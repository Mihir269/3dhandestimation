"""
hand_ekf.py
===========
Two-block Extended Kalman Filter over the 21 v12 articulation angles, with a
personalised (calibrated) kinematic hand model. Pure numpy, no I/O, no camera.

STATE
-----
x = [theta, theta_dot, theta_ddot]  (21 x 3 = 63), DEGREES.
Process: continuous white-noise jerk (CWNJ). Jerk is the only stochastic input,
so "minimise jerk" is literally "small `jerk_psd`" -- one knob.

MEASUREMENTS (applied sequentially -- see the correlated-blocks caveat)
------------
Block 1  z1 = 21 angles (corrected by v12, or raw MediaPipe for the benchmark)
         linear, H1 = [I 0 0]. Innovation is WRAPPED (wrap_deg) because atan2
         flexion lives on a circle: Thumb_IP_Flex spans -177.9..160.7 deg.
Block 2  z2 = the 15 FINGER keypoints of MediaPipe's world landmarks (mm),
         45-dim, nonlinear through FK. FK is evaluated at the current
         estimate, similarity-aligned to the observation (Umeyama), and the
         residual is taken in the OBSERVATION frame so the anisotropic
         diag(sxy^2, sxy^2, sz^2) per keypoint down-weights MediaPipe depth
         where it belongs. NIS target is therefore 45, not 63.

ALIGNMENT (changed from the 2026-09-05 design -- measured, see selftest.py)
---------
The similarity is fitted on the wrist + 5 MCPs only. In v12 FK those six
points depend on bone lengths and the calibrated fan, NOT on the angles, so
the alignment is independent of theta and the Jacobian is exact.

Two things were tried and rejected on a synthetic clip (sxy 1.5, sz 6 mm):
* Aligning on all 21 points: some angle combinations (all MCPs flexing
  together) look like a palm rotation, become unobservable, and drift.
  Block-2-only RMSE 92-105 deg.
* Palm alignment with a plain diagonal R2: six noisy points give a jittery
  root rotation whose error is SHARED by every fingertip and grows with lever
  arm. Unmodelled, it made block 2 worse than raw (12-19 deg vs 6 deg) with
  NIS ~210 at a nominally matched R.
So the palm points are used ONLY for alignment (they carry no angle
information anyway), and the alignment's own error covariance -- rotation and
translation noise propagated from the six palm points -- is added to R2 as a
full 45x45 block.

JACOBIAN
--------
Geometric, analytic. For joint j with world rotation axis a_j:
    d p_k / d theta_j = a_j x (p_k - p_j)     for every keypoint k downstream.
v12 builds F_child = F_parent @ M_x(flex) @ M_z(abd):
    flex axis = F_parent[:, 0]
    abd  axis = -(F_parent @ M_x(flex))[:, 2]   (M_z(a) is a rotation by -a)
The result is mm per RADIAN; the state is in DEGREES, so it is multiplied by
pi/180. Forgetting that inflates the gain 57x and the filter diverges (RMSE
41,386 deg, measured). Validated against finite differences in selftest.py.

FK positions reuse `anatomical_angles_v12.compose` and are checked against
`forward_kinematics_v12` when the model is built, so there is still exactly one
angle definition.

HONEST LIMITATIONS
------------------
* Kalman filters remove noise, not bias. A systematic corrector error that the
  keypoints agree with comes out smooth and confidently wrong.
* Block 1 (corrected angles) is computed FROM the same keypoints block 2 uses,
  so the two residuals are correlated and P shrinks faster than it should.
  `r_scale_*` is a blunt hedge; ablate with block 2 on/off.
* Real hands carry a structural FK residual (flat palm, zero hinge abduction)
  of a few mm. It belongs in sigma_xy before you read NIS.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import anatomical_angles_v12 as A12
import hand_angles_v2 as HA

N = HA.N_ANGLES                       # 21
DEG = np.pi / 180.0
PALM_IDS = (0, 1, 5, 9, 13, 17)       # wrist + 5 MCPs: independent of theta in FK
FINGER_IDS = tuple(k for k in range(21) if k not in PALM_IDS)   # 15, block 2
WRAP = HA.wrap_deg


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


# --------------------------------------------------------------------------
# Root orientation smoothing (DISPLAY ONLY)
# --------------------------------------------------------------------------
def mat_to_quat(R):
    m = np.asarray(R, float)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(m[i, i] - m[j, j] - m[k, k] + 1.0, 1e-12)) * 2
        q = np.zeros(4)
        q[0] = (m[k, j] - m[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (m[j, i] + m[i, j]) / s
        q[k + 1] = (m[k, i] + m[i, k]) / s
    return q / np.linalg.norm(q)


def quat_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class RootSmoother:
    """Exponential smoother on the per-frame palm-alignment rotation.

    The root pose is not in the filter state (Flow.md A30): it is re-fitted
    every frame by Umeyama on 6 palm points, so it carries MediaPipe's full
    landmark noise. The joint angles can be perfectly smooth while the whole
    drawn hand still shakes -- which is what the eye notices first.

    Smoothing it here is display-only: block 2 keeps using the raw rotation for
    its measurement update, so the estimate is untouched. `tau` is a time
    constant in SECONDS, so the amount of smoothing per second does not change
    with frame rate (important: webcams run well below 30 fps)."""

    def __init__(self):
        self.q = None

    def update(self, R, tau, dt):
        q = mat_to_quat(R)
        if self.q is None or tau <= 0.0:
            self.q = q
            return np.asarray(R, float)
        if np.dot(self.q, q) < 0.0:
            q = -q                                   # same rotation, nearer hemisphere
        a = 1.0 - np.exp(-max(dt, 1e-4) / tau)
        self.q = self.q + a * (q - self.q)
        self.q /= np.linalg.norm(self.q)
        return quat_to_mat(self.q)


# --------------------------------------------------------------------------
# Process model
# --------------------------------------------------------------------------
def cwnj_F(dt):
    return np.array([[1.0, dt, 0.5 * dt * dt],
                     [0.0, 1.0, dt],
                     [0.0, 0.0, 1.0]])


def cwnj_Q(dt, q):
    """Continuous white-noise jerk, discretised. q in deg^2 / s^5."""
    return q * np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                         [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                         [dt ** 3 / 6, dt ** 2 / 2, dt]])


# --------------------------------------------------------------------------
# Similarity alignment
# --------------------------------------------------------------------------
def umeyama(src, dst, with_scale=True):
    """Least-squares s, R, t with dst ~= s * R @ src + t. Proper rotation only."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(Y.T @ X / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (X ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale and var_s > 0 else 1.0
    return s, R, mu_d - s * R @ mu_s


# --------------------------------------------------------------------------
# Calibration -> personalised model
# --------------------------------------------------------------------------
def fan_from_palm_dirs(palm_dirs):
    """calib_adapter's (5,3) wrist->MCP unit directions (Thumb..Pinky, palm
    plane, +x radial) -> the 4 v12 fan angles, wrapped and order-checked."""
    P = np.asarray(palm_dirs, float).reshape(5, 3)
    az = np.degrees(np.arctan2(P[:, 0], P[:, 1]))
    t, i, m, r, p = az
    return HA.as_fan_dict({"Thumb_Index_Fan": t - i, "Index_Middle_Fan": i - m,
                           "Middle_Ring_Fan": m - r, "Ring_Pinky_Fan": r - p})


class PersonalizedHandModel:
    """Calibrated bone lengths (mm) + palm fan -> FK and its angle Jacobian."""

    def __init__(self, bone_lengths_mm, fan_deg, name="hand"):
        self.name = name
        self.bones = HA.as_bone_dict(bone_lengths_mm)
        self.fan = HA.as_fan_dict(fan_deg)
        self._az = {f: np.radians(a) for f, a in HA.fan_azimuths(self.fan).items()}
        self._palm = A12._orthonormalize(np.array([0.0, 1.0, 0.0]),
                                         np.array([0.0, 0.0, 1.0]))
        # joint table: finger -> [(kp index, flex state idx, abd state idx | None)]
        self._joints = {}
        # Metacarpal heads and each finger's first joint frame depend only on
        # bones + fan, never on the angles, so they are built once here rather
        # than on every FK call (FK runs several times per frame live).
        self._base = {}
        self.palm_points = np.zeros((21, 3))           # wrist + 5 MCPs, fixed
        for finger, chain in A12.FINGER_CHAINS.items():
            rows = []
            for label, hinge in A12.JOINT_LAYOUT[finger]:
                rows.append((HA.ANGLE_INDEX[f"{label}_Flex"],
                             None if hinge else HA.ANGLE_INDEX[f"{label}_Abd"]))
            self._joints[finger] = rows
            a = self._az[finger]
            ray = self._palm @ np.array([np.sin(a), np.cos(a), 0.0])
            mcp = self.bones[(A12.WRIST, chain[0])] * ray
            self._base[finger] = (mcp, A12._orthonormalize(ray, self._palm[:, 2]))
            self.palm_points[chain[0]] = mcp
        self._check_against_v12()

    @classmethod
    def from_calibration(cls, path):
        with open(path) as fh:
            c = json.load(fh)
        if "lengths_mm" in c:                                 # hand_calibration_ekf/v1
            bones = c["lengths_mm"]
            fan = (fan_from_palm_dirs(c["palm_dirs"]) if "palm_dirs" in c
                   else c["fan_angles_deg"])
        elif "bone_lengths_mm" in c:                          # hand_calibration/v2
            bones = {tuple(int(x) for x in k.split("-")): v
                     for k, v in c["bone_lengths_mm"].items()}
            fan = c["palm_template"]["fan_angles_deg"]
        else:
            raise KeyError(f"{path}: unrecognised calibration schema")
        if "palm_dirs_measured" in c and not c["palm_dirs_measured"]:
            print(f"[hand_ekf] WARNING {path}: palm fan is NOMINAL, not measured")
        return cls(bones, fan, name=str(c.get("hand", path)))

    @property
    def hand_length_mm(self):
        return sum(self.bones[b] for b in ((0, 9), (9, 10), (10, 11), (11, 12)))

    def fk(self, theta_deg):
        return self.fk_and_jacobian(theta_deg, jacobian=False)[0]

    def fk_and_jacobian(self, theta_deg, jacobian=True):
        """theta (21,) deg -> kps (21,3) mm in the canonical palm frame, wrist at
        origin; and J (21,3,21) in mm per DEGREE (None if jacobian=False)."""
        th = np.radians(np.asarray(theta_deg, float))
        kps = np.zeros((21, 3))
        J = np.zeros((21, 3, N)) if jacobian else None

        for finger, chain in A12.FINGER_CHAINS.items():
            kps[chain[0]], F = self._base[finger]
            axes = []
            for k, (i_f, i_a) in enumerate(self._joints[finger]):
                flex = th[i_f]
                abd = 0.0 if i_a is None else th[i_a]
                if jacobian:
                    ax_f = F[:, 0].copy()
                    ax_a = -(F @ A12._M_x(flex))[:, 2]
                    axes.append((ax_f, ax_a))
                direction, F = A12.compose(F, abd, flex)
                kps[chain[k + 1]] = kps[chain[k]] + \
                    self.bones[(chain[k], chain[k + 1])] * direction
            if jacobian:
                for k, (i_f, i_a) in enumerate(self._joints[finger]):
                    ax_f, ax_a = axes[k]
                    down = list(chain[k + 1:])
                    levers = kps[down] - kps[chain[k]]          # (m,3)
                    J[down, :, i_f] = levers @ skew(ax_f).T     # a x lever, row-wise
                    if i_a is not None:
                        J[down, :, i_a] = levers @ skew(ax_a).T
        if jacobian:
            J *= DEG
        return kps, J

    def _check_against_v12(self):
        rng = np.random.default_rng(1)
        th = np.where([n.endswith("_Flex") for n in HA.ARTICULATION_ANGLES],
                      rng.uniform(-10, 80, N), rng.uniform(-15, 15, N))
        ref = HA.forward_kinematics(th, self.bones, fan_angles=self.fan)
        err = float(np.abs(self.fk(th) - ref).max())
        if err > 1e-9:
            raise AssertionError(
                f"hand_ekf FK disagrees with forward_kinematics_v12 by {err:.2e} mm")


# --------------------------------------------------------------------------
# Filter
# --------------------------------------------------------------------------
@dataclass
class EKFConfig:
    # Defaults measured on DexYCB (100 sequences, GT 3D joints) -- Flow.md 7.5.
    jerk_psd: float = 3e4            # deg^2/s^5. smaller = smoother, laggier
    sigma_angle: object = 6.0        # deg. scalar or (21,) array
    sigma_xy: float = 15.0           # mm. covers MediaPipe noise AND the FK model
                                     # mismatch; 6.0 scored worse on DexYCB even
                                     # though its NIS2 looked closer to target
    sigma_z: float = 25.0            # mm, MediaPipe depth noise
    r_scale_angles: float = 2.0      # variance multipliers: correlated-blocks hedge
    r_scale_kps: float = 2.0
    use_block1: bool = True
    use_block2: bool = True
    align_noise: bool = True         # add alignment error covariance to R2
    # Adaptive process noise: scale jerk_psd per angle by how fast that angle is
    # moving, so a resting finger is smoothed hard and a moving one is not held
    # back. Same idea as the 1-euro filter, applied to Q instead of to a cutoff.
    # multiplier = clip((|theta_dot| / adapt_ref_speed)^2, adapt_lo, adapt_hi)
    # Measured on clip.mp4 at matched fidelity to the input: adaptive gives ~6 %
    # less jitter than the equivalent fixed jerk_psd, and 27 % less than the
    # default fixed value at +16 % deviation. Neutral on DexYCB accuracy
    # (13.87° vs 13.86°). Modest but free; --no-adaptive restores fixed Q.
    adaptive: bool = True
    adapt_ref_speed: float = 120.0   # deg/s at which the multiplier passes 1
    adapt_lo: float = 0.02           # floor: how much smoother a still finger gets
    adapt_hi: float = 8.0            # ceiling: how responsive a fast finger gets
    adapt_tau_s: float = 0.08        # smoothing of the speed estimate itself
    root_tau_s: float = 0.08         # DISPLAY-ONLY smoothing of the hand's overall
                                     # orientation, seconds. 0 = off. The root pose
                                     # is not in the state, so it is otherwise raw
                                     # per-frame MediaPipe noise -- the jitter the
                                     # eye notices most on a webcam.
    # Robust (innovation-weighted) measurement noise. Each frame, every angle's
    # innovation is compared with what the filter expected, d = |y| / sqrt(S)
    # (S = predicted variance + R). A measurement that jumps further than
    # `robust_k` sigmas from the physics prediction is down-weighted by
    # inflating its R by (d / k)^p, so it still pulls, but less. p = 1 is the
    # Huber weight; p = 2 makes its pull fall with distance.
    # Block 2 does the same per finger keypoint (d from the 3-D Mahalanobis
    # distance / sqrt 3). Off by default: enable with robust=True.
    robust: bool = False
    robust_k: float = 3.0            # sigmas before an angle is down-weighted
    robust_k_kp: float = 3.0         # same for a block-2 keypoint
    robust_power: float = 1.0        # 1 = Huber, 2 = redescending-like
    scale_ema: float = 0.05        # smoothing of the per-frame Umeyama scale
    init_std: tuple = (10.0, 200.0, 2000.0)   # theta, theta_dot, theta_ddot
    max_coast_frames: int = 8        # predict-only frames before a reset
    # While coasting (no detection), multiply velocity and acceleration by this
    # per frame, so the prediction settles instead of extrapolating a
    # constant-acceleration curve for 8 frames (DexYCB: coasting error grew to
    # 60 deg+ in the worst clips). Measured on DexYCB s01: coast MAE 21.9 ->
    # 18.2 deg at 0.3, detected-frame MAE unchanged. 1.0 = old behaviour.
    coast_decay: float = 0.3

    def angle_sigma_vector(self):
        s = np.asarray(self.sigma_angle, float)
        return np.full(N, float(s)) if s.ndim == 0 else s.reshape(N)


def robust_inflation(d, k, power):
    """R multiplier for normalised innovations d (>= 1; 1 inside the gate)."""
    d = np.asarray(d, float)
    return np.where(d > k, (d / max(k, 1e-6)) ** power, 1.0)


def load_angle_sigma_csv(path, column_hint="Corr"):
    """Per-angle sigma from freihand_correction_v12/test_per_angle_improvement.csv.
    Picks the column containing 'RMSE' and `column_hint` ('Corr' or 'MP').
    RMSE^2 there is block 1's error variance against GT (bias included)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    cols = list(rows[0])
    name_col = next(c for c in cols if c.lower() in ("angle", "name"))
    val_col = next((c for c in cols if "rmse" in c.lower()
                    and column_hint.lower() in c.lower()), None)
    if val_col is None:
        raise KeyError(f"{path}: no RMSE column matching {column_hint!r} in {cols}")
    table = {r[name_col]: float(r[val_col]) for r in rows}
    missing = [n for n in HA.ARTICULATION_ANGLES if n not in table]
    if missing:
        raise KeyError(f"{path}: missing angles {missing}")
    return np.array([table[n] for n in HA.ARTICULATION_ANGLES])


class HandEKF:
    def __init__(self, model: PersonalizedHandModel, cfg: Optional[EKFConfig] = None):
        self.model = model
        self.cfg = cfg or EKFConfig()
        c = self.cfg
        self.R1_diag = c.angle_sigma_vector() ** 2 * c.r_scale_angles
        self.Sigma_kp = np.diag([c.sigma_xy ** 2, c.sigma_xy ** 2,
                                 c.sigma_z ** 2]) * c.r_scale_kps   # per keypoint, obs frame
        self.R2_diag = np.tile(np.diag(self.Sigma_kp), len(FINGER_IDS))
        self._align_ids = list(PALM_IDS)
        self._meas_ids = list(FINGER_IDS)
        self._Fq_cache = {}
        self.reset()

    # ---- bookkeeping -----------------------------------------------------
    def reset(self):
        self.x = np.zeros(3 * N)
        self.P = np.eye(3 * N)
        self.initialised = False
        self.coast = 0
        self.scale = None
        self.last_R = np.eye(3)          # raw fit: used by block 2
        self.display_R = np.eye(3)       # smoothed: used for drawing only
        self._root = RootSmoother()
        self._dt = 1.0 / 30.0
        self._speed = np.zeros(N)        # smoothed |theta_dot|, deg/s
        self.last_t = np.zeros(3)
        self.last_nis1 = np.nan
        self.last_nis2 = np.nan
        self.last_resid_mm = np.nan
        self.last_infl1 = np.ones(N)                 # robust R multipliers, last frame
        self.last_infl2 = np.ones(len(FINGER_IDS))

    @property
    def angles(self):
        return self.x[:N].copy()

    @property
    def angle_std(self):
        return np.sqrt(np.clip(np.diag(self.P)[:N], 0, None))

    def _q_scale(self):
        """Per-angle multiplier on jerk_psd from the smoothed angular speed."""
        c = self.cfg
        if not c.adaptive:
            return None
        return np.clip((self._speed / max(c.adapt_ref_speed, 1e-6)) ** 2,
                       c.adapt_lo, c.adapt_hi)

    def _FQ(self, dt):
        key = round(dt, 5)
        if key not in self._Fq_cache:
            if len(self._Fq_cache) > 64:
                self._Fq_cache.clear()
            self._Fq_cache[key] = (np.kron(cwnj_F(dt), np.eye(N)), cwnj_Q(dt, 1.0))
        F, Qu = self._Fq_cache[key]
        scale = self._q_scale()
        # kron(A, diag(q)) scales each angle's own 3x3 process block by q_i
        diag = np.diag(np.full(N, self.cfg.jerk_psd) if scale is None
                       else self.cfg.jerk_psd * scale)
        return F, np.kron(Qu, diag)

    def _init_state(self, z_angles):
        s = self.cfg.init_std
        self.x[:] = 0.0
        self.x[:N] = WRAP(z_angles)
        self.P = np.diag(np.repeat(np.square(s), N))
        self.initialised = True
        self.coast = 0

    # ---- filter steps ----------------------------------------------------
    def predict(self, dt):
        F, Q = self._FQ(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def _apply(self, K, S, y):
        self.x += K @ y
        self.P -= K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def update_angles(self, z):
        y = WRAP(np.asarray(z, float) - self.x[:N])
        R1 = self.R1_diag
        if self.cfg.robust:
            d = np.abs(y) / np.sqrt(np.diag(self.P)[:N] + R1)
            self.last_infl1 = robust_inflation(d, self.cfg.robust_k, self.cfg.robust_power)
            R1 = R1 * self.last_infl1
        S = self.P[:N, :N] + np.diag(R1)
        K = np.linalg.solve(S, self.P[:N, :]).T
        self.last_nis1 = float(y @ np.linalg.solve(S, y))
        self._apply(K, S, y)

    def align(self, kps_mm, p_fk=None):
        """Similarity FK -> observation on the alignment points. Updates the
        smoothed scale and returns (s, R, t)."""
        p = self.model.fk(self.x[:N]) if p_fk is None else p_fk
        ids = self._align_ids
        s, R, _ = umeyama(p[ids], kps_mm[ids])
        e = self.cfg.scale_ema
        self.scale = s if self.scale is None else (1 - e) * self.scale + e * s
        t = kps_mm[ids].mean(0) - self.scale * R @ p[ids].mean(0)
        self.last_R, self.last_t = R, t
        self.display_R = self._root.update(R, self.cfg.root_tau_s, self._dt)
        return self.scale, R, t

    def _alignment_cov(self, s, R, p):
        """Covariance, in the observation frame, of the FK prediction error that
        the per-frame palm alignment itself introduces at the finger keypoints.

        Palm noise n_i perturbs the fitted rotation by omega and translation by
        tau; a finger point at lever y_k (from the palm centroid) then moves by
        -skew(y_k) omega + tau. Unweighted Umeyama gives
            cov(omega) = M^-1 B M^-1,  M = sum A_i^T A_i,  B = sum A_i^T Sn A_i
            cov(tau)   = Sn / n_palm,  A_i = skew(x_i)
        and omega, tau are uncorrelated because the points are centred."""
        pa = p[self._align_ids]
        c = pa.mean(0)
        Sn = self.Sigma_kp
        M = np.zeros((3, 3))
        B = np.zeros((3, 3))
        for xi in (s * (pa - c) @ R.T):
            A = skew(xi)
            M += A.T @ A
            B += A.T @ Sn @ A
        Mi = np.linalg.inv(M)
        S_om = Mi @ B @ Mi
        G = np.concatenate([skew(y) for y in s * (p[self._meas_ids] - c) @ R.T])
        m = len(self._meas_ids)
        return G @ S_om @ G.T + np.kron(np.ones((m, m)), Sn / len(pa))

    def update_keypoints(self, kps_mm):
        z = np.asarray(kps_mm, float).reshape(21, 3)
        p, J = self.model.fk_and_jacobian(self.x[:N])
        s, R, t = self.align(z, p)
        ids = self._meas_ids
        m3 = 3 * len(ids)
        pred = s * p[ids] @ R.T + t
        r = (z[ids] - pred).reshape(m3)
        H = (s * np.einsum("ij,kjn->kin", R, J[ids])).reshape(m3, N)
        R2 = np.diag(self.R2_diag)
        if self.cfg.align_noise:
            R2 = R2 + self._alignment_cov(s, R, p)
        PHt = self.P[:, :N] @ H.T                      # (63, 45)
        S = H @ PHt[:N] + R2
        if self.cfg.robust:
            m = len(ids)
            d = np.empty(m)
            for k in range(m):
                b = slice(3 * k, 3 * k + 3)
                d[k] = np.sqrt(r[b] @ np.linalg.solve(S[b, b], r[b]) / 3.0)
            self.last_infl2 = robust_inflation(d, self.cfg.robust_k_kp, self.cfg.robust_power)
            if np.any(self.last_infl2 > 1.0):
                # inflate only the keypoint's own noise, not the shared alignment term
                extra = np.repeat(self.last_infl2 - 1.0, 3) * self.R2_diag
                R2 = R2 + np.diag(extra)
                S = S + np.diag(extra)
        K = np.linalg.solve(S, PHt.T).T
        self.last_nis2 = float(r @ np.linalg.solve(S, r))
        self.last_resid_mm = float(np.linalg.norm(r.reshape(-1, 3), axis=1).mean())
        self._apply(K, S, r)

    def step(self, z_angles, kps_mm, dt):
        """One frame. Either measurement may be None (dropout). Returns the
        filtered angles, or None when the filter has no track."""
        have = z_angles is not None or kps_mm is not None
        if not self.initialised:
            if not have:
                return None
            z0 = z_angles if z_angles is not None else HA.compute_angles(kps_mm)
            self._init_state(z0)
            if kps_mm is not None:
                self.align(np.asarray(kps_mm, float))
            return self.angles

        self._dt = max(float(dt), 1e-4)
        self.predict(self._dt)
        self.last_nis1 = self.last_nis2 = self.last_resid_mm = np.nan
        self.last_infl1 = np.ones(N)
        self.last_infl2 = np.ones(len(FINGER_IDS))
        if not have:
            self.coast += 1
            if self.coast > self.cfg.max_coast_frames:
                self.reset()
                return None
            if self.cfg.coast_decay < 1.0:
                self.x[N:] *= self.cfg.coast_decay
            return self.angles
        self.coast = 0

        if z_angles is not None and self.cfg.use_block1:
            self.update_angles(z_angles)
        if kps_mm is not None:
            if self.cfg.use_block2:
                self.update_keypoints(kps_mm)
            else:
                self.align(np.asarray(kps_mm, float))   # display orientation only
        self.x[:N] = WRAP(self.x[:N])
        self._track_speed(self.x[N:2 * N])
        return self.angles

    def _track_speed(self, vel):
        a = 1.0 - np.exp(-self._dt / max(self.cfg.adapt_tau_s, 1e-6))
        self._speed += a * (np.abs(vel) - self._speed)

    def set_measurement_model(self, model: PersonalizedHandModel):
        """Swap the skeleton block 2 compares against, keeping the angle track."""
        self.model = model
        self.scale = None

    def skeleton(self, oriented=True, model: Optional[PersonalizedHandModel] = None):
        """FK at the current estimate, mm, wrist at origin. `oriented` applies
        the last alignment rotation (no scale) so it turns with the real hand.
        `model` lets the display use the calibrated hand even when block 2
        runs against a MediaPipe-estimated skeleton."""
        p = (model or self.model).fk(self.x[:N])
        return p @ self.display_R.T if oriented else p


class MediaPipeSkeletonEstimator:
    """Median bone lengths + fan over the first `n_frames` detections.

    Why: MediaPipe's world landmarks do not have the subject's proportions.
    Measured on Live_Webcam_testing/clip.mp4 against the right-hand
    calibration (ratios to Middle MCP->PIP): wrist->MCP +38..48 %, distal
    phalanges +34..60 %. FK with calibrated bones cannot fit those points
    better than ~26 mm mean per finger keypoint; with MediaPipe's own
    skeleton the floor is ~13 mm. Block 2 then pulls angles toward whatever
    makes the wrong skeleton fit -- a bias, which a Kalman filter does not
    remove. Using this skeleton for block 2 trades personal metric geometry
    for a smaller, less biased residual. Benchmark both."""

    def __init__(self, n_frames=15):
        self.n = n_frames
        self.bones, self.fans = [], []

    @property
    def done(self):
        return len(self.bones) >= self.n

    def add(self, kps):
        if self.done:
            return None
        _, fan = HA.compute_angles(kps, return_fan=True)
        bl = HA.bone_lengths_from_kps(kps)
        self.bones.append([bl[b] for b in HA.BONES])
        self.fans.append(WRAP(fan))
        if not self.done:
            return None
        return PersonalizedHandModel(np.median(self.bones, 0),
                                     np.median(self.fans, 0), name="mediapipe")
