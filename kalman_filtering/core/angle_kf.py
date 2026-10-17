"""
angle_kf.py
===========
Joint-angle-only, jerk-minimising Kalman filter. No keypoint block, no FK in
the estimate -- the calibrated FK model is used only to DRAW the result.

WHY A SEPARATE, SIMPLER FILTER
------------------------------
With only the angle measurement (H = [I 0 0]), a diagonal R, a diagonal
initial P and a process model that does not couple angles, the 63-state
HandEKF never builds cross-angle covariance. It is therefore EXACTLY 21
independent 3-state filters, one per angle. Running them that way is the same
estimate (tests/selftest.py checks agreement to ~1e-9 deg) at a fraction of
the cost: 21 x 3x3 operations instead of 63x63.

PER-ANGLE MODEL
---------------
state      s = [theta, theta_dot, theta_ddot]           deg, deg/s, deg/s^2
process    s_k = F s_{k-1} + w,   w ~ N(0, Q)           CWNJ, see hand_ekf.cwnj_Q
           F = [[1, dt, dt^2/2], [0, 1, dt], [0, 0, 1]]
measure    z_k = theta_k + v,     v ~ N(0, sigma^2 * r_scale)
innovation y = wrap_deg(z - theta)                      atan2 flexion is circular

The jerk PSD q is the only process knob: the filter explains the measurements
with the smallest random jerk consistent with the assumed noise, which is the
causal (filtering) counterpart of a minimum-jerk fit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import hand_angles_v2 as HA
from hand_ekf import (N, PALM_IDS, EKFConfig, PersonalizedHandModel, RootSmoother,
                      WRAP, cwnj_F, cwnj_Q, robust_inflation, umeyama)


class AngleKF:
    """Drop-in for HandEKF in core/pipeline.py (same attributes it reads)."""

    def __init__(self, model: PersonalizedHandModel, cfg: Optional[EKFConfig] = None):
        self.model = model                         # display only
        self.cfg = cfg or EKFConfig()
        self.cfg.use_block2 = False
        self.r = self.cfg.angle_sigma_vector() ** 2 * self.cfg.r_scale_angles   # (21,)
        self._cache = {}
        self.reset()

    # ---- bookkeeping -----------------------------------------------------
    def reset(self):
        self.x = np.zeros((N, 3))
        self.P = np.zeros((N, 3, 3))
        self.initialised = False
        self.coast = 0
        self.scale = None
        self.last_R = np.eye(3)
        self.display_R = np.eye(3)       # smoothed root orientation, drawing only
        self._root = RootSmoother()
        self._dt = 1.0 / 30.0
        self._speed = np.zeros(N)        # smoothed |theta_dot|, deg/s
        self.last_nis1 = np.nan
        self.last_nis2 = np.nan                    # no keypoint block
        self.last_resid_mm = np.nan
        self.last_infl1 = np.ones(N)

    @property
    def angles(self):
        return self.x[:, 0].copy()

    @property
    def angle_std(self):
        return np.sqrt(np.clip(self.P[:, 0, 0], 0, None))

    def _FQ(self, dt):
        """F is (3,3); Q is (21,3,3) so each angle can carry its own jerk PSD."""
        key = round(dt, 5)
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = (cwnj_F(dt), cwnj_Q(dt, 1.0))
        F, Qu = self._cache[key]
        c = self.cfg
        q = np.full(N, c.jerk_psd)
        if c.adaptive:
            q = q * np.clip((self._speed / max(c.adapt_ref_speed, 1e-6)) ** 2,
                            c.adapt_lo, c.adapt_hi)
        return F, Qu[None, :, :] * q[:, None, None]

    # ---- filter ----------------------------------------------------------
    def predict(self, dt):
        F, Q = self._FQ(dt)
        self.x = self.x @ F.T                                     # (21,3)
        self.P = F @ self.P @ F.T + Q                             # broadcast (21,3,3)

    def update(self, z):
        y = WRAP(np.asarray(z, float) - self.x[:, 0])             # (21,)
        r = self.r
        if self.cfg.robust:                                       # see EKFConfig.robust
            d = np.abs(y) / np.sqrt(self.P[:, 0, 0] + r)
            self.last_infl1 = robust_inflation(d, self.cfg.robust_k, self.cfg.robust_power)
            r = r * self.last_infl1
        S = self.P[:, 0, 0] + r                                   # (21,)
        K = self.P[:, :, 0] / S[:, None]                          # (21,3)
        self.x += K * y[:, None]
        self.P -= K[:, :, None] * self.P[:, 0, :][:, None, :]     # (I - K H) P
        self.P = 0.5 * (self.P + self.P.transpose(0, 2, 1))
        self.last_nis1 = float(np.sum(y * y / S))

    def step(self, z_angles, kps_mm, dt):
        """Same contract as HandEKF.step. `kps_mm` is used only to orient the
        drawn skeleton; it never touches the estimate."""
        if not self.initialised:
            if z_angles is None:
                return None
            s = self.cfg.init_std
            self.x[:] = 0.0
            self.x[:, 0] = WRAP(z_angles)
            self.P[:] = np.diag(np.square(s))
            self.initialised = True
            self.coast = 0
            self._orient(kps_mm)
            return self.angles

        self._dt = max(float(dt), 1e-4)
        self.predict(self._dt)
        self.last_nis1 = np.nan
        self.last_infl1 = np.ones(N)
        if z_angles is None:
            self.coast += 1
            if self.coast > self.cfg.max_coast_frames:
                self.reset()
                return None
            if self.cfg.coast_decay < 1.0:                        # see EKFConfig.coast_decay
                self.x[:, 1:] *= self.cfg.coast_decay
            return self.angles
        self.coast = 0
        self.update(z_angles)
        self.x[:, 0] = WRAP(self.x[:, 0])
        a = 1.0 - np.exp(-self._dt / max(self.cfg.adapt_tau_s, 1e-6))
        self._speed += a * (np.abs(self.x[:, 1]) - self._speed)
        self._orient(kps_mm)
        return self.angles

    # ---- display ---------------------------------------------------------
    def _orient(self, kps_mm):
        if kps_mm is None:
            return
        ids = list(PALM_IDS)                   # palm points do not depend on the angles
        _, self.last_R, _ = umeyama(self.model.palm_points[ids],
                                    np.asarray(kps_mm, float)[ids])
        self.display_R = self._root.update(self.last_R, self.cfg.root_tau_s, self._dt)

    def set_measurement_model(self, model):
        pass                                                      # no keypoint block

    def skeleton(self, oriented=True, model: Optional[PersonalizedHandModel] = None):
        p = (model or self.model).fk(self.x[:, 0])
        return p @ self.display_R.T if oriented else p
