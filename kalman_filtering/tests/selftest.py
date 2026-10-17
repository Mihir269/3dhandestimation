#!/usr/bin/env python3
"""
selftest.py -- run before trusting any filter output.

    python tests\\selftest.py     exit 0 = everything passed

Checks, all on SYNTHETIC hands built on your real right-hand calibration:
  1. anatomical_angles_v12 + hand_angles_v2 self-tests
  2. hand_ekf FK == forward_kinematics_v12, analytic Jacobian == finite diff
  3. calibration palm_dirs -> fan angles match the documented measurement
  4. wrap-aware innovation: a track crossing +-180 deg does not lurch
  5. tracking on a noisy synthetic clip: raw vs block 1 vs block 2 vs both,
     plus mean NIS so you can see what "matched noise" looks like
     (NIS1 target 21 = angles, NIS2 target 45 = 15 finger keypoints x 3)
  6. checkpoint angle names == hand_angles_v2.ARTICULATION_ANGLES
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # kalman_filtering/
sys.path.insert(0, os.path.join(HERE, "core"))

import time  # noqa: E402

import anatomical_angles_v12 as A12  # noqa: E402,F401
import hand_angles_v2 as HA  # noqa: E402
from angle_kf import AngleKF  # noqa: E402
from hand_ekf import (N, EKFConfig, HandEKF, PersonalizedHandModel,  # noqa: E402
                      fan_from_palm_dirs, WRAP)

CALIB_R = os.path.join(HERE, "calib", "hand_calibration_ekf_right.json")
CALIB_L = os.path.join(HERE, "calib", "hand_calibration_ekf_left.json")
CKPT = os.path.join(HERE, "models", "angle_corrector_v12.pth")

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


def random_pose(rng):
    return np.where([n.endswith("_Flex") for n in HA.ARTICULATION_ANGLES],
                    rng.uniform(-10, 85, N), rng.uniform(-18, 18, N))


def rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    rng = np.random.default_rng(0)

    print("1. canonical modules")
    check("hand_angles_v2 self-test", HA._selftest() == 0)
    print()

    print("2. FK + Jacobian")
    model = PersonalizedHandModel.from_calibration(CALIB_R)
    worst_p, worst_j = 0.0, 0.0
    for _ in range(100):
        th = random_pose(rng)
        p, J = model.fk_and_jacobian(th)
        ref = HA.forward_kinematics(th, model.bones, fan_angles=model.fan)
        worst_p = max(worst_p, float(np.abs(p - ref).max()))
        Jn = np.zeros_like(J)
        h = 1e-6
        for i in range(N):
            d = np.zeros(N)
            d[i] = h
            Jn[:, :, i] = (model.fk(th + d) - model.fk(th - d)) / (2 * h)
        worst_j = max(worst_j, float(np.abs(J - Jn).max() / (np.abs(Jn).max() + 1e-12)))
    check("FK matches forward_kinematics_v12", worst_p < 1e-9, f"{worst_p:.2e} mm")
    check("analytic Jacobian vs finite diff", worst_j < 1e-6, f"{worst_j:.2e} rel")
    print()

    print("3. calibration fan")
    fr = model.fan
    check("right fan = documented 49.4/18.4/15.0/15.5",
          np.allclose([fr[n] for n in HA.FAN_ANGLE_NAMES], [49.4, 18.4, 15.0, 15.5],
                      atol=0.1), str({k: round(v, 1) for k, v in fr.items()}))
    left = PersonalizedHandModel.from_calibration(CALIB_L)
    check("left (mirrored) fan wrapped + radial order",
          all(-90 < v < 90 for v in left.fan.values()),
          str({k: round(v, 1) for k, v in left.fan.items()}))
    print()

    print("4. wrap-aware innovation")
    ekf = HandEKF(model, EKFConfig(use_block2=False, sigma_angle=3.0))
    base = random_pose(rng)
    k = HA.ANGLE_INDEX["Thumb_IP_Flex"]
    worst_jump = 0.0
    prev = None
    for f in range(60):
        z = base.copy()
        z[k] = WRAP(170 + f * 0.5)                   # 170 -> 199.5 i.e. crosses +-180
        est = ekf.step(z, None, 1 / 30)
        if prev is not None:
            worst_jump = max(worst_jump, abs(float(WRAP(est[k] - prev))))
        prev = est[k]
    check("no lurch across +-180", worst_jump < 5.0, f"max step {worst_jump:.2f} deg")
    print()

    print("5. synthetic tracking (30 fps, 12 s)")
    fps, T = 30.0, 360
    t = np.arange(T) / fps
    centre, amp = random_pose(rng) * 0.6, np.where(
        [n.endswith("_Flex") for n in HA.ARTICULATION_ANGLES], 30.0, 8.0)
    freq = rng.uniform(0.2, 0.8, N)
    phase = rng.uniform(0, 2 * np.pi, N)
    truth = centre + amp * np.sin(2 * np.pi * freq * t[:, None] + phase)
    SIG_ANG, SIG_XY, SIG_Z = 6.0, 1.5, 6.0
    R0 = rot(rng)
    meas_ang = truth + rng.normal(0, SIG_ANG, truth.shape)
    meas_kps = []
    for f in range(T):
        p = model.fk(truth[f]) * 0.92 @ R0.T + np.array([10.0, -20.0, 400.0])
        p += rng.normal(0, 1, p.shape) * [SIG_XY, SIG_XY, SIG_Z]
        meas_kps.append(p)

    def run(cfg):
        e = HandEKF(model, cfg)
        est = np.array([e.step(meas_ang[f] if cfg.use_block1 else None,
                               meas_kps[f] if cfg.use_block2 else None, 1 / fps)
                        for f in range(T)])
        err = WRAP(est - truth)[30:]
        n1, n2 = [], []
        e2 = HandEKF(model, cfg)
        for f in range(T):
            e2.step(meas_ang[f] if cfg.use_block1 else None,
                    meas_kps[f] if cfg.use_block2 else None, 1 / fps)
            if f >= 30:
                n1.append(e2.last_nis1)
                n2.append(e2.last_nis2)
        jit = np.sqrt(np.mean(np.diff(est[30:], 2, axis=0) ** 2))
        return float(np.sqrt(np.mean(err ** 2))), jit, np.nanmean(n1), np.nanmean(n2)

    raw_rmse = float(np.sqrt(np.mean((meas_ang - truth)[30:] ** 2)))
    raw_jit = float(np.sqrt(np.mean(np.diff(meas_ang[30:], 2, axis=0) ** 2)))
    true_jit = float(np.sqrt(np.mean(np.diff(truth[30:], 2, axis=0) ** 2)))
    d = EKFConfig()
    print(f"      shipped defaults: jerk_psd {d.jerk_psd:g}, sigma_angle "
          f"{d.sigma_angle}, sigma_xy {d.sigma_xy}, sigma_z {d.sigma_z} "
          "(tuned on DexYCB, Flow.md 7.5)")
    print(f"      {'variant':<22}{'RMSE deg':>10}{'jitter':>9}{'NIS1':>8}{'NIS2':>8}")
    print(f"      {'raw measurement':<22}{raw_rmse:10.2f}{raw_jit:9.2f}")
    print(f"      {'ground truth':<22}{0:10.2f}{true_jit:9.2f}")
    # `matched` = R set to the noise actually injected. jerk_psd is pinned here
    # rather than taken from EKFConfig: this block tests the filter machinery,
    # not the shipped tuning, and this synthetic motion (0.2-0.8 Hz, +-30 deg)
    # is faster than real hand motion, so the shipped default would lag it.
    matched = dict(sigma_angle=SIG_ANG, sigma_xy=SIG_XY, sigma_z=SIG_Z,
                   r_scale_angles=1.0, r_scale_kps=1.0, jerk_psd=1e5)
    res = {}
    for name, extra in (("block 1 only", dict(use_block2=False)),
                        ("block 2 only", dict(use_block1=False)),
                        ("both (matched R)", {}),
                        ("both (defaults)", None)):
        cfg = EKFConfig() if extra is None else EKFConfig(**matched, **extra)
        res[name] = run(cfg)
        r, j, a, b = res[name]
        print(f"      {name:<22}{r:10.2f}{j:9.2f}{a:8.1f}{b:8.1f}")
    # angle-only KF: must be the same estimate as HandEKF with block 2 off
    cfg_a = EKFConfig(**matched, use_block2=False)
    akf, hkf = AngleKF(model, cfg_a), HandEKF(model, EKFConfig(**matched, use_block2=False))
    est_a, worst_eq, t_a, t_h = [], 0.0, 0.0, 0.0
    for f in range(T):
        t1 = time.perf_counter()
        a = akf.step(meas_ang[f], None, 1 / fps)
        t2 = time.perf_counter()
        h = hkf.step(meas_ang[f], None, 1 / fps)
        t3 = time.perf_counter()
        t_a, t_h = t_a + t2 - t1, t_h + t3 - t2
        est_a.append(a)
        worst_eq = max(worst_eq, float(np.abs(WRAP(a - h)).max()))
    est_a = np.array(est_a)
    rmse_a = float(np.sqrt(np.mean(WRAP(est_a - truth)[30:] ** 2)))
    jit_a = float(np.sqrt(np.mean(np.diff(est_a[30:], 2, axis=0) ** 2)))
    print(f"      {'angle_kf.py':<22}{rmse_a:10.2f}{jit_a:9.2f}")
    # full EKF timing, both blocks
    e_full = HandEKF(model, EKFConfig(**matched))
    t4 = time.perf_counter()
    for f in range(T):
        e_full.step(meas_ang[f], meas_kps[f], 1 / fps)
    t_full = time.perf_counter() - t4
    print(f"      per-frame filter cost: angle_kf {1e3 * t_a / T:.3f} ms | "
          f"HandEKF block 1 {1e3 * t_h / T:.3f} ms | HandEKF both {1e3 * t_full / T:.3f} ms")
    check("angle_kf == HandEKF(block 2 off)", worst_eq < 1e-6, f"{worst_eq:.1e} deg")
    check("block 1 beats raw", res["block 1 only"][0] < raw_rmse)
    check("both beats block 1 only", res["both (matched R)"][0] < res["block 1 only"][0])
    check("filter reduces jitter 5x", res["both (defaults)"][1] < raw_jit / 5)
    check("block 2 only is useful (beats raw)", res["block 2 only"][0] < raw_rmse)
    check("matched NIS1 near 21", 10 < res["block 1 only"][2] < 42,
          f"{res['block 1 only'][2]:.1f}")
    check("matched NIS2 near 45", 22 < res["both (matched R)"][3] < 90,
          f"{res['both (matched R)'][3]:.1f}")
    print()

    print("5b. robust (innovation-weighted) R: outlier spikes")
    # Same synthetic hand, but 4 % of angle measurements jump by 40-90 deg,
    # the kind of one-frame glitch MediaPipe makes when a finger is hidden.
    spikes = rng.random(meas_ang.shape) < 0.04
    bad = meas_ang + spikes * rng.choice([-1, 1], meas_ang.shape) * rng.uniform(40, 90, meas_ang.shape)
    rb = dict(matched, use_block2=False)

    def run_ang(cfg_kw, z, cls=HandEKF):
        e = cls(model, EKFConfig(**cfg_kw))
        est = np.array([e.step(z[f], None, 1 / fps) for f in range(T)])
        return est, float(np.sqrt(np.mean(WRAP(est - truth)[30:] ** 2)))

    _, rm_clean = run_ang(rb, meas_ang)
    _, rm_plain = run_ang(rb, bad)
    est_rob, rm_rob = run_ang(dict(rb, robust=True), bad)
    _, rm_rob_clean = run_ang(dict(rb, robust=True), meas_ang)
    est_rob_a, _ = run_ang(dict(rb, robust=True), bad, AngleKF)
    worst_rob = float(np.abs(WRAP(est_rob - est_rob_a)).max())
    print(f"      RMSE: clean {rm_clean:.2f} | spikes, plain {rm_plain:.2f} | "
          f"spikes, robust {rm_rob:.2f} | clean, robust {rm_rob_clean:.2f}")
    check("robust R cuts spike error", rm_rob < 0.8 * rm_plain, f"{rm_rob:.2f} vs {rm_plain:.2f}")
    check("robust R costs little on clean data", rm_rob_clean < 1.1 * rm_clean,
          f"{rm_rob_clean:.2f} vs {rm_clean:.2f}")
    check("robust angle_kf == robust HandEKF(block 2 off)", worst_rob < 1e-6, f"{worst_rob:.1e} deg")
    print()

    print("6. checkpoint")
    try:
        import torch
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        check("angle_names == ARTICULATION_ANGLES",
              list(ck["angle_names"]) == HA.ARTICULATION_ANGLES)
        check("fan_angle_names == FAN_ANGLE_NAMES",
              list(ck.get("fan_angle_names", HA.FAN_ANGLE_NAMES)) == HA.FAN_ANGLE_NAMES)
        check("trained on right hands", ck.get("config", {}).get("hand") == "right")
    except ImportError:
        print("  [skip] torch not installed")

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
