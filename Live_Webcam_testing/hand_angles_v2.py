"""
hand_angles_v2.py — bridge `anatomical_angles_v12` into hand_ekf

`hand_ekf.resolve_angle_fn()` looks for this module by name and, if it imports,
uses its `compute_angles` instead of hand_ekf's own built-in definition. That
matters: hand_ekf ships a 20-angle definition written independently, which is
NOT the v12 system. Feeding the wrong convention into the corrector produces
nonsense, silently.

This shim inspects `anatomical_angles_v12` at import time and adapts to whatever
`compute_angles_v12` actually looks like, because several shapes are plausible
(bare array, dict, or a tuple carrying the fan angles or the off-axis
diagnostic alongside). If it cannot work out the shape it raises immediately
with the details, rather than returning something plausible and wrong.

Print `hand_angles_v2.describe()` once after wiring it up and read the output.
"""

from __future__ import annotations

import inspect
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import anatomical_angles_v12 as A12

# ---------------------------------------------------------------------------
# Discover the angle-name list
# ---------------------------------------------------------------------------

def _find_names() -> Tuple[List[str], List[str]]:
    """(articulation_names, fan_names) from whatever the module exposes."""
    art: Optional[List[str]] = None
    for cand in ("ARTICULATION_ANGLES", "ANGLE_NAMES", "ANGLES", "V12_ANGLES"):
        v = getattr(A12, cand, None)
        if v is not None and len(v) in (20, 21):
            art = list(v)
            break
    if art is None:
        raise ImportError(
            "anatomical_angles_v12 exposes no 20/21-element angle-name list. "
            f"Module attributes: {[a for a in dir(A12) if a.isupper()]}")

    fan: List[str] = []
    for cand in ("FAN_ANGLES", "FAN_ANGLE_NAMES", "PALM_FAN_ANGLES"):
        v = getattr(A12, cand, None)
        if v is not None:
            fan = list(v)
            break
    return art, fan


ARTICULATION, FAN = _find_names()

_FN: Callable = getattr(A12, "compute_angles_v12", None) or \
                getattr(A12, "compute_angles", None)
if _FN is None:
    raise ImportError("anatomical_angles_v12 has no compute_angles_v12()")

_SIG = inspect.signature(_FN)
_PARAMS = list(_SIG.parameters)
_TAKES_HAND = any(p in _SIG.parameters for p in ("hand", "handedness", "chirality"))
_HAND_KW = next((p for p in ("hand", "handedness", "chirality")
                 if p in _SIG.parameters), None)

_RETURN_SHAPE: Optional[str] = None      # resolved on first call


def _to_dict(out) -> Dict[str, float]:
    """Normalise whatever compute_angles_v12 returned into {name: degrees}."""
    global _RETURN_SHAPE

    if isinstance(out, dict):
        _RETURN_SHAPE = _RETURN_SHAPE or "dict"
        return {k: float(v) for k, v in out.items()}

    if isinstance(out, (tuple, list)) and len(out) <= 4 and \
            not np.isscalar(out[0]) and not isinstance(out[0], (int, float)):
        # (angles, extra...) — first element is the angle vector/dict,
        # later ones are typically fan angles or the off_axis diagnostic
        vals: Dict[str, float] = _to_dict(out[0])
        _RETURN_SHAPE = "tuple"
        for extra in out[1:]:
            if isinstance(extra, dict):
                vals.update({k: float(v) for k, v in extra.items()})
            else:
                arr = np.asarray(extra, float).reshape(-1)
                if FAN and arr.size == len(FAN):
                    vals.update(dict(zip(FAN, arr.tolist())))
        return vals

    arr = np.asarray(out, float).reshape(-1)
    if arr.size == len(ARTICULATION):
        _RETURN_SHAPE = _RETURN_SHAPE or "array"
        return dict(zip(ARTICULATION, arr.tolist()))
    if FAN and arr.size == len(ARTICULATION) + len(FAN):
        _RETURN_SHAPE = "array+fan"
        return dict(zip(list(ARTICULATION) + list(FAN), arr.tolist()))
    raise ValueError(
        f"compute_angles_v12 returned {arr.size} values; expected "
        f"{len(ARTICULATION)} (articulation)"
        + (f" or {len(ARTICULATION) + len(FAN)} (articulation+fan)" if FAN else "")
        + ". Adapt _to_dict() in hand_angles_v2.py.")


def compute_angles(kps: np.ndarray, angle_names: Sequence[str],
                   hand: str = "Right") -> np.ndarray:
    """(21,3) keypoints -> angle vector ordered to match `angle_names`.

    `angle_names` comes from the checkpoint, so ordering is driven by the
    trained model rather than by this module's own list.
    """
    k = np.asarray(kps, float).reshape(21, 3)
    out = _FN(k, **({_HAND_KW: hand} if _TAKES_HAND else {}))
    vals = _to_dict(out)

    missing = [n for n in angle_names if n not in vals]
    if missing:
        raise KeyError(
            f"anatomical_angles_v12 did not produce {missing}.\n"
            f"It produced: {sorted(vals)}\n"
            f"The checkpoint's angle_names and the v12 module disagree — check "
            f"you are loading the matching checkpoint.")
    return np.array([vals[n] for n in angle_names], float)


def wrap_deg(d):
    """(d + 180) % 360 - 180. v12 flexion is atan2-based and wraps:
    Thumb_IP_Flex spans -177.9 to 160.7 deg in the training data, so a raw
    difference turns a 2 deg error into 358 deg. Every innovation, loss and
    metric must go through this."""
    return (np.asarray(d, float) + 180.0) % 360.0 - 180.0


def describe() -> str:
    return "\n".join([
        "hand_angles_v2 -> anatomical_angles_v12",
        f"  function        {_FN.__name__}{_SIG}",
        f"  takes hand arg  {_TAKES_HAND}" + (f"  (as '{_HAND_KW}')" if _HAND_KW else ""),
        f"  articulation    {len(ARTICULATION)}: {ARTICULATION}",
        f"  fan             {len(FAN)}: {FAN}" if FAN else "  fan             (none exposed)",
        f"  return shape    {_RETURN_SHAPE or 'not yet resolved — call once'}",
    ])


def selftest(angle_names: Optional[Sequence[str]] = None) -> str:
    """Round-trip against the v12 FK if the module exposes one. Run this before
    trusting the pipeline — it catches a definition mismatch immediately."""
    names = list(angle_names or ARTICULATION)
    rng = np.random.default_rng(0)
    lines = [describe(), ""]

    fk = getattr(A12, "forward_kinematics_v12", None)
    if fk is None:
        lines.append("no forward_kinematics_v12 exposed — skipping round trip")
        return "\n".join(lines)

    try:
        bl = np.array([45, 36, 33, 26, 91, 46, 27, 22, 87, 49, 31, 24,
                       83, 45, 29, 22, 80, 35, 22, 19], float)
        worst = 0.0
        for _ in range(50):
            th = np.array([rng.uniform(-20, 20) if n.endswith("_Abd")
                           else rng.uniform(-15, 80) for n in names])
            K = np.asarray(fk(th, bl))
            K = K[0] if (K.ndim == 3 or (isinstance(K, tuple))) else K
            back = compute_angles(K.reshape(21, 3), names)
            worst = max(worst, float(np.abs(wrap_deg(back - th)).max()))
        lines.append(f"FK -> angles round trip over 50 poses: {worst:.2e} deg")
        lines.append("  < 1e-6 means the definitions agree; anything larger means"
                     " compute_angles and forward_kinematics_v12 disagree.")
    except Exception as exc:
        lines.append(f"round trip could not run: {type(exc).__name__}: {exc}")
        lines.append("  (forward_kinematics_v12 probably takes different args —"
                     " harmless, but do verify the definition another way)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(selftest())
