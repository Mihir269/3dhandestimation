"""
unity_stream.py -- send the filtered hand to Unity (or anything) over UDP.

One small JSON datagram per processed frame, to 127.0.0.1:5065 by default:

    {"t": 12.345,            seconds since start
     "h": "right",           physical hand
     "d": 1,                 1 = MediaPipe saw the hand this frame, 0 = filter coasting
     "j": [63 floats],       21 joints x (x, y, z), METRES, Unity axes, wrist at origin
     "a": [21 floats],       filtered v12 angles, degrees (ARTICULATION_ANGLES order)
     "w": [u, v]}            wrist position in the displayed image, 0-1 (u right, v down)

Unity axes: x right, y up, z away from the viewer (left-handed). MediaPipe's
world landmarks are x right, y down, z away from the camera, so the
conversion is (x, -y, z): flipping one axis turns MediaPipe's right-handed
frame into Unity's left-handed one without mirroring the hand. The joints are
the hand as SEEN on screen (so a selfie-mirrored webcam stays mirrored), using
the calibrated skeleton and the smoothed hand orientation, exactly what the
3-D panel draws.

Check the stream without Unity:
    python core/unity_stream.py --listen          # prints what arrives on :5065
"""

from __future__ import annotations

import json
import socket

import numpy as np

UNITY_FROM_MP = np.array([1.0, -1.0, 1.0])


class UnityStream:
    def __init__(self, target="127.0.0.1:5065"):
        host, _, port = str(target).rpartition(":")
        self.addr = (host or "127.0.0.1", int(port or 5065))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent = 0

    def send(self, t, hand, detected, joints_mm_mp, angles=None, wrist_uv=None):
        """joints_mm_mp: (21,3) mm, MediaPipe axes, wrist at origin, as displayed."""
        if joints_mm_mp is None:
            msg = {"t": round(float(t), 4), "h": hand, "d": 0, "j": [], "a": [], "w": []}
        else:
            j = np.asarray(joints_mm_mp, float) / 1000.0 * UNITY_FROM_MP
            msg = {"t": round(float(t), 4), "h": hand, "d": int(bool(detected)),
                   "j": [round(float(v), 5) for v in j.ravel()],
                   "a": [] if angles is None else [round(float(v), 2) for v in angles],
                   "w": [] if wrist_uv is None else [round(float(v), 4) for v in wrist_uv]}
        try:
            self.sock.sendto(json.dumps(msg, separators=(",", ":")).encode(), self.addr)
            self.sent += 1
        except OSError:
            pass                     # nobody listening is fine for UDP

    def close(self):
        self.sock.close()


def _listen(port, n):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", port))
    print(f"listening on :{port} (Ctrl+C to stop)")
    for i in range(n or 10 ** 12):
        data, _ = s.recvfrom(65535)
        m = json.loads(data)
        j = np.array(m["j"]).reshape(-1, 3) if m["j"] else None
        tip = "" if j is None else (f"  index tip {np.round(j[8], 3)}  "
                                    f"hand length {np.linalg.norm(j[12] - j[0]) * 100:.1f} cm")
        print(f"t {m['t']:8.3f}  {m['h']:5s}  detected {m['d']}{tip}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", action="store_true")
    ap.add_argument("--port", type=int, default=5065)
    ap.add_argument("-n", type=int, default=0, help="stop after n packets")
    a = ap.parse_args()
    if a.listen:
        _listen(a.port, a.n)
