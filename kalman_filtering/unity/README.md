# Unity viewer for the filtered hand

The Python pipeline does the tracking: MediaPipe, the v12 corrector and the
EKF. It streams the result over UDP, and Unity only draws it, so you get a
lit, solid 3-D hand instead of the OpenCV line drawing. Unity needs no Python,
no MediaPipe and no camera access.

```
webcam → run_ekf_corrected.py --unity ──UDP :5065──► Unity: HandUdpReceiver → ProceduralHand / RiggedHandDriver
```

## 1 · Python side

```powershell
cd kalman_filtering
..\Live_Webcam_testing\myenv\Scripts\Activate.ps1
python run_ekf_corrected.py --unity                     # webcam → 127.0.0.1:5065
python run_ekf_corrected.py --unity 192.168.1.20:5065   # Unity on another machine
python run_ekf_corrected.py --source ..\Live_Webcam_testing\clip.mp4 --unity   # replay a video
python core\unity_stream.py --listen                    # check the stream without Unity
```

The OpenCV window still opens, and every filter flag works as usual
(`--robust`, `--jerk-psd`, ...). Each packet holds the 21 joints in metres
in Unity's axes (x right, y up, z away from you), with the wrist at the
origin. It also holds the 21 filtered angles, the physical hand, a
detected/coasting flag and the wrist's position in the image. The format is
documented in `core/unity_stream.py`.

## 2 · Unity side (any Unity 2021.3 or newer, including Unity 6; built-in, URP or HDRP)

1. Create a project, or open an existing one.
2. Copy `unity/Assets/HandTracking/` into the project's `Assets/` folder.
3. In a scene:
   - Create an empty GameObject `HandReceiver` and add **HandUdpReceiver**.
     The port must match the Python side (5065).
   - Create an empty GameObject `Hand` at, say, position (0, 1, 0.5) and add
     **ProceduralHand**. Drag `HandReceiver` into its *Receiver* slot (it also
     finds it automatically).
   - Point the Main Camera at the hand. For example, camera at (0, 1.1, 0.1)
     looking toward +z.
4. Press **Play**, then start the Python script. The top-left label shows
   packets per second and whether the hand is tracked or coasting.

Windows may ask to allow Unity through the firewall the first time. Allow it
on private networks.

### ProceduralHand options

| option | effect |
|---|---|
| Skin Material / Skin Colour | the hand's look; leave the material empty for an automatic skin-tone Lit material |
| Colour Fingers | the Python view's per-finger colours |
| Bone Radius, Palm Thickness | how thick the fingers and palm are |
| Extra Smoothing | display smoothing on top of the Kalman filter; normally 0 |
| Flip Z | use it if the fingers curl the wrong way (depth inverted) |
| Follow Image | move the hand around the scene with the wrist's position in the camera image |

The GameObject's transform places, turns and scales the whole hand.

### A realistic skinned hand (RiggedHandDriver)

For a proper mesh hand, import any rigged hand model, for example a free hand
from the Asset Store or a Mixamo character's hand. Add **RiggedHandDriver** to
a parent of the model, then assign:

- **Wrist**: the hand's root bone.
- **Thumb / Index / Middle / Ring / Pinky**: 4 transforms each, from the base
  joint to the fingertip. If the rig has no fingertip bone, add an empty child
  at the tip of each finger.

Keep the model in its rest pose when you press Play: the rest directions are
recorded in `Start()`. Every bone is then turned to point along the streamed
joint directions. The model keeps its own bone lengths, so any hand size works.

| option | when to use it |
|---|---|
| Mirror X | the rig is the other hand from the data. With the webcam's selfie view on (the default), your right hand arrives as a mirror image, so a right-hand rig needs Mirror X. Or run Python with `--no-mirror` |
| Flip Z | the fingers bend backwards |
| Drive Wrist | off = only the fingers move, and the hand keeps the orientation you gave it |

## 3 · Notes

- **Latency.** UDP on localhost adds well under 1 ms. What you see is the
  Python frame rate (about 25 fps with the threaded MediaPipe) plus the
  filter's lag (about 2 frames).
- **When the hand is lost.** While MediaPipe has lost the hand, the filter
  coasts for up to 8 frames (`d = 0` in the packet). After that Python stops
  sending joints, and Unity hides the hand after `Hide After` seconds.
- **Untested here.** These scripts were written without a Unity editor on the
  development machine, so they have not been compiled. The Python sender was
  tested with `core/unity_stream.py --listen`. If Unity reports a compile
  error, it will be a small API difference. One known example:
  `FindObjectOfType` is marked obsolete in Unity 2023+. That is only a
  warning; you can replace it with `FindFirstObjectByType`.
