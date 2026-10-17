// RiggedHandDriver.cs -- drives a skinned (rigged) hand model with the stream,
// for a realistic hand instead of capsules. Works with any rig that has one
// transform per finger joint, e.g. a hand from the Asset Store or Mixamo.
//
// Setup: assign the wrist bone and, per finger, the 3 joint bones plus the
// fingertip (base -> tip). If the rig has no fingertip bone, add an empty
// child GameObject at the fingertip. Leave the model in its rest pose in the
// scene: the rest directions are recorded in Start().
//
// Each frame, every bone is turned so it points from its joint to the next
// one, as in the stream (minimum rotation from its rest direction, so the
// rig's own twist is kept). The wrist is turned to match the palm's frame.
// Bone lengths stay the model's own: only directions are taken from the data.

using UnityEngine;

public class RiggedHandDriver : MonoBehaviour
{
    public HandUdpReceiver receiver;
    [Tooltip("Wrist / hand root bone of the rig")]
    public Transform wrist;
    [Tooltip("Thumb CMC, MCP, IP, tip")] public Transform[] thumb = new Transform[4];
    [Tooltip("Index MCP, PIP, DIP, tip")] public Transform[] index = new Transform[4];
    [Tooltip("Middle MCP, PIP, DIP, tip")] public Transform[] middle = new Transform[4];
    [Tooltip("Ring MCP, PIP, DIP, tip")] public Transform[] ring = new Transform[4];
    [Tooltip("Pinky MCP, PIP, DIP, tip")] public Transform[] pinky = new Transform[4];

    [Tooltip("Turn the whole hand with the data (off = only the fingers move)")]
    public bool driveWrist = true;
    [Tooltip("Set if the hand's depth looks inverted")]
    public bool flipZ = false;
    [Tooltip("Mirror the data left<->right, for a rig of the other hand")]
    public bool mirrorX = false;
    [Range(0f, 0.95f)] public float extraSmoothing = 0f;

    // MediaPipe joint indices of each finger chain, base -> tip
    static readonly int[][] Chains = {
        new[] { 1, 2, 3, 4 }, new[] { 5, 6, 7, 8 }, new[] { 9, 10, 11, 12 },
        new[] { 13, 14, 15, 16 }, new[] { 17, 18, 19, 20 } };

    Transform[][] _chains;
    Quaternion[][] _restRot;          // bone rotation relative to the wrist, at rest
    Vector3[][] _restDir;             // bone -> child direction in wrist space, at rest
    Quaternion _wristRest;            // wrist rotation relative to this transform
    Quaternion _palmRestInv;          // inverse of the rest palm frame in wrist space
    readonly Vector3[] _p = new Vector3[21];
    bool _ready, _havePose;

    void Start()
    {
        if (receiver == null) receiver = FindObjectOfType<HandUdpReceiver>();
        _chains = new[] { thumb, index, middle, ring, pinky };
        if (wrist == null) { Debug.LogError("RiggedHandDriver: assign the wrist bone"); return; }
        foreach (var c in _chains)
            if (c == null || c.Length != 4 || System.Array.Exists(c, t => t == null))
            { Debug.LogError("RiggedHandDriver: every finger needs 4 transforms (3 joints + tip)"); return; }

        Quaternion wInv = Quaternion.Inverse(wrist.rotation);
        _restRot = new Quaternion[5][];
        _restDir = new Vector3[5][];
        for (int f = 0; f < 5; f++)
        {
            _restRot[f] = new Quaternion[3];
            _restDir[f] = new Vector3[3];
            for (int k = 0; k < 3; k++)
            {
                _restRot[f][k] = wInv * _chains[f][k].rotation;
                _restDir[f][k] = (wInv * (_chains[f][k + 1].position - _chains[f][k].position)).normalized;
            }
        }
        _wristRest = Quaternion.Inverse(transform.rotation) * wrist.rotation;
        // rest palm frame in wrist space, from the rig's own knuckles
        Vector3 w = wrist.position;
        Quaternion palmRest = PalmFrame(wInv * (index[0].position - w), wInv * (pinky[0].position - w),
                                        wInv * (middle[0].position - w));
        _palmRestInv = Quaternion.Inverse(palmRest);
        _ready = true;
    }

    static Quaternion PalmFrame(Vector3 toIndex, Vector3 toPinky, Vector3 toMiddle)
    {
        Vector3 fwd = toMiddle.normalized;                       // wrist -> middle knuckle
        Vector3 across = (toIndex - toPinky).normalized;         // pinky -> index
        Vector3 normal = Vector3.Cross(fwd, across).normalized;  // palm normal
        return Quaternion.LookRotation(normal, fwd);
    }

    void LateUpdate()
    {
        if (!_ready) return;
        var pkt = receiver != null ? receiver.Latest : null;
        if (pkt == null || !pkt.HasJoints) return;
        float a = _havePose ? extraSmoothing : 0f;
        for (int i = 0; i < 21; i++)
        {
            Vector3 q = pkt.Joint(i);
            if (flipZ) q.z = -q.z;
            if (mirrorX) q.x = -q.x;
            _p[i] = Vector3.Lerp(q, _p[i], a);
        }
        _havePose = true;

        // Data is in this GameObject's local frame. The palm's data frame gives
        // the wrist rotation: R_wrist = R_palm_data * inverse(R_palm_rest).
        Quaternion palmData = PalmFrame(_p[5] - _p[0], _p[17] - _p[0], _p[9] - _p[0]);
        Quaternion wristLocal = driveWrist ? palmData * _palmRestInv : _wristRest;
        wrist.rotation = transform.rotation * wristLocal;

        Quaternion toWorld = wrist.rotation;
        Quaternion dataToWrist = Quaternion.Inverse(wristLocal);  // data (local) -> wrist space
        for (int f = 0; f < 5; f++)
        {
            int[] ids = Chains[f];
            for (int k = 0; k < 3; k++)
            {
                Vector3 d = _p[ids[k + 1]] - _p[ids[k]];
                if (d.sqrMagnitude < 1e-10f) continue;
                Vector3 target = (dataToWrist * d).normalized;           // in wrist space
                Quaternion turn = Quaternion.FromToRotation(_restDir[f][k], target);
                _chains[f][k].rotation = toWorld * (turn * _restRot[f][k]);
            }
        }
    }
}
