// ProceduralHand.cs -- a solid, lit 3-D hand built from capsules, spheres and a
// palm mesh, driven by HandUdpReceiver. No model or rig needed: drop it on an
// empty GameObject, assign the receiver, press Play.
//
// The joints arrive in the hand's own frame (metres, wrist at origin), so this
// GameObject's transform is where the hand sits in the scene; move, rotate or
// scale it freely. Optional: followImage moves it with the wrist in the camera
// image, so the hand travels across the scene as it does on screen.

using UnityEngine;

public class ProceduralHand : MonoBehaviour
{
    public HandUdpReceiver receiver;

    [Header("Look")]
    public Material skinMaterial;                 // optional; a skin-tone Lit material is made if empty
    public Color skinColor = new Color(0.93f, 0.76f, 0.65f);
    public bool colourFingers = false;            // the Python view's per-finger colours
    [Range(0.004f, 0.02f)] public float boneRadius = 0.0085f;   // metres
    [Range(0.002f, 0.02f)] public float palmThickness = 0.012f;

    [Header("Motion")]
    [Tooltip("Extra display smoothing on top of the Kalman filter (0 = none). " +
             "Usually not needed: the filter already smooths.")]
    [Range(0f, 0.95f)] public float extraSmoothing = 0f;
    [Tooltip("Hide the hand when no packet has arrived for this long (s)")]
    public float hideAfter = 0.5f;
    [Tooltip("Set if the hand's depth looks inverted (fingers curl the wrong way)")]
    public bool flipZ = false;
    [Tooltip("Move with the wrist's position in the camera image")]
    public bool followImage = false;
    public Vector2 imageRange = new Vector2(0.6f, 0.35f);   // metres for the full image width/height

    // MediaPipe's 21 joints: wrist 0; thumb 1-4; index 5-8; middle 9-12; ring 13-16; pinky 17-20
    static readonly int[,] Bones = {
        {0,1},{1,2},{2,3},{3,4},          // thumb
        {0,5},{5,6},{6,7},{7,8},          // index
        {9,10},{10,11},{11,12},           // middle (0-9 is covered by the palm)
        {13,14},{14,15},{15,16},          // ring
        {0,17},{17,18},{18,19},{19,20},   // pinky
        {5,9},{9,13},{13,17}              // knuckle line
    };
    static readonly int[] PalmOutline = { 0, 1, 5, 9, 13, 17 };
    static readonly Color[] FingerColours = {
        new Color(1f, 0.42f, 0.42f), new Color(0.31f, 0.80f, 0.77f), new Color(0.27f, 0.72f, 0.82f),
        new Color(0.59f, 0.81f, 0.71f), new Color(1f, 0.92f, 0.65f)
    };

    readonly Transform[] _joints = new Transform[21];
    Transform[] _bones;
    MeshFilter _palm;
    Mesh _palmMesh;
    readonly Vector3[] _p = new Vector3[21];
    bool _havePose;
    Vector3 _basePos;
    Renderer[] _renderers;
    bool _visible = true;

    void Start()
    {
        if (receiver == null) receiver = FindObjectOfType<HandUdpReceiver>();
        _basePos = transform.localPosition;
        Material skin = skinMaterial != null ? skinMaterial : MakeMaterial(skinColor);

        for (int i = 0; i < 21; i++)
        {
            var s = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            Destroy(s.GetComponent<Collider>());
            s.name = $"joint_{i}";
            s.transform.SetParent(transform, false);
            float r = boneRadius * (i == 0 ? 1.6f : IsTip(i) ? 0.95f : 1.12f) * FingerScale(i);
            s.transform.localScale = Vector3.one * 2f * r;
            s.GetComponent<Renderer>().sharedMaterial = PickMaterial(skin, i);
            _joints[i] = s.transform;
        }
        _bones = new Transform[Bones.GetLength(0)];
        for (int b = 0; b < _bones.Length; b++)
        {
            var c = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            Destroy(c.GetComponent<Collider>());
            c.name = $"bone_{Bones[b, 0]}_{Bones[b, 1]}";
            c.transform.SetParent(transform, false);
            c.GetComponent<Renderer>().sharedMaterial = PickMaterial(skin, Bones[b, 1]);
            _bones[b] = c.transform;
        }
        var palm = new GameObject("palm");
        palm.transform.SetParent(transform, false);
        _palm = palm.AddComponent<MeshFilter>();
        palm.AddComponent<MeshRenderer>().sharedMaterial = skin;
        _palmMesh = new Mesh { name = "palm" };
        _palmMesh.MarkDynamic();
        _palm.sharedMesh = _palmMesh;
        SetVisible(false);
    }

    void Update()
    {
        var pkt = receiver != null ? receiver.Latest : null;
        bool fresh = receiver != null && receiver.LastPacketTime >= 0 &&
                     Time.realtimeSinceStartup - receiver.LastPacketTime < hideAfter;
        if (pkt == null || !pkt.HasJoints || !fresh)
        {
            SetVisible(false);
            return;
        }
        float a = _havePose ? extraSmoothing : 0f;
        for (int i = 0; i < 21; i++)
        {
            Vector3 q = pkt.Joint(i);
            if (flipZ) q.z = -q.z;
            _p[i] = Vector3.Lerp(q, _p[i], a);
        }
        _havePose = true;
        if (followImage && pkt.w != null && pkt.w.Length == 2)
            transform.localPosition = _basePos + new Vector3((pkt.w[0] - 0.5f) * imageRange.x,
                                                             (0.5f - pkt.w[1]) * imageRange.y, 0f);
        Pose();
        SetVisible(true);
    }

    void Pose()
    {
        for (int i = 0; i < 21; i++) _joints[i].localPosition = _p[i];
        for (int b = 0; b < _bones.Length; b++)
        {
            Vector3 p0 = _p[Bones[b, 0]], p1 = _p[Bones[b, 1]];
            Vector3 d = p1 - p0;
            float len = d.magnitude;
            Transform t = _bones[b];
            if (len < 1e-5f) { t.gameObject.SetActive(false); continue; }
            t.gameObject.SetActive(true);
            float r = boneRadius * FingerScale(Bones[b, 1]) * (b >= _bones.Length - 3 ? 0.9f : 1f);
            t.localPosition = 0.5f * (p0 + p1);
            t.localRotation = Quaternion.FromToRotation(Vector3.up, d / len);
            // Unity's capsule: height 2 along y, radius 0.5
            t.localScale = new Vector3(2f * r, 0.5f * len, 2f * r);
        }
        BuildPalm();
    }

    void BuildPalm()
    {
        int n = PalmOutline.Length;
        Vector3 c = Vector3.zero;
        foreach (int i in PalmOutline) c += _p[i];
        c /= n;
        // palm normal from the wrist->index and wrist->pinky directions
        Vector3 normal = Vector3.Cross(_p[5] - _p[0], _p[17] - _p[0]).normalized;
        Vector3 off = normal * (0.5f * palmThickness);

        var v = new Vector3[2 * (n + 1)];
        v[0] = c + off;
        v[n + 1] = c - off;
        for (int k = 0; k < n; k++)
        {
            v[1 + k] = _p[PalmOutline[k]] + off;
            v[n + 2 + k] = _p[PalmOutline[k]] - off;
        }
        // two fans (front and back) plus the rim, so it reads as a solid slab
        var tris = new int[3 * n * 4];
        int ti = 0;
        for (int k = 0; k < n; k++)
        {
            int a = 1 + k, b = 1 + (k + 1) % n;
            tris[ti++] = 0; tris[ti++] = a; tris[ti++] = b;                       // front
            tris[ti++] = n + 1; tris[ti++] = n + 1 + b; tris[ti++] = n + 1 + a;   // back
            tris[ti++] = a; tris[ti++] = n + 1 + a; tris[ti++] = b;               // rim
            tris[ti++] = b; tris[ti++] = n + 1 + a; tris[ti++] = n + 1 + b;
        }
        _palmMesh.Clear();
        _palmMesh.vertices = v;
        _palmMesh.triangles = tris;
        _palmMesh.RecalculateNormals();
        _palmMesh.RecalculateBounds();
    }

    void SetVisible(bool on)
    {
        if (_renderers == null) _renderers = GetComponentsInChildren<Renderer>(true);
        if (_visible == on) return;
        _visible = on;
        foreach (var r in _renderers) r.enabled = on;
    }

    static bool IsTip(int i) => i == 4 || i == 8 || i == 12 || i == 16 || i == 20;

    static float FingerScale(int i)
    {
        if (i == 0) return 1.15f;
        if (i <= 4) return 1.15f;      // thumb
        if (i >= 17) return 0.85f;     // pinky
        return 1f;
    }

    Material PickMaterial(Material skin, int joint)
    {
        if (!colourFingers || joint == 0) return skin;
        var m = new Material(skin) { color = FingerColours[(joint - 1) / 4] };
        return m;
    }

    static Material MakeMaterial(Color col)
    {
        // works in the built-in pipeline, URP and HDRP
        Shader sh = Shader.Find("Universal Render Pipeline/Lit")
                    ?? Shader.Find("HDRP/Lit")
                    ?? Shader.Find("Standard");
        var m = new Material(sh) { color = col };
        if (m.HasProperty("_Smoothness")) m.SetFloat("_Smoothness", 0.35f);
        if (m.HasProperty("_Glossiness")) m.SetFloat("_Glossiness", 0.35f);
        if (m.HasProperty("_Cull")) m.SetFloat("_Cull", 0f);   // double sided (URP/Standard ignore if absent)
        return m;
    }
}
