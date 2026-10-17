// HandUdpReceiver.cs -- receives the filtered hand from kalman_filtering over UDP.
//
// Python side:  python run_ekf_corrected.py --unity            (sends to 127.0.0.1:5065)
// One JSON datagram per frame (see core/unity_stream.py):
//   t  seconds, h "right"/"left", d 1 = detected / 0 = filter coasting,
//   j  63 floats = 21 joints (x, y, z) in metres, Unity axes, wrist at origin,
//   a  21 filtered joint angles (deg), w  wrist position in the image (0-1).
//
// A background thread does the blocking socket read; Update() code reads
// Latest, which is swapped under a lock, so the main thread never waits.

using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

[Serializable]
public class HandPacket
{
    public float t;
    public string h;
    public int d;
    public float[] j;
    public float[] a;
    public float[] w;

    public bool HasJoints => j != null && j.Length >= 63;

    public Vector3 Joint(int i) => new Vector3(j[3 * i], j[3 * i + 1], j[3 * i + 2]);
}

public class HandUdpReceiver : MonoBehaviour
{
    [Tooltip("Must match --unity HOST:PORT on the Python side")]
    public int port = 5065;
    [Tooltip("Show packet rate and status in the corner of the Game view")]
    public bool showStatus = true;

    public HandPacket Latest { get { lock (_lock) return _latest; } }
    public int Received { get { lock (_lock) return _count; } }
    /// <summary>Time.realtimeSinceStartup of the newest packet (main-thread clock).</summary>
    public float LastPacketTime { get; private set; } = -1f;

    UdpClient _client;
    Thread _thread;
    volatile bool _running;
    readonly object _lock = new object();
    HandPacket _latest;
    int _count, _seenCount;
    float _rate, _rateClock;
    int _rateCount;

    void OnEnable()
    {
        try
        {
            _client = new UdpClient(port);
            _client.Client.ReceiveTimeout = 500;      // lets the thread notice shutdown
        }
        catch (SocketException e)
        {
            Debug.LogError($"HandUdpReceiver: cannot open UDP port {port}: {e.Message}");
            enabled = false;
            return;
        }
        _running = true;
        _thread = new Thread(ReceiveLoop) { IsBackground = true, Name = "HandUdpReceiver" };
        _thread.Start();
    }

    void ReceiveLoop()
    {
        var from = new IPEndPoint(IPAddress.Any, 0);
        while (_running)
        {
            try
            {
                byte[] data = _client.Receive(ref from);
                // JsonUtility may be called from background threads.
                var p = JsonUtility.FromJson<HandPacket>(Encoding.UTF8.GetString(data));
                lock (_lock)
                {
                    _latest = p;
                    _count++;
                }
            }
            catch (SocketException) { /* timeout: loop and check _running */ }
            catch (ObjectDisposedException) { break; }
            catch (Exception e) { Debug.LogWarning($"HandUdpReceiver: bad packet ({e.Message})"); }
        }
    }

    void Update()
    {
        int c = Received;
        if (c != _seenCount)
        {
            _rateCount += c - _seenCount;
            _seenCount = c;
            LastPacketTime = Time.realtimeSinceStartup;
        }
        _rateClock += Time.unscaledDeltaTime;
        if (_rateClock >= 1f)
        {
            _rate = _rateCount / _rateClock;
            _rateCount = 0;
            _rateClock = 0f;
        }
    }

    void OnDisable()
    {
        _running = false;
        _client?.Close();
        if (_thread != null && _thread.IsAlive)
            _thread.Join(600);
        _client = null;
        _thread = null;
    }

    void OnGUI()
    {
        if (!showStatus) return;
        var p = Latest;
        string state = p == null ? "waiting for Python (run with --unity)"
            : $"{p.h} hand  {(p.d == 1 ? "tracked" : "coasting")}  t={p.t:F1}s";
        GUI.Label(new Rect(10, 10, 600, 22), $"UDP :{port}   {_rate:F0} packets/s   {state}");
    }
}
