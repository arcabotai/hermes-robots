"""``ros``: any ROS 1 / ROS 2 robot (TurtleBot3 in Gazebo, a real base, turtlesim, ...) over a rosbridge websocket.

The Hermes host needs no ROS install: a small built-in rosbridge v2 client (``rosbridge_client.py``) imports
``websocket-client`` lazily and connects on the first tool call, never while the plugin registers.

Generic tools map to standard topics: ``robot_move`` publishes ``cmd_vel`` (Twist or TwistStamped) at 10 Hz for the
vetted duration and then zero velocity, aborting on e-stop or a runtime geofence breach; poses come from a persistent
``odom`` subscription; ``robot_look`` grabs one camera frame (Image or CompressedImage). ROS-specific extras go through
``robot_action``: ``topics``, ``echo`` (read) and ``publish``, ``call_service`` (write, allowlisted).

Ported from arcabotai/hermes-ros.
"""
from __future__ import annotations

import copy
import fnmatch
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from .. import imaging
from .base import CAMERA, MOVE, POSE, READ, WRITE, Action, BackendError, Blocked, MoveContext, RobotBackend, num
from .rosbridge_client import RosbridgeClient

ZERO_TWIST = {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}}
TWIST_TYPES = {"geometry_msgs/Twist", "geometry_msgs/msg/Twist", "geometry_msgs/TwistStamped", "geometry_msgs/msg/TwistStamped"}
TOPIC_DEFAULTS = {"cmd_vel": "/cmd_vel", "cmd_vel_type": "geometry_msgs/Twist", "odom": "/odom",
                  "odom_type": "nav_msgs/Odometry", "camera": "/camera/image_raw"}


def norm_name(name: Any) -> str:
    n = str(name or "").strip()
    return n if n.startswith("/") else "/" + n


def matches_any(name: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, norm_name(p)) for p in (patterns or []))


def extract_twist(message: Any) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """((lx, ly, lz), (ax, ay, az)) from a Twist or TwistStamped dict, else None."""
    if not isinstance(message, dict):
        return None
    tw = message.get("twist") if isinstance(message.get("twist"), dict) and "linear" not in message else message
    if not isinstance(tw, dict) or ("linear" not in tw and "angular" not in tw):
        return None

    def vec(key: str) -> Tuple[float, float, float]:
        v = tw.get(key) or {}
        if not isinstance(v, dict):
            raise Blocked("invalid_args", f"twist.{key} must be an object with x/y/z")
        return tuple(num(v.get(c, 0.0), f"twist.{key}.{c}", 0.0) for c in "xyz")  # type: ignore[return-value]

    return vec("linear"), vec("angular")


def set_twist(message: Dict[str, Any], vx: float, wz: float) -> Dict[str, Any]:
    out = copy.deepcopy(message)
    tw = out["twist"] if isinstance(out.get("twist"), dict) and "linear" not in out else out
    tw["linear"] = {"x": vx, "y": 0.0, "z": 0.0}
    tw["angular"] = {"x": 0.0, "y": 0.0, "z": wz}
    return out


def _quat_to_yaw(q: Dict[str, float]) -> float:
    x, y, z, w = (float(q.get(k, 0.0)) for k in ("x", "y", "z", "w"))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def parse_pose(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise nav_msgs/Odometry, PoseStamped, PoseWithCovarianceStamped or turtlesim/Pose to one dict."""
    out: Dict[str, Any] = {"frame_id": None, "child_frame_id": msg.get("child_frame_id")}
    hdr = msg.get("header") or {}
    out["frame_id"] = hdr.get("frame_id")
    if "theta" in msg and "x" in msg:  # turtlesim/Pose
        out.update(x=float(msg["x"]), y=float(msg["y"]), yaw_rad=float(msg["theta"]),
                   linear_vel=float(msg.get("linear_velocity", 0.0)), angular_vel=float(msg.get("angular_velocity", 0.0)))
        out["frame_id"] = out["frame_id"] or "turtlesim"
    else:
        pose = msg.get("pose")
        if isinstance(pose, dict) and "pose" in pose:  # Odometry / PoseWithCovarianceStamped
            pose = pose["pose"]
        if not isinstance(pose, dict) or "position" not in pose:
            raise BackendError("message has no pose (expected nav_msgs/Odometry, PoseStamped or turtlesim/Pose)")
        out["x"], out["y"] = float(pose["position"]["x"]), float(pose["position"]["y"])
        out["yaw_rad"] = _quat_to_yaw(pose.get("orientation") or {})
        tw = (msg.get("twist") or {}).get("twist") or {}
        out["linear_vel"] = float((tw.get("linear") or {}).get("x", 0.0))
        out["angular_vel"] = float((tw.get("angular") or {}).get("z", 0.0))
    out["yaw_rad"] = round(out["yaw_rad"], 4)
    out["yaw_deg"] = round(math.degrees(out["yaw_rad"]), 1)
    out["x"], out["y"] = round(out["x"], 4), round(out["y"], 4)
    return out


def summarize(obj: Any, max_str: int = 160, max_list: int = 16, depth: int = 0) -> Any:
    """Shrink big ROS messages (image data, covariance matrices) so they fit in the model context."""
    if isinstance(obj, dict):
        return {k: summarize(v, max_str, max_list, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if len(obj) > max_list:
            return [summarize(v, max_str, max_list, depth + 1) for v in obj[:max_list]] + [f"… (+{len(obj) - max_list} more)"]
        return [summarize(v, max_str, max_list, depth + 1) for v in obj]
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[:max_str] + f"… ({len(obj)} chars)"
    if isinstance(obj, float):
        return round(obj, 5)
    return obj


_SECTORS = (("front", 0.0), ("front_left", 45.0), ("left", 90.0), ("back_left", 135.0),
            ("back", 180.0), ("back_right", -135.0), ("right", -90.0), ("front_right", -45.0))


def scan_summary(msg: Any) -> Optional[Dict[str, Any]]:
    """Digest a sensor_msgs/LaserScan into what a model can use: nearest obstacle and the clearance per 45-degree sector.

    Angles follow REP-103 (0 = straight ahead, + = left). Out-of-range returns (null/inf/NaN, or outside
    [range_min, range_max]) count as "no obstacle within range_max".
    """
    if not isinstance(msg, dict) or not isinstance(msg.get("ranges"), list):
        return None
    try:
        a0, inc = float(msg["angle_min"]), float(msg["angle_increment"])
        rmin, rmax = float(msg.get("range_min") or 0.0), float(msg.get("range_max") or float("inf"))
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(a0) or not math.isfinite(inc) or inc == 0.0:
        return None
    pts = []
    for i, r in enumerate(msg["ranges"]):
        try:
            r = float(r)
        except (TypeError, ValueError):
            continue
        if math.isfinite(r) and rmin <= r <= rmax:
            deg = math.degrees(a0 + i * inc)
            pts.append((r, (deg + 180.0) % 360.0 - 180.0))
    sectors: Dict[str, Any] = {}
    for name, centre in _SECTORS:
        near = [r for r, d in pts if abs((d - centre + 180.0) % 360.0 - 180.0) <= 22.5]
        sectors[name] = round(min(near), 2) if near else None
    out: Dict[str, Any] = {"beams": len(msg["ranges"]), "valid_returns": len(pts),
                           "range_max_m": round(rmax, 2) if math.isfinite(rmax) else None,
                           "nearest_m_by_sector": sectors,
                           "note": "angles: 0 = straight ahead, + = left; null = nothing within range_max"}
    if pts:
        r, d = min(pts)
        out["nearest"] = {"distance_m": round(r, 2), "bearing_deg": round(d, 1)}
    return out


class RosBackend(RobotBackend):
    kind = "ros"
    capabilities = frozenset({MOVE, POSE, CAMERA})
    default_safety = {
        "max_publish_rate_hz": 30.0,
        "write_allowlist": {"topics": ["/cmd_vel"], "services": []},
        "motion_topic_patterns": ["*cmd_vel*"],
    }

    def __init__(self, name, cfg, data_dir):
        super().__init__(name, cfg, data_dir)
        self.host = str(cfg.get("host", "localhost"))
        self.port = int(cfg.get("port", 9090))
        self.connect_timeout = float(cfg.get("connect_timeout_s", 6.0))
        self.call_timeout = float(cfg.get("call_timeout_s", 8.0))
        self.topics_cfg = {**TOPIC_DEFAULTS, **(cfg.get("topics") or {})}
        self.description = f"rosbridge ws://{self.host}:{self.port}"
        self._ros: Any = None
        self._lock = threading.RLock()
        self._pubs: Dict[Tuple[str, str], Any] = {}
        self._odom_sub: Any = None
        self._odom_key: Optional[Tuple[str, str]] = None
        self._odom_last: Optional[Tuple[Dict[str, Any], float]] = None
        self._stop_evt = threading.Event()

    # ------------------------------------------------------------ connection
    def _client(self) -> Any:
        with self._lock:
            if self._ros is not None and self._ros.is_connected:
                return self._ros
            try:
                import websocket  # noqa: F401  (websocket-client; checked here so the error is actionable)
            except ImportError as exc:
                raise BackendError("websocket-client is not installed (pip install 'websocket-client>=1.6,<2')") from exc
            ros = RosbridgeClient(self.host, self.port)
            try:
                ros.connect(timeout=self.connect_timeout)
            except Exception as exc:
                ros.close()
                raise BackendError(f"cannot reach rosbridge at ws://{self.host}:{self.port} ({type(exc).__name__}: {exc}). "
                                   "Is rosbridge_server running and the host/port in plugin settings correct?") from exc
            self._ros = ros
            self._pubs.clear()
            self._odom_sub = None
            self._odom_key = None
            return ros

    def close(self) -> None:
        with self._lock:
            try:
                if self._odom_sub is not None and self._ros is not None:
                    self._ros.unsubscribe(self._odom_sub[0], self._odom_sub[1])
            except Exception:
                pass
            try:
                if self._ros is not None:
                    self._ros.close()
            except Exception:
                pass
            self._ros, self._pubs, self._odom_sub = None, {}, None

    
    # ---------------------------------------------------------------- helpers
    def _svc(self, name: str, typ: str, req: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        ros = self._client()
        try:
            return ros.call_service(name, typ, req or {}, timeout=timeout or self.call_timeout)
        except Exception as exc:
            raise BackendError(f"service {name} failed: {type(exc).__name__}: {exc}") from exc

    def _topic_type(self, topic: str) -> str:
        res = self._svc("/rosapi/topic_type", "rosapi/TopicType", {"topic": topic})
        t = res.get("type") or ""
        if not t:
            raise BackendError(f"topic {topic!r} has no publisher/known type; pass msg_type explicitly or check ros_list")
        return t

    # -------------------------------------------------------------- discovery
    def list_topics(self) -> List[Dict[str, str]]:
        res = self._svc("/rosapi/topics_and_raw_types", "rosapi/TopicsAndRawTypes")
        names, types = res.get("topics", []), res.get("types", [])
        if names and len(names) == len(types):
            return [{"name": n, "type": t} for n, t in zip(names, types)]
        res = self._svc("/rosapi/topics", "rosapi/Topics")
        return [{"name": n, "type": ""} for n in res.get("topics", [])]

    def list_services(self) -> List[Dict[str, str]]:
        names = self._svc("/rosapi/services", "rosapi/Services").get("services", [])
        out = []
        for n in names[:80]:
            try:
                t = self._svc("/rosapi/service_type", "rosapi/ServiceType", {"service": n}, timeout=2.0).get("type", "")
            except BackendError:
                t = ""
            out.append({"name": n, "type": t})
        out += [{"name": n, "type": ""} for n in names[80:]]
        return out

    def list_nodes(self) -> List[str]:
        return list(self._svc("/rosapi/nodes", "rosapi/Nodes").get("nodes", []))

    # ------------------------------------------------------------------- data
    def _first_message(self, topic: str, msg_type: Optional[str], timeout: float, **kw: Any) -> Dict[str, Any]:
        ros = self._client()
        if self._odom_key is not None and topic == self._odom_key[0] and not kw:
            # rosbridge drops *every* subscription of this client on a topic when one of them unsubscribes, so a
            # one-shot subscribe/unsubscribe here would silently kill the persistent odometry feed the safety
            # interlock depends on. Serve the request from that feed instead.
            t0 = time.time()
            self._ensure_odom()
            deadline = t0 + timeout
            while time.time() < deadline:
                last = self._odom_last
                if last is not None and last[1] >= t0:
                    return dict(last[0])
                time.sleep(0.02)
            raise BackendError(f"no message on {topic} within {timeout:.1f}s (nothing publishing? wrong type {msg_type}?)")
        msg_type = msg_type or self._topic_type(topic)
        got: Dict[str, Any] = {}
        ev = threading.Event()

        def cb(m: Dict[str, Any]) -> None:
            if not ev.is_set():
                got["m"] = dict(m)
                ev.set()

        sid = ros.subscribe(topic, msg_type, cb, queue_length=1, **kw)
        try:
            if not ev.wait(timeout):
                raise BackendError(f"no message on {topic} within {timeout:.1f}s (nothing publishing? wrong type {msg_type}?)")
        finally:
            try:
                ros.unsubscribe(topic, sid)
            except Exception:
                pass
        return got["m"]

    def echo_once(self, topic: str, msg_type: Optional[str], timeout: float) -> Dict[str, Any]:
        return self._first_message(topic, msg_type, timeout)

    def publish(self, topic: str, msg_type: str, message: Dict[str, Any]) -> None:
        ros = self._client()
        try:
            ros.publish(topic, msg_type, message)
        except Exception as exc:
            raise BackendError(f"publish to {topic} failed: {type(exc).__name__}: {exc}") from exc

    def call_service(self, service: str, srv_type: Optional[str], request: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        if not srv_type:
            res = self._svc("/rosapi/service_type", "rosapi/ServiceType", {"service": service}, timeout=3.0)
            srv_type = res.get("type")
            if not srv_type:
                raise BackendError(f"unknown service {service!r}; check ros_list kind=services")
        return self._svc(service, srv_type, request, timeout=timeout)

    def _ensure_odom(self) -> None:
        topic = self.topics_cfg.get("odom", "/odom")
        mtype = self.topics_cfg.get("odom_type") or self._topic_type(topic)
        key = (topic, mtype)
        with self._lock:
            ros = self._client()
            if self._odom_sub is not None and self._odom_key == key:
                return
            if self._odom_sub is not None:
                try:
                    ros.unsubscribe(self._odom_sub[0], self._odom_sub[1])
                except Exception:
                    pass

            def cb(m: Dict[str, Any]) -> None:
                self._odom_last = (dict(m), time.time())

            self._odom_sub = (topic, ros.subscribe(topic, mtype, cb, queue_length=1, throttle_rate=50))
            self._odom_key = key
            self._odom_last = None

    def latest_pose(self) -> Optional[Dict[str, Any]]:
        try:
            self._ensure_odom()
            deadline = time.time() + 1.5
            while self._odom_last is None and time.time() < deadline:
                time.sleep(0.05)
            if self._odom_last is None:
                return None
            msg, ts = self._odom_last
            if time.time() - ts > 2.0:  # feed went quiet: resubscribe once before reporting stale data
                try:
                    self._client().unsubscribe(self._odom_sub[0], self._odom_sub[1])
                except Exception:
                    pass
                self._odom_key, self._odom_sub, self._odom_last = None, None, None
                self._ensure_odom()
                deadline = time.time() + 1.5
                while self._odom_last is None and time.time() < deadline:
                    time.sleep(0.05)
                if self._odom_last is None:
                    return None
                msg, ts = self._odom_last
            pose = parse_pose(msg)
            pose["age_s"] = round(time.time() - ts, 3)
            pose["source"] = f"rosbridge:{self._odom_key[0] if self._odom_key else '?'}"
            return pose
        except BackendError:
            return None

    # --------------------------------------------------------- generic interface
    def connect(self):
        self._client()
        return {"connected": True, "via": self.description}

    def state(self):
        connected = bool(self._ros is not None and self._ros.is_connected)
        pose = self.latest_pose()
        if pose is not None:
            connected = True
        return {"pose": pose, "connected": connected, "url": f"ws://{self.host}:{self.port}", "topics": self.topics_cfg}

    def _stamped(self) -> bool:
        return "Stamped" in str(self.topics_cfg.get("cmd_vel_type", ""))

    def _twist(self, vx: float, wz: float) -> Dict[str, Any]:
        msg = {"linear": {"x": vx, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": wz}}
        return {"header": {"frame_id": "base_link"}, "twist": msg} if self._stamped() else msg

    def _send_zero(self, topic: str, msg_type: str, stamped: bool, n: int = 3) -> None:
        zero = {"header": {"frame_id": "base_link"}, "twist": ZERO_TWIST} if stamped else ZERO_TWIST
        for _ in range(n):
            self.publish(topic, msg_type, zero)
            time.sleep(0.02)

    def _run_timed(self, topic: str, msg_type: str, message: Dict[str, Any], duration: float, rate: float,
                   ctx: MoveContext, stamped: bool) -> Dict[str, Any]:
        """Publish ``message`` at ``rate`` Hz for ``duration`` s, aborting on e-stop / stop / fence breach, then stop."""
        self._stop_evt.clear()
        period = 1.0 / rate
        elapsed, sent, reason = 0.0, 0, "completed"
        before = self.latest_pose()
        try:
            while elapsed < duration - 1e-9:
                why = ctx.should_abort()
                if why:
                    reason = f"aborted: {why}"; break
                if self._stop_evt.is_set():
                    reason = "aborted: stop requested"; break
                f = ctx.fence
                if f:
                    p = self.latest_pose()
                    if p and not (f["x_min"] <= p["x"] <= f["x_max"] and f["y_min"] <= p["y"] <= f["y_max"]):
                        reason = "aborted: robot reached the geofence edge"; break
                self.publish(topic, msg_type, message)
                sent += 1
                step = min(period, duration - elapsed)
                time.sleep(step)
                elapsed += step
        finally:
            try:
                self._send_zero(topic, msg_type, stamped)
            except BackendError:
                pass
        after = self.latest_pose()
        disp = round(math.hypot(after["x"] - before["x"], after["y"] - before["y"]), 3) if before and after else None
        return {"topic": topic, "published": sent, "ran_for_s": round(elapsed, 2), "end_reason": reason,
                "pose_before": before, "pose_after": after, "displacement_m": disp}

    def move(self, linear_mps, angular_rps, duration_s, ctx):
        topic = norm_name(self.topics_cfg["cmd_vel"])
        return self._run_timed(topic, self.topics_cfg["cmd_vel_type"], self._twist(linear_mps, angular_rps), duration_s,
                               10.0, ctx, self._stamped())

    def stop(self):
        self._stop_evt.set()
        topic = norm_name(self.topics_cfg["cmd_vel"])
        self._send_zero(topic, self.topics_cfg["cmd_vel_type"], self._stamped(), n=5)
        return {"stopped": True, "stop_sent_to": topic}

    def views(self):
        return ("camera",)

    def look(self, view, width, height):
        topic = norm_name(self.cfg.get("camera_topic") or self.topics_cfg.get("camera"))
        timeout = float(self.cfg.get("camera_timeout_s", 6.0))
        msg = self._first_message(topic, None, timeout, compression="none", throttle_rate=100)
        if "format" in msg and "width" not in msg:  # sensor_msgs/CompressedImage
            data = imaging.compressed_msg_to_bytes(msg)
            imaging.sniff_image(data)
            return data, f"rosbridge:{topic} (CompressedImage {msg.get('format')})"
        try:
            png = imaging.image_msg_to_png(msg)
        except (KeyError, ValueError) as exc:
            raise BackendError(f"cannot decode image from {topic}: {exc}") from exc
        return png, f"rosbridge:{topic} (Image {msg.get('width')}x{msg.get('height')} {msg.get('encoding')})"

    # ------------------------------------------------------------------ extras
    def actions(self):
        return {
            "topics": Action("topics", "List topics, services and nodes. params.kind: topics|services|nodes|all, "
                             "params.filter: substring.", READ, {"kind": {"type": "string"}, "filter": {"type": "string"}}),
            "echo": Action("echo", "Read one message from params.topic (optional params.msg_type, params.timeout_s). "
                           "LaserScans also get a scan_summary (nearest obstacle per 45° sector).", READ,
                           {"topic": {"type": "string"}, "msg_type": {"type": "string"}, "timeout_s": {"type": "number"}}),
            "publish": Action("publish", "Publish params.message (JSON) of params.msg_type to an allowlisted params.topic. "
                              "A Twist on a velocity topic gets the same speed / duration / geofence checks as robot_move "
                              "and is followed by zero velocity.", WRITE,
                              {"topic": {"type": "string"}, "msg_type": {"type": "string"}, "message": {"type": "object"},
                               "duration_s": {"type": "number"}, "rate_hz": {"type": "number"}}),
            "call_service": Action("call_service", "Call an allowlisted params.service with params.request "
                                   "(optional params.srv_type, params.timeout_s).", WRITE,
                                   {"service": {"type": "string"}, "srv_type": {"type": "string"},
                                    "request": {"type": "object"}, "timeout_s": {"type": "number"}}),
        }

    @staticmethod
    def _allow(kind: str, name: str, limits: Dict[str, Any]) -> None:
        allow = (limits.get("write_allowlist") or {}).get(kind + "s") or []
        if not matches_any(name, allow):
            raise Blocked(f"{kind}_not_allowlisted", f"{kind} {name!r} is not on the write allowlist {list(allow)}; a "
                          f"human can add it under safety.write_allowlist.{kind}s")

    def check_action(self, action, params, guard):
        lim = guard.limits
        if action == "call_service":
            name = norm_name(params.get("service"))
            if not isinstance(params.get("request") or {}, dict):
                raise Blocked("invalid_args", "request must be an object")
            self._allow("service", name, lim)
            return {**params, "service": name}
        if action != "publish":
            return params
        topic, msg_type, message = norm_name(params.get("topic")), str(params.get("msg_type") or ""), params.get("message")
        if not isinstance(message, dict):
            raise Blocked("invalid_args", "message must be an object (the ROS message as JSON)")
        self._allow("topic", topic, lim)
        twist = extract_twist(message)
        motion_topic = matches_any(topic, lim.get("motion_topic_patterns") or []) or msg_type in TWIST_TYPES
        if twist is None:
            if motion_topic:
                raise Blocked("unparseable_motion_message", f"{topic} looks like a velocity topic but the message is not a "
                              "readable Twist/TwistStamped; refusing because its speed cannot be verified")
            return {"topic": topic, "msg_type": msg_type, "message": message, "twist": False}
        (lx, ly, lz), (ax, ay, az) = twist
        if any(abs(c) > 0 for c in (ly, lz, ax, ay)):
            raise Blocked("unsupported_twist", "only linear.x and angular.z can be commanded (the interlock cannot "
                          "verify sideways, vertical or roll/pitch motion)")
        v = guard.check_velocity(lx, az, params.get("duration_s"))
        rate = num(params.get("rate_hz"), "rate_hz", 10.0)
        if rate <= 0:
            raise Blocked("invalid_args", "rate_hz must be > 0")
        notes = list(v["notes"])
        max_rate = float(lim.get("max_publish_rate_hz", 30.0))
        if rate > max_rate:
            notes.append(f"rate clamped {rate:.0f} -> {max_rate:.0f} Hz"); rate = max_rate
        return {"topic": topic, "msg_type": msg_type, "message": set_twist(message, v["linear_mps"], v["angular_rps"]),
                "twist": True, "stop": v["stop"], "duration_s": v["duration_s"], "rate_hz": rate, "_notes": notes}

    def run_action(self, action, params, ctx):
        if action == "topics":
            kind, flt = str(params.get("kind") or "all").lower(), str(params.get("filter") or "").lower()

            def keep(item: Any) -> bool:
                s = (item if isinstance(item, str) else f"{item['name']} {item.get('type', '')}").lower()
                return not flt or flt in s
            out: Dict[str, Any] = {"via": self.description}
            if kind in ("topics", "all"):
                out["topics"] = [t for t in self.list_topics() if keep(t)]
            if kind in ("services", "all"):
                out["services"] = [s for s in self.list_services() if keep(s)]
            if kind in ("nodes", "all"):
                out["nodes"] = [n for n in self.list_nodes() if keep(n)]
            return out
        if action == "echo":
            topic = norm_name(params.get("topic"))
            timeout = float(params.get("timeout_s") or 5.0)
            msg = self.echo_once(topic, params.get("msg_type") or None, timeout)
            scan = scan_summary(msg)
            return {"topic": topic, **({"scan_summary": scan} if scan is not None else {}), "message": summarize(msg)}
        if action == "publish":
            if not params.get("twist"):
                self.publish(params["topic"], params["msg_type"], params["message"])
                return {"topic": params["topic"], "published": 1}
            if params.get("stop"):
                self._send_zero(params["topic"], params["msg_type"] or self.topics_cfg["cmd_vel_type"],
                                "twist" in params["message"] and "linear" not in params["message"])
                return {"topic": params["topic"], "published": "zero velocity"}
            return self._run_timed(params["topic"], params["msg_type"], params["message"], params["duration_s"],
                                   params["rate_hz"], ctx, "twist" in params["message"] and "linear" not in params["message"])
        if action == "call_service":
            res = self.call_service(params["service"], params.get("srv_type") or None, params.get("request") or {},
                                    float(params.get("timeout_s") or self.call_timeout))
            return {"service": params["service"], "response": summarize(res)}
        raise BackendError(f"unknown action {action!r}")
