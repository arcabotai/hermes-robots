"""``microduck``: Pollen Robotics' open-source bipedal duck, driven by its real RL policies in MuJoCo.

Two modes (``robots.<name>.mode``):

* ``sim``  - HTTP client for ``sim/microduck/server.py`` (MuJoCo physics + Pollen's trained ONNX walking / kick
             policies). Start the server first; see the README.
* ``mock`` - a pure-Python kinematic duck with the same API (no physics, no policies) for CI and quick tries.

Ported from arcabotai/hermes-microduck. The walking policy is trained only up to 0.25 m/s forward, so that is a hard
ceiling the settings cannot raise.
"""
from __future__ import annotations

import json
import math
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib
from typing import Any, Dict, Optional

from .base import CAMERA, MOVE, POSE, TURN, MOTION, Action, BackendError, Blocked, MoveContext, RobotBackend, num


class SimLink:
    """HTTP client for sim/microduck/server.py (MuJoCo + Pollen's ONNX policies)."""
    name = "sim"

    def __init__(self, url: str, timeout_s: float = 90):
        self.url = url.rstrip("/")
        self.timeout = float(timeout_s)
        self.description = f"MuJoCo Microduck sim at {self.url} (Pollen Robotics ONNX policies)"

    def _req(self, method: str, path: str, body: Optional[Dict[str, Any]] = None, raw: bool = False) -> Any:
        data = json.dumps(body or {}).encode() if method == "POST" else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = r.read()
        except urllib.error.HTTPError as exc:
            try:
                msg = json.loads(exc.read()).get("error")
            except Exception:
                msg = str(exc)
            raise BackendError(f"sim server error: {msg}")
        except (urllib.error.URLError, OSError) as exc:
            raise BackendError(f"cannot reach the Microduck sim at {self.url} ({exc}); start it with "
                               "`python sim/microduck/server.py` or set mode: mock")
        if raw:
            return payload
        out = json.loads(payload)
        if isinstance(out, dict) and out.get("error"):
            raise BackendError(f"sim server: {out['error']}")
        return out

    def status(self) -> Dict[str, Any]:
        return self._req("GET", "/status")

    def frame(self, view: str, w: int, h: int) -> bytes:
        return self._req("GET", f"/frame?view={view}&w={int(w)}&h={int(h)}", raw=True)

    def walk(self, forward_mps: float, turn_rps: float, duration_s: float, fence: Optional[Dict[str, float]] = None):
        return self._req("POST", "/walk", {"forward_mps": forward_mps, "turn_rps": turn_rps, "duration_s": duration_s, "fence": fence})

    def turn(self, degrees: float):
        return self._req("POST", "/turn", {"degrees": degrees})

    def kick_ball(self, foot: str = "auto"):
        return self._req("POST", "/kick_ball", {"foot": foot})

    def kick(self, foot: str):
        return self._req("POST", "/kick", {"foot": foot})

    def sit(self):
        return self._req("POST", "/sit")

    def stand(self):
        return self._req("POST", "/stand")

    def head(self, yaw: float, pitch: float, hold_s: float):
        # head command slots are [neck_pitch, head_pitch, head_yaw, head_roll]; + neck_pitch = up, + head_yaw = left
        return self._req("POST", "/head", {"head_yaw": yaw, "neck_pitch": pitch, "hold_s": hold_s})

    def stop(self):
        return self._req("POST", "/stop")

    def reset(self, ball_x: Optional[float] = None, ball_y: Optional[float] = None):
        body = {} if ball_x is None else {"ball_x": ball_x, "ball_y": ball_y or 0.0}
        return self._req("POST", "/reset", body)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------------------------------- mock
ARENA = 1.5
GOAL_X, GOAL_HALF_W = ARENA - 0.08, 0.3


def _png(w: int, h: int, rgb: bytearray) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw += rgb[y * w * 3:(y + 1) * w * 3]
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))


class KinematicDuck:
    """Kinematic stand-in: unicycle duck (~0.1 m/s real speed like the policy), ball that rolls ~1 m when kicked."""
    name = "mock"
    description = "built-in kinematic mock duck (no MuJoCo, no policies)"

    def __init__(self):
        self._lock = threading.RLock()
        self.reset()

    def reset(self, ball_x: Optional[float] = None, ball_y: Optional[float] = None):
        with self._lock:
            self.x, self.y, self.yaw = -0.6, 0.0, 0.0
            self.bx, self.by = (0.2, -0.05) if ball_x is None else (ball_x, ball_y or 0.0)
            self.posture = "standing"
            self.t = 0.0
            self.head_pose = (0.0, 0.0)
        return {"ok": True, "status": self.status()}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            dx, dy = self.bx - self.x, self.by - self.y
            bearing = math.degrees((math.atan2(dy, dx) - self.yaw + math.pi) % (2 * math.pi) - math.pi)
            return {"sim_time_s": round(self.t, 2),
                    "pose": {"x": round(self.x, 3), "y": round(self.y, 3), "yaw_deg": round(math.degrees(self.yaw), 1), "trunk_height_m": 0.116},
                    "upright": True, "posture": self.posture, "policy": "mock",
                    "ball": {"x": round(self.bx, 3), "y": round(self.by, 3), "distance_m": round(math.hypot(dx, dy), 3), "bearing_deg": round(bearing, 1)},
                    "goal": {"x": GOAL_X, "y": 0.0, "width_m": 2 * GOAL_HALF_W}, "arena_half_size_m": ARENA, "behaviour": "idle"}

    def _moved(self, before, reason="completed", **extra):
        after = self.status()
        b, a = before["pose"], after["pose"]
        out = {"ok": True, "end_reason": reason, "pose_before": b, "pose_after": a,
               "displacement_m": round(math.hypot(a["x"] - b["x"], a["y"] - b["y"]), 3),
               "turned_deg": round(a["yaw_deg"] - b["yaw_deg"], 1), "status": after}
        out.update(extra)
        return out

    def walk(self, forward_mps, turn_rps, duration_s, fence=None):
        before = self.status()
        reason = "completed"
        with self._lock:
            if self.posture == "sitting":
                return self._moved(before, "refused: the duck is sitting (robot_action stand first)")
            real = 0.4 * forward_mps if abs(forward_mps) >= 0.2 else 0.0  # the policy only gaits near its trained speed
            dt, t = 0.05, 0.0
            while t < duration_s - 1e-9:
                self.x += real * math.cos(self.yaw) * dt
                self.y += real * math.sin(self.yaw) * dt
                self.yaw += (0.5 * turn_rps if real else 0.1 * turn_rps) * dt
                t += dt; self.t += dt
                if fence and not (fence["x_min"] <= self.x <= fence["x_max"] and fence["y_min"] <= self.y <= fence["y_max"]):
                    reason = "aborted: reached the geofence edge"; break
        return self._moved(before, reason)

    def turn(self, degrees):
        before = self.status()
        with self._lock:
            self.yaw = (self.yaw + math.radians(degrees) + math.pi) % (2 * math.pi) - math.pi
            self.t += abs(degrees) / 30.0
        return self._moved(before)

    def kick_ball(self, foot="auto"):
        before = self.status()
        with self._lock:
            u = (GOAL_X - self.bx, -self.by); n = math.hypot(*u) or 1.0; u = (u[0] / n, u[1] / n)
            self.x, self.y = self.bx - 0.09 * u[0], self.by - 0.09 * u[1]
            self.yaw = math.atan2(u[1], u[0])
            self.t += 10 + before["ball"]["distance_m"] * 10
            travel = 1.1
            nx, ny = self.bx + travel * u[0], self.by + travel * u[1]
            goal = nx >= GOAL_X - 0.03 and abs(self.by + (GOAL_X - self.bx) * u[1] / max(u[0], 1e-6)) < GOAL_HALF_W
            self.bx, self.by = min(nx, ARENA - 0.06), ny
            foot = foot if foot != "auto" else ("right" if before["ball"]["bearing_deg"] <= 0 else "left")
        after = self.status()
        return self._moved(before, foot=foot, lined_up=True, ball_before=before["ball"], ball_after=after["ball"],
                           ball_travel_m=round(math.hypot(after["ball"]["x"] - before["ball"]["x"], after["ball"]["y"] - before["ball"]["y"]), 3),
                           goal_scored=bool(goal))

    def kick(self, foot):
        before = self.status()
        with self._lock:
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            fx, fy = c * (self.bx - self.x) + s * (self.by - self.y), -s * (self.bx - self.x) + c * (self.by - self.y)
            side = -0.045 if foot == "right" else 0.045
            if 0.06 <= fx <= 0.12 and abs(fy - side) < 0.04:
                self.bx += 0.8 * c; self.by += 0.8 * s
        after = self.status()
        return self._moved(before, ball_travel_m=round(math.hypot(after["ball"]["x"] - before["ball"]["x"], after["ball"]["y"] - before["ball"]["y"]), 3))

    def sit(self):
        before = self.status(); self.posture = "sitting"; return self._moved(before)

    def stand(self):
        before = self.status(); self.posture = "standing"; return self._moved(before)

    def head(self, yaw, pitch, hold_s):
        self.head_pose = (yaw, pitch); return {"ok": True, "status": self.status()}

    def stop(self):
        return {"ok": True, "status": self.status()}

    def frame(self, view: str, w: int, h: int) -> bytes:
        """Top-down sketch of the arena (the mock has no 3-D renderer): grey floor, yellow walls, white goal, red ball, duck."""
        w, h = max(64, min(int(w), 640)), max(64, min(int(h), 480))
        img = bytearray([58, 66, 80] * (w * h))
        scale = min(w, h) / (2 * ARENA + 0.2)

        def px(x, y):
            return int(w / 2 + x * scale), int(h / 2 - y * scale)

        def disc(cx, cy, r, col):
            X, Y = px(cx, cy); R = max(2, int(r * scale))
            for yy in range(max(0, Y - R), min(h, Y + R + 1)):
                for xx in range(max(0, X - R), min(w, X + R + 1)):
                    if (xx - X) ** 2 + (yy - Y) ** 2 <= R * R:
                        img[(yy * w + xx) * 3:(yy * w + xx) * 3 + 3] = bytes(col)
        x0, y0 = px(-ARENA, ARENA); x1, y1 = px(ARENA, -ARENA)
        for xx in range(max(0, x0), min(w, x1 + 1)):
            for yy in (y0, y1):
                if 0 <= yy < h: img[(yy * w + xx) * 3:(yy * w + xx) * 3 + 3] = bytes((240, 190, 30))
        for yy in range(max(0, y0), min(h, y1 + 1)):
            for xx in (x0, x1):
                if 0 <= xx < w: img[(yy * w + xx) * 3:(yy * w + xx) * 3 + 3] = bytes((240, 190, 30))
        for gy in (-GOAL_HALF_W, GOAL_HALF_W):
            disc(GOAL_X, gy, 0.03, (245, 245, 245))
        with self._lock:
            disc(self.bx, self.by, 0.05, (230, 50, 40))
            disc(self.x, self.y, 0.08, (250, 180, 0))
            disc(self.x + 0.08 * math.cos(self.yaw), self.y + 0.08 * math.sin(self.yaw), 0.03, (255, 255, 255))
        return _png(w, h, img)

    def close(self) -> None:
        pass


class MicroduckBackend(RobotBackend):
    kind = "microduck"
    capabilities = frozenset({MOVE, TURN, POSE, CAMERA})
    hard_limits = {"max_linear_mps": 0.25}  # the walking policy's trained command range
    default_safety = {
        "max_linear_mps": 0.25, "max_reverse_mps": 0.2, "max_angular_rps": 1.0,
        "max_duration_s": 8.0, "default_duration_s": 2.0, "max_head_rad": 0.6,
        "geofence": {"enabled": True, "x_min": -1.5, "x_max": 1.5, "y_min": -1.5, "y_max": 1.5, "margin_m": 0.2},
    }
    turn_clearance_m = 0.1  # a walking turn sweeps a small arc

    def __init__(self, name, cfg, data_dir):
        super().__init__(name, cfg, data_dir)
        mode = str(cfg.get("mode", "mock")).lower()
        if mode == "sim":
            self.link: Any = SimLink(str(cfg.get("url", "http://127.0.0.1:8765")), float(cfg.get("timeout_s", 90)))
        elif mode == "mock":
            self.link = KinematicDuck()
        else:
            raise BackendError(f"microduck mode must be 'mock' or 'sim', got {mode!r}")
        self.mode = mode
        self.description = self.link.description

    def connect(self):
        st = self.link.status()  # proves the sim server answers
        return {"connected": True, "mode": self.mode, "via": self.description, "policy": st.get("policy")}

    def state(self):
        st = dict(self.link.status())
        st.setdefault("pose", {})
        st["mode"] = self.mode
        return st

    @staticmethod
    def _clean(res: Dict[str, Any]) -> Dict[str, Any]:
        res = dict(res)
        res.pop("ok", None)
        return res

    def move(self, linear_mps, angular_rps, duration_s, ctx: MoveContext):
        return self._clean(self.link.walk(linear_mps, angular_rps, duration_s, ctx.fence))

    def turn(self, degrees, ctx, angular_rps):
        return self._clean(self.link.turn(degrees))

    def stop(self):
        return self._clean(self.link.stop())

    def views(self):
        return ("duck_eye", "chase", "overhead")

    def look(self, view, width, height):
        view = view if view in self.views() else "duck_eye"
        return self.link.frame(view, width, height), f"microduck {self.mode} view {view}"

    def reset(self, ball_x: Optional[float] = None, ball_y: Optional[float] = None, **_: Any):
        return self._clean(self.link.reset(ball_x, ball_y))

    # ---------------------------------------------------------------- extras
    def actions(self):
        foot = {"type": "string", "enum": ["left", "right"]}
        return {
            "kick_ball": Action("kick_ball", "Walk to the ball, line up behind it and kick it toward the goal with the kick "
                                "policy. Optional params.foot: left|right (default: auto).", MOTION, {"foot": foot}),
            "kick": Action("kick", "Kick in place with params.foot (left|right). Only useful when the ball is right at "
                           "that foot.", MOTION, {"foot": foot}),
            "sit": Action("sit", "Sit down (the duck cannot walk while sitting).", MOTION),
            "stand": Action("stand", "Stand back up after sitting.", MOTION),
            "head": Action("head", "Point the head: params.yaw (+ left) and params.pitch (+ up) in radians, held for "
                           "params.hold_s seconds.", MOTION,
                           {"yaw": {"type": "number"}, "pitch": {"type": "number"}, "hold_s": {"type": "number"}}),
        }

    def check_action(self, action, params, guard):
        if action == "head":
            hmax = float(guard.limits.get("max_head_rad", 0.6))
            out = {}
            for k in ("yaw", "pitch"):
                v = num(params.get(k), k, 0.0)
                if abs(v) > hmax + 1e-9:
                    raise Blocked("over_head_limit", f"head {k} {v:.2f} rad exceeds the limit of ±{hmax:.2f} rad")
                out[k] = v
            hold = num(params.get("hold_s"), "hold_s", 1.0)
            if not 0 <= hold <= 5:
                raise Blocked("invalid_args", "hold_s must be between 0 and 5 seconds")
            out["hold_s"] = hold
            return out
        if action == "kick":
            if params.get("foot") not in ("left", "right"):
                raise Blocked("invalid_args", "foot must be 'left' or 'right'")
            return {"foot": params["foot"]}
        if action == "kick_ball":
            fence = guard.fence()
            if fence:
                b = (guard.backend.state() or {}).get("ball") or {}
                if "x" not in b:
                    raise Blocked("no_pose", "cannot see where the ball is; refusing to chase it blind")
                # the goal sits at the +x edge, so only the far wall is special
                if not (fence["y_min"] <= b["y"] <= fence["y_max"] and fence["x_min"] <= b["x"]):
                    raise Blocked("geofence", f"the ball is at ({b['x']:.2f}, {b['y']:.2f}), outside the geofence; the duck "
                                  "will not chase it there")
                if b["x"] > fence["x_max"]:
                    raise Blocked("geofence", f"the ball is at ({b['x']:.2f}, {b['y']:.2f}), too close to the far wall to "
                                  "walk behind it safely")
            foot = params.get("foot")
            return {"foot": foot if foot in ("left", "right") else "auto"}
        return {}

    def run_action(self, action, params, ctx):
        if action == "head":
            return self._clean(self.link.head(params["yaw"], params["pitch"], params["hold_s"]))
        if action == "kick":
            return self._clean(self.link.kick(params["foot"]))
        if action == "kick_ball":
            return self._clean(self.link.kick_ball(params.get("foot", "auto")))
        if action == "sit":
            return self._clean(self.link.sit())
        if action == "stand":
            return self._clean(self.link.stand())
        raise BackendError(f"unknown action {action!r}")
