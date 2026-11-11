"""``mock``: a built-in 2D differential-drive robot (no dependencies) for tests, CI and quick tries.

World: a rectangular arena ``[-half_x, half_x] x [-half_y, half_y]`` metres with red traffic cones. The robot (radius
0.2 m) stops dead on walls and cones and reports ``collided``. Time is simulated, so moves return instantly unless
``realtime: true``. Views: ``camera`` (software-raycast forward camera) and ``overhead`` (top-down map with the trail).
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from .. import imaging
from .base import CAMERA, MOVE, POSE, READ, WRITE, Action, BackendError, Blocked, MoveContext, RobotBackend, num

ROBOT_RADIUS = 0.2
CONE_RADIUS = 0.2
SIM_DT = 0.02


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class MockBackend(RobotBackend):
    kind = "mock"
    description = "built-in 2D differential-drive sim (no dependencies)"
    capabilities = frozenset({MOVE, POSE, CAMERA})
    default_safety = {"max_linear_mps": 0.5, "max_reverse_mps": 0.3,
                      "geofence": {"enabled": True, "x_min": -4.0, "x_max": 4.0, "y_min": -4.0, "y_max": 4.0, "margin_m": 0.25}}

    def __init__(self, name, cfg, data_dir):
        super().__init__(name, cfg, data_dir)
        self._lock = threading.RLock()
        self.half_x = float(cfg.get("half_x", 5.0))
        self.half_y = float(cfg.get("half_y", 5.0))
        self.cones: List[Tuple[float, float]] = [(float(c["x"]), float(c["y"])) for c in cfg.get("cones", [{"x": 3.5, "y": 0.8}])]
        self.realtime = bool(cfg.get("realtime", False))
        self.reset()

    def reset(self, x: Optional[float] = None, y: Optional[float] = None, yaw_deg: Optional[float] = None,
              **_: Any) -> Dict[str, Any]:
        start = self.cfg.get("start") or {}
        with self._lock:
            self.x = float(x if x is not None else start.get("x", 0.0))
            self.y = float(y if y is not None else start.get("y", 0.0))
            self.yaw = math.radians(float(yaw_deg if yaw_deg is not None else start.get("yaw_deg", 0.0)))
            self.v = self.w = 0.0
            self.t = 0.0
            self.collided: Optional[str] = None
            self.trail: Deque[Tuple[float, float]] = deque([(self.x, self.y)], maxlen=400)
            self._stop = threading.Event()
        return {"reset": True, "pose": self._pose()}

    # ------------------------------------------------------------- physics
    def _blocked_at(self, nx: float, ny: float, margin: float = 0.0) -> Optional[str]:
        if abs(nx) > self.half_x - ROBOT_RADIUS - margin or abs(ny) > self.half_y - ROBOT_RADIUS - margin:
            return "wall"
        for cx, cy in self.cones:
            if math.hypot(nx - cx, ny - cy) < ROBOT_RADIUS + CONE_RADIUS + margin:
                return "cone"
        return None

    def _step(self, lin: float, ang: float, dt: float) -> None:
        nx, ny = self.x + lin * math.cos(self.yaw) * dt, self.y + lin * math.sin(self.yaw) * dt
        hit = self._blocked_at(nx, ny)
        if hit:
            self.collided, self.v = hit, 0.0
        else:
            if lin != 0.0 and not self._blocked_at(nx, ny, margin=0.02):
                self.collided = None
            self.x, self.y, self.v = nx, ny, lin
        self.w = ang
        self.yaw = _wrap(self.yaw + ang * dt)
        self.t += dt

    def _pose(self) -> Dict[str, Any]:
        return {"x": round(self.x, 4), "y": round(self.y, 4), "yaw_deg": round(math.degrees(self.yaw), 1),
                "age_s": 0.0, "frame_id": "odom"}

    # ------------------------------------------------------------- interface
    def state(self) -> Dict[str, Any]:
        with self._lock:
            return {"pose": self._pose(), "velocity": {"linear_mps": round(self.v, 3), "angular_rps": round(self.w, 3)},
                    "collided": self.collided, "sim_time_s": round(self.t, 2),
                    "arena": {"half_x": self.half_x, "half_y": self.half_y},
                    "cones": [{"x": cx, "y": cy} for cx, cy in self.cones]}

    def move(self, linear_mps, angular_rps, duration_s, ctx: MoveContext):
        before = self.state()["pose"]
        self._stop.clear()
        elapsed, reason = 0.0, "completed"
        while elapsed < duration_s - 1e-9:
            why = ctx.should_abort()
            if why:
                reason = f"aborted: {why}"; break
            if self._stop.is_set():
                reason = "aborted: stop requested"; break
            dt = min(SIM_DT, duration_s - elapsed)
            with self._lock:
                self._step(linear_mps, angular_rps, dt)
                f = ctx.fence
                if f and not (f["x_min"] <= self.x <= f["x_max"] and f["y_min"] <= self.y <= f["y_max"]):
                    reason = "aborted: reached the geofence edge"
                    elapsed += dt
                    break
            elapsed += dt
            if self.realtime:
                time.sleep(dt)
        with self._lock:
            self.v = self.w = 0.0
            self.trail.append((self.x, self.y))
            after = self._pose()
        return {"end_reason": reason, "ran_for_s": round(elapsed, 2), "pose_before": before, "pose_after": after,
                "displacement_m": round(math.hypot(after["x"] - before["x"], after["y"] - before["y"]), 3),
                "collided": self.collided}

    def stop(self):
        self._stop.set()
        with self._lock:
            self.v = self.w = 0.0
        return {"stopped": True}

    def views(self):
        return ("camera", "overhead")

    def look(self, view, width, height):
        with self._lock:
            if view == "overhead":
                return (imaging.render_topdown(self.x, self.y, self.yaw, self.half_x, self.half_y, self.cones, list(self.trail)),
                        "mock overhead map")
            return imaging.render_forward(self.x, self.y, self.yaw, self.half_x, self.half_y, self.cones), \
                "mock forward camera (software raycaster, 320x240)"

    # ---------------------------------------------------------------- extras
    def actions(self):
        return {
            "cones": Action("cones", "List the traffic cones (positions in the odom frame).", READ),
            "teleport": Action("teleport", "Place the mock robot at x, y (metres) facing yaw_deg. Must be inside the geofence.",
                               WRITE, {"x": {"type": "number"}, "y": {"type": "number"}, "yaw_deg": {"type": "number"}}),
        }

    def check_action(self, action, params, guard):
        if action == "teleport":
            x, y = num(params.get("x"), "x"), num(params.get("y"), "y")
            yaw = num(params.get("yaw_deg"), "yaw_deg", 0.0)
            f = guard.fence()
            if f and not (f["x_min"] <= x <= f["x_max"] and f["y_min"] <= y <= f["y_max"]):
                raise Blocked("geofence", f"teleport target ({x:.2f}, {y:.2f}) is outside the geofence")
            return {"x": x, "y": y, "yaw_deg": yaw}
        return params

    def run_action(self, action, params, ctx):
        with self._lock:
            if action == "cones":
                return {"cones": [{"x": cx, "y": cy} for cx, cy in self.cones]}
            if action == "teleport":
                if self._blocked_at(params["x"], params["y"]):
                    raise BackendError("teleport target is inside a wall or cone")
                self.x, self.y, self.yaw = params["x"], params["y"], _wrap(math.radians(params["yaw_deg"]))
                self.trail.append((self.x, self.y))
                return {"pose": self._pose()}
        raise BackendError(f"unknown action {action!r}")
