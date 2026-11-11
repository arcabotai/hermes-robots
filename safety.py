"""One safety interlock for every robot, enforced from Hermes' ``pre_tool_call`` hook.

Rules (per robot: generic base < backend defaults < ``settings.safety`` < ``robots.<name>.safety``):

* speed caps     - forward / reverse / angular magnitudes; a backend's ``hard_limits`` cannot be raised by config
* duration cap   - every motion is bounded; a default is injected when the model leaves it out
* geofence       - optional rectangle in the robot's odometry frame; the *projected* path (dead-reckoned from the
                   current pose) must stay inside it minus a margin, and the pose must be fresh. Moving backends
                   also get the fence to abort at runtime.
* e-stop latch   - one latch for all robots. While latched every motion/write is refused; only a human clears it
                   with ``/robot clear`` (the agent has no tool for that). Survives restarts (``estop.json``).
* fail closed    - handlers act only on a single-use token the hook minted for exactly the vetted arguments;
                   any interlock error blocks the call.
* audit trail    - every refusal goes to the log, a ring buffer, ``blocked.jsonl`` and ``/robot log``.

Stopping is never blocked, and a zero-velocity move is treated as a stop.
"""
from __future__ import annotations

import json
import logging
import math
import secrets
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from .backends.base import CAMERA, MOTION, MOVE, POSE, READ, TURN, Blocked, num  # noqa: F401

logger = logging.getLogger("hermes_robots.interlock")

VETTED_TOOLS = {"robot_move", "robot_turn", "robot_action"}
READ_TOOLS = {"robot_list", "robot_connect", "robot_state", "robot_look"}
STOP_TOOLS = {"robot_stop"}
ALL_TOOLS = VETTED_TOOLS | READ_TOOLS | STOP_TOOLS

AFTER_BLOCK = ("The robot did not move. Do not retry with a smaller, shorter or slower version on your own: tell the "
               "user what was refused and why, offer a safe alternative, and wait for their answer.")


def project_path(x: float, y: float, yaw: float, vx: float, wz: float, duration: float,
                 dt: float = 0.1) -> List[Tuple[float, float]]:
    """Dead-reckon a body-frame (vx, wz) command for ``duration`` seconds from (x, y, yaw)."""
    pts = [(x, y)]
    steps = max(1, int(math.ceil(duration / dt)))
    step = duration / steps
    for _ in range(steps):
        x += vx * math.cos(yaw) * step
        y += vx * math.sin(yaw) * step
        yaw += wz * step
        pts.append((x, y))
    return pts


class Guard:
    """The interlock's view of one robot: its effective limits and the shared checks. Backends get one in
    ``check_action`` so their extras reuse the same speed / duration / geofence logic."""

    def __init__(self, robot: str, backend: Any, limits: Dict[str, Any], interlock: "Interlock"):
        self.robot, self.backend, self.limits, self.interlock = robot, backend, limits, interlock

    @property
    def clamp(self) -> bool:
        return str(self.limits.get("over_limit_action", "block")).lower() == "clamp"

    def require_not_latched(self) -> None:
        if self.interlock.latched:
            raise Blocked("estop_latched", f"e-stop is latched ({self.interlock.estop_status().get('reason')}); all "
                          "motion is refused until a human clears it with /robot clear")

    def fence(self) -> Optional[Dict[str, float]]:
        gf = self.limits.get("geofence") or {}
        if not gf.get("enabled"):
            return None
        m = float(gf.get("margin_m", 0.0))
        return {"x_min": float(gf["x_min"]) + m, "x_max": float(gf["x_max"]) - m,
                "y_min": float(gf["y_min"]) + m, "y_max": float(gf["y_max"]) - m}

    def pose(self) -> Dict[str, Any]:
        """Fresh pose for a geofence decision, or :class:`Blocked`. Never moves blind."""
        try:
            st = self.backend.state()
        except Exception as exc:
            raise Blocked("no_pose", f"cannot read {self.robot}'s pose ({exc}); the geofence is on, so refusing to move blind")
        pose = (st or {}).get("pose")
        if not isinstance(pose, dict) or "x" not in pose or "y" not in pose:
            raise Blocked("no_pose", f"{self.robot} reports no pose; the geofence is on, so refusing to move blind")
        age = float(pose.get("age_s") or 0.0)
        max_age = float((self.limits.get("geofence") or {}).get("max_pose_age_s", 5.0))
        if age > max_age:
            raise Blocked("pose_stale", f"{self.robot}'s latest pose is {age:.1f}s old (max {max_age:.1f}s); refusing to "
                          "move on stale data")
        return pose

    def check_geofence(self, vx: float, wz: float, duration: float) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        fence = self.fence()
        if fence is None:
            return None, None
        p = self.pose()
        yaw = math.radians(float(p.get("yaw_deg", 0.0))) if "yaw_deg" in p else float(p.get("yaw_rad", 0.0))
        pts = project_path(float(p["x"]), float(p["y"]), yaw, vx, wz, duration)
        for i, (px, py) in enumerate(pts):
            if not (fence["x_min"] <= px <= fence["x_max"] and fence["y_min"] <= py <= fence["y_max"]):
                t = duration * i / max(1, len(pts) - 1)
                raise Blocked("geofence", f"the projected path leaves the geofence x[{fence['x_min']:.2f},{fence['x_max']:.2f}] "
                              f"y[{fence['y_min']:.2f},{fence['y_max']:.2f}] after ≈{t:.1f}s near ({px:.2f}, {py:.2f}); "
                              f"{self.robot} is at ({float(p['x']):.2f}, {float(p['y']):.2f}) facing {math.degrees(yaw):.0f}°. "
                              "Choose a shorter, slower or different move.")
        return fence, f"geofence ok (ends near ({pts[-1][0]:.2f}, {pts[-1][1]:.2f}))"

    def check_clearance(self, clearance: float) -> Optional[Dict[str, float]]:
        """For turns: the robot must sit at least ``clearance`` inside the fence."""
        fence = self.fence()
        if fence is None:
            return None
        p = self.pose()
        x, y = float(p["x"]), float(p["y"])
        if not (fence["x_min"] + clearance <= x <= fence["x_max"] - clearance and
                fence["y_min"] + clearance <= y <= fence["y_max"] - clearance):
            raise Blocked("geofence", f"{self.robot} is too close to the geofence edge at ({x:.2f}, {y:.2f}) to turn "
                          f"(needs {clearance:.2f} m clearance); move back inside first")
        return fence

    def check_velocity(self, linear: float, angular: float, duration_arg: Any) -> Dict[str, Any]:
        """Speed caps, duration cap and geofence for a body-velocity command. Returns the vetted command."""
        lim, notes = self.limits, []
        if linear == 0.0 and angular == 0.0:
            return {"linear_mps": 0.0, "angular_rps": 0.0, "duration_s": 0.0, "fence": None, "stop": True,
                    "notes": ["zero-velocity command: treated as a stop (always allowed)"]}
        self.require_not_latched()
        fmax, rmax, amax = float(lim["max_linear_mps"]), float(lim["max_reverse_mps"]), float(lim["max_angular_rps"])
        if linear > fmax + 1e-9:
            if not self.clamp:
                raise Blocked("over_speed_forward", f"forward speed {linear:.2f} m/s exceeds {self.robot}'s limit of {fmax:.2f} m/s")
            notes.append(f"forward speed clamped {linear:.2f} -> {fmax:.2f} m/s"); linear = fmax
        if -linear > rmax + 1e-9:
            if not self.clamp:
                raise Blocked("over_speed_reverse", f"reverse speed {-linear:.2f} m/s exceeds {self.robot}'s limit of {rmax:.2f} m/s")
            notes.append(f"reverse speed clamped {-linear:.2f} -> {rmax:.2f} m/s"); linear = -rmax
        if abs(angular) > amax + 1e-9:
            if not self.clamp:
                raise Blocked("over_speed_angular", f"turn rate {abs(angular):.2f} rad/s exceeds {self.robot}'s limit of {amax:.2f} rad/s")
            notes.append(f"turn rate clamped {abs(angular):.2f} -> {amax:.2f} rad/s"); angular = math.copysign(amax, angular)
        dmax = float(lim["max_duration_s"])
        if duration_arg is None:
            duration = float(lim["default_duration_s"])
            notes.append(f"no duration given; defaulted to {duration:.1f}s")
        else:
            duration = num(duration_arg, "duration_s")
            if duration <= 0:
                raise Blocked("invalid_args", "duration_s must be > 0 for a moving command")
            if duration > dmax + 1e-9:
                if not self.clamp:
                    raise Blocked("over_duration", f"duration {duration:.1f}s exceeds {self.robot}'s maximum of {dmax:.1f}s per command")
                notes.append(f"duration clamped {duration:.1f}s -> {dmax:.1f}s"); duration = dmax
        fence, note = self.check_geofence(linear, angular, duration)
        if note:
            notes.append(note)
        return {"linear_mps": linear, "angular_rps": angular, "duration_s": duration, "fence": fence, "stop": False,
                "notes": notes}


class Interlock:
    """Shared by all robots. ``resolve(robot_arg) -> (name, backend, limits)`` comes from the runtime."""

    def __init__(self, resolve: Callable[[Any], Tuple[str, Any, Dict[str, Any]]],
                 data_dir_fn: Callable[[], Optional[Path]] = lambda: None):
        self._resolve = resolve
        self._data_dir_fn = data_dir_fn
        self._lock = threading.RLock()
        self.nonce = secrets.token_hex(12)
        self._tokens: Dict[str, Dict[str, Any]] = {}
        self._latched: Optional[Dict[str, Any]] = None
        self._latch_loaded = False
        self.blocked: Deque[Dict[str, Any]] = deque(maxlen=500)
        self.counters = {"allowed": 0, "blocked": 0, "modified": 0}

    # ------------------------------------------------------------------ e-stop
    def _latch_path(self) -> Optional[Path]:
        d = self._data_dir_fn()
        return None if d is None else Path(d) / "estop.json"

    def _load_latch(self) -> None:
        if self._latch_loaded:
            return
        self._latch_loaded = True
        p = self._latch_path()
        if p and p.exists():
            try:
                data = json.loads(p.read_text())
                if data.get("latched"):
                    self._latched = data
            except Exception:
                self._latched = {"latched": True, "reason": "estop.json unreadable (failing closed)", "ts": time.time()}

    def estop_status(self) -> Dict[str, Any]:
        with self._lock:
            self._load_latch()
            return dict(self._latched) if self._latched else {"latched": False}

    @property
    def latched(self) -> bool:
        with self._lock:
            self._load_latch()
            return self._latched is not None

    def engage_estop(self, reason: str = "", source: str = "agent") -> Dict[str, Any]:
        with self._lock:
            self._load_latch()
            rec = {"latched": True, "reason": reason or "e-stop engaged", "source": source, "ts": time.time()}
            self._latched = rec
            self._tokens.clear()  # anything vetted before the latch must not run after it
            self._write_latch(rec)
        logger.warning("E-STOP engaged by %s: %s", source, rec["reason"])
        return rec

    def clear_estop(self, source: str = "human") -> bool:
        """Human-only: called from ``/robot clear``. No tool reaches this."""
        with self._lock:
            self._load_latch()
            was = self._latched is not None
            self._latched = None
            self._write_latch({"latched": False, "cleared_by": source, "ts": time.time()})
        if was:
            logger.warning("e-stop cleared by %s", source)
        return was

    def _write_latch(self, rec: Dict[str, Any]) -> None:
        p = self._latch_path()
        if not p:
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec))
            tmp.replace(p)
        except OSError as exc:
            logger.error("could not persist the e-stop latch: %s", exc)

    # --------------------------------------------------------------- audit log
    def record_block(self, tool: str, args: Dict[str, Any], rule: str, reason: str) -> Dict[str, Any]:
        safe = {k: v for k, v in (args or {}).items() if not str(k).startswith("_")}
        text = json.dumps(safe, default=str)
        entry = {"ts": time.time(), "iso": time.strftime("%H:%M:%S"), "tool": tool, "rule": rule, "reason": reason,
                 "args": text if len(text) < 400 else text[:400] + "…"}
        with self._lock:
            self.blocked.append(entry)
            self.counters["blocked"] += 1
        logger.warning("BLOCKED %s [%s]: %s | args=%s", tool, rule, reason, entry["args"])
        d = self._data_dir_fn()
        if d:
            try:
                path = Path(d) / "blocked.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")
                if path.stat().st_size > 1_000_000:
                    path.write_text("\n".join(path.read_text(encoding="utf-8").splitlines()[-300:]) + "\n", encoding="utf-8")
            except OSError:
                pass
        return entry

    def recent_blocked(self, n: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.blocked)[-n:]

    # ------------------------------------------------------------------- rules
    def guard_for(self, robot_arg: Any) -> Guard:
        name, backend, limits = self._resolve(robot_arg)
        return Guard(name, backend, limits, self)

    def evaluate(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return the vetted call ``{"robot", ..., "notes"}`` or raise :class:`Blocked`."""
        try:
            g = self.guard_for(args.get("robot"))
        except Blocked:
            raise
        except Exception as exc:
            raise Blocked("unknown_robot", str(exc))
        caps = g.backend.capabilities
        if tool == "robot_move":
            if MOVE not in caps:
                raise Blocked("unsupported", f"{g.robot} ({g.backend.kind}) cannot be driven with robot_move")
            v = g.check_velocity(num(args.get("linear_mps"), "linear_mps", 0.0), num(args.get("angular_rps"), "angular_rps", 0.0),
                                 args.get("duration_s"))
            return {"robot": g.robot, **v}
        if tool == "robot_turn":
            if TURN not in caps and MOVE not in caps:
                raise Blocked("unsupported", f"{g.robot} ({g.backend.kind}) cannot turn")
            deg = num(args.get("degrees"), "degrees")
            if abs(deg) > 360:
                raise Blocked("invalid_args", "degrees must be between -360 and 360")
            g.require_not_latched()
            rate = min(float(g.limits.get("turn_rate_rps", 0.5)), float(g.limits["max_angular_rps"]))
            notes = []
            if TURN not in caps:  # emulated with a timed rotation: the duration cap applies
                dur = abs(math.radians(deg)) / rate if rate > 0 else math.inf
                if dur > float(g.limits["max_duration_s"]) + 1e-9:
                    raise Blocked("over_duration", f"turning {abs(deg):.0f}° at {rate:.2f} rad/s takes {dur:.1f}s, more than "
                                  f"{g.robot}'s maximum of {float(g.limits['max_duration_s']):.1f}s; turn in smaller steps")
            fence = g.check_clearance(float(getattr(g.backend, "turn_clearance_m", 0.0)))
            if fence:
                notes.append("geofence ok")
            return {"robot": g.robot, "degrees": deg, "turn_rate_rps": rate, "fence": fence, "notes": notes}
        if tool == "robot_action":
            name = str(args.get("action") or "")
            acts = g.backend.actions()
            if name not in acts:
                raise Blocked("unknown_action", f"{g.robot} ({g.backend.kind}) has no action {name!r}; "
                              f"available: {', '.join(sorted(acts)) or 'none'}")
            params = args.get("params") or {}
            if not isinstance(params, dict):
                raise Blocked("invalid_args", "params must be an object")
            notes: List[str] = []
            if acts[name].kind != READ:
                g.require_not_latched()
                params = g.backend.check_action(name, dict(params), g)
                if isinstance(params, dict) and "_notes" in params:
                    notes = list(params.pop("_notes"))
            return {"robot": g.robot, "action": name, "params": params, "kind": acts[name].kind, "fence": g.fence(),
                    "notes": notes}
        raise Blocked("unknown_tool", f"{tool} is not vetted by this interlock")

    # -------------------------------------------------------------- the hook
    def pre_tool_call(self, tool_name: str = "", args: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Hermes ``pre_tool_call`` callback. Fails closed for every tool that can move or write."""
        if tool_name not in VETTED_TOOLS:
            return None
        args = args if isinstance(args, dict) else {}
        try:
            vetted = self.evaluate(tool_name, args)
        except Blocked as b:
            self.record_block(tool_name, args, b.rule, b.reason)
            return {"action": "block", "message": f"hermes-robots interlock BLOCKED {tool_name} [{b.rule}]: {b.reason.rstrip('.')}. {AFTER_BLOCK}"}
        except Exception as exc:  # never fail open
            logger.exception("interlock error")
            self.record_block(tool_name, args, "interlock_error", repr(exc))
            return {"action": "block", "message": f"hermes-robots interlock BLOCKED {tool_name} [interlock_error]: {exc!r} "
                    "(failing closed)"}
        token = secrets.token_hex(8)
        with self._lock:
            self.counters["modified" if vetted["notes"] else "allowed"] += 1
            if len(self._tokens) > 256:
                self._tokens.clear()
            self._tokens[token] = {"tool": tool_name, "vetted": vetted}
        if vetted["notes"]:
            logger.info("%s vetted with notes: %s", tool_name, "; ".join(vetted["notes"]))
        return {"action": "modify", "args": {"_interlock": {"nonce": self.nonce, "token": token}}}

    def redeem(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handlers call this first. Returns the vetted call (the args the hook approved), exactly once."""
        tok = args.get("_interlock")
        if not isinstance(tok, dict) or tok.get("nonce") != self.nonce:
            raise Blocked("interlock_not_applied", "this call was not vetted by the hermes-robots safety interlock "
                          "(the pre_tool_call hook did not run); refusing to act")
        with self._lock:
            rec = self._tokens.pop(str(tok.get("token")), None)
        if rec is None or rec["tool"] != tool:
            raise Blocked("interlock_token_invalid", "the interlock approval for this call is missing, used, or was "
                          "voided by an e-stop; refusing to act")
        if self.latched and not rec["vetted"].get("stop") and rec["vetted"].get("kind") != READ:
            raise Blocked("estop_latched", "the e-stop was latched after this call was approved; refusing to act")
        return rec["vetted"]
