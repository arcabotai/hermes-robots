"""The backend interface every robot implements, plus the small types shared with the interlock.

A backend is one *kind* of robot (``mock``, ``microduck``, ``ros``, ...). The plugin creates one backend instance per
robot entry in the settings. Backends never talk to the agent directly: the generic tools call them only after the
shared safety interlock (``safety.py``) vetted the call, and they report what happened as plain dicts.

Adding a robot kind means subclassing :class:`RobotBackend`, filling in the parts it supports (declared in
``capabilities``), and registering the class with :func:`backends.register_backend`. The interlock, tools, ``/robot``
command and audit log work unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Optional, Tuple

# capability names the generic tools understand
MOVE = "move"          # robot_move: body velocity (linear m/s, angular rad/s) for a bounded duration
TURN = "turn"          # robot_turn: rotate in place by an angle
POSE = "pose"          # state() reports pose {x, y, yaw_deg} in the geofence frame
CAMERA = "camera"      # robot_look returns an image

READ, MOTION, WRITE = "read", "motion", "write"


class BackendError(RuntimeError):
    """A recoverable backend failure (not connected, timeout, unknown topic, ...). Shown to the agent as an error."""


class Blocked(Exception):
    """Raised by the interlock (or a backend's action validator) to refuse a call. ``rule`` is a stable id."""

    def __init__(self, rule: str, reason: str):
        super().__init__(reason)
        self.rule, self.reason = rule, reason


@dataclass(frozen=True)
class Action:
    """A backend-specific extra exposed through ``robot_action``.

    ``kind`` decides how the interlock treats it: ``read`` never moves anything; ``motion`` and ``write`` are refused
    while the e-stop is latched and must pass the backend's ``check_action`` validator.
    """
    name: str
    description: str
    kind: str = MOTION
    params: Dict[str, Any] = field(default_factory=dict)  # JSON-schema "properties"


@dataclass
class MoveContext:
    """What the interlock hands a moving backend call: the vetted geofence (or None) and an abort check."""
    fence: Optional[Dict[str, float]] = None
    should_abort: Callable[[], Optional[str]] = lambda: None


def num(v: Any, what: str, default: Optional[float] = None) -> float:
    """Strict finite-number parsing for tool arguments; raises :class:`Blocked` (rule ``invalid_args``)."""
    if v is None:
        if default is None:
            raise Blocked("invalid_args", f"{what} is required")
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise Blocked("invalid_args", f"{what} must be a number, got {type(v).__name__}")
    try:
        f = float(v)
    except ValueError:
        raise Blocked("invalid_args", f"{what} must be a number, got {v!r}")
    if not math.isfinite(f):
        raise Blocked("invalid_args", f"{what} must be finite, got {v!r}")
    return f


class RobotBackend:
    """Base class. Only ``state`` and ``stop`` are mandatory; everything else depends on ``capabilities``."""

    kind = "base"
    description = ""
    capabilities: FrozenSet[str] = frozenset()
    # safety defaults for this kind of robot (merged under the user's settings)
    default_safety: Dict[str, Any] = {}
    # ceilings the settings cannot raise (e.g. a learned policy's trained speed range). Keys like default_safety.
    hard_limits: Dict[str, float] = {}
    # a turn in place sweeps this radius (a walking robot turns on a small arc); the geofence keeps this clearance
    turn_clearance_m = 0.0

    def __init__(self, name: str, cfg: Dict[str, Any], data_dir: Path):
        self.name = name
        self.cfg = cfg
        self.data_dir = data_dir

    # ----------------------------------------------------------------- lifecycle
    def connect(self) -> Dict[str, Any]:
        """Open the link if there is one and return a short description. Called lazily, never at plugin load."""
        return {"connected": True}

    def close(self) -> None:
        pass

    # ---------------------------------------------------------------------- read
    def state(self) -> Dict[str, Any]:
        """Current state. With the ``pose`` capability it must include ``pose: {x, y, yaw_deg, age_s?}``."""
        raise NotImplementedError

    def look(self, view: str, width: int, height: int) -> Tuple[bytes, str]:
        """``(png_or_jpeg_bytes, source description)`` for the ``camera`` capability."""
        raise BackendError(f"{self.kind} robots have no camera")

    def views(self) -> Tuple[str, ...]:
        return ("camera",)

    # -------------------------------------------------------------------- motion
    def move(self, linear_mps: float, angular_rps: float, duration_s: float, ctx: MoveContext) -> Dict[str, Any]:
        raise BackendError(f"{self.kind} robots cannot be driven with robot_move")

    def turn(self, degrees: float, ctx: MoveContext, angular_rps: float) -> Dict[str, Any]:
        """Default: rotate in place with ``move`` at ``angular_rps`` (already within the limits)."""
        if degrees == 0:
            return {"end_reason": "completed", "turned_deg": 0.0}
        w = math.copysign(abs(angular_rps), degrees)
        return self.move(0.0, w, abs(math.radians(degrees)) / abs(w), ctx)

    def stop(self) -> Dict[str, Any]:
        """Stop all motion now. Must be safe to call at any time, including while a move is running."""
        raise NotImplementedError

    # ------------------------------------------------------------------- extras
    def actions(self) -> Dict[str, Action]:
        return {}

    def check_action(self, action: str, params: Dict[str, Any], guard: Any) -> Dict[str, Any]:
        """Validate a ``motion``/``write`` action before it runs; raise :class:`Blocked` to refuse.

        ``guard`` is the interlock's :class:`safety.Guard` for this robot (limits, ``check_velocity``,
        ``check_geofence``). Return the (possibly adjusted) params; they reach :meth:`run_action` unchanged.
        """
        return params

    def run_action(self, action: str, params: Dict[str, Any], ctx: MoveContext) -> Dict[str, Any]:
        raise BackendError(f"unknown action {action!r}")

    def reset(self, **kw: Any) -> Dict[str, Any]:
        """Human-only (``/robot reset``): respawn a simulated robot. Real robots refuse."""
        raise BackendError(f"{self.kind} robots cannot be reset")
