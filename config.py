"""Plugin settings: ``plugins.entries.hermes-robots.settings`` read through ``ctx.get_config`` on every call.

Layout::

    active: duck                 # robot the tools use when they are not given one (default: the first robot)
    robots:                      # name -> {backend: <kind>, ...backend options, safety: {...per-robot limits}}
      duck: {backend: microduck, mode: sim, url: http://127.0.0.1:8765}
      tb3:  {backend: ros, host: localhost, port: 9090}
    safety: {...}                # limits for every robot (a robot's own ``safety`` wins)
    camera: {width, height, return_mode: path|inline, keep_last}

If ``robots`` is set it replaces the built-in pair (a mock base and a mock Microduck) entirely.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict

DEFAULTS: Dict[str, Any] = {
    "active": "",
    "robots": {
        "mock": {"backend": "mock"},
        "duck": {"backend": "microduck", "mode": "mock"},
    },
    "safety": {},
    "camera": {"width": 640, "height": 480, "return_mode": "path", "keep_last": 20},
}

# Generic limits every robot starts from; a backend's ``default_safety`` and then the settings are merged over them.
SAFETY_BASE: Dict[str, Any] = {
    "max_linear_mps": 0.3,       # forward
    "max_reverse_mps": 0.2,      # backward
    "max_angular_rps": 1.0,
    "max_duration_s": 5.0,       # per robot_move
    "default_duration_s": 1.0,   # injected when the model leaves duration out
    "turn_rate_rps": 0.5,        # robot_turn rotation speed for backends without a native turn
    "over_limit_action": "block",  # block | clamp
    "geofence": {"enabled": False, "x_min": -2.0, "x_max": 2.0, "y_min": -2.0, "y_max": 2.0,
                 "margin_m": 0.2, "max_pose_age_s": 5.0},
}


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Settings:
    """Defaults + host config (+ test overrides). Re-read on every call so config edits apply without a reload."""

    def __init__(self, getter: Callable[[str, Any], Any] | None = None, overrides: Dict[str, Any] | None = None):
        self._getter = getter
        self._overrides = overrides or {}

    def _raw(self, key: str) -> Any:
        if key in self._overrides:
            return copy.deepcopy(self._overrides[key])
        if self._getter is None:
            return None
        try:
            return self._getter(key, None)
        except Exception:
            return None

    def snapshot(self) -> Dict[str, Any]:
        out = copy.deepcopy(DEFAULTS)
        for key, default in DEFAULTS.items():
            value = self._raw(key)
            if value is None:
                continue
            if key == "robots":
                if isinstance(value, dict) and value:
                    out[key] = copy.deepcopy(value)  # replaces the built-in robots
            elif isinstance(default, dict) and isinstance(value, dict):
                out[key] = deep_merge(default, value)
            elif not isinstance(default, dict):
                out[key] = value
        return out

    def __getitem__(self, key: str) -> Any:
        return self.snapshot()[key]
