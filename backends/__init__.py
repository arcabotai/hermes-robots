"""Backend registry: robot kind name -> backend class.

Built-in kinds are imported lazily so loading the plugin stays cheap and offline. A new kind (e.g. Reachy Mini or a
real base) is one module with a :class:`~.base.RobotBackend` subclass plus one ``register_backend`` call.
"""
from __future__ import annotations

import importlib
from typing import Dict, List, Type

from .base import (CAMERA, MOTION, MOVE, POSE, READ, TURN, WRITE, Action, BackendError, Blocked,  # noqa: F401  (re-exported)
                   MoveContext, RobotBackend, num)

_BUILTIN = {
    "mock": ("mock", "MockBackend"),
    "microduck": ("microduck", "MicroduckBackend"),
    "ros": ("ros", "RosBackend"),
}
_REGISTERED: Dict[str, Type[RobotBackend]] = {}


def register_backend(kind: str, cls: Type[RobotBackend]) -> None:
    """Make ``backend: <kind>`` available in the robot settings."""
    if not (isinstance(cls, type) and issubclass(cls, RobotBackend)):
        raise TypeError("backend classes must subclass RobotBackend")
    _REGISTERED[kind] = cls


def unregister_backend(kind: str) -> None:
    _REGISTERED.pop(kind, None)


def available() -> List[str]:
    return sorted(set(_BUILTIN) | set(_REGISTERED))


def get_backend_class(kind: str) -> Type[RobotBackend]:
    if kind in _REGISTERED:
        return _REGISTERED[kind]
    if kind in _BUILTIN:
        mod, attr = _BUILTIN[kind]
        return getattr(importlib.import_module(f"{__name__}.{mod}"), attr)
    raise BackendError(f"unknown robot backend {kind!r}; available: {', '.join(available())}")
