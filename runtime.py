"""Plugin runtime: settings, the robot roster, backend lifecycle, the shared interlock, the data directory."""
from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .backends import BackendError, RobotBackend, get_backend_class
from .config import SAFETY_BASE, Settings, deep_merge
from .safety import Interlock


class Runtime:
    def __init__(self, get_config: Optional[Callable[[str, Any], Any]] = None,
                 data_dir: Optional[Path] = None, overrides: Optional[Dict[str, Any]] = None):
        self.settings = Settings(get_config, overrides)
        self._data_dir = Path(data_dir) if data_dir else None
        self._lock = threading.RLock()
        self._backends: Dict[str, Tuple[str, RobotBackend]] = {}
        self._selected: Optional[str] = None
        self.interlock = Interlock(self.resolve, lambda: self.data_dir)

    # ----------------------------------------------------------------- paths
    @property
    def data_dir(self) -> Path:
        if self._data_dir is None:  # only when the host gave us none (bare unit use)
            self._data_dir = Path(tempfile.mkdtemp(prefix="hermes-robots-"))
        return self._data_dir

    def snapshot_dir(self) -> Path:
        d = self.data_dir / "snapshots"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---------------------------------------------------------------- roster
    def robots(self) -> Dict[str, Dict[str, Any]]:
        robots = self.settings.snapshot()["robots"]
        return {str(k): (v if isinstance(v, dict) else {}) for k, v in robots.items()}

    def active_name(self) -> str:
        robots = self.robots()
        if not robots:
            raise BackendError("no robots are configured (plugins.entries.hermes-robots.settings.robots)")
        if self._selected in robots:
            return self._selected  # type: ignore[return-value]
        configured = str(self.settings.snapshot().get("active") or "")
        return configured if configured in robots else next(iter(robots))

    def select(self, name: str) -> str:
        if name not in self.robots():
            raise BackendError(f"no robot named {name!r}; configured: {', '.join(self.robots())}")
        self._selected = name
        return name

    def backend(self, name: Optional[str] = None) -> RobotBackend:
        name = name or self.active_name()
        robots = self.robots()
        if name not in robots:
            raise BackendError(f"no robot named {name!r}; configured: {', '.join(robots)}")
        cfg = robots[name]
        kind = str(cfg.get("backend") or "")
        key = kind + json.dumps({k: v for k, v in cfg.items() if k != "safety"}, sort_keys=True, default=str)
        with self._lock:
            cur = self._backends.get(name)
            if cur is None or cur[0] != key:
                if cur is not None:
                    try:
                        cur[1].close()
                    except Exception:
                        pass
                cls = get_backend_class(kind)
                d = self.data_dir / "robots" / name
                d.mkdir(parents=True, exist_ok=True)
                self._backends[name] = (key, cls(name, cfg, d))
            return self._backends[name][1]

    def loaded(self) -> List[Tuple[str, RobotBackend]]:
        with self._lock:
            return [(n, b) for n, (_, b) in self._backends.items()]

    def limits(self, name: str, backend: RobotBackend) -> Dict[str, Any]:
        """Effective limits: generic base < backend defaults < settings.safety < robots.<name>.safety, then the
        backend's hard ceilings (which config cannot raise)."""
        snap = self.settings.snapshot()
        lim = deep_merge(SAFETY_BASE, backend.default_safety)
        lim = deep_merge(lim, snap.get("safety") or {})
        lim = deep_merge(lim, (self.robots().get(name) or {}).get("safety") or {})
        for k, cap in (backend.hard_limits or {}).items():
            try:
                if float(lim.get(k, cap)) > float(cap):
                    lim[k] = float(cap)
            except (TypeError, ValueError):
                lim[k] = float(cap)
        return lim

    def resolve(self, robot_arg: Any = None) -> Tuple[str, RobotBackend, Dict[str, Any]]:
        name = str(robot_arg) if robot_arg not in (None, "") else self.active_name()
        be = self.backend(name)
        return name, be, self.limits(name, be)

    def stop_all(self) -> Dict[str, Any]:
        out = {}
        for name, be in self.loaded():
            try:
                be.stop()
                out[name] = "stopped"
            except Exception as exc:
                out[name] = f"stop failed: {exc}"
        return out

    def shutdown(self) -> None:
        with self._lock:
            for _, be in self.loaded():
                try:
                    be.stop()
                except Exception:
                    pass
                try:
                    be.close()
                except Exception:
                    pass
            self._backends.clear()
