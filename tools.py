"""Tool handlers. Every handler returns JSON text (or a multimodal envelope for robot_look) and never raises."""
from __future__ import annotations

import base64
import json
import time
from typing import Any, Callable, Dict, List

from . import imaging
from .backends import BackendError, MoveContext, RobotBackend, available
from .backends.base import Blocked
from .runtime import Runtime

LIMIT_KEYS = ("max_linear_mps", "max_reverse_mps", "max_angular_rps", "max_duration_s", "default_duration_s",
              "over_limit_action", "geofence")


def _ok(**kw: Any) -> str:
    return json.dumps({"ok": True, **kw}, default=str)


def _err(msg: str, **kw: Any) -> str:
    return json.dumps({"ok": False, "error": msg, **kw}, default=str)


def describe(rt: Runtime, name: str, be: RobotBackend, limits: Dict[str, Any]) -> Dict[str, Any]:
    extra = {k: v for k, v in limits.items() if k not in LIMIT_KEYS and k != "turn_rate_rps"}
    return {"robot": name, "kind": be.kind, "via": getattr(be, "description", ""),
            "capabilities": sorted(be.capabilities), "views": list(be.views()),
            "limits": {**{k: limits[k] for k in LIMIT_KEYS if k in limits}, **extra},
            "hard_limits": dict(be.hard_limits),
            "actions": {a.name: {"kind": a.kind, "description": a.description, "params": a.params} for a in be.actions().values()}}


def make_handlers(rt: Runtime) -> Dict[str, Callable[..., Any]]:
    def guarded(fn: Callable[[Dict[str, Any]], Any]) -> Callable[..., Any]:
        def handler(args: Dict[str, Any], **kwargs: Any) -> Any:
            args = args if isinstance(args, dict) else {}
            try:
                return fn(args)
            except Blocked as b:
                rt.interlock.record_block(fn.__name__, args, b.rule, b.reason)
                return _err(f"hermes-robots interlock BLOCKED {fn.__name__} [{b.rule}]: {b.reason}")
            except BackendError as exc:
                return _err(str(exc))
            except Exception as exc:  # tool contract: always JSON, never raise
                return _err(f"{type(exc).__name__}: {exc}")
        handler.__name__ = fn.__name__
        return handler

    def ctx_for(vetted: Dict[str, Any]) -> MoveContext:
        return MoveContext(fence=vetted.get("fence"),
                           should_abort=lambda: "e-stop latched" if rt.interlock.latched else None)

    # ------------------------------------------------------------------ read
    def robot_list(a):
        active = rt.active_name()
        loaded = dict(rt.loaded())
        rows: List[Dict[str, Any]] = []
        for name, cfg in rt.robots().items():
            row: Dict[str, Any] = {"robot": name, "kind": cfg.get("backend"), "active": name == active,
                                   "loaded": name in loaded}
            if cfg.get("mode"):
                row["mode"] = cfg["mode"]
            try:
                be = rt.backend(name)  # constructing a backend opens no connection
                row.update(capabilities=sorted(be.capabilities), actions=sorted(be.actions()), via=getattr(be, "description", ""))
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)
        return _ok(robots=rows, active=active, backends_available=available(), estop=rt.interlock.estop_status())

    def robot_connect(a):
        name = rt.select(str(a.get("robot") or ""))
        name, be, limits = rt.resolve(name)
        info = be.connect()
        return _ok(active=name, connection=info, **{k: v for k, v in describe(rt, name, be, limits).items() if k != "robot"},
                   state=be.state(), estop=rt.interlock.estop_status())

    def robot_state(a):
        name, be, limits = rt.resolve(a.get("robot"))
        return _ok(**describe(rt, name, be, limits), state=be.state(), estop=rt.interlock.estop_status(),
                   counters=rt.interlock.counters,
                   recent_blocked=[f"{r['iso']} {r['tool']} [{r['rule']}] {r['reason']}" for r in rt.interlock.recent_blocked(3)])

    def robot_look(a):
        name, be, _ = rt.resolve(a.get("robot"))
        cam = rt.settings.snapshot()["camera"]
        views = be.views()
        view = a.get("view") if a.get("view") in views else views[0]
        data, source = be.look(view, int(cam["width"]), int(cam["height"]))
        mime, ext, size = imaging.sniff_image(data)
        out_dir = rt.snapshot_dir()
        path = out_dir / f"{name}-{view}-{time.strftime('%Y%m%d-%H%M%S')}-{int((time.time() % 1) * 1000):03d}.{ext}"
        path.write_bytes(data)
        snaps = sorted((p for p in out_dir.iterdir() if p.is_file()), key=lambda p: p.stat().st_mtime)
        for old in snaps[:-max(1, int(cam.get("keep_last", 20)))]:
            try:
                old.unlink()
            except OSError:
                pass
        inline = str(cam.get("return_mode", "path")).lower() == "inline"
        text = _ok(robot=name, view=view, path=str(path), mime=mime, width=size[0] if size else None,
                   height=size[1] if size else None, source=source, media_tag=f"MEDIA:{path}",
                   next_step=("" if inline else f"Call vision_analyze with image_url='{path}' to see it. ")
                   + f"To show the picture to the user, include the line MEDIA:{path} in your reply.")
        if inline:
            return {"_multimodal": True,
                    "content": [{"type": "text", "text": text},
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(data).decode()}}],
                    "text_summary": text}
        return text

    # ---------------------------------------------------------------- vetted
    def robot_move(a):
        v = rt.interlock.redeem("robot_move", a)
        be = rt.backend(v["robot"])
        if v["stop"]:
            return _ok(robot=v["robot"], interlock=v["notes"], **be.stop())
        res = be.move(v["linear_mps"], v["angular_rps"], v["duration_s"], ctx_for(v))
        return _ok(robot=v["robot"], command={k: v[k] for k in ("linear_mps", "angular_rps", "duration_s")},
                   interlock=v["notes"], **res)

    def robot_turn(a):
        v = rt.interlock.redeem("robot_turn", a)
        be = rt.backend(v["robot"])
        return _ok(robot=v["robot"], degrees=v["degrees"], interlock=v["notes"],
                   **be.turn(v["degrees"], ctx_for(v), v["turn_rate_rps"]))

    def robot_action(a):
        v = rt.interlock.redeem("robot_action", a)
        be = rt.backend(v["robot"])
        return _ok(robot=v["robot"], action=v["action"], interlock=v["notes"], **be.run_action(v["action"], v["params"], ctx_for(v)))

    def robot_stop(a):
        target = str(a.get("robot") or "")
        latched = None
        if a.get("estop"):
            latched = rt.interlock.engage_estop(str(a.get("reason") or "agent e-stop"), source="agent")
        if latched or target == "all":
            results = rt.stop_all()
            if not results:  # nothing loaded yet: still stop the active robot
                target = ""
        else:
            results = {}
        if not results:
            name, be, _ = rt.resolve(target or None)
            try:
                be.stop()
                results[name] = "stopped"
            except Exception as exc:
                results[name] = f"stop failed: {exc}"
        failed = {k: r for k, r in results.items() if r != "stopped"}
        return json.dumps({"ok": not failed, "stopped": results, "estop": latched or rt.interlock.estop_status(),
                           "note": "e-stop latched: only the human can clear it with /robot clear" if latched else ""},
                          default=str)

    fns = [robot_list, robot_connect, robot_state, robot_look, robot_move, robot_turn, robot_action, robot_stop]
    return {f.__name__: guarded(f) for f in fns}
