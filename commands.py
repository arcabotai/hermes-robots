"""``/robot`` slash command: the human's controls. The agent cannot type slash commands, so ``clear`` is human-only."""
from __future__ import annotations

import json
from typing import Callable

from .runtime import Runtime

HELP = """hermes-robots - robot link and safety controls
  /robot [status] [name]     active robot (or name): pose, limits, e-stop
  /robot list                configured robots
  /robot use <name>          make <name> the active robot
  /robot estop [reason]      latch the e-stop now and stop every robot (all motion refused)
  /robot clear               clear the e-stop latch (human-only)
  /robot log [n]             last n blocked calls with reasons
  /robot stop                stop every robot now (no latch)
  /robot reset [args...]     respawn a simulated robot (mock: x y yaw_deg; microduck: ball_x ball_y)"""


def make_command(rt: Runtime) -> Callable[[str], str]:
    from .tools import make_handlers
    h = make_handlers(rt)

    def handle(raw_args: str = "") -> str:
        parts = (raw_args or "").strip().split()
        sub = parts[0].lower() if parts else "status"
        if sub in ("help", "-h", "--help"):
            return HELP
        if sub == "estop":
            rec = rt.interlock.engage_estop(" ".join(parts[1:]) or "human e-stop via /robot estop", source="human")
            stopped = rt.stop_all()
            return f"E-STOP LATCHED ({rec['reason']}). All robot motion is refused. Stopped: {stopped or 'no robot loaded'}. " \
                   "Clear with /robot clear."
        if sub == "clear":
            return "E-stop cleared." if rt.interlock.clear_estop("human") else "E-stop was not latched."
        if sub == "stop":
            return f"Stopped: {rt.stop_all() or 'no robot loaded'}"
        if sub == "use":
            if len(parts) < 2:
                return "usage: /robot use <name>"
            try:
                return f"Active robot: {rt.select(parts[1])}"
            except Exception as exc:
                return str(exc)
        if sub == "log":
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
            rows = rt.interlock.recent_blocked(n)
            if not rows:
                return "No blocked calls this session."
            return "\n".join(f"{r['iso']}  {r['tool']}  [{r['rule']}]  {r['reason']}" for r in rows)
        if sub == "reset":
            try:
                name, be, _ = rt.resolve(None)
                nums = [float(x) for x in parts[1:]]
                if be.kind == "microduck":
                    kw = {"ball_x": nums[0], "ball_y": nums[1] if len(nums) > 1 else 0.0} if nums else {}
                else:
                    kw = dict(zip(("x", "y", "yaw_deg"), nums))
                be.reset(**kw)
                return f"{name} respawned."
            except Exception as exc:
                return f"reset failed: {exc}"
        if sub == "list":
            res = json.loads(h["robot_list"]({}))
            if not res.get("ok"):
                return res.get("error", "list failed")
            return "\n".join(f"{'*' if r['active'] else ' '} {r['robot']:<12} {r['kind']}{' (' + r['mode'] + ')' if r.get('mode') else ''}"
                             f"  actions: {', '.join(r.get('actions') or []) or '-'}" for r in res["robots"])
        if sub == "status":
            res = json.loads(h["robot_state"]({"robot": parts[1]} if len(parts) > 1 else {}))
            if not res.get("ok"):
                return res.get("error", "status failed")
            p, lim, est = (res["state"] or {}).get("pose") or {}, res["limits"], res["estop"]
            pose = f"x={p['x']:.2f} y={p['y']:.2f} yaw={p.get('yaw_deg', 0):.0f}°" if "x" in p else "unknown"
            gf = lim.get("geofence") or {}
            return "\n".join([
                f"robot: {res['robot']} ({res['kind']}, {res['via']})",
                f"pose: {pose}",
                f"limits: ≤{lim['max_linear_mps']} m/s fwd, ≤{lim['max_reverse_mps']} m/s back, ≤{lim['max_angular_rps']} rad/s, "
                f"≤{lim['max_duration_s']} s per move ({lim['over_limit_action']})",
                f"geofence: {'x[{x_min},{x_max}] y[{y_min},{y_max}] margin {margin_m}'.format(**gf) if gf.get('enabled') else 'off'}",
                f"e-stop: {'LATCHED - ' + str(est.get('reason')) if est.get('latched') else 'clear'}",
                f"interlock: {rt.interlock.counters}"])
        return f"unknown subcommand {sub!r}\n{HELP}"

    return handle
