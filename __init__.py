"""hermes-robots: drive robots from Hermes (Microduck, ROS 2 robots, a mock) behind one shared safety interlock."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import schemas
from .backends import register_backend  # noqa: F401  (public: lets other code add robot kinds)
from .commands import make_command
from .runtime import Runtime
from .tools import make_handlers

_RUNTIME: Optional[Runtime] = None


def get_runtime() -> Optional[Runtime]:
    return _RUNTIME


def register(ctx) -> None:
    """Register the robot tools, the pre_tool_call interlock, /robot and the bundled skill. Opens no connections."""
    global _RUNTIME
    try:
        data_dir = Path(ctx.state.data_dir)
    except Exception:
        data_dir = None
    rt = Runtime(get_config=ctx.get_config, data_dir=data_dir)
    _RUNTIME = rt
    handlers = make_handlers(rt)
    for schema in schemas.ALL:
        ctx.register_tool(name=schema["name"], toolset="robots", schema=schema, handler=handlers[schema["name"]],
                          emoji="🤖", description=schema["description"].split(". ")[0])
    ctx.register_hook("pre_tool_call", rt.interlock.pre_tool_call)
    ctx.register_command("robot", make_command(rt), description="Robot link and safety: status, list, use, estop, clear, log, stop, reset",
                         args_hint="[status|list|use|estop|clear|log|stop|reset]")
    skill = Path(__file__).parent / "skills" / "robot-operation" / "SKILL.md"
    if skill.exists():
        ctx.register_skill("robot-operation", skill)
    ctx.on_unload(rt.shutdown)
