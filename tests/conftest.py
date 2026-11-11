"""Load the repo root as the importable package ``hermes_robots`` (the repo *is* the plugin directory)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if "hermes_robots" not in sys.modules:
    spec = importlib.util.spec_from_file_location("hermes_robots", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hermes_robots"] = mod
    spec.loader.exec_module(mod)

from hermes_robots.runtime import Runtime  # noqa: E402
from hermes_robots.tools import make_handlers  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--live-ros", default="", help="host:port of a real rosbridge (TurtleBot3 / turtlesim) for tests/live")
    parser.addoption("--live-microduck", default="", help="URL of a running sim/microduck/server.py for tests/live")


class Harness:
    """Runs tool calls the way Hermes does: pre_tool_call hook first (its modify args shallow-merged), then the handler."""

    def __init__(self, tmp_path, overrides=None):
        self.rt = Runtime(None, tmp_path, overrides)
        self.handlers = make_handlers(self.rt)

    def call(self, tool, args=None):
        args = dict(args or {})
        directive = self.rt.interlock.pre_tool_call(tool_name=tool, args=args)
        if directive and directive.get("action") == "block":
            return {"blocked": True, "message": directive["message"]}
        if directive and directive.get("action") == "modify":
            args = {**args, **directive["args"]}
        out = self.handlers[tool](args)
        if isinstance(out, dict):  # multimodal envelope
            out = out["text_summary"]
        res = json.loads(out)
        res["blocked"] = False
        return res

    def raw(self, tool, args=None):
        """Call the handler *without* the hook (a missing or crashed hook)."""
        return json.loads(self.handlers[tool](dict(args or {})))


@pytest.fixture
def make(tmp_path):
    made = []

    def _make(overrides=None):
        h = Harness(tmp_path, overrides)
        made.append(h)
        return h
    yield _make
    for h in made:
        h.rt.shutdown()


@pytest.fixture
def bot(make):
    return make()
