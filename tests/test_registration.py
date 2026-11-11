"""Plugin contract: what register() exposes matches plugin.yaml, it opens no connections, and /robot works."""
import socket
from pathlib import Path

import pytest
import yaml

import hermes_robots
from hermes_robots import config

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = yaml.safe_load((ROOT / "plugin.yaml").read_text())


class FakeCtx:
    def __init__(self, tmp_path, cfg=None):
        self.tools, self.hooks, self.commands, self.skills, self.unload = {}, {}, {}, {}, []
        self.cfg = cfg or {}
        self.state = type("S", (), {"data_dir": tmp_path})()

    def get_config(self, key, default=None):
        return self.cfg.get(key, default)

    def register_tool(self, **kw):
        self.tools[kw["name"]] = kw

    def register_hook(self, name, fn):
        self.hooks.setdefault(name, []).append(fn)

    def register_command(self, name, handler, **kw):
        self.commands[name] = (handler, kw)

    def register_skill(self, name, path, description=""):
        self.skills[name] = Path(path)

    def on_unload(self, fn):
        self.unload.append(fn)


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("register() must not open connections")
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    c = FakeCtx(tmp_path)
    hermes_robots.register(c)
    yield c
    monkeypatch.undo()
    for fn in c.unload:
        fn()


def test_tools_match_manifest(ctx):
    assert sorted(ctx.tools) == sorted(MANIFEST["provides_tools"])
    for name, t in ctx.tools.items():
        assert t["toolset"] == "robots" and callable(t["handler"]) and t["schema"]["name"] == name


def test_hook_command_skill(ctx):
    assert list(ctx.hooks) == MANIFEST["provides_hooks"] == ["pre_tool_call"]
    assert "robot" in ctx.commands
    assert ctx.skills["robot-operation"].exists()


def test_manifest_defaults_match_config():
    for key, spec in MANIFEST["config_schema"].items():
        assert spec["default"] == config.DEFAULTS[key], key


def test_every_moving_tool_is_vetted_and_reads_are_not():
    from hermes_robots.safety import ALL_TOOLS, VETTED_TOOLS
    assert ALL_TOOLS == set(MANIFEST["provides_tools"])
    assert VETTED_TOOLS == {"robot_move", "robot_turn", "robot_action"}


def test_slash_command_round_trip(ctx):
    cmd = ctx.commands["robot"][0]
    assert "robot: mock" in cmd("status")
    assert "duck" in cmd("list")
    assert "Active robot: duck" in cmd("use duck")
    assert "LATCHED" in cmd("estop testing")
    assert "cleared" in cmd("clear")
    assert "not latched" in cmd("clear")
    assert "No blocked" in cmd("log")
    assert "respawned" in cmd("reset 0.5 0.1")
    assert "unknown subcommand" in cmd("dance")


def test_config_robots_replace_the_builtin_pair(tmp_path):
    c = FakeCtx(tmp_path, {"robots": {"tb3": {"backend": "ros", "host": "10.0.0.5"}}, "active": "tb3"})
    hermes_robots.register(c)
    rt = hermes_robots.get_runtime()
    assert list(rt.robots()) == ["tb3"] and rt.active_name() == "tb3"
    for fn in c.unload:
        fn()
