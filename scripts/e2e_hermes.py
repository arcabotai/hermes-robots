#!/usr/bin/env python3
"""Exercise an *installed* hermes-robots through Hermes' real tool dispatch (pre_tool_call hooks included).

Run it with the Python of a Hermes install, against a throwaway HERMES_HOME where the plugin was installed:

    export HERMES_HOME=$(mktemp -d)
    hermes plugins install arcabotai/hermes-robots --enable
    hermes plugins validate "$HERMES_HOME/plugins/hermes-robots"
    <hermes-venv>/bin/python scripts/e2e_hermes.py [--ros host:port] [--microduck http://127.0.0.1:8765]

It writes the plugin settings into $HERMES_HOME/config.yaml (mock + mock Microduck always; a turtlesim/TurtleBot3
robot and the MuJoCo Microduck when given), then drives each robot through ``model_tools.handle_function_call``,
the same entry point the agent loop uses, and the ``/robot`` command handler. Exit code 0 = every check passed.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--ros", default="", help="host:port of a rosbridge with turtlesim (scripts/ros_demo) or a TurtleBot3")
ap.add_argument("--microduck", default="", help="URL of sim/microduck/server.py")
opt = ap.parse_args()

home = Path(os.environ.get("HERMES_HOME", ""))
if not (home / "plugins" / "hermes-robots").exists():
    sys.exit("HERMES_HOME must point at a home where hermes-robots is installed")

robots = {"mock": {"backend": "mock"}, "duck": {"backend": "microduck", "mode": "mock"}}
if opt.ros:
    host, _, port = opt.ros.partition(":")
    robots["turtle"] = {"backend": "ros", "host": host, "port": int(port or 9090),
                        "topics": {"cmd_vel": "/turtle1/cmd_vel", "odom": "/turtle1/pose", "odom_type": "turtlesim/msg/Pose"},
                        "safety": {"max_linear_mps": 1.0, "max_angular_rps": 1.5,
                                   "write_allowlist": {"topics": ["/turtle1/cmd_vel"], "services": ["/turtle1/teleport_absolute"]},
                                   "geofence": {"enabled": True, "x_min": 1.0, "x_max": 10.0, "y_min": 1.0, "y_max": 10.0}}}
if opt.microduck:
    robots["mujoco_duck"] = {"backend": "microduck", "mode": "sim", "url": opt.microduck}

import yaml  # noqa: E402

cfg_path = home / "config.yaml"
cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
cfg = cfg or {}
cfg.setdefault("plugins", {}).setdefault("entries", {}).setdefault("hermes-robots", {})["settings"] = {"robots": robots}
cfg_path.write_text(yaml.safe_dump(cfg))

from hermes_cli.plugins import discover_plugins, get_plugin_command_handler  # noqa: E402

discover_plugins(force=True)
import model_tools  # noqa: E402

failures = []


def call(tool, args=None, expect="ok"):
    out = model_tools.handle_function_call(tool, dict(args or {}), task_id="e2e", session_id="e2e")
    try:
        res = json.loads(out)
    except Exception:
        res = {"raw": out}
    text = json.dumps(res)
    blocked = "interlock BLOCKED" in text
    good = (expect == "ok" and res.get("ok") is True) or (expect == "blocked" and blocked)
    tag = "PASS" if good else "FAIL"
    if not good:
        failures.append((tool, args, text[:400]))
    print(f"[{tag}] {tool} {json.dumps(args or {})} -> {text[:220]}")
    return res


def section(name):
    print(f"\n=== {name}")


section("plugin loaded through Hermes")
lst = call("robot_list")
print("    robots:", [r["robot"] + ":" + str(r["kind"]) for r in lst.get("robots", [])])
cmd = get_plugin_command_handler("robot")
print("    /robot list ->", (cmd("list") if cmd else "MISSING").replace("\n", " | "))
if not cmd:
    failures.append(("/robot", None, "command not registered"))


def drive(robot, fwd, slow_ok=True):
    section(f"{robot}: connect, look, move, turn, interlock, e-stop")
    call("robot_connect", {"robot": robot})
    call("robot_state", {"robot": robot})
    call("robot_look", {"robot": robot})
    if slow_ok:
        r = call("robot_move", {"robot": robot, "linear_mps": fwd, "duration_s": 2})
        print("    displacement_m:", r.get("displacement_m"), "end_reason:", r.get("end_reason"))
    call("robot_turn", {"robot": robot, "degrees": 30})
    call("robot_move", {"robot": robot, "linear_mps": 5.0, "duration_s": 30}, expect="blocked")
    call("robot_move", {"robot": robot, "linear_mps": fwd, "duration_s": 999}, expect="blocked")
    call("robot_stop", {"robot": robot, "estop": True, "reason": "e2e"})
    call("robot_move", {"robot": robot, "linear_mps": fwd * 0.5, "duration_s": 1}, expect="blocked")
    print("    /robot clear ->", cmd("clear") if cmd else None)
    call("robot_move", {"robot": robot, "linear_mps": fwd, "duration_s": 1})


drive("mock", 0.3)
drive("duck", 0.25)
section("duck: actions")
call("robot_action", {"robot": "duck", "action": "kick_ball"})
call("robot_action", {"robot": "duck", "action": "head", "params": {"yaw": 1.5}}, expect="blocked")
if opt.ros:
    call("robot_action", {"robot": "turtle", "action": "call_service",
                          "params": {"service": "/turtle1/teleport_absolute", "request": {"x": 5.5, "y": 5.5, "theta": 0.0}}})
    drive("turtle", 0.5)
    call("robot_action", {"robot": "turtle", "action": "topics", "params": {"kind": "topics"}})
    call("robot_action", {"robot": "turtle", "action": "call_service", "params": {"service": "/kill"}}, expect="blocked")
if opt.microduck:
    drive("mujoco_duck", 0.25)
    call("robot_action", {"robot": "mujoco_duck", "action": "kick_ball"})

section("/robot log")
print(cmd("log 20") if cmd else "")
print(f"\nRESULT: {'FAILED ' + str(len(failures)) if failures else 'ALL PASSED'}")
for f in failures:
    print("  FAIL", f)
sys.exit(1 if failures else 0)
