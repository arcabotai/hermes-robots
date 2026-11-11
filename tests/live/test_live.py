"""End-to-end against REAL robots. Opt-in; skipped unless pointed at one:

    pytest tests/live --live-ros=localhost:9090            # rosbridge + turtlesim (scripts/ros_demo) or a TurtleBot3
    pytest tests/live --live-microduck=http://127.0.0.1:8765   # sim/microduck/server.py (MuJoCo + ONNX policies)

scripts/blaxel_e2e.py brings up both in a Blaxel sandbox and runs these.
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import Harness  # noqa: E402
from helpers import red_pixels  # noqa: E402


# ------------------------------------------------------------------------------------------------ ROS 2
@pytest.fixture(scope="module")
def ros(request, tmp_path_factory):
    target = request.config.getoption("--live-ros")
    if not target:
        pytest.skip("pass --live-ros=host:port to run against a live rosbridge")
    host, _, port = target.partition(":")
    base = {"backend": "ros", "host": host, "port": int(port or 9090), "connect_timeout_s": 10}
    probe = Harness(tmp_path_factory.mktemp("probe"), {"robots": {"r": base}})
    names = {t["name"] for t in probe.call("robot_action", {"action": "topics", "params": {"kind": "topics"}})["topics"]}
    probe.rt.shutdown()
    if "/turtle1/cmd_vel" in names:  # turtlesim: arena 0..11, pose on /turtle1/pose
        robot = {**base, "topics": {"cmd_vel": "/turtle1/cmd_vel", "odom": "/turtle1/pose", "odom_type": "turtlesim/msg/Pose"},
                 "safety": {"max_linear_mps": 1.0, "max_angular_rps": 1.5, "max_duration_s": 6.0,
                            "write_allowlist": {"topics": ["/turtle1/cmd_vel"], "services": ["/turtle1/teleport_absolute"]},
                            "geofence": {"enabled": True, "x_min": 1.0, "x_max": 10.0, "y_min": 1.0, "y_max": 10.0, "margin_m": 0.2}}}
    else:  # TurtleBot3 (or any base with /cmd_vel + /odom)
        robot = {**base, "safety": {"max_linear_mps": 0.2, "max_angular_rps": 1.0}}
    h = Harness(tmp_path_factory.mktemp("ros"), {"robots": {"live": robot}, "camera": {"keep_last": 200}})
    h.turtlesim = "/turtle1/cmd_vel" in names
    h.names = names
    yield h
    h.call("robot_stop")
    h.rt.shutdown()


def test_ros_connect_and_pose(ros):
    con = ros.call("robot_connect", {"robot": "live"})
    assert con["ok"], con
    pose = ros.call("robot_state")["state"]["pose"]
    assert pose and math.isfinite(pose["x"]) and pose["age_s"] < 5


def test_ros_move_turn_and_stop(ros):
    if ros.turtlesim:
        assert ros.call("robot_action", {"action": "call_service", "params": {
            "service": "/turtle1/teleport_absolute", "request": {"x": 5.5, "y": 5.5, "theta": 0.0}}})["ok"]
    r = ros.call("robot_move", {"linear_mps": 0.5 if ros.turtlesim else 0.1, "duration_s": 2})
    assert r["ok"] and r["end_reason"] == "completed", r
    assert r["displacement_m"] > (0.5 if ros.turtlesim else 0.05), r
    t = ros.call("robot_turn", {"degrees": 45})
    assert t["ok"] and t["end_reason"] == "completed", t
    assert ros.call("robot_stop")["ok"]


def test_ros_interlock_blocks_on_the_real_robot(ros):
    before = ros.call("robot_state")["state"]["pose"]
    res = ros.call("robot_move", {"linear_mps": 3.0, "duration_s": 10})
    assert res["blocked"] and "[over_speed_forward]" in res["message"]
    after = ros.call("robot_state")["state"]["pose"]
    assert math.hypot(after["x"] - before["x"], after["y"] - before["y"]) < 0.02


def test_ros_geofence_blocks_on_live_odom(ros):
    if not ros.turtlesim:
        pytest.skip("geofence frame only known for turtlesim")
    assert ros.call("robot_action", {"action": "call_service", "params": {
        "service": "/turtle1/teleport_absolute", "request": {"x": 9.5, "y": 5.5, "theta": 0.0}}})["ok"]
    import time
    time.sleep(0.5)
    res = ros.call("robot_move", {"linear_mps": 1.0, "duration_s": 3})
    assert res["blocked"] and "[geofence]" in res["message"]


def test_ros_camera_and_echo(ros):
    if "/camera/image_raw" not in ros.names:
        pytest.skip("no camera topic")
    look = ros.call("robot_look")
    assert look["ok"] and look["width"] and Path(look["path"]).stat().st_size > 1000
    echo = ros.call("robot_action", {"action": "echo", "params": {"topic": "/turtle1/pose" if ros.turtlesim else "/odom"}})
    assert echo["ok"] and echo["message"]


def test_ros_estop_aborts_and_latches(ros):
    ros.rt.interlock.engage_estop("live test", "human")
    assert "[estop_latched]" in ros.call("robot_move", {"linear_mps": 0.1})["message"]
    assert ros.call("robot_stop")["ok"]
    ros.rt.interlock.clear_estop("human")


# ------------------------------------------------------------------------------------------- Microduck
@pytest.fixture(scope="module")
def duck(request, tmp_path_factory):
    url = request.config.getoption("--live-microduck")
    if not url:
        pytest.skip("pass --live-microduck=http://host:port (sim/microduck/server.py) to run against the MuJoCo sim")
    h = Harness(tmp_path_factory.mktemp("duck"), {"robots": {"duck": {"backend": "microduck", "mode": "sim", "url": url}},
                                                  "camera": {"keep_last": 200}})
    h.rt.backend().reset()
    yield h
    h.rt.shutdown()


def test_duck_connects_to_the_real_policies(duck):
    con = duck.call("robot_connect", {"robot": "duck"})
    assert con["ok"] and con["connection"]["mode"] == "sim", con
    st = duck.call("robot_state")["state"]
    assert st["upright"] and "x" in st["pose"]


def test_duck_walks_with_the_policy(duck):
    r = duck.call("robot_move", {"linear_mps": 0.25, "duration_s": 4})
    assert r["ok"], r
    assert r["displacement_m"] > 0.1 and r["status"]["upright"], r


def test_duck_turns_and_looks(duck):
    t = duck.call("robot_turn", {"degrees": 45})
    assert t["ok"], t
    for view in ("duck_eye", "overhead"):
        look = duck.call("robot_look", {"view": view})
        assert look["ok"] and Path(look["path"]).stat().st_size > 1000


def test_duck_interlock_blocks_over_speed_before_the_sim(duck):
    before = duck.call("robot_state")["state"]["pose"]
    res = duck.call("robot_move", {"linear_mps": 1.0, "duration_s": 30})
    assert res["blocked"] and "[over_speed_forward]" in res["message"]
    after = duck.call("robot_state")["state"]["pose"]
    assert math.hypot(after["x"] - before["x"], after["y"] - before["y"]) < 0.05


def test_duck_kicks_the_ball(duck):
    duck.rt.backend().reset()
    r = duck.call("robot_action", {"action": "kick_ball"})
    assert r["ok"], r
    assert r["ball_travel_m"] > 0.1, json.dumps(r)[:500]
