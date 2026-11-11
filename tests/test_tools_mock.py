"""Generic tools against the dependency-free mock base."""
import pytest

from helpers import red_pixels


def test_list_connect_state(bot):
    lst = bot.call("robot_list")
    assert [r["robot"] for r in lst["robots"]] == ["mock", "duck"] and lst["active"] == "mock"
    con = bot.call("robot_connect", {"robot": "duck"})
    assert con["active"] == "duck" and "kick_ball" in con["actions"]
    assert bot.call("robot_state")["robot"] == "duck"
    assert not bot.call("robot_connect", {"robot": "nobody"})["ok"]


def test_drive_to_the_cone_and_see_it_grow(bot):
    first = bot.call("robot_look")
    assert first["mime"] == "image/png" and first["media_tag"].startswith("MEDIA:")
    near = red_pixels(open(first["path"], "rb").read())
    for _ in range(2):
        assert bot.call("robot_move", {"linear_mps": 0.5, "duration_s": 2})["ok"]
    after = bot.call("robot_look")
    assert red_pixels(open(after["path"], "rb").read()) > near
    assert bot.call("robot_look", {"view": "overhead"})["view"] == "overhead"


def test_collision_is_reported_and_stops_the_robot(make):
    h = make({"robots": {"m": {"backend": "mock", "cones": [{"x": 1.0, "y": 0.0}]}}})
    res = h.call("robot_move", {"linear_mps": 0.5, "duration_s": 4})
    assert res["ok"] and res["collided"] == "cone" and res["pose_after"]["x"] < 0.62


def test_read_and_write_actions(bot):
    assert bot.call("robot_action", {"action": "cones"})["cones"] == [{"x": 3.5, "y": 0.8}]
    assert bot.call("robot_action", {"action": "teleport", "params": {"x": 1, "y": 1}})["pose"]["x"] == 1.0
    assert "[geofence]" in bot.call("robot_action", {"action": "teleport", "params": {"x": 9, "y": 0}})["message"]
    bot.call("robot_stop", {"estop": True})
    assert bot.call("robot_action", {"action": "cones"})["ok"]  # reads still work under e-stop
    assert "[estop_latched]" in bot.call("robot_action", {"action": "teleport", "params": {"x": 0, "y": 0}})["message"]


def test_estop_stops_every_loaded_robot(bot):
    bot.call("robot_state", {"robot": "mock"})
    bot.call("robot_state", {"robot": "duck"})
    res = bot.call("robot_stop", {"estop": True, "reason": "person in the arena"})
    assert set(res["stopped"]) == {"mock", "duck"} and res["estop"]["source"] == "agent"


def test_inline_camera_mode_returns_an_image_part(make):
    h = make({"camera": {"return_mode": "inline"}})
    env = h.handlers["robot_look"]({})
    assert env["_multimodal"] and env["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_snapshots_are_pruned(make):
    h = make({"camera": {"keep_last": 2}})
    for _ in range(4):
        h.call("robot_look")
    assert len(list(h.rt.snapshot_dir().iterdir())) == 2
