"""Behaviour contract of the shared interlock, run against every built-in simulated robot."""
import json

import pytest

ROBOTS = ["mock", "duck"]  # 2D mock base, kinematic Microduck


@pytest.fixture(params=ROBOTS)
def r(request, make):
    h = make({"active": request.param})
    h.name = request.param
    return h


def test_safe_move_is_allowed_and_moves(r):
    res = r.call("robot_move", {"linear_mps": 0.25, "duration_s": 2})
    assert res["ok"], res
    assert res["robot"] == r.name and res["pose_after"]["x"] > res["pose_before"]["x"]


@pytest.mark.parametrize("args,rule", [
    ({"linear_mps": 2.0, "duration_s": 1}, "over_speed_forward"),
    ({"linear_mps": -2.0, "duration_s": 1}, "over_speed_reverse"),
    ({"linear_mps": 0.1, "angular_rps": 5, "duration_s": 1}, "over_speed_angular"),
    ({"linear_mps": 0.1, "duration_s": 60}, "over_duration"),
    ({"linear_mps": 0.1, "duration_s": -1}, "invalid_args"),
    ({"linear_mps": "fast"}, "invalid_args"),
    ({"linear_mps": float("nan")}, "invalid_args"),
])
def test_limits_block_and_nothing_moves(r, args, rule):
    before = r.call("robot_state")["state"]["pose"]
    res = r.call("robot_move", args)
    assert res["blocked"] and f"[{rule}]" in res["message"]
    assert "Do not retry" in res["message"]
    assert r.call("robot_state")["state"]["pose"] == before
    assert r.rt.interlock.recent_blocked(1)[0]["rule"] == rule


def test_missing_duration_gets_the_default(r):
    res = r.call("robot_move", {"linear_mps": 0.2})
    lim = r.call("robot_state")["limits"]
    assert res["ok"] and res["command"]["duration_s"] == lim["default_duration_s"]
    assert any("defaulted" in n for n in res["interlock"])


def test_geofence_blocks_a_projected_exit(r):
    lim = r.call("robot_state")["limits"]["geofence"]
    assert lim["enabled"]
    # face the +x edge from close to it: any forward move long enough leaves the fence
    r.rt.backend().reset(**({"x": lim["x_max"] - lim["margin_m"] - 0.1} if r.name == "mock" else {}))
    if r.name == "duck":
        r.rt.backend().link.x = lim["x_max"] - lim["margin_m"] - 0.1
    res = r.call("robot_move", {"linear_mps": 0.25, "duration_s": 3})
    assert res["blocked"] and "[geofence]" in res["message"]


def test_estop_latches_everything_until_a_human_clears_it(r):
    assert r.call("robot_stop", {"estop": True, "reason": "test"})["estop"]["latched"]
    for tool, args in [("robot_move", {"linear_mps": 0.1}), ("robot_turn", {"degrees": 30})]:
        res = r.call(tool, args)
        assert res["blocked"] and "[estop_latched]" in res["message"]
    # stopping and reading still work
    assert r.call("robot_stop")["ok"] and r.call("robot_state")["ok"]
    assert r.call("robot_move", {"linear_mps": 0, "angular_rps": 0})["ok"]  # zero velocity == stop
    # the agent has no tool that clears it: no tool argument resets the latch
    for tool in ("robot_stop", "robot_connect", "robot_state", "robot_list"):
        r.call(tool, {"estop": False, "clear": True, "robot": r.name})
    assert r.rt.interlock.latched
    assert r.rt.interlock.clear_estop("human")
    assert r.call("robot_move", {"linear_mps": 0.1, "duration_s": 0.5})["ok"]


def test_estop_survives_a_restart(r, tmp_path):
    r.call("robot_stop", {"estop": True, "reason": "persist"})
    from conftest import Harness
    again = Harness(tmp_path, {"active": r.name})
    res = again.call("robot_move", {"linear_mps": 0.1})
    assert res["blocked"] and "persist" in res["message"]


def test_corrupt_latch_file_fails_closed(make, tmp_path):
    (tmp_path / "estop.json").write_text("{not json")
    res = make().call("robot_move", {"linear_mps": 0.1})
    assert res["blocked"] and "[estop_latched]" in res["message"]


def test_handlers_refuse_without_the_hook(r):
    for tool, args in [("robot_move", {"linear_mps": 0.1}), ("robot_turn", {"degrees": 10}),
                       ("robot_action", {"action": "teleport" if r.name == "mock" else "sit"})]:
        res = r.raw(tool, args)
        assert not res["ok"] and "interlock_not_applied" in res["error"]


def test_forged_or_replayed_tokens_are_refused(r):
    forged = r.raw("robot_move", {"linear_mps": 3.0, "_interlock": {"nonce": "guess", "token": "x"}})
    assert "interlock_not_applied" in forged["error"]
    d = r.rt.interlock.pre_tool_call(tool_name="robot_move", args={"linear_mps": 0.1, "duration_s": 0.2})
    args = {"linear_mps": 0.1, "duration_s": 0.2, **d["args"]}
    assert json.loads(r.handlers["robot_move"](dict(args)))["ok"]
    replay = json.loads(r.handlers["robot_move"](dict(args)))
    assert not replay["ok"] and "interlock_token_invalid" in replay["error"]


def test_handler_runs_the_vetted_args_not_later_edits(r):
    """Another hook (or anything else) editing args after the interlock cannot raise the speed."""
    d = r.rt.interlock.pre_tool_call(tool_name="robot_move", args={"linear_mps": 0.2, "duration_s": 1})
    res = json.loads(r.handlers["robot_move"]({"linear_mps": 9.0, "duration_s": 99, **d["args"]}))
    assert res["ok"] and res["command"] == {"linear_mps": 0.2, "angular_rps": 0.0, "duration_s": 1.0}


def test_estop_voids_calls_approved_before_it(r):
    d = r.rt.interlock.pre_tool_call(tool_name="robot_move", args={"linear_mps": 0.2, "duration_s": 1})
    r.rt.interlock.engage_estop("between hook and handler", "human")
    res = json.loads(r.handlers["robot_move"]({"linear_mps": 0.2, **d["args"]}))
    assert not res["ok"] and "BLOCKED" in res["error"]


def test_interlock_errors_fail_closed(r, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(r.rt.interlock, "evaluate", boom)
    res = r.call("robot_move", {"linear_mps": 0.1})
    assert res["blocked"] and "[interlock_error]" in res["message"]


def test_unreadable_pose_blocks_when_the_geofence_is_on(r, monkeypatch):
    be = r.rt.backend()

    def broken():
        raise RuntimeError("odom gone")
    monkeypatch.setattr(be, "state", broken)
    res = r.call("robot_move", {"linear_mps": 0.1, "duration_s": 1})
    assert res["blocked"] and "[no_pose]" in res["message"]


def test_stale_pose_blocks(make, monkeypatch):
    h = make()
    be = h.rt.backend()
    orig = be.state
    monkeypatch.setattr(be, "state", lambda: {**orig(), "pose": {**orig()["pose"], "age_s": 30.0}})
    res = h.call("robot_move", {"linear_mps": 0.1, "duration_s": 1})
    assert res["blocked"] and "[pose_stale]" in res["message"]


def test_clamp_mode_adjusts_instead_of_blocking(make):
    h = make({"safety": {"over_limit_action": "clamp"}})
    res = h.call("robot_move", {"linear_mps": 3.0, "angular_rps": -4, "duration_s": 60})
    assert res["ok"]
    assert res["command"] == {"linear_mps": 0.5, "angular_rps": -1.0, "duration_s": 5.0}
    assert len([n for n in res["interlock"] if "clamped" in n]) == 3


def test_hard_limits_cannot_be_raised_by_config(make):
    h = make({"active": "duck", "safety": {"max_linear_mps": 5.0},
              "robots": {"duck": {"backend": "microduck", "mode": "mock", "safety": {"max_linear_mps": 9.0}}}})
    assert h.call("robot_state")["limits"]["max_linear_mps"] == 0.25
    assert h.call("robot_move", {"linear_mps": 0.5, "duration_s": 1})["blocked"]


def test_per_robot_limits_win_over_global(make):
    h = make({"safety": {"max_linear_mps": 0.4},
              "robots": {"a": {"backend": "mock", "safety": {"max_linear_mps": 0.1}}, "b": {"backend": "mock"}}})
    assert h.call("robot_move", {"robot": "a", "linear_mps": 0.2, "duration_s": 1})["blocked"]
    assert h.call("robot_move", {"robot": "b", "linear_mps": 0.2, "duration_s": 1})["ok"]


def test_the_vetted_robot_is_pinned(make):
    """Switching the active robot between hook and handler must not redirect an approved command."""
    h = make()
    d = h.rt.interlock.pre_tool_call(tool_name="robot_move", args={"linear_mps": 0.2, "duration_s": 1})
    h.rt.select("duck")
    res = json.loads(h.handlers["robot_move"]({"linear_mps": 0.2, **d["args"]}))
    assert res["ok"] and res["robot"] == "mock"


def test_unknown_robot_and_action_are_blocked(bot):
    assert "[unknown_robot]" in bot.call("robot_move", {"robot": "ghost", "linear_mps": 0.1})["message"]
    assert "[unknown_action]" in bot.call("robot_action", {"action": "fly"})["message"]


def test_turn_emulated_by_a_timed_rotation_respects_the_duration_cap(make):
    h = make({"safety": {"turn_rate_rps": 0.2}})
    res = h.call("robot_turn", {"degrees": 180})  # pi / 0.2 = 15.7 s > 5 s
    assert res["blocked"] and "[over_duration]" in res["message"]
    ok = h.call("robot_turn", {"degrees": 45})
    assert ok["ok"] and ok["pose_after"]["yaw_deg"] == pytest.approx(45, abs=1)


def test_blocked_calls_are_audited_to_disk(bot, tmp_path):
    bot.call("robot_move", {"linear_mps": 9})
    rows = [json.loads(line) for line in (tmp_path / "blocked.jsonl").read_text().splitlines()]
    assert rows[-1]["rule"] == "over_speed_forward" and rows[-1]["tool"] == "robot_move"
