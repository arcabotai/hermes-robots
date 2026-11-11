"""Adding a robot kind is one subclass + one register_backend call: the interlock, tools and /robot work unchanged."""
import pytest

from hermes_robots.backends import (MOTION, Action, Blocked, RobotBackend, available, register_backend,
                                    unregister_backend)
from hermes_robots.backends import POSE, num


class ToyArm(RobotBackend):
    """A stand-in for e.g. Reachy Mini: no base, one extra motion action with its own check."""
    kind = "toyarm"
    capabilities = frozenset({POSE})
    default_safety = {"max_wave_deg": 30}

    def __init__(self, name, cfg, data_dir):
        super().__init__(name, cfg, data_dir)
        self.waves = []

    def state(self):
        return {"pose": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}, "waves": len(self.waves)}

    def stop(self):
        return {"stopped": True}

    def actions(self):
        return {"wave": Action("wave", "Wave by params.deg degrees.", MOTION, {"deg": {"type": "number"}})}

    def check_action(self, action, params, guard):
        deg = num(params.get("deg"), "deg")
        if abs(deg) > guard.limits["max_wave_deg"]:
            raise Blocked("over_wave", f"wave {deg} exceeds {guard.limits['max_wave_deg']}")
        return {"deg": deg}

    def run_action(self, action, params, ctx):
        self.waves.append(params["deg"])
        return {"waved": params["deg"]}


@pytest.fixture
def arm(make):
    register_backend("toyarm", ToyArm)
    yield make({"robots": {"arm": {"backend": "toyarm"}}})
    unregister_backend("toyarm")


def test_registered_kind_is_listed_and_usable(arm):
    assert arm.rt.backend().waves == []
    assert "toyarm" in available()
    lst = arm.call("robot_list")
    assert lst["robots"][0]["kind"] == "toyarm" and lst["robots"][0]["actions"] == ["wave"]
    assert arm.call("robot_action", {"action": "wave", "params": {"deg": 20}})["waved"] == 20


def test_registered_kind_gets_the_shared_interlock(arm):
    assert "[over_wave]" in arm.call("robot_action", {"action": "wave", "params": {"deg": 90}})["message"]
    assert "[unsupported]" in arm.call("robot_move", {"linear_mps": 0.1})["message"]
    arm.call("robot_stop", {"estop": True})
    assert "[estop_latched]" in arm.call("robot_action", {"action": "wave", "params": {"deg": 5}})["message"]
    assert arm.rt.backend().waves == []  # no refused call reached the robot
    assert arm.raw("robot_action", {"action": "wave", "params": {"deg": 5}})["ok"] is False


def test_only_robot_backend_subclasses_register():
    with pytest.raises(TypeError):
        register_backend("bad", object)


def test_unknown_backend_kind_is_a_clean_error(make):
    h = make({"robots": {"x": {"backend": "warpdrive"}}})
    res = h.call("robot_state")
    assert not res["ok"] and "unknown robot backend 'warpdrive'" in res["error"]
    assert "[unknown_robot]" in h.call("robot_move", {"linear_mps": 0.1})["message"]
