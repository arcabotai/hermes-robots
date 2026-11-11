"""microduck backend: its extra actions through the shared interlock, and the sim HTTP client against a stub server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def duck(make):
    return make({"active": "duck"})


def test_kick_ball_moves_the_ball_toward_the_goal(duck):
    res = duck.call("robot_action", {"action": "kick_ball"})
    assert res["ok"] and isinstance(res["goal_scored"], bool) and res["ball_travel_m"] > 0.5
    assert res["ball_after"]["x"] > res["ball_before"]["x"]


def test_sitting_duck_refuses_to_walk_until_it_stands(duck):
    assert duck.call("robot_action", {"action": "sit"})["ok"]
    res = duck.call("robot_move", {"linear_mps": 0.25, "duration_s": 1})
    assert res["ok"] and res["end_reason"].startswith("refused") and res["displacement_m"] == 0
    assert duck.call("robot_action", {"action": "stand"})["ok"]
    assert duck.call("robot_move", {"linear_mps": 0.25, "duration_s": 2})["displacement_m"] > 0


@pytest.mark.parametrize("params,rule", [
    ({"yaw": 1.2}, "over_head_limit"),
    ({"pitch": -0.9}, "over_head_limit"),
    ({"yaw": 0.1, "hold_s": 30}, "invalid_args"),
])
def test_head_limits(duck, params, rule):
    res = duck.call("robot_action", {"action": "head", "params": params})
    assert res["blocked"] and f"[{rule}]" in res["message"]


def test_kick_needs_a_foot(duck):
    assert "[invalid_args]" in duck.call("robot_action", {"action": "kick", "params": {"foot": "both"}})["message"]
    assert duck.call("robot_action", {"action": "kick", "params": {"foot": "left"}})["ok"]


def test_kick_ball_refuses_a_ball_outside_the_geofence(duck):
    duck.rt.backend().reset(ball_x=0.0, ball_y=1.45)
    res = duck.call("robot_action", {"action": "kick_ball"})
    assert res["blocked"] and "[geofence]" in res["message"]


def test_native_turn_and_three_views(duck):
    res = duck.call("robot_turn", {"degrees": 90})
    assert res["ok"] and res["turned_deg"] == pytest.approx(90, abs=1)
    for view in ("duck_eye", "chase", "overhead"):
        assert duck.call("robot_look", {"view": view})["mime"] == "image/png"


def test_actions_refused_while_estopped(duck):
    duck.call("robot_stop", {"estop": True})
    for action in ("kick_ball", "sit", "stand"):
        assert "[estop_latched]" in duck.call("robot_action", {"action": action})["message"]


# ------------------------------------------------------------------ sim mode (HTTP client) against a stub server
class StubSim:
    def __init__(self):
        self.posts = []
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj, ctype="application/json"):
                body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/status"):
                    return self._send(200, {"pose": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}, "policy": "walk.onnx",
                                            "ball": {"x": 0.2, "y": 0.0}, "posture": "standing"})
                if self.path.startswith("/frame"):
                    from hermes_robots import imaging
                    return self._send(200, imaging.encode_png(2, 2, bytes(12)), "image/png")
                self._send(404, {"error": "nope"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                stub.posts.append((self.path, body))
                if self.path == "/kick":
                    return self._send(500, {"error": "policy crashed"})
                self._send(200, {"ok": True, "end_reason": "completed", "status": {}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def sim():
    s = StubSim()
    yield s
    s.close()


def test_sim_mode_sends_the_vetted_walk_with_the_fence(make, sim):
    h = make({"robots": {"duck": {"backend": "microduck", "mode": "sim", "url": sim.url, "timeout_s": 5}}})
    assert h.call("robot_connect", {"robot": "duck"})["connection"]["policy"] == "walk.onnx"
    assert h.call("robot_move", {"linear_mps": 0.25, "duration_s": 2})["ok"]
    path, body = sim.posts[-1]
    assert path == "/walk" and body["forward_mps"] == 0.25 and body["duration_s"] == 2.0
    assert body["fence"] == {"x_min": -1.3, "x_max": 1.3, "y_min": -1.3, "y_max": 1.3}
    assert h.call("robot_move", {"linear_mps": 0.5, "duration_s": 1})["blocked"]  # never reaches the server
    assert [p for p, _ in sim.posts].count("/walk") == 1
    assert h.call("robot_look", {})["mime"] == "image/png"


def test_sim_errors_and_an_absent_server_are_clean_tool_errors(make, sim):
    h = make({"robots": {"duck": {"backend": "microduck", "mode": "sim", "url": sim.url, "timeout_s": 5}}})
    res = h.call("robot_action", {"action": "kick", "params": {"foot": "left"}})
    assert not res["ok"] and "policy crashed" in res["error"]
    gone = make({"robots": {"duck": {"backend": "microduck", "mode": "sim", "url": "http://127.0.0.1:1", "timeout_s": 2}}})
    res = gone.call("robot_connect", {"robot": "duck"})
    assert not res["ok"] and "cannot reach the Microduck sim" in res["error"]
    # with the geofence on and no pose, motion is refused rather than attempted blind
    assert "[no_pose]" in gone.call("robot_move", {"linear_mps": 0.25, "duration_s": 1})["message"]
