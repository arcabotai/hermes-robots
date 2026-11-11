"""The ros backend over the wire, against an in-process fake rosbridge server (no ROS needed, runs in CI).

The fake speaks the real rosbridge v2 JSON protocol: rosapi services, subscribe/unsubscribe, advertise/publish.
tests/live covers a real ROS 2 rosbridge (TurtleBot3 / turtlesim).
"""
import json
import threading
import time

import pytest

pytest.importorskip("websocket")
sync_server = pytest.importorskip("websockets.sync.server")

from conftest import Harness  # noqa: E402

FENCE = {"enabled": True, "x_min": -1.0, "x_max": 3.0, "y_min": -2.0, "y_max": 2.0, "margin_m": 0.0}


class FakeRosbridge:
    def __init__(self):
        self.received_twists = []
        self.other_publishes = []
        self.pose = {"x": 0.0, "y": 0.0}
        self.unsub_count = 0
        self.log = []
        self.clients = []
        self._stop = threading.Event()
        self.server = sync_server.serve(self._handle, "127.0.0.1", 0)
        self.port = self.server.socket.getsockname()[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self._stop.set()
        threading.Thread(target=self.server.shutdown, daemon=True).start()  # may linger on a half-closed client

    def drop_clients(self):
        for ws in list(self.clients):  # the server-side close handshake can linger: do it off the test thread
            threading.Thread(target=lambda w=ws: w.close(), daemon=True).start()

    def _handle(self, ws):
        self.clients.append(ws)
        subs = {}
        lock = threading.Lock()

        def send(obj):
            with lock:
                try:
                    ws.send(json.dumps(obj))
                except Exception:
                    pass

        def feeder():
            while not self._stop.is_set():
                for topic in list(subs):
                    msg = {"pose": {"pose": {"position": {"x": self.pose["x"], "y": self.pose["y"], "z": 0.0},
                                            "orientation": {"x": 0, "y": 0, "z": 0, "w": 1.0}}},
                           "twist": {"twist": {"linear": {"x": 0.0}, "angular": {"z": 0.0}}},
                           "header": {"frame_id": "odom"}}
                    send({"op": "publish", "topic": topic, "msg": msg})
                time.sleep(0.05)

        threading.Thread(target=feeder, daemon=True).start()
        try:
            for raw in ws:
                m = json.loads(raw)
                self.log.append(m)
                op = m["op"]
                if op == "subscribe":
                    subs[m["topic"]] = m
                elif op == "unsubscribe":
                    self.unsub_count += 1
                    subs.pop(m["topic"], None)  # like rosbridge: drops *all* of this client's subs on the topic
                elif op == "publish" and m["topic"] == "/cmd_vel":
                    self.received_twists.append(m["msg"])
                elif op == "publish":
                    self.other_publishes.append(m)
                elif op == "call_service":
                    v, ok = self._service(m["service"], m.get("args", {}))
                    send({"op": "service_response", "service": m["service"], "id": m["id"], "values": v, "result": ok})
        except Exception:
            pass

    @staticmethod
    def _service(name, args):
        if name == "/rosapi/topics_and_raw_types":
            return {"topics": ["/cmd_vel", "/odom"], "types": ["geometry_msgs/Twist", "nav_msgs/Odometry"]}, True
        if name == "/rosapi/topic_type":
            return {"type": {"/odom": "nav_msgs/Odometry", "/cmd_vel": "geometry_msgs/Twist"}.get(args.get("topic"), "")}, True
        if name == "/rosapi/nodes":
            return {"nodes": ["/fake_base"]}, True
        if name == "/rosapi/services":
            return {"services": ["/reset"]}, True
        if name == "/rosapi/service_type":
            return {"type": "std_srvs/Empty"}, True
        if name == "/reset":
            return {}, True
        if name == "/explode":
            return "service blew up", False
        return "no such service", False


@pytest.fixture
def fake():
    f = FakeRosbridge()
    yield f
    f.close()


def overrides(port, topics=None, **safety):
    robot = {"backend": "ros", "host": "127.0.0.1", "port": port, "connect_timeout_s": 3.0, "call_timeout_s": 3.0,
             "safety": {"max_duration_s": 3.0, "write_allowlist": {"topics": ["/cmd_vel", "/chatter"],
                                                                     "services": ["/reset", "/explode"]}, **safety}}
    if topics:
        robot["topics"] = topics
    return {"robots": {"tb3": robot}}


@pytest.fixture
def bot(fake, tmp_path):
    h = Harness(tmp_path, overrides(fake.port))
    yield h
    h.rt.shutdown()


def moving(fake, x):
    return [t for t in fake.received_twists if (t.get("twist") or t)["linear"]["x"] == pytest.approx(x)]


def test_discovery_over_the_wire(bot):
    r = bot.call("robot_action", {"action": "topics"})
    assert r["ok"], r
    assert {"name": "/odom", "type": "nav_msgs/Odometry"} in r["topics"] and r["nodes"] == ["/fake_base"]
    assert bot.call("robot_action", {"action": "topics", "params": {"kind": "topics", "filter": "odom"}})["topics"] == \
        [{"name": "/odom", "type": "nav_msgs/Odometry"}]


def test_pose_comes_from_the_odom_feed(bot, fake):
    fake.pose.update(x=1.5, y=-0.5)
    st = bot.call("robot_state")["state"]
    assert st["connected"] and st["pose"]["x"] == pytest.approx(1.5) and st["pose"]["y"] == pytest.approx(-0.5)


def test_move_publishes_twists_then_zero_twists(bot, fake):
    r = bot.call("robot_move", {"linear_mps": 0.2, "duration_s": 0.5})
    assert r["ok"], r
    assert 3 <= len(moving(fake, 0.2)) <= 8  # ~10 Hz for 0.5 s
    assert fake.received_twists[-1]["linear"]["x"] == 0.0 and fake.received_twists[-1]["angular"]["z"] == 0.0
    adverts = [m for m in fake.log if m["op"] == "advertise"]
    assert len(adverts) == 1 and adverts[0]["type"] == "geometry_msgs/Twist"


def test_twist_stamped_robots_get_stamped_messages(fake, tmp_path):
    bot = Harness(tmp_path, overrides(fake.port, topics={"cmd_vel_type": "geometry_msgs/msg/TwistStamped"}))
    try:
        assert bot.call("robot_move", {"linear_mps": 0.1, "duration_s": 0.3})["ok"]
        assert moving(fake, 0.1) and all("twist" in t for t in fake.received_twists)
        assert fake.received_twists[-1]["twist"]["linear"]["x"] == 0.0
    finally:
        bot.rt.shutdown()


def test_blocked_move_never_reaches_the_wire(bot, fake):
    assert bot.call("robot_move", {"linear_mps": 2.0, "duration_s": 1})["blocked"]
    assert not [t for t in fake.received_twists if t["linear"]["x"]]


def test_turn_is_a_timed_rotation_then_zero(bot, fake):
    r = bot.call("robot_turn", {"degrees": -30})
    assert r["ok"], r
    spins = [t for t in fake.received_twists if t["angular"]["z"] < 0]
    assert spins and all(t["angular"]["z"] == pytest.approx(-0.5) for t in spins)
    assert fake.received_twists[-1]["angular"]["z"] == 0.0


def test_estop_mid_move_aborts_and_zeroes(bot, fake):
    t = threading.Timer(0.4, lambda: bot.rt.interlock.engage_estop("human hit the button", "human"))
    t.start()
    r = bot.call("robot_move", {"linear_mps": 0.2, "duration_s": 3})
    assert r["ok"] and r["end_reason"] == "aborted: e-stop latched" and r["ran_for_s"] < 1.5
    assert fake.received_twists[-1]["linear"]["x"] == 0.0


def test_one_shot_echo_of_odom_does_not_kill_the_persistent_feed(fake, tmp_path):
    """Regression: rosbridge drops all of a client's subscriptions on a topic when one unsubscribes."""
    bot = Harness(tmp_path, overrides(fake.port, geofence=FENCE))
    try:
        assert bot.call("robot_move", {"linear_mps": 0.1, "duration_s": 0.2})["ok"]  # starts the feed
        for _ in range(3):
            assert bot.call("robot_state")["ok"]
            assert bot.call("robot_action", {"action": "echo", "params": {"topic": "/odom"}})["ok"]
        assert fake.unsub_count == 0
        assert bot.call("robot_move", {"linear_mps": 0.1, "duration_s": 0.2})["ok"]  # geofence still sees fresh odom
    finally:
        bot.rt.shutdown()


def test_geofence_uses_live_odom_from_the_wire(fake, tmp_path):
    bot = Harness(tmp_path, overrides(fake.port, geofence=FENCE))
    try:
        fake.pose.update(x=FENCE["x_max"] - 0.1, y=0.0)
        r = bot.call("robot_move", {"linear_mps": 0.25, "duration_s": 2})
        assert r["blocked"] and "[geofence]" in r["message"]
    finally:
        bot.rt.shutdown()


def test_service_calls_allowlist_and_failures(bot):
    assert bot.call("robot_action", {"action": "call_service", "params": {"service": "/reset"}})["ok"]
    failed = bot.call("robot_action", {"action": "call_service", "params": {"service": "/explode"}})
    assert not failed["blocked"] and not failed["ok"] and "blew up" in json.dumps(failed)
    res = bot.call("robot_action", {"action": "call_service", "params": {"service": "/rosapi/nodes"}})
    assert res["blocked"] and "[service_not_allowlisted]" in res["message"]


def test_publish_rules(bot, fake):
    assert "[topic_not_allowlisted]" in bot.call("robot_action", {"action": "publish", "params": {
        "topic": "/arm/joint_cmd", "msg_type": "std_msgs/Float64", "message": {"data": 1.0}}})["message"]
    assert bot.call("robot_action", {"action": "publish", "params": {
        "topic": "/chatter", "msg_type": "std_msgs/String", "message": {"data": "hi"}}})["published"] == 1
    deadline = time.time() + 2
    while not fake.other_publishes and time.time() < deadline:
        time.sleep(0.02)
    assert fake.other_publishes[-1]["msg"] == {"data": "hi"}
    fast = {"topic": "/cmd_vel", "msg_type": "geometry_msgs/Twist", "message": {"linear": {"x": 3.0}}, "duration_s": 1}
    assert "[over_speed_forward]" in bot.call("robot_action", {"action": "publish", "params": fast})["message"]
    side = {"topic": "/cmd_vel", "msg_type": "geometry_msgs/Twist", "message": {"linear": {"y": 0.1}}}
    assert "[unsupported_twist]" in bot.call("robot_action", {"action": "publish", "params": side})["message"]
    junk = {"topic": "/cmd_vel", "msg_type": "geometry_msgs/Twist", "message": {"data": 1}}
    assert "[unparseable_motion_message]" in bot.call("robot_action", {"action": "publish", "params": junk})["message"]
    ok = {"topic": "/cmd_vel", "msg_type": "geometry_msgs/Twist", "message": {"linear": {"x": 0.1}}, "duration_s": 0.3}
    r = bot.call("robot_action", {"action": "publish", "params": ok})
    assert r["ok"] and r["published"] >= 2 and fake.received_twists[-1]["linear"]["x"] == 0.0


def test_stop_sends_zero_velocity(bot, fake):
    r = bot.call("robot_stop")
    assert r["ok"] and r["stopped"] == {"tb3": "stopped"}
    assert len(fake.received_twists) >= 3 and all(t["linear"]["x"] == 0.0 for t in fake.received_twists)


def test_unreachable_rosbridge_is_a_clean_error(tmp_path):
    bot = Harness(tmp_path, overrides(1))  # nothing listens on port 1
    try:
        r = bot.call("robot_connect", {"robot": "tb3"})
        assert not r["ok"] and "rosbridge" in r["error"].lower()
        stop = bot.call("robot_stop")
        assert stop["ok"] is False and "stop failed" in stop["stopped"]["tb3"]
    finally:
        bot.rt.shutdown()


def test_server_going_away_fails_calls_fast_and_reconnects(fake, tmp_path):
    bot = Harness(tmp_path, overrides(fake.port))
    try:
        assert bot.call("robot_action", {"action": "topics"})["ok"]
        fake.drop_clients()
        time.sleep(0.3)
        assert bot.call("robot_action", {"action": "topics"})["ok"]
    finally:
        bot.rt.shutdown()
