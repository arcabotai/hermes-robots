# SPDX-License-Identifier: Apache-2.0  (port of Pollen Robotics microduck-simulator logic; see NOTICE)
"""Microduck MuJoCo sim server: real-time 50 Hz physics + ONNX policies behind a tiny local HTTP API.

The hermes-microduck plugin's `http` backend talks to this. Endpoints (JSON in/out, localhost only):
  GET  /status                         pose, posture, ball, goal
  GET  /frame?view=duck_eye|chase|overhead&w=&h=   PNG bytes
  POST /walk      {forward_mps, turn_rps, duration_s, fence:{x_min..}}   blocks until done
  POST /turn      {degrees, timeout_s}
  POST /kick      {foot}
  POST /kick_ball {foot, timeout_s}
  POST /sit, /stand, /stop, /head {neck_pitch, head_pitch, head_yaw, head_roll}
  POST /reset     {ball_x, ball_y}
  POST /log/start {path}  POST /log/stop     (records the trajectory for offline video rendering)
"""
import argparse, io, json, math, os, queue, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
from microduck_sim import MicroduckSim, Approach, wrap, CTRL_DT, ARENA_HALF, VEL_FWD, VEL_ANG  # noqa: E402

GOAL_X, GOAL_HALF_W = ARENA_HALF - 0.08, 0.3


class Behaviour:
    """A closed-loop task ticked inside the 50 Hz loop."""
    name = "idle"
    def tick(self, sim): return True  # return True when finished
    result = None


class TimedWalk(Behaviour):
    name = "walk"
    def __init__(self, vx, wz, duration, fence=None):
        self.vx, self.wz, self.n = vx, wz, int(round(duration / CTRL_DT)); self.k = 0; self.fence = fence
        self.end_reason = "completed"
    def tick(self, sim):
        if self.k == 0:
            sim.set_velocity(self.vx, 0, self.wz)
        self.k += 1
        if self.fence:
            x, y = sim.data.qpos[0], sim.data.qpos[1]
            f = self.fence
            if not (f["x_min"] <= x <= f["x_max"] and f["y_min"] <= y <= f["y_max"]):
                self.end_reason = "aborted: reached the geofence edge"
                sim.set_velocity(0, 0, 0); return True
        if self.k >= self.n:
            sim.set_velocity(0, 0, 0); return True
        return False


class Turn(Behaviour):
    """Turn by `degrees`. The walking policy barely rotates in place, so this arcs (forward 0.25 m/s + turn).
    Tracks the unwrapped rotation so 180+ degree turns go the requested way."""
    name = "turn"
    def __init__(self, sim, degrees, timeout_s=25.0):
        # the gait keeps settling ~10 deg back toward the start after the stop, so aim slightly past the target
        over = math.copysign(math.radians(10), degrees) if abs(degrees) >= 30 else 0.0
        self.goal = math.radians(degrees) + over
        self.turned = 0.0; self.prev = sim.yaw(); self.k = 0; self.max = int(timeout_s / CTRL_DT)
        self.settle = None; self.end_reason = "completed"
    def tick(self, sim):
        self.k += 1
        y = sim.yaw(); self.turned += wrap(y - self.prev); self.prev = y
        err = self.goal - self.turned
        if self.settle is not None:
            self.settle += 1
            return self.settle >= 40
        if abs(err) < math.radians(7) or self.k >= self.max or (err * self.goal < 0):
            if self.k >= self.max: self.end_reason = "timeout"
            sim.set_velocity(0, 0, 0); self.settle = 0; return False
        sim.set_velocity(VEL_FWD, 0, math.copysign(VEL_ANG, err) if abs(err) > 0.35 else max(-1, min(1, 2.5 * err)))
        return False


class KickBall(Behaviour):
    name = "kick_ball"
    def __init__(self, sim, foot="auto", timeout_s=45.0):
        self.ap = Approach(sim, foot=foot, stop_dist=0.09); self.k = 0; self.max = int(timeout_s / CTRL_DT)
        self.stage = "approach"; self.after = 0; self.end_reason = "completed"
        self.ball0 = None; self.goal = False; self.maxx = -9
    def tick(self, sim):
        self.k += 1
        if self.stage == "approach":
            if self.k >= self.max:
                sim.set_velocity(0, 0, 0); self.end_reason = "timeout: could not line up"; return True
            self.ap.tick()
            if self.ap.done:
                sim.set_velocity(0, 0, 0)
                self.ball0 = sim.status()["ball"]
                if not self.ap.result["lined_up"]:
                    self.end_reason = "could not line up a clean kick (ball not in front of a foot)"
                sim.kick(self.ap.foot); self.stage = "kick"
            return False
        if self.stage == "kick":
            self.after += 1
            bx, by = sim.data.qpos[sim.ball_q], sim.data.qpos[sim.ball_q + 1]
            if bx >= GOAL_X - 0.03 and abs(by) < GOAL_HALF_W - 0.01:
                self.goal = True
            return self.after >= 150
        return True


class Server:
    def __init__(self, ball_xy=(0.2, -0.05), realtime=1.0):
        self.sim = MicroduckSim(ball_xy=ball_xy)
        self.realtime = realtime
        self.behaviour = None
        self.done_evt = threading.Event()
        self.log = None
        self.log_lock = threading.Lock()
        self.render_q = queue.Queue()
        self.events = []
        self.t0 = time.time()
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._render_worker, daemon=True).start()

    # ------------------------------------------------------------- loop
    def _loop(self):
        nxt = time.perf_counter()
        while True:
            with self.sim.lock:
                b = self.behaviour
                if b is not None:
                    try:
                        if b.tick(self.sim):
                            self.behaviour = None; self.done_evt.set()
                    except Exception as exc:  # never kill the loop
                        b.end_reason = f"error: {exc}"; self.behaviour = None; self.sim.set_velocity(0, 0, 0); self.done_evt.set()
                self.sim.step()
                if self.log is not None:
                    self._log_frame()
            nxt += CTRL_DT / self.realtime
            dt = nxt - time.perf_counter()
            if dt > 0: time.sleep(dt)
            else: nxt = time.perf_counter()

    def _log_frame(self):
        s = self.sim
        st_label = s.mode if not s.recovery else "stand"
        self.log["t"].append(time.time())
        self.log["qpos"].append(s.data.qpos.copy())
        self.log["label"].append(f"{st_label}|{self.behaviour.name if self.behaviour else 'idle'}|{s.vel[0]:.2f}|{s.vel[2]:.2f}")

    def run_behaviour(self, b, timeout):
        with self.sim.lock:
            if self.behaviour is not None:
                self.behaviour.end_reason = "pre-empted"; self.done_evt.set()
            self.done_evt = threading.Event(); evt = self.done_evt
            self.behaviour = b
        evt.wait(timeout)
        return b

    def stop(self):
        with self.sim.lock:
            if self.behaviour is not None:
                self.behaviour.end_reason = "stopped"; self.behaviour = None; self.done_evt.set()
            self.sim.set_velocity(0, 0, 0)

    # ----------------------------------------------------------- render
    def _render_worker(self):
        renderers = {}
        data = mujoco.MjData(self.sim.model)
        while True:
            view, w, h, qpos, out = self.render_q.get()
            try:
                key = (w, h)
                if key not in renderers:
                    renderers[key] = mujoco.Renderer(self.sim.model, h, w)
                r = renderers[key]
                data.qpos[:] = qpos; mujoco.mj_forward(self.sim.model, data)
                cam = make_camera(view, data, self.sim)
                r.update_scene(data, cam)
                out.put(r.render().copy())
            except Exception as exc:
                out.put(exc)

    def frame(self, view="duck_eye", w=640, h=480):
        with self.sim.lock:
            qpos = self.sim.data.qpos.copy()
        out = queue.Queue()
        self.render_q.put((view, w, h, qpos, out))
        img = out.get(timeout=30)
        if isinstance(img, Exception): raise img
        from PIL import Image
        buf = io.BytesIO(); Image.fromarray(img).save(buf, "PNG"); return buf.getvalue()


def make_camera(view, data, sim):
    if view == "duck_eye":
        return "duck_eye"
    cam = mujoco.MjvCamera()
    x, y = data.qpos[0], data.qpos[1]
    if view == "overhead":
        cam.lookat[:] = [0, 0, 0]; cam.distance = 4.2; cam.azimuth = 180; cam.elevation = -89.9
        return cam
    q = data.qpos[3:7]; yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    cam.lookat[:] = [x + 0.15 * math.cos(yaw), y + 0.15 * math.sin(yaw), 0.08]
    cam.distance = 0.95; cam.elevation = -22; cam.azimuth = math.degrees(yaw) + 200
    return cam


def make_handler(srv: Server):
    sim = srv.sim

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/status":
                st = sim.status(); st["behaviour"] = srv.behaviour.name if srv.behaviour else "idle"
                st["events"] = sim.events[-5:]; return self._json(st)
            if u.path == "/frame":
                png = srv.frame(q.get("view", "duck_eye"), int(q.get("w", 640)), int(q.get("h", 480)))
                self.send_response(200); self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png))); self.end_headers(); self.wfile.write(png); return
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                a = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return self._json({"error": "bad json"}, 400)
            p = urlparse(self.path).path
            try:
                return self._json(self.route(p, a))
            except Exception as exc:
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def route(self, p, a):
            before = sim.status()
            if p == "/walk":
                vx = float(a.get("forward_mps", 0)); wz = float(a.get("turn_rps", 0)); d = float(a.get("duration_s", 1))
                b = srv.run_behaviour(TimedWalk(vx, wz, d, a.get("fence")), d + 5)
                time.sleep(0.6)
                return self.moved(before, end_reason=b.end_reason)
            if p == "/turn":
                b = srv.run_behaviour(Turn(sim, float(a.get("degrees", 0)), float(a.get("timeout_s", 25))), 40)
                return self.moved(before, end_reason=b.end_reason)
            if p == "/kick_ball":
                b = srv.run_behaviour(KickBall(sim, a.get("foot", "auto"), float(a.get("timeout_s", 45))), 70)
                out = self.moved(before, end_reason=b.end_reason)
                out.update({"foot": b.ap.foot, "lined_up": bool(b.ap.result and b.ap.result["lined_up"]),
                            "ball_before": before["ball"], "ball_after": out["status"]["ball"],
                            "ball_travel_m": round(math.hypot(out["status"]["ball"]["x"] - before["ball"]["x"],
                                                              out["status"]["ball"]["y"] - before["ball"]["y"]), 3),
                            "goal_scored": b.goal})
                return out
            if p == "/kick":
                ok = sim.kick(a.get("foot", "right")); time.sleep(1.5)
                out = self.moved(before, end_reason="completed" if ok else "refused: duck busy")
                out["ball_travel_m"] = round(math.hypot(out["status"]["ball"]["x"] - before["ball"]["x"], out["status"]["ball"]["y"] - before["ball"]["y"]), 3)
                return out
            if p == "/sit":
                srv.stop(); ok = sim.sit(); time.sleep(3.5); return self.moved(before, end_reason="completed" if ok else "refused")
            if p == "/stand":
                srv.stop(); ok = sim.stand(); time.sleep(3.0); return self.moved(before, end_reason="completed" if ok else "refused")
            if p == "/head":
                sim.set_head(*(float(a.get(k, 0)) for k in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")))
                time.sleep(float(a.get("hold_s", 1.0))); return {"ok": True, "status": sim.status()}
            if p == "/stop":
                srv.stop(); time.sleep(0.3); return {"ok": True, "status": sim.status()}
            if p == "/reset":
                srv.stop()
                with sim.lock:
                    if "ball_x" in a: sim.ball_home = (float(a["ball_x"]), float(a.get("ball_y", 0)))
                    sim.reset()
                return {"ok": True, "status": sim.status()}
            if p == "/log/start":
                with sim.lock: srv.log = {"t": [], "qpos": [], "label": [], "path": a.get("path", "/tmp/duck_log.npz")}
                return {"ok": True}
            if p == "/log/stop":
                with sim.lock: lg, srv.log = srv.log, None
                if lg:
                    np.savez_compressed(lg["path"], t=np.array(lg["t"]), qpos=np.array(lg["qpos"]), label=np.array(lg["label"]))
                    return {"ok": True, "frames": len(lg["t"]), "path": lg["path"]}
                return {"ok": False}
            return {"error": "not found"}

        def moved(self, before, end_reason):
            after = sim.status()
            b, a2 = before["pose"], after["pose"]
            return {"ok": True, "end_reason": end_reason, "pose_before": b, "pose_after": a2,
                    "displacement_m": round(math.hypot(a2["x"] - b["x"], a2["y"] - b["y"]), 3),
                    "turned_deg": round(math.degrees(wrap(math.radians(a2["yaw_deg"] - b["yaw_deg"]))), 1),
                    "status": after, "events": sim.events[-5:]}
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ball", default="0.2,-0.05"); ap.add_argument("--realtime", type=float, default=1.0)
    a = ap.parse_args()
    bx, by = (float(v) for v in a.ball.split(","))
    srv = Server((bx, by), a.realtime)
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(srv))
    print(f"microduck sim listening on http://{a.host}:{a.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
