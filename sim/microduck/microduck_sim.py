# SPDX-License-Identifier: Apache-2.0  (port of Pollen Robotics microduck-simulator logic; see NOTICE)
"""Headless MuJoCo + onnxruntime port of the Microduck playground control loop.

Replicates pollen-robotics/microduck-simulator (app/src/game/game.js, Apache-2.0):
61-D obs [gyro(3), projected_gravity(3), q-q0(14), qd(14), last_action(14), cmd(13)],
14 position targets = DEFAULT_POSE + action, 50 Hz control (dt 0.005, decimation 4).
"""
import math, os, re, threading, time
import numpy as np
import mujoco
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.environ.get("MICRODUCK_ASSETS", os.path.join(HERE, "assets"))
JOINT_NAMES = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
               "neck_pitch", "head_pitch", "head_yaw", "head_roll",
               "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
DEFAULT_POSE = np.array([0, -0.08726646259971647, -0.457924, -0.004940, 0.452984,
                         0.3490658503988659, 0.3490658503988659, 0, 0,
                         0, 0.08726646259971647, 0.457924, 0.004940, -0.452984], dtype=np.float32)
TIMESTEP, DECIMATION = 0.005, 4
CTRL_DT = TIMESTEP * DECIMATION
VEL_FWD, VEL_BACK, VEL_ANG = 0.25, -0.2, 1.0
ARENA_HALF = 1.5
BALL_R = 0.05
SPAWN = (-0.6, 0.0)
KICK_STEPS, POST_KICK_LOCK = 25, 20
POLICIES = {"walk": "BEST_alpha_walking.onnx", "sitstand": "BEST_alpha_sitstand.onnx",
            "roll": "roulade.onnx", "kickL": "ball_kick_left.onnx", "kickR": "ball_kick_right.onnx",
            "stand": "BEST_alpha_stand.onnx", "groundpick": "alpha_ground_pick.onnx"}


def build_xml(ball_xy=(0.0, 0.0), goal=True):
    src = open(os.path.join(ASSETS, "robot", "robot_allcollisions.xml")).read()
    src = src.replace('meshdir="assets"', f'meshdir="{os.path.join(ASSETS, "robot", "meshes")}"')
    # Make visual meshes visible in the default render groups (group 2 is hidden by Renderer default? keep 2).
    extra_asset = """
    <texture name="grid" type="2d" builtin="checker" rgb1="0.20 0.24 0.30" rgb2="0.16 0.19 0.24" width="512" height="512" mark="edge" markrgb="0.35 0.42 0.52"/>
    <material name="floor_mat" texture="grid" texrepeat="10 10" reflectance="0.0"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.62 0.76 0.92" rgb2="0.30 0.36 0.46" width="512" height="512"/>
    <material name="wall_mat" rgba="0.95 0.75 0.10 1"/>
    <material name="ball_mat" rgba="0.95 0.25 0.20 1" specular="0.6" shininess="0.6"/>
    <material name="goal_mat" rgba="0.95 0.95 0.95 1"/>
    """
    src = src.replace("</asset>", extra_asset + "</asset>", 1)
    m = re.search(r'<site group="3" name="head_camera" pos="([^"]+)" quat="([^"]+)"/>', src)
    if m:
        src = src.replace(m.group(0), m.group(0) + f'\n<camera name="duck_eye" pos="{m.group(1)}" xyaxes="0 -1 0 1 0 0" fovy="80"/>', 1)
    o = ARENA_HALF + 0.025
    s = ARENA_HALF + 0.05
    world = f"""
    <light name="sun" pos="0 0 3" dir="0 0 -1" directional="true" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2" castshadow="true"/>
    <light name="fill" pos="-1.5 -1.5 2" dir="1 1 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor_mat"/>
    <geom name="wall_px" type="box" pos="{o} 0 0.125" size="0.025 {s} 0.125" material="wall_mat"/>
    <geom name="wall_nx" type="box" pos="{-o} 0 0.125" size="0.025 {s} 0.125" material="wall_mat"/>
    <geom name="wall_py" type="box" pos="0 {o} 0.125" size="{s} 0.025 0.125" material="wall_mat"/>
    <geom name="wall_ny" type="box" pos="0 {-o} 0.125" size="{s} 0.025 0.125" material="wall_mat"/>
    """
    if goal:
        gx = ARENA_HALF - 0.08
        world += f"""
    <geom name="goal_l" type="cylinder" pos="{gx} 0.3 0.15" size="0.015 0.15" material="goal_mat" contype="0" conaffinity="0"/>
    <geom name="goal_r" type="cylinder" pos="{gx} -0.3 0.15" size="0.015 0.15" material="goal_mat" contype="0" conaffinity="0"/>
    <geom name="goal_bar" type="cylinder" pos="{gx} 0 0.3" euler="1.5708 0 0" size="0.015 0.315" material="goal_mat" contype="0" conaffinity="0"/>
    """
    world += f"""
    <body name="ball" pos="{ball_xy[0]} {ball_xy[1]} {BALL_R}">
      <freejoint name="ball_freejoint"/>
      <geom name="ball_geom" type="sphere" size="{BALL_R}" mass="0.03" friction="0.4 0.01 0.003" solref="0.03 0.4" condim="6" material="ball_mat"/>
    </body>
    """
    # append to the end of worldbody (after robot so freejoint stays first)
    idx = src.rfind("</worldbody>")
    src = src[:idx] + world + src[idx:]
    src = src.replace("<mujoco model=\"microduck\">",
                      f"<mujoco model=\"microduck\">\n  <option timestep=\"{TIMESTEP}\"/>\n"
                      "  <visual><global offwidth=\"1920\" offheight=\"1080\"/><quality shadowsize=\"4096\"/>"
                      "<headlight ambient=\"0.35 0.35 0.35\" diffuse=\"0.4 0.4 0.4\"/></visual>", 1)
    return src


class MicroduckSim:
    def __init__(self, ball_xy=(0.0, 0.0)):
        self.ball_home = ball_xy
        self.model = mujoco.MjModel.from_xml_string(build_xml(ball_xy))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = [m.joint(n).qposadr[0] for n in JOINT_NAMES]
        self.dadr = [m.joint(n).dofadr[0] for n in JOINT_NAMES]
        self.act_ids = [m.actuator(n).id for n in JOINT_NAMES]
        self.gyro_adr = m.sensor("imu_ang_vel").adr[0]
        self.trunk = m.body("trunk_base").id
        self.ball_q = m.joint("ball_freejoint").qposadr[0]
        self.ball_d = m.joint("ball_freejoint").dofadr[0]
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        pdir = os.path.join(ASSETS, "policies")
        self.sess = {k: ort.InferenceSession(os.path.join(pdir, f), so, providers=["CPUExecutionProvider"])
                     for k, f in POLICIES.items()}
        self.lock = threading.RLock()
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self):
        with self.lock:
            m, d = self.model, self.data
            mujoco.mj_resetData(m, d)
            d.qpos[0:7] = [SPAWN[0], SPAWN[1], 0.12, 1, 0, 0, 0]
            for i, a in enumerate(self.qadr):
                d.qpos[a] = DEFAULT_POSE[i]
            self.place_ball(*self.ball_home)
            d.ctrl[self.act_ids] = DEFAULT_POSE
            mujoco.mj_forward(m, d)
            self.last_action = np.zeros(14, np.float32)
            self.cmd = np.zeros(13, np.float32)
            self.vel = np.zeros(3, np.float32)      # commanded twist (vx, vy, wz)
            self.head_target = np.zeros(4, np.float32)  # [neck_pitch, head_pitch, head_yaw, head_roll] offsets
            self.head_smooth = np.zeros(4, np.float32)
            self.mode = "walk"
            self.sit_flag = 0.0
            self.kick_steps = 0
            self.post_kick = 0
            self.recovery = None
            self.fall_debounce = 0
            self.steps = 0
            self.events = []
            self.timers = []   # (step_due, fn)

    def place_ball(self, x, y):
        d = self.data
        d.qpos[self.ball_q:self.ball_q + 7] = [x, y, BALL_R + 0.005, 1, 0, 0, 0]
        d.qvel[self.ball_d:self.ball_d + 6] = 0

    def proj_gravity(self):
        q = self.data.xquat[self.trunk]
        R = np.zeros(9); mujoco.mju_quat2Mat(R, q); R = R.reshape(3, 3)
        return R.T @ np.array([0, 0, -1.0])

    def yaw(self):
        q = self.data.qpos[3:7]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def build_obs(self):
        d = self.data
        o = np.zeros(61, np.float32)
        o[0:3] = d.sensordata[self.gyro_adr:self.gyro_adr + 3]
        o[3:6] = self.proj_gravity()
        o[6:20] = d.qpos[self.qadr] - DEFAULT_POSE
        o[20:34] = d.qvel[self.dadr]
        o[34:48] = self.last_action
        cmd = np.zeros(13, np.float32)
        if self.mode == "sitstand":
            cmd[0] = self.sit_flag
        elif self.mode == "walk" and self.recovery is None and self.post_kick == 0:
            cmd[0:3] = self.vel
        self.head_smooth += 0.2 * (self.head_target - self.head_smooth)   # runtime EMA, alpha 0.2 @ 50 Hz
        if self.recovery is None:
            cmd[3:7] = self.head_smooth
        self.cmd = cmd
        o[48:61] = cmd
        return o

    def active_session(self):
        if self.recovery and self.recovery["state"] == "recovering":
            return self.sess["stand"]
        return self.sess[self.mode]

    # ---------------------------------------------------------------- loop
    def step(self):
        with self.lock:
            d, m = self.data, self.model
            if not (self.recovery and self.recovery["state"] == "fallen"):
                s = self.active_session()
                act = s.run(None, {s.get_inputs()[0].name: self.build_obs()[None]})[0][0]
                if np.all(np.isfinite(act)):
                    self.last_action = act.astype(np.float32)
                    d.ctrl[self.act_ids] = DEFAULT_POSE + self.last_action
            for _ in range(DECIMATION):
                mujoco.mj_step(m, d)
            self.steps += 1
            # timers
            due = [t for t in self.timers if t[0] <= self.steps]
            self.timers = [t for t in self.timers if t[0] > self.steps]
            for _, fn in due:
                fn()
            gz = self.proj_gravity()[2]
            z = d.qpos[2]
            if not np.isfinite(z):
                self.events.append("physics exploded -> reset"); self.reset(); return
            fallen = gz > -0.5 or z < 0.02
            if self.recovery:
                self.recovery["steps"] += 1
                if self.recovery["state"] == "fallen" and self.recovery["steps"] >= 15:
                    self.recovery = {"state": "recovering", "steps": 0, "up": 0}
                    self.last_action[:] = 0
                elif self.recovery["state"] == "recovering":
                    self.recovery["up"] = self.recovery["up"] + 1 if gz < -0.85 else 0
                    if self.recovery["up"] >= 50:
                        self.recovery = None; self.mode = "walk"; self.last_action[:] = 0
                        self.events.append("recovered from fall")
                    elif self.recovery["steps"] >= 300:
                        self.events.append("could not get up -> reset"); self.reset()
            elif fallen and self.mode == "walk" and self.post_kick == 0:
                self.fall_debounce += 1
                if self.fall_debounce >= 10:
                    self.fall_debounce = 0
                    self.vel[:] = 0
                    self.recovery = {"state": "fallen", "steps": 0}
                    self.events.append("fell over -> running stand-up policy")
            else:
                self.fall_debounce = 0
            if self.post_kick > 0 and self.mode == "walk":
                self.post_kick -= 1
            if self.mode in ("kickL", "kickR"):
                self.kick_steps += 1
                if self.kick_steps >= KICK_STEPS:
                    self.mode = "walk"; self.post_kick = POST_KICK_LOCK
            # ball escaped arena -> respawn near center
            bx, by = d.qpos[self.ball_q], d.qpos[self.ball_q + 1]
            if abs(bx) > ARENA_HALF + 0.1 or abs(by) > ARENA_HALF + 0.1:
                self.place_ball(0.0, 0.0)

    # ------------------------------------------------------------ commands
    def busy(self):
        return self.mode in ("kickL", "kickR") or self.recovery is not None or bool(self.timers)

    def set_velocity(self, vx, vy, wz):
        with self.lock:
            self.vel[:] = [float(np.clip(vx, VEL_BACK, VEL_FWD)), 0.0, float(np.clip(wz, -VEL_ANG, VEL_ANG))]

    def set_head(self, neck_pitch=0.0, head_pitch=0.0, head_yaw=0.0, head_roll=0.0):
        with self.lock:
            self.head_target[:] = np.clip([neck_pitch, head_pitch, head_yaw, head_roll], -0.8, 0.8)

    def kick(self, foot):
        with self.lock:
            if self.mode != "walk" or self.busy():
                return False
            self.vel[:] = 0
            self.mode = "kickL" if foot == "left" else "kickR"
            self.kick_steps = 0
            return True

    def sit(self):
        with self.lock:
            if self.recovery or self.mode in ("kickL", "kickR"):
                return False
            if self.mode == "sitstand" and self.sit_flag == 1:
                return True
            self.vel[:] = 0
            self.mode = "sitstand"; self.sit_flag = 0.0; self.last_action[:] = 0
            def go():
                if self.mode == "sitstand": self.sit_flag = 1.0
            self.timers.append((self.steps + 40, go))
            return True

    def stand(self):
        with self.lock:
            if self.mode != "sitstand":
                return True
            self.sit_flag = 0.0
            def back():
                self.mode = "walk"; self.last_action[:] = 0
            self.timers.append((self.steps + 100, back))
            return True

    def status(self):
        with self.lock:
            d = self.data
            bx, by = float(d.qpos[self.ball_q]), float(d.qpos[self.ball_q + 1])
            x, y = float(d.qpos[0]), float(d.qpos[1])
            yaw = self.yaw()
            dx, dy = bx - x, by - y
            bearing = math.atan2(dy, dx) - yaw
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
            posture = ("recovering" if self.recovery else "kicking" if self.mode.startswith("kick")
                       else "sitting" if (self.mode == "sitstand" and self.sit_flag == 1)
                       else "transitioning" if self.timers else "standing" if self.mode == "sitstand"
                       else "walking" if np.any(np.abs(self.vel) > 1e-3) else "standing")
            return {
                "sim_time_s": round(self.steps * CTRL_DT, 2),
                "pose": {"x": round(x, 3), "y": round(y, 3), "yaw_deg": round(math.degrees(yaw), 1),
                         "trunk_height_m": round(float(d.qpos[2]), 3)},
                "upright": bool(self.proj_gravity()[2] < -0.85),
                "posture": posture, "policy": "stand" if self.recovery else self.mode,
                "commanded_velocity": {"vx": round(float(self.vel[0]), 3), "wz": round(float(self.vel[2]), 3)},
                "ball": {"x": round(bx, 3), "y": round(by, 3), "distance_m": round(math.hypot(dx, dy), 3),
                         "bearing_deg": round(math.degrees(bearing), 1)},
                "goal": {"x": ARENA_HALF - 0.08, "y": 0.0, "width_m": 0.6},
                "arena_half_size_m": ARENA_HALF,
            }


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Approach:
    """Closed-loop 'walk to the ball and line up a kick toward the goal' behaviour.
    The Microduck walking policy only gaits for forward commands near its 0.25 m/s
    training limit, so this steers with wz while walking (pure pursuit), then stops."""
    KICK_DX = (0.06, 0.12)
    FOOT_DY = {"right": -0.045, "left": 0.045}
    # measured in this sim: the right-foot kick leaves ~16 deg right of the heading, the left ~4 deg right
    KICK_BIAS_DEG = {"right": -16.0, "left": -4.0}

    def __init__(self, sim, foot="auto", vx=0.25, target=None, stop_dist=0.17):
        self.sim, self.vx = sim, vx
        self.target = target or (ARENA_HALF - 0.08, 0.0)
        self.phase = "travel"
        self.stop_dist = stop_dist
        st = sim.status()
        if foot == "auto":
            foot = "right" if st["ball"]["bearing_deg"] <= 0 else "left"
        self.foot = foot
        self.done = False
        self.result = None
        self.tries = 0
        self.settle = 0

    def geometry(self):
        s = self.sim; d = s.data
        b = np.array([d.qpos[s.ball_q], d.qpos[s.ball_q + 1]])
        p = np.array([d.qpos[0], d.qpos[1]]); yaw = s.yaw()
        u = np.array(self.target) - b; u /= (np.linalg.norm(u) + 1e-9)
        bias = -math.radians(self.KICK_BIAS_DEG[self.foot])
        cb, sb = math.cos(bias), math.sin(bias)
        u = np.array([cb * u[0] - sb * u[1], sb * u[0] + cb * u[1]])
        n = np.array([-u[1], u[0]])
        lat = -self.FOOT_DY[self.foot]
        line0 = b + lat * n           # point the duck centre should pass through
        c, s_ = math.cos(yaw), math.sin(yaw)
        r = b - p
        rel = np.array([c * r[0] + s_ * r[1], -s_ * r[0] + c * r[1]])
        return b, p, yaw, u, n, line0, rel

    def tick(self):
        b, p, yaw, u, n, line0, rel = self.geometry()
        # signed progress along the kick line and cross-track error
        along = float(np.dot(p - line0, u))       # negative = behind the ball
        cross = float(np.dot(p - line0, n))
        if self.phase == "travel":
            # aim for a point on the kick line ahead of the duck's projection (pure pursuit)
            if along > -0.12 or abs(cross) > 0.25:
                goal = line0 - 0.40 * u           # loop round to get behind the ball
            else:
                goal = line0 + (along + 0.18) * u
            if along < -0.12 and abs(cross) < 0.03 and abs(wrap(math.atan2(u[1], u[0]) - yaw)) < 0.2:
                self.phase = "final"
            v = goal - p
            err = wrap(math.atan2(v[1], v[0]) - yaw)
            self.sim.set_velocity(self.vx, 0, float(np.clip(2.5 * err, -1, 1)))
            return
        if self.phase == "final":
            goal = line0 + (along + 0.15) * u
            v = goal - p
            err = wrap(math.atan2(v[1], v[0]) - yaw)
            if rel[0] <= self.stop_dist:
                self.sim.set_velocity(0, 0, 0)
                self.phase = "settle"; self.settle = 0
                return
            if abs(cross) > 0.08 or along > -0.03:
                self.phase = "travel"; return
            self.sim.set_velocity(self.vx, 0, float(np.clip(2.5 * err, -1, 1)))
            return
        if self.phase == "settle":
            self.settle += 1
            if self.settle >= 60:
                dy = rel[1] - self.FOOT_DY[self.foot]
                ok = self.KICK_DX[0] <= rel[0] <= self.KICK_DX[1] and abs(dy) < 0.035
                self.tries += 1
                if ok or self.tries >= 3:
                    self.done = True
                    self.result = {"lined_up": bool(ok), "foot": self.foot, "attempts": self.tries,
                                   "ball_in_duck_frame_m": {"forward": round(float(rel[0]), 3), "left": round(float(rel[1]), 3)}}
                else:
                    self.phase = "travel"
