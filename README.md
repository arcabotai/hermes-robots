# hermes-robots 🤖

**One [Hermes Agent](https://hermes-agent.nousresearch.com) plugin for driving robots: one set of tools, pluggable
backends, and one safety interlock the model can't talk its way past.**

`hermes-robots` merges [hermes-microduck](https://github.com/arcabotai/hermes-microduck) and
[hermes-ros](https://github.com/arcabotai/hermes-ros) into a single plugin. The agent uses the same eight tools whether it
is walking Pollen Robotics' Microduck in MuJoCo, driving a TurtleBot3 in Gazebo over rosbridge, or playing with the
built-in mock. Every moving or writing call goes through the same `pre_tool_call` interlock, which enforces speed and
duration caps, a projected-path geofence, and an e-stop that only a human can clear. It fails closed.

| Backend | Robot | Needs |
|---|---|---|
| `mock` | 2D differential-drive base, a raycast camera and an overhead map | nothing |
| `microduck` | [Microduck](https://huggingface.co/spaces/pollen-robotics/microduck-simulator), Pollen Robotics' bipedal duck running its **real trained RL policies** (`mode: sim`), or a kinematic stand-in (`mode: mock`) | `sim/microduck/server.py` (MuJoCo + onnxruntime) for `sim` |
| `ros` | any ROS 1 / ROS 2 robot over a rosbridge websocket (TurtleBot3 + Gazebo, turtlesim, a real base) | `rosbridge_server` on the robot side; no ROS on the Hermes host |

Adding a robot kind (Reachy Mini, a real base, ...) means writing one `RobotBackend` subclass and adding one registry entry. The tools, the interlock,
`/robot` and the audit log work with it unchanged. See [Adding a robot](#adding-a-robot).

## Tools

| Tool | What it does | Interlock |
|---|---|---|
| `robot_list` | configured robots, their kind, capabilities and extra actions; e-stop state | read-only |
| `robot_connect` | make a robot active and open its link; returns limits, views and actions | read-only |
| `robot_state` | pose `{x, y, yaw_deg}`, kind-specific state (ball and goal for the duck), limits, e-stop | read-only |
| `robot_look` | one camera frame saved to disk (`MEDIA:` tag for the user, path for `vision_analyze`; `return_mode: inline` sends it to the model directly) | read-only |
| `robot_move` | body velocity `linear_mps` / `angular_rps` for `duration_s`, then stop | speed caps, duration cap, projected-path geofence, e-stop |
| `robot_turn` | turn in place by `degrees` (the duck's own turn controller, or a timed rotation) | duration cap, edge clearance, e-stop |
| `robot_action` | a backend-specific extra (below) | `read` actions pass; `motion`/`write` actions get the e-stop check plus the backend's validator, which reuses the shared checks |
| `robot_stop` | stop now; `estop=true` latches the e-stop and stops **every** robot | never blocked |

Backend extras through `robot_action`:

* **microduck:** `kick_ball` (walk to the ball, line it up and kick it toward the goal; reports `goal_scored`), `kick`,
  `sit`, `stand`, `head` (head yaw and pitch capped by `max_head_rad`).
* **ros:** `topics` and `echo` (read; laser scans get a `scan_summary` with the nearest obstacle per 45° sector), plus
  `publish` and `call_service` (writes restricted to `write_allowlist`; a Twist published on a velocity topic gets the
  same speed, duration and geofence checks as `robot_move` and is followed by zero velocity).
* **mock:** `cones` (read), `teleport` (write, target must be inside the geofence).

`/robot status | list | use <name> | estop [reason] | clear | log [n] | stop | reset [...]` are the human's controls.
The agent cannot type slash commands, so only a human can clear the e-stop.

## The interlock

One `Interlock` (`safety.py`) serves every robot. For each vetted call it resolves the robot and computes its limits:
generic base, then the backend's defaults, then `settings.safety`, then `robots.<name>.safety`, then the backend's
**hard limits**, which config cannot raise (the Microduck walking policy is only trained up to 0.25 m/s).

* **Speed:** forward, reverse and angular caps. Over-limit calls are blocked, or clamped with `over_limit_action: clamp`.
* **Duration:** every motion is bounded. A default is injected when the model leaves it out.
* **Geofence:** a rectangle in the robot's odometry frame. The command's path is dead-reckoned from the current pose and
  must stay inside the fence minus a margin. A stale or unreadable pose blocks the move. Moving backends also receive
  the fence and abort at runtime if the robot reaches it.
* **E-stop:** one latch for all robots, persisted in `estop.json` so it survives restarts (an unreadable file counts as
  latched). It voids approvals already issued, aborts running ROS and mock moves, and is cleared only by `/robot clear`.
* **Fail closed:** the hook mints a single-use token bound to the exact arguments it approved, and the handlers execute
  only those arguments. A call that skipped the hook, a forged or replayed token, or an edit made to the arguments after
  vetting does nothing. An exception inside the interlock blocks the call.
* **Audit:** every refusal is logged with a stable rule id (`over_speed_forward`, `geofence`, `estop_latched`,
  `topic_not_allowlisted`, ...). See `/robot log` and `blocked.jsonl`.

## Quickstart

```bash
hermes plugins install arcabotai/hermes-robots --enable
hermes chat -t robots
> List the robots, connect to the duck and kick the ball into the goal.
> Drive the mock robot to the red cone.
> Full speed for 30 seconds!          # blocked: over_speed_forward / over_duration
> /robot log
```

With no settings you get two dependency-free robots, `mock` and `duck` (a kinematic Microduck). Real robots:

```yaml
# ~/.hermes/config.yaml
plugins:
  entries:
    hermes-robots:
      settings:
        active: duck
        robots:                                   # replaces the built-in pair
          duck:
            backend: microduck
            mode: sim                             # MuJoCo + Pollen's ONNX policies
            url: http://127.0.0.1:8765
          tb3:
            backend: ros
            host: localhost
            port: 9090
            topics: {cmd_vel: /cmd_vel, cmd_vel_type: geometry_msgs/msg/Twist, odom: /odom, camera: /camera/image_raw}
            safety:
              max_linear_mps: 0.2                 # TurtleBot3 waffle tops out at 0.26 m/s
              geofence: {enabled: true, x_min: -2, x_max: 2, y_min: -2, y_max: 2}
              write_allowlist: {topics: [/cmd_vel], services: []}
        camera: {return_mode: path}               # or inline for multimodal models
```

**Microduck in MuJoCo** (Python 3.10+; downloads Pollen's MJCF, meshes and policies from their Hugging Face Space, about 7 MB):

```bash
cd ~/.hermes/plugins/hermes-robots/sim/microduck
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python fetch_assets.py
MUJOCO_GL=egl .venv/bin/python server.py           # or MUJOCO_GL=glfw with a display / Xvfb
```

**TurtleBot3 in Gazebo** (ROS 2 Humble, Ubuntu 22.04):

```bash
sudo apt install ros-humble-ros-base ros-humble-rosbridge-server ros-humble-rosapi ros-humble-turtlebot3-gazebo
source /opt/ros/humble/setup.bash && export TURTLEBOT3_MODEL=waffle      # waffle has the camera
ros2 launch rosbridge_server rosbridge_websocket_launch.xml port:=9090 &
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

On Jazzy with Gazebo Harmonic, TurtleBot3 expects `cmd_vel_type: geometry_msgs/msg/TwistStamped`. A lighter ROS 2
target is turtlesim with a synthetic camera: `scripts/ros_demo/run_demo.sh`.

## Settings (`plugins.entries.hermes-robots.settings`)

| Key | Meaning |
|---|---|
| `active` | robot used when a tool names none (default: the first robot) |
| `robots.<name>.backend` | `mock`, `microduck`, `ros`, or a registered kind |
| `robots.<name>.*` | backend options. microduck: `mode`, `url`, `timeout_s`. ros: `host`, `port`, `connect_timeout_s`, `call_timeout_s`, `topics{cmd_vel, cmd_vel_type, odom, odom_type, camera}`. mock: `half_x`, `half_y`, `cones`, `start`, `realtime` |
| `safety` / `robots.<name>.safety` | `max_linear_mps`, `max_reverse_mps`, `max_angular_rps`, `max_duration_s`, `default_duration_s`, `turn_rate_rps`, `over_limit_action`, `geofence{enabled, x_min, x_max, y_min, y_max, margin_m, max_pose_age_s}`; microduck `max_head_rad`; ros `write_allowlist{topics, services}`, `motion_topic_patterns`, `max_publish_rate_hz` |
| `camera` | `width`, `height`, `return_mode` (`path` or `inline`), `keep_last` |

Settings are re-read on every call. A changed robot entry reconnects on its next use.

## Adding a robot

Put a module in `backends/` and add it to `_BUILTIN` in `backends/__init__.py` (or call `register_backend`):

```python
# backends/reachy_mini.py
from .base import MOTION, POSE, Action, Blocked, RobotBackend, num

class ReachyMiniBackend(RobotBackend):
    kind = "reachy_mini"
    capabilities = frozenset({POSE})                 # add MOVE / TURN / CAMERA as supported
    default_safety = {"max_head_rad": 0.5}
    def state(self): ...                              # {"pose": {...}, ...}
    def stop(self): ...
    def actions(self): return {"look_at": Action("look_at", "Point the head ...", MOTION, {...})}
    def check_action(self, name, params, guard): ...  # raise Blocked(rule, reason); guard has limits + shared checks
    def run_action(self, name, params, ctx): ...
```

`tests/test_registry.py` registers a toy arm the same way and shows that the shared interlock covers it.

## Verification

* **Unit and behaviour tests:** `python -m pytest tests` (no ROS or MuJoCo needed). They cover every interlock rule
  against both simulated robots, the Microduck actions, the sim HTTP client against a stub server, the ros backend
  against an in-process fake rosbridge that speaks the real v2 protocol, a custom backend, registration and `/robot`.
* **Real robots:** `pytest tests/live --live-ros=host:9090 --live-microduck=http://127.0.0.1:8765`.
  `scripts/blaxel_e2e.py` runs these inside a short-lived Blaxel sandbox: ROS 2 Jazzy, rosbridge and turtlesim in an
  Ubuntu chroot, plus the MuJoCo Microduck server. It always deletes the sandbox.
* **Through Hermes:** `scripts/e2e_hermes.py` drives an installed copy through `model_tools.handle_function_call`,
  the agent loop's own dispatch path with the real `pre_tool_call` hook, and through the `/robot` handler.

Results are in [docs/verification](docs/verification).

## Disclosure

* **Network:** `mock` robots make no connections. `microduck` in `sim` mode sends plain HTTP to the sim server you
  configure (default `127.0.0.1:8765`). `ros` robots open a websocket to the rosbridge you configure. Links open on
  first use, never when the plugin loads. No telemetry.
* **Setup download:** `sim/microduck/fetch_assets.py`, which you run yourself, downloads Pollen's Microduck assets
  (about 7 MB) from the `pollen-robotics/microduck-simulator` Hugging Face Space at a pinned commit. The plugin downloads
  nothing at runtime and never updates itself.
* **Background processes:** none. You start the sim server or ROS stack yourself.
* **Files:** camera snapshots, `blocked.jsonl` and `estop.json`, all under the plugin's data dir.
* **Credentials:** none.
* **Physical effects:** `ros` robots can be real hardware. The interlock limits this plugin's own tools. It does not stop
  other tools (terminal, code execution) from reaching the robot or editing `config.yaml`, so restrict those toolsets
  when real hardware is attached. Nothing here has been run on a physical robot.

## Credits and license

Microduck, its model and policies are by **Pollen Robotics / Hugging Face** (Apache-2.0). They are downloaded at setup,
not redistributed. TurtleBot3 is by ROBOTIS. This is an independent community plugin, affiliated with neither.
Plugin code: MIT. `sim/microduck/`: Apache-2.0. See [NOTICE](NOTICE).

An AI agent from the Arca Computer, Inc. agent family wrote this code.

— Opened by an AI agent from Arca. All code here is stewarded and reviewed by Luis Felipe Abarca ([@felirami](https://github.com/felirami)) of Arca Computer, Inc. (https://arca.computer). All Arca OSS contributions: https://oss.arcabot.ai
