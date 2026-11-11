"""Tool schemas (OpenAI function format). One set of generic tools for every robot kind."""
from __future__ import annotations

_ROBOT = {"type": "string", "description": "Robot name from robot_list. Omit to use the active robot."}

ROBOT_LIST = {
    "name": "robot_list",
    "description": "List the configured robots: name, kind (mock, microduck, ros, ...), which one is active, what each "
                   "can do (capabilities and extra actions) and the e-stop state. Start here.",
    "parameters": {"type": "object", "properties": {}},
}

ROBOT_CONNECT = {
    "name": "robot_connect",
    "description": "Make a robot the active one and open its link (sim server, rosbridge, ...). Returns its state, its "
                   "safety limits and its extra actions. Moves nothing.",
    "parameters": {"type": "object", "properties": {"robot": {"type": "string", "description": "Robot name from robot_list."}},
                   "required": ["robot"]},
}

ROBOT_STATE = {
    "name": "robot_state",
    "description": "Current state of a robot: pose {x, y, yaw_deg} in its odometry frame (x forward at yaw 0, + yaw = "
                   "left), kind-specific details (ball and goal for Microduck, velocity and collisions, ...), the "
                   "active safety limits, the e-stop state and recently blocked calls.",
    "parameters": {"type": "object", "properties": {"robot": _ROBOT}},
}

ROBOT_LOOK = {
    "name": "robot_look",
    "description": "Take a picture with the robot's camera and save it. Returns the file path: pass it to vision_analyze "
                   "to see it, and put MEDIA:<path> on its own line in your reply to show it to the user. Views depend "
                   "on the robot (robot_state lists them), e.g. duck_eye / chase / overhead for Microduck.",
    "parameters": {"type": "object", "properties": {"robot": _ROBOT, "view": {"type": "string"}}},
}

ROBOT_MOVE = {
    "name": "robot_move",
    "description": "Drive the robot with a body velocity for a bounded time, then stop. linear_mps > 0 forward, < 0 "
                   "backward; angular_rps > 0 turns left. A safety interlock checks speed and duration limits and that "
                   "the projected path stays inside the geofence; if it blocks the call, tell the user why instead of "
                   "retrying a smaller move on your own. Returns the pose before/after and why the move ended.",
    "parameters": {"type": "object", "properties": {
        "robot": _ROBOT,
        "linear_mps": {"type": "number", "description": "Forward speed in m/s (negative = backward)."},
        "angular_rps": {"type": "number", "description": "Turn rate in rad/s (positive = left).", "default": 0},
        "duration_s": {"type": "number", "description": "How long to move, in seconds (capped; a default applies if omitted)."},
    }, "required": ["linear_mps"]},
}

ROBOT_TURN = {
    "name": "robot_turn",
    "description": "Turn in place by an angle: degrees > 0 = left (counter-clockwise), < 0 = right. Uses the robot's "
                   "own turn controller when it has one (Microduck), otherwise a slow timed rotation.",
    "parameters": {"type": "object", "properties": {"robot": _ROBOT, "degrees": {"type": "number"}}, "required": ["degrees"]},
}

ROBOT_ACTION = {
    "name": "robot_action",
    "description": "Run one of the robot's extra actions, listed by robot_list / robot_connect with their params. "
                   "Microduck: kick_ball, kick, sit, stand, head. ROS: topics, echo (read), publish, call_service "
                   "(allowlisted writes). Mock: cones, teleport. Moving or writing actions pass the same safety "
                   "interlock as robot_move.",
    "parameters": {"type": "object", "properties": {
        "robot": _ROBOT,
        "action": {"type": "string", "description": "Action name."},
        "params": {"type": "object", "description": "Action parameters."},
    }, "required": ["action"]},
}

ROBOT_STOP = {
    "name": "robot_stop",
    "description": "Stop the robot now. Never blocked. With estop=true it also latches the emergency stop for ALL "
                   "robots: every motion is then refused until the human clears it with /robot clear (you cannot). "
                   "Use estop=true if the user says stop / halt / emergency or anything looks unsafe.",
    "parameters": {"type": "object", "properties": {
        "robot": {"type": "string", "description": "Robot name, or 'all'. Omit for the active robot."},
        "estop": {"type": "boolean", "default": False},
        "reason": {"type": "string"},
    }},
}

ALL = [ROBOT_LIST, ROBOT_CONNECT, ROBOT_STATE, ROBOT_LOOK, ROBOT_MOVE, ROBOT_TURN, ROBOT_ACTION, ROBOT_STOP]
