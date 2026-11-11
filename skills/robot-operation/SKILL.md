---
name: robot-operation
description: How to drive robots safely with the hermes-robots tools (Microduck, ROS 2 robots such as TurtleBot3, the mock) - pre-flight, small verified moves, kind-specific actions, reading interlock refusals, e-stop discipline.
---

# Driving robots with hermes-robots

You are controlling something that can hit a wall, a person or itself. A `pre_tool_call` interlock enforces hard limits,
but it is the *last* line of defence. You are the first. Work like a careful pilot.

## Pre-flight
1. `robot_list` - which robots exist, which one is active, what each can do, whether the e-stop is latched.
2. `robot_connect robot=<name>` - select the robot and open its link. Read its `limits`, `views` and `actions`.
3. `robot_state` - pose `{x, y, yaw_deg}` (yaw 0 = +x, positive angles turn left) and kind-specific details.
4. `robot_look`, then `vision_analyze` on the returned path - look before you move and say what you see.

## Moving: the loop
Never plan a long blind route. Repeat: **look → one small move → verify → look again**.
- `robot_move linear_mps=… angular_rps=… duration_s=…`: short (1-3 s) and well under the limits. Turn first, then drive.
- `robot_turn degrees=…` to change heading (positive = left).
- After every move read `pose_after`, `displacement_m`, `end_reason` (and `collided` when present). If the robot did
  not do what you expected, stop and re-assess.
- Stop at least ~0.5 m before obstacles. Avoid reversing: you cannot see behind the robot.

## Kind-specific notes
- **microduck** (25 cm bipedal duck on Pollen Robotics' trained RL policy; `mode` sim = MuJoCo, mock = kinematic):
  0.25 m/s forward is the trained walking speed (also its hard limit); below ~0.2 m/s it just stands, and real speed is
  about 0.1 m/s. Turns arc 10-20 cm. Arena 3 x 3 m, +x points at the goal. Views: `duck_eye`, `chase`, `overhead`.
  Actions: `kick_ball` (approach, line up and kick toward the goal; reports `goal_scored`), `kick`, `sit`, `stand`,
  `head` (`yaw` + left, `pitch` + up, radians). It cannot walk while sitting.
- **ros** (TurtleBot3 in Gazebo, a real base, ...): check the topics exist first with
  `robot_action action=topics params={"filter":"cmd_vel"}`. `robot_action action=echo params={"topic":"/scan"}` adds a
  `scan_summary` (nearest obstacle per 45° sector) to laser scans - use it before driving. `publish` and
  `call_service` only work on allowlisted names.
- **mock**: a 2D base in a 10 x 10 m arena with a red cone; views `camera` and `overhead`.

## When the interlock says no
`hermes-robots interlock BLOCKED <tool> [rule]: reason` is final for that request. Do **not** retry it, split it into
many smaller "legal" moves to reach the same result, switch to another robot or tool, or touch the robot any other
way (terminal, python, publishing to a different topic). Tell the user what was refused and why, offer a safe
alternative, and run it only after they agree.
- `over_speed_*` / `over_duration` / `over_head_limit`: the request is too aggressive.
- `geofence` / `no_pose` / `pose_stale`: the move would leave the allowed area, or the position is unknown.
- `*_not_allowlisted`: you may not write there; only a human changes the allowlist.
- `estop_latched`: stop trying to move. Only the user can clear it, with `/robot clear`.

## E-stop discipline
- If the user says stop / halt / wait, or anything looks unsafe (collision, a fall, odd odometry, lost camera):
  `robot_stop estop=true` **first** (it stops every robot), explain afterwards.
- `robot_stop` without `estop` is a brake with no latch.
- Never try to clear the latch; you cannot, and you must not look for a way.

## Honesty
Report what the robot actually did (returned poses, `end_reason`, `goal_scored`), not what you intended. If the camera
or pose is unavailable, say so and do not guess. Show pictures with the `MEDIA:<path>` line from `robot_look`.
