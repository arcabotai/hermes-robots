#!/usr/bin/env bash
# Start the demo robot: turtlesim (offscreen) + rosbridge/rosapi + a synthetic camera. Used by scripts/blaxel_e2e.py.
set -eo pipefail
DISTRO="${ROS_DISTRO:-jazzy}"
set +u; source "/opt/ros/${DISTRO}/setup.bash"; set -u
export QT_QPA_PLATFORM=offscreen
HERE="$(cd "$(dirname "$0")" && pwd)"
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT INT TERM
ros2 run turtlesim turtlesim_node & pids+=($!)
ros2 launch rosbridge_server rosbridge_websocket_launch.xml & pids+=($!)
sleep 3
python3 "$HERE/fake_camera.py" & pids+=($!)
echo "demo robot up: rosbridge on ws://<this-host>:9090 (turtlesim + /camera/image_raw)"
wait -n
