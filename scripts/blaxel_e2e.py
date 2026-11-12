#!/usr/bin/env python3
"""Prove hermes-robots against REAL robots in a short-lived Blaxel sandbox instead of your own machine.

The sandbox (an Alpine microVM, no docker) gets an Ubuntu 24.04 chroot with ROS 2 Jazzy + rosbridge + turtlesim
(scripts/ros_demo) and the MuJoCo Microduck sim server running Pollen's ONNX policies (sim/microduck). Then it runs
the whole pytest suite with the live tests enabled against both, copies the logs and camera frames back, and ALWAYS
deletes the sandbox.

    export BLAXEL_API_KEY=...          # never printed
    python scripts/blaxel_e2e.py [--ref main] [--out runs/e2e] [--memory 4096]

Requires `pip install blaxel`. About 4 GB RAM for ~10 minutes. A 45 minute server-side TTL is the safety net.
"""
import argparse
import asyncio
import base64
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path

API = "https://api.blaxel.ai/v0"
REPO = "https://github.com/arcabotai/hermes-robots"
HERE = Path(__file__).resolve().parent

REMOTE = r"""#!/bin/bash
set -u
L=/tmp/logs; mkdir -p $L
say() { echo "=== $*"; }
in_root() { chroot /ros /bin/bash -c "$1"; }
say "1/7 host packages"; apk add --no-cache curl tar git bash xz >$L/apk.log 2>&1 || { tail -5 $L/apk.log; exit 1; }
say "2/7 ubuntu 24.04 rootfs"; mkdir -p /ros
curl -fsSL https://cdimage.ubuntu.com/ubuntu-base/releases/24.04/release/ubuntu-base-24.04.5-base-amd64.tar.gz | tar -xz -C /ros || exit 1
mount -t proc proc /ros/proc; mount --rbind /dev /ros/dev; cp /etc/resolv.conf /ros/etc/resolv.conf
printf '127.0.0.1 localhost\n' > /ros/etc/hosts
say "3/7 ROS 2 jazzy + rosbridge + turtlesim + OSMesa"; cp /tmp/chroot_setup.sh /ros/tmp/
chroot /ros /bin/bash /tmp/chroot_setup.sh >$L/ros_setup.log 2>&1 || { tail -20 $L/ros_setup.log; exit 1; }
tail -2 $L/ros_setup.log
say "4/7 plugin @ $REF"; in_root "git clone -q $REPO /opt/hermes-robots && cd /opt/hermes-robots && git checkout -q $REF && git log -1 --format='%H %s'"
say "5/7 ROS 2 robot: turtlesim + rosbridge + synthetic camera"
nohup chroot /ros /bin/bash -c 'ROS_DISTRO=jazzy bash /opt/hermes-robots/scripts/ros_demo/run_demo.sh' > $L/ros_demo.log 2>&1 &
in_root 'for i in $(seq 1 90); do python3 -c "import socket;socket.create_connection((\"127.0.0.1\",9090),1)" 2>/dev/null && exit 0; sleep 1; done; exit 1' \
  || { tail -20 $L/ros_demo.log; exit 1; }
sleep 4; tail -2 $L/ros_demo.log
say "6/7 Microduck: MuJoCo + Pollen ONNX policies"
in_root 'cd /opt/hermes-robots/sim/microduck && python3 -m venv /opt/simvenv && /opt/simvenv/bin/pip install -q -r requirements.txt pyopengl && /opt/simvenv/bin/python fetch_assets.py' > $L/sim_setup.log 2>&1 \
  || { tail -20 $L/sim_setup.log; exit 1; }
tail -2 $L/sim_setup.log
nohup chroot /ros /bin/bash -c 'cd /opt/hermes-robots/sim/microduck && MUJOCO_GL=osmesa exec /opt/simvenv/bin/python server.py' > $L/sim_server.log 2>&1 &
in_root 'for i in $(seq 1 120); do curl -sf http://127.0.0.1:8765/status >/dev/null && exit 0; sleep 1; done; exit 1' \
  || { tail -20 $L/sim_server.log; exit 1; }
curl -s http://127.0.0.1:8765/status | head -c 300; echo
say "7/7 pytest: unit + live (real ROS 2 over rosbridge, real Microduck policies in MuJoCo)"
in_root 'python3 -m venv /opt/venv && /opt/venv/bin/pip install -q websocket-client websockets pytest pyyaml && cd /opt/hermes-robots && /opt/venv/bin/python -m pytest tests -rA -p no:cacheprovider -o addopts=--import-mode=importlib --live-ros=127.0.0.1:9090 --live-microduck=http://127.0.0.1:8765' 2>&1 \
  | tee $L/pytest.log | grep -E "PASSED|FAILED|ERROR|passed|failed|error" | sed 's/^PASSED /  ok  /'
RC=${PIPESTATUS[0]}
grep -qE "[0-9]+ (failed|errors?)( |,|$)" $L/pytest.log && RC=1
mkdir -p /tmp/art; cp /ros/tmp/pytest-of-*/pytest-*/*/snapshots/* /tmp/art/ 2>/dev/null; cd /tmp/art && tar -czf /tmp/artifacts.tgz . 2>/dev/null
echo "RC=$RC" | tee $L/rc.txt
"""


def api(path):
    req = urllib.request.Request(API + path, headers={"Authorization": f"Bearer {os.environ['BLAXEL_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def setup_env():
    if not os.environ.get("BLAXEL_API_KEY"):
        sys.exit("BLAXEL_API_KEY not set")
    ws = os.environ.get("BL_WORKSPACE") or api("/workspaces")[0]
    if isinstance(ws, dict):
        ws = ws["name"]
    os.environ["BL_WORKSPACE"], os.environ["BL_API_KEY"] = ws, os.environ["BLAXEL_API_KEY"]
    return ws


async def main(a):
    ws = setup_env()
    from blaxel.core.sandbox import SandboxInstance
    name = f"hermes-robots-e2e-{uuid.uuid4().hex[:8]}"
    out = Path(a.out or f"runs/{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[blaxel-e2e] workspace={ws} sandbox={name} memory={a.memory}MB ref={a.ref}")
    rc, t0 = 2, time.time()
    try:
        sb = await SandboxInstance.create({"name": name, "image": "blaxel/base-image:latest", "memory": a.memory,
                                           "ttl": "45m", "region": os.environ.get("BL_REGION", "us-was-1"),
                                           "labels": {"purpose": "hermes-robots-e2e"}})
        await sb.fs.write("/tmp/chroot_setup.sh", (HERE / "chroot_setup.sh").read_text())
        await sb.fs.write("/tmp/run.sh", f"export REF={a.ref} REPO={REPO}\n" + REMOTE)
        try:
            await sb.process.exec({"name": "run", "command": "bash /tmp/run.sh", "wait_for_completion": True,
                                   "timeout": a.timeout, "keep_alive": True,
                                   "on_log": lambda l: print(l, end="" if l.endswith("\n") else "\n", flush=True)})
        except Exception as exc:  # the log stream can drop; the logs are on disk
            print(f"[blaxel-e2e] stream error ({type(exc).__name__}); fetching logs anyway")
        for n in ("pytest", "ros_demo", "ros_setup", "sim_setup", "sim_server", "rc"):
            try:
                path = "/tmp/logs/rc.txt" if n == "rc" else f"/tmp/logs/{n}.log"
                (out / f"{n}.log").write_text(await sb.fs.read(path))
            except Exception:
                pass
        try:
            p = await sb.process.exec({"name": "b64", "command": "base64 -w0 /tmp/artifacts.tgz", "wait_for_completion": True,
                                       "timeout": 60})
            blob = (getattr(p, "logs", "") or "").strip()
            if blob:
                (out / "artifacts.tgz").write_bytes(base64.b64decode(blob))
        except Exception:
            pass
        rc_txt = (out / "rc.log").read_text() if (out / "rc.log").exists() else ""
        rc = 0 if "RC=0" in rc_txt else 1 if "RC=1" in rc_txt else 2
    finally:
        for _ in range(5):
            try:
                await SandboxInstance.delete(name)
                print(f"[blaxel-e2e] deleted {name}")
                break
            except Exception as exc:
                if "404" in str(exc) or "not found" in str(exc).lower():
                    break
                await asyncio.sleep(3)
        print(f"[blaxel-e2e] elapsed={time.time() - t0:.0f}s logs={out} rc={rc}")
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="main", help="git ref of the plugin to test (default main)")
    ap.add_argument("--memory", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--out")
    sys.exit(asyncio.run(main(ap.parse_args())))
