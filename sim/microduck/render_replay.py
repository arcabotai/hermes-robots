# SPDX-License-Identifier: Apache-2.0  (port of Pollen Robotics microduck-simulator logic; see NOTICE)
"""Render a trajectory log recorded by server.py (/log/start, /log/stop) into video frames.

The live session runs physics + policies in real time without rendering; this replays the exact logged qpos
(no re-simulation) through MuJoCo's renderer with a smoothed chase camera, a duck-eye picture-in-picture and a
small HUD, and pipes raw frames into ffmpeg.

  python sim/microduck/render_replay.py log.npz out.mp4 --start <unix_ts> --end <unix_ts> --w 1120 --h 1080 --fps 25
"""
import argparse, math, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from microduck_sim import build_xml  # noqa: E402

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def yaw_of(q):
    return math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log"); ap.add_argument("out")
    ap.add_argument("--start", type=float); ap.add_argument("--end", type=float)
    ap.add_argument("--w", type=int, default=1120); ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--still", default="", help="comma list of times (s from start) to also save as PNG")
    ap.add_argument("--stills-only", action="store_true")
    a = ap.parse_args()
    lg = np.load(a.log)
    T, Q, L = lg["t"], lg["qpos"], lg["label"]
    t0 = a.start or T[0]; t1 = a.end or T[-1]
    n = int((t1 - t0) * a.fps)
    model = mujoco.MjModel.from_xml_string(build_xml((0, 0)))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, a.w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, a.h)
    data = mujoco.MjData(model)
    r = mujoco.Renderer(model, a.h, a.w)
    pip = mujoco.Renderer(model, 270, 360)
    cam = mujoco.MjvCamera()
    fb = ImageFont.truetype(FONT_B, 26); fr = ImageFont.truetype(FONT_R, 22); fs = ImageFont.truetype(FONT_R, 18)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.w}x{a.h}",
                           "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                           "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    stills = {int(float(s) * a.fps): s for s in a.still.split(",") if s}
    sm_yaw = None; sm_xy = None; ball_q = model.joint("ball_freejoint").qposadr[0]
    sm_look = None
    for k in range(n):
        if a.stills_only and k not in stills and k % 5:
            continue
        tw = t0 + k / a.fps
        i = int(np.clip(np.searchsorted(T, tw), 0, len(T) - 1))
        data.qpos[:] = Q[i]; mujoco.mj_forward(model, data)
        q = Q[i]; yaw = yaw_of(q)
        if sm_yaw is None:
            sm_yaw, sm_xy = yaw, np.array(q[:2])
        d = (yaw - sm_yaw + math.pi) % (2 * math.pi) - math.pi
        sm_yaw += 0.04 * d
        sm_xy = sm_xy + 0.12 * (np.array(q[:2]) - sm_xy)
        # look between the duck and the ball so kicks stay in frame
        # sideline "broadcast" camera: looks along +y from the -y side, tracks the duck and leans toward the ball
        bxy = np.array(q[ball_q:ball_q + 2])
        tgt = np.array(q[:2]) + 0.4 * np.clip(bxy - np.array(q[:2]), -0.9, 0.9)
        sm_look = tgt if k == 0 else sm_look + 0.06 * (tgt - sm_look)
        cam.lookat[:] = [sm_look[0], sm_look[1] + 0.22, 0.05]
        cam.distance = 1.4; cam.elevation = -20; cam.azimuth = 80
        r.update_scene(data, cam); img = Image.fromarray(r.render())
        pip.update_scene(data, "duck_eye"); eye = Image.fromarray(pip.render())
        dr = ImageDraw.Draw(img, "RGBA")
        dr.rectangle([0, 0, a.w, 52], fill=(13, 17, 23, 230))
        dr.text((18, 12), "Microduck in MuJoCo", font=fb, fill=(255, 255, 255))
        dr.text((18 + dr.textlength("Microduck in MuJoCo", font=fb) + 18, 17), "Pollen Robotics / Hugging Face RL policies  ·  real time", font=fs, fill=(160, 200, 255))
        x0, y0 = a.w - 360 - 16, 68
        img.paste(eye, (x0, y0)); dr.rectangle([x0 - 2, y0 - 2, x0 + 361, y0 + 271], outline=(255, 255, 255, 200), width=2)
        dr.rectangle([x0, y0 + 240, x0 + 360, y0 + 270], fill=(0, 0, 0, 160))
        dr.text((x0 + 8, y0 + 244), "duck's head camera", font=fs, fill=(255, 255, 255))
        pol, beh, vx, wz = str(L[i]).split("|")
        pol_name = {"walk": "walk (BEST_alpha_walking)", "sitstand": "sit/stand (BEST_alpha_sitstand)",
                    "kickL": "KICK left (ball_kick_left)", "kickR": "KICK right (ball_kick_right)",
                    "stand": "get-up (BEST_alpha_stand)"}.get(pol, pol)
        dr.rectangle([0, a.h - 92, a.w, a.h], fill=(13, 17, 23, 200))
        dr.text((18, a.h - 82), f"policy: {pol_name}", font=fr, fill=(255, 220, 120))
        dr.text((18, a.h - 48), f"behaviour: {beh}    command: vx {float(vx):+.2f} m/s   wz {float(wz):+.2f} rad/s", font=fs, fill=(220, 220, 220))
        if not a.stills_only:
            ff.stdin.write(img.tobytes())
        if k in stills:
            img.save(os.path.splitext(a.out)[0] + f"-still-{stills[k]}.png")
        if k % 250 == 0:
            print(f"{k}/{n}", flush=True)
    ff.stdin.close(); ff.wait()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
