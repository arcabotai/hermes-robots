"""Download the Microduck MJCF, meshes and ONNX policies from Pollen Robotics' Hugging Face Space (Apache-2.0).

Nothing from the Space is vendored in this repo; run this once:  python sim/microduck/fetch_assets.py
Files land in sim/microduck/assets/ (git-ignored).
"""
import os
import sys
import urllib.request

SPACE = "https://huggingface.co/spaces/pollen-robotics/microduck-simulator/resolve/{rev}/app/public/"
REV = os.environ.get("MICRODUCK_SPACE_REV", "efedf47f4d4ae52d17ad2e108a5b8795d41537a3")
POLICIES = ["BEST_alpha_walking.onnx", "BEST_alpha_sitstand.onnx", "BEST_alpha_stand.onnx", "ball_kick_left.onnx",
            "ball_kick_right.onnx", "roulade.onnx", "alpha_ground_pick.onnx"]
MESHES = ["ankle_left", "ankle_right", "banana_pcb_locker", "bearing_roll", "bottom_head_shell", "elec_rpi_robot_hat_pcb",
          "face_part", "foot_left", "foot_right", "hip_l", "jaw", "jaw_soft", "left_shell", "leg", "lens", "m12_lens_holder",
          "motor_support", "neck", "neck_pitch", "noenoeil", "np_f970", "pcb__raspberry_pi_zero_2_w", "power_support",
          "right_shell", "seeed_bearing__configuration__22x16x4", "seeed_bearing__configuration_default", "soft_mouth_top",
          "sole_left", "sole_right", "speaker", "top_head_shell", "trunk_base", "upper_leg_left", "upper_leg_right",
          "upper_leg_rigidity_plate", "xl330", "yaw2roll", "yaw_roll_motion"]


def get(rel, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 200:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = SPACE.format(rev=REV) + rel
    with urllib.request.urlopen(url, timeout=120) as r, open(dest + ".part", "wb") as f:
        f.write(r.read())
    os.replace(dest + ".part", dest)
    print("  ", rel)


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    print(f"fetching Microduck assets from pollen-robotics/microduck-simulator@{REV[:10]} -> {root}")
    get("robot/mjlab/robot_allcollisions.xml", os.path.join(root, "robot", "robot_allcollisions.xml"))
    for m in MESHES:
        get(f"robot/mjlab/meshes/{m}.stl", os.path.join(root, "robot", "meshes", f"{m}.stl"))
    for p in POLICIES:
        get(f"policies/{p}", os.path.join(root, "policies", p))
    print("done")


if __name__ == "__main__":
    sys.exit(main())
