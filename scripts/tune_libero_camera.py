"""
Interactive camera tuning for LIBERO environments.

Loads a LIBERO task from a BDDL file, then lets you move a camera
with the keyboard. Prints pos/quat so you can paste them into
_setup_camera() or scene XMLs.

Controls:
  w/s         zoom in/out
  a/d         pan left/right
  r/f         pan up/down
  arrow keys  rotate view direction
  . /         roll camera
  p           print current camera pose

Usage:
  python scripts/tune_libero_camera.py --bddl_file <path> --camera agentview
"""

import argparse

import init_path
import numpy as np
from pynput.keyboard import Key, Listener

from robosuite import load_controller_config
import robosuite.utils.transform_utils as T

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING

DELTA_POS = 0.05
DELTA_ROT = 1  # degrees


class DirectCameraMover:
    """Moves a MuJoCo camera by directly writing sim.model.cam_pos/cam_quat.
    No XML modification or env reset needed — avoids the segfault from CameraMover."""

    def __init__(self, sim, camera_name):
        self.sim = sim
        self.camera_id = sim.model.camera_name2id(camera_name)

    def get_pos(self):
        return self.sim.model.cam_pos[self.camera_id].copy()

    def get_quat_wxyz(self):
        return self.sim.model.cam_quat[self.camera_id].copy()

    def get_rot(self):
        """Return rotation matrix from camera quat (stored as wxyz in MuJoCo)."""
        q_wxyz = self.get_quat_wxyz()
        q_xyzw = T.convert_quat(q_wxyz, to="xyzw")
        return T.quat2mat(q_xyzw)

    def set_pose(self, pos, rot):
        q_xyzw = T.mat2quat(rot)
        q_wxyz = T.convert_quat(q_xyzw, to="wxyz")
        self.sim.model.cam_pos[self.camera_id] = pos
        self.sim.model.cam_quat[self.camera_id] = q_wxyz

    def move(self, direction, scale):
        """Move camera along a direction in camera frame."""
        rot = self.get_rot()
        world_dir = rot @ np.array(direction)
        new_pos = self.get_pos() + world_dir * scale
        self.set_pose(new_pos, rot)

    def rotate(self, axis, angle_deg):
        """Rotate camera about an axis in camera frame."""
        pos = self.get_pos()
        rot = self.get_rot()
        rad = np.deg2rad(angle_deg)
        # Build rotation matrix about the given axis in camera frame
        delta_rot = T.rotation_matrix(rad, axis, point=None)[:3, :3]
        new_rot = rot @ delta_rot
        self.set_pose(pos, new_rot)

    def print_pose(self, camera_name):
        pos = self.get_pos()
        q = self.get_quat_wxyz()
        print("\n" + "=" * 60)
        print("For _setup_camera() in env file (Python):")
        print(f"  pos=[{pos[0]}, {pos[1]}, {pos[2]}],")
        print(f"  quat=[{q[0]}, {q[1]}, {q[2]}, {q[3]}],")
        print()
        print("For scene XML:")
        print(f'  <camera mode="fixed" name="{camera_name}" '
              f'pos="{pos[0]} {pos[1]} {pos[2]}" '
              f'quat="{q[0]} {q[1]} {q[2]} {q[3]}"/>')
        print("=" * 60)


class KeyboardHandler:
    def __init__(self, cam):
        self.cam = cam
        self.listener = Listener(on_press=self.on_press, on_release=lambda k: None)
        self.listener.start()

    def on_press(self, key):
        try:
            if key == Key.up:
                self.cam.rotate([1.0, 0.0, 0.0], DELTA_ROT)
            elif key == Key.down:
                self.cam.rotate([-1.0, 0.0, 0.0], DELTA_ROT)
            elif key == Key.left:
                self.cam.rotate([0.0, 1.0, 0.0], DELTA_ROT)
            elif key == Key.right:
                self.cam.rotate([0.0, -1.0, 0.0], DELTA_ROT)
            elif key.char == "w":
                self.cam.move([0.0, 0.0, -1.0], DELTA_POS)
            elif key.char == "s":
                self.cam.move([0.0, 0.0, 1.0], DELTA_POS)
            elif key.char == "a":
                self.cam.move([-1.0, 0.0, 0.0], DELTA_POS)
            elif key.char == "d":
                self.cam.move([1.0, 0.0, 0.0], DELTA_POS)
            elif key.char == "r":
                self.cam.move([0.0, 1.0, 0.0], DELTA_POS)
            elif key.char == "f":
                self.cam.move([0.0, -1.0, 0.0], DELTA_POS)
            elif key.char == ".":
                self.cam.rotate([0.0, 0.0, 1.0], DELTA_ROT)
            elif key.char == "/":
                self.cam.rotate([0.0, 0.0, -1.0], DELTA_ROT)
            elif key.char == "p":
                self.cam.print_pose(args.camera)
        except AttributeError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively tune camera poses in LIBERO environments")
    parser.add_argument("--bddl_file", type=str, required=True, help="Path to BDDL task file")
    parser.add_argument("--camera", type=str, default="agentview", help="Camera name to tune (default: agentview)")
    parser.add_argument("--robots", nargs="+", type=str, default=["Panda"], help="Robot(s) to use")
    args = parser.parse_args()

    # Load LIBERO environment
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    problem_name = problem_info["problem_name"]
    print(f"Loading: {problem_info['language_instruction']}")
    print(f"Problem: {problem_name}")

    controller_config = load_controller_config(default_controller="OSC_POSE")

    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        robots=args.robots,
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()

    # Set up direct camera mover (no XML modification, no env reset)
    cam = DirectCameraMover(env.sim, args.camera)
    camera_id = env.sim.model.camera_name2id(args.camera)
    env.viewer.set_camera(camera_id=camera_id)

    # Start keyboard handler
    KeyboardHandler(cam)

    print("\n--- Camera Tuner ---")
    print("w/s: zoom | a/d: pan L/R | r/f: pan U/D | arrows: rotate | ./: roll")
    print("p: print current pose | Ctrl+C: quit\n")

    # Print initial pose
    cam.print_pose(args.camera)

    step = 0
    while True:
        env.step(np.zeros(env.action_dim))
        env.render()
        step += 1
        if step % 500 == 0:
            cam.print_pose(args.camera)
