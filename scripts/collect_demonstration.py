import argparse
import cv2
import keyboard
import datetime
import h5py
import init_path
import json
import numpy as np
import os
import robosuite as suite
import time
from glob import glob
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.input_utils import input2action

from libero.success_functions import lift_white_yellow_mug, false, lift_white_yellow_mug_and_place_on_plate
is_success = false

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *

from pynput import keyboard
import threading

# SAM3 and camera utilities for position estimation
from libero.libero.utils.sam3_client import SAM3StreamClient
from robosuite.utils.camera_utils import (
    get_real_depth_map,
    get_camera_transform_matrix,
)

def bilinear_interpolate(im, x, y):
    """
    Bilinear sampling for pixel coordinates x and y from source image im.
    Taken from https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def transform_from_pixels_to_world(pixels, depth_map, camera_to_world_transform):
    pixels = pixels.astype(float)
    z = bilinear_interpolate(im=depth_map, x=pixels[..., 1], y=pixels[..., 0])
    z = z.reshape(-1, 1)  # shape [..., 1]

    # form 4D homogenous camera vector to transform - [x * z, y * z, z, 1]
    # (note that we need to swap the first 2 dimensions of pixels to go from pixel indices
    # to camera coordinates)
    cam_pts = [pixels[..., 1:2] * z, pixels[..., 0:1] * z, z, np.ones_like(z)]
    cam_pts = np.concatenate(cam_pts, axis=-1)  # shape [..., 4]

    # batch matrix multiplication of 4 x 4 matrix and 4 x 1 vectors to do camera to robot frame transform
    mat_reshape = [1] * len(cam_pts.shape[:-1]) + [4, 4]
    cam_trans = camera_to_world_transform.reshape(mat_reshape)  # shape [..., 4, 4]
    points = np.matmul(cam_trans, cam_pts[..., None])[..., 0]  # shape [..., 4]
    return points[..., :3]

# Configurable subtask states for demonstration collection
# Cycle starts at "0", then goes through each subtask in this list
# Final 'l' press after last subtask sends task completion signal
SUBTASK_STATES = ["place"]

class KeyboardInputManager:
    """Manages the pynput listener and provides a thread-safe flag."""
    
    def __init__(self):
        # 1. The shared flag to signal an 'A' press
        self.task_completion_key_pressed = threading.Event()
        self.gripper_trigger_key_pressed = threading.Event()
        self.subtask_cycle_key_pressed = threading.Event()
        self.current_subtask_index = -1  # Start at -1, so first press gives index 0

        # 2. Setup the listener
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        # Allows the main program to exit without waiting for the listener thread
        self.listener.daemon = True 
        self.listener.start()
        
    def _on_press(self, key):
        """Called by the listener thread when a key is pressed."""
        try:
            if key.char == 'p':
                # Set the event flag
                self.task_completion_key_pressed.set()
            if key.char == 'g':
                # Set the event flag
                if self.gripper_trigger_key_pressed.is_set():
                    self.gripper_trigger_key_pressed.clear()
                else:
                    self.gripper_trigger_key_pressed.set()
            if key.char == 'l':
                # Set the event flag for subtask cycling
                self.subtask_cycle_key_pressed.set()
        except AttributeError:
            # Handle special keys if needed
            pass

    def _on_release(self, key):
        """Called by the listener thread when a key is released."""
        # You can handle key releases here if your logic requires it
        pass
        
    def check_task_completion_signal(self):
        """Checks the flag and immediately clears it for the next check."""
        # check() returns True if the flag is set
        is_set = self.task_completion_key_pressed.is_set()

        # clear() resets the flag immediately so we only register one press per keydown
        if is_set:
            self.task_completion_key_pressed.clear()

        return is_set

    def cycle_subtask(self):
        """Cycles through subtask states with zero states between transitions

        Pattern: 0 -> subtask1 -> 0 -> subtask2 -> 0 -> ... -> subtaskN -> terminate

        Returns:
            tuple: (current_subtask, should_send_completion_signal)
        """
        is_set = self.subtask_cycle_key_pressed.is_set()

        if is_set:
            self.subtask_cycle_key_pressed.clear()
            self.current_subtask_index += 1

            # Check if this is a subtask index (even) or a return-to-zero index (odd)
            if self.current_subtask_index % 2 == 0:
                # Even index: this is an actual subtask
                subtask_number = self.current_subtask_index // 2
                if subtask_number < len(SUBTASK_STATES):
                    print("Switched to subtask:", SUBTASK_STATES[subtask_number])
                    return (SUBTASK_STATES[subtask_number], False)
                else:
                    # Shouldn't reach here, but handle gracefully
                    self.current_subtask_index = -1
                    return ("0", True)
            else:
                # Odd index: return to 0 or terminate
                subtask_just_completed = (self.current_subtask_index - 1) // 2
                if subtask_just_completed < len(SUBTASK_STATES) - 1:
                    # Not the last subtask, return to 0
                    print("Switched to subtask: 0")
                    return ("0", False)
                else:
                    # Just completed the last subtask, terminate
                    self.current_subtask_index = -1
                    return ("0", True)

        # Return current state without changes
        if self.current_subtask_index == -1:
            return ("0", False)
        elif self.current_subtask_index % 2 == 0:
            # Even index: in a subtask
            subtask_number = self.current_subtask_index // 2
            return (SUBTASK_STATES[subtask_number], False)
        else:
            # Odd index: in a 0 state between subtasks
            return ("0", False)

    def stop(self):
        """Stops the listener thread cleanly."""
        self.listener.stop()

input_manager = KeyboardInputManager()

def collect_human_trajectory(
    env, device, arm, env_configuration, problem_info, remove_directory=[], sam3_client=None, args=None
):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """
    reset_success = False
    while not reset_success:
        try:
            env.reset()
            input_manager.gripper_trigger_key_pressed.clear()
            input_manager.current_subtask_index = -1  # Reset subtask index to beginning
            env.current_subtask = "0"  # Initialize env subtask
            reset_success = True
        except:
            continue

    # ID = 2 always corresponds to agentview
    env.render()

    task_completion_hold_count = (
        -1
    )  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    # Loop until we get a reset from the input or the task completes
    saving = True
    count = 0
    received_task_completion_signal = False
    env._start_new_episode() # auto-start new episode

    # Visual servoing: target position from SAM3
    visual_servo_target = None
    obs = None  # Store observations from previous step
    world_pos_3d = None


    while True:
        count += 1
        # Set active robot
        active_robot = (
            env.robots[0]
            if env_configuration == "bimanual"
            else env.robots[arm == "left"]
        )

        # Get the newest action of action[0:5] -- ee pose, action[6] gripper state (-1 open, 1 close)
        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration,
        )
        
        # If action is none, then this a reset so we should break
        if action is None:
            print("Break")
            saving = False
            break

        # trigger gripper via keyboard; needs to be done after checking if action is None
        action[-1] = 1.0 if input_manager.gripper_trigger_key_pressed.is_set() else -1.0

        # Apply visual servoing if target is set
        if visual_servo_target is not None and obs is not None:
            # Get current end-effector position from observations
            ee_pos_key = "robot0_eef_pos" if "robot0_eef_pos" in obs else "eef_pos"
            if ee_pos_key in obs:
                current_ee_pos = obs[ee_pos_key]  # [x, y, z]

                # Calculate position delta: target - current
                visual_servo_delta = visual_servo_target - current_ee_pos

                # Override the position component of the action (first 3 elements are position delta)
                # Scale down the delta to avoid large jumps (gain factor)
                action[0:3] = visual_servo_delta * 0.5

                # print(f"[Visual Servo] Target: {visual_servo_target}, Current EE: {current_ee_pos}, Delta applied: {action[0:3]}")

        # Run environment step

        obs, _, _, _ = env.step(action)

        # SAM3-based position estimation
        if sam3_client is not None and args is not None and count % args.sam3_query_freq == 0 and world_pos_3d is None:
            try:
                camera_name = args.camera  # Use the camera specified in command line
                rgb_key = f"{camera_name}_image"
                depth_key = f"{camera_name}_depth"

                if rgb_key in obs and depth_key in obs:
                    # Flip images vertically to correct robosuite's upside-down orientation
                    rgb_image = obs[rgb_key][::-1]  # Shape: (H, W, 3), uint8
                    depth_normalized = obs[depth_key][::-1]  # Shape: (H, W, 1), float [0,1]

                    # Send frame to SAM3 server
                    sent = sam3_client.send_frame(
                        rgb_image=rgb_image,
                        sam3_stage=0,  # Use fixed stage for now
                        prompt=args.sam3_prompt
                    )

                    if sent:
                        # Try to receive segmented frame (non-blocking)
                        mask = sam3_client.receive_segmented_frame()

                        if mask is not None and mask.sum() > 0:
                            # Convert normalized depth to real depth (meters)
                            depth_real = get_real_depth_map(env.sim, depth_normalized)

                            # Save depth images for debugging
                            if args.sam3_debug_dir:
                                os.makedirs(args.sam3_debug_dir, exist_ok=True)

                                # Save normalized depth (0-1 range) as greyscale
                                depth_norm_vis = (depth_normalized * 255).astype(np.uint8)
                                depth_norm_path = os.path.join(args.sam3_debug_dir, f"depth_normalized_step_{count:04d}.png")
                                cv2.imwrite(depth_norm_path, depth_norm_vis)

                                # Save real depth as greyscale (normalize to visible range)
                                depth_real_vis = depth_real.squeeze()  # Remove channel dimension
                                # Normalize to 0-255 for visualization (clip to reasonable range)
                                depth_min, depth_max = depth_real_vis.min(), depth_real_vis.max()
                                if depth_max > depth_min:
                                    depth_real_vis = ((depth_real_vis - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                                else:
                                    depth_real_vis = np.zeros_like(depth_real_vis, dtype=np.uint8)
                                depth_real_path = os.path.join(args.sam3_debug_dir, f"depth_real_step_{count:04d}.png")
                                cv2.imwrite(depth_real_path, depth_real_vis)

                            # Get camera-to-world transformation matrix
                            # Camera dimensions (images are 256x256)
                            camera_height, camera_width = 256, 256

                            # Get full transform matrix (includes intrinsics)
                            world_to_camera = get_camera_transform_matrix(
                                env.sim,
                                camera_name,
                                camera_height,
                                camera_width
                            )
                            # Invert to get camera-to-world transform
                            camera_to_world = np.linalg.inv(world_to_camera)

                            # Transform centroid pixel + depth to 3D world coordinates
                            coords = np.argwhere(mask > 0)
                            pixels = np.array([coords[:, 0], coords[:, 1]])
                            world_pos_3d = transform_from_pixels_to_world(
                                pixels=pixels.T,
                                depth_map=depth_real[..., 0],
                                camera_to_world_transform=camera_to_world
                            )

                            world_pos_3d = np.array([
                                (np.percentile(world_pos_3d[:, 0], 95) + np.percentile(world_pos_3d[:, 0], 5)) / 2,
                                (np.percentile(world_pos_3d[:, 1], 95) + np.percentile(world_pos_3d[:, 1], 5)) / 2,
                                (np.percentile(world_pos_3d[:, 2], 95) + np.percentile(world_pos_3d[:, 2], 5)) / 2,
                            ])

                            # Extract 3D position
                            world_x, world_y, world_z = world_pos_3d

                            # Update visual servoing target
                            visual_servo_target = world_pos_3d.copy()

                            # Print position estimate to console
                            print(f"\n[SAM3 Step {count}] '{args.sam3_prompt}' Position Estimate:")
                            print( f"  Global Position: X={world_x:.4f}m, Y={world_y:.4f}m, Z={world_z:.4f}m")
                            print(f"Expected Global Position: {obs['red_coffee_mug_1_pos']}")
                            print()

                            # Update visualization markers
                            try:
                                # Update green marker to computed position
                                env.set_indicator_pos("computed_position_marker", [world_x, world_y, world_z])

                                # Update blue marker to expected position
                                expected_pos = obs['red_coffee_mug_1_pos']
                                env.set_indicator_pos("expected_position_marker", expected_pos)
                            except Exception as marker_error:
                                print(f"Warning: Could not update markers: {marker_error}")

                        else:
                            print(f"[SAM3 Step {count}] Warning: No valid mask received for '{args.sam3_prompt}'")
                else:
                    print(f"[SAM3 Step {count}] Warning: Missing observations {rgb_key} or {depth_key}")

            except Exception as e:
                print(f"[SAM3 Step {count}] Error: {e}")
                import traceback
                traceback.print_exc()

        # Cycle through subtasks and update env
        new_subtask, should_send_completion = input_manager.cycle_subtask()
        env.current_subtask = new_subtask

        # If 'l' was pressed after completing all subtasks, send completion signal
        if should_send_completion:
            received_task_completion_signal = True

        env.render()
        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        # pressing a once should be sufficient to trigger task completion
        if input_manager.check_task_completion_signal() or received_task_completion_signal or is_success(env):
            received_task_completion_signal = True
            # start recording if not running already
            if not env.started_new_episode: 
                env._start_new_episode()
                received_task_completion_signal = False
            # if pressed again: save recording
            else:
                if task_completion_hold_count > 0:
                    task_completion_hold_count -= 1  # latched state, decrement count
                else:
                    task_completion_hold_count = 10  # reset count on first success timestep; control_freq=20, so ~1.5 sec
        else:
            task_completion_hold_count = -1  # null the counter if there's no success
            

    print(count)
    # cleanup for end of data collection episodes
    if not saving:
        try:
            remove_directory.append(env.ep_directory.split("/")[-1])
        except:
            # most likely path does not exist because recording was not started
            pass
    env.close()
    return saving


def gather_demonstrations_as_hdf5(
    directory, out_dir, env_info, args, remove_directory=[]
):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        # print(ep_directory)
        if ep_directory in remove_directory:
            # print("Skipping")
            continue
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        subtasks = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            subtasks.extend(dic["subtasks"])

        if len(states) == 0:
            continue

        # Delete the first actions and the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action.
        del states[-1]
        del subtasks[-1]
        assert len(states) == len(actions) == len(subtasks)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        ep_data_grp.create_dataset("subtasks", data=np.array(subtasks, dtype='S'))

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    grp.attrs["problem_info"] = json.dumps(problem_info)
    grp.attrs["bddl_file_name"] = args.bddl_file
    grp.attrs["bddl_file_content"] = str(open(args.bddl_file, "r", encoding="utf-8"))

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default="demonstration_data",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="robot0_eye_in_hand",
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="OSC_POSE",
        help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'",
    )
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.5,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--num-demonstration",
        type=int,
        default=50,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument("--bddl-file", type=str)

    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50741)

    # SAM3-based position estimation arguments
    parser.add_argument(
        "--enable-sam3",
        action="store_true",
        help="Enable SAM3-based object position estimation"
    )
    parser.add_argument(
        "--sam3-send-endpoint",
        type=str,
        default="tcp://localhost:5555",
        help="SAM3 ZMQ send endpoint"
    )
    parser.add_argument(
        "--sam3-recv-endpoint",
        type=str,
        default="tcp://localhost:5556",
        help="SAM3 ZMQ receive endpoint"
    )
    parser.add_argument(
        "--sam3-prompt",
        type=str,
        default="yellow and white mug",
        help="Text prompt for SAM3 segmentation"
    )
    parser.add_argument(
        "--sam3-query-freq",
        type=int,
        default=10,
        help="Query SAM3 every N steps (default: 10)"
    )
    parser.add_argument(
        "--sam3-debug-dir",
        type=str,
        default="output_sam3",
        help="Directory to save SAM3 debug visualizations"
    )

    args = parser.parse_args()
    
    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    assert os.path.exists(args.bddl_file)
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    # Check if we're using a multi-armed environment and use env_configuration argument if so

    # Create environment
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    if "TwoArm" in problem_name:
        config["env_configuration"] = args.config
    print(language_instruction)
    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=True,
        has_offscreen_renderer=not False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20,
        camera_depths=True
    )

    # Wrap this with visualization wrapper with markers
    env = VisualizationWrapper(env, indicator_configs=[

        {
            "name": "computed_position_marker",
            "type": "sphere",
            "size": [0.04],          # 0.05 is good value for visibility
            "rgba": [0, 1, 0, 1],    # Solid green - computed position from SAM3
            "pos": [0, 0, -10]       # Start off-screen, updated during runtime
        },
        {
            "name": "expected_position_marker",
            "type": "sphere",
            "size": [0.04],          # 0.05 is good value for visibility
            "rgba": [0, 0, 1, 1],    # Solid blue - expected/ground truth position
            "pos": [0, 0, -10]       # Start off-screen, updated during runtime
        }
    ])

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "demonstration_data/tmp/{}_ln_{}/{}".format(
        problem_name,
        language_instruction.replace(" ", "_").strip('""'),
        str(time.time()).replace(".", "_"),
    )

    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
        )
        # env.viewer.add_keypress_callback("any", device.on_press)
        # env.viewer.add_keyup_callback("any", device.on_release)
        # env.viewer.add_keyrepeat_callback("any", device.on_press)
        env.viewer.add_keypress_callback(device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            args.vendor_id,
            args.product_id,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(
        args.directory,
        f"{domain_name}_ln_{problem_name}_{t1}_{t2}_"
        + language_instruction.replace(" ", "_").strip('""'),
    )

    os.makedirs(new_dir)

    # Initialize SAM3 client if enabled
    sam3_client = None
    sam3_stage = 0  # Counter for SAM3 server reset tracking

    if args.enable_sam3:
        try:
            sam3_client = SAM3StreamClient(
                send_endpoint=args.sam3_send_endpoint,
                recv_endpoint=args.sam3_recv_endpoint,
                target_size=(256, 256),  # Matches observation size
                original_size=(256, 256),
                output_dir=args.sam3_debug_dir
            )
            print(f"SAM3 enabled: debug output -> {args.sam3_debug_dir}")
            print(f"  Query frequency: every {args.sam3_query_freq} steps")
            print(f"  Segmentation prompt: '{args.sam3_prompt}'")

            # Drain any stale messages from previous runs
            sam3_client.drain_stale_messages()

        except Exception as e:
            print(f"Failed to initialize SAM3 client: {e}")
            print("Continuing without SAM3...")
            sam3_client = None

    # collect demonstrations

    remove_directory = []
    i = 0
    while i < args.num_demonstration:
        print(i)
        saving = collect_human_trajectory(
            env, device, args.arm, args.config, problem_info, remove_directory, sam3_client, args
        )
        if saving:
            print(remove_directory)
            gather_demonstrations_as_hdf5(
                tmp_directory, new_dir, env_info, args, remove_directory
            )
            i += 1

    # Cleanup SAM3 client
    if sam3_client is not None:
        sam3_client.close()
