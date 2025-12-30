"""
Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments.

Notes:
    - We save image observations at 256x256px resolution (instead of 128x128).
    - OPTINOAL: We filter out transitions with "no-op" (zero) actions that do not change the robot's state.
    - OPTIONAL: We filter out unsuccessful demonstrations.
    - We rotate the images by 180 degrees similar to opanvla and openpi.

Usage:
    python experiments/robot/libero/regenerate_libero_dataset.py \
        --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 ] \
        --libero_task_id <TASK ID> \
        --libero_raw_data_file <PATH TO RAW HDF5 DATASET FILE> \
        --libero_target_dir <PATH TO TARGET DIR>
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

only_render = False

def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env_args["has_renderer"] = only_render
    env_args["has_offscreen_renderer"] = not only_render
    env_args["use_camera_obs"] = not only_render
    env = ControlEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


IMAGE_RESOLUTION = 256


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def main(args):
    print(f"Regenerating {args.libero_task_suite} dataset!")

    # Create target directory
    if os.path.isdir(args.libero_target_dir):
        user_input = input(f"Target directory already exists at path: {args.libero_target_dir}\nEnter 'y' to overwrite the directory, or anything else to exit: ")
        if user_input != 'y':
            exit()
    os.makedirs(args.libero_target_dir, exist_ok=True)


    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    task_ids = [args.libero_task_id]

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0

    for task_id in task_ids:
        # Get task in suite
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
        print(f"Regenerating dataset for task '{task_description}' (ID: {task_id})...")

        # Get dataset for task
        orig_data_path = os.path.join(args.libero_raw_data_file)
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        # Create new HDF5 file for regenerated demos
        new_data_path = os.path.join(args.libero_target_dir, "demo.hdf5")
        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")

        for _, demo in enumerate(orig_data.keys()):
            # Get demo data
            demo_data = orig_data[demo]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]

            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()
            env.set_init_state(orig_states[0])
            for _ in range(10):
                obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

            # Set up new data lists
            states = []
            actions = []
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            agentview_images = []
            eye_in_hand_images = []

            # Replay original demo actions in environment and record observations
            for _, action in enumerate(orig_actions):
                # Skip transitions with no-op actions
                # prev_action = actions[-1] if len(actions) > 0 else None
                # if is_noop(action, prev_action):
                #     print(f"\tSkipping no-op action: {action}")
                #     num_noops += 1
                #     continue

                if states == []:
                    # In the first timestep, since we're using the original initial state to initialize the environment,
                    # copy the initial state (first state in episode) over from the original HDF5 to the new one
                    states.append(orig_states[0])
                    robot_states.append(demo_data["robot_states"][0])
                else:
                    # For all other timesteps, get state from environment and record it
                    states.append(env.sim.get_state().flatten())
                    robot_states.append(
                        np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                    )

                # Record original action (from demo)
                actions.append(action)

                # Record data returned by environment
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                joint_states.append(obs["robot0_joint_pos"])
                ee_states.append(
                    np.hstack(
                        (
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                        )
                    )
                )
                if not only_render:
                    agentview_images.append(obs["agentview_image"][::-1, ::-1]) # rotate by 180 degree similar to openvla
                    eye_in_hand_images.append(obs["robot0_eye_in_hand_image"][::-1, ::-1]) # rotate by 180 degree similar to openvla
                else:
                    env.env.render()

                # Execute demo action in environment
                obs, reward, done, info = env.step(action.tolist())
            

            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            if not only_render:
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1
                assert len(actions) == len(agentview_images)

                ep_data_grp = grp.create_group(demo)
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
                obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
                obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
                ep_data_grp.create_dataset("actions", data=actions)
                ep_data_grp.create_dataset("states", data=np.stack(states))
                ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                ep_data_grp.create_dataset("rewards", data=rewards)
                ep_data_grp.create_dataset("dones", data=dones)

                num_success += 1

            num_replays += 1

            # Count total number of successful replays so far
            print(
                f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
            )

            # Report total number of no-op actions filtered out so far
            print(f"  Total # no-op actions filtered out: {num_noops}")

        # Close HDF5 files
        orig_data_file.close()
        new_data_file.close()
        print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")

    print(f"Dataset regeneration complete! Saved new dataset at: {args.libero_target_dir}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, default="libero_10", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial")
    parser.add_argument("--libero_task_id", type=int, default=4,
                        help="LIBERO task ID.")
    parser.add_argument("--libero_raw_data_file", type=str, default="/home/admin_07/project_repos/LIBERO/libero/datasets/libero_10",
                        help="Path to raw HDF5 dataset file. Example: ./LIBERO/libero/datasets/libero_spatial")
    parser.add_argument("--libero_target_dir", type=str, default="/home/admin_07/project_repos/LIBERO/libero/datasets/libero_10_regenerated_openvla",
                        help="Path to regenerated dataset path. Example: ./LIBERO/libero/datasets/libero_spatial_no_noops")
    args = parser.parse_args()

    # Start data regeneration
    main(args)