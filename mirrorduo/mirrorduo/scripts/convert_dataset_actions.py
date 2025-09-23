import pickle
import collections
from tqdm import tqdm
import pathlib
import click
import shutil
import multiprocessing
import numpy as np
import h5py
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.config import config_factory
import mimicgen  # noqa

from mirrorduo.utils.core_utils import (
    update_env_meta_action_mode,
    goal_poses_to_delta_actions,
    VALID_ACTION_MODES,
    get_action_key,
    update_env_meta_action_mode,
)

from typing import Tuple


def convert_actions(
    env, states: np.ndarray, actions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actions = actions.copy()  # avoid modifying the original actions

    stacked_actions = actions.reshape(*actions.shape[:-1], -1, 7).astype(np.float64)
    num_frames, num_robots = stacked_actions.shape[:2]
    assert num_robots == 1, f"Multi-robot not supported yet, {stacked_actions.shape}"

    abs_goal_pos = np.zeros((num_frames, num_robots, 3), dtype=np.float64)
    abs_goal_ori = np.zeros((num_frames, num_robots, 3), dtype=np.float64)
    rel_goal_pos = np.zeros_like(abs_goal_pos)
    rel_goal_ori = np.zeros_like(abs_goal_ori)
    delta_goal_pos = np.zeros_like(abs_goal_pos)
    delta_goal_ori = np.zeros_like(abs_goal_ori)
    action_gripper = stacked_actions[..., [-1]]

    robot_eef_rots = np.zeros((num_frames, num_robots, 9), dtype=np.float64)

    for i in range(num_frames):
        env.reset_to({"states": states[i]})
        for idx, robot in enumerate(env.env.robots):
            robot.control(stacked_actions[i, idx], policy_step=True)
            controller = robot.controller

            goal_pos, goal_ori = controller.goal_pos, controller.goal_ori
            ee_pos, ee_ori_mat = controller.ee_pos, controller.ee_ori_mat
            max_dpos, max_drot = controller.output_max[0], controller.output_max[3]

            abs_goal_pos[i, idx] = goal_pos
            abs_goal_ori[i, idx] = Rotation.from_matrix(goal_ori).as_rotvec()

            # robomimic scales the control for relative actions
            rel_rot = ee_ori_mat.T @ goal_ori  # inv(R_0) @ R_t
            rel_pos = ee_ori_mat.T @ (goal_pos - ee_pos)  # inv(R_0) @ (p_t - p_0)
            rel_goal_pos[i, idx] = rel_pos / max_dpos
            rel_goal_ori[i, idx] = Rotation.from_matrix(rel_rot).as_rotvec() / max_drot

            delta_rot, delta_pos = goal_poses_to_delta_actions(goal_ori, goal_pos)
            delta_goal_pos[i, idx] = delta_pos
            delta_goal_ori[i, idx] = Rotation.from_matrix(delta_rot).as_rotvec()

            robot_eef_rots[i, idx] = ee_ori_mat.flatten()

    abs_actions = np.concatenate([abs_goal_pos, abs_goal_ori, action_gripper], axis=-1).reshape(
        actions.shape
    )
    rel_actions = np.concatenate([rel_goal_pos, rel_goal_ori, action_gripper], axis=-1).reshape(
        actions.shape
    )
    delta_actions = np.concatenate(
        [delta_goal_pos, delta_goal_ori, action_gripper], axis=-1
    ).reshape(actions.shape)

    converted_actions = {
        "absolute": abs_actions.astype(np.float32),
        "relative": rel_actions.astype(np.float32),
        "delta": delta_actions.astype(np.float32),
    }
    for mode in VALID_ACTION_MODES:
        assert mode in converted_actions, f"Invalid action mode: {mode}"
    return converted_actions, robot_eef_rots


class RobomimicActionConverter:
    """
    Converts actions from the robomimic dataset into different control modes for
    evaluation and playback. The conversion is based on the original robomimic
    OpenSim Controller (OSC) and the specific control modes defined in the
    robomimic environment.
    The conversion convert the original robomimic OSC controller actions into three additional
    control modes: "absolute", "delta", and "relative":

        - "absolute": Standard absolute control mode, where the action specifies the desired
        end-effector pose directly in the world coordinate frame.

        - "delta": Actions are expressed relative to a fixed reference frame — specifically,
        the robot's home pose. Although the controller is configured in absolute mode,
        delta actions are later converted into absolute SE(3) poses using the known home
        transformation during evaluation or playback.

        - "relative": True relative control mode, where actions specify transformations relative
        to the current end-effector pose (i.e., inv(current_pose) @ goal_pose). This requires
        both 'control_delta' and 'true_relative' to be set to True in the controller config.

    Note:
    The original robomimic OSC controller uses 'control_delta' to implement first-order
    relative control, where actions represent deltas in position (goal_pos - curr_pos)
    and orientation (goal_rot.T @ curr_rot). In contrast, the 'true_relative' mode used here
    operates on full SE(3) transformations.
    """

    def __init__(self, dataset_path, action_mode="default"):
        config = config_factory(algo_name="bc")
        ObsUtils.initialize_obs_utils_with_config(config)
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        env_meta = update_env_meta_action_mode(env_meta, action_mode)

        self.envs = {}
        self.file = h5py.File(dataset_path, "r")

        valid_modes = VALID_ACTION_MODES + ["default"]
        assert action_mode in valid_modes, (
            f"Invalid source action mode: {action_mode}. " f"Valid modes are: {valid_modes}"
        )

        for mode in valid_modes:
            updated_env_meta = update_env_meta_action_mode(env_meta, mode)
            self.envs[mode] = EnvUtils.create_env_from_metadata(
                env_meta=updated_env_meta,
                render=False,
                render_offscreen=False,
                use_image_obs=False,
            )

        self.action_key = get_action_key(action_mode)
        self.action_mode = action_mode
        self.env = self.envs[action_mode]

    def __len__(self):
        return len(self.file["data"])

    def convert_idx(self, idx):
        demo = self.file[f"data/demo_{idx}"]
        assert (
            self.action_key in demo
        ), f"source action key '{self.action_key}' not found in demo {idx}"
        source_action = demo[self.action_key][:]
        return convert_actions(self.env, demo["states"][:], source_action)

    def convert_and_eval_idx(self, idx):
        demo = self.file[f"data/demo_{idx}"]
        assert (
            self.action_key in demo
        ), f"source action key '{self.action_key}' not found in demo {idx}"
        source_action = demo[self.action_key][:]
        states = demo["states"][:]
        converted_actions, robot_eef_rots = convert_actions(self.env, states, source_action)

        robot0_eef_pos = demo["obs"]["robot0_eef_pos"][:]
        robot0_eef_quat = demo["obs"]["robot0_eef_quat"][:]

        info = {}

        for mode, actions_to_eval in converted_actions.items():
            info[mode] = self.evaluate_rollout_error(
                env=self.envs[mode],
                action_mode=mode,
                states=states,
                actions=actions_to_eval,
                robot0_eef_pos=robot0_eef_pos,
                robot0_eef_quat=robot0_eef_quat,
            )

        return converted_actions, robot_eef_rots, info

    @staticmethod
    def validate_controller(env, action_mode):
        controller = env.env.robots[0].controller
        if action_mode == "relative":
            assert controller.true_relative and controller.use_relative
        elif action_mode == "delta":
            assert (
                not controller.true_relative
                and not controller.use_relative
                and controller.true_delta
            )
        elif action_mode == "absolute":
            assert not controller.true_relative and not controller.use_relative
        elif action_mode == "default":
            pass
        else:
            raise ValueError(f"Unknown action mode: {action_mode}")

    def evaluate_rollout_error(
        self,
        env,
        action_mode,
        states,
        actions,
        robot0_eef_pos,
        robot0_eef_quat,
        metric_skip_steps=1,
    ):
        self.validate_controller(env, action_mode)
        rollout_next_states, rollout_next_eef_pos, rollout_next_eef_quat = [], [], []

        actions = actions.copy()  # avoid modifying the original actions
        states = states.copy()  # avoid modifying the original states

        for i in range(len(states)):
            env.reset_to({"states": states[i]})
            obs, _, _, _ = env.step(actions[i])
            obs = env.get_observation()
            rollout_next_states.append(env.get_state()["states"])
            rollout_next_eef_pos.append(obs["robot0_eef_pos"])
            rollout_next_eef_quat.append(obs["robot0_eef_quat"])

        next_state_diff = states[1:] - np.array(rollout_next_states[:-1])
        next_eef_pos_diff = robot0_eef_pos[1:] - np.array(rollout_next_eef_pos[:-1])
        next_eef_rot_diff = (
            Rotation.from_quat(robot0_eef_quat[1:])
            * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
        )

        return {
            "state": np.max(np.abs(next_state_diff[metric_skip_steps:])),
            "pos": np.max(np.linalg.norm(next_eef_pos_diff[metric_skip_steps:], axis=-1)),
            "rot": np.max(next_eef_rot_diff.magnitude()[metric_skip_steps:]),
        }


def worker(x):
    path, action_mode, idx, do_eval = x
    converter = RobomimicActionConverter(path, action_mode)
    return converter.convert_and_eval_idx(idx) if do_eval else (*converter.convert_idx(idx), {})


@click.command()
@click.option("-i", "--input_file", required=True, help="input hdf5 path")
@click.option("-o", "--output", required=True, help="output hdf5 path")
@click.option("-e", "--eval_dir", default=None, help="directory to output evaluation metrics")
@click.option("-b", "--batch_size", default=None, type=int)
@click.option("-n", "--num_demos", default=None, type=int)
@click.option("-a", "--action_mode", default="default", type=str)
def main(input_file, output, eval_dir, num_demos, batch_size, action_mode):
    input_file, output = pathlib.Path(input_file).expanduser(), pathlib.Path(output).expanduser()
    assert input_file.is_file() and output.parent.is_dir() and not output.is_dir()

    do_eval = bool(eval_dir)
    if do_eval:
        eval_dir = pathlib.Path(eval_dir).expanduser()
        assert eval_dir.parent.exists()

    converter = RobomimicActionConverter(input_file, action_mode=action_mode)
    total_demos = len(converter)

    # Batch jobs in chunks of `batch_size`
    if num_demos is None:
        num_demos = total_demos
    batch_size = num_demos if batch_size is None else batch_size
    all_indices = list(range(num_demos))
    results = []

    for i in range(0, num_demos, batch_size):
        batch = all_indices[i : i + batch_size]
        print(
            f"Processing batch {i // batch_size + 1} / {(num_demos + batch_size - 1) // batch_size}..."
        )

        with multiprocessing.Pool(processes=batch_size) as pool:
            batch_results = pool.map(
                worker, [(input_file, action_mode, idx, do_eval) for idx in batch]
            )
            results.extend(batch_results)

    # If output file is different, copy and open new h5py.File
    if input_file != output:
        shutil.copy(str(input_file), str(output))
        out_file = h5py.File(str(output), "r+")
    else:
        out_file = converter.file  # already opened

    for i, (converted_actions, robot_eef_rots, _) in enumerate(
        tqdm(results, desc="Writing to output")
    ):
        demo = out_file[f"data/demo_{i}"]
        for action_mode_, actions_ in converted_actions.items():
            if action_mode_ == "default":
                continue
            act_name = get_action_key(action_mode_)
            if act_name in demo:
                del demo[act_name]
            demo.create_dataset(act_name, data=actions_)

        for bot_idx in range(robot_eef_rots.shape[1]):
            key = f"robot{bot_idx}_eef_rot"
            path = f"data/demo_{i}/obs/{key}"
            if key in demo["obs"]:
                demo["obs"][key][:] = robot_eef_rots[:, bot_idx]
            else:
                out_file.create_dataset(path, data=robot_eef_rots[:, bot_idx])

    if do_eval:
        eval_dir.mkdir(parents=False, exist_ok=True)
        infos = [info for *_, info in results]
        pickle.dump(infos, eval_dir.joinpath("error_stats.pkl").open("wb"))

        metrics_dicts = {m: collections.defaultdict(list) for m in ["pos", "rot"]}
        for info in infos:
            for k, v in info.items():
                for m in metrics_dicts:
                    metrics_dicts[m][k].append(v[m])

        plt.switch_backend("PDF")
        fig, axes = plt.subplots(1, len(metrics_dicts), figsize=(10, 4))
        for ax, (metric, data) in zip(axes, metrics_dicts.items()):
            for key, values in data.items():
                ax.plot(values, label=key)
            ax.set_title(metric)
            ax.legend()
        fig.savefig(str(eval_dir / "error_stats.pdf"))
        fig.savefig(str(eval_dir / "error_stats.png"))


if __name__ == "__main__":
    main()
