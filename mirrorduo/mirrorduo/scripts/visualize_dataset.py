import pathlib

import cv2
import h5py
import mimicgen  # noqa
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
from tqdm import tqdm

from mirrorduo.utils.core_utils import (
    VALID_ACTION_MODES,
    get_action_key,
    update_env_meta_action_mode,
)
from mirrorduo.utils.junk_utils import divider, plain_divider, print_dict
import numpy as np


class DatasetReplayer:
    def __init__(self, dataset_path, action_mode, start_from=0):
        assert action_mode in VALID_ACTION_MODES + [
            "default"
        ], f"Invalid action mode: {action_mode}. Valid modes are: {VALID_ACTION_MODES}"
        # default BC config
        config = config_factory(algo_name="bc")

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        # must ran before create dataset
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

        env_meta = update_env_meta_action_mode(env_meta=env_meta, action_mode=action_mode)

        divider("Controller Configs", char="=")
        print_dict(env_meta["env_kwargs"]["controller_configs"])
        plain_divider(char="=")

        env = EnvUtils.create_env_for_data_processing(
            env_meta=env_meta,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_height=256,
            camera_width=256,
            reward_shaping=False,
        )

        self.env = env
        self.action_mode = action_mode
        self.h5_file = h5py.File(dataset_path, "r")
        self.start_from = start_from

    def __len__(self):
        return len(self.file["data"])

    def replay(self, reset_per_step=True):
        env = self.env
        h5_file = self.h5_file
        # first step have high error for some reason, not representative

        num_demos = len(h5_file["data"])

        for idx in tqdm(range(self.start_from, num_demos), desc="Replaying Demos"):
            # constantly monitor keyboard input to skip to next demo
            demo = h5_file[f"data/demo_{idx}"]
            model_file = h5_file[f"data/demo_{idx}"].attrs["model_file"]
            # input
            states = demo["states"][:]
            actions = demo[get_action_key(self.action_mode)][:]
            env.reset_to({"model": model_file, "states": states[0]})
            for i in range(len(states)):
                # if reset_per_step:
                # obs, reward, done, info = env.step(actions[i])
                # obs = env.get_observation()
                demo_debug_dir = f'debug_three_piece/demo_{idx}'
                if not pathlib.Path(demo_debug_dir).exists():
                    pathlib.Path(demo_debug_dir).mkdir(parents=True, exist_ok=True)
                if i % 1 == 0:
                    obs = env.reset_to({"states": states[i]})
                    im = obs["agentview_image"]
                    im_eih = obs['robot0_eye_in_hand_image']
                    im_eih = cv2.cvtColor(im_eih, cv2.COLOR_RGB2BGR)
                    im_eih_mirrored = im_eih[:, ::-1, :]
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                    im_mirrored = im[:, ::-1, :]
                    im = np.concatenate([im, im_eih], axis=1)
                    im_mirrored = np.concatenate([im_mirrored, im_eih_mirrored], axis=1)
                    cv2.imwrite(f"{demo_debug_dir}/stack_three_demo_{idx}_step_{i}.png", im)
                    cv2.imwrite(f"{demo_debug_dir}/mirrored_stack_three_demo_{idx}_step_{i}.png", im_mirrored)
                    cv2.imshow("image", im)
                    cv2.waitKey(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset_path", type=str, required=True, help="Path to the dataset to replay"
    )
    parser.add_argument(
        "-a", "--action_mode", type=str, default="default", help="Action mode for the dataset"
    )
    parser.add_argument(
        "-r", "--reset_per_step", action="store_true", help="Reset the environment at each step"
    )
    parser.add_argument(
        "-s", "--start_from", type=int, default=0, help="Start replaying from this demo index"
    )
    args = parser.parse_args()
    dataset_path = args.dataset_path
    action_mode = args.action_mode

    # Check if the dataset path exists
    if not pathlib.Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    replayer = DatasetReplayer(dataset_path, action_mode=args.action_mode, start_from=args.start_from)
    replayer.replay(reset_per_step=args.reset_per_step)
