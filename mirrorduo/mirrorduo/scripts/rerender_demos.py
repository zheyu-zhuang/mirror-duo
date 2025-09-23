import os
import shutil
import json
import h5py
from tqdm import tqdm

from mirrorduo.utils.core_utils import (
    extract_trajectory,
    update_env_meta_action_mode,
    get_action_key,
)

import mimicgen  # noqa
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from mirrorduo.scripts.convert_dataset_actions import convert_actions
import numpy as np
import cv2


class DatasetRerender:
    def __init__(self, dataset_path, output_path, env_name=None, debug=False):
        """
        Initialize the DatasetRerender class.

        Args:
            dataset_path (str): Path to the input dataset.
            output_path (str): Path to the output dataset.
            env_name (str): Name of the environment for rerendering.
        """
        assert dataset_path, "Dataset path must be provided!"
        assert os.path.exists(dataset_path), f"Dataset {dataset_path} not found!"

        self.check_output_destination(output_path)

        self.input_path = dataset_path
        self.output_path = output_path

        # Load environment metadata
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
        env_meta = update_env_meta_action_mode(env_meta, action_mode="absolute")
        # USE ABSOLUTE CONTROL
        print(f"Creating environment from metadata:\n{env_meta}\n")
        print("Using absolute control for rerendering.")

        if env_name is not None:
            env_meta["env_name"] = env_name
        else:
            print("No env name provided, using the original env name from the dataset.")

        self.record_mirrored_demos = True if "Mirror" in env_meta["env_name"] else False

        self.env_meta = env_meta
        #
        output_dir = os.path.dirname(self.output_path)
        self.debug_dir = os.path.join(output_dir, f"rerender_debug_{env_name}")
        if os.path.exists(self.debug_dir):
            shutil.rmtree(self.debug_dir)

        self.env = EnvUtils.create_env_for_data_processing(
            env_meta=self.env_meta,
            camera_names=self.env_meta["env_kwargs"]["camera_names"],
            camera_height=self.env_meta["env_kwargs"]["camera_heights"] if not debug else 256,
            camera_width=self.env_meta["env_kwargs"]["camera_widths"] if not debug else 256,
            reward_shaping=False,
        )
        self.camera_names = self.env_meta["env_kwargs"]["camera_names"]
        self.debug = debug

    def rerender_demonstration(self, h5_file, ep, reset_every_step=True):
        """
        Rerender a single demonstration.

        Args:
            h5_file (h5py.File): HDF5 file object.
            ep (str): Episode key.

        Returns:
            dict: Rerendered observations.
        """
        is_robosuite_env = EnvUtils.is_robosuite_env(self.env_meta)

        states = h5_file[f"data/{ep}/states"][()]

        assert "absolute_actions" in h5_file[f"data/{ep}"], (
            f"Episode {ep} does not contain absolute actions. "
            "Please rerender the dataset with absolute actions."
        )
        actions = h5_file[f"data/{ep}/absolute_actions"][()]
        initial_state = {"states": states[0]}

        if is_robosuite_env:
            initial_state["model"] = h5_file[f"data/{ep}"].attrs["model_file"]

        if self.record_mirrored_demos and reset_every_step:
            reset_every_step = False

        traj, success = extract_trajectory(
            env=self.env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            mirror=self.record_mirrored_demos,
            reset_every_step=reset_every_step,
        )
        return traj, success

    def render(self, num_demos=None):
        data_writer = h5py.File(self.output_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

        demo_counter = 0
        with h5py.File(self.input_path, "r") as h5_file:
            demos = sorted(h5_file["data"].keys(), key=lambda x: int(x[5:]))

            if num_demos is not None:
                demos = demos[:num_demos]

            p_bar = tqdm(demos[:5] if self.debug else demos, desc="Rerendering demonstrations")

            for ep in p_bar:
                if self.debug:
                    # in the parent folder of the output path, save debug images
                    debug_img_dir = os.path.join(self.output_path.replace(".hdf5", ""), "debug_images", ep)
                    os.makedirs(debug_img_dir, exist_ok=True)

                traj, success = self.rerender_demonstration(h5_file, ep)
                postfix_str = "success" if success else "failed, skipping"
                p_bar.set_postfix_str(f"ep {ep}: {postfix_str}")

                if not success and not self.debug:
                    continue  # skip failed demos

                demo_id = f"demo_{demo_counter}"
                ep_data_grp = data_grp.create_group(demo_id)

                converted_actions, robot_eef_rots = convert_actions(
                    self.env, traj["states"], traj["actions"]
                )
                for mode in converted_actions:
                    act_key = get_action_key(mode)
                    ep_data_grp.create_dataset(act_key, data=converted_actions[mode])
                ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
                ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
                ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))

                for k in traj["obs"]:
                    ep_data_grp.create_dataset(f"obs/{k}", data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset(f"next_obs/{k}", data=np.array(traj["next_obs"][k]))
                    if self.debug:
                        if "agentview" in k:
                            imgs = traj["obs"][k]
                            for i in range(imgs.shape[0]):
                                img = imgs[i]
                                img = img.astype(np.uint8)
                                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                img_name = os.path.join(debug_img_dir, f"{ep}_{k}_{i}.png")
                                cv2.imwrite(img_name, img)

                # Episode metadata
                if "model" in traj["initial_state_dict"]:
                    ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]

                ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
                total_samples += traj["actions"].shape[0]
                demo_counter += 1

            data_grp.attrs["env_args"] = json.dumps(self.env.serialize(), indent=4)
            data_grp.attrs["total"] = demo_counter
            print(f"Wrote {demo_counter} successfully rerendered demos to {self.output_path}")

        data_writer.close()

    @staticmethod
    def camera_obs_check(obs_key, camera_names):
        """
        Check if the observation key corresponds to a camera.

        Args:
            obs_key (str): Observation key.
            camera_names (list): List of camera names.

        Returns:
            bool: True if the key corresponds to a camera, False otherwise.
        """
        return any(camera_name in obs_key for camera_name in camera_names)

    @staticmethod
    def add_data_to_new_key(h5_file, ep, obs_key, new_key, data):
        """
        Add data to a new key in the HDF5 file.

        Args:
            h5_file (h5py.File): HDF5 file object.
            ep (str): Episode key.
            obs_key (str): Observation key.
            new_key (str): New key to add data to.
            data (np.ndarray): Data to add.
        """
        group_path = f"data/{ep}/{obs_key}"
        if new_key in h5_file[group_path]:
            del h5_file[group_path][new_key]
        h5_file.create_dataset(f"{group_path}/{new_key}", data=data)

    @staticmethod
    def check_output_destination(output_path):
        """
        Validate the output destination.

        Args:
            output_path (str): Path to the output file.

        Returns:
            bool: True if the destination is valid, False otherwise.
        """
        assert output_path.endswith(".hdf5"), "Output path must end with .hdf5!"

        if os.path.exists(output_path):
            user_input = (
                input(f"File {output_path} already exists. Overwrite? (y/n): ").strip().lower()
            )
            if user_input != "y":
                print("Copy operation canceled.")
                return
            os.remove(output_path)

        parent_dir = os.path.dirname(output_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rerender dataset observations.")
    parser.add_argument("-i", "--dataset", type=str, required=True, help="Path to the dataset.")
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Path to the output dataset."
    )
    parser.add_argument("--env_name", type=str, help="Name of the environment.")
    parser.add_argument("--debug", action="store_true", help="Enable debugging mode.")
    parser.add_argument(
        "-n", "--num_demos", type=int, default=200, help="Number of demos to rerender."
    )

    args = parser.parse_args()

    rerenderer = DatasetRerender(
        dataset_path=args.dataset, output_path=args.output, env_name=args.env_name, debug=args.debug
    )
    rerenderer.render(num_demos=args.num_demos)
