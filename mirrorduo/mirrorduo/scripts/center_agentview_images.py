import os

import h5py
import mimicgen  # noqa
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.scripts.dataset_states_to_obs import extract_trajectory
from robosuite.utils.camera_utils import get_camera_transform_matrix
from tqdm import tqdm

from mirrorduo.utils import image_utils as ImageUtils, HOME_X



class CenterAgentviewImages:
    def __init__(self, dataset_path, debug_path=None):
        """
        Args:
            dataset_path (str): Path to the dataset
            crop_size (int): Size of the crop
            resize_size (int): Size to resize the image to
            render_size (int): Size to increase the resolution to before cropping, for higher quality crops
            keys_for_cropping (list): List of keys to crop
            gripper_overlay (bool): Flag to enable gripper overlay on the images
            proprio_in_which_cam (str): The camera name in which the proprio states are expressed
        """

        assert dataset_path is not None, "Dataset path must be provided!"
        assert os.path.exists(dataset_path), f"Dataset {dataset_path} not found!"

        self.dataset_path = dataset_path
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)

        print(f"Creating environment from metadata:\n {env_meta} \n")
        self.env_meta = env_meta
        self.env = EnvUtils.create_env_for_data_processing(
            env_meta=self.env_meta,
            camera_names=self.env_meta["env_kwargs"]["camera_names"],
            camera_height=env_meta["env_kwargs"]["camera_heights"],
            camera_width=env_meta["env_kwargs"]["camera_widths"],
            reward_shaping=False,
        )
        self.camera_names = self.env_meta["env_kwargs"]["camera_names"]
        self.camera_h = self.env_meta["env_kwargs"]["camera_heights"]
        self.camera_w = self.env_meta["env_kwargs"]["camera_widths"]
        self.debug_path = debug_path
        self.home_pose = np.array(HOME_X["robomimic"])

    def rerender_demonstration(self, h5_file, ep):
        is_robosuite_env = EnvUtils.is_robosuite_env(self.env_meta)
        states = h5_file["data/{}/states".format(ep)][()]
        actions = h5_file["data/{}/actions".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = h5_file["data/{}".format(ep)].attrs["model_file"]
        traj, camera_info = extract_trajectory(
            env=self.env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            done_mode=0,
        )
        return traj["obs"]

    def center_agnetview(self, num_demos=None, mode="per_traj"):
        assert mode in ["per_traj", "per_frame"], f"mode {mode} not supported"
        with h5py.File(self.dataset_path, "a") as h5_file:
            demos = list(h5_file["data"].keys())
            inds = np.argsort([int(elem[5:]) for elem in demos])
            if num_demos is not None and num_demos < len(inds):
                inds = inds[:num_demos]
            demos = [demos[i] for i in inds]
            debug_mode = self.debug_path is not None
            p_bar = tqdm(range(min(10, len(demos)))) if debug_mode else tqdm(range(len(demos)))

            camera_matrix = get_camera_transform_matrix(
                self.env.env.sim, "agentview", self.camera_h, self.camera_w
            )  # [4, 4]
            for ind in p_bar:
                p_bar.set_description(f"Convert RGB for {demos[ind]}")
                ep = demos[ind]
                # prepare initial state to reload from
                obs = h5_file[f"data/{ep}/obs"]
                assert "agentview_image" in obs.keys(), f"agentview_image not found in {ep}"
                images = obs["agentview_image"][:]
                self.add_data_to_new_key(
                    h5_file, ep, "obs", "agentview_camera_matrix", camera_matrix
                )
                eef_pos = obs["robot0_eef_pos"][:]
                eef_rot = obs["robot0_eef_rot"][:]
                tcp_coord = ImageUtils.get_tcp_coords(
                    camera_matrix=camera_matrix, eef_pos=eef_pos, eef_rot=eef_rot
                )
                if mode == "per_traj":
                    tcp_coord = ImageUtils.get_tcp_coords(
                        camera_matrix=camera_matrix, eef_pos=self.home_pose[:3, 3], eef_rot=self.home_pose[:3, :3]
                    )
                    n_frames = images.shape[0]
                    tcp_coord = np.tile(tcp_coord, (n_frames, 1))
                centered_imgs = ImageUtils.tcp_center_hori_rolled(
                    images,
                    tcp_coord,
                    img_prefix=f"agentview_image_tcp_centered_{ep}",
                    debug_path=self.debug_path,
                )
                self.add_data_to_new_key(
                    h5_file, ep, "obs", "agentview_image_tcp_centered", centered_imgs
                )
            h5_file.close()

    @staticmethod
    def add_data_to_new_key(h5_file, ep, obs_key, new_key, data):
        """
        write to /data/ep/obs_key/new_key
        """
        if new_key in h5_file[f"data/{ep}/{obs_key}"].keys():
            del h5_file[f"data/{ep}/{obs_key}/{new_key}"]
        h5_file.create_dataset(f"data/{ep}/{obs_key}/{new_key}", data=data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Path to the dataset",
    )

    parser.add_argument(
        "-m",
        "--mode",
        default="per_traj",
        type=str,
        choices=["per_traj", "per_frame"],
        help="center the image per trajectory or per frame",
    )

    parser.add_argument(
        "--debug_path",
        type=str,
        default=None,
        help="Path to the debug dataset",
    )

    parser.add_argument(
        "-n",
        "--num_demos",
        type=int,
        default=None,
        help="Number of demos to process",
    )

    args = parser.parse_args()

    helper = CenterAgentviewImages(dataset_path=args.dataset, debug_path=args.debug_path)

    helper.center_agnetview(args.num_demos, args.mode)
