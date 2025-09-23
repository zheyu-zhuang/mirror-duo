# Standard library
import collections
import math
import os
import pathlib
from typing import List

import dill
import h5py
import mimicgen  # noqa
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import torch
import tqdm
import wandb
import wandb.sdk.data_types.video as wv
from einops import rearrange
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.env_runner.base_image_runner import BaseImageRunner
from equi_diffpo.gym_util.async_vector_env import AsyncVectorEnv
from equi_diffpo.gym_util.multistep_wrapper import MultiStepWrapper
from equi_diffpo.gym_util.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from equi_diffpo.model.common.rotation_transformer import RotationTransformer
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from robosuite.utils.camera_utils import get_camera_transform_matrix

from mirrorduo.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from mirrorduo.utils.core_utils import (
    get_delta_positions,
    get_delta_rotations,
    tcp_center_agentview,
    update_env_meta_action_mode,
)
from mirrorduo.utils.junk_utils import find_all_keys


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    return env


class MirrorDuoRobomimicImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(
        self,
        output_dir,
        dataset_path,
        shape_meta: dict,
        action_mode: str,
        n_train=10,
        n_train_vis=3,
        train_start_idx=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        render_obs_key="agentview_image",
        fps=10,
        crf=22,
        past_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
        env_name=None,
    ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        env_meta = update_env_meta_action_mode(env_meta, action_mode)

        # disable object state observation
        env_meta["env_kwargs"]["use_object_obs"] = False
        if env_name is not None:
            env_meta["env_name"] = env_name

        self.camera_h = env_meta["env_kwargs"]["camera_heights"]
        self.camera_w = env_meta["env_kwargs"]["camera_widths"]

        rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

        self.center_agentview = False

        default_shape_meta = {
            "obs": {
                "agentview_image": {
                    "shape": [3, 84, 84],
                    "type": "rgb",
                },
                "robot0_eye_in_hand_image": {
                    "shape": [3, 84, 84],
                    "type": "rgb",
                },
                "robot0_eef_pos": {
                    "shape": [3],
                },
                "robot0_eef_rot": {
                    "shape": [9],
                },
                "robot0_gripper_qpos": {
                    "shape": [2],
                },
            },
            "action": {
                "shape": [10],
            },
        }
        # change tcp_centered to agentview_image for correct rendering

        env_temp = create_env(env_meta=env_meta, shape_meta=default_shape_meta)
        self.camera_matrix = get_camera_transform_matrix(
            env_temp.env.sim, "agentview", self.camera_h, self.camera_w
        )
        del env_temp

        def env_fn():
            robomimic_env = create_env(env_meta=env_meta, shape_meta=default_shape_meta)
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=default_shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, shape_meta=default_shape_meta, enable_render=False
            )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=default_shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, "r") as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f"data/demo_{train_idx}/states"][0]

                def init_fn(env, init_state=init_state, enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            "media", wv.util.generate_id() + ".mp4"
                        )
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append("train/")
                env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.tqdm_interval_sec = tqdm_interval_sec
        self.max_rewards = {}
        self.shape_meta = shape_meta
        self.default_shape_meta = default_shape_meta
        for prefix in self.env_prefixs:
            self.max_rewards[prefix] = 0

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta["env_name"]
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {env_name} Image {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            done = False
            while not done:
                self.preprocess_obs(obs)

                # create obs
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict["past_action"] = past_action[:, -(self.n_obs_steps - 1) :].astype(
                        np.float32
                    )

                # device transfer
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())

                action = np_action_dict["action"]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                # step env
                env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[this_local_slice]
        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f"sim_max_reward_{seed}"] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f"sim_video_{seed}"] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix + "mean_score"
            value = np.mean(value)
            log_data[name] = value
            self.max_rewards[prefix] = max(self.max_rewards[prefix], value)
            log_data[prefix + "max_score"] = self.max_rewards[prefix]

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

    def preprocess_obs(self, obs):
        """
        Preprocess observations by applying delta transformations to end-effector or centering the views.
        """
        shape_keys = self.shape_meta["obs"].keys()
        default_shape_keys = self.default_shape_meta["obs"].keys()
        # Center the views first, based on proprioceptive states before being centered !!!
        tcp_center_key = find_all_keys("tcp_centered", shape_keys, only_one_match=True)
        if tcp_center_key is not None:
            tcp_center_agentview(obs, self.camera_matrix, fixed_init=True)

        delta_pos_key = find_all_keys("delta_pos", shape_keys, only_one_match=True)
        pos_key = find_all_keys("pos", default_shape_keys, only_one_match=True, reject=["delta"])
        if delta_pos_key is not None:
            obs[delta_pos_key] = self._reshape_apply(get_delta_positions, obs[pos_key])
            del obs[pos_key]

        delta_rot_key = find_all_keys("delta_rot", shape_keys, only_one_match=True)
        rot_key = find_all_keys("rot", default_shape_keys, only_one_match=True, reject=["delta"])

        if delta_rot_key is not None:
            obs[delta_rot_key] = self._reshape_apply(get_delta_rotations, obs[rot_key])
            del obs[rot_key]
            rot_key = delta_rot_key
        is_6d = self.shape_meta["obs"][rot_key]["shape"] == [6]
        # not the first 2 cols, but the first 2 rows of the 3x3 matrix
        obs[rot_key] = obs[rot_key][..., :6] if is_6d else obs[rot_key]

    @staticmethod
    def _reshape_apply(func, inputs: List):
        """
        Reshape inputs from (B, T, D) to (B*T, D), apply func, then reshape back to (B, T, D).
        """
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]

        B, T = inputs[0].shape[:2]
        reshaped = [rearrange(x, "b t d -> (b t) d") for x in inputs]

        outputs = func(*reshaped)
        outputs = [outputs] if isinstance(outputs, np.ndarray) else list(outputs)
        reshaped_outputs = [rearrange(x, "(b t) d -> b t d", b=B, t=T) for x in outputs]

        return reshaped_outputs[0] if len(reshaped_outputs) == 1 else reshaped_outputs
