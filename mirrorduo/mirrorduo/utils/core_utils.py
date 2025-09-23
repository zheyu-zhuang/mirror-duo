from copy import deepcopy

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
from einops import rearrange
from robomimic.envs.env_base import EnvBase
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
from scipy.spatial.transform import Rotation as R

import mirrorduo.utils.image_utils as ImageUtils

# HARDCODE: hardcoded home pose for robomimic
from mirrorduo.utils import HOME_X
import cv2

# ====================================================================================
# core_utils.py - MirrorDuo Core Utility Functions
# ====================================================================================
#
# 1. Constants and Imports
#
# 2. Visual Utilities
#     - center_agentview
#     - camera_pose_from_xml
#
# 3. Pose Centering Utilities
#     - get_delta_rotations
#     - get_delta_positions
#
# 4. Action Format & Conversion
#     - update_env_meta_action_mode
#     - get_action_key
#     - delta_to_absolute_actions
#
# 5. Mirroring Utilities
#     - mirror_object
#     - mirror_object_placements
#     - mirror_actions
#
# 6. Trajectory Extraction
#     - extract_mirrored_trajectory
#
# ====================================================================================


VALID_ACTION_MODES = ["absolute", "relative", "delta"]


# ------------------------------------------------------------------------------------ #
# Visual Utilities
# ------------------------------------------------------------------------------------ #


def tcp_center_agentview(obs, camera_matrix, fixed_init=False):
    agentview_images = obs["agentview_image"]
    b, t, c, h, w = agentview_images.shape
    assert c == 3, f"agentview_image should have 3 channels, but got {c}"

    if fixed_init is not None:
        eef_pos = np.array(HOME_X["robomimic"])[:3, 3]
        eef_rot = np.array(HOME_X["robomimic"])[:3, :3]
        eef_pos = np.tile(eef_pos, (b * t, 1))
        eef_rot = np.tile(eef_rot, (b * t, 1, 1))
    else:
        eef_pos = rearrange(obs["robot0_eef_pos"], "b t d -> (b t) d")
        eef_rot = rearrange(obs["robot0_eef_rot"], "b t d -> (b t) d")
    images = rearrange(agentview_images, "b t c h w -> (b t) h w c")

    tcp_coord = ImageUtils.get_tcp_coords(
        camera_matrix=camera_matrix, eef_pos=eef_pos, eef_rot=eef_rot
    )
    centered_imgs = ImageUtils.tcp_center_hori_rolled(images, tcp_coord)

    obs["agentview_image_tcp_centered"] = rearrange(
        centered_imgs, "(b t) h w c -> b t c h w", b=b, t=t, h=h, w=w
    )

    del obs["agentview_image"]


def camera_pose_from_xml(pos, quat):
    X_C = np.eye(4)
    obj_quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
    X_C[:3, :3] = R.from_quat(obj_quat_xyzw).as_matrix()
    X_C[:3, 3] = pos
    camera_axis_correction = np.diag([1, -1, -1, 1])
    return X_C @ camera_axis_correction


# ------------------------------------------------------------------------------------ #
# Delta Pose Utilities
# ------------------------------------------------------------------------------------ #


def get_delta_rotations(rot: np.ndarray, return_6d=False) -> np.ndarray:
    """
    Centeralize the poses to the home position of the robot,
    i.e. express the poses in the robot's home frame.
    (home frame is defined in mirrorduo/__init__.py).
    """
    assert rot.ndim == 2, "rot should be 2D"
    assert rot.shape[1] == 9, "rot should be of shape (N, 9)"

    rot_mat = rot.reshape(-1, 3, 3)

    X0 = np.array(HOME_X["robomimic"], dtype=np.float32)
    R0 = X0[:3, :3]
    centered_rot = R0.T @ rot_mat
    centered_rot = centered_rot.reshape(-1, 9).astype(np.float32)
    if return_6d:
        return centered_rot[:, :6]
    return centered_rot


def get_delta_positions(pos: np.ndarray) -> np.ndarray:
    """
    Centeralize the poses to the home position of the robot,
    i.e. express the poses in the robot's home frame.
    (home frame is defined in mirrorduo/__init__.py).
    """
    assert pos.ndim == 2, "pos should be 2D"
    assert pos.shape[1] == 3, "pos should be of shape (N, 3)"

    X0 = np.array(HOME_X["robomimic"], dtype=np.float32)
    X0_inv = np.linalg.inv(X0)
    R0_inv = X0_inv[:3, :3]
    t0_inv = X0_inv[:3, 3]

    out = R0_inv @ pos.reshape(-1, 3, 1) + t0_inv.reshape(1, 3, 1)
    return out.reshape(-1, 3).astype(np.float32)


def goal_poses_to_delta_actions(goal_rot, goal_pos):
    assert goal_rot.shape == (3, 3)
    assert goal_pos.shape == (3,)
    X0 = np.array(HOME_X["robomimic"], dtype=np.float32)
    R0 = X0[:3, :3]
    t0 = X0[:3, 3]
    delta_rot = R0.T @ goal_rot
    delta_pos = R0.T @ (goal_pos - t0)
    return delta_rot, delta_pos


def delta_actions_to_goal_poses(delta_rot, delta_pos):
    assert delta_rot.shape == (3, 3)
    assert delta_pos.shape == (3,)
    X0 = np.array(HOME_X["robomimic"], dtype=np.float32)
    R0 = X0[:3, :3]
    t0 = X0[:3, 3]
    goal_rot = R0 @ delta_rot
    goal_pos = R0 @ delta_pos + t0
    return goal_rot, goal_pos


# ------------------------------------------------------------------------------------ #
# Action Format & Conversion
# ------------------------------------------------------------------------------------ #


def update_env_meta_action_mode(env_meta, action_mode):
    if action_mode == "default":
        return env_meta
    env_meta = deepcopy(env_meta)  # Avoid modifying the original env_meta
    assert action_mode in VALID_ACTION_MODES, f"Invalid action mode: {action_mode}"
    control_modes = {
        "absolute": {"control_delta": False, "true_relative": False},
        "delta": {"control_delta": False, "true_relative": False, "true_delta": True},
        "relative": {"control_delta": True, "true_relative": True},
    }
    env_meta["env_kwargs"]["controller_configs"].update(control_modes[action_mode])
    return env_meta


def get_action_key(action_mode, original_key="actions"):
    if action_mode == "all":
        return [f"{mode}_{original_key}" for mode in VALID_ACTION_MODES] + [original_key]
    if action_mode == "default":
        return original_key
    assert action_mode in VALID_ACTION_MODES
    return f"{action_mode}_{original_key}"


# ------------------------------------------------------------------------------------ #
# Mirroring Utilities
# ------------------------------------------------------------------------------------ #


def mirror_object(X_C, obj_pos, obj_quat, rot_180=False):
    C_X = np.linalg.inv(X_C)
    E = np.diag([-1, 1, 1, 1])

    X_G = np.eye(4)
    if obj_quat is not None:
        quat_xyzw = [obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]
        X_G[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    X_G[:3, 3] = obj_pos

    C_X_G = C_X @ X_G
    mirrored = E @ C_X_G @ E
    X_G_mirror = X_C @ mirrored

    obj_pos = X_G_mirror[:3, 3]
    if obj_quat is None:
        return obj_pos, None

    Rz = R.from_rotvec([0, 0, np.pi]).as_matrix() if rot_180 else np.eye(3)
    new_quat = R.from_matrix(X_G_mirror[:3, :3] @ Rz).as_quat()
    return obj_pos, [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]


def mirror_object_placements(env, object_placements, mirror=False):
    if not mirror:
        return
    X_C = env.get_agentview_extrincs()
    for obj_pos, obj_quat, obj in object_placements.values():
        if mirror:
            obj_pos, obj_quat = mirror_object(X_C, obj_pos, obj_quat)
        env.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([obj_pos, obj_quat]))


def mirror_actions(actions: np.ndarray, action_mode, X_C=None) -> np.ndarray:
    assert isinstance(actions, np.ndarray)
    if actions.ndim == 3:
        raise ValueError("Actions should be 2D or 1D, not 3D, reshape it before mirroring.")
    actions = actions.reshape(-1, actions.shape[-1])

    B = actions.shape[0]
    E = np.diag([-1, 1, 1, 1])

    if action_mode == "relative":
        delta_X = np.tile(np.eye(4), (B, 1, 1))
        delta_X[:, :3, 3] = actions[:, :3]
        delta_X[:, :3, :3] = R.from_rotvec(actions[:, 3:6]).as_matrix()
        mirrored = E @ delta_X @ E
        actions[:, :3] = mirrored[:, :3, 3]
        actions[:, 3:6] = R.from_matrix(mirrored[:, :3, :3]).as_rotvec()
    elif action_mode == "absolute":
        assert X_C is not None, "X_C is required for absolute mirroring"
        X_H = np.tile(np.eye(4), (B, 1, 1))
        X_H[:, :3, 3] = actions[:, :3]
        X_H[:, :3, :3] = R.from_rotvec(actions[:, 3:6]).as_matrix()
        C_X = np.linalg.inv(X_C)
        mirrored = X_C @ (E @ (C_X @ X_H) @ E)
        actions[:, :3] = mirrored[:, :3, 3]
        actions[:, 3:6] = R.from_matrix(mirrored[:, :3, :3]).as_rotvec()
    elif action_mode == "delta":
        raise NotImplementedError("Delta action mirroring is not implemented.")
    else:
        raise ValueError("Unsupported action format or action mode.")
    return actions[0] if B == 1 else actions


# ------------------------------------------------------------------------------------ #
# Trajectory Extraction
# ------------------------------------------------------------------------------------ #


def extract_trajectory(env, initial_state, states, actions, mirror=False, reset_every_step=False):
    assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0]

    env.reset()
    initial_state["model"] = env.env.sim.model.get_xml()
    env.reset_to(initial_state)

    if mirror:
        try:
            X_C = env.env.get_agentview_extrincs()
        except AttributeError:
            raise ValueError("env does not have get_agentview_extrincs method")
        actions = mirror_actions(actions, X_C=X_C, action_mode="absolute")
        env.env.mirror_objects()
        env.env.sim.forward()

    obs = env.get_observation()
    state_dict = env.get_state()

    traj = dict(
        obs=[],
        next_obs=[],
        rewards=[],
        dones=[],
        states=[],
        actions=actions,
        initial_state_dict=state_dict,
    )

    for t in range(1, states.shape[0] + 1):
        if reset_every_step and t < states.shape[0]:
            next_obs = env.reset_to({"states": states[t]})
        else:
            next_obs, _, _, _ = env.step(actions[t - 1])
        # im_viz = obs["agentview_image"]
        # if im_viz is not None:
        #     im_viz = cv2.cvtColor(im_viz, cv2.COLOR_RGB2BGR)
        #     cv2.imshow("agentview", im_viz)
        #     cv2.waitKey(1)
        r = env.get_reward()
        done = env.is_success()["task"]
        done = int(done)

        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["states"].append(state_dict["states"])

        obs = deepcopy(next_obs)
        state_dict = env.get_state()

    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for sub_k in traj[k]:
                traj[k][sub_k] = np.array(traj[k][sub_k])
        else:
            traj[k] = np.array(traj[k])

    return traj, done


def test():
    from scipy.spatial.transform import Rotation as R

    B = 30

    X_H = np.tile(np.eye(4), (B, 1, 1))
    X_H[:, :3, 3] = np.random.rand(B, 3)
    X_H[:, :3, :3] = R.random(B).as_matrix()

    W_X = np.array(HOME_X["robomimic"], dtype=np.float32)

    delta_X = np.linalg.inv(W_X) @ X_H

    for i in range(B):
        a = delta_X[i]
        a_prim = np.linalg.inv(W_X) @ X_H[i]
        assert np.allclose(a, a_prim), f"Failed for {i}"

    rot = X_H[:, :3, :3].reshape(B, 9)

    delta_rot = get_delta_rotations(rot)
    delta_pos = get_delta_positions(X_H[:, :3, 3])
    assert np.allclose(delta_rot.reshape(-1, 3, 3), delta_X[:, :3, :3]), "Rotation mismatch"
    assert np.allclose(delta_pos, delta_X[:, :3, 3]), "Position mismatch"


if __name__ == "__main__":
    test()
