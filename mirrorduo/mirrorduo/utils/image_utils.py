import os

import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageDraw


class ShapeHelper:
    def __init__(self, debug_path=None):
        if debug_path is None:
            self.debug = False
        else:
            debug_save_dir = os.path.join(debug_path, "debug")
            if not os.path.exists(debug_save_dir):
                os.makedirs(debug_save_dir)
            self.debug_save_dir = debug_save_dir
            self.debug = True

        self.reset_meta()

    def reset_meta(self):
        self.meta = {
            "is_set": False,
            "dtype": None,
            "input_order": None,
            "current_order": None,
            "input_ndim": None,
            "current_ndim": None,
            "input_shape": None,
        }

    def set_meta(self, img):
        if self.meta["is_set"]:
            return

        assert isinstance(img, np.ndarray), "Input image must be a numpy array!"
        assert img.ndim in [3, 4], "Input must have 3 or 4 dimensions!"
        channel_order = "bhwc" if img.shape[-1] in [1, 3, 4] else "bchw"
        self.meta.update(
            {
                "is_set": True,
                "input_shape": img.shape,
                "dtype": img.dtype,
                "input_ndim": img.ndim,
                "current_ndim": img.ndim,
                "input_order": channel_order,
                "current_order": channel_order,
            }
        )

    # TODO: add eniops-like order specification, i.e. "from_shape -> to_shape"
    def reorder_channels(self, img, order, backend="torch"):
        assert order in ["bchw", "bhwc"], "Order must be either 'bchw' or 'bhwc'!"
        assert backend in ["torch", "numpy"], "Backend must be either 'torch' or 'numpy'!"
        self.set_meta(img)
        if self.meta["current_ndim"] == 3:
            img = np.expand_dims(img, axis=0)
            self.meta["current_ndim"] = 4
        # Rearrange the channels
        if self.meta["current_order"] != order:
            pattern = f"{' '.join(self.meta['current_order'])} -> {' '.join(order)}"
            img = rearrange(img, pattern)
            self.meta["current_order"] = order
        return torch.tensor(img, dtype=torch.float32) if backend == "torch" else img

    def restore_channel_orders(self, img, backend="numpy"):
        """
        Restore the original shape and dtype of the image.
        """
        assert self.meta["is_set"], "Meta data must be set before restoring the image!"
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        if self.meta["input_order"] != self.meta["current_order"]:
            img = self.reorder_channels(img, self.meta["input_order"], backend=backend)
        if self.meta["input_ndim"] == 3:
            img = np.squeeze(img, axis=0)
        return img.astype(self.meta["dtype"])

    def save_debug_images(self, np_img, img_name):
        if not self.debug:
            return
        assert isinstance(np_img, np.ndarray), "Input image must be a numpy array!"
        self.reset_meta()
        np_img = self.reorder_channels(np_img, "bhwc", backend="numpy")
        for i in range(np_img.shape[0]):
            img_ = np_img[i]
            # Handle grayscale images (H, W, 1) by squeezing the channel dimension
            if img_.shape[-1] == 1:
                img_ = img_.squeeze(-1)
            # Scale the image to 0-255 if it's in a 0-1 range
            img_ = img_ * 255 if np.amax(img_) <= 1 else img_
            img_ = Image.fromarray(img_.astype(np.uint8))
            img_.save(f"{self.debug_save_dir}/{img_name}_{i}.png")


def get_tcp_coords(camera_matrix, eef_pos, eef_rot):
    # TODO: add support to different grippers
    assert isinstance(eef_pos, np.ndarray), "End-effector pose must be a numpy array!"
    assert isinstance(eef_rot, np.ndarray), "End-effector rotations must be a numpy array!"
    has_batch_dim = eef_pos.ndim == 2
    eef_pos = np.expand_dims(eef_pos, axis=0) if not has_batch_dim else eef_pos
    eef_rot = np.expand_dims(eef_rot, axis=0) if not has_batch_dim else eef_rot
    X = np.eye(4).reshape(1, 4, 4).repeat(eef_pos.shape[0], axis=0)
    X[:, :3, :3] = eef_rot.reshape(-1, 3, 3)
    X[:, :3, 3] = eef_pos
    tcp_coord = world_pose_to_camera_coords(X, camera_matrix)  # [B, 2]
    # normalize the coordinates
    return tcp_coord


def tcp_center_hori_rolled(
    images: np.ndarray,
    coords: np.ndarray,
    overlay_watermark=True,
    img_prefix=None,
    debug_path=None,
) -> np.ndarray:
    """
    Horizontally shifts a batch of images such that the given coordinates are at the center of each image.

    Args:
        images (np.ndarray): Batch of images of shape (B, H, W, C) or (B, H, W) (grayscale).
        coords (np.ndarray): Array of shape (B, 2), where each row is (x, y) for the coordinate to center.

    Returns:
        np.ndarray: Batch of shifted images with the same shape as the input.
    """
    assert isinstance(images, np.ndarray), "Images must be a numpy array."
    assert isinstance(coords, np.ndarray), "Coordinates must be a numpy array."

    img_helper = ShapeHelper(debug_path=debug_path)
    images = img_helper.reorder_channels(images, "bhwc", backend="numpy")
    coords = coords.reshape(-1, 2)

    assert images.shape[0] == coords.shape[0], "Number of images and coordinates must match."

    B, H, W = images.shape[:3]  # Batch size, height, width
    shifted_images = np.zeros_like(images)  # Initialize output array
    masks = np.ones_like(images)  # Initialize masks

    # Calculate horizontal shifts for each image
    x_shifts = W // 2 - coords[:, 0].astype(np.int32)

    mask_colour_filter = np.array([0, 1, 0], dtype=np.float32)  # Green color for mask

    for i in range(B):
        shifted_images[i] = np.roll(images[i], shift=x_shifts[i], axis=1)
        # Generate mask for rolled portions
        if x_shifts[i] > 0:
            # Green channel for left rolled portion
            masks[i, :, : x_shifts[i], :] = mask_colour_filter  # Green color
        elif x_shifts[i] < 0:
            # Green channel for right rolled portion
            masks[i, :, x_shifts[i] :, :] = mask_colour_filter

    if overlay_watermark:
        shifted_images = shifted_images * masks

    img_helper.save_debug_images(shifted_images, img_name=f"{img_prefix}_rolled_shift")

    shifted_images = img_helper.restore_channel_orders(shifted_images, backend="numpy")

    return shifted_images


def draw_arrow(draw, start, end, color, width=1):
    """
    Draw an arrow from start to end.
    """
    arrow_length = 10  # Length of the arrow head
    arrow_width = 5  # Width of the arrow head

    # Calculate the direction of the arrow
    direction = np.array(end) - np.array(start)
    length = np.linalg.norm(direction)
    if length == 0:
        return
    direction = direction / length

    # Calculate the points for the arrow head
    left = np.array([-direction[1], direction[0]]) * arrow_width
    right = np.array([direction[1], -direction[0]]) * arrow_width
    head_base = np.array(end) - direction * arrow_length

    arrow_head = [tuple(end), tuple(head_base + left), tuple(head_base + right)]

    # Draw the arrow line
    draw.line([tuple(start), tuple(head_base)], fill=color, width=width)
    # Draw the arrow head
    draw.polygon(arrow_head, fill=color)


def tcp_overlay(img, tcp_coord, lf_coord, rf_coord, width, overlay_only=False):
    img = img.copy()
    # TODO: add noise to the overlay
    img_helper = ShapeHelper()
    img = img_helper.reorder_channels(img, "bhwc", backend="numpy")
    max_range = 1 if np.amax(img) <= 1 else 255
    img = img * 255 if max_range == 1 else img
    overlayed_imgs = []
    for i in range(img.shape[0]):
        # draw a line from the end-effector to the gripper joints
        if not overlay_only:
            im_ = Image.fromarray(img[i].astype(np.uint8))
            draw = ImageDraw.Draw(im_)
        else:
            im_ = Image.fromarray(np.zeros_like(img[i]).astype(np.uint8))
            draw = ImageDraw.Draw(im_)
        tcp = (int(tcp_coord[i][0]), int(tcp_coord[i][1]))
        lf = (int(lf_coord[i][0]), int(lf_coord[i][1]))
        rf = (int(rf_coord[i][0]), int(rf_coord[i][1]))
        # draw a line between the tps
        draw.line([lf, rf], fill=(255, 220, 10), width=width)
        # draw dots at the tps and tcp
        r = width
        draw.ellipse([lf[0] - r, lf[1] - r, lf[0] + r, lf[1] + r], fill=(0, 255, 0))
        draw.ellipse([rf[0] - r, rf[1] - r, rf[0] + r, rf[1] + r], fill=(0, 255, 0))
        draw.ellipse([tcp[0] - r, tcp[1] - r, tcp[0] + r, tcp[1] + r], fill=(0, 0, 255))
        overlayed_imgs.append(np.array(im_))
    overlayed_imgs = np.stack(overlayed_imgs, axis=0)
    overlayed_imgs = overlayed_imgs / 255 if max_range == 1 else overlayed_imgs
    overlayed_imgs = img_helper.restore_channel_orders(overlayed_imgs, backend="numpy")
    return overlayed_imgs


def world_pose_to_camera_coords(SE3, camera_matrix):
    assert isinstance(SE3, np.ndarray), "Pose matrix must be a numpy array!"
    assert isinstance(camera_matrix, np.ndarray), "Camera matrix must be a numpy array!"
    has_batch_dim = SE3.ndim == 3
    SE3 = SE3.reshape(-1, 4, 4) if not has_batch_dim else SE3
    # to homogeneous coordinates
    pos = np.concatenate([SE3[:, :3, 3], np.ones((SE3.shape[0], 1))], axis=1)  # [B, 4]
    # Transform end-effector poses to image coordinates
    coord = np.matmul(pos, camera_matrix.T)  # [B, 4]
    coord = coord[:, :2] / coord[:, 2:3]
    return coord


def world_point_to_camera_coords(pt, camera_matrix):
    assert isinstance(pt, np.ndarray), "Pose matrix must be a numpy array!"
    assert isinstance(camera_matrix, np.ndarray), "Camera matrix must be a numpy array!"
    has_batch_dim = pt.ndim == 2 and pt.shape[-1] > 1
    pt = np.expand_dims(pt, axis=0) if not has_batch_dim else pt
    # to homogeneous coordinates
    if pt.shape[1] == 3:
        pos = np.concatenate([pt, np.ones((pt.shape[0], 1))], axis=1)
    elif pt.shape[1] == 4 and np.all(pt[:, 3] == 1):
        pos = pt
    else:
        raise ValueError("Point must be in 3D or 4D homogeneous coordinates!")
    # Transform end-effector poses to image coordinates
    coord = np.matmul(pos, camera_matrix.T)  # [B, 4]
    coord = coord[:, :2] / coord[:, 2:3]
    return coord if has_batch_dim else coord[0]
