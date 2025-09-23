from collections import OrderedDict

import equi_diffpo.model.vision.crop_randomizer as dmvc
import torch
from einops import rearrange
from equi_diffpo.model.common.module_attr_mixin import ModuleAttrMixin
from equi_diffpo.model.equi.equi_encoder import EquiResBlock
from escnn import gspaces, nn
from escnn.group import dihedral_group

from mirrorduo.utils.junk_utils import find_all_keys


class MirrorEquivariantResEncoder76(torch.nn.Module):
    def __init__(
        self,
        obs_channel: int = 3,
        n_out: int = 128,
        initialize: bool = True,
        return_torch_tensor=False,
    ):
        super().__init__()
        self.obs_channel = obs_channel
        self.n_out = n_out
        self.group = gspaces.flip2dOnR2()
        self.return_torch_tensor = return_torch_tensor
        self.conv = torch.nn.Sequential(
            # 76x76
            nn.R2Conv(
                nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                kernel_size=5,
                padding=0,
                initialize=initialize,
            ),
            # 72x72
            nn.ReLU(
                nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True
            ),
            EquiResBlock(self.group, n_out // 8, n_out // 8, initialize=True),
            EquiResBlock(self.group, n_out // 8, n_out // 8, initialize=True),
            nn.PointwiseMaxPool(
                nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2
            ),
            # 36x36
            EquiResBlock(self.group, n_out // 8, n_out // 4, initialize=True),
            EquiResBlock(self.group, n_out // 4, n_out // 4, initialize=True),
            nn.PointwiseMaxPool(
                nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), 2
            ),
            # 18x18
            EquiResBlock(self.group, n_out // 4, n_out // 2, initialize=True),
            EquiResBlock(self.group, n_out // 2, n_out // 2, initialize=True),
            nn.PointwiseMaxPool(
                nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2
            ),
            # 9x9
            EquiResBlock(self.group, n_out // 2, n_out, initialize=True),
            EquiResBlock(self.group, n_out, n_out, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out * [self.group.regular_repr]), 3),
            # 3x3
            nn.R2Conv(
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                kernel_size=3,
                padding=0,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, x) -> nn.GeometricTensor:
        B = x.shape[0]
        assert len(x.shape) == 4, f"Input shape should be (B, C, H, W), but got {x.shape}"
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(
                x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr])
            )
        return self.conv(x).tensor.reshape(B, -1)  # B, n_out

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert len(input_shape) == 3

        return [self.n_out * 2]


class MirrorDuoObsEnc(ModuleAttrMixin):
    def __init__(
        self,
        obs_shape=(3, 84, 84),
        crop_shape=(76, 76),
        n_hidden=128,
        initialize=True,
    ):
        super().__init__()

        self.crop_randomizer = dmvc.CropRandomizer(
            input_shape=obs_shape,
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )

        obs_channel = obs_shape[0]
        self.n_out = n_hidden
        self.group = gspaces.no_base_space(dihedral_group(1))

        self.linear_out_type = nn.FieldType(self.group, self.n_out * [self.group.regular_repr])
        self.enc_obs = MirrorEquivariantResEncoder76(obs_channel, self.n_out, initialize)
        self.enc_ih = MirrorEquivariantResEncoder76(obs_channel, self.n_out, initialize)
        # xyz_6d as proprioception: [-x, (y, z,  r₁₁)  -(r₁₂  r₁₃  r₂₁)  (r₂₂  r₂₃ + qpos (2d))]
        regrep = self.group.regular_repr
        trivrep = self.group.trivial_repr
        irrep = self.group.irrep(1, 0)
        # rot_is_6d
        linear_in_type = nn.FieldType(
            self.group,
            2 * n_hidden * [regrep] + [irrep] + 3 * [trivrep] + 3 * [irrep] + 4 * [trivrep],
        )
        self.enc_out = nn.Linear(linear_in_type, self.linear_out_type)

    def forward(self, nobs):
        # hardcoded this for necessary inputs, for now
        agentview_key = find_all_keys("agentview_image", nobs.keys())
        in_hand_key = find_all_keys("eye_in_hand_image", nobs.keys())
        eef_pos_key = find_all_keys("pos", nobs.keys(), reject="gripper")  # avoid qpos
        eef_rot_key = find_all_keys("rot", nobs.keys())
        gripper_qpos_key = find_all_keys("gripper_qpos", nobs.keys())

        assert len(agentview_key) == 1, f"Found {len(agentview_key)} agentview keys"
        assert len(in_hand_key) == 1, f"Found {len(in_hand_key)} in_hand keys"
        assert len(eef_pos_key) == 1, f"Found {len(eef_pos_key)} eef_pos keys"
        assert len(eef_rot_key) == 1, f"Found {len(eef_rot_key)} eef_rot keys"
        assert len(gripper_qpos_key) == 1, f"Found {len(gripper_qpos_key)} gripper_qpos keys"

        agentview_key = agentview_key[0]
        in_hand_key = in_hand_key[0]
        eef_pos_key = eef_pos_key[0]
        eef_rot_key = eef_rot_key[0]
        gripper_qpos_key = gripper_qpos_key[0]

        obs = nobs[agentview_key]
        ih = nobs[in_hand_key]
        eef_pos = nobs[eef_pos_key]
        eef_rot = nobs[eef_rot_key]
        gripper_qpos = nobs[gripper_qpos_key]
        # B, T, C, H, W
        batch_size = obs.shape[0]
        t = obs.shape[1]
        obs = rearrange(obs, "b t c h w -> (b t) c h w")
        ih = rearrange(ih, "b t c h w -> (b t) c h w")
        eef_pos = rearrange(eef_pos, "b t d -> (b t) d")
        eef_rot = rearrange(eef_rot, "b t d -> (b t) d")
        gripper_qpos = rearrange(gripper_qpos, "b t d -> (b t) d")

        obs = self.crop_randomizer(obs)
        ih = self.crop_randomizer(ih)
        enc_out = self.enc_obs(obs).reshape(batch_size * t, -1)  # b d
        ih_out = self.enc_ih(ih).reshape(batch_size * t, -1)  # b d
        features = torch.cat((enc_out, ih_out, eef_pos, eef_rot, gripper_qpos), dim=1)
        features = nn.GeometricTensor(features, self.enc_out.in_type)
        out = self.enc_out(features).tensor
        return rearrange(out, "(b t) d -> b t d", b=batch_size)


if __name__ == "__main__":
    with torch.no_grad():
        b = 64
        h, w = 84, 84
        obs = torch.randn(b, 1, 3, h, w)
        ih = torch.randn(b, 1, 3, h, w)
        eef_pos = torch.randn(b, 1, 3)
        #
        eef_rot = torch.randn(b, 1, 3, 3).reshape(b, 1, 9)[:, :, :6]
        gripper_qpos = torch.randn(b, 1, 2)

        nobs = {
            "agentview_image": obs,
            "robot0_eye_in_hand_image": ih,
            "robot0_eef_delta_pos": eef_pos,
            "robot0_eef_delta_rot": eef_rot,
            "robot0_gripper_qpos": gripper_qpos,
        }

        mirror_obs = obs.flip(dims=[-1])
        mirror_ih = ih.flip(dims=[-1])

        mirror_eef_pos = eef_pos * torch.tensor([-1, 1, 1], dtype=torch.float32)
        rot_vec_flip = torch.tensor([1, -1, -1, -1, 1, 1], dtype=torch.float32)
        mirror_eef_rot = eef_rot * rot_vec_flip

        mirror_nobs = {
            "agentview_image": mirror_obs,
            "robot0_eye_in_hand_image": mirror_ih,
            "robot0_eef_delta_pos": mirror_eef_pos,
            "robot0_eef_delta_rot": mirror_eef_rot,
            "robot0_gripper_qpos": gripper_qpos,
        }

        model = MirrorDuoObsEnc(
            obs_shape=(3, h, w),
            crop_shape=(76, 76),
            n_hidden=128,
            initialize=True,
        )

        model.eval()

        idx = 0

        out = model(nobs)[:, 0, :]
        mirror_out = model(mirror_nobs)[:, 0, :]

        out = rearrange(out, "b (c n) -> b n c", n=2)
        mirror_out = rearrange(mirror_out, "b (c n) -> b n c", n=2)

        def check_is_mirror(out, mirror_out):
            for i in range(out.shape[0]):
                feat = out[i]
                mirror_feat = mirror_out[i]
                unmirrored_feat = torch.flip(mirror_feat, dims=[0])
                assert torch.allclose(
                    feat, unmirrored_feat, atol=1e-4
                ), f"Feature mismatch at index {i}"

        check_is_mirror(out, mirror_out)
