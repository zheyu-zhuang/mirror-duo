from typing import Union

import torch
from einops import rearrange, repeat
from equi_diffpo.model.diffusion.conditional_unet1d import ConditionalUnet1D
from escnn import gspaces, nn
from escnn.group import dihedral_group


class MirrorDuoDiffusionUNet(torch.nn.Module):
    def __init__(
        self,
        act_emb_dim,
        local_cond_dim,
        global_cond_dim,
        diffusion_step_embed_dim,
        down_dims,
        kernel_size,
        n_groups,
        cond_predict_scale,
    ):
        """
        Initializes the MirrorDuoDiffusionUNet model.

        Args:
            act_emb_dim (int): Action embedding dimension.
            local_cond_dim (int): Local conditioning dimension.
            global_cond_dim (int): Global conditioning dimension.
            diffusion_step_embed_dim (int): Diffusion step embedding dimension.
            down_dims (list): Dimensions for the downsampling layers.
            kernel_size (int): Kernel size for convolutional layers.
            n_groups (int): Number of groups for group normalization.
            cond_predict_scale (float): Scale for conditional prediction.
        """
        super().__init__()
        self.unet = ConditionalUnet1D(
            input_dim=act_emb_dim,
            local_cond_dim=local_cond_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.group = gspaces.no_base_space(dihedral_group(1))

        self.act_feat_type = nn.FieldType(self.group, act_emb_dim * [self.group.regular_repr])

        self.act_decoder = nn.Linear(self.act_feat_type, self.get_act_field_type())

        self.act_encoder = nn.SequentialModule(
            nn.Linear(self.get_act_field_type(), self.act_feat_type), nn.ReLU(self.act_feat_type)
        )

    def get_act_field_type(self):
        """
        Defines the field type for the action tensor.

        Returns:
            nn.FieldType: The field type for the action tensor.
        """
        # [-x, y, z, r₁₁  -r₁₂  -r₁₃  -r₂₁  r₂₂  r₂₃] + gripper (1d)
        return nn.FieldType(
            self.group,
            1 * [self.group.irrep(1, 0)]  # x
            + 3 * [self.group.trivial_repr]  # y, z, r₁₁
            + 3 * [self.group.irrep(1, 0)]  # r₁₂, r₁₃, r₂₁
            + 3 * [self.group.trivial_repr],  # r₂₂, r₂₃, gripper act (binary)
        )

    def get_act_geometric_tensor(self, act):
        """
        Converts an action tensor into a geometric tensor.

        Args:
            act (torch.Tensor): Input action tensor.

        Returns:
            nn.GeometricTensor: Geometric tensor with the appropriate field type.
        """
        return nn.GeometricTensor(act, self.get_act_field_type())

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond=None,
        global_cond=None,
        **kwargs
    ):
        """
        Forward pass of the model.

        Args:
            sample (torch.Tensor): Input tensor of shape (B, T, input_dim).
            timestep (Union[torch.Tensor, float, int]): Diffusion step, either as a scalar or tensor of shape (B,).
            local_cond (torch.Tensor, optional): Local conditioning tensor of shape (B, T, local_cond_dim).
            global_cond (torch.Tensor, optional): Global conditioning tensor of shape (B, global_cond_dim).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, input_dim).
        """
        # Validate input shapes
        B, T = sample.shape[:2]
        if local_cond is not None and local_cond.shape[:2] != (B, T):
            raise ValueError("local_cond must have shape (B, T, local_cond_dim)")
        if global_cond is not None and global_cond.shape[0] != B:
            raise ValueError("global_cond must have shape (B, global_cond_dim)")

        # Reshape and encode the action tensor
        sample = rearrange(sample, "b t d -> (b t) d")
        sample = self.get_act_geometric_tensor(sample)
        act_equi_feat = self.act_encoder(sample).tensor.reshape(B, T, -1)
        
        act_equi_feat = rearrange(act_equi_feat, "b t (c n) -> (b n) t c", n=2)

        # Process timestep
        if isinstance(timestep, torch.Tensor) and len(timestep.shape) == 1:
            timestep = repeat(timestep, "b -> (b n)", n=2)

        # Process local and global conditions
        if local_cond is not None:
            local_cond = rearrange(local_cond, "b t (c n) -> (b n) t c", n=2)
        if global_cond is not None:
            global_cond = rearrange(global_cond, "b (c n) -> (b n) c", n=2)

        # Decode the action
        out = self.unet(act_equi_feat, timestep, local_cond, global_cond, **kwargs)
        out = rearrange(out, "(b n) t c -> (b t) (c n)", n=2)
        out = nn.GeometricTensor(out, self.act_feat_type)
        out = self.act_decoder(out).tensor.reshape(B * T, -1)
        return rearrange(out, "(b t) c -> b t c", b=B)  # b, t, act_dim
