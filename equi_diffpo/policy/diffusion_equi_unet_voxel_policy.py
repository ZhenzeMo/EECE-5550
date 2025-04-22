from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from equi_diffpo.model.equi.equi_obs_encoder import EquivariantObsEncVoxel
from equi_diffpo.model.equi.equi_conditional_unet1d import EquiDiffusionUNet
from equi_diffpo.model.diffusion.mask_generator import LowdimMaskGenerator
from equi_diffpo.model.common.normalizer import LinearNormalizer
from equi_diffpo.model.vision.voxel_rot_randomizer import VoxelRotRandomizer
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.common.pytorch_util import dict_apply

class DiffusionEquiUNetPolicyVoxel(BaseImagePolicy):
    """
    Equivariant Diffusion Policy model using voxel observations.
    """
    def __init__(self, 
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 horizon: int, 
                 n_action_steps: int, 
                 n_obs_steps: int,
                 num_inference_steps: int = None,
                 crop_shape: Tuple[int, int, int] = (58, 58, 58),
                 N: int = 8,
                 enc_n_hidden: int = 64,
                 diffusion_step_embed_dim: int = 256,
                 down_dims: Tuple[int] = (256, 512, 1024),
                 kernel_size: int = 5,
                 n_groups: int = 8,
                 cond_predict_scale: bool = True,
                 rot_aug: bool = False,
                 initialize: bool = True,
                 color: bool = True,
                 depth: bool = True,
                 **kwargs):
        super().__init__()

        # Parse action dimension
        self.action_dim = shape_meta['action']['shape'][0]

        # Set observation channels
        self.obs_channel = (4 if color and depth else 3 if color else 1)

        # Initialize encoders and diffusion model
        self.enc = EquivariantObsEncVoxel(
            obs_shape=(self.obs_channel, 64, 64, 64), 
            crop_shape=crop_shape, 
            n_hidden=enc_n_hidden, 
            N=N,
            initialize=initialize,
        )

        global_cond_dim = enc_n_hidden * n_obs_steps

        self.diff = EquiDiffusionUNet(
            act_emb_dim=64,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            N=N,
        )

        # Utilities
        self.mask_generator = LowdimMaskGenerator(self.action_dim, 0, n_obs_steps, fix_obs_steps=True, action_visible=False)
        self.normalizer = LinearNormalizer()
        self.rot_randomizer = VoxelRotRandomizer()

        # Hyperparameters
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.crop_shape = crop_shape
        self.obs_feature_dim = enc_n_hidden
        self.rot_aug = rot_aug
        self.kwargs = kwargs

        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = num_inference_steps or noise_scheduler.config.num_train_timesteps

    # ================= Training =================
    def set_normalizer(self, normalizer: LinearNormalizer):
        """
        Load pre-trained normalizer.
        """
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(self, weight_decay: float, learning_rate: float, betas: Tuple[float, float], eps: float) -> torch.optim.Optimizer:
        """
        Initialize AdamW optimizer.
        """
        return torch.optim.AdamW(self.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)

    def compute_loss(self, batch) -> torch.Tensor:
        """
        Compute MSE loss for training diffusion model.
        """
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if self.rot_aug:
            nobs, nactions = self.rot_randomizer(nobs, nactions)

        B, T = nactions.shape[:2]
        global_cond = self.enc(nobs).reshape(B, -1)

        condition_mask = self.mask_generator(nactions.shape)
        noise = torch.randn_like(nactions)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=nactions.device).long()

        noisy_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)
        noisy_actions[condition_mask] = nactions[condition_mask]

        pred = self.diff(noisy_actions, timesteps, global_cond=global_cond)
        target = noise if self.noise_scheduler.config.prediction_type == 'epsilon' else nactions

        loss = F.mse_loss(pred, target, reduction='none')
        loss = (loss * (~condition_mask).float()).mean(dim=list(range(1, loss.ndim))).mean()

        return loss

    # ================= Inference =================
    def conditional_sample(self, condition_data, condition_mask, local_cond=None, global_cond=None, generator=None, **kwargs):
        """
        Perform conditional denoising diffusion sampling.
        """
        model, scheduler = self.diff, self.noise_scheduler
        traj = torch.randn_like(condition_data, generator=generator)

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            traj[condition_mask] = condition_data[condition_mask]
            model_output = model(traj, t, local_cond=local_cond, global_cond=global_cond)
            traj = scheduler.step(model_output, t, traj, generator=generator, **kwargs).prev_sample

        traj[condition_mask] = condition_data[condition_mask]
        return traj

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict future actions given the current observation.
        """
        assert 'past_action' not in obs_dict  # not supported yet

        if 'agentview_image' in obs_dict:
            del obs_dict['agentview_image']
        
        obs_dict['voxels'][:, :, 1:] /= 255.0
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        
        global_cond = self.enc(nobs).reshape(B, -1)

        cond_data = torch.zeros((B, self.horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        sampled_traj = self.conditional_sample(cond_data, cond_mask, global_cond=global_cond, **self.kwargs)
        naction_pred = sampled_traj[..., :self.action_dim]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            'action': action,
            'action_pred': action_pred
        }


    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if self.rot_aug:
            nobs, nactions = self.rot_randomizer(nobs, nactions)

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        # reshape B, T, ... to B*T
        # this_nobs = dict_apply(nobs, 
        #     lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.enc(nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(batch_size, -1)

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.diff(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss