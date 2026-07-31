# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025]. 
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].



import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT, SelfAttentionTransformer

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple

class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # import ipdb; ipdb.set_trace()
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,) or (B, T)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        elif timesteps.dim() == 2 and timesteps.shape == (B, T):
            pass
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) or (B, T)."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x



class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024, num_embodiments=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,) or (B, T)
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        elif timesteps.dim() == 2 and timesteps.shape == (B, T):
            pass
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) or (B, T)."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)




DiTConfig = {"num_layers": 36, "input_embedding_dim": 2048, "attention_head_dim": 64, "num_attention_heads": 32} # default for qwen2.5-vl


class LayerwiseFlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        global_config,
        **kwargs,
    ):
        super().__init__()
        action_config = global_config.framework.action_model
        diffusion_model_cfg = action_config.diffusion_model_cfg

        # 更新 DiTConfig 到 diffusion_model_cfg
        # The action expert may consume only the last N VLM hidden layers. Honor
        # the configured DiT depth instead of silently expanding it to every VLM layer.
        DiTConfig["num_layers"] = int(
            getattr(diffusion_model_cfg, "num_layers", global_config.framework.qwenvl.num_vl_layers)
        )
        DiTConfig["input_embedding_dim"] = global_config.framework.qwenvl.vl_hidden_dim
        DiTConfig["num_attention_heads"] = DiTConfig["input_embedding_dim"] // DiTConfig["attention_head_dim"]
        diffusion_model_cfg.update(DiTConfig)
        # diffusion_model_cfg["interleave_self_attention"] = False
        diffusion_model_cfg.cross_attention_dim = DiTConfig["input_embedding_dim"] # should match vl embedding dim, but for some case we might want to change it for cross + self attention
        self.input_embedding_dim = global_config.framework.qwenvl.vl_hidden_dim
        self.model = DiT(**diffusion_model_cfg) # TODO better way is copy LLM from VLM
        self.dit_out_hidden_size = self.input_embedding_dim
        self.action_dim = action_config.action_dim
        self.action_horizon = action_config.future_action_window_size + 1
        self.num_inference_timesteps = action_config.num_inference_timesteps

        self.state_encoder = MLP(
            input_dim=action_config.state_dim,
            output_dim=self.input_embedding_dim,
        ) if action_config.state_dim else None

        self.action_encoder = ActionEncoder(
            action_dim=action_config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.input_embedding_dim,
            hidden_dim=1024,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(action_config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(action_config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config
        rtc_config = getattr(global_config.framework, "rtc", None)
        self.rtc_enabled = self._as_bool(getattr(rtc_config, "enable", False)) if rtc_config is not None else False
        self.rtc_simulated_delay = int(getattr(rtc_config, "simulated_delay", 0) or 0) if rtc_config is not None else 0
        self.rtc_delay_sampling = (
            str(getattr(rtc_config, "delay_sampling", "exponential"))
            if rtc_config is not None
            else "exponential"
        )
        self.rtc_conditioning_mode = (
            str(getattr(rtc_config, "conditioning_mode", "action_encoder"))
            if rtc_config is not None
            else "action_encoder"
        )
        valid_rtc_modes = {"action_encoder", "dit_token"}
        if self.rtc_conditioning_mode not in valid_rtc_modes:
            raise ValueError(
                f"Unsupported rtc.conditioning_mode={self.rtc_conditioning_mode!r}. "
                f"Expected one of {sorted(valid_rtc_modes)}."
            )
        valid_delay_sampling = {"exponential", "uniform"}
        if self.rtc_delay_sampling not in valid_delay_sampling:
            raise ValueError(
                f"Unsupported rtc.delay_sampling={self.rtc_delay_sampling!r}. "
                f"Expected one of {sorted(valid_delay_sampling)}."
            )

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @staticmethod
    def _rank() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def _debug_concat_shapes(
        self,
        context: str,
        future_tokens: torch.Tensor,
        action_features: torch.Tensor,
        state_features: torch.Tensor = None,
    ) -> None:
        should_print = os.environ.get("STARVLA_DEBUG_ACTION_SHAPES", "0") == "1"
        tensors = [future_tokens, action_features]
        if state_features is not None:
            tensors = [state_features] + tensors

        mismatch = False
        if any(t.dim() != 3 for t in tensors):
            mismatch = True
        else:
            batch_sizes = {int(t.shape[0]) for t in tensors}
            hidden_sizes = {int(t.shape[-1]) for t in tensors}
            mismatch = len(batch_sizes) != 1 or len(hidden_sizes) != 1

        if should_print or mismatch:
            print(
                f"[rank{self._rank()}][{context}] "
                f"future_tokens_weight_shape={tuple(self.future_tokens.weight.shape)} "
                f"future_tokens_shape={tuple(future_tokens.shape)} "
                f"action_features_shape={tuple(action_features.shape)} "
                f"state_features_shape={None if state_features is None else tuple(state_features.shape)} "
                f"future_num_embeddings={self.future_tokens.num_embeddings} "
                f"future_embedding_dim={self.future_tokens.embedding_dim} "
                f"action_dtype={action_features.dtype} "
                f"future_dtype={future_tokens.dtype}",
                flush=True,
            )

        if mismatch:
            raise RuntimeError(
                f"Action head concat shape mismatch in {context}. "
                f"future_tokens={tuple(future_tokens.shape)}, "
                f"action_features={tuple(action_features.shape)}, "
                f"state_features={None if state_features is None else tuple(state_features.shape)}, "
                f"future_tokens.weight={tuple(self.future_tokens.weight.shape)}"
            )

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def sample_rtc_delay(self, batch_size, action_horizon, device):
        max_delay = min(self.rtc_simulated_delay, action_horizon)
        if max_delay <= 1:
            return torch.zeros(batch_size, device=device, dtype=torch.long)

        if self.rtc_delay_sampling == "uniform":
            return torch.randint(max_delay, (batch_size,), device=device)

        weights = torch.exp(
            torch.arange(max_delay - 1, -1, -1, device=device, dtype=torch.float32)
        )
        weights = weights / weights.sum()
        return torch.multinomial(weights, batch_size, replacement=True).long()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)


    def forward(
        self,
        vl_embs_list: list,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        action_mask: torch.Tensor = None,
    ):
        """
        vl_embs: list of torch.Tensor, each shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        """
        device = actions.device
        num_layers = len(vl_embs_list)
        B, L, D = vl_embs_list[0].shape
        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        base_t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = base_t[:, None].expand(-1, actions.shape[1])
        loss_mask = torch.ones(actions.shape[:2], device=actions.device, dtype=torch.bool)
        if action_mask is not None:
            action_mask = torch.as_tensor(action_mask, device=actions.device, dtype=torch.bool)
            if tuple(action_mask.shape) != tuple(actions.shape[:2]):
                raise ValueError(
                    f"action_mask must have shape {tuple(actions.shape[:2])}, got {tuple(action_mask.shape)}"
                )
            loss_mask &= action_mask
        self.latest_rtc_delay = None

        if self.rtc_enabled and self.rtc_simulated_delay > 0:
            delay = self.sample_rtc_delay(
                batch_size=actions.shape[0],
                action_horizon=actions.shape[1],
                device=actions.device,
            )
            self.latest_rtc_delay = delay.detach()
            step_ids = torch.arange(actions.shape[1], device=actions.device)
            prefix_mask = step_ids.unsqueeze(0) < delay.unsqueeze(1)
            t = torch.where(prefix_mask, torch.ones_like(t), t)
            loss_mask &= ~prefix_mask

        noisy_trajectory = (1 - t[..., None]) * noise + t[..., None] * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed.
        # RTC mode A: per-action timestep only conditions ActionEncoder.
        # RTC mode B: the same per-action timestep also conditions DiT action tokens.
        t_discretized = (t * self.num_timestep_buckets).long().clamp(max=self.num_timestep_buckets - 1)
        temb_discretized = (base_t * self.num_timestep_buckets).long().clamp(max=self.num_timestep_buckets - 1)
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        # Embed state
        state_features = self.state_encoder(state) if state is not None else None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        future_token_ids = torch.arange(
            self.future_tokens.num_embeddings,
            dtype=torch.long,
            device=device,
        )
        future_tokens = self.future_tokens(future_token_ids).unsqueeze(0).expand(B, -1, -1)
        self._debug_concat_shapes(
            "LayerwiseFlowmatchingActionHead.forward",
            future_tokens=future_tokens,
            action_features=action_features,
            state_features=state_features,
        )
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
            if state_features is not None else torch.cat((future_tokens, action_features), dim=1)
        
        # Encode timesteps
        if self.rtc_enabled and self.rtc_conditioning_mode == "dit_token":
            non_action_len = sa_embs.shape[1] - action_features.shape[1]
            non_action_timesteps = temb_discretized[:, None].expand(-1, non_action_len)
            sa_t_discretized = torch.cat((non_action_timesteps, t_discretized), dim=1)
            temb = self.model.timestep_encoder(sa_t_discretized)
        else:
            temb = self.model.timestep_encoder(temb_discretized)

        # Layerwise cross-attention with vl_embs
        model_output = sa_embs
        for layer_idx, layer in enumerate(self.model.transformer_blocks):
            model_output = layer(
                hidden_states=model_output,
                encoder_hidden_states=vl_embs_list[layer_idx],  # Use layer-specific vl_embs
                temb=temb,
            )
        
        # TODO miss self att and _process_output, but work well
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        squared_error = (pred_actions - velocity) ** 2
        if action_mask is not None or (self.rtc_enabled and self.rtc_simulated_delay > 0):
            loss_mask = loss_mask.unsqueeze(-1).to(dtype=squared_error.dtype)
            denom = loss_mask.sum().clamp_min(1.0) * squared_error.shape[-1]
            loss = (squared_error * loss_mask).sum() / denom
            per_dim_denom = loss_mask.sum().clamp_min(1.0)
            per_dim_loss = (squared_error * loss_mask).sum(dim=(0, 1)) / per_dim_denom
        else:
            loss = squared_error.mean()
            per_dim_loss = squared_error.mean(dim=(0, 1))
        self.latest_action_dim_loss = per_dim_loss.detach().float()
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        prev_action_chunk: torch.Tensor = None,
        inference_delay=0,
    ) -> torch.Tensor:
        """Generate a chunk, optionally pinning its RTC prefix to the previous chunk.

        ``prev_action_chunk`` must already be aligned to the current observation and
        normalized with the same statistics as the training actions. The first
        ``inference_delay`` actions are treated as clean (t=1) conditions.
        """
        # Set initial actions as the sampled noise.
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        dtype = vl_embs_list[0].dtype
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=dtype,
            device=device,
        )

        prefix_mask = None
        if prev_action_chunk is not None:
            if not self.rtc_enabled:
                raise ValueError("RTC inference requires framework.rtc.enable=true.")
            prev_action_chunk = torch.as_tensor(prev_action_chunk, device=device, dtype=dtype)
            if prev_action_chunk.dim() == 2:
                prev_action_chunk = prev_action_chunk.unsqueeze(0)
            expected_shape = (batch_size, self.action_horizon, self.action_dim)
            if tuple(prev_action_chunk.shape) != expected_shape:
                raise ValueError(
                    f"prev_action_chunk must have shape {expected_shape}, "
                    f"got {tuple(prev_action_chunk.shape)}."
                )

            delay = torch.as_tensor(inference_delay, device=device, dtype=torch.long)
            if delay.dim() == 0:
                delay = delay.expand(batch_size)
            if tuple(delay.shape) != (batch_size,):
                raise ValueError(
                    f"inference_delay must be a scalar or shape ({batch_size},), "
                    f"got {tuple(delay.shape)}."
                )
            if bool(((delay < 0) | (delay > self.action_horizon)).any()):
                raise ValueError(
                    f"inference_delay must be in [0, {self.action_horizon}], "
                    f"got {delay.tolist()}."
                )
            step_ids = torch.arange(self.action_horizon, device=device)
            prefix_mask = step_ids.unsqueeze(0) < delay.unsqueeze(1)
        elif torch.as_tensor(inference_delay).ne(0).any():
            raise ValueError("inference_delay requires prev_action_chunk.")

        num_steps = self.num_inference_timesteps
        if not num_steps or num_steps <= 0:
            raise ValueError(f"num_inference_timesteps must be positive, got {num_steps}.")
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized_int = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )

            action_timesteps = timesteps_tensor
            if prefix_mask is not None:
                actions = torch.where(prefix_mask.unsqueeze(-1), prev_action_chunk, actions)
                action_timesteps = timesteps_tensor.unsqueeze(1).expand(-1, self.action_horizon)
                action_timesteps = torch.where(
                    prefix_mask,
                    torch.full_like(action_timesteps, self.num_timestep_buckets - 1),
                    action_timesteps,
                )

            # Embed current action trajectory with timestep
            action_features = self.action_encoder(actions, action_timesteps)

            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_token_ids = torch.arange(
                self.future_tokens.num_embeddings,
                dtype=torch.long,
                device=device,
            )
            future_tokens = self.future_tokens(future_token_ids).unsqueeze(0).expand(batch_size, -1, -1)
            self._debug_concat_shapes(
                "LayerwiseFlowmatchingActionHead.predict_action",
                future_tokens=future_tokens,
                action_features=action_features,
                state_features=state_features,
            )
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            # RTC dit_token mode conditions each action token on its own timestep.
            if prefix_mask is not None and self.rtc_conditioning_mode == "dit_token":
                non_action_len = sa_embs.shape[1] - action_features.shape[1]
                non_action_timesteps = timesteps_tensor[:, None].expand(-1, non_action_len)
                sa_timesteps = torch.cat((non_action_timesteps, action_timesteps), dim=1)
                temb = self.model.timestep_encoder(sa_timesteps)
            else:
                temb = self.model.timestep_encoder(timesteps_tensor)

            # Layerwise cross-attention with vl_embs_list
            model_output = sa_embs
            for layer_idx, layer in enumerate(self.model.transformer_blocks):
                model_output = layer(
                    hidden_states=model_output,
                    encoder_hidden_states=vl_embs_list[layer_idx],
                    temb=temb,
                )
            # TODO miss self att and _process_output 
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon :]

            # Euler integration
            actions = actions + dt * pred_velocity
        if prefix_mask is not None:
            actions = torch.where(prefix_mask.unsqueeze(-1), prev_action_chunk, actions)
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype



def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.
    
    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return LayerwiseFlowmatchingActionHead(
        global_config=config
    )



if __name__ == "__main__":
    # TODO make each backbone.py can be debug independently

    pass
