# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
Qwen-GROOT Framework
A lightweight implementation that Qwen2.5-vl + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5, but a sample MoE inspired by PI_0
"""
from typing import List
import re
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image



from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import get_action_model, LayerwiseFlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

####################################################
# ⚠️ Warning: This framework has been restructured and is NOT compatible with checkpoints created before 2025-10-20.
####################################################

@FRAMEWORK_REGISTRY.register("QwenPI")
class Qwen_PI(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise cross DiT diffusion head 
      

    Focus: Predict future continuous actions conditioned on images + instruction.
    """
# 
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """

        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Dynamic VLM dimensions keep Qwen3/Qwen3.5/action heads aligned.
        model_config = self.qwen_vl_interface.model.config
        text_config = getattr(model_config, "text_config", None)
        llm_hidden_size = getattr(model_config, "hidden_size", None)
        if llm_hidden_size is None and text_config is not None:
            llm_hidden_size = getattr(text_config, "hidden_size", None)
        if llm_hidden_size is None:
            raise ValueError("Unable to infer VLM hidden_size from model config.")

        num_vl_layers = getattr(model_config, "num_hidden_layers", None)
        if num_vl_layers is None and text_config is not None:
            num_vl_layers = getattr(text_config, "num_hidden_layers", None)
        if num_vl_layers is None:
            num_vl_layers = 36
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
    def _router_tokens(self) -> tuple[str, str]:
        datasets_cfg = _cfg_get(self.config, "datasets", None)
        router_cfg = _cfg_get(datasets_cfg, "router_data", None)
        signal_cfg = _cfg_get(datasets_cfg, "vlm_signal_data", None)
        pred_action_token = _cfg_get(router_cfg, "pred_action_token", None)
        if pred_action_token is None:
            pred_action_token = _cfg_get(signal_cfg, "pred_action_token", "<|pred_action|>")
        pred_bbox_token = _cfg_get(router_cfg, "pred_bbox_token", "<|pred_bbox|>")
        return str(pred_action_token), str(pred_bbox_token)

    def _configured_route_tokens(self, *, allow_bbox: bool = False) -> dict[str, str]:
        datasets_cfg = _cfg_get(self.config, "datasets", None)
        router_cfg = _cfg_get(datasets_cfg, "router_data", None)
        configured = _cfg_get(router_cfg, "route_tokens", None)
        if not configured:
            pred_action_token, pred_bbox_token = self._router_tokens()
            return {"action": pred_action_token, "bbox": pred_bbox_token}

        main_routes = [str(route) for route in _cfg_get(router_cfg, "main_routes", configured.keys())]
        tokens = {route: str(_cfg_get(configured, route)) for route in main_routes}
        bbox_cfg = _cfg_get(router_cfg, "bbox", None)
        bbox_allowed = (
            allow_bbox
            and bool(_cfg_get(bbox_cfg, "implementation_enabled", True))
            and bool(_cfg_get(bbox_cfg, "allow_route_prediction", False))
        )
        if bbox_allowed:
            tokens["bbox"] = str(_cfg_get(router_cfg, "pred_bbox_token", "<|pred_bbox|>"))
        return tokens

    def _single_router_token_ids(self, route_tokens: dict[str, str] | None = None) -> dict[str, int] | None:
        route_tokens = route_tokens or self._configured_route_tokens()
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        result = {}
        for route, token in route_tokens.items():
            token_ids = tokenizer(token, add_special_tokens=False).input_ids
            if len(token_ids) != 1:
                logger.warning(
                    "Router tokens must be single tokenizer ids for first-token routing. "
                    "Got %s -> %s. Falling back to generate-and-parse routing.",
                    token,
                    token_ids,
                )
                return None
            result[route] = int(token_ids[0])
        return result

    def _select_action_hidden_states(self, hidden_states, indices: torch.Tensor | None = None) -> list[torch.Tensor]:
        expected_layers = len(self.action_model.model.transformer_blocks)
        vl_embs_list = list(hidden_states[-expected_layers:])
        if indices is not None:
            vl_embs_list = [hidden.index_select(0, indices.to(hidden.device)) for hidden in vl_embs_list]
        return vl_embs_list

    def action_loss_from_hidden_states(
        self,
        hidden_states,
        examples: List[dict],
        indices: torch.Tensor | None = None,
        *,
        detach_vlm_hidden_states: bool = False,
    ) -> torch.Tensor:
        """Compute action-head loss from already-computed Qwen hidden states."""
        if not examples:
            raise ValueError("examples must be non-empty when computing action loss.")

        vl_embs_list = self._select_action_hidden_states(hidden_states, indices=indices)
        if detach_vlm_hidden_states:
            vl_embs_list = [hidden.detach() for hidden in vl_embs_list]
        base_hidden = vl_embs_list[-1]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        action_mask = [example["action_mask"] for example in examples] if "action_mask" in examples[0] else None
        action_dim_mask = (
            [example["action_dim_mask"] for example in examples]
            if "action_dim_mask" in examples[0]
            else None
        )

        device_type = "cuda" if base_hidden.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.float32, enabled=base_hidden.is_cuda):
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]

            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=base_hidden.device, dtype=base_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_mask_repeated = None
            if action_mask is not None:
                action_mask = torch.tensor(
                    np.array(action_mask), device=base_hidden.device, dtype=torch.bool
                )
                action_mask = action_mask[:, -(self.future_action_window_size + 1):]
                action_mask_repeated = action_mask.repeat(repeated_diffusion_steps, 1)

            action_dim_mask_repeated = None
            if action_dim_mask is not None:
                action_dim_mask = torch.tensor(
                    np.array(action_dim_mask), device=base_hidden.device, dtype=torch.bool
                )
                action_dim_mask_repeated = action_dim_mask.repeat(repeated_diffusion_steps, 1)

            action_loss = self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                state_repeated,
                action_mask=action_mask_repeated,
                action_dim_mask=action_dim_mask_repeated,
            )

        return action_loss

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # 取与 DiT 层数匹配的最后 N 层隐藏态，按层喂给 DiT
            all_hidden = qwenvl_outputs.hidden_states

        # Step 4: Action Expert Forward and Loss
        action_loss = self.action_loss_from_hidden_states(all_hidden, examples)
        action_dim_loss = getattr(self.action_model, "latest_action_dim_loss", None)


        return {"action_loss": action_loss, "action_dim_loss": action_dim_loss}

    def forward_action_with_route(
        self,
        examples: List[dict],
        route_token: str | None = None,
    ) -> dict:
        """Run Qwen with assistant route token included, then train the action expert."""
        if type(examples) is not list:
            examples = [examples]
        if route_token is None:
            route_token, _ = self._router_tokens()

        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        solutions = [route_token for _ in examples]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            solutions=solutions,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

        action_loss = self.action_loss_from_hidden_states(qwenvl_outputs.hidden_states, examples)
        action_dim_loss = getattr(self.action_model, "latest_action_dim_loss", None)
        return {
            "action_loss": action_loss,
            "action_dim_loss": action_dim_loss,
            "vlm_loss": getattr(qwenvl_outputs, "loss", None),
            "qwen_outputs": qwenvl_outputs,
        }

    def _train_image_size(self):
        datasets_cfg = _cfg_get(self.config, "datasets", None)
        vla_cfg = _cfg_get(datasets_cfg, "vla_data", None)
        router_cfg = _cfg_get(datasets_cfg, "router_data", None)
        return _cfg_get(vla_cfg, "image_size", _cfg_get(router_cfg, "image_size", None))

    @torch.inference_mode()
    def predict_action( # TODO align  predict_action with forward, make api more flexible
        self,
        examples: List[dict] = None,
        solutions: Optional[List[str]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        from deployment.model_server.tools.image_tools import to_pil_preserve
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
    
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        
        train_obs_image_size = self._train_image_size()
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            solutions=solutions,
        )
        if solutions is not None:
            qwen_inputs.pop("labels", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = qwenvl_outputs.hidden_states
            vl_embs_list = self._select_action_hidden_states(all_hidden)
            base_hidden = vl_embs_list[-1]

        state = torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype) if state is not None else None
        prev_action_chunk = kwargs.pop("prev_action_chunk", kwargs.pop("action_prefix", None))
        inference_delay = kwargs.pop("inference_delay", 0)
        if kwargs:
            logger.warning("Ignoring unsupported predict_action kwargs: %s", sorted(kwargs))
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs_list,
                state,
                prev_action_chunk=prev_action_chunk,
                inference_delay=inference_delay,
            )  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        if torch.is_tensor(inference_delay):
            delay_metadata = inference_delay.detach().cpu().tolist()
        elif isinstance(inference_delay, np.ndarray):
            delay_metadata = inference_delay.tolist()
        else:
            delay_metadata = inference_delay
        return {
            "normalized_actions": normalized_actions,
            "rtc": {
                "enabled": bool(self.action_model.rtc_enabled),
                "applied": prev_action_chunk is not None,
                "inference_delay": delay_metadata,
            },
        }

    @torch.inference_mode()
    def predict_action_with_route_token(
        self,
        examples: List[dict] = None,
        route_token: str | None = None,
        **kwargs,
    ) -> dict:
        if type(examples) is not list:
            examples = [examples]
        if route_token is None:
            route_token, _ = self._router_tokens()
        return self.predict_action(
            examples=examples,
            solutions=[route_token for _ in examples],
            **kwargs,
        )

    @torch.inference_mode()
    def predict_route(
        self,
        examples: List[dict] = None,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.2,
        top_p: float = 0.95,
        route_mode: str = "first_token",
        continue_bbox: bool = True,
        continue_action: bool = False,
        route_confidence_threshold: float | None = None,
        allow_bbox: bool = False,
        **generate_kwargs,
    ) -> dict:
        """Predict the router decision.

        Default ``route_mode="first_token"`` reads the first-token logits and
        chooses among the configured route tokens. This avoids waiting for
        a full autoregressive answer before dispatching to the action expert.
        """
        if type(examples) is not list:
            examples = [examples]

        from deployment.model_server.tools.image_tools import to_pil_preserve

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        train_obs_image_size = self._train_image_size()
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs.update({"temperature": temperature, "top_p": top_p})
        generation_kwargs.update(generate_kwargs)

        route_tokens = self._configured_route_tokens(allow_bbox=allow_bbox)
        token_ids = self._single_router_token_ids(route_tokens)
        if route_mode == "first_token" and token_ids is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwen_output = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )

            first_logits = qwen_output.logits[:, -1, :]
            route_names = list(route_tokens)
            route_token_ids = torch.tensor(
                [token_ids[route] for route in route_names],
                device=first_logits.device,
                dtype=torch.long,
            )
            route_logits = first_logits.index_select(-1, route_token_ids)
            route_probs = torch.softmax(route_logits.float(), dim=-1)
            route_choice = torch.argmax(route_probs, dim=-1)
            route_confidence = torch.max(route_probs, dim=-1).values

            raw_first_ids = torch.argmax(first_logits, dim=-1)
            raw_first_texts = self.qwen_vl_interface.processor.batch_decode(
                raw_first_ids[:, None],
                skip_special_tokens=False,
            )

            chosen_route_token_ids = []
            routes = []
            for idx, choice_tensor in enumerate(route_choice):
                choice = int(choice_tensor.item())
                confidence = float(route_confidence[idx].item())
                if route_confidence_threshold is not None and confidence < float(route_confidence_threshold):
                    route = "unknown"
                    route_token = None
                    route_token_id = int(raw_first_ids[idx].item())
                    generated_text = raw_first_texts[idx].strip()
                else:
                    route = route_names[choice]
                    route_token = route_tokens[route]
                    route_token_id = token_ids[route]
                    generated_text = route_token

                chosen_route_token_ids.append(route_token_id)
                routes.append(
                    {
                        "route": route,
                        "route_token": route_token,
                        "generated_text": generated_text,
                        "route_confidence": confidence,
                        "route_probs": {
                            name: float(route_probs[idx, route_idx].item())
                            for route_idx, name in enumerate(route_names)
                        },
                        "raw_first_token_id": int(raw_first_ids[idx].item()),
                        "raw_first_token_text": raw_first_texts[idx].strip(),
                    }
                )

            routes_to_continue = {"bbox"} if continue_bbox else set()
            if continue_action:
                routes_to_continue.update(route for route in route_names if route != "bbox")

            if any(item["route"] in routes_to_continue for item in routes):
                forced_inputs = {
                    key: value
                    for key, value in qwen_inputs.items()
                    if key != "labels"
                }
                input_ids = forced_inputs["input_ids"]
                route_ids = torch.tensor(
                    chosen_route_token_ids,
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )[:, None]
                forced_inputs["input_ids"] = torch.cat([input_ids, route_ids], dim=1)
                if "attention_mask" in forced_inputs:
                    attention_mask = forced_inputs["attention_mask"]
                    forced_inputs["attention_mask"] = torch.cat(
                        [attention_mask, torch.ones_like(route_ids)],
                        dim=1,
                    )

                generated = self.qwen_vl_interface.generate(**forced_inputs, **generation_kwargs)
                gen_ids = generated.sequences if hasattr(generated, "sequences") else generated
                prompt_len = int(qwen_inputs["input_ids"].shape[1])
                new_ids = gen_ids[:, prompt_len:]
                texts = self.qwen_vl_interface.processor.batch_decode(new_ids, skip_special_tokens=False)
                for route_item, text in zip(routes, texts):
                    if route_item["route"] in routes_to_continue:
                        route_item["generated_text"] = text.strip()

            return {
                "routes": routes,
                "route_mode": "first_token",
            }

        if route_mode != "generate":
            logger.warning("Unsupported route_mode=%s; falling back to generate-and-parse routing.", route_mode)

        generated = self.qwen_vl_interface.generate(**qwen_inputs, **generation_kwargs)
        gen_ids = generated.sequences if hasattr(generated, "sequences") else generated
        prompt_len = int(qwen_inputs["input_ids"].shape[1])
        new_ids = gen_ids[:, prompt_len:]
        texts = self.qwen_vl_interface.processor.batch_decode(new_ids, skip_special_tokens=False)

        routes = []
        for text in texts:
            stripped = text.strip()
            route = "unknown"
            route_token = None
            for candidate_route, candidate_token in route_tokens.items():
                if stripped.startswith(candidate_token):
                    route = candidate_route
                    route_token = candidate_token
                    break
            routes.append(
                {
                    "route": route,
                    "route_token": route_token,
                    "generated_text": stripped,
                }
            )
        return {
            "routes": routes,
            "route_mode": "generate",
        }

    @torch.inference_mode()
    def predict_router(
        self,
        examples: List[dict] = None,
        max_new_tokens: int = 64,
        **generate_kwargs,
    ) -> dict:
        """Route each example, then run the action expert only for action-route items."""
        if type(examples) is not list:
            examples = [examples]
        route_output = self.predict_route(
            examples=examples,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        routes = route_output["routes"]
        action_indices = [idx for idx, item in enumerate(routes) if item["route"] in {"action", "nav"}]
        if action_indices:
            action_examples = [examples[idx] for idx in action_indices]
            action_solutions = [routes[idx].get("generated_text") for idx in action_indices]
            action_output = self.predict_action(
                examples=action_examples,
                solutions=action_solutions,
            )
            normalized_actions = action_output["normalized_actions"]
            for local_idx, sample_idx in enumerate(action_indices):
                routes[sample_idx]["normalized_actions"] = normalized_actions[local_idx]
        return {"routes": routes}

    @staticmethod
    def _parse_subtask(text: str) -> str | None:
        match = re.search(r"<\|subtask\|>(.*?)<\|end_subtask\|>", str(text), flags=re.DOTALL)
        return match.group(1).strip() if match else None

    @torch.inference_mode()
    def predict_typed_action(
        self,
        instruction: str | None = None,
        head_images=None,
        wrist_image=None,
        state=None,
        allow_bbox: bool = False,
        examples: List[dict] | None = None,
        **kwargs,
    ) -> dict:
        """Predict a Go2 route and decode only the dimensions owned by that route."""
        if examples is None:
            images = []
            if head_images is not None:
                images.extend(head_images if isinstance(head_images, (list, tuple)) else [head_images])
            if wrist_image is not None:
                images.append(wrist_image)
            example = {"image": images, "lang": str(instruction or "")}
            if state is not None:
                example["state"] = state
            examples = [example]
        elif type(examples) is not list:
            examples = [examples]

        prev_action_chunk = kwargs.pop("prev_action_chunk", None)
        inference_delay = kwargs.pop("inference_delay", 0)
        route_output = self.predict_route(
            examples=examples,
            continue_action=True,
            allow_bbox=allow_bbox,
            **kwargs,
        )
        routes = route_output["routes"]
        action_indices = [
            idx for idx, item in enumerate(routes) if item["route"] in {"nav", "grasp", "place"}
        ]
        if action_indices:
            action_examples = [examples[idx] for idx in action_indices]
            solutions = [routes[idx]["generated_text"] for idx in action_indices]
            action_output = self.predict_action(
                examples=action_examples,
                solutions=solutions,
                prev_action_chunk=prev_action_chunk,
                inference_delay=inference_delay,
            )
            for local_idx, sample_idx in enumerate(action_indices):
                action_chunk = action_output["normalized_actions"][local_idx]
                if routes[sample_idx]["route"] == "nav":
                    routes[sample_idx]["nav_waypoints"] = action_chunk[:, :3]
                else:
                    routes[sample_idx]["arm_targets_base"] = action_chunk[:, 3:10]

        results = []
        for item in routes:
            results.append(
                {
                    "route": item["route"],
                    "subtask": self._parse_subtask(item.get("generated_text", "")),
                    "nav_waypoints": item.get("nav_waypoints"),
                    "arm_targets_base": item.get("arm_targets_base"),
                    "stop_probability": None,
                    "target_name": None,
                    "grasp_primitive": None,
                    "raw_text": item.get("generated_text", ""),
                    "route_confidence": item.get("route_confidence"),
                    "route_probs": item.get("route_probs"),
                }
            )
        return results[0] if len(results) == 1 else {"results": results}

    @torch.inference_mode()
    def predict_locked_action(
        self,
        instruction: str | None = None,
        head_images=None,
        wrist_image=None,
        state=None,
        examples: List[dict] | None = None,
        locked_route: str | None = None,
        locked_subtask: str | None = None,
        **kwargs,
    ) -> dict:
        """Predict only the action expert for an already-locked route/subtask.

        Skips the autoregressive router/subtask generation entirely: the caller
        provides the decided route and subtask, and the action head is run
        directly with the fixed solution as the assistant prefix. This is the
        fast path used by real/sim clients once a subtask is locked, reducing
        single inference latency from several seconds to a few hundred ms.
        """
        if examples is None:
            images = []
            if head_images is not None:
                images.extend(
                    head_images
                    if isinstance(head_images, (list, tuple))
                    else [head_images]
                )
            if wrist_image is not None:
                images.append(wrist_image)
            example: dict[str, Any] = {"image": images, "lang": str(instruction or "")}
            if state is not None:
                example["state"] = state
            examples = [example]
        elif type(examples) is not list:
            examples = [examples]

        route = str(locked_route or "").strip().lower()
        if route not in {"nav", "grasp", "place"}:
            raise ValueError(
                f"predict_locked_action requires locked_route in "
                f"nav/grasp/place, got {locked_route!r}"
            )

        router_cfg = _cfg_get(_cfg_get(self.config, "datasets", None), "router_data", None)
        route_tokens = self._configured_route_tokens()
        route_token = str(_cfg_get(route_tokens, route, ""))
        subtask_start = str(
            _cfg_get(router_cfg, "subtask_start_token", "<|subtask|>")
        )
        subtask_end = str(
            _cfg_get(router_cfg, "subtask_end_token", "<|end_subtask|>")
        )
        subtask = str(locked_subtask or "").strip()
        solution = f"{route_token}{subtask_start}{subtask}{subtask_end}"

        action_output = self.predict_action(
            examples=examples,
            solutions=[solution],
            prev_action_chunk=kwargs.pop("prev_action_chunk", None),
            inference_delay=kwargs.pop("inference_delay", 0),
        )
        action_chunk = action_output["normalized_actions"][0]
        if route == "nav":
            nav_waypoints = action_chunk[:, :3]
            arm_targets_base = None
        else:
            nav_waypoints = None
            arm_targets_base = action_chunk[:, 3:10]

        return {
            "route": route,
            "subtask": subtask or None,
            "nav_waypoints": nav_waypoints,
            "arm_targets_base": arm_targets_base,
            "stop_probability": None,
            "target_name": None,
            "grasp_primitive": None,
            "raw_text": solution,
            "route_confidence": None,
            "route_probs": None,
            "locked_action": True,
        }

    @torch.inference_mode()
    def predict_bbox(
        self,
        instruction: str,
        head_images=None,
        wrist_image=None,
        max_new_tokens: int = 48,
    ) -> dict:
        """Run the preserved bbox text branch explicitly, independently of main routing."""
        datasets_cfg = _cfg_get(self.config, "datasets", None)
        router_cfg = _cfg_get(datasets_cfg, "router_data", None)
        bbox_cfg = _cfg_get(router_cfg, "bbox", None)
        if not bool(_cfg_get(bbox_cfg, "implementation_enabled", True)):
            raise RuntimeError("BBox implementation is disabled by configuration.")

        images = []
        if head_images is not None:
            images.extend(head_images if isinstance(head_images, (list, tuple)) else [head_images])
        if wrist_image is not None:
            images.append(wrist_image)
        bbox_token = str(_cfg_get(router_cfg, "pred_bbox_token", "<|pred_bbox|>"))
        prompt = (
            f"{instruction}\nReturn the target bounding box as "
            f"{bbox_token}<point>[x1, y1, x2, y2]</point>."
        )
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=[images], instructions=[prompt]
        )
        forced_inputs = {key: value for key, value in qwen_inputs.items() if key != "labels"}
        token_ids = self.qwen_vl_interface.processor.tokenizer(
            bbox_token, add_special_tokens=False
        ).input_ids
        if len(token_ids) != 1:
            raise RuntimeError(f"BBox token must map to one token id, got {token_ids}")
        input_ids = forced_inputs["input_ids"]
        route_id = torch.full(
            (input_ids.shape[0], 1),
            int(token_ids[0]),
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        forced_inputs["input_ids"] = torch.cat((input_ids, route_id), dim=1)
        if "attention_mask" in forced_inputs:
            forced_inputs["attention_mask"] = torch.cat(
                (forced_inputs["attention_mask"], torch.ones_like(route_id)), dim=1
            )
        generated = self.qwen_vl_interface.generate(
            **forced_inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        raw_text = self.qwen_vl_interface.processor.batch_decode(
            sequences[:, input_ids.shape[1] :], skip_special_tokens=False
        )[0].strip()
        match = re.search(
            r"<point>\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
            r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*</point>",
            raw_text,
        )
        bbox = [float(value) for value in match.groups()] if match else None
        return {
            "bbox": bbox,
            "camera": "head" if head_images is not None else "wrist" if wrist_image is not None else None,
            "parse_success": bbox is not None,
            "raw_text": raw_text,
        }



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    

    model = Qwen_PI(cfg)
    # ckpt="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1011_qwenpi/checkpoints/need_steps_10000_pytorch_model.pt"
    # model = Qwen_PI.from_pretrained(ckpt)
    print(model)


    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake instruction for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
