# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Shijie LIAN/ Huazhong University of Science & Technology] in [2026].
# Design and Merged by [Jinhui YE / HKUST University] in [2026].

import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Optional, List

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoProcessor, BatchFeature

from qwen_vl_utils import process_vision_info
from starVLA.model.modules.vlm.special_tokens import prepare_qwen_special_tokens
from accelerate.logging import get_logger

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError as import_error:
    raise ImportError(
        "Qwen3.5 model class is unavailable. Please install transformers >= 5.2.0 or check your transformers version."
    ) from import_error

logger = get_logger(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 248056
VIDEO_TOKEN_INDEX = 248057
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 248077 # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = 248077 + 2047 # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md


import torch.nn as nn


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class _QWen3_5_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3.5-VL (Qwen3_5ForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3.5-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3.5-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3.5-VL-4B-Instruct")

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"
        prepared_tokens = prepare_qwen_special_tokens(
            model=model,
            tokenizer=processor.tokenizer,
            config=config,
            logger=logger,
            default_policy="auto_add",
        )

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3.5 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        if prepared_tokens.action_token_range is not None:
            self._ACTION_TOKEN_MIN, self._ACTION_TOKEN_MAX = prepared_tokens.action_token_range
        elif "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen3.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).

        When solutions are provided (supervised VLM training), labels are built by
        keeping only the assistant answer span and masking all other tokens.
        """

        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"

        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            datasets_cfg = _cfg_get(self.config, "datasets", None)
            prompt_cfg = _cfg_get(datasets_cfg, "vla_data", None)
            if prompt_cfg is None:
                prompt_cfg = _cfg_get(datasets_cfg, "router_data", None)
            if prompt_cfg is not None and "CoT_prompt" in prompt_cfg:
                cot_prompt = prompt_cfg.get("CoT_prompt", "")
                prompt = cot_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        add_generation_prompt = solutions is None
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=add_generation_prompt)
            for m in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        batch_inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if solutions is not None:
            labels = torch.full_like(batch_inputs["input_ids"], IGNORE_INDEX)
            prefix_ids = self.processor.tokenizer(
                "<|im_start|>assistant\n", add_special_tokens=False
            ).input_ids
            prefix = torch.tensor(prefix_ids, device=batch_inputs["input_ids"].device)
            empty_think_ids = self.processor.tokenizer(
                "<think>\n\n</think>\n\n", add_special_tokens=False
            ).input_ids
            empty_think = torch.tensor(empty_think_ids, device=batch_inputs["input_ids"].device)
            im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
            pad_id = self.processor.tokenizer.pad_token_id

            for i in range(batch_inputs["input_ids"].size(0)):
                seq = batch_inputs["input_ids"][i]
                start = None
                max_j = seq.numel() - prefix.numel() + 1
                if max_j > 0:
                    for j in range(max_j):
                        if torch.equal(seq[j : j + prefix.numel()], prefix):
                            start = j + prefix.numel()
                            break

                if start is None:
                    continue

                if empty_think.numel() > 0:
                    think_end = start + empty_think.numel()
                    if think_end <= seq.numel() and torch.equal(seq[start:think_end], empty_think):
                        start = think_end

                end = start
                if im_end_id is None:
                    end = seq.numel()
                else:
                    while end < seq.numel() and int(seq[end].item()) != im_end_id:
                        end += 1

                if end > start:
                    labels[i, start:end] = seq[start:end]

            if pad_id is not None:
                labels[batch_inputs["input_ids"] == pad_id] = IGNORE_INDEX
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)




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
    
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3.5-VL-4B-Instruct"
    qwen_vl = _QWen3_5_VL_Interface(cfg)
    pass
