# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""Fast Action Tokenizer Adapter
"this file is adapted from https://huggingface.co/physical-intelligence/fast"

Overview:
    This module encapsulates a lightweight "action → language model-readable sequence" converter (Fast_Action_Tokenizer).
    Its core objective is to convert continuous/discrete raw robot actions (raw_actions) into
    pseudo-natural language token strings like <robot_action_12><robot_action_3><robot_action_87> ...
    This facilitates direct integration into multimodal large models (VLM/LLM) dialogue templates,
    leveraging their language modeling capabilities for action prediction.
"""

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch.nn as nn
from transformers import AutoProcessor, PreTrainedTokenizerFast


def _format_action_token(prefix: str, token_id: int) -> str:
    if prefix.endswith("{i}"):
        return prefix.format(i=int(token_id))
    return f"{prefix}{int(token_id)}>"


def _load_local_fast_processor(path: str):
    path_obj = Path(path)
    processor_py = path_obj / "processing_action_tokenizer.py"
    tokenizer_json = path_obj / "tokenizer.json"
    if not tokenizer_json.exists():
        tokenizer_json = path_obj / "bpe_tokenizer" / "tokenizer.json"
    processor_config = path_obj / "processor_config.json"
    if not processor_py.exists() or not tokenizer_json.exists():
        raise FileNotFoundError(
            f"Local FAST tokenizer path `{path}` must contain processing_action_tokenizer.py and tokenizer.json"
        )

    spec = importlib.util.spec_from_file_location("starvla_local_fast_processor", processor_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import local FAST processor from `{processor_py}`")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        clean_up_tokenization_spaces=False,
    )

    config: dict[str, Any] = {}
    if processor_config.exists():
        with open(processor_config, "r", encoding="utf-8") as f:
            config = json.load(f)

    return module.UniversalActionProcessor(
        tokenizer,
        scale=config.get("scale", 10),
        vocab_size=config.get("vocab_size", 2048),
        min_token=config.get("min_token", 0),
        action_dim=config.get("action_dim", None),
        time_horizon=config.get("time_horizon", None),
    )


def load_fast_action_processor(tokenizer_path: str):
    if os.path.isdir(tokenizer_path):
        return _load_local_fast_processor(tokenizer_path)
    try:
        return AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception:
        raise


class FastActionTokenFormatter:
    """Format FAST processor ids as Qwen action special-token strings."""

    def __init__(
        self,
        tokenizer_path: str = "physical-intelligence/fast",
        *,
        token_prefix: str = "<robot_action_",
        token_count: int = 2048,
    ) -> None:
        self.processor = load_fast_action_processor(tokenizer_path)
        self.token_prefix = str(token_prefix)
        self.token_count = int(token_count)

    def encode_ids(self, raw_actions: np.ndarray | list[np.ndarray]) -> list[list[int]]:
        actions = np.asarray(raw_actions, dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[None, ...]
        if actions.ndim != 3:
            raise ValueError(f"FAST action input must have shape [B,T,D] or [T,D], got {actions.shape}")

        token_batches = self.processor(actions)
        encoded: list[list[int]] = []
        for tokens in token_batches:
            ids = [int(token_id) for token_id in tokens]
            invalid = [token_id for token_id in ids if token_id < 0 or token_id >= self.token_count]
            if invalid:
                preview = invalid[:8]
                raise ValueError(
                    f"FAST tokenizer emitted ids outside [0,{self.token_count - 1}]: {preview}"
                )
            encoded.append(ids)
        return encoded

    def encode_strings(self, raw_actions: np.ndarray | list[np.ndarray]) -> list[str]:
        return [
            "".join(_format_action_token(self.token_prefix, token_id) for token_id in token_ids)
            for token_ids in self.encode_ids(raw_actions)
        ]

    def encode_string(self, raw_action: np.ndarray) -> str:
        return self.encode_strings(np.asarray(raw_action, dtype=np.float32))[0]



class Fast_Action_Tokenizer(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, fast_tokenizer_name="playground/Pretrained_models/fast"):
        super().__init__()
        self.fast_tokenizer = load_fast_action_processor(fast_tokenizer_name)


    def encoder_action2fastoken(self, raw_actions):
        # x: (batch_size, chunck, dim)
        batch_actions = np.stack(raw_actions, axis=0)  # (B, T, D)
        batch_fast_tokens = self.fast_tokenizer(batch_actions)

        return batch_fast_tokens # List[str]
    
    def decoder_action(self, generated_ids):
        # api https://huggingface.co/physical-intelligence/fast
        # return: (batch_size, chunck, dim)
        pred_actions = self.fast_tokenizer.decode([generated_ids - self._ACTION_TOKEN_MIN])
        return pred_actions
    

    def fit_tokenizer_on_datasets(self, action_dataset, datasets_path="<your_local_path>", ):
        # 如果 datasets_path 存在， 直接读取
        if os.path.exists(datasets_path):

            self.fast_tokenizer = AutoProcessor.from_pretrained(
            datasets_path, trust_remote_code=True
        )
            return
        else:
            # 如果不存在，Fit the tokenizer on the new dataset
            new_tokenizer = self.fast_tokenizer.tokenizer.fit(action_dataset)
            self.fast_tokenizer = new_tokenizer

            # Save the new tokenizer, optionally push it to the Hugging Face model hub
            self.fast_tokenizer.save_pretrained(datasets_path)


def get_action_model(config=None):
    """
    Factory: build ActionModel from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).
    Returns:
        ActionModel: Initialized diffusion action head.
    """
    action_model = Fast_Action_Tokenizer()

    return action_model


def start_debugpy_once():
    """start debugpy once"""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10094))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10094 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True

if __name__ == "__main__":

    start_debugpy_once()

    fast_tokenizer_name = "physical-intelligence/fast"
    fast_tokenizer = Fast_Action_Tokenizer(fast_tokenizer_name=fast_tokenizer_name)
    raw_actions = [np.random.randn(16, 7), np.random.randn(16, 7)]

    # Load the tokenizer from the Hugging Face hub
    tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_name, trust_remote_code=True)

    # basic test
    # Tokenize & decode action chunks (we use dummy data here)
    action_data = np.random.rand(2, 16, 7)    # one batch of action chunks
    tokens = tokenizer(action_data)              # tokens = list[int]
    decoded_actions = tokenizer.decode(tokens)

    # self func test
    vlm_tokens = fast_tokenizer.encoder_action2vlmtoken(raw_actions)
    print(vlm_tokens)
    pred_actions = fast_tokenizer.decoder_action(np.array([12,3,45,87]))
    print(pred_actions)


