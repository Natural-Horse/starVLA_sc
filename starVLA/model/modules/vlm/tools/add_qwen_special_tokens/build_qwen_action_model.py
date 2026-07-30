from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from starVLA.model.modules.vlm.special_tokens import (
    action_tokens,
    initialize_token_embeddings,
    resolve_action_token_range,
    token_id,
)


DEFAULT_ROUTER_TOKENS = ["<|pred_action|>", "<|pred_bbox|>"]


def parse_token_file(path: Path) -> list[str]:
    tokens: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if token:
            tokens.append(token)
    return tokens


def unique_keep_order(tokens: list[str]) -> list[str]:
    seen = set()
    out = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def torch_dtype_from_name(name: str):
    if name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_qwen_vl_model(
    model_id: str,
    *,
    model_type: str,
    dtype: Any,
    attn_implementation: str | None,
    device_map: str | None,
):
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if device_map:
        kwargs["device_map"] = device_map

    if model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration

        if dtype != "auto":
            kwargs["dtype"] = dtype
        return Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)

    if model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        kwargs["torch_dtype"] = dtype
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)

    if model_type in {"qwen3_5_vl", "qwen3_5"}:
        from transformers import Qwen3_5ForConditionalGeneration

        kwargs["torch_dtype"] = dtype
        return Qwen3_5ForConditionalGeneration.from_pretrained(model_id, **kwargs)

    raise ValueError(f"Unsupported model_type={model_type!r}. Expected qwen2_5_vl, qwen3_vl, or qwen3_5_vl.")


def ensure_output_dir(save_dir: Path, overwrite: bool) -> None:
    if save_dir.exists() and any(save_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{save_dir} already exists and is not empty. Use --overwrite to replace it.")
    save_dir.mkdir(parents=True, exist_ok=True)


def resize_embeddings_to_cover(model: torch.nn.Module, tokenizer: Any, ids: list[int]) -> None:
    current_size = model.get_input_embeddings().weight.shape[0]
    required_size = max([current_size, len(tokenizer)] + [idx + 1 for idx in ids])
    if required_size > current_size:
        model.resize_token_embeddings(required_size)


def tokenizes_as_single_id(tokenizer: Any, token: str, expected_id: int) -> bool:
    encoded = tokenizer(token, add_special_tokens=False).input_ids
    return len(encoded) == 1 and int(encoded[0]) == int(expected_id)


def save_processor(model_id: str, tokenizer: Any, save_dir: Path, padding_side: str) -> None:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.tokenizer = tokenizer
    processor.tokenizer.padding_side = padding_side
    processor.save_pretrained(save_dir)


def main() -> None:
    default_fast_tokens = Path(__file__).with_name("fast_tokens.txt")
    parser = argparse.ArgumentParser(description="Build a local Qwen-VL ActionRouter base model.")
    parser.add_argument("--model-id", required=True, help="Source Qwen-VL model path or HF/ModelScope cache path.")
    parser.add_argument("--save-dir", required=True, help="Output directory for the generated ActionRouter model.")
    parser.add_argument("--fast-tokens-file", default=str(default_fast_tokens))
    parser.add_argument("--action-token-prefix", default="<robot_action_")
    parser.add_argument("--action-token-count", type=int, default=2048)
    parser.add_argument("--router-token", action="append", default=None)
    parser.add_argument("--no-router-tokens", action="store_true")
    parser.add_argument("--init-strategy", default="normal", choices=["normal", "avg", "zero", "none"])
    parser.add_argument("--padding-side", default="left", choices=["left", "right"])
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device-map", default=None, help="Optional HF device_map, e.g. auto, cuda, cpu.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    ensure_output_dir(save_dir, overwrite=args.overwrite)

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    model_type = str(getattr(config, "model_type", ""))
    print(f"[INFO] Source model: {args.model_id}")
    print(f"[INFO] Detected model_type: {model_type}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = args.padding_side

    model = load_qwen_vl_model(
        args.model_id,
        model_type=model_type,
        dtype=torch_dtype_from_name(args.dtype),
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )

    if Path(args.fast_tokens_file).exists():
        fast_tokens = parse_token_file(Path(args.fast_tokens_file))
    else:
        fast_tokens = action_tokens(prefix=args.action_token_prefix, count=args.action_token_count)

    router_tokens = [] if args.no_router_tokens else (args.router_token or DEFAULT_ROUTER_TOKENS)
    requested_tokens = unique_keep_order(fast_tokens + router_tokens)
    missing_tokens = [token for token in requested_tokens if token not in tokenizer.get_vocab()]

    old_tokenizer_len = len(tokenizer)
    old_embedding_size = int(model.get_input_embeddings().weight.shape[0])
    print(f"[INFO] tokenizer len before: {old_tokenizer_len}")
    print(f"[INFO] embedding rows before: {old_embedding_size}")
    print(f"[INFO] requested tokens: {len(requested_tokens)}")
    print(f"[INFO] missing tokens to add: {len(missing_tokens)}")

    added = 0
    if missing_tokens:
        added = tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})

    mapping = {token: token_id(tokenizer, token) for token in requested_tokens}
    if any(idx is None for idx in mapping.values()):
        bad = [token for token, idx in mapping.items() if idx is None]
        raise RuntimeError(f"Failed to map token(s) after adding: {bad[:10]}")

    mapped_ids = [int(idx) for idx in mapping.values() if idx is not None]
    resize_embeddings_to_cover(model, tokenizer, mapped_ids)

    added_ids = [int(mapping[token]) for token in missing_tokens if mapping[token] is not None]
    initialize_token_embeddings(model, added_ids, init_strategy=args.init_strategy)

    action_range = resolve_action_token_range(
        tokenizer,
        prefix=args.action_token_prefix,
        count=args.action_token_count,
        require=True,
    )
    assert action_range is not None

    for idx in range(args.action_token_count):
        token = action_tokens(prefix=args.action_token_prefix, count=args.action_token_count)[idx]
        expected_id = int(mapping[token])
        if not tokenizes_as_single_id(tokenizer, token, expected_id):
            raise RuntimeError(f"{token} does not tokenize as a single id {expected_id}.")

    for token in router_tokens:
        expected_id = int(mapping[token])
        if not tokenizes_as_single_id(tokenizer, token, expected_id):
            raise RuntimeError(f"{token} does not tokenize as a single id {expected_id}.")

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    save_processor(args.model_id, tokenizer, save_dir, args.padding_side)

    metadata = {
        "source_model": args.model_id,
        "model_type": model_type,
        "tokenizer_len_before": old_tokenizer_len,
        "tokenizer_len_after": len(tokenizer),
        "embedding_rows_before": old_embedding_size,
        "embedding_rows_after": int(model.get_input_embeddings().weight.shape[0]),
        "added_token_count": int(added),
        "init_strategy": args.init_strategy,
        "action_token_prefix": args.action_token_prefix,
        "action_token_count": args.action_token_count,
        "action_token_min": int(action_range[0]),
        "action_token_max": int(action_range[1]),
        "router_tokens": router_tokens,
        "router_token_ids": {token: int(mapping[token]) for token in router_tokens},
    }
    (save_dir / "action_token_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (save_dir / "added_custom_token_id_map.json").write_text(
        json.dumps({token: int(idx) for token, idx in mapping.items() if idx is not None}, indent=2),
        encoding="utf-8",
    )

    reloaded_tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)
    reloaded_range = resolve_action_token_range(
        reloaded_tokenizer,
        prefix=args.action_token_prefix,
        count=args.action_token_count,
        require=True,
    )
    if reloaded_range != action_range:
        raise RuntimeError(f"Reloaded action token range changed: {reloaded_range} != {action_range}")
    AutoProcessor.from_pretrained(save_dir, trust_remote_code=True)

    print(f"[OK] Saved ActionRouter model to: {save_dir}")
    print(f"[OK] FAST action token range: [{action_range[0]}, {action_range[1]}]")
    if router_tokens:
        print(f"[OK] Router token ids: {metadata['router_token_ids']}")


if __name__ == "__main__":
    main()
