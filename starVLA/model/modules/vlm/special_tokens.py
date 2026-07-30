from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def collect_router_tokens(config: Any) -> list[str]:
    tokens: list[str] = []
    datasets_cfg = cfg_get(config, "datasets", None)
    for section_name in ("router_data", "vlm_signal_data"):
        section_cfg = cfg_get(datasets_cfg, section_name, None)
        if section_cfg is None:
            continue
        for token_key in ("pred_action_token", "pred_bbox_token"):
            token = cfg_get(section_cfg, token_key, None)
            if token:
                tokens.append(str(token))

    qwenvl_cfg = cfg_get(cfg_get(config, "framework", None), "qwenvl", None)
    special_cfg = cfg_get(qwenvl_cfg, "special_tokens", None)
    for token in cfg_get(special_cfg, "router_tokens", []) or []:
        if token:
            tokens.append(str(token))

    unique_tokens: list[str] = []
    for token in tokens:
        if token and token not in unique_tokens:
            unique_tokens.append(token)
    return unique_tokens


def get_special_tokens_cfg(config: Any) -> Any:
    framework_cfg = cfg_get(config, "framework", None)
    qwenvl_cfg = cfg_get(framework_cfg, "qwenvl", None)
    return cfg_get(qwenvl_cfg, "special_tokens", None)


def token_exists(tokenizer: Any, token: str) -> bool:
    return token in tokenizer.get_vocab()


def token_id(tokenizer: Any, token: str) -> int | None:
    idx = tokenizer.convert_tokens_to_ids(token)
    if idx is None:
        return None
    if isinstance(idx, int) and idx < 0:
        return None
    return int(idx)


def action_token(prefix: str, index: int) -> str:
    if prefix.endswith("{i}"):
        return prefix.format(i=index)
    return f"{prefix}{index}>"


def action_tokens(prefix: str = "<robot_action_", count: int = 2048) -> list[str]:
    return [action_token(prefix, idx) for idx in range(int(count))]


def resolve_action_token_range(
    tokenizer: Any,
    *,
    prefix: str = "<robot_action_",
    count: int = 2048,
    require: bool = False,
) -> tuple[int, int] | None:
    tokens = action_tokens(prefix=prefix, count=count)
    ids = [token_id(tokenizer, token) for token in tokens]
    missing = [token for token, idx in zip(tokens, ids) if idx is None]
    if missing:
        if require:
            preview = ", ".join(missing[:5])
            more = "" if len(missing) <= 5 else f", ... ({len(missing)} missing)"
            raise ValueError(f"Missing FAST action tokens: {preview}{more}")
        return None

    assert all(idx is not None for idx in ids)
    int_ids = [int(idx) for idx in ids]
    expected = list(range(int_ids[0], int_ids[0] + len(int_ids)))
    if int_ids != expected:
        if require:
            raise ValueError(
                "FAST action token ids must be contiguous and ordered. "
                f"First ids={int_ids[:5]}, last ids={int_ids[-5:]}"
            )
        return None
    return int_ids[0], int_ids[-1]


@dataclass
class PreparedSpecialTokens:
    router_tokens: list[str]
    added_tokens: list[str]
    action_token_range: tuple[int, int] | None
    policy: str


def _embedding_modules(model: torch.nn.Module) -> list[torch.nn.Embedding]:
    modules: list[torch.nn.Embedding] = []
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None:
        modules.append(input_embeddings)
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and output_embeddings is not input_embeddings:
        modules.append(output_embeddings)
    return modules


def _resize_embeddings_to_cover_token_ids(model: torch.nn.Module, tokenizer: Any, token_ids: list[int]) -> None:
    current_size = model.get_input_embeddings().weight.shape[0]
    required_size = max([current_size, len(tokenizer)] + [idx + 1 for idx in token_ids])
    if required_size > current_size:
        model.resize_token_embeddings(required_size)


def initialize_token_embeddings(
    model: torch.nn.Module,
    token_ids: list[int],
    *,
    init_strategy: str = "normal",
) -> None:
    if not token_ids:
        return
    unique_ids = sorted(set(int(idx) for idx in token_ids))
    with torch.no_grad():
        for embedding in _embedding_modules(model):
            weight = embedding.weight
            valid_ids = [idx for idx in unique_ids if 0 <= idx < weight.shape[0]]
            if not valid_ids:
                continue
            if init_strategy == "avg":
                ref_vec = weight.mean(dim=0)
                for idx in valid_ids:
                    weight[idx].copy_(ref_vec)
            elif init_strategy == "zero":
                for idx in valid_ids:
                    weight[idx].zero_()
            elif init_strategy == "normal":
                for idx in valid_ids:
                    torch.nn.init.normal_(weight[idx], mean=0.0, std=0.02)
            elif init_strategy == "none":
                continue
            else:
                raise ValueError(f"Unknown token init strategy: {init_strategy}")


def prepare_qwen_special_tokens(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    config: Any,
    logger: Any = None,
    default_policy: str = "auto_add",
) -> PreparedSpecialTokens:
    special_cfg = get_special_tokens_cfg(config)
    policy = str(cfg_get(special_cfg, "policy", default_policy))
    if policy not in {"auto_add", "strict", "none"}:
        raise ValueError(f"Unsupported framework.qwenvl.special_tokens.policy={policy!r}")

    router_tokens = collect_router_tokens(config)
    missing_router_tokens = [token for token in router_tokens if not token_exists(tokenizer, token)]
    added_tokens: list[str] = []

    if missing_router_tokens and policy == "strict":
        raise ValueError(
            "Tokenizer is missing router special tokens under strict policy: "
            f"{missing_router_tokens}. Generate an ActionRouter base model first or set policy=auto_add."
        )

    if missing_router_tokens and policy == "auto_add":
        added = tokenizer.add_special_tokens({"additional_special_tokens": missing_router_tokens})
        added_tokens = list(missing_router_tokens)
        added_ids = [token_id(tokenizer, token) for token in added_tokens]
        added_ids = [idx for idx in added_ids if idx is not None]
        _resize_embeddings_to_cover_token_ids(model, tokenizer, added_ids)
        init_strategy = str(cfg_get(special_cfg, "init_strategy", "normal"))
        initialize_token_embeddings(model, added_ids, init_strategy=init_strategy)
        if logger is not None:
            logger.info("Added %d router token(s) to tokenizer: %s", added, added_tokens)

    require_action = bool(cfg_get(special_cfg, "require_fast_action_tokens", False))
    action_prefix = str(cfg_get(special_cfg, "action_token_prefix", "<robot_action_"))
    action_count = int(cfg_get(special_cfg, "action_token_count", 2048))
    action_range = resolve_action_token_range(
        tokenizer,
        prefix=action_prefix,
        count=action_count,
        require=require_action,
    )

    return PreparedSpecialTokens(
        router_tokens=router_tokens,
        added_tokens=added_tokens,
        action_token_range=action_range,
        policy=policy,
    )
