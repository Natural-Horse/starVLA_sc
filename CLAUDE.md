# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StarVLA is a modular framework for developing Vision-Language-Action (VLA) models. It pairs swappable VLM backbones (Qwen2.5-VL, Qwen3-VL, Qwen3.5-VL, Florence-2, Cosmos-Reason2) with pluggable action heads (Flow Matching/DiT, MLP, FAST tokenizer, VLA-Adapter) in a Lego-like architecture.

## Build & Lint Commands

```bash
pip install -e .                    # Install package
make check                          # Lint check (black --check + ruff check), does NOT modify files
make autoformat                     # Format in-place (black + ruff --fix)
make clean                          # Remove .pyc / __pycache__
```

- **Code style**: black (line-length 121, target py310), ruff (select A,B,E,F,I,RUF,W; ignore F722)
- No formal test suite exists. Each framework file has a `if __name__ == "__main__"` smoke test.

## Training Entry Points

Each training strategy is a standalone script under `starVLA/training/`:

| Script | Purpose |
|--------|---------|
| `train_starvla.py` | Basic single-dataset VLA training |
| `train_starvla_cotrain.py` | Cotraining VLA + VLM-bbox + VLM-signal |
| `train_starvla_cotrain_router.py` | Unified router cotraining (WallX) |
| `train_starvlm.py` | VLM-only pretraining |
| `train_starvlm_cotrain_vlm.py` | VLM cotraining |

Launch via: `torchrun --nproc_per_node=N starVLA/training/train_starvla_cotrain.py --config_yaml path/to/config.yaml`

CLI overrides use OmegaConf dotlist: `--framework.qwenvl.base_vlm Qwen/Qwen3-VL-4B-Instruct`

## Architecture

### Config System

Single OmegaConf YAML file per experiment (in `starVLA/config/training/`). DeepSpeed configs live in `starVLA/config/deepseeds/`. All training hyperparameters, module-specific learning rates, and freeze lists are in the YAML.

### Framework Registration Pattern

Frameworks register via `@FRAMEWORK_REGISTRY.register("Name")` in their module file. The `framework/__init__.py` auto-imports all modules in the directory to trigger registration. `build_framework(cfg)` resolves by `cfg.framework.name`.

When adding a new framework:
1. Create `starVLA/model/framework/YourFramework.py`
2. Decorate the class with `@FRAMEWORK_REGISTRY.register("YourName")`
3. The auto-import in `__init__.py` picks it up automatically
4. Set `framework.name: "YourName"` in your training YAML

### Model Frameworks

Each framework implements `forward(examples: List[dict]) -> dict` (training) and `predict_action(examples: List[dict]) -> dict` (inference). The input dict keys are `image` (PIL), `lang` (str), `action` (np.ndarray), optionally `state` (np.ndarray).

| Framework | VLM Backbone | Action Head | Method |
|-----------|-------------|-------------|--------|
| QwenGR00T | Qwen VL | FlowmatchingActionHead (DiT) | Dual-system flow matching |
| QwenPI | Qwen VL | LayerwiseFlowmatchingActionHead | Layer-wise cross-attention flow matching |
| QwenFast | Qwen VL | FAST tokenizer | Autoregressive discrete tokens |
| QwenOFT | Qwen VL | MLP | Parallel decode L1 regression |

### Dataloader Factory

`build_dataloader(cfg, dataset_py=...)` in `starVLA/dataloader/__init__.py` dispatches by dataset type string. Datasets return raw dicts (no model-specific preprocessing). All tokenization/encoding happens inside the framework's `forward()`.

WallX datasets (`wallx_cotrain_datasets.py`) read LeRobot v2 parquet format, support episode subsetting, history keyframes, photometric augmentation, and quantile-based action normalization.

### Per-Module Learning Rates & Freezing

- `cfg.trainer.learning_rate` is a dict mapping module name patterns to LR values
- `cfg.trainer.freeze_modules` is a comma-separated list of module paths to freeze (e.g., `"qwen_vl_interface.model.model.visual"`)
- `auto_get_trainable_modules()` in `tools.py` provides introspection

### Inference/Serving

- WebSocket server: `deployment/model_server/server_policy.py` loads via `baseframework.from_pretrained(checkpoint_path)`
- WallX closed-loop: `scripts/serve_wallx_closed_loop.py` (VQA, bbox routing, multi-GPU, msgpack)
- Open-loop eval: `scripts/eval_open_loop_wallx.py`

## Key Directories

- `starVLA/model/framework/` — VLA model implementations (the core)
- `starVLA/model/modules/vlm/` — VLM backbone wrappers
- `starVLA/model/modules/action_model/` — Action head implementations
- `starVLA/dataloader/` — Dataset classes (model-agnostic)
- `starVLA/training/trainer_utils/` — Trainer helpers, param/LR group builders
- `starVLA/config/training/` — Experiment YAML configs
- `examples/` — Benchmark-specific scripts (LIBERO, SimplerEnv, RoboCasa, etc.)
- `scripts/` — Serving and eval scripts
- `deployment/` — WebSocket serving infrastructure
