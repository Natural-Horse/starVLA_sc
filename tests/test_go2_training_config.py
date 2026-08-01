import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from starVLA.training.train_starvla_cotrain_router import (
    _apply_router_framework_overrides,
    _validate_go2_training_config,
    build_accelerator,
)
from starVLA.training.trainer_utils.trainer_tools import adapt_padded_vocab_state_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
GO2_CONFIG = REPO_ROOT / "starVLA/config/training/starvla_go2_qwen3vl_waypoint_router.yaml"


class Go2TrainingConfigTest(unittest.TestCase):
    def test_legacy_qwen_vocab_rows_expand_without_moving_token_rows(self):
        class FakeTokenizer:
            def __len__(self):
                return 5

        class FakeQwen(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = torch.nn.Module()
                self.language_model.embed_tokens = torch.nn.Embedding(8, 3)
                self.lm_head = torch.nn.Linear(3, 8, bias=False)
                self.processor = SimpleNamespace(tokenizer=FakeTokenizer())

        module = FakeQwen()
        old_embed = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        old_head = old_embed + 100.0
        adapted, keys = adapt_padded_vocab_state_dict(
            module,
            {
                "language_model.embed_tokens.weight": old_embed,
                "lm_head.weight": old_head,
            },
        )
        self.assertEqual(len(keys), 2)
        torch.testing.assert_close(
            adapted["language_model.embed_tokens.weight"][:5], old_embed
        )
        torch.testing.assert_close(adapted["lm_head.weight"][:5], old_head)
        self.assertEqual(tuple(adapted["lm_head.weight"].shape), (8, 3))

    def test_checked_in_config_contract(self):
        cfg = OmegaConf.load(GO2_CONFIG)
        _apply_router_framework_overrides(cfg)
        _validate_go2_training_config(cfg)

    def test_horizon_mismatch_is_rejected(self):
        cfg = OmegaConf.load(GO2_CONFIG)
        cfg.datasets.router_data.action_horizon = 5
        with self.assertRaisesRegex(ValueError, "action horizon mismatch"):
            _validate_go2_training_config(cfg)

    def test_enabled_bbox_training_is_rejected(self):
        cfg = OmegaConf.load(GO2_CONFIG)
        cfg.datasets.router_data.bbox.train_enabled = True
        with self.assertRaisesRegex(ValueError, "bbox"):
            _validate_go2_training_config(cfg)

    def test_missing_route_token_is_rejected(self):
        cfg = OmegaConf.load(GO2_CONFIG)
        del cfg.framework.qwenvl.special_tokens.router_tokens[0]
        with self.assertRaisesRegex(ValueError, "missing"):
            _validate_go2_training_config(cfg)

    def test_unknown_include_route_is_rejected(self):
        cfg = OmegaConf.load(GO2_CONFIG)
        cfg.datasets.router_data.include_routes = ["nav", "bbox"]
        with self.assertRaisesRegex(ValueError, "include_routes"):
            _validate_go2_training_config(cfg)

    def test_vlm_stage_contract(self):
        cfg = OmegaConf.load(GO2_CONFIG)
        cfg.trainer.stage = "vlm"
        cfg.framework.qwenvl.freeze = False
        cfg.framework.action_model.freeze = True
        cfg.framework.router.action_loss_grad_to_vlm = False
        cfg.trainer.loss_scale.vlm = 1.0
        cfg.trainer.loss_scale.action = 0.0
        _apply_router_framework_overrides(cfg)
        _validate_go2_training_config(cfg)

    def test_action_stage_requires_checkpoint(self):
        cfg = OmegaConf.load(GO2_CONFIG)
        cfg.trainer.stage = "action"
        cfg.framework.qwenvl.freeze = True
        cfg.framework.action_model.freeze = False
        cfg.framework.router.action_loss_grad_to_vlm = False
        cfg.trainer.loss_scale.vlm = 0.0
        cfg.trainer.loss_scale.action = 1.0
        with self.assertRaisesRegex(ValueError, "pretrained_checkpoint"):
            _validate_go2_training_config(cfg)

    def test_action_stage_contract(self):
        cfg = OmegaConf.load(GO2_CONFIG)
        cfg.trainer.stage = "action"
        cfg.framework.qwenvl.freeze = True
        cfg.framework.action_model.freeze = False
        cfg.framework.router.action_loss_grad_to_vlm = False
        cfg.trainer.loss_scale.vlm = 0.0
        cfg.trainer.loss_scale.action = 1.0
        cfg.trainer.pretrained_checkpoint = "/path/validated/by/launcher.pt"
        _apply_router_framework_overrides(cfg)
        _validate_go2_training_config(cfg)

    @patch("starVLA.training.train_starvla_cotrain_router.Accelerator")
    @patch("starVLA.training.train_starvla_cotrain_router.DeepSpeedPlugin")
    def test_deepspeed_uses_configured_gradient_clipping(self, plugin_cls, accelerator_cls):
        cfg = OmegaConf.load(GO2_CONFIG)
        build_accelerator(cfg)
        self.assertEqual(plugin_cls.call_args.kwargs["gradient_clipping"], 5.0)
        accelerator_cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()
