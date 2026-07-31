import unittest
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.model.framework.QwenPI import Qwen_PI
from starVLA.training.train_starvla_cotrain_router import VLARouterTrainer


class _FakeActionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(transformer_blocks=[object(), object()])
        self.received_mask = None

    def forward(self, hidden, actions, state, action_mask=None):
        self.received_mask = action_mask.detach().cpu()
        return actions.square().mean()


def _config():
    return OmegaConf.create(
        {
            "framework": {
                "action_model": {"future_action_window_size": 3, "action_dim": 3, "state_dim": 3},
            },
            "datasets": {
                "router_data": {
                    "action_horizon": 4,
                    "include_state": True,
                    "main_routes": ["nav", "grasp", "place", "done", "recover"],
                    "route_tokens": {
                        "nav": "<|nav|>",
                        "grasp": "<|grasp|>",
                        "place": "<|place|>",
                        "done": "<|done|>",
                        "recover": "<|recover|>",
                    },
                    "pred_bbox_token": "<|pred_bbox|>",
                    "bbox": {
                        "implementation_enabled": True,
                        "allow_route_prediction": False,
                        "train_enabled": False,
                        "evaluation_enabled": False,
                    },
                }
            },
            "trainer": {"repeated_diffusion_steps": 2},
        }
    )


class Go2RouterModelContractTest(unittest.TestCase):
    def test_action_mask_reaches_flow_head(self):
        model = Qwen_PI.__new__(Qwen_PI)
        torch.nn.Module.__init__(model)
        model.config = _config()
        model.future_action_window_size = 3
        model.action_model = _FakeActionModel()
        hidden = [torch.zeros((1, 5, 8)), torch.zeros((1, 5, 8))]
        examples = [
            {
                "action": np.ones((4, 3), dtype=np.float32),
                "action_mask": np.array([1, 1, 0, 0], dtype=np.float32),
                "state": np.zeros((1, 3), dtype=np.float32),
            }
        ]

        loss = model.action_loss_from_hidden_states(hidden, examples)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(model.action_model.received_mask.shape), (2, 4))
        np.testing.assert_array_equal(
            model.action_model.received_mask.numpy(),
            [[True, True, False, False], [True, True, False, False]],
        )

    def test_main_route_map_excludes_bbox(self):
        model = Qwen_PI.__new__(Qwen_PI)
        model.config = _config()
        self.assertEqual(
            set(model._configured_route_tokens(allow_bbox=False)),
            {"nav", "grasp", "place", "done", "recover"},
        )
        self.assertNotIn("bbox", model._configured_route_tokens(allow_bbox=True))

    def test_trainer_filters_disabled_bbox_and_selects_only_actions(self):
        trainer = VLARouterTrainer.__new__(VLARouterTrainer)
        trainer.config = _config()
        batch = [
            {"route": "nav", "action": np.zeros((4, 3))},
            {"route": "grasp"},
            {"route": "bbox"},
        ]
        filtered = trainer._filter_bbox_samples(batch, training=True)
        self.assertEqual([item["route"] for item in filtered], ["nav", "grasp"])
        self.assertEqual(trainer._action_indices(filtered), [0])
        self.assertNotIn("bbox", trainer._route_counts(filtered))


if __name__ == "__main__":
    unittest.main()
