import os
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from starVLA.dataloader.go2_waypoint_dataset import (
    DEFAULT_ROUTE_TOKENS,
    Go2WaypointRouterDataset,
    _Episode,
    _load_task_instructions,
)


class Go2TaskInstructionUnitTest(unittest.TestCase):
    def test_task_metadata_uses_explicit_task_indices(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_path = Path(temp_dir) / "tasks.jsonl"
            rows = [
                {"task_index": 7, "task": "Task seven."},
                {"task_index": 2, "task": "Task two."},
            ]
            tasks_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(_load_task_instructions(tasks_path), {7: "Task seven.", 2: "Task two."})

    def test_sample_prompt_uses_its_own_task_index(self):
        dataset = Go2WaypointRouterDataset.__new__(Go2WaypointRouterDataset)
        dataset.samples = [(0, 0), (1, 0)]
        dataset.task_instructions = {7: "Task seven.", 2: "Task two."}
        dataset.router_prompt = "Main task: {instruction}"
        dataset.route_tokens = dict(DEFAULT_ROUTE_TOKENS)
        dataset.subtask_start_token = "<|subtask|>"
        dataset.subtask_end_token = "<|end_subtask|>"
        dataset.include_state = True
        dataset.action_horizon = 4
        dataset._image = lambda *_: Image.new("RGB", (8, 8))

        def episode(episode_index, task_index):
            return _Episode(
                episode_index=episode_index,
                task_indices=np.asarray([task_index]),
                poses=np.zeros((1, 3)),
                base_velocity=np.zeros((1, 3), dtype=np.float32),
                actions=np.zeros((1, 10), dtype=np.float32),
                stages=np.asarray(["pick"], dtype=object),
                subtasks=np.asarray(["arm_contact"], dtype=object),
                instructions=np.asarray(["Pick the object."], dtype=object),
                done=np.asarray([False]),
                waypoint_indices_by_frame={},
            )

        dataset.episodes = {0: episode(0, 7), 1: episode(1, 2)}
        self.assertEqual(dataset[0]["lang"], "Main task: Task seven.")
        self.assertEqual(dataset[0]["task_index"], 7)
        self.assertEqual(dataset[1]["lang"], "Main task: Task two.")
        self.assertEqual(dataset[1]["task_index"], 2)


class Go2WaypointDatasetIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get("GO2_TEST_DATASET_ROOT")
        if not root or not Path(root).exists():
            raise unittest.SkipTest("GO2_TEST_DATASET_ROOT is not available")
        cls.dataset_root = Path(root)
        cls.dataset = Go2WaypointRouterDataset(
            OmegaConf.create(
                {
                    "root": root,
                    "image_size": [96, 96],
                    "action_horizon": 4,
                    "include_state": True,
                    "episode_start": 0,
                    "num_episodes": 1,
                    "done_repeat": 1,
                    "bbox": {"train_enabled": False},
                }
            )
        )

    def test_routes_and_nav_contract(self):
        counts = Counter(
            self.dataset._route_for_frame(self.dataset.episodes[episode], frame)
            for episode, frame in self.dataset.samples
        )
        self.assertGreater(counts["nav"], 0)
        self.assertGreater(counts["grasp"], 0)
        self.assertGreater(counts["place"], 0)
        self.assertEqual(counts["done"], 1)
        self.assertEqual(counts["recover"], 0)

        nav_index = next(
            idx
            for idx, (episode, frame) in enumerate(self.dataset.samples)
            if self.dataset._route_for_frame(self.dataset.episodes[episode], frame) == "nav"
        )
        sample = self.dataset[nav_index]
        expected_task = self.dataset.task_instructions[sample["task_index"]]
        self.assertIn(expected_task, sample["lang"])
        self.assertEqual(sample["action"].shape, (4, 10))
        self.assertEqual(sample["action_mask"].shape, (4,))
        self.assertEqual(sample["action_dim_mask"].shape, (10,))
        np.testing.assert_array_equal(sample["action_dim_mask"], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(sample["state"].shape, (1, 10))
        self.assertEqual([image.size for image in sample["image"]], [(96, 96), (96, 96)])
        self.assertTrue(sample["solution"].startswith("<|nav|>"))

    def test_grasp_has_cartesian_arm_target(self):
        grasp_index = next(
            idx
            for idx, (episode, frame) in enumerate(self.dataset.samples)
            if self.dataset._route_for_frame(self.dataset.episodes[episode], frame) == "grasp"
        )
        sample = self.dataset[grasp_index]
        self.assertEqual(sample["action"].shape, (4, 10))
        self.assertEqual(sample["action_mask"].shape, (4,))
        np.testing.assert_array_equal(sample["action_dim_mask"], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        self.assertIn(sample["subtask_text"], {"arm_approach", "arm_contact", "arm_retreat"})

    def test_route_filter_keeps_only_requested_routes(self):
        cfg = OmegaConf.create(
            {
                "root": str(self.dataset_root),
                "image_size": [96, 96],
                "action_horizon": 4,
                "episode_start": 0,
                "num_episodes": 2,
                "include_routes": ["grasp", "place"],
            }
        )
        filtered = Go2WaypointRouterDataset(cfg)
        self.assertTrue(filtered.samples)
        routes = {
            filtered._route_for_frame(filtered.episodes[episode], frame)
            for episode, frame in filtered.samples
        }
        self.assertEqual(routes, {"grasp", "place"})


if __name__ == "__main__":
    unittest.main()
