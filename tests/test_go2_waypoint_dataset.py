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
    _build_episode_instruction,
    _first_turn_direction,
    _load_task_instructions,
)


class Go2TaskInstructionUnitTest(unittest.TestCase):
    def test_episode_instruction_contains_box_directions(self):
        stages = np.asarray(
            ["nav_to_pick", "nav_to_pick", "pick", "nav_to_place", "nav_to_place"],
            dtype=object,
        )
        instructions = np.asarray(
            [
                "Pick up the coke can from the box in front of you.",
                "Turn toward your front-left to find the box with the coke can.",
                "Pick up the coke can from the box in front of you.",
                "Turn toward your back-right to find the box where you can place the coke can.",
                "Turn toward your front to find the box where you can place the coke can.",
            ],
            dtype=object,
        )

        self.assertEqual(
            _build_episode_instruction("Move the coke can.", stages, instructions),
            "Move the coke can. Box1 is to the robot's front-left from its initial pose. "
            "Box2 is to the robot's back-right from its first pose after grasping.",
        )

    def test_missing_direction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nav_to_pick"):
            _first_turn_direction(
                np.asarray(["nav_to_pick"], dtype=object),
                np.asarray(["Pick up the object."], dtype=object),
                stage="nav_to_pick",
            )

    def test_existing_matching_global_instruction_is_not_duplicated(self):
        stages = np.asarray(["nav_to_pick", "nav_to_place"], dtype=object)
        instructions = np.asarray(
            [
                "Turn toward your left to find the box with the coke can.",
                "Turn toward your back to find the box where you can place the coke can.",
            ],
            dtype=object,
        )
        task = (
            "Move the coke can. Box1 is to the robot's left from its initial pose. "
            "Box2 is to the robot's back from its first pose after grasping."
        )
        self.assertEqual(_build_episode_instruction(task, stages, instructions), task)

    def test_existing_mismatched_global_instruction_is_rejected(self):
        stages = np.asarray(["nav_to_pick", "nav_to_place"], dtype=object)
        instructions = np.asarray(
            [
                "Turn toward your left to find the box with the coke can.",
                "Turn toward your back to find the box where you can place the coke can.",
            ],
            dtype=object,
        )
        task = (
            "Move the coke can. Box1 is to the robot's right from its initial pose. "
            "Box2 is to the robot's back from its first pose after grasping."
        )
        with self.assertRaisesRegex(ValueError, "disagree"):
            _build_episode_instruction(task, stages, instructions)

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
                global_instruction=f"Task {task_index} with directions.",
            )

        dataset.episodes = {0: episode(0, 7), 1: episode(1, 2)}
        self.assertEqual(dataset[0]["lang"], "Main task: Task 7 with directions.")
        self.assertEqual(dataset[0]["subtask_text"], "Pick the object.")
        self.assertEqual(dataset[0]["phase_label"], "arm_contact")
        self.assertEqual(dataset[0]["task_index"], 7)
        self.assertEqual(dataset[1]["lang"], "Main task: Task 2 with directions.")
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
        self.assertIn(sample["global_instruction"], sample["lang"])
        self.assertIn("Box1 is to the robot's", sample["global_instruction"])
        self.assertIn("Box2 is to the robot's", sample["global_instruction"])
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
        self.assertIn(sample["phase_label"], {"arm_approach", "arm_contact", "arm_retreat"})
        self.assertEqual(sample["subtask_text"], "Pick up the coke can from the box in front of you.")

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
