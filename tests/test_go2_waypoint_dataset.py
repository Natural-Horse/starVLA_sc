import os
import unittest
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf

from starVLA.dataloader.go2_waypoint_dataset import Go2WaypointRouterDataset


class Go2WaypointDatasetIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get("GO2_TEST_DATASET_ROOT")
        if not root or not Path(root).exists():
            raise unittest.SkipTest("GO2_TEST_DATASET_ROOT is not available")
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
        self.assertEqual(sample["action"].shape, (4, 3))
        self.assertEqual(sample["action_mask"].shape, (4,))
        self.assertEqual(sample["state"].shape, (1, 3))
        self.assertEqual([image.size for image in sample["image"]], [(96, 96), (96, 96)])
        self.assertTrue(sample["solution"].startswith("<|nav|>"))

    def test_non_nav_has_no_continuous_target(self):
        grasp_index = next(
            idx
            for idx, (episode, frame) in enumerate(self.dataset.samples)
            if self.dataset._route_for_frame(self.dataset.episodes[episode], frame) == "grasp"
        )
        sample = self.dataset[grasp_index]
        self.assertNotIn("action", sample)
        self.assertNotIn("action_mask", sample)


if __name__ == "__main__":
    unittest.main()
