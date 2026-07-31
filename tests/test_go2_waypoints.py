import unittest

import numpy as np

from deployment.go2.waypoint_adapter import WaypointAdapter, WaypointAdapterConfig
from starVLA.dataloader.go2_waypoints import (
    WaypointExtractionConfig,
    extract_sparse_waypoints,
    future_waypoint_chunk,
    world_waypoints_to_body,
)


class SparseWaypointTest(unittest.TestCase):
    def test_straight_path_is_sparse_and_bounded(self):
        poses = np.stack((np.linspace(0.0, 2.0, 101), np.zeros(101), np.zeros(101)), axis=-1)
        cfg = WaypointExtractionConfig(max_translation_m=0.5)
        indices, waypoints = extract_sparse_waypoints(poses, cfg)

        self.assertLess(len(indices), 10)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 100)
        self.assertLessEqual(np.diff(waypoints[:, 0]).max(), 0.52)

    def test_in_place_turn_keeps_discrete_yaw_targets(self):
        poses = np.stack(
            (np.zeros(101), np.zeros(101), np.linspace(0.0, np.pi, 101)), axis=-1
        )
        cfg = WaypointExtractionConfig(max_yaw_rad=np.deg2rad(25.0))
        _, waypoints = extract_sparse_waypoints(poses, cfg)

        unwrapped = np.unwrap(waypoints[:, 2])
        self.assertGreater(len(waypoints), 2)
        self.assertLess(len(waypoints), 20)
        self.assertLessEqual(np.diff(unwrapped).max(), np.deg2rad(27.0))

    def test_body_transform_and_future_mask(self):
        poses = np.array([[1.0, 2.0, np.pi / 2], [1.0, 3.0, np.pi / 2], [1.0, 4.0, np.pi]])
        body = world_waypoints_to_body(poses[0], poses[1:])
        np.testing.assert_allclose(body[0], [1.0, 0.0, 0.0], atol=1e-6)

        chunk, mask = future_waypoint_chunk(poses, 1, np.array([0, 1, 2]), horizon=4)
        self.assertEqual(chunk.shape, (4, 3))
        np.testing.assert_array_equal(mask, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(chunk[0], chunk[-1])


class WaypointAdapterTest(unittest.TestCase):
    def test_nonholonomic_adapter_anchors_and_stops(self):
        adapter = WaypointAdapter(WaypointAdapterConfig(holonomic=False))
        adapter.set_waypoints(np.array([[1.0, 0.2, 0.0]]), np.array([0.0, 0.0, 0.0]))

        command = adapter.compute_command(np.array([0.0, 0.0, 0.0]), dt_s=0.02)
        self.assertGreater(command[0], 0.0)
        self.assertEqual(command[1], 0.0)
        self.assertGreater(command[2], 0.0)

        stopped = adapter.compute_command(np.array([1.0, 0.2, 0.0]), dt_s=0.02)
        np.testing.assert_allclose(stopped, np.zeros(3))
        self.assertTrue(adapter.goal_reached)

    def test_behind_target_rotates_without_forward_motion(self):
        adapter = WaypointAdapter(WaypointAdapterConfig(holonomic=False))
        adapter.set_waypoints(np.array([[-1.0, 0.0, np.pi]]), np.array([0.0, 0.0, 0.0]))
        command = adapter.compute_command(np.array([0.0, 0.0, 0.0]), dt_s=0.1)
        self.assertAlmostEqual(command[0], 0.0)
        self.assertNotEqual(command[2], 0.0)


if __name__ == "__main__":
    unittest.main()
