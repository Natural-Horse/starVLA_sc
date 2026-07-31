"""Convert sparse body-frame waypoints into bounded Go2 velocity commands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starVLA.dataloader.go2_waypoints import wrap_to_pi


@dataclass(frozen=True)
class WaypointAdapterConfig:
    holonomic: bool = False
    linear_gain: float = 0.8
    lateral_gain: float = 0.8
    heading_gain: float = 1.5
    final_yaw_gain: float = 1.0
    max_vx_mps: float = 0.45
    max_vy_mps: float = 0.20
    max_wz_rps: float = 0.80
    max_linear_accel_mps2: float = 0.80
    max_angular_accel_rps2: float = 1.50
    waypoint_position_tolerance_m: float = 0.12
    waypoint_yaw_tolerance_rad: float = np.deg2rad(8.0)
    final_position_tolerance_m: float = 0.08
    final_yaw_tolerance_rad: float = np.deg2rad(5.0)
    final_alignment_distance_m: float = 0.20


class WaypointAdapter:
    """Stateful sparse-waypoint tracker with odometry anchoring and slew limits."""

    def __init__(self, config: WaypointAdapterConfig | None = None):
        self.config = config or WaypointAdapterConfig()
        self._world_waypoints = np.empty((0, 3), dtype=np.float64)
        self._target_index = 0
        self._last_command = np.zeros(3, dtype=np.float64)
        self._goal_reached = True

    @property
    def goal_reached(self) -> bool:
        return self._goal_reached

    @property
    def target_index(self) -> int:
        return self._target_index

    def clear(self) -> None:
        self._world_waypoints = np.empty((0, 3), dtype=np.float64)
        self._target_index = 0
        self._last_command.fill(0.0)
        self._goal_reached = True

    def set_waypoints(
        self,
        body_waypoints: np.ndarray,
        current_world_xyyaw: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        waypoints = np.asarray(body_waypoints, dtype=np.float64).reshape(-1, 3)
        current = np.asarray(current_world_xyyaw, dtype=np.float64).reshape(3)
        if valid_mask is not None:
            mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
            if len(mask) != len(waypoints):
                raise ValueError(f"valid_mask length {len(mask)} != waypoint count {len(waypoints)}")
            waypoints = waypoints[mask]
        if len(waypoints) == 0:
            self.clear()
            return

        cosine = np.cos(current[2])
        sine = np.sin(current[2])
        world_x = current[0] + cosine * waypoints[:, 0] - sine * waypoints[:, 1]
        world_y = current[1] + sine * waypoints[:, 0] + cosine * waypoints[:, 1]
        world_yaw = wrap_to_pi(current[2] + waypoints[:, 2])
        self._world_waypoints = np.stack((world_x, world_y, world_yaw), axis=-1)
        self._target_index = 0
        self._goal_reached = False

    def _target_error(self, current_world_xyyaw: np.ndarray) -> tuple[float, float, float]:
        current = np.asarray(current_world_xyyaw, dtype=np.float64).reshape(3)
        target = self._world_waypoints[self._target_index]
        world_delta = target[:2] - current[:2]
        cosine = np.cos(current[2])
        sine = np.sin(current[2])
        dx = cosine * world_delta[0] + sine * world_delta[1]
        dy = -sine * world_delta[0] + cosine * world_delta[1]
        dyaw = float(wrap_to_pi(target[2] - current[2]))
        return float(dx), float(dy), dyaw

    def _advance_reached_waypoints(self, current_world_xyyaw: np.ndarray) -> None:
        while self._target_index < len(self._world_waypoints):
            dx, dy, dyaw = self._target_error(current_world_xyyaw)
            distance = float(np.hypot(dx, dy))
            final = self._target_index == len(self._world_waypoints) - 1
            position_tolerance = (
                self.config.final_position_tolerance_m
                if final
                else self.config.waypoint_position_tolerance_m
            )
            yaw_tolerance = (
                self.config.final_yaw_tolerance_rad
                if final
                else self.config.waypoint_yaw_tolerance_rad
            )
            if distance > position_tolerance or abs(dyaw) > yaw_tolerance:
                break
            self._target_index += 1
        if self._target_index >= len(self._world_waypoints):
            self._goal_reached = True

    @staticmethod
    def _slew(target: np.ndarray, previous: np.ndarray, limits: np.ndarray) -> np.ndarray:
        return previous + np.clip(target - previous, -limits, limits)

    def compute_command(self, current_world_xyyaw: np.ndarray, dt_s: float) -> np.ndarray:
        if dt_s <= 0:
            raise ValueError(f"dt_s must be positive, got {dt_s}")
        if self._goal_reached or len(self._world_waypoints) == 0:
            self._last_command.fill(0.0)
            return self._last_command.copy()

        self._advance_reached_waypoints(current_world_xyyaw)
        if self._goal_reached:
            self._last_command.fill(0.0)
            return self._last_command.copy()

        dx, dy, dyaw = self._target_error(current_world_xyyaw)
        distance = float(np.hypot(dx, dy))
        path_heading = float(np.arctan2(dy, dx))
        alignment_weight = np.clip(
            1.0 - distance / max(self.config.final_alignment_distance_m, 1e-6), 0.0, 1.0
        )
        yaw_error = (1.0 - alignment_weight) * path_heading + alignment_weight * dyaw

        if self.config.holonomic:
            vx = self.config.linear_gain * dx
            vy = self.config.lateral_gain * dy
        else:
            # Reduce forward motion while facing away; lateral error is corrected through yaw.
            vx = self.config.linear_gain * distance * max(0.0, np.cos(path_heading))
            vy = 0.0
        wz = self.config.heading_gain * yaw_error + alignment_weight * self.config.final_yaw_gain * dyaw

        target = np.array(
            [
                np.clip(vx, -self.config.max_vx_mps, self.config.max_vx_mps),
                np.clip(vy, -self.config.max_vy_mps, self.config.max_vy_mps),
                np.clip(wz, -self.config.max_wz_rps, self.config.max_wz_rps),
            ],
            dtype=np.float64,
        )
        slew_limits = np.array(
            [
                self.config.max_linear_accel_mps2 * dt_s,
                self.config.max_linear_accel_mps2 * dt_s,
                self.config.max_angular_accel_rps2 * dt_s,
            ]
        )
        self._last_command = self._slew(target, self._last_command, slew_limits)
        return self._last_command.copy()
