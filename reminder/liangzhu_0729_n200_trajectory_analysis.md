# Go2 Navigation Trajectory Analysis

- Dataset: `/home/natural/pct_scene_outputs/liangzhu_0729_n200/lerobot_dataset`
- Episodes: `130`
- Frames: `30200` at `5.0` FPS
- Episode duration: 46.462 s +/- 2.908 s (min 40.800, median 46.200, max 53.400)

## Sparse Waypoint Parameters

- `rdp_epsilon_m`: `0.12`
- `yaw_metric_scale_m_per_rad`: `0.45`
- `max_translation_m`: `0.5`
- `max_yaw_rad`: `0.4363323129985824`

## Navigation Stages

### `nav_to_pick`

- Segments: `130`
- Duration: 7.925 s +/- 1.825 s (min 4.000, median 7.700, max 12.800)
- Trajectory length: 1.210 m +/- 0.336 m (min 0.519, median 1.176, max 2.212)
- Start x/y mean: `-0.577`, `5.089` m
- End x/y mean: `-0.555`, `6.051` m
- Sparse waypoints: 7.162 +/- 1.949 (min 3.000, median 7.000, max 13.000)
- Retained-frame ratio: 0.179 +/- 0.020 (min 0.120, median 0.182, max 0.216)

### `nav_to_place`

- Segments: `130`
- Duration: 19.448 s +/- 1.879 s (min 15.400, median 19.400, max 24.800)
- Trajectory length: 2.664 m +/- 0.393 m (min 1.725, median 2.670, max 3.500)
- Start x/y mean: `-0.552`, `6.053` m
- End x/y mean: `-0.578`, `4.042` m
- Sparse waypoints: 14.154 +/- 1.438 (min 11.000, median 14.000, max 18.000)
- Retained-frame ratio: 0.146 +/- 0.011 (min 0.117, median 0.144, max 0.167)
