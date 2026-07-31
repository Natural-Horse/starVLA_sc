# Go2 Sparse Waypoint Pipeline

- Added n200 trajectory analysis and shared sparse SE(2) extraction; 130 episodes/30200 frames yield 7.16 pick and 14.15 place waypoints on average.
- Added pct_scene dual-camera loader, five-route Qwen3-VL router, NAV-only masked Flow Matching head, explicit bbox isolation/API, and nonholonomic waypoint-to-velocity adapter.
- Local validation: 10 tests pass, full analysis pass, 2-worker video loading pass, and 117/13 episode split pass. Formal model training was not started locally.
- Synced commits `b2cbf35` and `048a8de` to GitHub and `zju-server`. Remote validation passed: 10 tests, full 130-episode analysis, shell syntax, trainer import/help, dataset/model path checks. Training was not started.
