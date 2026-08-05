# StarVLA 远程推理服务启动指南

本文描述如何在一台有 GPU 的服务器（例如 `zju-server`）上启动 Go2 StarVLA
远程推理服务。服务只监听回环地址 `127.0.0.1`，评测机/机器狗通过 SSH 隧道访问，
不直接暴露公网。

推理服务跑在 GPU 服务器上，仓库路径按服务器为准（例如
`/hdd4/MaTianran/pct_workspace/starVLA_sc`）。启动前先登录服务器，
并确认代码已同步（`git pull` 或 rsync）。

## 1. 配置（YAML）

所有启动参数集中在 `configs/vla_eval/server.yaml`：

```yaml
server:
  host: 127.0.0.1
  port: 10093
  checkpoint: /path/to/final_model/pytorch_model.pt
  device: cuda
  use_bf16: true
  gpu: 5
  tmux_session: go2_vla_eval_server
  log_dir: results/evaluation_logs
  python_bin: /hdd4/MaTianran/rtc_starvla/envs/mtr_star/bin/python
```

常用可调项：

- `server.checkpoint`：完整 QwenPI 权重，必须是 10 维 v2 协议训练产物；
- `server.gpu`：`CUDA_VISIBLE_DEVICES` 使用的物理 GPU；启动前脚本会拒绝已占用卡；
- `server.port`：回环监听端口，默认 `10093`；
- `server.tmux_session`：tmux 会话名，避免与他人冲突；
- `server.use_bf16`：默认 `true`，显存不足时可关掉（会慢）；
- `server.python_bin`：含 torch/starVLA 的 Python 解释器。

## 1.1 模型/归一化/夹爪/RTC（当前单 head 管线）

- 训练产物使用 **按 step 索引归一化**：`dataset_statistics.json` 的 `q01/q99`
  为 `[action_horizon, 10]`（每个 horizon step 独立统计，避免 waypoint 首步被
  跨 step 合并统计压坏）；server 端反归一化自动按步查表（兼容旧 pooled 10 维）。
- 夹爪维语义：训练目标夹爪维使用 **control.action 的夹爪指令**（0=闭合,1=张开），
  不是实测指位；模型输出即夹爪目标指令，`0..1` 映射到 X5 `0..0.088m`。
- RTC 开关：`framework.rtc.enable` 为总开关（训练/推理一致生效）；当前 go2 动作
  训练用最简单方式（单 head + 按步归一化，无 RTC），server 加载对应 checkpoint
  后自动跳过 prev-chunk 条件化。

## 2. 启动

```bash
ssh zju-server
cd /hdd4/MaTianran/pct_workspace/starVLA_sc   # 服务器上的仓库
bash scripts/evaluation/start_vla_inference.sh --config configs/vla_eval/server.yaml
```

先看将执行的参数（不真正启动）：

```bash
bash scripts/evaluation/start_vla_inference.sh --config configs/vla_eval/server.yaml --dry-run
```

启动后检查：

```bash
tmux ls | grep go2_vla_eval_server
tail -f results/evaluation_logs/go2_vla_eval_server.log
curl http://127.0.0.1:10093/healthz   # 期望 ok
```

协议 smoke test（不加载模型）：

```bash
CUDA_VISIBLE_DEVICES=5 python -B -m deployment.go2_remote.server --mock-route nav --port 10093
```

## 3. 停止与日志

```bash
tmux kill-session -t go2_vla_eval_server
```

日志在 `results/evaluation_logs/go2_vla_eval_server.log`，每请求记录
`inference_ms` / `server_total_ms` 与协议版本。

## 4. 与评测机/真机联通

- 评测机（pct_scene 仿真）：工作站 `ssh -L 127.0.0.1:10093:127.0.0.1:10093 zju-server`
- 真机（gx-real）：工作站双跳隧道 `scripts/evaluation/manage_remote_vla_tunnel.sh start zju-server robodog`，
  机器狗客户端连接本机 `ws://127.0.0.1:10093`。

注意：`robodog` 没有默认路由，隧道由工作站主动发起，机器狗无需出网。
三台机器（服务器/工作站/robodog）各自登录、各自启动对应进程。
