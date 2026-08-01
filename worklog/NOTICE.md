每次工作前将工作记录写入日志，工作完成后将计划转为工作记录并压缩，保持日志文件精炼且准确。

本机训练代码仓库目录为 /home/natural/Desktop/mtr/starVLA_sc
github 仓库为 github.com/Natural-Horse/starVLA_sc.git
目前工作分支为 robodog
远程服务器见 /home/natural/.ssh/config zju-server
服务器训练代码仓库目录为 /hdd4/MaTianran/pct_workspace/starVLA_sc
服务器工作分支为 robodog

另本地数据采集工作目录为 /home/natural/Desktop/mtr/pct_scene
数采仓库分支为 mtr_dev
数采仓库 origin 为 git@github.com:Natural-Horse/arm-vla-grasp-sim.git
数采仓库 upstream 为 https://github.com/yagami-light7/arm-vla-grasp-sim.git
已采集数据目录为 ~/pct_scene_outputs

仿真闭环测评中，starVLA_sc 负责模型加载、版本化远程协议和 route/subtask/机体系稀疏 waypoint 输出；pct_scene 负责观测编码、世界系变换、DWA/RL 速度适配、Isaac 状态机及 cuRobo 抓放。远端服务只绑定 127.0.0.1，由评测机 SSH 本地转发访问，不直接暴露公网。

工作时，在本地修改代码，验证提交后，再在远程服务器目录下拉取新版本代码并运行。
