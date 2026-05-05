# Franka Three-Stage Lift Project

独立 Isaac Lab 项目，包含以下任务：

- Isaac-Lift-LongCuboid-Franka-ThreeStage-v0
- Isaac-Lift-LongCuboid-Franka-ThreeStage-Play-v0

该仓库已从主 IsaacLab 仓库迁移，方便你使用自己的 git 历史独立管理。

## 项目结构

- 任务注册入口：source/frank_with_handlight/frank_with_handlight/tasks/manager_based/frank_with_handlight/__init__.py
- 环境配置：source/frank_with_handlight/frank_with_handlight/tasks/manager_based/frank_with_handlight/joint_pos_env_cfg.py
- 三阶段环境骨架：source/frank_with_handlight/frank_with_handlight/tasks/manager_based/frank_with_handlight/franka_three_stage_env_cfg.py
- MDP 逻辑（观测/奖励/终止）：source/frank_with_handlight/frank_with_handlight/tasks/manager_based/frank_with_handlight/mdp/
- PPO 配置：source/frank_with_handlight/frank_with_handlight/tasks/manager_based/frank_with_handlight/agents/rsl_rl_ppo_cfg.py

## 前置条件

1. 已安装 Isaac Lab（推荐 conda 安装方式）。
2. 已有 conda 环境 isaaclab。
3. 在该项目根目录执行命令。

## 安装

```bash
source ~/.bashrc
conda activate isaaclab
cd /home/ubuntu/frank/frank_with_handlight
python -m pip install -e source/frank_with_handlight
```

## 验证任务注册

```bash
source ~/.bashrc
conda activate isaaclab
cd /home/ubuntu/frank/frank_with_handlight
python scripts/list_envs.py
```

预期会看到两个任务：

- Isaac-Lift-LongCuboid-Franka-ThreeStage-v0
- Isaac-Lift-LongCuboid-Franka-ThreeStage-Play-v0

## 训练（Headless）

```bash
source ~/.bashrc
conda activate isaaclab
cd /home/ubuntu/frank/frank_with_handlight
python scripts/rsl_rl/train.py \
    --task Isaac-Lift-LongCuboid-Franka-ThreeStage-v0 \
    --headless \
    --num_envs 1024 \
    --max_iterations 3000
```

快速冒烟测试：

```bash
python scripts/rsl_rl/train.py \
    --task Isaac-Lift-LongCuboid-Franka-ThreeStage-v0 \
    --headless \
    --num_envs 1024 \
    --max_iterations 3
```

## 可视化（Play）

```bash
source ~/.bashrc
conda activate isaaclab
cd /home/ubuntu/frank/frank_with_handlight
python scripts/rsl_rl/play.py \
    --task Isaac-Lift-LongCuboid-Franka-ThreeStage-Play-v0 \
    --num_envs 1 \
    --enable_cameras
```

## 常见说明

1. 如果看到 Overriding environment ... already in registry 警告，通常是因为主 IsaacLab 仓库和本项目同时注册了同名任务，这是预期现象。
2. 训练日志默认输出到 logs/rsl_rl/ 目录。
3. 调参优先改 PPO 配置和奖励权重：
     - source/frank_with_handlight/frank_with_handlight/tasks/manager_based/frank_with_handlight/agents/rsl_rl_ppo_cfg.py
     - source/frank_with_handlight/frank_with_handlight/tasks/manager_based/frank_with_handlight/franka_three_stage_env_cfg.py

## Git 管理建议

```bash
cd /home/ubuntu/frank/frank_with_handlight
git add -A
git commit -m "init standalone franka three-stage project"
```

建议后续按以下粒度提交：

- feat(task): 任务结构或接口修改
- feat(reward): 奖励函数与终止条件修改
- exp(train): 训练超参实验