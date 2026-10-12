# 第三方 DQN-grid

## 运行

在项目根目录运行：

```powershell
python AE/dqn_grid/run.py train
python AE/dqn_grid/run.py infer --device xpu
python AE/dqn_grid/run.py evaluate
```

数据读取 `AE/DataSet_AE`，直接调用本目录 `third_party/camera-exposure-rl` 的 DQN 训练代码。
默认训练配置 `configs/original_xpu.yaml`。cuda 上使用 `train --config AE/dqn_grid/configs/original_cuda.yaml`；CPU 使用 `original_cpu.yaml`。
训练输出 `runs/`，推理输出 `results/`。推理默认读取本目录 `weights/policy_best.pt`；更换模型用 `--checkpoint <路径>`。

## 指标

Exposure error = `abs(log2(t/t_AE) + (gain_db−gain_AE_db)/(20×log10(2)))`，单位 EV，越低越接近左相机内置 AE。

已有实验使用原训练累计奖励选出的最佳检查点，代表选图为 seed=0：

| 场景范围 | 场景数 | Mean ± Std，EV ↓ |
|---|---:|---:|
| 选模4场景 | 4 | 0.991931 ± 0.653769 |
| 全部25场景 | 25 | 0.850370 ± 0.644487 |

四个 eval 场景参加过选模，其余21场景参加过训练。
