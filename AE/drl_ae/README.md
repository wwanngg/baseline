# DRL-AE（SAC）

## 运行

在项目根目录运行：

```powershell
python AE/drl_ae/run.py train --device cuda
python AE/drl_ae/run.py infer --device cuda --split all
python AE/drl_ae/run.py evaluate
```

数据读取 `AE/DataSet_AE`，唯一配置为 `configs/config.yaml`。训练从零开始，训练输出 `runs/`；推理输出 `results/`。
`infer` 默认读取本目录 `weights/policy_best.pt`。使用新训练模型时传入 `--checkpoint AE/drl_ae/runs/<运行目录>/policy_best.pt`。
默认每场景运行 5 个预声明回合，代表选图固定为首个 seed，不按指标挑选回合。

## 指标

Exposure error = `abs(log2(t/t_AE) + (gain_db−gain_AE_db)/(20×log10(2)))`，单位 EV，越低越接近左相机内置 AE。

终端输出 Exposure error 及其均值、样本标准差。已有实验以每场景 seed=0 的实际终态选图计分：

| 场景范围 | 有效场景 | Mean ± Std，EV ↓ |
|---|---:|---:|
| 共同4场景 | 4/4 | 0.805735 ± 0.392089 |
| 全部25场景 | 25/25 | 0.735639 ± 0.461193 |

共同4场景是当前 SAC 的独立测试集；全部25场景包含训练和验证场景。
