# AE / AF / AWB 基线

## 运行

所有命令从项目根目录执行。根目录包含 `AE/`、`AF/`、`AWB/`；各基线从自己的 `third_party/` 加载源码。数据和权重均位于项目内，新推理输出位于各基线的 `results/`。
代码包含训练、推理、评价及必要依赖。指标直接在终端输出。

| 方向 | 基线与运行说明 |
|---|---|
| AE | [Noise-Aware AE](AE/noise_aware/README.md) |
| AE | [Global AE](AE/global_ae/README.md) |
| AE | [Entropy AE（Shannon）](AE/entropy_ae/README.md) |
| AE | [DRL-AE（SAC）](AE/drl_ae/README.md) |
| AE | [第三方 DQN-grid 对照](AE/dqn_grid/README.md) |
| AF | [Shape from Focus](AF/shape_from_focus/README.md) |
| AWB | [Time-Aware AWB](AWB/time_aware_awb/README.md) |
| AWB | [RL-AWB](AWB/rl_awb/README.md) |
| AWB | [GCC](AWB/gcc/README.md) |

每个基线有独立 `third_party/` 和 `weights/`（需要权重的方法）。第三方可复用的网络和数学函数直接从该目录引用；必要的数据适配与算法改造保留在基线代码中。
数据路径：`AE/DataSet_AE`、`AF/DataSet_AF`、`AWB/S24_val`。


权重位置：`AE/drl_ae/weights/policy_best.pt`、`AE/dqn_grid/weights/policy_best.pt`、`AWB/time_aware_awb/weights/`、`AWB/rl_awb/weights/`、`AWB/gcc/weights/`。
各基线独立包含评价模块，不引用其他基线的代码。

`AE/drl_ae/weights/policy_best.pt` 和 `AE/dqn_grid/weights/policy_best.pt` 这两个权重是自己跑的，不是官方权重。

以下为整理前已完成实验的结果。三项任务的数据和指标不同，分别评价。

## AE 主表：Exposure error

Exposure error = `abs(log2(t/t_AE) + (gain_db−gain_AE_db)/(20×log10(2)))`，单位 EV，越低越接近左相机内置 AE。

共同4场景是当前 SAC 的独立测试集，每场景使用预先声明的 seed=0 终态选图。全部25场景包含 SAC 的训练、验证场景，仅作补充。

| 方法 | 4场景 Mean ± Std，EV ↓ | 全25场景 Mean ± Std，EV ↓ |
|---|---:|---:|
| Noise-Aware AE | 1.235921 ± 0.888725 | 1.109620 ± 0.835055 |
| Global AE | 1.789359 ± 1.316132 | 0.976201 ± 0.884335 |
| Entropy AE（Shannon） | 0.541259 ± 0.339195 | 0.707334 ± 0.593816 |
| DRL-AE（SAC） | 0.805735 ± 0.392089 | 0.735639 ± 0.461193 |

第三方 DQN-grid 的4场景 Mean=0.991931 EV；全25场景 Mean=0.850370 EV。

## AF 主表：Shape from Focus

数据为第三方自带图像栈，采用 SML + Gaussian。缺少真值，精度列填写 N/A。

| 方法 | 场景 | 尺寸 | 帧数 | MAE | RMSE | 单次纯计算时间，s |
|---|---|---|---:|---|---|---:|---:|
| SFF | xsim | 360×360 | 97 | N/A | N/A | 0.7564 |
| SFF | xsin | 300×300 | 60 | N/A | N/A | 0.3245 |

耗时是已有实验的单次记录，没有重复测量的均值和标准差。

## AWB 主表：中性角误差

数据为 S24_val 的205张配对图像，mask=without，对照 `gt_illum`。采用 recovery angular error，单位度，全部指标越低越好。原 RAW 与去噪 RAW 是同一批场景的两种条件。

| 方法 / 权重 | 输入 | 有效/总数 | Mean ↓ | Median ↓ | Best25% ↓ | Worst25% ↓ | Worst5% ↓ | Tri-mean ↓ | Max ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Time-Aware AWB / awb-w-noise-snr | 原 RAW | 205/205 | 1.7431 | 1.2985 | 0.4005 | 3.7806 | 6.7871 | 1.4047 | 18.6403 |
| RL-AWB / NCC | 原 RAW | 205/205 | 3.8930 | 2.6410 | 0.5771 | 9.4096 | 17.2458 | 2.8950 | 31.1960 |
| RL-AWB / LEVI | 原 RAW | 205/205 | 3.8019 | 2.5723 | 0.5580 | 9.3237 | 16.4895 | 2.8171 | 30.1189 |
| RL-AWB / Gehler | 原 RAW | 205/205 | 3.5876 | 2.3560 | 0.5428 | 8.7522 | 15.4328 | 2.6181 | 30.6787 |
| GCC / nus8 | 原 RAW | 205/205 | 2.9593 | 2.1224 | 0.7645 | 6.5554 | 10.6950 | 2.3912 | 15.2924 |
| GCC / gehler | 原 RAW | 205/205 | 3.0290 | 2.3860 | 0.9461 | 6.1917 | 9.4344 | 2.5435 | 18.0106 |
| Time-Aware AWB / awb-w-noise-snr | 去噪 RAW | 205/205 | 2.1363 | 1.3040 | 0.3431 | 5.3654 | 11.0986 | 1.4816 | 18.8463 |
| RL-AWB / NCC | 去噪 RAW | 205/205 | 3.8496 | 2.7227 | 0.4727 | 9.2654 | 15.6126 | 2.9522 | 27.1188 |
| RL-AWB / LEVI | 去噪 RAW | 205/205 | 3.6913 | 2.5488 | 0.4710 | 8.7673 | 13.7030 | 2.8601 | 19.3599 |
| RL-AWB / Gehler | 去噪 RAW | 205/205 | 3.7148 | 2.6279 | 0.4906 | 8.8787 | 14.3578 | 2.8293 | 22.1371 |
| GCC / nus8 | 去噪 RAW | 205/205 | 2.9355 | 2.1370 | 0.7852 | 6.4190 | 10.4183 | 2.3535 | 13.2190 |
| GCC / gehler | 去噪 RAW | 205/205 | 3.0336 | 2.4177 | 0.9768 | 6.1783 | 9.2714 | 2.5360 | 16.8182 |

角误差=`acos(dot(pred,gt)/(norm(pred)×norm(gt)))×180/pi`，float64，余弦裁剪至[-1,1]。
Best25%/Worst25%/Worst5% 分别为第25/75/95百分位边界内的尾部均值，保留边界并列样本；Tri-mean=`(Q25+2×Median+Q75)/4`。
不同权重来源分别列行。
