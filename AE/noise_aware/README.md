# Noise-Aware AE

## 运行

在项目根目录运行：

```powershell
python AE/noise_aware/run.py prepare
python AE/noise_aware/run.py infer --seed 0
python AE/noise_aware/run.py evaluate
```

数据直接读取 `AE/DataSet_AE`；CPU 运行，无需训练。`prepare` 为离线控制建立评分曲面，随后运行 `infer`。
终态图和搜索工作区保存到本目录 `results/`。本地 MATLAB 源码参数与已有 Python 数值实现共同用于搜索。

## 指标

Exposure error = `abs(log2(t/t_AE) + (gain_db−gain_AE_db)/(20×log10(2)))`，单位 EV，越低越接近左相机内置 AE。

终端只输出 Exposure error 及其均值、样本标准差。已有实验以每场景 seed=0 的实际终态选图计分：

| 场景范围 | 有效场景 | Mean ± Std，EV ↓ |
|---|---:|---:|
| 共同4场景 | 4/4 | 1.235921 ± 0.888725 |
| 全部25场景 | 25/25 | 1.109620 ± 0.835055 |
