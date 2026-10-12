# Entropy AE（Shannon）

## 运行

在项目根目录运行：

```powershell
python AE/entropy_ae/run.py infer
python AE/entropy_ae/run.py evaluate
```

数据直接读取 `AE/DataSet_AE`，CPU 运行，无需训练。默认固定增益 0 dB、采样步长 8；终态图输出到本目录 `results/`。
采用 Shannon 最大熵选图；不执行联合增益搜索。

## 指标

Exposure error = `abs(log2(t/t_AE) + (gain_db−gain_AE_db)/(20×log10(2)))`，单位 EV，越低越接近左相机内置 AE。

终端只输出 Exposure error 及其均值、样本标准差。已有实验以每场景 seed=0 的实际终态选图计分：

| 场景范围 | 有效场景 | Mean ± Std，EV ↓ |
|---|---:|---:|
| 共同4场景 | 4/4 | 0.541259 ± 0.339195 |
| 全部25场景 | 25/25 | 0.707334 ± 0.593816 |
