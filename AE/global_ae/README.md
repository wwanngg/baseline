# Global AE

## 运行

在项目根目录运行：

```powershell
python AE/global_ae/run.py infer
python AE/global_ae/run.py evaluate
```

数据直接读取 `AE/DataSet_AE`，CPU 运行，无需训练。默认固定增益 0 dB、采样步长 8；终态图输出到本目录 `results/`。
运行参数为线性化 JPEG 亮度代理、目标 0.13、饱和阈值 0.9。

## 指标

Exposure error = `abs(log2(t/t_AE) + (gain_db−gain_AE_db)/(20×log10(2)))`，单位 EV，越低越接近左相机内置 AE。

终端只输出 Exposure error 及其均值、样本标准差。已有实验以每场景 seed=0 的实际终态选图计分：

| 场景范围 | 有效场景 | Mean ± Std，EV ↓ |
|---|---:|---:|
| 共同4场景 | 4/4 | 1.789359 ± 1.316132 |
| 全部25场景 | 25/25 | 0.976201 ± 0.884335 |
