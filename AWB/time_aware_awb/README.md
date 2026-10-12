# Time-Aware AWB

## 运行

在项目根目录运行：

```powershell
python AWB/time_aware_awb/run.py infer --model awb-w-noise-snr --variant both --target both --device xpu
```

直接读取项目内的 `AWB/S24_val`，权重使用 `AWB/time_aware_awb/weights`。模型和数学函数从本目录 `third_party/` 加载。
原 RAW 与去噪 RAW 分开运行，默认不加 mask。结果写入本目录 `results/without_mask/`；`--limit 1` 可检查一张图像。
推理同时执行评价并在终端输出角误差。

## 指标

中性目标对照 `gt_illum`，使用 float64 recovery angular error，单位度，越低越好。两种 RAW 条件共享同一批205个场景。

| 权重 | 输入 | 有效/总数 | Mean ↓ | Median ↓ | Best25% ↓ | Worst25% ↓ | Worst5% ↓ | Tri-mean ↓ | Max ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| awb-w-noise-snr | 原 RAW | 205/205 | 1.7431 | 1.2985 | 0.4005 | 3.7806 | 6.7871 | 1.4047 | 18.6403 |
| awb-w-noise-snr | 去噪 RAW | 205/205 | 2.1363 | 1.3040 | 0.3431 | 5.3654 | 11.0986 | 1.4816 | 18.8463 |

Best25%/Worst25%/Worst5% 为包含百分位边界并列值的尾部均值；Tri-mean=`(Q25+2×Median+Q75)/4`。S24_val 是 validation，与原论文测试集不同。

`--target both` 还使用专门偏好权重评价 `pref_illum`；偏好权重的 Mean 为原 RAW 1.2457°、去噪 RAW 1.1812°。
