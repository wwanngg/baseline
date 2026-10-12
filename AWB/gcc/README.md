# GCC

## 运行

在项目根目录运行：

```powershell
python AWB/gcc/run.py infer --model both --variant both --device xpu --dtype float16
```

直接读取项目内的 `AWB/S24_val`，权重使用 `AWB/gcc/weights`。模型和数学函数从本目录 `third_party/` 加载。
原 RAW 与去噪 RAW 分开运行，默认不加 mask。结果写入本目录 `results/without_mask/`；`--limit 1` 可检查一张图像。
推理同时执行评价并在终端输出角误差。

## 指标

中性目标对照 `gt_illum`，使用 float64 recovery angular error，单位度，越低越好。两种 RAW 条件共享同一批205个场景。

| 权重 | 输入 | 有效/总数 | Mean ↓ | Median ↓ | Best25% ↓ | Worst25% ↓ | Worst5% ↓ | Tri-mean ↓ | Max ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nus8 | 原 RAW | 205/205 | 2.9593 | 2.1224 | 0.7645 | 6.5554 | 10.6950 | 2.3912 | 15.2924 |
| nus8 | 去噪 RAW | 205/205 | 2.9355 | 2.1370 | 0.7852 | 6.4190 | 10.4183 | 2.3535 | 13.2190 |
| gehler | 原 RAW | 205/205 | 3.0290 | 2.3860 | 0.9461 | 6.1917 | 9.4344 | 2.5435 | 18.0106 |
| gehler | 去噪 RAW | 205/205 | 3.0336 | 2.4177 | 0.9768 | 6.1783 | 9.2714 | 2.5360 | 16.8182 |

Best25%/Worst25%/Worst5% 为包含百分位边界并列值的尾部均值；Tri-mean=`(Q25+2×Median+Q75)/4`。
