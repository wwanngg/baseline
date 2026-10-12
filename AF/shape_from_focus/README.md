# Shape from Focus

## 运行

在项目根目录运行：

```powershell
python AF/shape_from_focus/run.py run
python AF/shape_from_focus/run.py run --mode upstream
```

默认使用论文的 5×5 SML、阈值7、高斯插值；`upstream` 直接调用本目录第三方代码的离散 ML/SML。
数据放在 `AF/DataSet_AF`，CPU 运行，无需训练和权重。结果写入本目录 `results/<模式>/`。

## 指标

当前模式为 SML + Gaussian。数据没有深度、焦点索引真值或台面标定，MAE/RMSE 无法计算。

| 场景 | 尺寸 | 帧数 | MAE | RMSE | 单次纯计算时间，s |
|---|---|---:|---|---|---:|---:|
| xsim | 360×360 | 97 | N/A | N/A | 0.7564 |
| xsin | 300×300 | 60 | N/A | N/A | 0.3245 |

上述耗时为已有实验的单次记录，不是重复测量的平均值。终端输出有效覆盖率和纯计算耗时。
