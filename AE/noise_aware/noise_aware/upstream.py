from pathlib import Path
import re

UPSTREAM = Path(__file__).resolve().parents[1] / "third_party" / "Noise-AwareCameraExposureControl"

def parameters():
    text = (UPSTREAM / "Utils_Metric" / "Metric_Our.m").read_text(encoding="utf-8")
    result = {}
    for name in ("Alpha", "Beta", "Kg", "Ke", "Lambda", "Gamma", "p", "T_lower", "T_upper", "Resize_factor"):
        match = re.search(r"(?m)^\s*" + name + r"\s*=\s*([0-9.eE+\-]+)(?:\s*\^\s*([0-9]+))?\s*;", text)
        if match is None:
            raise ValueError(f"第三方 MATLAB 参数缺失：{name}")
        value = float(match.group(1))
        result[name] = value ** int(match.group(2)) if match.group(2) else value
    return result

PARAMETERS = parameters()
