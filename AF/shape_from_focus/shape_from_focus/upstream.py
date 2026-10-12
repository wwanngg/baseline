from pathlib import Path
import numpy as np
from .source_loader import definitions

UPSTREAM = Path(__file__).resolve().parents[1] / "third_party" / "Shape_from_Focus"
_namespace = definitions(UPSTREAM / "tr_SFF.py", ("ML", "SML"), {"np": np})
ML = _namespace["ML"]
SML = _namespace["SML"]
