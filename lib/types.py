import enum
from typing import Any, Literal

type KWArgs = dict[str, Any]
type JSONDict = dict[str, Any]  # Must be JSON-serializable.
type AMPDType = Literal['bfloat16', 'float16']

# ==================================================================================
# NOTE[tabular]
# ==================================================================================


type DataKey = str  # 'x_num', 'x_bin', 'x_cat', 'y', ...
type PartKey = str  # 'train', 'val', 'test', ...


class TaskType(enum.Enum):
    REGRESSION = 'regression'
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'


class PredictionType(enum.Enum):
    LABELS = 'labels'
    PROBS = 'probs'
    LOGITS = 'logits'
