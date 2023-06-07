# Standard imports
from enum import Enum


class Engine(Enum):
    """
    This Enum specifies the supported engines (for benchmarking).
    """

    ONNXRUNTIME = "onnxruntime"
    TENSORFLOW = "tensorflow"
    TORCH = "torch"
    TORCHSCRIPT = "torchscript"
