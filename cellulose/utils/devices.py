# Standard imports
from enum import Enum


class Device(Enum):
    """
    This Enum specifies the supported devices (for benchmarking).
    """

    CPU = "cpu"
    CUDA = "cuda"
