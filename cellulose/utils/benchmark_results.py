# Standard imports
from dataclasses import dataclass
from datetime import datetime, timezone

# Cellulose imports
from cellulose.metrics.latency import LatencyMetrics
from cellulose.utils.devices import Device

# return {
#     # "precision": precision,
#     "io_binding": "",
#     # "model_name": model_name,
#     "inputs": 1,
#     # "threads": num_threads,
#     # "sequence_length": sequence_length,
#     # "custom_layer_num": config_modifier.get_layer_num(),
# }

USE_GPU = False


@dataclass
class BenchmarkResult:
    """
    This Enum specifies the supported engines (for benchmarking).
    """

    engine: str
    version: str
    latency_metrics: LatencyMetrics
    batch_size: int
    optimizer: str = "N/A"
    providers: str = "N/A"
    device: str = str(Device.CUDA.value) if USE_GPU else str(Device.CPU.value)
    timestamp: str = str(datetime.now(timezone.utc).isoformat())
