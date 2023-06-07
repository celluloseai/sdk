# Standard imports
import logging

# Third party imports
import numpy as np
from pydantic.dataclasses import dataclass

# Cellulose imports
from cellulose.metrics.latency import LatencyMetrics
from cellulose.utils.benchmark_results import BenchmarkResult
from cellulose.utils.engines import Engine

logger = logging.getLogger(__name__)


@dataclass
class Benchmark:
    batch_size: int
    engine = Engine | None
    version = str | None

    def calculate_latency_metrics(
        self, latency_list_sec: int
    ) -> LatencyMetrics:
        latency_ms = (
            sum(latency_list_sec) / float(len(latency_list_sec)) * 1000.0
        )
        variance = np.var(latency_list_sec, dtype=np.float64)
        throughput = self.batch_size * (1000.0 / latency_ms)

        return LatencyMetrics(
            num_samples=len(latency_list_sec),
            variance=variance,
            mean_ms=latency_ms,
            p50_ms=np.percentile(latency_list_sec, 50) * 1000.0,
            p90_ms=np.percentile(latency_list_sec, 90) * 1000.0,
            p95_ms=np.percentile(latency_list_sec, 95) * 1000.0,
            p99_ms=np.percentile(latency_list_sec, 99) * 1000.0,
            throughput_per_sec=throughput,
        )

    def generate_result(self, runtime_sec) -> BenchmarkResult:
        """
        This method generates the results.
        """
        if self.engine is None:
            error_msg = (
                "Expected engine to be an Engine type, but got None instead"
            )
            logger.error(error_msg)
            raise Exception(error_msg)

        if self.version is None or self.version == "":
            logger.error(
                "Expected version to be a str type, but got {} instead".format(
                    type(self.version)
                )
            )
            raise
        return BenchmarkResult(
            engine=self.engine.value,
            version=self.version,
            batch_size=self.batch_size,
            latency_metrics=self.calculate_latency_metrics(
                latency_list_sec=runtime_sec
            ),
        )
