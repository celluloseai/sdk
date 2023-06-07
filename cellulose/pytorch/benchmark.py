# Standard imports
import logging
import timeit

# Third party imports
import torch
from pydantic.dataclasses import dataclass

# Cellulose imports
from cellulose.base.benchmark import Benchmark
from cellulose.utils.benchmark_results import BenchmarkResult
from cellulose.utils.engines import Engine

logger = logging.getLogger(__name__)


@dataclass
class PyTorchBenchmark(Benchmark):
    batch_size: int
    engine: Engine = Engine.TORCH
    version: str = torch.__version__

    def benchmark(
        self, torch_model, input, num_iterations: int
    ) -> list[BenchmarkResult]:
        results = []
        """
        This class method loads the given torch_model and input, then
        runs a set of benchmarks and returns them.
        """
        runtime_sec = timeit.repeat(
            lambda: torch_model(input), repeat=num_iterations, number=1
        )
        result = self.generate_result(runtime_sec=runtime_sec)
        logger.info(result)
        results.append(result)

        return results
