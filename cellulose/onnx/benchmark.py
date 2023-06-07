# Standard imports
import logging
import timeit
from pathlib import Path

# Third party imports
import onnxruntime

# Cellulose imports
from cellulose.base.benchmark import Benchmark
from cellulose.utils.benchmark_results import BenchmarkResult
from cellulose.utils.engines import Engine

logger = logging.getLogger(__name__)


class ONNXRuntimeBenchmark(Benchmark):
    batch_size: int
    engine: Engine = Engine.ONNXRUNTIME
    version: str = onnxruntime.__version__

    def benchmark(
        self, onnx_file: Path, input, num_iterations: int
    ) -> list[BenchmarkResult]:
        """
        This class method loads the given ONNX model and input, then
        runs a set of benchmarks and returns them.
        """
        results = []
        ort_session = onnxruntime.InferenceSession(str(onnx_file))

        ort_session.get_modelmeta()
        first_input_name = ort_session.get_inputs()[0].name
        first_output_name = ort_session.get_outputs()[0].name
        logger.info("first input name: {}".format(first_input_name))
        logger.info("first output name: {}".format(first_output_name))

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: input}
        # ort_outs = ort_session.run(None, ort_inputs)
        runtime_sec = timeit.repeat(
            lambda: ort_session.run(None, ort_inputs),
            repeat=num_iterations,
            number=1,
        )
        result = self.generate_result(
            runtime_sec=runtime_sec,
        )

        logger.info(result)
        results.append(result)

        return results
