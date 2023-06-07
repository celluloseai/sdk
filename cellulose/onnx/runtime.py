# Standard imports
import logging
from pathlib import Path

# Third party imports
import onnxruntime
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ONNXRuntime:
    def run(self, onnx_file: Path, input):
        """
        This class method loads the ONNX model as specified by the file.
        """
        ort_session = onnxruntime.InferenceSession(str(onnx_file))

        ort_session.get_modelmeta()
        first_input_name = ort_session.get_inputs()[0].name
        first_output_name = ort_session.get_outputs()[0].name
        logger.info("first input name: {}".format(first_input_name))
        logger.info("first output name: {}".format(first_output_name))

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: input}
        ort_outs = ort_session.run(None, ort_inputs)

        logger.info("Output: {}".format(ort_outs))
        return ort_outs
