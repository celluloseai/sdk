# Standard imports
import logging

# Third party imports
import numpy as np
from pydantic.dataclasses import dataclass

DEFAULT_ATOL = 1e-05
DEFAULT_RTOL = 1e-03

logger = logging.getLogger(__name__)


@dataclass
class Validation:
    atol: float = DEFAULT_ATOL
    rtol: float = DEFAULT_RTOL

    def compare_numpy(self, expected_outputs, actual_outputs):
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(
            expected_outputs, actual_outputs, rtol=self.rtol, atol=self.atol
        )

        logger.info(
            "Exported model has been tested with ONNXRuntime, and the result looks good!"  # noqa: E501
        )
