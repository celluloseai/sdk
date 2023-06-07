# Third party imports
from onnx.checker import check_model

# Cellulose imports
from cellulose.base.checker import Checker


class ONNXChecker(Checker):
    def check(self, onnx_model):
        """
        This class method runs simple checks on the given ONNX model
        """
        check_model(onnx_model)
