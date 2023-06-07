# Third party imports
from onnx import load

# Cellulose imports
from cellulose.base.loader import Loader


class ONNXLoader(Loader):
    def load(self, file):
        """
        This class method loads the ONNX model as specified by the file.
        """
        with open(file, "rb") as f:
            onnx_model = load(f)
        return onnx_model


if __name__ == "__main__":
    loader = ONNXLoader()

    onnx_model = loader.load("linear_regression.onnx")
    # display
    print(onnx_model)
