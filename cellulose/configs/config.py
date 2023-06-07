# Third party imports
from pydantic.dataclasses import dataclass


@dataclass
class ConfigBase:
    """
    Base dataclass for all configurations within Cellulose.
    """

    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class OutputConfigInternal(ConfigBase):
    """
    This internal dataclass contains internal bookkeeping attributes
    """

    onnx_output_filename: str
    torchscript_output_filename: str


# -------------------------- PUBLIC DATACLASSES ---------------------------
# NOTE: These dataclasses should have 1:1 parity with all attributes specified
# in .cellulose_config.toml, with the exception of DecoratorConfig.
# Those are exclusive to each Cellulose decorator use.
@dataclass
class ExportConfig(ConfigBase):
    batch_size: int
    enable_export_validation: bool
    enable_onnx_checks: bool
    enable_onnx_export: bool
    enable_torchscript_export: bool


@dataclass
class BenchmarkConfig(ConfigBase):
    enable_onnxruntime: bool
    enable_pytorch: bool
    num_iterations: int


@dataclass
class OutputConfig(ConfigBase):
    enable_csv: bool
    enable_stdout: bool


@dataclass
class DecoratorConfig(ConfigBase):
    """
    This dataclass contains minimally required attributes for
    config arguments exclusive to each Cellulose() decorator use.
    """

    input_names: list[str]  # The list of input_names to the underlying model.
    output_names: list[
        str
    ]  # The list of output_names to the underlying model.


@dataclass
class AllConfig(ConfigBase):
    """
    Internal dataclass to store all forms of configs.
    """

    export_config: ExportConfig
    benchmark_config: BenchmarkConfig
    output_config: OutputConfig
    decorator_config: DecoratorConfig | None = None

    # for Cellulose use only.
    output_config_internal = OutputConfigInternal(
        onnx_output_filename="", torchscript_output_filename=""
    )
