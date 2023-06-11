# Standard imports
import logging
from pathlib import Path

# Third party imports
import click
from pydantic.dataclasses import dataclass

# Cellulose imports
from cellulose.artifact.cellulose_artifact_manager import (
    CelluloseArtifactManager,
)
from cellulose.configs.loader import ConfigLoader
from cellulose.dashboard.api_resources.models.onnx import upload_onnx_model

logger = logging.getLogger(__name__)

DEFAULT_TARGET_DIRECTORY = Path.cwd()


@dataclass
class CelluloseContext:
    config_loader = ConfigLoader()
    artifact_manager = CelluloseArtifactManager()

    def __init__(self, api_key: str):
        logger.info("Initializing Cellulose context...")
        logger.info("Loading Cellulose configs...")
        self.load_config()
        logger.info("Initializing Cellulose artifact manager...")
        self.artifact_manager.init()
        self.api_key = api_key

    def load_config(self):
        """
        Internal!
        Loads the Cellulose configs
        """
        # Initialize and parse workspace level configs.
        self.config_loader.parse()

    def export(self, torch_model, input, **export_args):
        """
        This method exports the given PyTorch model and input. Please refer
        to our documentation for full set of options.

        Params
        ------
        torch_model (nn.Module): The PyTorch model.
        input: Input tensors to the PyTorch model.
        export_args: Other (both required and optional) arguments specific to
        the underlying export workflow. Please read our documentation for full
        details.
        """
        export_output = torch_model.cellulose.export_model(
            config_loader=self.config_loader,
            artifact_manager=self.artifact_manager,
            torch_model=torch_model,
            input=input,
            **export_args,
        )

        click.secho(
            "Uploading ONNX model to Cellulose dashboard...", fg="yellow"
        )
        response = upload_onnx_model(
            api_key=self.api_key, onnx_file=export_output.onnx.onnx_file
        )

        if response.status_code == 200 or response.status_code == 201:
            click.secho("Done!", fg="green")
        else:
            click.secho(
                "Failed to upload ONNX model to Cellulose dashboard",
                fg="red",
            )
            click.secho("Please check your API key and try again.", fg="red")
            click.secho(
                "If the problem persists, please contact us at support@cellulose.ai",
                fg="red",
            )

    def benchmark(self, torch_model, input, **benchmark_args):
        """
        This method benchmarks the given PyTorch model and input. Please refer
        to our documentation for full set of options.

        Params
        ------
        torch_model (nn.Module): The PyTorch model.
        input: Input tensors to the PyTorch model.
        benchmark_args: Other (both required and optional) arguments specific
        to the underlying benchmarking workflow. Please read our documentation
        for full details.
        """
        torch_model.cellulose.benchmark_model(
            config_loader=self.config_loader,
            artifact_manager=self.artifact_manager,
            torch_model=torch_model,
            input=input,
            **benchmark_args,
        )

    def flush(
        self, name: str, target_directory: str = str(DEFAULT_TARGET_DIRECTORY)
    ):
        """
        This method takes in a name for the Cellulose output artifact and
        target directory for where to place it.

        Params
        ------
        name: str - The name of the Cellulose artifact.
        For example, "my_benchmarks" to export a "my_benchmarks.cellulose.zip"
        target_directory: str - The target directory for where to place this
        generated ".cellulose" artifact. Current directory by default.
        """
        self.artifact_manager.export(
            name=name,
            target_directory=target_directory,
        )
