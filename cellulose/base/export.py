# Third party imports
from pydantic.dataclasses import dataclass

# Cellulose imports
from cellulose.artifact.cellulose_artifact_manager import (
    CelluloseArtifactManager,
)
from cellulose.configs.config import AllConfig


@dataclass
class Export:
    model_config: AllConfig
    artifact_manager: CelluloseArtifactManager
