# Standard imports
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Metadata:
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass
class CelluloseArtifact:
    """
    This class encapsulates all Cellulose zip artifact modules
    """

    file_paths: list[Path]
    metadata: Metadata
