# Standard imports
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from cellulose.artifact.cellulose_artifact import CelluloseArtifact, Metadata

"""
Write unit tests for all classes in cellulose/artifact/cellulose_artifact.py
Test invalid cases too.
"""

def test_valid_cellulose_artifact():
    test = CelluloseArtifact(
        file_paths=[Path("test")],
        metadata=Metadata(
            title="test",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    )
    assert test.file_paths == [Path("test")]
    assert test.metadata.title == "test"
    assert isinstance(test.metadata.created_at, datetime)
    assert isinstance(test.metadata.updated_at, datetime)

def test_invalid_cellulose_artifact():
    test = CelluloseArtifact(
        file_paths=[Path("test")],
        metadata=Metadata(
            title="test",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    )
    assert test.file_paths != [Path("test2")]
    assert test.metadata.title != "test2"
    assert not isinstance(test.metadata.created_at, str)
    assert not isinstance(test.metadata.updated_at, str)
    
def test_metadata():
    test = Metadata(
        title="test",
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    assert test.title == "test"
    assert isinstance(test.created_at, datetime)
    assert isinstance(test.updated_at, datetime)