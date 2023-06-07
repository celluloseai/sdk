# Standard imports
import csv
import os
import tempfile
from pathlib import Path
from typing import Any


def generate_output_csv_name(name: str) -> str:
    """
    This function takes in an arbitrary name (string) and returns it with a
    .csv extension.
    """
    return "{name}.csv".format(name=name)


def generate_output_csv_stream(
    output_file_name: str, input_list: list[dict[str, Any]]
) -> Path:
    """
    This function takes in an output filename (.csv) and a list of dictionaries
    then writes to it.
    """
    temp_output_file_path = Path(
        os.path.join(tempfile.gettempdir(), output_file_name)
    )
    if len(input_list) == 0:
        return temp_output_file_path

    with open(temp_output_file_path, "w", newline="") as csvfile:
        headers = input_list[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=headers)

        writer.writeheader()
        writer.writerows(input_list)
    return temp_output_file_path
