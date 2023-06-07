#!/bin/bash

# This bash script is used by our CI system to lint Python source files.

set -euo pipefail

echo "Running black..."
python3 -m black --check .

echo "Running flake8..."
python3 -m flake8 ./**/*.py
python3 -m flake8 ./**/**/*.py

echo "Running isort..."
python3 -m isort --check-only .

echo "Done!"
