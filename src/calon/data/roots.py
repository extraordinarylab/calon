"""Locations of data / runs and of model weights, set by the CALON_ROOT and CALON_MODELS environment variables."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("CALON_ROOT", "."))
MODELS = Path(os.environ.get("CALON_MODELS", "models"))
