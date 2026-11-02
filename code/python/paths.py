"""
paths.py: project root shared by the analysis scripts.

ROOT is the folder that holds data/, output/ and report/. Set the environment
variable ANOREXIA_ROOT to point to it; otherwise the folder that contains this
repository is used.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("ANOREXIA_ROOT", Path(__file__).resolve().parents[3]))
