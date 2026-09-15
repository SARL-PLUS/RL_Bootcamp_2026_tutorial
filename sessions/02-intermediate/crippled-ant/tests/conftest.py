import sys
from pathlib import Path

# make `envs` and `utils` importable when running pytest from this directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
