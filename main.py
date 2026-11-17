"""Prepara datos. El EDA completo se ejecuta en notebooks/01_EDA.ipynb."""
from pathlib import Path
import os
from src.pipeline import run
if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    run()
