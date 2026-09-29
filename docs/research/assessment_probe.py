"""Replay the original assessment against the frozen, pre-upgrade implementation."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory() as directory:
    with ZipFile(ROOT / "evaluations/baselines/original_runtime.zip") as archive:
        archive.extractall(directory)
        source = archive.read("assessment_probe.py").decode()
    source = source.replace("ROOT=Path(__file__).resolve().parents[2]", f"ROOT=Path({str(ROOT)!r})")
    subprocess.run([sys.executable, "-c", source], check=True,
                   env=dict(os.environ, PYTHONPATH=directory))
