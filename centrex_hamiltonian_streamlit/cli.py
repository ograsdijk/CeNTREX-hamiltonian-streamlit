import subprocess
from pathlib import Path

filepath = Path(__file__)


def run():
    subprocess.run(f"streamlit run {filepath.parent / 'main.py'}")
