# One-command dev bootstrap for Windows
# Usage:  scripts\dev.ps1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
pytest -q
