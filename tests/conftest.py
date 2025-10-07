# Ensure repository root is on sys.path for direct test execution without editable install.
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
