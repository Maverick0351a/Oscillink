# Ensure repository root is on sys.path for direct test execution without editable install.
import pathlib
import sys
import warnings

root = pathlib.Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Silence upstream PendingDeprecationWarning in Starlette when importing
# the legacy 'multipart' name. We depend on python-multipart; the warning is
# informational and not actionable in our codepaths.
warnings.filterwarnings(
    "ignore",
    message=r"Please use `import python_multipart` instead\.",
    category=PendingDeprecationWarning,
    module=r"starlette\.formparsers",
)
