import json
import subprocess
import sys
from pathlib import Path


def test_export_openapi_script(tmp_path: Path):
    out_path = tmp_path / "schema.json"
    # Run module invocation equivalent to CLI
    cmd = [sys.executable, "-m", "scripts.export_openapi", "--out", str(out_path)]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert out_path.exists(), "OpenAPI export did not produce file"
    data = json.loads(out_path.read_text("utf-8"))
    for p in ["/v1/settle", "/v1/receipt"]:
        assert p in data.get("paths", {}), f"missing {p} in exported schema"
    # Basic sanity: title and version fields
    assert "info" in data and "title" in data["info"]
