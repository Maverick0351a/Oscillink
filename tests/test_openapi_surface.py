from fastapi.testclient import TestClient

from cloud.app.main import app


def test_openapi_contains_core_paths():
    client = TestClient(app)
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    data = resp.json()
    paths = data.get("paths", {})
    expected = ["/v1/settle", "/v1/receipt", "/v1/bundle", "/v1/chain/receipt"]
    for p in expected:
        assert p in paths, f"missing path {p} in OpenAPI spec"
