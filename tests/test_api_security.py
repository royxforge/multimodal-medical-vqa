"""Phase 1 API hardening tests: explicit CORS, jailed path reads, upload size cap.

Importing the API pulls in torch/transformers, so the whole module is skipped
when the ML dependencies are unavailable.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("torch", reason="med-VQA API requires torch")
pytest.importorskip("transformers", reason="med-VQA API requires transformers")

from fastapi.testclient import TestClient  # noqa: E402

from api import main as api_main  # noqa: E402


def test_cors_origins_are_explicit_and_never_wildcard():
    assert api_main._ALLOWED_ORIGINS
    assert "*" not in api_main._ALLOWED_ORIGINS


def test_upload_size_cap_is_10mb():
    assert api_main.MAX_UPLOAD_BYTES == 10 * 1024 * 1024


def test_predict_get_rejects_path_outside_data_root(tmp_path):
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"not-an-image")
    client = TestClient(api_main.app)
    response = client.get(
        "/predict/", params={"image_path": str(outside), "question": "Is this normal?"}
    )
    assert response.status_code == 400
    assert "data/" in response.json()["detail"]


def test_predict_get_rejects_traversal_path():
    client = TestClient(api_main.app)
    response = client.get(
        "/predict/",
        params={"image_path": "../../etc/passwd", "question": "Is this normal?"},
    )
    assert response.status_code == 400


def test_predict_get_requires_question():
    client = TestClient(api_main.app)
    response = client.get("/predict/", params={"image_path": "data/raw/x.png"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Question is required"