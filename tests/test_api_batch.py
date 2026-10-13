"""Tests for the ``/predict-batch`` endpoint.

The real pipeline is stubbed so these tests exercise request handling, per-item
isolation, and temp-file cleanup without loading Mistral-7B. Importing the API
pulls in torch/transformers, so the module is skipped when those are missing.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("torch", reason="med-VQA API requires torch")
pytest.importorskip("transformers", reason="med-VQA API requires transformers")

from fastapi.testclient import TestClient  # noqa: E402

from api import main as api_main  # noqa: E402


class _FakePipeline:
    """Minimal stand-in for MedVQAPipeline that records the questions it saw."""

    def __init__(self, fail_on_bytes: bytes | None = None) -> None:
        self._is_api_mode = False
        self.api_llm_client = None
        self.seen_questions: list[str] = []
        self._fail_on_bytes = fail_on_bytes

    def predict(self, image_path, question, generate_heatmap=False, use_mc_dropout=False):
        self.seen_questions.append(question)
        # Uploads are persisted under a generated temp name, so faults are keyed
        # off file content rather than the original client-side filename.
        if self._fail_on_bytes is not None:
            if self._fail_on_bytes in Path(image_path).read_bytes():
                raise RuntimeError("simulated inference failure")
        return SimpleNamespace(
            answer="No acute abnormality is visible.",
            confidence=0.91,
            uncertainty_flag=False,
            heatmap_path=None,
            latency_ms=12.5,
            predictive_entropy=0.08,
        )


@pytest.fixture
def client_and_pipeline(monkeypatch):
    pipeline = _FakePipeline()
    monkeypatch.setattr(api_main, "get_pipeline", lambda: pipeline)
    return TestClient(api_main.app), pipeline


def _png(name: str = "scan.png") -> tuple[str, bytes, str]:
    return (name, b"\x89PNG\r\n\x1a\nfake-bytes", "image/png")


def test_upload_temp_names_are_unique_per_request():
    """A timestamp-only temp name collides for concurrent same-second uploads."""

    source = Path(api_main.__file__).read_text(encoding="utf-8")
    assert "upload_{uuid.uuid4().hex}" in source


def test_batch_answers_every_image_in_order(client_and_pipeline):
    client, pipeline = client_and_pipeline
    response = client.post(
        "/predict-batch",
        data={"question": "Is there a pleural effusion?"},
        files=[("files", _png("a.png")), ("files", _png("b.png"))],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["succeeded"] == 2
    assert body["failed"] == 0
    assert [item["filename"] for item in body["results"]] == ["a.png", "b.png"]
    assert all(item["status"] == "ok" for item in body["results"])
    assert all(item["result"]["confidence"] == pytest.approx(0.91) for item in body["results"])
    assert len(pipeline.seen_questions) == 2


def test_batch_per_item_questions_override_shared_question(client_and_pipeline):
    client, pipeline = client_and_pipeline
    response = client.post(
        "/predict-batch",
        data={
            "question": "Default question",
            "questions": '["Is the heart enlarged?", ""]',
        },
        files=[("files", _png("first.png")), ("files", _png("second.png"))],
    )

    assert response.status_code == 200
    assert pipeline.seen_questions[0].endswith("Is the heart enlarged?")
    assert pipeline.seen_questions[1].endswith("Default question")


def test_one_bad_item_does_not_discard_the_rest(client_and_pipeline):
    client, _ = client_and_pipeline
    response = client.post(
        "/predict-batch",
        data={"question": "Any fracture?"},
        files=[
            ("files", _png("good.png")),
            ("files", ("notes.txt", b"not an image", "text/plain")),
            ("files", _png("also_good.png")),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert body["succeeded"] == 2
    assert body["failed"] == 1
    assert [item["status"] for item in body["results"]] == ["ok", "error", "ok"]
    assert body["results"][1]["detail"].startswith("Unsupported file type: .txt")
    assert body["results"][1]["result"] is None
    assert body["results"][1]["filename"] == "notes.txt"


def test_pipeline_exception_is_reported_per_item(monkeypatch):
    pipeline = _FakePipeline(fail_on_bytes=b"boom-bytes")
    monkeypatch.setattr(api_main, "get_pipeline", lambda: pipeline)
    client = TestClient(api_main.app)

    response = client.post(
        "/predict-batch",
        data={"question": "Any nodule?"},
        files=[
            ("files", ("boom.png", b"loom-bytes boom-bytes", "image/png")),
            ("files", _png("fine.png")),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["succeeded"] == 1
    assert body["failed"] == 1
    assert "simulated inference failure" in body["results"][0]["detail"]


def test_batch_rejects_oversized_batch(client_and_pipeline):
    client, _ = client_and_pipeline
    files = [("files", _png(f"scan_{i}.png")) for i in range(api_main.MAX_BATCH_SIZE + 1)]
    response = client.post("/predict-batch", data={"question": "Any nodule?"}, files=files)

    assert response.status_code == 400
    assert str(api_main.MAX_BATCH_SIZE) in response.json()["detail"]


def test_batch_requires_a_question(client_and_pipeline):
    client, _ = client_and_pipeline
    response = client.post("/predict-batch", files=[("files", _png())])

    assert response.status_code == 400
    assert response.json()["detail"] == "Question is required"


def test_batch_rejects_malformed_questions_json(client_and_pipeline):
    client, _ = client_and_pipeline
    response = client.post(
        "/predict-batch",
        data={"question": "Any nodule?", "questions": '{"not": "an array"}'},
        files=[("files", _png())],
    )

    assert response.status_code == 400
    assert "JSON array" in response.json()["detail"]


def test_batch_leaves_no_temp_uploads_behind(client_and_pipeline):
    client, _ = client_and_pipeline
    temp_dir = Path("data/temp")
    before = set(temp_dir.iterdir()) if temp_dir.exists() else set()

    response = client.post(
        "/predict-batch",
        data={"question": "Any nodule?"},
        files=[("files", _png("one.png")), ("files", _png("two.png"))],
    )

    assert response.status_code == 200
    after = set(temp_dir.iterdir()) if temp_dir.exists() else set()
    assert after - before == set()
