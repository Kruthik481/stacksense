import embeddings
import numpy as np
import pytest
from config import settings


class FakeTextEmbedding:
    instances = 0

    def __init__(self, model_name: str, cache_dir: str | None = None, **kwargs):
        FakeTextEmbedding.instances += 1
        self.model_name = model_name
        self.cache_dir = cache_dir

    def embed(self, texts):
        for _ in texts:
            yield np.ones(384, dtype=np.float64)


@pytest.fixture
def fake_model(monkeypatch):
    FakeTextEmbedding.instances = 0
    monkeypatch.setattr(embeddings, "TextEmbedding", FakeTextEmbedding)
    embeddings.get_model.cache_clear()
    yield FakeTextEmbedding
    embeddings.get_model.cache_clear()


class TestEmbed:
    def test_returns_float32_vector_of_model_dimension(self, fake_model):
        vec = embeddings.embed("def login(): pass")
        assert vec.shape == (embeddings.EMBEDDING_DIM,)
        assert vec.dtype == np.float32

    def test_loads_model_once_across_calls(self, fake_model):
        embeddings.embed("first")
        embeddings.embed("second")
        assert fake_model.instances == 1

    def test_does_not_load_model_until_first_embed(self, fake_model):
        assert fake_model.instances == 0

    def test_uses_configured_model_and_cache_dir(self, fake_model):
        model = embeddings.get_model()
        assert model.model_name == settings.embedding_model
        assert model.cache_dir == str(settings.embedding_cache_dir)


class TestRealModel:
    """Runs the real ONNX model (downloads ~90MB on first run)."""

    def test_vectors_are_unit_length(self):
        vec = embeddings.embed("def login(username, password): return True")
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)

    def test_related_code_is_closer_than_unrelated_code(self):
        query = embeddings.embed("how does user authentication work?")
        related = embeddings.embed("def login(username, password): check_password(password)")
        unrelated = embeddings.embed("def resize_image(img, width, height): return img")
        assert np.linalg.norm(query - related) < np.linalg.norm(query - unrelated)
