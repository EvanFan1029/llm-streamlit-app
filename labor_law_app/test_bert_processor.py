from __future__ import annotations

import numpy as np
import pytest

from labor_law_app.bert_processor import BERTConfig, BERTProcessor


@pytest.fixture(autouse=True)
def reset_singleton():
    BERTProcessor.reset_instance()
    yield
    BERTProcessor.reset_instance()


@pytest.mark.slow
def test_bert_processor_loads_model():
    processor = BERTProcessor.get_instance()
    processor._ensure_loaded()
    assert processor.is_loaded


@pytest.mark.slow
def test_embed_single_returns_correct_shape():
    processor = BERTProcessor.get_instance()
    emb = processor.embed_single("劳动者未签书面劳动合同，被口头辞退")
    assert emb.ndim == 1
    assert emb.shape[0] == 1024


@pytest.mark.slow
def test_embed_texts_batch():
    processor = BERTProcessor.get_instance()
    texts = ["劳动关系", "劳务关系", "未签书面劳动合同"]
    embs = processor.embed_texts(texts)
    assert embs.shape == (3, 1024)


@pytest.mark.slow
def test_embedding_is_deterministic():
    processor = BERTProcessor.get_instance()
    text = "劳动者主张双倍工资和违法解除赔偿金"
    emb1 = processor.embed_single(text)
    emb2 = processor.embed_single(text)
    assert np.allclose(emb1, emb2, atol=1e-6)


@pytest.mark.slow
def test_cosine_similarity_identical_is_one():
    processor = BERTProcessor.get_instance()
    emb = processor.embed_single("劳动争议")
    sim = processor.cosine_similarity(emb, emb)
    assert abs(sim - 1.0) < 1e-5


@pytest.mark.slow
def test_cosine_similarity_different():
    processor = BERTProcessor.get_instance()
    emb_a = processor.embed_single("劳动关系")
    emb_b = processor.embed_single("劳务关系")
    sim = processor.cosine_similarity(emb_a, emb_b)
    assert 0.0 < sim < 1.0


@pytest.mark.slow
def test_batch_similarity_shape():
    processor = BERTProcessor.get_instance()
    query = processor.embed_single("未签书面劳动合同")
    candidates = processor.embed_texts([
        "未签书面劳动合同",
        "违法解除/辞退",
        "加班费争议",
    ])
    sims = processor.batch_similarity(query, candidates)
    assert sims.shape == (3,)
    assert sims[0] > sims[1]
    assert sims[0] > sims[2]


@pytest.mark.slow
def test_embed_options_precomputes():
    processor = BERTProcessor.get_instance()
    options = ("劳动关系倾向", "劳务/承揽关系倾向", "事实不清")
    result = processor.embed_options(options, group="test_options")
    assert len(result) == 3
    assert "劳动关系倾向" in result
    assert result["劳动关系倾向"].shape == (1024,)


@pytest.mark.slow
def test_empty_texts_returns_empty():
    processor = BERTProcessor.get_instance()
    embs = processor.embed_texts([])
    assert embs.shape == (0, 1024)


@pytest.mark.slow
def test_cache_save_load_roundtrip(tmp_path):
    BERTProcessor.reset_instance()
    config = BERTConfig(cache_dir=str(tmp_path))
    processor = BERTProcessor.get_instance(config)
    processor._ensure_loaded()

    text = "双倍工资"
    emb1 = processor.embed_single(text)

    BERTProcessor.reset_instance()
    processor2 = BERTProcessor.get_instance(config)
    processor2._ensure_loaded()
    emb2 = processor2.embed_single(text)

    assert np.allclose(emb1, emb2, atol=1e-6)
