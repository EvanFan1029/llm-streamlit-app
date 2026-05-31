from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
_DEFAULT_DEVICE = "cpu"
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_CACHE_DIR = str(Path(__file__).resolve().parent / ".bert_cache")


@dataclass
class BERTConfig:
    model_name: str = _DEFAULT_MODEL
    device: str = _DEFAULT_DEVICE
    batch_size: int = _DEFAULT_BATCH_SIZE
    cache_dir: str = _DEFAULT_CACHE_DIR
    normalize_embeddings: bool = True


class BERTProcessor:
    _instance: Optional["BERTProcessor"] = None
    _model: Any = None
    _config: Optional[BERTConfig] = None

    def __init__(self, config: Optional[BERTConfig] = None):
        if BERTProcessor._instance is not None:
            raise RuntimeError("Use BERTProcessor.get_instance() instead")
        self._cfg = config or BERTConfig()
        self._model = None
        self._option_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._text_cache: Dict[str, np.ndarray] = {}

    @classmethod
    def get_instance(cls, config: Optional[BERTConfig] = None) -> "BERTProcessor":
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None
        cls._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as ex:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            ) from ex

        self._model = SentenceTransformer(
            self._cfg.model_name,
            device=self._cfg.device,
        )
        if self._cfg.normalize_embeddings:
            self._model.encode = self._make_normalized_encode(self._model)

        os.makedirs(self._cfg.cache_dir, exist_ok=True)
        self._load_cache()

    def _make_normalized_encode(self, model: Any) -> Any:
        original_encode = model.encode

        def normalized_encode(sentences, **kwargs):
            kwargs.setdefault("normalize_embeddings", True)
            return original_encode(sentences, **kwargs)

        return normalized_encode

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        self._ensure_loaded()
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        cached: List[Optional[np.ndarray]] = []
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._text_cache:
                cached.append(self._text_cache[key])
            else:
                cached.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            embeddings = self._model.encode(
                uncached_texts,
                batch_size=self._cfg.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for idx, text in zip(uncached_indices, uncached_texts):
                emb = embeddings[uncached_indices.index(idx)]
                if emb.ndim == 1:
                    emb = emb.reshape(1, -1)
                self._text_cache[self._cache_key(text)] = emb
                cached[idx] = emb

        result = np.vstack([c for c in cached if c is not None])
        return result

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_options(self, options: Tuple[str, ...], group: str = "default") -> Dict[str, np.ndarray]:
        self._ensure_loaded()
        if group in self._option_cache:
            return self._option_cache[group]

        opt_list = list(options)
        embeddings = self.embed_texts(opt_list)
        result = {opt: embeddings[i] for i, opt in enumerate(opt_list)}
        self._option_cache[group] = result
        self._save_cache()
        return result

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a.flatten()
        b = b.flatten()
        dot = float(np.dot(a, b))
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def batch_similarity(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
    ) -> np.ndarray:
        query = query_embedding.flatten()
        if candidate_embeddings.ndim == 1:
            candidate_embeddings = candidate_embeddings.reshape(1, -1)
        dot = np.dot(candidate_embeddings, query)
        na = np.linalg.norm(query)
        nb = np.linalg.norm(candidate_embeddings, axis=1)
        denom = na * nb
        denom[denom == 0] = 1.0
        return dot / denom

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_path(self) -> str:
        safe_name = self._cfg.model_name.replace("/", "_").replace(":", "_")
        return os.path.join(self._cfg.cache_dir, f"{safe_name}_cache.json")

    def _save_cache(self) -> None:
        try:
            data: Dict[str, Any] = {
                "text_cache": {k: v.tolist() for k, v in self._text_cache.items()},
                "option_cache": {
                    group: {opt: emb.tolist() for opt, emb in group_dict.items()}
                    for group, group_dict in self._option_cache.items()
                },
            }
            with open(self._cache_path(), "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
        except Exception:
            pass

    def _load_cache(self) -> None:
        path = self._cache_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._text_cache = {
                k: np.array(v, dtype=np.float32)
                for k, v in (data.get("text_cache", {}) or {}).items()
            }
            self._option_cache = {
                group: {opt: np.array(emb, dtype=np.float32) for opt, emb in group_dict.items()}
                for group, group_dict in (data.get("option_cache", {}) or {}).items()
            }
        except Exception:
            self._text_cache = {}
            self._option_cache = {}

    def clear_cache(self) -> None:
        self._text_cache.clear()
        self._option_cache.clear()
        path = self._cache_path()
        if os.path.exists(path):
            os.remove(path)


def precompute_labor_option_embeddings(
    processor: Optional[BERTProcessor] = None,
) -> Dict[str, np.ndarray]:
    if processor is None:
        processor = BERTProcessor.get_instance()

    return processor.embed_options(options, group="labor_options")
