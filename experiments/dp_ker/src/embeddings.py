from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from data_loader import SEED


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


@dataclass
class EmbeddingBackend:
    name: str
    dimension: int

    def fit(self, texts: Iterable[str]) -> "EmbeddingBackend":
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)


class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        super().__init__(name=f"sentence-transformers:{model_name}", dimension=self.model.get_sentence_embedding_dimension())

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        encoded = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(encoded, dtype=np.float32)


class TfidfSvdBackend(EmbeddingBackend):
    def __init__(self, max_features: int = 6000, max_components: int = 128) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
        )
        self.svd: TruncatedSVD | None = None
        self.max_components = max_components
        super().__init__(name="tfidf-svd", dimension=0)

    def fit(self, texts: Iterable[str]) -> "EmbeddingBackend":
        corpus = list(texts)
        matrix = self.vectorizer.fit_transform(corpus)
        n_features = matrix.shape[1]
        n_samples = matrix.shape[0]
        if n_features > 2 and n_samples > 1:
            n_components = min(self.max_components, n_features - 1, n_samples - 1)
            if n_components >= 2:
                self.svd = TruncatedSVD(n_components=n_components, random_state=SEED)
                self.svd.fit(matrix)
                self.dimension = n_components
            else:
                self.svd = None
                self.dimension = n_features
        else:
            self.svd = None
            self.dimension = n_features
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(list(texts))
        if self.svd is not None:
            dense = self.svd.transform(matrix)
        else:
            dense = matrix.toarray()
        dense = np.asarray(dense, dtype=np.float32)
        return _l2_normalize(dense)


def build_embedding_backend(texts: Iterable[str], prefer_sentence_transformers: bool = True) -> EmbeddingBackend:
    corpus = list(texts)
    if prefer_sentence_transformers:
        try:
            backend = SentenceTransformerBackend()
            backend.fit(corpus)
            return backend
        except Exception as exc:  # pragma: no cover - optional dependency fallback
            warnings.warn(f"sentence-transformers 不可用，回退到 TF-IDF/SVD embedding。原因: {exc}")
    backend = TfidfSvdBackend()
    backend.fit(corpus)
    return backend

