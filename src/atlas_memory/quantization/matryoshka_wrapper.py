"""
Matryoshka Representation Learning (MRL) Quantization Wrapper (arXiv:2205.13147).

Zapewnia adaptacyjną precyzję wektorów embeddingów poprzez hierarchiczne
wymiary (np. 64 -> 128 -> 256 -> 512 -> 1536), umożliwiając błyskawiczne
shortlistowanie na niskich wymiarach i precyzyjny reranking na wyższych.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class MatryoshkaResult(BaseModel):
    """
    Struktura przechowująca wynik adaptacyjnego wyszukiwania Matryoshka MRL.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_indices: List[int] = Field(
        ..., description="Indeksy wektorów z korpusu uporządkowane malejąco wg podobieństwa"
    )
    dimensions_used: List[int] = Field(
        ..., description="Lista wymiarów użytych w poszczególnych etapach wyszukiwania"
    )
    recall_achieved: float = Field(
        ..., ge=0.0, le=1.0, description="Osiągnięta dokładność Recall@k względem pełnego wymiaru"
    )
    n_candidates: int = Field(
        ..., ge=0, description="Liczba kandydatów przekazanych do finalnego rerankingu"
    )


class MatryoshkaEmbedding:
    """
    Silnik hierarchicznego wyszukiwania Matryoshka Representation Learning (MRL).

    Umożliwia wielopoziomowe przeszukiwanie przestrzeni wektorowych:
    - shortlist(): natychmiastowe filtrowanie kandydatów na najniższym wymiarze (np. 64-dim)
    - rerank(): precyzyjny reranking wyselekcjonowanych wektorów na wymiarze docelowym (max_dim)
    - adaptive_search(): dynamiczny pipeline podbijający wymiar aż do osiągnięcia zadanego recall_target
    """

    def __init__(
        self,
        max_dim: int = 1536,
        dimensions: Optional[List[int]] = None,
    ) -> None:
        """
        Inicjalizuje wrapper Matryoshka MRL.

        Args:
            max_dim: Maksymalny (pełny) wymiar embeddingu (np. 1536 lub 384).
            dimensions: Lista rosnących wymiarów sub-wektorów MRL.
        """
        if max_dim <= 0:
            raise ValueError(f"Wymiar max_dim musi być dodatni, otrzymano: {max_dim}")

        dims = dimensions or [64, 128, 256, 512, 1536]
        if not dims:
            raise ValueError("Lista wymiarów nie może być pusta")

        sorted_dims = sorted(list(set(dims)))
        if sorted_dims[0] <= 0:
            raise ValueError("Wszystkie wymiary muszą być dodatnie")
        if sorted_dims[-1] > max_dim:
            raise ValueError(
                f"Wymiar w dimensions ({sorted_dims[-1]}) nie może przekraczać max_dim ({max_dim})"
            )
        if max_dim not in sorted_dims:
            sorted_dims.append(max_dim)
            sorted_dims.sort()

        self.max_dim = max_dim
        self.dimensions = sorted_dims

    def shortlist(
        self,
        query: np.ndarray,
        corpus: np.ndarray,
        top_k: int = 100,
    ) -> np.ndarray:
        """
        Szybkie shortlistowanie kandydatów przy użyciu najniższego wymiaru Matryoshka (np. 64-dim).

        Args:
            query: Wektor zapytania o wymiarze (D,) lub (1, D).
            corpus: Macierz wektorów dokumentów o kształcie (N, D).
            top_k: Liczba najlepszych kandydatów do wyselekcjonowania.

        Returns:
            Tablica indeksów najlepszych kandydatów w typie np.int64.
        """
        c = np.asarray(corpus, dtype=np.float32)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        n_vecs = c.shape[0]
        if n_vecs == 0:
            return np.empty((0,), dtype=np.int64)

        k = max(1, min(top_k, n_vecs))
        dim_0 = self.dimensions[0]
        dim_use = min(dim_0, c.shape[1])

        q = np.asarray(query, dtype=np.float32).reshape(-1)[:dim_use]
        q_norm = float(np.linalg.norm(q))
        q_unit = q / max(q_norm, 1e-12)

        c_sub = c[:, :dim_use]
        c_norms = np.linalg.norm(c_sub, axis=1, keepdims=True)
        c_unit = c_sub / np.maximum(c_norms, 1e-12)

        sims = c_unit @ q_unit
        top_indices = np.argsort(-sims)[:k]
        return top_indices.astype(np.int64)

    def rerank(
        self,
        query: np.ndarray,
        candidates: np.ndarray,
    ) -> np.ndarray:
        """
        Pełne rerankowanie wyselekcjonowanych wektorów kandydatów na pełnym wymiarze max_dim.

        Args:
            query: Wektor zapytania.
            candidates: Macierz wektorów kandydatów o kształcie (K, D).

        Returns:
            Względne indeksy uporządkowane malejąco wg podobieństwa kosinusowego.
        """
        c = np.asarray(candidates, dtype=np.float32)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        n_candidates = c.shape[0]
        if n_candidates == 0:
            return np.empty((0,), dtype=np.int64)

        dim_use = min(self.max_dim, c.shape[1])
        q = np.asarray(query, dtype=np.float32).reshape(-1)[:dim_use]
        q_norm = float(np.linalg.norm(q))
        q_unit = q / max(q_norm, 1e-12)

        c_sub = c[:, :dim_use]
        c_norms = np.linalg.norm(c_sub, axis=1, keepdims=True)
        c_unit = c_sub / np.maximum(c_norms, 1e-12)

        sims = c_unit @ q_unit
        ranked_order = np.argsort(-sims)
        return ranked_order.astype(np.int64)

    def shortlist_then_rerank(
        self,
        query: np.ndarray,
        corpus: np.ndarray,
        top_k: int = 10,
        shortlist_k: int = 100,
    ) -> np.ndarray:
        """
        Kompletny dwustopniowy potok MRL:
        1. Shortlist na najniższym wymiarze (dimensions[0]).
        2. Rerank na max_dim.

        Args:
            query: Wektor zapytania.
            corpus: Macierz wektorów korpusu.
            top_k: Liczba docelowych wyników.
            shortlist_k: Rozmiar wstępnej listy kandydatów.

        Returns:
            Indeksy z korpusu posortowane malejąco.
        """
        cand_indices = self.shortlist(query, corpus, top_k=shortlist_k)
        if len(cand_indices) == 0:
            return np.empty((0,), dtype=np.int64)

        cand_vecs = corpus[cand_indices]
        ranked_order = self.rerank(query, cand_vecs)
        final_indices = cand_indices[ranked_order][:top_k]
        return final_indices.astype(np.int64)

    def adaptive_search(
        self,
        query: np.ndarray,
        corpus: np.ndarray,
        recall_target: float = 0.95,
        max_candidates: int = 100,
        k_eval: int = 10,
    ) -> MatryoshkaResult:
        """
        Adaptacyjny pipeline MRL:
        Shortlistuje na najniższym wymiarze -> sprawdza recall -> jeśli < recall_target,
        podbija wymiar do kolejnego poziomu aż do osiągnięcia celu lub max_dim.

        Args:
            query: Wektor zapytania.
            corpus: Macierz wektorów korpusu (N, D).
            recall_target: Docelowy próg dokładności Recall@k (domyślnie 0.95).
            max_candidates: Maksymalna liczba kandydatów w puli.
            k_eval: Liczba top-k elementów używana do ewaluacji recall.

        Returns:
            MatryoshkaResult ze szczegółami wyszukiwania.
        """
        c = np.asarray(corpus, dtype=np.float32)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        n_vecs = c.shape[0]
        if n_vecs == 0:
            return MatryoshkaResult(
                final_indices=[],
                dimensions_used=[self.dimensions[0]],
                recall_achieved=1.0,
                n_candidates=0,
            )

        k_eval = max(1, min(k_eval, n_vecs))
        cand_k = max(k_eval, min(max_candidates, n_vecs))

        # Obliczenie dokładnego rankingu referencyjnego na max_dim
        dim_full = min(self.max_dim, c.shape[1])
        q_full = np.asarray(query, dtype=np.float32).reshape(-1)[:dim_full]
        q_norm = float(np.linalg.norm(q_full))
        q_full_unit = q_full / max(q_norm, 1e-12)

        c_full = c[:, :dim_full]
        c_full_norms = np.linalg.norm(c_full, axis=1, keepdims=True)
        c_full_unit = c_full / np.maximum(c_full_norms, 1e-12)

        true_sims = c_full_unit @ q_full_unit
        ground_truth_topk = set(np.argsort(-true_sims)[:k_eval].tolist())

        dimensions_used: List[int] = []
        current_candidates = np.arange(n_vecs)
        recall_achieved = 0.0

        for dim in self.dimensions:
            dimensions_used.append(dim)
            dim_use = min(dim, c.shape[1])

            q_sub = q_full[:dim_use]
            q_sub_unit = q_sub / max(float(np.linalg.norm(q_sub)), 1e-12)

            cand_vecs = c[current_candidates, :dim_use]
            cand_norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
            cand_unit = cand_vecs / np.maximum(cand_norms, 1e-12)

            sims = cand_unit @ q_sub_unit
            ranked_local = np.argsort(-sims)

            # Zawężenie do cand_k
            top_local = ranked_local[:cand_k]
            current_candidates = current_candidates[top_local]

            # Obliczenie recall@k_eval dla obecnego zestawu kandydatów
            intersection = len(
                set(current_candidates[:k_eval].tolist()).intersection(ground_truth_topk)
            )
            recall_achieved = float(intersection / float(k_eval))

            if recall_achieved >= recall_target or dim >= self.max_dim:
                break

        # Finalny reranking kandydatów na pełnym wymiarze
        cand_full_vecs = c[current_candidates]
        ranked_order = self.rerank(query, cand_full_vecs)
        final_indices = current_candidates[ranked_order].tolist()

        # Ostateczny zmierzony recall
        final_intersection = len(
            set(final_indices[:k_eval]).intersection(ground_truth_topk)
        )
        final_recall = float(final_intersection / float(k_eval))

        return MatryoshkaResult(
            final_indices=final_indices,
            dimensions_used=dimensions_used,
            recall_achieved=final_recall,
            n_candidates=len(final_indices),
        )

    async def adaptive_search_async(
        self,
        query: np.ndarray,
        corpus: np.ndarray,
        recall_target: float = 0.95,
        max_candidates: int = 100,
        k_eval: int = 10,
    ) -> MatryoshkaResult:
        """
        Asynchroniczny wariant metody adaptive_search.
        """
        return self.adaptive_search(
            query=query,
            corpus=corpus,
            recall_target=recall_target,
            max_candidates=max_candidates,
            k_eval=k_eval,
        )
