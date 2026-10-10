"""Lokalny enkoder dense ONNX (ONNX Runtime) — hot-path bez torch.

Model referencyjny: OPI-PIB/PolDense-68M (ModernBERT, hidden_size=512, 19 warstw,
max_position_embeddings=7999, pooling CLS + L2-normalizacja w grafie). Artefakt
ONNX (opset 17) ma wejścia ``input_ids`` / ``attention_mask`` i wyjście
``embedding`` o kształcie ``[batch, 512]``.

Prefiksy wymagane przez producenta modelu (README OPI-PIB):
  * ``encode_query`` → ``'[query]: '`` (asymetryczny retrieval),
  * ``'[sts]: '`` dla symetrii (``decode_sts_prefix``/``encode_sts``),
  * dokumenty przechowywane w pamięci kodujemy BEZ prefiksu.

Doktryna:
  * ZERO importu torch, ZERO nowych zależności (``onnxruntime`` wchodzi
    tranzytywnie z ``fastembed``, ``tokenizers`` z ``fastembed``/``transformers``) —
    dlatego NIE dopisujemy nic do ``pyproject.toml``.
  * Importy inferencyjne są LENIWE (w ``_initialize``) — sam import modułu nie
    ładuje ONNX Runtime ani tokenizerów.
  * Brak artefaktów ⇒ ``EncoderUnavailableError``. Żadnego cichego fallbacku do
    n-gramów (to była dziura obecnego ``FastEmbedEncoder``).
  * Wymiar wektora pochodzi WYŁĄCZNIE z ``config.json`` modelu i NIGDY nie jest
    po cichu skracany; niezgodność z ``dimension`` ⇒ ``ValueError``.

Etykieta licencji w karcie modelu OPI-PIB/PolDense-68M: ``gemma`` (odnotowane
wprost — przed użyciem produkcyjnym wymaga potwierdzenia z właścicielem modelu).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ENV_MODEL_DIR = "ATLAS_EMBED_MODEL_DIR"
DEFAULT_MODEL_DIR = Path.home() / ".hermes" / "atlas" / "models" / "poldense68"

QUERY_PREFIX = "[query]: "
STS_PREFIX = "[sts]: "

ONNX_RELPATH = "onnx/model.onnx"
TOKENIZER_RELPATH = "tokenizer.json"
CONFIG_RELPATH = "config.json"

# V88b/T7: uniwersalny korpus centrowania (post-hoc centering osadzeń). Frazy są
# tematycznie neutralne i różnorodne (pol. + ang., 15 rekordów) — średnia z nich
# estymuje anizotropowy komponent wspólny PolDense-68M bez zależności od domeny
# użytkownika i bez wczytywania czegokolwiek poza repo (zero hardcodowanych ścieżek).
DEFAULT_CENTERING_CORPUS: Tuple[str, ...] = (
    "konfiguracja serwera nginx jako reverse proxy dla ruchu http",
    "raport finansowy kwartalny przychody koszty zysk",
    "trening silowy plan tygodniowy dla początkujących",
    "pogoda jutro opady deszczu temperatura wiatr",
    "recenzja książki science fiction o podróżach kosmicznych",
    "testy obciążeniowe api apache bench przepustowość",
    "montaż półki ściennej kołki rozporowe wkrętarka",
    "pieczenie chleba na zakwasie przepis domowy",
    "aktualizacja sterownika amd rocm linux",
    "wakacje nad morzem plaża wladyslawowo atrakcje",
    "restart usługi atlas-memory przez systemctl",
    "stan pamięci ATLAS w bazie SQLite katalog domowy",
    "sleep cycle consolidation gc węzłów grafu",
    "zapis faktu klucz fact confidence",
    "pomiar recall mediana 12 ms 512d poldense68 cpu",
)

# Kolejność ma znaczenie: to jest to, co raportujemy użytkownikowi/Orchestratorowi.
REQUIRED_ARTIFACTS = (ONNX_RELPATH, TOKENIZER_RELPATH, CONFIG_RELPATH)

DEFAULT_INTRA_OP_THREADS = 6
DEFAULT_MAX_LENGTH = 512
NORM_EPS = 1e-12


class EncoderUnavailableError(RuntimeError):
    """Brak artefaktów modelu / brak biblioteki inferencyjnej — twardy błąd.

    Świadomie NIE ma fallbacku: cicha degradacja do n-gramów maskuje brak
    enkodera i zatruwa kolekcję wektorową wektorami o innej przestrzeni.
    """


def resolve_model_dir(model_dir: Optional[str] = None) -> Path:
    """Kolejność rozwiązywania katalogu modelu: argument → ENV → domyślny."""
    if model_dir:
        return Path(model_dir).expanduser()
    env_value = os.environ.get(ENV_MODEL_DIR, "").strip()
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_MODEL_DIR


def missing_artifacts(directory: Path) -> List[str]:
    """Zwraca listę brakujących plików wymaganych przez enkoder (pusta = OK)."""
    directory = Path(directory)
    return [rel for rel in REQUIRED_ARTIFACTS if not (directory / rel).is_file()]


class OnnxDenseEncoder:
    """Dense encoder ONNX (CLS pooling + L2) z leniwą sesją i twardym kontraktem wymiaru."""

    def __init__(
        self,
        dimension: Optional[int] = None,
        model_dir: Optional[str] = None,
        lazy_init: bool = True,
        *,
        intra_op_num_threads: int = DEFAULT_INTRA_OP_THREADS,
        max_length: int = DEFAULT_MAX_LENGTH,
        corpus_centering: bool = False,
        centering_corpus: Optional[Sequence[str]] = None,
    ) -> None:
        self._dimension = int(dimension) if dimension is not None else None
        self.model_dir = str(model_dir) if model_dir else None
        self.intra_op_num_threads = max(1, int(intra_op_num_threads))
        self.max_length = max(1, int(max_length))

        self._session = None
        self._tokenizer = None
        self._output_name = "embedding"
        self._initialized = False
        self._lock = threading.Lock()

        # V88b/T7: post-hoc centering (odjęcie średniej korpusu + renormalizacja L2).
        # Domyślnie WYŁĄCZONE — włączone zmienia przestrzeń wektorową kolekcji, więc
        # wymaga świadomej decyzji (env ATLAS_EMBED_CENTERING / konstruktor) i re-embedu.
        self._centering_enabled = bool(corpus_centering)
        self._centering_corpus: List[str] = list(
            centering_corpus or (DEFAULT_CENTERING_CORPUS if corpus_centering else ())
        )
        self._centering_mean: Optional[np.ndarray] = None

        if not lazy_init:
            self._initialize()

    # ------------------------------------------------------------------ setup
    @property
    def dimension(self) -> Optional[int]:
        """Wymiar wektora wykryty z modelu (None dopóki sesja nie jest zainicjalizowana)."""
        return self._dimension

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def resolved_model_dir(self) -> Path:
        return resolve_model_dir(self.model_dir)

    @property
    def model_path(self) -> Path:
        return self.resolved_model_dir / ONNX_RELPATH

    def _initialize(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return

            directory = self.resolved_model_dir
            missing = missing_artifacts(directory)
            if missing:
                raise EncoderUnavailableError(
                    f"Brak artefaktów enkodera w {directory}: {', '.join(missing)}. "
                    f"Skopiuj model (onnx/model.onnx, tokenizer.json, config.json) albo ustaw "
                    f"{ENV_MODEL_DIR}. Enkoder nie ma fallbacku."
                )

            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
            except ImportError as exc:  # pragma: no cover - zależność tranzytywna fastembed
                raise EncoderUnavailableError(
                    "Brak biblioteki inferencyjnej: wymagane 'onnxruntime' i 'tokenizers' "
                    f"(przychodzą tranzytywnie z fastembed). Import error: {exc}"
                ) from exc

            config = json.loads((directory / CONFIG_RELPATH).read_text(encoding="utf-8"))
            hidden_size = config.get("hidden_size")
            if not isinstance(hidden_size, int) or hidden_size <= 0:
                raise EncoderUnavailableError(
                    f"config.json w {directory} nie zawiera poprawnego 'hidden_size': {hidden_size!r}"
                )
            if self._dimension is not None and self._dimension != hidden_size:
                raise ValueError(
                    f"Wykryty wymiar modelu ({hidden_size}) != wymiar oczekiwany ({self._dimension}). "
                    "Odmawiam cichego obcięcia/uzupełnienia wektora — popraw konfigurację kolekcji."
                )
            self._dimension = hidden_size

            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = self.intra_op_num_threads
            session = ort.InferenceSession(
                str(directory / ONNX_RELPATH),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            input_names = [i.name for i in session.get_inputs()]
            for required in ("input_ids", "attention_mask"):
                if required not in input_names:
                    raise EncoderUnavailableError(
                        f"Graf ONNX nie ma wejścia '{required}' (ma: {input_names})"
                    )
            output_shape = session.get_outputs()[0].shape
            graph_dim = output_shape[-1] if isinstance(output_shape[-1], int) else None
            if graph_dim is not None and graph_dim != hidden_size:
                raise ValueError(
                    f"Graf ONNX zwraca wymiar {graph_dim}, a config.json deklaruje {hidden_size}."
                )

            tokenizer = Tokenizer.from_file(str(directory / TOKENIZER_RELPATH))
            tokenizer.enable_truncation(max_length=self.max_length)
            tokenizer.enable_padding(
                pad_id=int(config.get("pad_token_id", 0)),
                pad_token="[PAD]",
            )

            # Publikujemy dopiero kompletny, spójny stan.
            self._session = session
            self._output_name = session.get_outputs()[0].name
            self._tokenizer = tokenizer
            self._initialized = True

    # ------------------------------------------------------------- centering
    @property
    def corpus_centering(self) -> bool:
        """Czy post-hoc centering (odjęcie średniej korpusu + L2) jest włączony."""
        return self._centering_enabled

    def enable_corpus_centering(self, corpus: Optional[Sequence[str]] = None) -> None:
        """Włącza post-hoc centering osadzeń (V88b/T7).

        `corpus` (None = wbudowany DEFAULT_CENTERING_CORPUS) ESTYMUJE średnią
        anizotropowego komponentu wspólnego. Transform jest przyłożony do wszystkich
        wyjść identycznie, więc kosinus dwóch wyjść = kosinus centrowany. UWAGA:
        zmienia przestrzeń wektorową — istniejąca kolekcja wymaga re-embedu.
        """
        corpus = corpus or DEFAULT_CENTERING_CORPUS
        self._centering_corpus = [str(t) for t in corpus]
        self._centering_enabled = True
        self._centering_mean = None  # wymuś re-fit przy najbliższym wywołaniu

    def disable_corpus_centering(self) -> None:
        """Wyłącza post-hoc centering (M8: krótko-circuit transform musi być wykrywalny)."""
        self._centering_enabled = False
        self._centering_mean = None

    def _fit_centering_mean(self) -> None:
        """Osadza korpus centrowania RAW (bez transformu) i estymuje średnią."""
        if not self._centering_corpus:
            self._centering_mean = None
            return
        raw = self._infer(list(self._centering_corpus))
        mean = raw.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        self._centering_mean = mean / norm if norm > NORM_EPS else mean

    def _apply_post_hoc(self, matrix: np.ndarray) -> np.ndarray:
        """Odjęcie średniej korpusu + renormalizacja L2 (post-hoc, bez zmiany wymiaru)."""
        if self._centering_mean is None:
            return matrix
        centered = matrix - self._centering_mean[None, :]
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        return centered / np.maximum(norms, NORM_EPS)

    # ------------------------------------------------------------------ infer
    def _infer(self, texts: List[str]) -> np.ndarray:
        """Surowy przebieg ONNX + pooling CLS + normalizacja L2 (bez post-hoc)."""
        self._initialize()
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f"OnnxDenseEncoder oczekuje str, dostał {type(text).__name__}")

        tokenizer = self._tokenizer
        session = self._session
        if tokenizer is None or session is None:  # pragma: no cover - _initialize gwarantuje stan
            raise EncoderUnavailableError("Sesja ONNX nie została zainicjalizowana")

        encodings = tokenizer.encode_batch(texts, add_special_tokens=True)
        input_ids = np.asarray([enc.ids for enc in encodings], dtype=np.int64)
        attention_mask = np.asarray([enc.attention_mask for enc in encodings], dtype=np.int64)

        outputs = session.run(
            [self._output_name],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        matrix = np.asarray(outputs[0], dtype=np.float64)
        if matrix.ndim == 3:  # graf bez poolingu - bierzemy token CLS
            matrix = matrix[:, 0, :]
        if matrix.shape[0] != len(texts):
            raise EncoderUnavailableError(
                f"Graf zwrócił {matrix.shape[0]} wektorów dla {len(texts)} tekstów"
            )
        if matrix.shape[1] != self._dimension:
            raise ValueError(
                f"Graf zwrócił wektor o wymiarze {matrix.shape[1]}, "
                f"a wykryty wymiar modelu to {self._dimension}"
            )

        return matrix

    def _run(self, texts: List[str]) -> List[List[float]]:
        """Znormalizowane osadzenia z opcjonalnym post-hoc centering (V88b/T7)."""
        matrix = self._infer(texts)
        if self._centering_enabled and self._centering_mean is None:
            self._fit_centering_mean()
        if self._centering_mean is None:
            return [row.tolist() for row in matrix]
        if self._initialized and matrix.shape[1] != self._centering_mean.shape[0]:
            raise ValueError(
                f"Wymiar osadzenia ({matrix.shape[1]}) != wymiar średniej centrowania "
                f"({self._centering_mean.shape[0]})"
            )
        matrix = self._apply_post_hoc(matrix)
        return [row.tolist() for row in matrix]

    def encode(self, text: Any) -> Any:
        """Embedding dokumentu (bez prefiksu) — do zapisu w pamięci.

        Akceptuje pojedynczy ciąg znaków (zwraca List[float]) lub sekwencję (zwraca List[List[float]]).
        """
        if isinstance(text, str):
            return self._run([text])[0]
        if isinstance(text, (list, tuple)):
            return self._run(list(text))
        return self._run([str(text)])[0]

    def encode_query(self, text: str) -> List[float]:
        """Embedding zapytania retrieval (prefiks ``'[query]: '`` wymagany przez producenta)."""
        return self._run([QUERY_PREFIX + text])[0]

    def encode_sts(self, text: str) -> List[float]:
        """Embedding do podobieństwa symetrycznego (prefiks ``'[sts]: '``)."""
        return self._run([STS_PREFIX + text])[0]

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """True batching: jeden przebieg sesji ONNX na partię tekstów."""
        if not texts:
            return []
        return self._run(list(texts))


__all__ = [
    "DEFAULT_MODEL_DIR",
    "ENV_MODEL_DIR",
    "EncoderUnavailableError",
    "OnnxDenseEncoder",
    "QUERY_PREFIX",
    "REQUIRED_ARTIFACTS",
    "STS_PREFIX",
    "missing_artifacts",
    "resolve_model_dir",
]
