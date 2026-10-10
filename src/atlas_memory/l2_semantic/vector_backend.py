"""Wybór backendu wektorowego ATLAS (512D PolDense-68M ONNX).

Jedno miejsce decyzyjne. Plik `~/.hermes/atlas/vector_backend.json` (albo ścieżka z
`ATLAS_VECTOR_CONFIG`) opisuje: enkoder, wymiar i kolekcję.
Domyślny i jedyny produkcyjny standard: PolDense-68M (512D, kolekcja `hermes_episodic_memory_68m`).
Stare 384D (bge-small / fastembed) zostało trwale wycofane.

Doktryna:
  * ŻADNEGO cichego fallbacku: nieznany `kind` albo brak artefaktów modelu ⇒ twardy
    wyjątek. Cicha degradacja do innego enkodera zatruwa kolekcję wektorami z innej
    przestrzeni (nieodwracalne bez re-embedu).
  * Zgodność wymiaru jest sprawdzana PRZED otwarciem magazynu: `dimension` musi
    zgadzać się z `config.json` modelu, inaczej ValueError.
  * Stary config nie jest modyfikowany — plik jest tylko czytany.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from atlas_memory.l2_semantic.onnx_encoder import (
    EncoderUnavailableError,
    OnnxDenseEncoder,
    missing_artifacts,
    resolve_model_dir,
)

logger = logging.getLogger(__name__)

ENV_CONFIG_PATH = "ATLAS_VECTOR_CONFIG"
DEFAULT_CONFIG_PATH = Path.home() / ".hermes" / "atlas" / "vector_backend.json"

KIND_ONNX68 = "onnx68"
KIND_FASTEMBED = "fastembed"  # Wycofany

POST_HOC_NONE = ""
POST_HOC_CENTER = "center"
KNOWN_POST_HOC = (POST_HOC_NONE, POST_HOC_CENTER)
KNOWN_KINDS = (KIND_ONNX68,)

DEFAULT_COLLECTION = "hermes_episodic_memory_68m"
DEFAULT_DIMENSION = 512
ONNX68_DIMENSION = 512


@dataclass(frozen=True)
class VectorBackend:
    """Opis aktywnego backendu wektorowego (serializowany do get_stats)."""

    kind: str = KIND_ONNX68
    collection: str = DEFAULT_COLLECTION
    dimension: int = DEFAULT_DIMENSION
    model_dir: Optional[str] = None
    post_hoc: str = POST_HOC_NONE
    source: str = "defaults"

    def describe(self) -> dict:
        data = asdict(self)
        if self.kind == KIND_ONNX68:
            data["model_dir"] = str(resolve_model_dir(self.model_dir))
            missing = missing_artifacts(Path(data["model_dir"]))
            data["missing_artifacts"] = missing
        return data


def config_path(path: Optional[Path | str] = None) -> Path:
    if path:
        return Path(path).expanduser()
    env_value = os.environ.get(ENV_CONFIG_PATH, "").strip()
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_CONFIG_PATH


def load_vector_backend(path: Optional[Path | str] = None) -> VectorBackend:
    """Czyta konfigurację backendu. Brak pliku ⇒ domyślny 512D PolDense-68M."""
    resolved = config_path(path)
    if not resolved.is_file():
        return VectorBackend()

    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Nieczytelny {resolved}: {exc}") from exc

    kind = str(raw.get("kind", "")).strip()
    if kind == KIND_FASTEMBED:
        raise ValueError(
            f"{resolved}: kind='fastembed' (384D) został trwale usunięty z ATLAS. "
            f"Dozwolone: {sorted(KNOWN_KINDS)} (512D PolDense-68M)."
        )
    if kind not in KNOWN_KINDS:
        raise ValueError(
            f"{resolved}: kind={kind!r} nieznany; dozwolone: {sorted(KNOWN_KINDS)} "
            "(celowo bez fallbacku — zły backend nie może po cichu udawać dobrego)"
        )

    backend = VectorBackend(
        kind=kind,
        collection=str(raw.get("collection") or DEFAULT_COLLECTION),
        dimension=int(raw.get("dimension") or ONNX68_DIMENSION),
        model_dir=raw.get("model_dir"),
        post_hoc=str(raw.get("post_hoc", POST_HOC_NONE)).strip(),
        source=str(resolved),
    )
    _validate(backend)
    return backend


def _validate(backend: VectorBackend) -> None:
    if backend.post_hoc not in KNOWN_POST_HOC:
        raise ValueError(
            f"post_hoc={backend.post_hoc!r} nieznany; dozwolone: {sorted(KNOWN_POST_HOC)} "
            "(celowo bez fallbacku — cicha zmiana przestrzeni osadzen zatruwa kolekcje)"
        )
    if backend.kind != KIND_ONNX68:
        return
    model_dir = resolve_model_dir(backend.model_dir)
    missing = missing_artifacts(model_dir)
    if missing:
        raise EncoderUnavailableError(
            f"Backend {KIND_ONNX68}: brak artefaktów w {model_dir}: {missing}. "
            "Brak enkodera to twardy błąd — nie ma przełączenia na inne modele."
        )
    hidden = _hidden_size(model_dir)
    if hidden is not None and hidden != backend.dimension:
        raise ValueError(
            f"Backend {KIND_ONNX68}: dimension={backend.dimension} w konfiguracji, "
            f"ale config.json modelu mówi hidden_size={hidden}. "
            "Zgodność wymiaru jest obowiązkowa (inaczej wektory trafiają do złej przestrzeni)."
        )


def _hidden_size(model_dir: Path) -> Optional[int]:
    config_file = Path(model_dir) / "config.json"
    try:
        payload = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("hidden_size")
    return int(value) if isinstance(value, (int, float)) else None


_ENCODER_CACHE: dict[tuple[str, int, str, str], Any] = {}


def build_encoder(backend: VectorBackend) -> Any:
    """Zwraca enkoder dla backendu (OnnxDenseEncoder dla PolDense-68M) z cache per-sygnatura."""
    if backend.kind == KIND_ONNX68:
        resolved_dir = resolve_model_dir(backend.model_dir)
        cache_key = (backend.kind, backend.dimension, str(resolved_dir), backend.post_hoc)
        if not missing_artifacts(resolved_dir) and cache_key in _ENCODER_CACHE:
            return _ENCODER_CACHE[cache_key]
        encoder = OnnxDenseEncoder(dimension=backend.dimension, model_dir=backend.model_dir)
        encoder_dim = encoder.dimension
        if encoder_dim is not None and encoder_dim != backend.dimension:
            raise ValueError(
                f"Enkoder ONNX deklaruje {encoder_dim}D, konfiguracja {backend.dimension}D"
            )
        if backend.post_hoc == POST_HOC_CENTER:
            encoder.enable_corpus_centering()
            logger.warning(
                "Backend %s: post_hoc=center aktywny — kolekcja %s wymaga re-embeddingu "
                "(przestrzen osadzen rozni sie od wektorow zapisanych bez centrowania)",
                backend.kind, backend.collection,
            )
        _ENCODER_CACHE[cache_key] = encoder
        return encoder
    raise ValueError(f"Nieobsługiwany kind={backend.kind!r}")
