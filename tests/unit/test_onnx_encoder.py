"""Testy kontraktowe enkodera dense ONNX (V70.2 — PolDense-68M / ModernBERT 512D).

Testy dzielą się na dwie grupy:
  * kontraktowe bez artefaktów (brak artefaktów ⇒ twardy wyjątek, brak cichego
    obcięcia wymiaru, brak fallbacku do n-gramów, leniwa inicjalizacja),
  * inferencyjne — wykonywane TYLKO gdy katalog modelu istnieje; w przeciwnym
    razie czysto się pomijają (SKIPPED), nigdy nie padają.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from atlas_memory.l2_semantic.onnx_encoder import (
    ENV_MODEL_DIR,
    QUERY_PREFIX,
    EncoderUnavailableError,
    OnnxDenseEncoder,
    missing_artifacts,
    resolve_model_dir,
)

pytest.importorskip("onnxruntime", reason="onnxruntime jest zależnością tranzytywną fastembed")

MODEL_DIR = resolve_model_dir()
MISSING = missing_artifacts(MODEL_DIR)

needs_model = pytest.mark.skipif(
    bool(MISSING),
    reason=f"brak artefaktów modelu w {MODEL_DIR} (brakuje: {MISSING}) — testy inferencyjne pominięte",
)


# --------------------------------------------------------------------- kontrakt
def test_resolve_model_dir_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # 1) argument wygrywa
    assert resolve_model_dir(str(tmp_path)) == tmp_path
    # 2) potem ENV
    monkeypatch.setenv(ENV_MODEL_DIR, str(tmp_path / "z-env"))
    assert resolve_model_dir() == tmp_path / "z-env"
    # 3) na końcu katalog domyślny
    monkeypatch.delenv(ENV_MODEL_DIR, raising=False)
    assert resolve_model_dir().name == "poldense68"


def test_missing_artifacts_raise_encoder_unavailable(tmp_path: Path) -> None:
    assert missing_artifacts(tmp_path), "pusty katalog nie może raportować kompletnych artefaktów"
    with pytest.raises(EncoderUnavailableError) as excinfo:
        OnnxDenseEncoder(dimension=None, model_dir=str(tmp_path), lazy_init=False)
    message = str(excinfo.value)
    assert "onnx/model.onnx" in message
    assert ENV_MODEL_DIR in message


def test_lazy_init_does_not_touch_disk(tmp_path: Path) -> None:
    encoder = OnnxDenseEncoder(dimension=None, model_dir=str(tmp_path), lazy_init=True)
    assert encoder.is_initialized is False
    assert encoder.dimension is None
    # dopiero pierwsze użycie inicjalizuje - i wtedy twardo pada (żadnego fallbacku)
    with pytest.raises(EncoderUnavailableError):
        encoder.encode("cokolwiek")


def test_no_silent_dimension_truncation(tmp_path: Path) -> None:
    """ANTYWZORZEC FastEmbedEncoder (emb[:dimension]) jest niedozwolony."""
    _write_artifacts(tmp_path, hidden_size=512)
    for wrong in (384, 128, 1024):
        encoder = OnnxDenseEncoder(dimension=wrong, model_dir=str(tmp_path), lazy_init=True)
        with pytest.raises(ValueError) as excinfo:
            encoder._initialize()  # noqa: SLF001 - jawny test kontraktu inicjalizacji
        assert str(wrong) in str(excinfo.value)
        assert "512" in str(excinfo.value)


def test_onnx_encoder_source_never_imports_torch() -> None:
    source = Path(__file__).resolve().parents[2] / "src" / "atlas_memory" / "l2_semantic" / "onnx_encoder.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported, f"hot-path nie może importować torch: {sorted(imported)}"
    # Dozwolone są wyłącznie biblioteki hot-path (ONNX Runtime + tokenizer HF) oraz stdlib/numpy.
    # "torch" pozostaje ZAKAZANY (ADR-002); jego obecność łapie asercja powyżej.
    assert imported <= {
        "json", "logging", "os", "threading", "pathlib", "typing", "numpy", "__future__",
        "onnxruntime", "tokenizers",
    }


def _write_artifacts(directory: Path, hidden_size: int) -> None:
    (directory / "onnx").mkdir(parents=True, exist_ok=True)
    (directory / "onnx" / "model.onnx").write_bytes(b"nieprawdziwy-graf")
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    (directory / "config.json").write_text(
        json.dumps({"hidden_size": hidden_size, "model_type": "modernbert"}), encoding="utf-8"
    )


# ------------------------------------------------------------------ inferencja
@needs_model
def test_dimension_comes_from_config_json() -> None:
    expected = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))["hidden_size"]
    encoder = OnnxDenseEncoder(dimension=None, model_dir=str(MODEL_DIR), lazy_init=False)
    assert encoder.dimension == expected
    assert encoder.dimension != 384, "384 to wymiar bge-small - pomylka konfiguracji"


@needs_model
def test_vector_length_equals_model_dimension() -> None:
    encoder = OnnxDenseEncoder(lazy_init=False)
    vector = encoder.encode("Zażółć gęślą jaźń - test wymiaru wektora.")
    assert len(vector) == encoder.dimension
    assert all(isinstance(value, float) for value in vector)


@needs_model
def test_vector_is_l2_normalized() -> None:
    encoder = OnnxDenseEncoder(lazy_init=False)
    norm = sum(value * value for value in encoder.encode("Kaucja wynosi 5000 złotych.")) ** 0.5
    assert abs(norm - 1.0) <= 1e-4


@needs_model
def test_encode_is_deterministic_bit_for_bit() -> None:
    encoder = OnnxDenseEncoder(lazy_init=False)
    text = "Umowa najmu mieszkania w Łodzi."
    assert encoder.encode(text) == encoder.encode(text)
    other_encoder = OnnxDenseEncoder(lazy_init=False)
    assert other_encoder.encode(text) == encoder.encode(text)


@needs_model
def test_different_texts_give_different_vectors() -> None:
    encoder = OnnxDenseEncoder(lazy_init=False)
    first = encoder.encode("Termin płatności faktury mija w piątek.")
    second = encoder.encode("Parametry techniczne pompy ciepła COP 4.2.")
    assert first != second
    cosine = sum(a * b for a, b in zip(first, second, strict=True))
    assert cosine < 0.999


@needs_model
def test_encode_query_differs_from_encode_by_prefix() -> None:
    encoder = OnnxDenseEncoder(lazy_init=False)
    text = "limity kredytowe dla spółki"
    document = encoder.encode(text)
    query = encoder.encode_query(text)
    assert query != document
    assert query == encoder.encode(QUERY_PREFIX + text)


@needs_model
def test_encode_batch_matches_single_encode() -> None:
    encoder = OnnxDenseEncoder(lazy_init=False)
    texts = ["pierwszy tekst o windykacji", "drugi tekst o księgowości", "trzeci"]
    batch = encoder.encode_batch(texts)
    assert len(batch) == len(texts)
    for single, batched in zip(texts, batch, strict=True):
        assert batched == encoder.encode(single)
    assert encoder.encode_batch([]) == []


@needs_model
def test_stored_text_normalization_then_encode_is_encodable() -> None:
    from atlas_memory.l2_semantic.textnorm import normalize_stored_text

    encoder = OnnxDenseEncoder(lazy_init=False)
    raw = "Wniosek \\u0142\\u0105czy si\\u0119 z\\nza\\u0142\\u0105cznikiem \\ud83d"
    vector = encoder.encode(normalize_stored_text(raw))
    assert len(vector) == encoder.dimension
    assert abs(sum(value * value for value in vector) ** 0.5 - 1.0) <= 1e-4
