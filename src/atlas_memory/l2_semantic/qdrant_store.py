from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
import zlib
from hashlib import blake2b
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from atlas_memory.l2_semantic.sparse_encoder import (
    SPARSE_VECTOR_NAME,
    encode_sparse_document,
    encode_sparse_query,
)

logger = logging.getLogger(__name__)

# V70.1 (L2-OWNER): pojedynczy właściciel magazynu wektorów na ścieżce dyskowej.
OWNER_FILENAME = "owner.json"

# V72.2 (W3/W5): pole payloadu wiążące punkt z rekordem KV oraz produkcyjny prefiks
# tego klucza (scripts/rebuild_qdrant_68m.py pisze ``upsert_key="kv:<klucz>"``).
UPSERT_KEY_PAYLOAD_FIELD = "upsert_key"
KV_UPSERT_PREFIX = "kv:"
# Górne ograniczenie skanu kolekcji przy liczeniu pokrycia (H6) — nie blokujemy RPC.
_PREFIX_SCAN_PAGE_SIZE = 256
_PREFIX_SCAN_MAX_PAGES = 400
# V90 (BUG-Q2): schemat ID z blake2b-63bit zamiast crc32-31bit. Kolizja ID = ciche
# nadpisanie cudzego rekordu; prawdopodobienstwo maleje z 8,89% @20k (crc32) do ~1e-9 @20k.
_LEGACY_ID_MASK = (1 << 31) - 1          # crc32 & 0x7FFFFFFF — identyfikacja legacy punktow
_POINT_ID_MASK = (1 << 63) - 1           # 63 bity: qdrant zapisuje unsigned, maska trzyma margines
_POINT_ID_SAFETY_PAGES = 10_000          # straznik scrolla: 10k stron = ~2,5 mln punktow


def normalize_upsert_key(k: Optional[str]) -> str:
    """Canonicalizes upsert keys by removing 'kv:' prefix and whitespace."""
    if not k:
        return ""
    s = str(k).strip()
    if s.startswith(KV_UPSERT_PREFIX):
        s = s[len(KV_UPSERT_PREFIX):].strip()
    return s


def _point_id_for_key(key: str) -> int:
    """JEDNO zrodlo deterministycznego ID punktu (V90, BUG-Q2).

    blake2b(digest_size=8) -> uint64 & (2^63-1). Zastepuje crc32 & 0x7FFFFFFF (31 bitow).
    UWAGA: ID punktow zapisanych przed V90 ma schemat legacy — sweep
    migrate_legacy_point_ids_sync() przepisuje je przyrostowo.
    """
    canon = normalize_upsert_key(key)
    return int.from_bytes(blake2b(canon.encode("utf-8"), digest_size=8).digest(), "big") & _POINT_ID_MASK


class VectorStoreLockedError(RuntimeError):
    """Magazyn Qdrant na ścieżce dyskowej jest zajęty przez ŻYWY proces.

    V70.1: wcześniej kod bezwarunkowo kasował ``.lock`` i ponawiał otwarcie,
    co przy żywym właścicielu dopuszczało DWÓCH pisarzy do jednego katalogu
    (ryzyko korupcji). Teraz odzyskuję blokadę wyłącznie po DOWODZIE śmierci
    właściciela, a tutaj podnoszę ten wyjątek z instrukcją użycia RPC daemona.
    """

    def __init__(
        self,
        location: str,
        owner_pid: Optional[int] = None,
        owner_host: Optional[str] = None,
        started_at: Optional[float] = None,
    ) -> None:
        self.location = str(location)
        self.owner_pid = owner_pid
        self.owner_host = owner_host
        self.started_at = started_at
        detail = (
            f"Qdrant store at '{self.location}' is held by a LIVE process "
            f"(pid={owner_pid}, host={owner_host}"
            + (f", started_at={time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(started_at))}" if started_at else "")
            + "). Refusing to delete .lock — two concurrent writers on one Qdrant directory "
            "risks corruption. Use the ATLAS daemon RPC over the Unix socket "
            "(~/.hermes/atlas.sock) instead of opening this path directly."
        )
        super().__init__(detail)


def _host_name() -> str:
    return socket.gethostname() or socket.getfqdn() or "localhost"


def pid_is_alive(pid: int) -> bool:
    """True, jeśli proces o danym pid istnieje na TYM hoście.

    ``PermissionError`` traktujemy jako 'żywy' — proces istnieje, tylko należy
    do innego użytkownika (brak dowodu śmierci => nie odzyskujemy blokady).
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def read_vector_store_owner(location: str | os.PathLike[str]) -> Optional[Dict[str, Any]]:
    """Odczytuje token właściciela (``owner.json``) lub None, gdy go nie ma.

    Fallback dla katalogów utworzonych przed V70.1: treść ``.lock`` bywa samym
    pidem (np. w testach H21) — parsujemy ją wtedy jako pid bez hosta.
    """
    base = Path(location)
    token_path = base / OWNER_FILENAME
    try:
        if token_path.is_file():
            data = json.loads(token_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable ownership token %s: %s", token_path, exc)
    try:
        lock_path = base / ".lock"
        if lock_path.is_file():
            raw = lock_path.read_text(encoding="utf-8", errors="replace").strip()
            if raw.isdigit():
                return {"owner": "unknown(legacy .lock)", "pid": int(raw), "started_at": None, "host": None}
    except OSError:
        return None
    return None


try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

from atlas_memory.core.models import HERMES_DEFAULT_SESSION, HERMES_GLOBAL_SESSION, MemoryRecord
from atlas_memory.l2_semantic.textnorm import normalize_stored_text


class _StubDenseEncoder:
    """Deterministyczny stub enkodera (n-gram CRC32 projection) dla środowisk testowych."""

    def __init__(self, dimension: int = 512, **kwargs: Any) -> None:
        self.dimension = dimension

    def encode(self, text: Any) -> Any:
        if isinstance(text, (list, tuple)):
            return [self.encode(t) for t in text]
        text_clean = str(text or "").lower().strip()
        if not text_clean:
            return [0.0] * self.dimension
        rng = np.random.default_rng(42)
        proj = rng.standard_normal((1024, self.dimension))
        ngrams = [text_clean[i : i + 3] for i in range(max(1, len(text_clean) - 2))]
        vec = np.zeros(self.dimension)
        for ng in ngrams:
            idx = zlib.crc32(ng.encode("utf-8")) % 1024
            vec += proj[idx]
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
        return vec.tolist()

    def encode_batch(self, texts: Any) -> List[List[float]]:
        return [self.encode(t) for t in texts]

    def encode_query(self, text: str) -> List[float]:
        return self.encode(f"[query]: {text}")



class QdrantVectorStore:
    """
    L2: Produkcyjny Magazyn Wektorowy Qdrant (Embedded / On-Disk) + PolDense-68M ONNX (512D).
    
    Obsługuje tryb embedded (:memory: lub katalog na dysku ./data/qdrant)
    bez konieczności uruchamiania zewnętrznego kontenera Dockera.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        location: str = ":memory:",
        dimension: int = 512,
        encoder: Optional[Any] = None,  # strukturalnie: cokolwiek z .encode(str)->List[float]
        owner: str = "qdrant_vector_store",
        path: Optional[str] = None,
    ):
        if path is not None:
            location = str(path)
        self._explicit_collection = collection_name is not None
        self.collection_name = collection_name or "hermes_episodic_memory_68m"
        self.location = location
        self.owner_label = owner
        self.owner_path: Optional[Path] = None
        self.owner_token: Optional[Dict[str, Any]] = None
        self.dimension = dimension
        if encoder is not None:
            self.encoder = encoder
        elif dimension != 512:
            self.encoder = _StubDenseEncoder(dimension=dimension)
        else:
            try:
                from atlas_memory.l2_semantic.onnx_encoder import OnnxDenseEncoder
                self.encoder = OnnxDenseEncoder(dimension=dimension, lazy_init=True)
            except Exception:
                self.encoder = _StubDenseEncoder(dimension=dimension)
        self._lock = asyncio.Lock()

        if HAS_QDRANT:
            if location == ":memory:":
                self.client = QdrantClient(location=":memory:")
            else:
                self.client = self._open_disk_client(location)
                self._write_owner_token(location)

            try:
                existing_cols = [c.name for c in self.client.get_collections().collections]
                if existing_cols and self.collection_name not in existing_cols:
                    if "hermes_episodic_memory_68m" in existing_cols:
                        self.collection_name = "hermes_episodic_memory_68m"
                    else:
                        self.collection_name = existing_cols[0]
            except Exception as e_col:
                logger.debug("Failed to inspect existing collections: %s", e_col)

            # V90 (BUG-Q4): tryb lokalny wykrywamy na KLiencie, nie z nazwy ścieżki.
            self._local_mode = self._detect_local_mode()
            self._ensure_collection()
        else:
            self.client = None
            self._local_mode = False
            self._records: List[Dict[str, Any]] = []
            self._vectors: List[np.ndarray] = []

    def _detect_local_mode(self) -> bool:
        """True, gdy klient to wbudowany tryb lokalny (QdrantLocal)."""
        return self._client_is_local()

    def _client_is_local(self) -> bool:
        """V90 (BUG-Q4): rozpoznanie trybu wbudowanego po TYPIE klienta wewnętrznego.

        Dokumentacja i sam qdrant-client 1.19.0 mówią wprost: „Payload indexes have no
        effect in the local Qdrant" (ostrzeżenie biblioteki przy create_payload_index).
        W trybie serwerowym wewnętrzny klient to QdrantRemote, więc isinstance jest
        rozstrzygające i nie zgadujemy po kształcie ścieżki.
        """
        try:
            from qdrant_client.local.qdrant_local import QdrantLocal
        except Exception:  # pragma: no cover
            return False
        client = getattr(self, "client", None)
        if client is None:
            return False
        for attr in ("_client", "_local_client"):
            target = getattr(client, attr, None)
            if isinstance(target, QdrantLocal):
                return True
        return False

    def _open_disk_client(self, location: str) -> "QdrantClient":
        """Otwiera klienta na dysku, odzyskując ``.lock`` TYLKO po dowodzie śmierci właściciela (V70.1)."""
        try:
            return QdrantClient(path=location)
        except Exception as exc:
            text = str(exc).lower()
            if not ("lock" in text or "already accessed" in text):
                raise
            lock_file = Path(location) / ".lock"
            owner = read_vector_store_owner(location)
            owner_pid = owner.get("pid") if owner else None
            owner_host = owner.get("host") if owner else None
            started_at = owner.get("started_at") if owner else None
            this_host = _host_name()

            if owner_pid is None:
                raise VectorStoreLockedError(location, None, owner_host, started_at) from exc
            if owner_host and owner_host != this_host:
                # Właściciel na innym hoście — brak dowodu śmierci, nie ruszamy blokady.
                raise VectorStoreLockedError(location, int(owner_pid), owner_host, started_at) from exc
            if pid_is_alive(int(owner_pid)):
                raise VectorStoreLockedError(location, int(owner_pid), owner_host or this_host, started_at) from exc

            logger.warning(
                "Recovering stale Qdrant lock at %s: owner '%s' pid=%s on host=%s (started_at=%s) "
                "is DEAD at %s (pid %s not present on host %s). Removing .lock + owner.json.",
                location,
                (owner or {}).get("owner", "unknown"),
                owner_pid,
                owner_host or "unknown",
                time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_at)) if started_at else "unknown",
                time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                owner_pid,
                this_host,
            )
            lock_file.unlink(missing_ok=True)
            (Path(location) / OWNER_FILENAME).unlink(missing_ok=True)
            return QdrantClient(path=location)

    def _write_owner_token(self, location: str) -> None:
        """Zapisuje token właściciela (owner/pid/started_at/host) przy tworzeniu klienta na dysku."""
        token = {
            "owner": self.owner_label,
            "pid": os.getpid(),
            "started_at": time.time(),
            "host": _host_name(),
        }
        path = Path(location) / OWNER_FILENAME
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(token, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write ownership token %s: %s", path, exc)
            return
        self.owner_path = path
        self.owner_token = token
        logger.info(
            "Qdrant ownership acquired: file=%s owner=%s pid=%d host=%s",
            path, token["owner"], token["pid"], token["host"],
        )

    def close(self) -> None:
        """Zamyka klienta i zwalnia token właściciela, jeśli należy do tego procesu."""
        try:
            if self.owner_path is not None and self.owner_token is not None:
                current = read_vector_store_owner(self.owner_path.parent) or {}
                if current.get("pid") == os.getpid() and current.get("host") == self.owner_token.get("host"):
                    self.owner_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("Could not release ownership token %s: %s", self.owner_path, exc)
        if self.client is not None:
            try:
                self.client.close()
            except Exception as exc:
                logger.debug("Qdrant client close failed: %s", exc)
        self.owner_path = None
        self.owner_token = None

    # V72.4 (BUG-061): indeksy payloadu tworzone w schemacie kolekcji.
    #
    # Cytat z docs Qdrant (concepts/indexing): „Payload indexes should be created before
    # ingesting data. Qdrant's filterable HNSW index only benefits from additional
    # filter-aware edges when it is generated after the payload indexes have been
    # created. If you create a payload index after data has already been ingested, you
    # need to rebuild the HNSW index to take advantage of the new edges."
    # → każdy wpis bez indeksu poszerza dług; zakładamy je, póki kolekcja jest mała.
    #
    # Zakres = WYŁĄCZNIE pola, które realnie istnieją w payloadzie (insert_sync:
    # text/record/timestamp/confidence/metadata/upsert_key) I mają konsumenta filtra.
    # Pole zagnieżdżone idzie notacją kropkową (docs: „You can use dot notation to
    # specify a nested field for indexing").
    # Płaskie `origin` / `session_id` NIE występują w payloadzie — indeks na nich byłby
    # no-opem, a filtr pod nimi wiecznym zerem trafień (audyt #3).
    # `timestamp` to epoch float (models.py: `timestamp: float = Field(default_factory=time.time)`),
    # więc FLOAT, a nie DATETIME (DATETIME w Qdrancie = RFC 3339).
    PAYLOAD_INDEX_SPECS: tuple = (
        ("upsert_key", "KEYWORD"),
        ("timestamp", "FLOAT"),
        ("confidence", "FLOAT"),
        ("metadata.is_superseded", "BOOL"),
    )

    def _ensure_collection(self) -> None:
        if self.client is None:
            return
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.dimension,
                    distance=qmodels.Distance.COSINE,
                ),
                # V92: drugi (NAZWANY) wektor = leksykalny kanał wyszukiwania. IDF dokłada
                # silnik (Modifier.IDF), wiec nie utrzymujemy drugiej implementacji IDF
                # w Pythonie — patrz l2_semantic/sparse_encoder.py.
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(
                        index=qmodels.SparseIndexParams(on_disk=False),
                        modifier=qmodels.Modifier.IDF,
                    )
                },
            )
        self._ensure_payload_indexes()
        # V92: kolekcje tworzone przed V92 nie mają schematu sparse — sygnalizujemy RAZ,
        # bez cichego fallbacku do dense-only (cichy fallback udawał hybrydę).
        self._sparse_schema_notice()

    def _sparse_schema_ready(self) -> bool:
        """True, gdy kolekcja faktycznie ma nazwany wektor sparse (sonda, nie zgadywanie)."""
        if self.client is None:
            return False
        cached = getattr(self, "_sparse_schema_cached", None)
        if cached is not None:
            return cached
        try:
            info = self.client.get_collection(self.collection_name)
            cfg = getattr(getattr(info, "config", None), "params", None)
            ready = bool(getattr(cfg, "sparse_vectors", None))
        except Exception as exc:
            logger.debug("V92: sonda schematu sparse nieudana: %s", exc)
            ready = False
        self._sparse_schema_cached = ready
        return ready

    def _sparse_schema_notice(self) -> None:
        if getattr(self, "_sparse_notice_shown", False):
            return
        self._sparse_notice_shown = True
        if not self._sparse_schema_ready():
            logger.warning(
                "V92: kolekcja '%s' NIE ma wektora sparse — kanał hybrydowy będzie gęsto-single-modalny. "
                "Migracja (addytywna, bez rekodowania dense): scripts/migrate_qdrant_sparse.py --store <sciezka> --apply",                self.collection_name,
            )

    def _ensure_payload_indexes(self) -> None:
        """
        V72.4 (BUG-061): zakłada indeksy payloadu raz na instancję magazynu.

        Idempotentne: powtórne wywołanie dla istniejącego indeksu jest dla silnika
        no-opem/„already exists", a każdy błąd pojedynczego indeksu jest logowany
        i **nie** przerywa otwarcia kolekcji (brak indeksu to niższa wydajność filtra,
        nie utrata danych).
        """
        if self.client is None or getattr(self, "_payload_indexes_done", False):
            return
        if getattr(self, "_local_mode", False):
            # V90 (BUG-Q4): w trybie wbudowanym indeksy payloadu są NO-OPEM (biblioteka
            # sama ostrzega), a każde wywołanie to szum w logu startowym. Specyfikacja
            # PAYLOAD_INDEX_SPECS zostaje — po migracji na serwer działa bez zmian.
            logger.info(
                "V90/BUG-Q4: tryb lokalny Qdrant — indeksy payloadu pominięte (no-op). "
                "Specyfikacja (%d pól) czeka na tryb serwerowy.",
                len(self.PAYLOAD_INDEX_SPECS),
            )
            self._payload_indexes_done = True
            return
        for field_name, type_name in self.PAYLOAD_INDEX_SPECS:
            schema = getattr(qmodels.PayloadSchemaType, type_name, None)
            if schema is None:
                logger.debug("PayloadSchemaType.%s niedostępne — pomijam indeks %s", type_name, field_name)
                continue
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception as exc:
                logger.debug("create_payload_index(%s) nie powiodło się: %s", field_name, exc)
        self._payload_indexes_done = True

    def _encode_text(self, text: str) -> List[float]:
        """Jedno wejście tekstu do enkodera: najpierw normalizacja zapisu z bazy (V70.2).

        Produkcyjne ``state_variables.value`` jest podwójnie zakodowanym JSON-em, więc
        w tekście leżą LITERALNE sekwencje ``\\uXXXX`` (np. ``si\\u0119`` zamiast ``się``).
        Bez odwinicia model tokenizuje ucieczki zamiast polskich znaków — na prawdziwym
        PolDense-68M kosztuje to 3.3% cosine (0.9666 między wektorem znormalizowanego a
        surowego tekstu). Normalizacja jest idempotentna, więc podwójne wywołanie nie
        zmienia danych poprawnych.
        """
        return self.encoder.encode(normalize_stored_text(text))

    def _encode_query(self, text: str) -> List[float]:
        """Kodowanie zapytania: używa encode_query (z prefiksem query:) jeśli dostępne."""
        norm_text = normalize_stored_text(text)
        if hasattr(self.encoder, "encode_query"):
            return self.encoder.encode_query(norm_text)
        return self.encoder.encode(norm_text)

    def insert_sync(self, record: MemoryRecord, raw_text: Optional[str] = None, *, upsert_key: Optional[str] = None) -> str:
        """Synchronously inserts a memory record into Qdrant or in-memory fallback storage.

        ``upsert_key`` (V66, BUG-011): gdy podany, punkt ma DETERMINISTYCZNE ID
        (V90/BUG-Q2: blake2b-63bit; do V90 był to crc32-31bit), więc ponowny zapis
        tego samego klucza nadpisuje istniejący punkt zamiast tworzyć duplikat.
        Używane do rejestracji SOP-a w przestrzeni wektorowej przy bake — re-bake
        tej samej procedury nie mnoży wektorów.
        """
        text = raw_text or f"{record.effective_subject} {record.predicate} {record.object}"
        vec = record.vector if (record.vector and len(record.vector) == self.dimension) else self._encode_text(text)

        doc_id = f"vec_{int(time.time() * 1000)}_{abs(hash(text)) % 100000}"
        payload = {
            "text": text,
            # V90 (BUG-Q1): wektor NIE wchodzi do payloadu. Daemon ustawia record.vector
            # przed zapisem (atlas_daemon.py:1321), więc model_dump() wkładał 512
            # floatów DRUGI raz — audyt 2026-09-18: 4 z 271 punktów nosiło 45 403
            # z 47 908 B wszystkich payloadów (94,8%). Wektor jest w polu `vector`
            # punktu, a żaden konsument nie czyta payload["record"]["vector"].
            "record": record.model_dump(exclude={"vector"}),
            "timestamp": record.timestamp,
            "confidence": record.confidence,
            "metadata": record.metadata,
            "upsert_key": normalize_upsert_key(upsert_key) if upsert_key is not None else None,
        }
        # V91 (BUG-S4): kanał gęsty był martwy w produkcji, bo odczyt nie miał z czego
        # odzyskać klucza KV. `candidates.py` czytał wyłącznie `meta["kv_key"]`, a write_path
        # i scripts/rebuild_qdrant_68m.py nigdy go nie ustawiały — dla 143/295 punktów
        # trafienie gęste lądowało pod `record.subject` (sam tytuć, np. 'Crawl4AI LLM default'
        # zamiast 'hermes:memory_md:Crawl4AI LLM default'), weryfikacja w KV je odrzucała
        # i fuzja degenerowała do leksykalnej (log: 0/43 linii RetrievalQuality z dense>0).
        # Kanoniczny klucz wpisujemy do payloadu przy każdym zapisie; punkty sprzed naprawy
        # są ratowane fallbackiem na `upsert_key` w `search()` (ma go 272/295).
        if upsert_key:
            payload["kv_key"] = normalize_upsert_key(upsert_key)

        if self.client is not None:
            if upsert_key is not None:
                canon_uk = normalize_upsert_key(upsert_key)
                point_id = _point_id_for_key(canon_uk)
                # V90 (BUG-Q2): punkt ze STARYM schematem (crc32) o tym samym kluczu
                # kasujemy — inaczej zmiana schematu ID tworzyłaby duplikat (dwa punkty
                # dla jednego rekordu KV), a delete-by-key zostałby z zombie.
                self._drop_legacy_point_id(upsert_key, point_id)
                self._drop_legacy_point_id(canon_uk, point_id)
            else:
                point_id = (int(time.time() * 1_000_000) ^ (zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF)) % 0x7FFFFFFF
            # V92: wektor zapisujemy jako słownik {"": dense, "sparse": ...}, gdy kolekcja
            # ma drugi (nazwany) kanał. Klucz "" to unnamed dense — zachowuje zgodność
            # z istniejącymi punktami i ID (probe: qdrant-client 1.19.0 local mode).
            point_vector: Any = vec
            if self._sparse_schema_ready():
                point_vector = {
                    "": vec,
                    SPARSE_VECTOR_NAME: encode_sparse_document(text),
                }
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    qmodels.PointStruct(
                        id=point_id,
                        vector=point_vector,
                        payload=payload,
                    )
                ],
            )
            # V71: zwracamy REALNE id punktu. Wcześniej przy `upsert_key` (id deterministyczne
            # z crc32) funkcja zwracała `doc_id` ze znacznikiem czasu, którego w kolekcji nie ma
            # — czyli ID-widmo. Bez `upsert_key` `doc_id` też nie istniał w Qdrant (punkt miał
            # id z XOR), więc zwrot realnego id jest poprawny w obu przypadkach.
            return str(point_id)
        else:
            if upsert_key is not None:
                canon_uk = normalize_upsert_key(upsert_key)
                target_keys = {canon_uk, f"{KV_UPSERT_PREFIX}{canon_uk}", str(upsert_key)}
                # In-memory dedupe: nadpisz istniejący wpis o tym samym upsert_key.
                for idx, existing in enumerate(self._records):
                    ex_uk = existing.get("upsert_key")
                    if ex_uk in target_keys or (ex_uk and normalize_upsert_key(ex_uk) == canon_uk):
                        self._records[idx] = {"id": existing.get("id", doc_id), **payload}
                        self._vectors[idx] = np.array(vec, dtype=np.float64)
                        return str(existing.get("id", doc_id))
            self._records.append({"id": doc_id, **payload})
            self._vectors.append(np.array(vec, dtype=np.float64))

        return doc_id

    async def insert(self, record: MemoryRecord, raw_text: Optional[str] = None, *, upsert_key: Optional[str] = None) -> str:
        """Asynchronously inserts a memory record under lock."""
        async with self._lock:
            return self.insert_sync(record, raw_text=raw_text, upsert_key=upsert_key)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.1,
        query_filter: Optional[Any] = None,
        hybrid: bool = True,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """`hybrid=False` wymusza czysty kanał gęsty — to ramię A pomiaru A/B (V92)."""
        if limit is not None:
            top_k = limit
        """
        Wyszukuje semantycznie pasujące rekordy pamięci.

        V70.1: lokalny pre-filtr MIB+Hamming usunięty z tej ścieżki (pomiar: p50 31.88 ms
        vs 5.12 ms bez niego, tie-tolerant recall@10 = 1.0 → identyczna dokładność za
        ~6x większy koszt). Decyzja i warunki powrotu: docs/baselines/qdrant_prefilter_owner.json
        (`revisit_when`). Sygnatura NIE przyjmuje już tego parametru, więc przypadkowe
        użycie kończy się natywnym TypeError Pythona, a nie wpisaną tu listą zakazów.
        """
        q_vec = self._encode_query(query)

        async with self._lock:
            if self.client is not None:
                search_res = []
                try:
                    # V72.4 (FINDING-063): usunięta martwa gałąź sprzed 2 linii niżej —
                    # dawne `hasattr` na metodzie `search` klienta.
                    # qdrant-client 1.19.0 NIE MA metody `.search` (sonda: hasattr -> False),
                    # więc ta gałąź nigdy nie dochodziła do głosu — Universal Query API
                    # (`query_points`) jest jedyną żywą ścieżką, a martwy kod udawał
                    # istnienie fallbacku.
                    #
                    # V72.4 (BUG-062): `query_filter` trafia do silnika. Wcześniejszy stan:
                    # żadna filtracja nie szła do Qdranta, wszystko działo się w Pythonie po
                    # pobraniu top-k (docs concepts/points: „we recommend server-side
                    # filtering”, bo filtrowanie klienta działa post-filterem i zwraca
                    # MNIEJ wyników niż żądano, nawet gdy w kolekcji są pasujące punkty).
                    # V92: kanał hybrydowy. Gdy kolekcja ma schemat sparse, zapytanie idzie
                    # jako DBSF-fuzja prefetch dense + prefetch sparse (server-side, Modifier.IDF).
                    # Gdy NIE ma (kolekcja przed migracją), zostaje dense-only — jawnie zalogowane
                    # raz per instancja, żeby hybryda nigdy nie udawała, że działa.
                    if hybrid and self._sparse_schema_ready() and hasattr(self.client, "query_points"):
                        try:
                            qp = self.client.query_points(
                                collection_name=self.collection_name,
                                prefetch=[
                                    qmodels.Prefetch(
                                        query=q_vec,
                                        using="",
                                        limit=max(top_k * 4, 24),
                                        filter=query_filter,
                                    ),
                                    qmodels.Prefetch(
                                        query=encode_sparse_query(query),
                                        using=SPARSE_VECTOR_NAME,
                                        limit=max(top_k * 4, 24),
                                        filter=query_filter,
                                    ),
                                ],
                                query=qmodels.FusionQuery(fusion=qmodels.Fusion.DBSF),
                                limit=top_k,
                                query_filter=query_filter,
                            )
                            search_res = getattr(qp, "points", [])
                        except Exception as exc:
                            logger.warning("V92: kanał hybrydowy nieudany, fallback dense: %s", exc)
                            qp = self.client.query_points(
                                collection_name=self.collection_name,
                                query=q_vec,
                                limit=top_k,
                                score_threshold=min_similarity,
                                query_filter=query_filter,
                            )
                            search_res = getattr(qp, "points", [])
                    elif hasattr(self.client, "query_points"):
                        qp = self.client.query_points(
                            collection_name=self.collection_name,
                            query=q_vec,
                            limit=top_k,
                            score_threshold=min_similarity,
                            query_filter=query_filter,
                        )
                        search_res = getattr(qp, "points", [])
                except Exception as exc:
                    logger.debug("query_points nie powiodło się: %s", exc)
                    search_res = []

                results = []
                for hit in search_res:
                    payload = hit.payload or {}
                    results.append({
                        "id": str(hit.id),
                        "text": payload.get("text", ""),
                        "record": payload.get("record", {}),
                        "score": float(hit.score),
                        "confidence": payload.get("confidence", 1.0),
                        "timestamp": payload.get("timestamp", 0.0),
                        # V91 (BUG-S4): `upsert_key`/`kv_key` MUSZĄ wyjść do odczytu.
                        # Wcześniej ten słownik je gubił, więc `candidates.py` nie miał jak
                        # odtworzyć klucza KV i odrzucał każde trafienie gęste.
                        "upsert_key": payload.get("upsert_key"),
                        "kv_key": payload.get("kv_key"),
                    })
                return results
            else:
                if not self._vectors:
                    return []

                candidate_indices = np.arange(len(self._vectors))
                matrix = np.stack(self._vectors)

                qv = np.array(q_vec, dtype=np.float64)
                qv_norm = np.linalg.norm(qv) + 1e-9
                m_norm = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
                sims = (matrix @ qv) / (m_norm.flatten() * qv_norm)

                results = []
                sub_indices = np.argsort(-sims)
                for s_idx in sub_indices:
                    score = float(sims[s_idx])
                    if score < min_similarity:
                        continue
                    real_idx = candidate_indices[s_idx]
                    entry = self._records[real_idx]
                    results.append({
                        "id": entry["id"],
                        "text": entry["text"],
                        "record": entry["record"],
                        "score": score,
                        "confidence": entry["confidence"],
                        "timestamp": entry["timestamp"],
                    })
                    if len(results) >= top_k:
                        break
                return results

    def _hits_to_dicts(self, search_res: List[Any]) -> List[Dict[str, Any]]:
        """Mapowanie trafień silnika na słowniki kandydatów (wspólne dla dense i hybrydy)."""
        out: List[Dict[str, Any]] = []
        for hit in search_res:
            payload = hit.payload or {}
            out.append({
                "id": str(hit.id),
                "text": payload.get("text", ""),
                "record": payload.get("record", {}),
                "score": float(hit.score),
                "confidence": payload.get("confidence", 1.0),
                "timestamp": payload.get("timestamp", 0.0),
                "upsert_key": payload.get("upsert_key"),
                "kv_key": payload.get("kv_key"),
            })
        return out

    @staticmethod
    def _build_session_filter(session_id: Optional[str]) -> Optional[Any]:
        """Filtr izolacji sesji dla zapytania hybrydowego (jedno źródło prawdy).

        Włącza się wyłącznie dla nazwanej sesji prywatnej — klasy domyślne/globalne
        (`""`, `default`, `hermes_default`, `global`) obsługuje post-filtr
        `is_session_accessible`. Punkty BEZ `metadata.session_id` (sesja wynika z
        prefiksu klucza, np. `hermes:`/`fact:`) MUSZĄ przechodzić: w Qdrant
        `IsNullCondition` NIE dopasowuje brakującego pola (zweryfikowane empirycznie),
        stąd dodatkowo `IsEmptyCondition`.
        """
        sid = str(session_id or "").strip()
        if sid in ("", "default", "global", HERMES_GLOBAL_SESSION, HERMES_DEFAULT_SESSION):
            return None
        clean_sess = sid.replace("session_", "").strip()
        field = "metadata.session_id"
        try:
            return qmodels.Filter(
                should=[
                    qmodels.FieldCondition(key=field, match=qmodels.MatchAny(any=[
                        clean_sess, f"session_{clean_sess}", "global", "hermes_persistent",
                    ])),
                    qmodels.IsNullCondition(is_null=qmodels.PayloadField(key=field)),
                    qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key=field)),
                ]
            )
        except Exception as f_err:
            logger.warning("Qdrant session filter build failed (isolation falls back to post-filter): %s", f_err)
            return None

    async def search_hybrid_async(
        self,
        query: str,
        limit: int = 10,
        session_id: Optional[str] = None,
        query_filter: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """V92: OŻYWIONA gałąź hybrydowa — gęsty + leksykalny (sparse) kanał w jednym zapytaniu.

        Przed V92 ta nazwa istniała wyłącznie w martwej gałęzi `candidates.py` (wywoływana,
        gdy magazyn NIE miał `search`, a QdrantVectorStore miał) — czyli hybrydowy retrieval
        był deklaratywny, nigdy niewykonany. Teraz to jest PIERWSZE ŹRÓDŁO kandydatów:
        `query_points(prefetch=[dense, sparse], fusion=DBSF)` po stronie silnika, plus
        osobny przebieg sparse-only do telemetrii (które kandydaty weszły WYŁĄCZNIE
        kanałem leksykalnym — bez tego nie da się udowodnić, że sparse coś dodał).

        Fuzja: DBSF (nie RRF). Ocena końcowa i bramki autorytetu pozostają w
        `cognitive/retrieval/score.py` (Static-Anchor DBSF).
        """
        # UWAGA: bez `self._lock` w tej gałęzi — `search()` sam bierze ten sam zamek,
        # a asyncio.Lock NIE jest reentrantny (poprzednia wersja robiła z tego deadlock).
        if self.client is None or not self._sparse_schema_ready():
            self.last_hybrid_stats = {
                "fused": 0, "sparse_branch_hits": 0, "dense_branch_hits": 0,
                "sparse_only": 0, "query_terms": 0, "fallback": "dense",
            }
            return await self.search(query, top_k=limit, query_filter=query_filter)

        eff_filter = query_filter if query_filter is not None else self._build_session_filter(session_id)

        q_vec = self._encode_query(query)
        q_sparse = encode_sparse_query(query)
        branch_limit = max(limit * 4, 24)
        self.last_hybrid_stats = {
            "fused": 0,
            "sparse_branch_hits": 0,
            "dense_branch_hits": 0,
            "sparse_only": 0,
            "query_terms": len(q_sparse.indices or []),
        }

        async with self._lock:
            try:
                fused = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        qmodels.Prefetch(query=q_vec, using="", limit=branch_limit, filter=eff_filter),
                        qmodels.Prefetch(
                            query=q_sparse, using=SPARSE_VECTOR_NAME, limit=branch_limit, filter=eff_filter
                        ),
                    ],
                    query=qmodels.FusionQuery(fusion=qmodels.Fusion.DBSF),
                    limit=limit,
                    query_filter=eff_filter,
                )
            except Exception as exc:
                # Fallback POZA zamkiem (niżej) — nie wołamy search() trzymając lock.
                logger.warning("V92: search_hybrid_async DBSF nieudane, fallback dense: %s", exc)
                self.last_hybrid_stats["fallback"] = "dense"
                fused = None

            hits = getattr(fused, "points", []) if fused is not None else []

            # Telemetria i surowe wyniki per-kanał (Single Source of Truth)
            via_sparse: set = set()
            via_dense: set = set()
            dense_scores: dict[str, float] = {}
            sparse_scores: dict[str, float] = {}
            try:
                s_only = self.client.query_points(
                    collection_name=self.collection_name,
                    query=q_sparse, using=SPARSE_VECTOR_NAME,
                    limit=branch_limit, query_filter=eff_filter,
                )
                for p in getattr(s_only, "points", []):
                    pid = str(p.id)
                    via_sparse.add(pid)
                    sparse_scores[pid] = float(getattr(p, "score", 0.0) or 0.0)

                d_only = self.client.query_points(
                    collection_name=self.collection_name,
                    query=q_vec, using="",
                    limit=branch_limit, query_filter=eff_filter,
                )
                for p in getattr(d_only, "points", []):
                    pid = str(p.id)
                    via_dense.add(pid)
                    dense_scores[pid] = float(getattr(p, "score", 0.0) or 0.0)
            except Exception as exc:
                logger.debug("V92: telemetria kanałów niedostępna: %s", exc)

            results = self._hits_to_dicts(hits)
            for row in results:
                rid = str(row.get("id", ""))
                row["via_sparse"] = rid in via_sparse
                row["via_dense"] = rid in via_dense
                row["fused_score"] = float(row.get("score", 0.0))
                row["dense_score"] = float(dense_scores.get(rid, 0.0))
                row["sparse_score"] = float(sparse_scores.get(rid, 0.0))
                # Canonical: score kandydata to surowy cosinus gęsty.
                # Jeśli punkt wszedł wyłącznie sparse, dense cosine to 0.0.
                row["score"] = row["dense_score"]
            self.last_hybrid_stats.update({
                "fused": len(results),
                "sparse_branch_hits": len(via_sparse),
                "dense_branch_hits": len(via_dense),
                "sparse_only": len([r for r in results if r.get("via_sparse") and not r.get("via_dense")]),
            })
            if fused is None:
                self.last_hybrid_stats["fallback"] = "dense"
            else:
                return results

        # Jedyna droga fallbacku: fuzja DBSF padła -> klasyczny dense przez `search()`.
        return await self.search(query, top_k=limit, query_filter=query_filter)

    async def count(self) -> int:
        async with self._lock:
            if self.client is not None:
                info = self.client.get_collection(self.collection_name)
                return info.points_count or 0
            return len(self._records)

    # ------------------------------------------------------------------------------
    # V72.2 (W3): kasowanie punktów.
    #
    # Do V72.2 magazyn NIE MIAŁ żadnej metody delete: ``delete`` rekordu KV kasował
    # wiersz w SQLite, a punkt wektorowy zostawał — ``atlas_recall`` potrafił więc
    # zwrócić fakt oficjalnie usunięty (udokumentowane na archiwalnej sondzie
    # logic-kv: dowód w historii gita, bez żywej referencji do wersji; ADR-017).
    # Kasowanie idzie po wartości pola payload ``upsert_key`` (format produkcyjny to
    # ``kv:<klucz>`` — patrz scripts/rebuild_qdrant_68m.py), a nie po ID, bo punkty
    # sprzed V66 mają ID losowe i nie da się ich odzyskać z klucza.
    # ------------------------------------------------------------------------------

    @staticmethod
    def point_id_for_upsert_key(upsert_key: str) -> int:
        """Deterministyczne ID punktu — deleguje do ``_point_id_for_key`` (jedno źródło, V90)."""
        return _point_id_for_key(upsert_key)

    def _drop_legacy_point_id(self, upsert_key: str, new_point_id: int) -> int:
        """V90 (BUG-Q2): kasuje punkt zapisany STARYM schematem (crc32) pod tym samym kluczem.

        Kasujemy wyłącznie po DOWODZIE, że pod legacy ID leży ten sam ``upsert_key``
        (odczyt payloadu), więc cudze punkty nie są zagrożone. Zwraca 1 gdy usunięto.
        """
        from qdrant_client.http import models as qmodels

        legacy_id = zlib.crc32(upsert_key.encode("utf-8")) & _LEGACY_ID_MASK
        if legacy_id == new_point_id:
            return 0
        try:
            probe = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[legacy_id],
                with_payload=[UPSERT_KEY_PAYLOAD_FIELD],
                with_vectors=False,
            )
        except Exception as exc:
            logger.debug("probe legacy ID %s nie powiodło się: %s", legacy_id, exc)
            return 0
        if not probe or str((probe[0].payload or {}).get(UPSERT_KEY_PAYLOAD_FIELD) or "") != upsert_key:
            return 0
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.PointIdsList(points=[legacy_id]),
            wait=True,
        )
        logger.info("V90/BUG-Q2: usunięto punkt legacy %s (klucz zapisany pod %s)", legacy_id, new_point_id)
        return 1

    def _scroll_all_matching_ids(
        self,
        *,
        scroll_filter: Optional[Any] = None,
        payload_keys: Optional[List[str]] = None,
        predicate: Optional[Any] = None,
    ) -> List[int]:
        """V90 (BUG-Q3): ID WSZYSTKICH pasujących punktów — strona po stronie, do wyczerpania.

        Koniec rozpoznajemy po ``offset is None`` / pustej stronie (semantyka scrolla
        Qdranta), a nie po sztywnym sufitcie stron. Przekroczenie bezpiecznika podnosi
        ``RuntimeError``: ciche obcięcie skanu zaniżało licznik pokrycia (H6) i zostawiało
        resztę punktów przy delete — defekt klasy BUG-Q3.
        """
        ids: List[int] = []
        offset: Any = None
        pages = 0
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=_PREFIX_SCAN_PAGE_SIZE,
                offset=offset,
                with_payload=list(payload_keys) if payload_keys else False,
                with_vectors=False,
            )
            for point in points:
                if predicate is None or predicate(point):
                    ids.append(point.id)
            pages += 1
            if not points or offset is None:
                return ids
            if pages >= _POINT_ID_SAFETY_PAGES:
                raise RuntimeError(
                    f"Skan kolekcji {self.collection_name} przekroczył bezpiecznik "
                    f"{_POINT_ID_SAFETY_PAGES} stron x {_PREFIX_SCAN_PAGE_SIZE} punktów. "
                    "Podnieś bezpiecznik albo filtruj po indeksowanym polu — ciche "
                    "obcięcie skanu jest defektem, nie optymalizacją (V90/BUG-Q3)."
                )

    @staticmethod
    def _upsert_key_filter(upsert_key: str) -> "qmodels.Filter":
        from qdrant_client.http import models as qmodels
        canon = normalize_upsert_key(upsert_key)
        targets = [canon, f"{KV_UPSERT_PREFIX}{canon}"] if canon else [upsert_key]
        should_conds: List[Any] = [
            qmodels.FieldCondition(key=UPSERT_KEY_PAYLOAD_FIELD, match=qmodels.MatchValue(value=t))
            for t in targets
        ]
        if canon:
            should_conds.append(qmodels.FieldCondition(key="kv_key", match=qmodels.MatchValue(value=canon)))
        return qmodels.Filter(should=should_conds)

    def migrate_legacy_point_ids_sync(self, *, batch: int = 64) -> Dict[str, int]:
        """V90 (BUG-Q2): przepisuje punkty z upsert_key na schemat ID blake2b.

        Dwufazowo, żeby nie dało się zapętlić na jednej stronie:
          faza 1 — PEŁNY skan (predykat: punkt ma klucz i ID != nowy schemat),
          faza 2 — batchowy retrieve z wektorami, re-upsert pod nowym ID, kasacja starego.
        Pętla powtarza się, aż faza 1 zwróci pustkę — punkt przepisany przestaje być
        stale, więc każdy przebieg maleje (idempotentność). Wymaga ZATRZYMANEGO daemona
        (drugi pisarz na katalogu = ryzyko korupcji, V70.1).
        """
        from qdrant_client.http import models as qmodels

        moved = 0
        rounds = 0
        if self.client is None:
            return {"moved": 0, "rounds": 0}

        def is_stale(point) -> bool:
            key = str((point.payload or {}).get(UPSERT_KEY_PAYLOAD_FIELD) or "")
            return bool(key) and point.id != _point_id_for_key(key)

        while True:
            rounds += 1
            if rounds > 50:
                raise RuntimeError(f"migrate_legacy_point_ids_sync: >{rounds} rund — zbiór maleć nie chce")
            stale_ids = self._scroll_all_matching_ids(
                payload_keys=[UPSERT_KEY_PAYLOAD_FIELD],
                predicate=is_stale,
            )
            if not stale_ids:
                return {"moved": moved, "rounds": rounds}
            for i in range(0, len(stale_ids), batch):
                chunk = stale_ids[i:i + batch]
                for point in self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=chunk,
                    with_payload=True,
                    with_vectors=True,
                ):
                    key = str((point.payload or {}).get(UPSERT_KEY_PAYLOAD_FIELD) or "")
                    if not key:
                        continue
                    vec = point.vector
                    if isinstance(vec, dict):  # named vectors: bierzemy pierwszą gałąź
                        vec = next(iter(vec.values()))
                    new_id = _point_id_for_key(key)
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=[qmodels.PointStruct(id=new_id, vector=vec, payload=dict(point.payload or {}))],
                        wait=True,
                    )
                    if point.id != new_id:
                        self.client.delete(
                            collection_name=self.collection_name,
                            points_selector=qmodels.PointIdsList(points=[point.id]),
                            wait=True,
                        )
                    moved += 1
                    logger.info("V90/BUG-Q2 sweep: %s -> %s (klucz %s)", point.id, new_id, key)

    def count_by_upsert_key_sync(self, upsert_key: str) -> int:
        """Liczba punktów o danym ``upsert_key`` (górny limit skanu: 1000)."""
        canon_key = normalize_upsert_key(upsert_key)
        targets = {canon_key, f"{KV_UPSERT_PREFIX}{canon_key}", str(upsert_key)}
        if self.client is None:
            return sum(
                1 for rec in self._records
                if rec.get(UPSERT_KEY_PAYLOAD_FIELD) in targets
                or (rec.get(UPSERT_KEY_PAYLOAD_FIELD) and normalize_upsert_key(rec.get(UPSERT_KEY_PAYLOAD_FIELD)) == canon_key)
            )
        count = 0
        offset: Any = None
        for _ in range(100):
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._upsert_key_filter(canon_key),
                limit=10,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            count += len(points)
            if not points or offset is None:
                break
        return count

    def delete_by_upsert_key_sync(self, upsert_key: Optional[str]) -> int:
        """
        W3 (V72.2): kasuje WSZYSTKIE punkty o podanym ``upsert_key``. Zwraca liczbę skasowanych.

        Semantyka: 0 gdy nic nie pasowało (bez wyjątku — delete rekordu niezaindeksowanego
        to normalna ścieżka), punkty bez klucza (``upsert_key`` = NULL, ścieżka SOP-bake)
        nigdy nie są trafiane. Ścieżka Qdrant ID-widmo (gdy filtr zwrócił zero, a punkt
        istnieje pod deterministycznym ID) też jest obsługiwana.
        """
        if not upsert_key:
            return 0
        raw_key = str(upsert_key)
        canon_key = normalize_upsert_key(raw_key)
        target_keys = {canon_key, f"{KV_UPSERT_PREFIX}{canon_key}", raw_key}

        if self.client is None:
            kept_records: List[Dict[str, Any]] = []
            kept_vectors: List[np.ndarray] = []
            removed = 0
            # strict=False: rekordy i wektory idą parami; gdyby historyczny stan był
            # rozjechany, kasowanie ma zostać ciche, nie podnieść wyjątku w ścieżce delete.
            for rec, vec in zip(self._records, self._vectors, strict=False):
                rec_uk = rec.get(UPSERT_KEY_PAYLOAD_FIELD)
                rec_canon = normalize_upsert_key(rec_uk) if rec_uk else ""
                if rec_uk in target_keys or (rec_canon and rec_canon == canon_key):
                    removed += 1
                    continue
                kept_records.append(rec)
                kept_vectors.append(vec)
            if removed:
                self._records = kept_records
                self._vectors = kept_vectors
            return removed

        try:
            from qdrant_client.http import models as qmodels  # lokalny import: qmodels bywa niezaimportowany
            # V72.4 (BUG-061): kasowanie PREDYKATEM SERWERA (FilterSelector) zamiast
            # scroll -> PointIdsList. Docs Qdrant (concepts/points) pokazują dokładnie ten
            # wzorzec: client.delete(points_selector=FilterSelector(filter=...)). Zysk:
            # (a) jedno wywołanie zamiast dwóch round-tripów (odczyt ID, potem kasowanie),
            # (b) brak okna wyścigu między odczytem a kasowaniem (punkt dopisany w tym
            # oknie zostawał zombie — defekt klasy BUG-057), (c) znika zgadywanie
            # paginacji: silnik dobiera strony sam.
            # `wait=True` utrzymuje determinizm licznika (W3 liczy zwrot, W5/H6 porównują
            # stan natychmiast po kasowaniu).
            before = self.count_by_upsert_key_sync(canon_key)
            if before:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=qmodels.FilterSelector(filter=self._upsert_key_filter(canon_key)),
                    wait=True,
                )
                # Odczyt ZWROTNY po kasowaniu: licznik nie może kłamać (lekcja V72.3 —
                # `gc_pruned` rósł na nieistniejącym kluczu). Gdy silnik nie zdążył
                # (wait=False w innym wariancie), zwracamy realny ubytek, nie zamiar.
                after = self.count_by_upsert_key_sync(canon_key)
                return max(0, before - after)
            # Fallback: punkt ID-widmo pod deterministycznym ID, gdy filtr nie zna klucza
            # (punkt zapisany przed wprowadzeniem indeksu payloadu).
            probe_ids = list({
                self.point_id_for_upsert_key(canon_key),
                self.point_id_for_upsert_key(f"{KV_UPSERT_PREFIX}{canon_key}"),
                self.point_id_for_upsert_key(raw_key),
            })
            probe = self.client.retrieve(
                collection_name=self.collection_name,
                ids=probe_ids,
                with_payload=False,
                with_vectors=False,
            )
            ghost_ids = [p.id for p in probe]
            if not ghost_ids:
                return 0
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.PointIdsList(points=ghost_ids),
                wait=True,
            )
            return len(ghost_ids)
        except Exception as exc:  # pragma: no cover - awaria silnika nie może wywalić delete
            logger.debug("delete_by_upsert_key (Qdrant) nie powiodło się dla %s: %s", raw_key, exc)
            return 0

    def get_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Zwraca rekord po kluczu lub upsert_key (kanonicznym lub z prefiksem kv:)."""
        if not key:
            return None
        canon_key = normalize_upsert_key(key)
        targets = {key, canon_key, f"{KV_UPSERT_PREFIX}{canon_key}"}
        if self.client is None:
            for rec in self._records:
                uk = rec.get(UPSERT_KEY_PAYLOAD_FIELD)
                rec_canon = normalize_upsert_key(uk) if uk else ""
                if uk in targets or (rec_canon and rec_canon == canon_key) or rec.get("key") in targets:
                    return rec
            return None
        try:
            pts = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[self.point_id_for_upsert_key(canon_key)],
                with_payload=True,
            )
            if pts and pts[0].payload:
                return dict(pts[0].payload)
            res, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._upsert_key_filter(canon_key),
                limit=1,
                with_payload=True,
            )
            if res and res[0].payload:
                return dict(res[0].payload)
            return None
        except Exception:
            return None

    async def delete_by_upsert_key(self, upsert_key: Optional[str]) -> int:
        """W3 (V72.2): asynchroniczny wrapper kasowania pod blokadą magazynu."""
        async with self._lock:
            return self.delete_by_upsert_key_sync(upsert_key)

    def count_with_upsert_key_prefix_sync(self, prefix: str = KV_UPSERT_PREFIX) -> int:
        """
        W5 (V72.2 / H6): ile PUNKTÓW ma ``upsert_key`` zaczynający się od ``prefix``.

        Liczy punkty, nie unikalne klucze — żeby mnożenie warstwy wektorowej (historyczny
        defekt: 9 969 punktów przy 511 tekstach) podniosło licznik tak samo jak ciche
        niezaindeksowanie go obniżyło. Skan jest paginowany z twardym limitem.
        """
        pref = str(prefix)
        if self.client is None:
            return sum(1 for rec in self._records if str(rec.get(UPSERT_KEY_PAYLOAD_FIELD) or "").startswith(pref))
        # V90 (BUG-Q3): dawniej `for _ in range(_PREFIX_SCAN_MAX_PAGES)` = sufit
        # 400x256 = 102 400 punktow i CICHE przerwanie -> licznik zaniżony.
        return len(self._scroll_all_matching_ids(
            payload_keys=[UPSERT_KEY_PAYLOAD_FIELD],
            predicate=lambda point: str((point.payload or {}).get(UPSERT_KEY_PAYLOAD_FIELD) or "").startswith(pref),
        ))

    async def count_with_upsert_key_prefix(self, prefix: str = KV_UPSERT_PREFIX) -> int:
        async with self._lock:
            return self.count_with_upsert_key_prefix_sync(prefix)

    def count_indexed_kv_points_sync(self) -> int:
        """
        V91 (BUG-S4): ile punktów ma KANONICZNY klucz KV w payloadzie (``kv_key``).

        Poprzedni licznik daemona (``count_with_upsert_key_prefix_sync("kv:")``) liczył
        tylko punkty z prefiksem ``kv:`` — produkcyjnie format jest MIESZANY (część
        writerów zapisuje ``upsert_key`` bez prefiksu), więc metryka pokazywała 152/213
        i ukrywała, że 61 rekordów KV nie ma wektora wcale. Tu liczymy po polu ``kv_key``,
        które po V91 ustawia każdy zapis, a punkty historyczne dostają je w reindeksie.
        """
        if self.client is None:
            return sum(1 for rec in self._records if rec.get("kv_key"))
        # Predicate po stronie skrolla (ten sam mechanizm co V90/BUG-Q3): liczymy punkty,
        # ktore REALNIE maja kanoniczny klucz, nie te z dowolnym upsert_key.
        ids = self._scroll_all_matching_ids(
            payload_keys=["kv_key"],
            predicate=lambda point: bool((getattr(point, "payload", None) or {}).get("kv_key")),
        )
        return len(ids)

    def delete_by_upsert_key_prefix_sync(self, prefix: str) -> int:
        """V72.7 (FINDING-075): Kasuje wszystkie punkty, których ``upsert_key`` zaczyna się od ``prefix``.

        Zwraca dokładną liczbę usuniętych wektorów.
        Ochrona przed przypadkowym usunięciem kluczy kanonicznych:
        Jeśli prefiks zaczyna się od 'hermes:' lub 'kv:hermes:', rzuca ValueError.
        """
        if not prefix:
            return 0
        pref = str(prefix)
        if pref.startswith("hermes:") or pref.startswith("kv:hermes:"):
            raise ValueError(f"Prefix matching hermes:* is protected from deletion: {pref}")
        if len(pref.strip()) < 5:
            raise ValueError(f"Prefix too short for deletion: {pref}")

        if self.client is None:
            kept_records: List[Dict[str, Any]] = []
            kept_vectors: List[np.ndarray] = []
            removed = 0
            for rec, vec in zip(self._records, self._vectors, strict=False):
                upsert_key = str(rec.get(UPSERT_KEY_PAYLOAD_FIELD) or "")
                if upsert_key.startswith(pref):
                    removed += 1
                    continue
                kept_records.append(rec)
                kept_vectors.append(vec)
            if removed:
                self._records = kept_records
                self._vectors = kept_vectors
            return removed

        try:
            from qdrant_client.http import models as qmodels

            # V90 (BUG-Q3): stronicowanie do wyczerpania (był sufit 400x256 i ciche
            # przerwanie — reszta prefiksu zostawała w kolekcji).
            matching_ids = self._scroll_all_matching_ids(
                payload_keys=[UPSERT_KEY_PAYLOAD_FIELD],
                predicate=lambda point: str((point.payload or {}).get(UPSERT_KEY_PAYLOAD_FIELD) or "").startswith(pref),
            )

            if not matching_ids:
                return 0

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.PointIdsList(points=matching_ids),
                wait=True,
            )
            return len(matching_ids)
        except Exception as exc:
            logger.error("delete_by_upsert_key_prefix_sync failed for prefix %s: %s", pref, exc)
            raise

    async def delete_by_upsert_key_prefix(self, prefix: str) -> int:
        async with self._lock:
            return self.delete_by_upsert_key_prefix_sync(prefix)

    def delete_by_text_sync(self, text: str) -> int:
        """
        W3 (V72.2): kasowanie po treści punktu, dla rekordów indeksowanych BEZ ``upsert_key``.

        Konieczne, bo ścieżka KV daemona zapisywała punkty bez klucza deterministycznego
        (ingest bez upsert_key), a ``atlas_recall`` przeszukuje kolekcję szeroko — bez
        tego delete zostawiał punkt. Match pełnej equality pola payload ``text``.
        """
        if not text:
            return 0
        needle = str(text)
        if self.client is None:
            kept_records: List[Dict[str, Any]] = []
            kept_vectors: List[np.ndarray] = []
            removed = 0
            for rec, vec in zip(self._records, self._vectors, strict=False):
                if rec.get("text") == needle:
                    removed += 1
                    continue
                kept_records.append(rec)
                kept_vectors.append(vec)
            if removed:
                self._records = kept_records
                self._vectors = kept_vectors
            return removed
        try:
            # V90 (BUG-Q3): dawniej limit=100 i ZWROT len(ids) jako "usunięte" —
            # przy >100 duplikatów tekstu reszta zostawała. Teraz wszystkie strony.
            ids = self._scroll_all_matching_ids(
                scroll_filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="text", match=qmodels.MatchValue(value=needle))]
                )
            )
            if not ids:
                return 0
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.PointIdsList(points=ids),
                wait=True,
            )
            return len(ids)
        except Exception as exc:  # pragma: no cover
            logger.debug("delete_by_text (Qdrant) nie powiodło się: %s", exc)
            return 0

