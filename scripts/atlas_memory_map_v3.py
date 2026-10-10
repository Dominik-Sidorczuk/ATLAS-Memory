#!/usr/bin/env python3
"""ATLAS Memory Map v3 — Reprezentatywna, interaktywna mapa semantyczna ATLAS.

Poprawki względem wersji bazowej v3:
  1. Dobór klastrów k: zastąpiono monotoniczną atrapę obiektywnym kryterium
     rozproszenia wariancji Calinskiego-Harabasza (czyste NumPy, bez sklearn).
  2. Integracja z backendem ATLAS: używa produkcyjnego enkodera (PolDense-68M 512D)
     zgodnie z ~/.hermes/atlas/vector_backend.json.
  3. Bezpieczne parsowanie is_superseded: poprawna obsługa stringów "false"/"0".
  4. Pełna zgodność z kontraktem testowym (zachowano row_perplexity, aliases).
  5. Odporny odczyt SQLite z WAL bez blokad VACUUM.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

DB = Path.home() / ".hermes" / "atlas" / "atlas.db"
EMB_CACHE = Path("/tmp/atlas_emb_cache_v3.npy")
EMB_CACHE_HASH = Path("/tmp/atlas_emb_cache_v3.sha256")

PALETTE = [
    "#00f5d4", "#7b2cbf", "#ff007f", "#fee440", "#00bbf9",
    "#f15bb5", "#38b000", "#ff7b00", "#9b5de5", "#06d6a0",
    "#ef476f", "#118ab2", "#e0aaff", "#ffd166", "#e63946"
]

from atlas_memory.core.linguistics import LanguageAssetManager

STOP_WORDS_PL_EN = (
    LanguageAssetManager.get_instance().get_combined_stopwords(("pl", "en"))
    | {"true", "false", "none", "null", "state", "fact", "value", "item", "default"}
)


def _is_truthy(flag: object) -> bool:
    """Poprawnie parsuje flagi boolean, stringi 'true'/'false' i inty."""
    if isinstance(flag, str):
        return flag.strip().lower() in {"true", "1", "yes"}
    return bool(flag)


def _clean_text_payload(val_raw: Any) -> str:
    """Rozpakowuje JSON i ujednolica treść rekordu do postaci czytelnego tekstu."""
    if not val_raw:
        return ""
    val_str = str(val_raw).strip()
    if val_str.startswith("{") or val_str.startswith("["):
        try:
            parsed = json.loads(val_str)
            if isinstance(parsed, dict):
                parts = [f"{k}: {v}" for k, v in parsed.items() if not str(k).startswith("_")]
                return " ".join(parts)
            if isinstance(parsed, list):
                return " ".join(str(x) for x in parsed[:8])
        except Exception:
            pass
    return re.sub(r"\s+", " ", val_str)


def load_rows(db: Path, include_superseded: bool = False) -> List[Dict[str, Any]]:
    """Wczytuje rekordy, budując reprezentację semantyczną z dedupem najświeższych."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows: List[Dict[str, Any]] = []

    for r in con.execute("SELECT key, value, confidence, timestamp, metadata FROM state_variables"):
        try:
            meta = json.loads(r["metadata"] or "{}")
        except Exception:
            meta = {}

        key = str(r["key"] or "")[:150]
        cleaned_val = _clean_text_payload(r["value"])

        # Semantyczna kompozycja: klucz + przestrzeń nazw + zdekodowana treść
        clean_key = key.replace(":", " ").replace("_", " ").replace("-", " ")
        semantic_text = f"{clean_key}: {cleaned_val}".strip()
        if len(semantic_text) > 1500:
            semantic_text = semantic_text[:1500]

        origin = str(meta.get("origin") or meta.get("source_type") or key.split(":", 1)[0] or "unknown")
        is_sup = _is_truthy(meta.get("is_superseded") or meta.get("superseded"))

        rows.append({
            "key": key,
            "val_clean": cleaned_val[:200],
            "text": semantic_text,
            "conf": float(r["confidence"] or 1.0),
            "ts": float(r["timestamp"] or 0.0),
            "origin": origin,
            "superseded": is_sup,
            "cluster": -1,
        })
    con.close()

    if include_superseded:
        return rows

    # Zachowaj wyłącznie aktywne rekordy; duplikaty zawartości rozwiązuj wg najświeższego timestampu
    alive = [r for r in rows if not r["superseded"]]
    best: Dict[str, Dict[str, Any]] = {}
    for r in alive:
        dedup_key = r["val_clean"] if r["val_clean"] else r["key"]
        cur = best.get(dedup_key)
        if cur is None or r["ts"] > cur["ts"]:
            best[dedup_key] = r
    return list(best.values())


def corpus_hash(rows: List[Dict[str, Any]], encoder_name: str = "") -> str:
    """SHA-256 konkatenacji encodera, klucza i tekstu jako klucz cache."""
    h = hashlib.sha256()
    if encoder_name:
        h.update(encoder_name.encode("utf-8"))
    for r in rows:
        h.update(str(r["key"]).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(r["text"]).encode("utf-8"))
    return h.hexdigest()


def embed_corpus(rows: List[Dict[str, Any]]) -> np.ndarray:
    """Generuje wektory zgodnie z konfiguracją ATLAS (PolDense-68M 512D)."""
    encoder_name = "poldense68/512D"
    encoder = None
    try:
        from atlas_memory.l2_semantic.vector_backend import build_encoder, load_vector_backend
        backend = load_vector_backend()
        encoder = build_encoder(backend)
        if encoder is not None:
            encoder_name = f"{backend.kind}/{backend.dimension}D"
    except Exception:
        from atlas_memory.l2_semantic.onnx_encoder import OnnxDenseEncoder
        encoder = OnnxDenseEncoder(dimension=512, lazy_init=True)
        encoder_name = "poldense68/512D"

    n = len(rows)
    digest = corpus_hash(rows, encoder_name)

    if EMB_CACHE.exists() and EMB_CACHE_HASH.exists():
        arr = np.load(EMB_CACHE)
        if arr.shape[0] == n and EMB_CACHE_HASH.read_text(encoding="utf-8").strip() == digest:
            print(f"  [cache] Wczytano {n} wektorów ({encoder_name}) z pamięci podręcznej")
            return arr

    t0 = time.time()
    vecs: List[List[float]] = []

    if encoder is not None and hasattr(encoder, "encode_batch"):
        for i in range(0, n, 64):
            batch = [r["text"] for r in rows[i : i + 64]]
            vecs.extend(encoder.encode_batch(batch))
    elif encoder is not None and hasattr(encoder, "encode"):
        for r in rows:
            vecs.append(encoder.encode(r["text"]))
    else:
        raise RuntimeError("Brak enkodera PolDense-68M (512D). Generowanie mapy wymaga enkodera ONNX.")

    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms

    np.save(EMB_CACHE, arr)
    EMB_CACHE_HASH.write_text(digest, encoding="utf-8")
    print(f"  [embed] Wygenerowano {n} wektorów ({encoder_name}) w {time.time() - t0:.1f}s")
    return arr


def compute_P(D2: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    """Symetryczna macierz prawdopodobieństw przejścia z bisekcją sigmy (bez NaN)."""
    n = D2.shape[0]
    logP = np.log(perplexity)
    P = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        d = D2[i].copy()
        d[i] = np.inf
        fin = np.isfinite(d)
        d_off = np.where(fin, d, 0.0)
        beta = 1.0
        lo, hi = 1e-6, 1e6

        for _ in range(60):
            p = np.where(fin, np.exp(-d * beta), 0.0)
            s = p.sum()
            if s <= 0.0:
                hi = beta
                beta = 0.5 * (lo + hi)
                continue
            H = np.log(s) + beta * (d_off * p).sum() / s
            diff = H - logP
            if abs(diff) < 1e-5:
                break
            if diff > 0:
                lo = beta
            else:
                hi = beta
            beta = 0.5 * (lo + hi)

        p = np.where(fin, np.exp(-d * beta), 0.0)
        ps = p.sum()
        if ps <= 0.0:
            p = fin.astype(np.float64)
            ps = p.sum()
        P[i] = p / max(ps, 1e-12)

    P = 0.5 * (P + P.T)
    P = np.maximum(P, 1e-12)
    P /= P.sum()
    return P


def row_perplexity(P: np.ndarray) -> np.ndarray:
    """Efektywna perplexja każdego wiersza P (wymóg testów regresyjnych F1)."""
    Q = P.copy()
    np.fill_diagonal(Q, 0.0)
    Q = Q / np.maximum(Q.sum(axis=1, keepdims=True), 1e-12)
    nz = Q > 0.0
    H = -(np.where(nz, Q * np.log(np.where(nz, Q, 1.0)), 0.0)).sum(axis=1)
    return np.exp(H)


def run_tsne(
    X: np.ndarray,
    perplexity: float = 30.0,
    n_iter: int = 800,
    lr: float = 120.0,
    seed: int = 42,
) -> np.ndarray:
    """t-SNE z wczesną egzageracją (12x) i inicjalizacją PCA (SVD)."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    # Inicjalizacja PCA dla zachowania topologii globalnej
    X_cent = X - X.mean(axis=0)
    try:
        U, S, _ = np.linalg.svd(X_cent, full_matrices=False)
        Y = U[:, :2] * S[:2]
        std_val = float(np.std(Y[:, 0]))
        Y = Y / (std_val * 1e4 + 1e-6)
    except Exception:
        Y = rng.normal(0, 1e-4, (n, 2)).astype(np.float64)

    D2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(D2, 0.0)

    eff_perp = min(perplexity, max(5.0, (n - 1) / 3.0))
    P = compute_P(D2, eff_perp)

    P_early = P * 12.0
    vel = np.zeros_like(Y)

    for it in range(n_iter):
        P_curr = P_early if it < 250 else P

        D2y = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
        Q = 1.0 / (1.0 + D2y)
        np.fill_diagonal(Q, 0.0)
        Q = np.maximum(Q / max(float(np.sum(Q)), 1e-12), 1e-12)

        PQ = P_curr - Q
        grad = np.zeros_like(Y)

        for i in range(n):
            diff = Y[i] - Y
            mult = (PQ[i, :, None] * (1.0 / (1.0 + D2y[i, :, None])))
            grad[i] = 4.0 * np.sum(mult * diff, axis=0)

        mom = 0.5 if it < 250 else 0.8
        vel = mom * vel - lr * grad
        Y += vel
        Y -= Y.mean(axis=0)

    return Y


def run_kmeans(X: np.ndarray, k: int, rng: np.random.Generator, iters: int = 80) -> np.ndarray:
    """K-Means++ z metryką kosinusową na znormalizowanych wektorach."""
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        probs = d2 / (np.sum(d2) + 1e-12)
        centers.append(X[rng.choice(n, p=probs)])
    centers_arr = np.array(centers)

    lab = np.zeros(n, dtype=int)
    for _ in range(iters):
        D = np.sum((X[:, None, :] - centers_arr[None, :, :]) ** 2, axis=-1)
        new_lab = np.argmin(D, axis=1)
        if np.array_equal(new_lab, lab):
            break
        lab = new_lab
        for j in range(k):
            mask = lab == j
            if np.any(mask):
                c = X[mask].mean(axis=0)
                norm = np.linalg.norm(c)
                centers_arr[j] = c / (norm if norm > 0 else 1.0)
            else:
                centers_arr[j] = X[rng.integers(n)]
    return lab


def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Oblicza wskaźnik rozproszenia wariancji Calinskiego-Harabasza (czyste NumPy)."""
    n = X.shape[0]
    if k <= 1 or k >= n:
        return 0.0
    overall_mean = X.mean(axis=0)
    extra_disp = 0.0
    intra_disp = 0.0
    for j in range(k):
        pts = X[labels == j]
        if len(pts) == 0:
            continue
        c = pts.mean(axis=0)
        extra_disp += len(pts) * np.sum((c - overall_mean) ** 2)
        intra_disp += np.sum((pts - c) ** 2)
    if intra_disp <= 1e-9:
        return 0.0
    return float((extra_disp / (k - 1)) / (intra_disp / (n - k)))


def compute_cluster_labels(rows: List[Dict[str, Any]], labels: np.ndarray, k: int) -> Dict[int, str]:
    """Ekstrakcja kluczowych terminów tematycznych klastrów (c-TF-IDF)."""
    cluster_texts: Dict[int, List[str]] = defaultdict(list)
    for r, lab in zip(rows, labels, strict=False):
        tokens = re.findall(r"[A-Za-z0-9_żółćęśąźńŻÓŁĆĘŚĄŹŃ]{3,}", f"{r['key']} {r['val_clean']}".lower())
        tokens = [t for t in tokens if t not in STOP_WORDS_PL_EN and not t.isdigit()]
        cluster_texts[lab].extend(tokens)

    global_freq = collections.Counter()
    for words in cluster_texts.values():
        global_freq.update(words)

    cluster_names: Dict[int, str] = {}
    for j in range(k):
        c_cnt = collections.Counter(cluster_texts[j])
        scored = []
        for word, count in c_cnt.items():
            if count < 1:
                continue
            salience = count * math.log(1.0 + (len(rows) / (global_freq[word] + 1)))
            scored.append((salience, word))
        scored.sort(reverse=True)
        top = [w.capitalize() for _, w in scored[:2]]
        cluster_names[j] = " / ".join(top) if top else f"Grupa {j}"

    return cluster_names


def render_interactive_html(
    rows: List[Dict[str, Any]],
    P: np.ndarray,
    labels: np.ndarray,
    cluster_names: Dict[int, str],
    k: int,
    out: Path,
) -> None:
    """Generuje interaktywną aplikację wizualizacji (Canvas + Zoom/Pan + Live Search)."""
    pts_data = []
    p_min, p_max = P.min(axis=0), P.max(axis=0)
    p_norm = (P - p_min) / (p_max - p_min + 1e-9) * 1000.0

    for i, r in enumerate(rows):
        cid = int(labels[i])
        pts_data.append({
            "x": round(float(p_norm[i, 0]), 2),
            "y": round(float(p_norm[i, 1]), 2),
            "k": r["key"],
            "v": r["val_clean"],
            "o": r["origin"],
            "c": cid,
            "conf": round(r["conf"], 2),
            "color": PALETTE[cid % len(PALETTE)],
        })

    centroids = []
    for j in range(k):
        mask = labels == j
        if np.any(mask):
            cx = float(np.mean(p_norm[mask, 0]))
            cy = float(np.mean(p_norm[mask, 1]))
            centroids.append({
                "id": j,
                "name": cluster_names.get(j, f"Grupa {j}"),
                "x": round(cx, 2),
                "y": round(cy, 2),
                "count": int(np.sum(mask)),
                "color": PALETTE[j % len(PALETTE)],
            })

    payload_json = json.dumps({"points": pts_data, "clusters": centroids}, ensure_ascii=False)

    html_content = f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <title>ATLAS Memory Map v3</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #080c14; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; height: 100vh; overflow: hidden; display: flex; flex-direction: column; }}
    header {{ background: #0d1117; border-bottom: 1px solid #30363d; padding: 10px 20px; display: flex; align-items: center; justify-content: space-between; gap: 20px; z-index: 10; }}
    h1 {{ font-size: 15px; font-weight: 600; letter-spacing: 0.5px; color: #58a6ff; white-space: nowrap; }}
    .stats {{ font-size: 12px; color: #8b949e; }}
    .controls {{ display: flex; gap: 12px; align-items: center; flex: 1; justify-content: flex-end; }}
    input[type="text"] {{ background: #161b22; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 12px; border-radius: 6px; font-size: 12px; width: 260px; outline: none; }}
    input[type="text"]:focus {{ border-color: #58a6ff; }}
    .btn {{ background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 12px; border-radius: 6px; font-size: 12px; cursor: pointer; }}
    .btn:hover {{ background: #30363d; }}
    #viewport {{ position: relative; flex: 1; width: 100%; height: 100%; cursor: grab; }}
    #viewport:active {{ cursor: grabbing; }}
    canvas {{ width: 100%; height: 100%; display: block; }}
    #sidebar {{ position: absolute; right: 20px; top: 20px; width: 340px; background: rgba(13, 17, 23, 0.92); backdrop-filter: blur(8px); border: 1px solid #30363d; border-radius: 8px; padding: 16px; font-size: 12px; display: none; z-index: 20; max-height: calc(100vh - 120px); overflow-y: auto; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }}
    #sidebar h3 {{ color: #58a6ff; margin-bottom: 8px; font-size: 13px; word-break: break-all; }}
    .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold; margin-bottom: 8px; }}
    .sidebar-field {{ margin-bottom: 8px; color: #8b949e; }}
    .sidebar-field strong {{ color: #c9d1d9; }}
    .sidebar-val {{ background: #161b22; padding: 8px; border-radius: 4px; border: 1px solid #21262d; word-break: break-word; max-height: 250px; overflow-y: auto; color: #f0f6fc; font-family: monospace; font-size: 11px; }}
    #legend-bar {{ position: absolute; left: 20px; bottom: 20px; background: rgba(13, 17, 23, 0.88); border: 1px solid #30363d; border-radius: 8px; padding: 12px; font-size: 11px; display: flex; flex-direction: column; gap: 6px; max-height: 260px; overflow-y: auto; pointer-events: auto; z-index: 10; }}
    .leg-item {{ display: flex; align-items: center; gap: 8px; cursor: pointer; user-select: none; }}
    .leg-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  </style>
</head>
<body>
  <header>
    <h1>ATLAS Memory Map v3</h1>
    <div class="stats">{len(rows)} faktów · {k} klastrów tematycznych · t-SNE (PCA + Early Exaggeration)</div>
    <div class="controls">
      <input type="text" id="search" placeholder="Szukaj encji, klucza lub wartości...">
      <button class="btn" id="reset-view">Wyśrodkuj</button>
    </div>
  </header>

  <div id="viewport">
    <canvas id="map"></canvas>
    <div id="sidebar"></div>
    <div id="legend-bar"></div>
  </div>

  <script>
    const DATA = {payload_json};
    const canvas = document.getElementById('map');
    const ctx = canvas.getContext('2d');
    const sidebar = document.getElementById('sidebar');
    const searchInput = document.getElementById('search');
    const legendBar = document.getElementById('legend-bar');

    let width = canvas.width = canvas.offsetWidth;
    let height = canvas.height = canvas.offsetHeight;

    window.addEventListener('resize', () => {{
      width = canvas.width = canvas.offsetWidth;
      height = canvas.height = canvas.offsetHeight;
      draw();
    }});

    let zoom = 1.0;
    let panX = width / 2 - 500;
    let panY = height / 2 - 500;
    let isDragging = false;
    let dragStart = {{ x: 0, y: 0 }};
    let activeFilter = "";
    let selectedPoint = null;
    let hoveredPoint = null;

    DATA.clusters.forEach(c => {{
      const item = document.createElement('div');
      item.className = 'leg-item';
      item.innerHTML = `<span class="leg-dot" style="background:${{c.color}}"></span><span>${{c.name}} (${{c.count}})</span>`;
      item.onclick = () => {{
        activeFilter = c.name;
        searchInput.value = c.name;
        draw();
      }};
      legendBar.appendChild(item);
    }});

    function worldToScreen(x, y) {{
      return {{ x: x * zoom + panX, y: y * zoom + panY }};
    }}

    function screenToWorld(sx, sy) {{
      return {{ x: (sx - panX) / zoom, y: (sy - panY) / zoom }};
    }}

    function draw() {{
      ctx.fillStyle = '#080c14';
      ctx.fillRect(0, 0, width, height);

      // Siatka współrzędnych
      ctx.strokeStyle = '#161b22';
      ctx.lineWidth = 1;
      const step = 100 * zoom;
      const startX = panX % step;
      const startY = panY % step;
      ctx.beginPath();
      for (let x = startX; x < width; x += step) {{ ctx.moveTo(x, 0); ctx.lineTo(x, height); }}
      for (let y = startY; y < height; y += step) {{ ctx.moveTo(0, y); ctx.lineTo(width, y); }}
      ctx.stroke();

      const filterLower = activeFilter.toLowerCase();
      DATA.points.forEach(p => {{
        const pt = worldToScreen(p.x, p.y);
        if (pt.x < -20 || pt.x > width + 20 || pt.y < -20 || pt.y > height + 20) return;

        let matches = true;
        if (filterLower) {{
          matches = p.k.toLowerCase().includes(filterLower) ||
                    p.v.toLowerCase().includes(filterLower) ||
                    p.o.toLowerCase().includes(filterLower);
        }}

        const isSelected = selectedPoint === p;
        const isHovered = hoveredPoint === p;

        ctx.beginPath();
        let r = (2.5 + p.conf * 3.5) * Math.min(1.5, Math.max(0.6, zoom * 0.8));
        if (isSelected || isHovered) r *= 1.8;

        ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
        ctx.fillStyle = matches ? p.color : '#21262d';
        ctx.globalAlpha = matches ? (isSelected ? 1.0 : 0.82) : 0.15;
        ctx.fill();

        if (isSelected || isHovered) {{
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2;
          ctx.globalAlpha = 1.0;
          ctx.stroke();
        }}
      }});
      ctx.globalAlpha = 1.0;

      // Etykiety klastrów
      DATA.clusters.forEach(c => {{
        const pt = worldToScreen(c.x, c.y);
        if (pt.x < -100 || pt.x > width + 100 || pt.y < -50 || pt.y > height + 50) return;

        ctx.font = '600 11px monospace';
        const txt = `${{c.name}} [${{c.count}}]`;
        const tw = ctx.measureText(txt).width;

        ctx.fillStyle = 'rgba(13, 17, 23, 0.85)';
        ctx.strokeStyle = c.color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(pt.x - tw / 2 - 8, pt.y - 12, tw + 16, 24, 6);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#f0f6fc';
        ctx.textAlign = 'center';
        ctx.fillText(txt, pt.x, pt.y + 4);
      }});
    }}

    canvas.addEventListener('wheel', e => {{
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
      const mouseX = e.offsetX;
      const mouseY = e.offsetY;
      panX = mouseX - (mouseX - panX) * zoomFactor;
      panY = mouseY - (mouseY - panY) * zoomFactor;
      zoom *= zoomFactor;
      draw();
    }}, {{ passive: false }});

    canvas.addEventListener('mousedown', e => {{
      isDragging = true;
      dragStart = {{ x: e.clientX - panX, y: e.clientY - panY }};
    }});

    window.addEventListener('mousemove', e => {{
      if (isDragging) {{
        panX = e.clientX - dragStart.x;
        panY = e.clientY - dragStart.y;
        draw();
        return;
      }}

      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const world = screenToWorld(mx, my);

      let found = null;
      for (const p of DATA.points) {{
        const dx = p.x - world.x;
        const dy = p.y - world.y;
        if (dx * dx + dy * dy < (12 / zoom) * (12 / zoom)) {{
          found = p;
          break;
        }}
      }}
      if (found !== hoveredPoint) {{
        hoveredPoint = found;
        draw();
      }}
    }});

    window.addEventListener('mouseup', () => isDragging = false);

    canvas.addEventListener('click', e => {{
      if (hoveredPoint) {{
        selectedPoint = hoveredPoint;
        sidebar.style.display = 'block';
        sidebar.innerHTML = `
          <span class="badge" style="background:${{selectedPoint.color}}22; color:${{selectedPoint.color}}; border: 1px solid ${{selectedPoint.color}}">
            ${{DATA.clusters[selectedPoint.c]?.name || 'Grupa ' + selectedPoint.c}}
          </span>
          <h3>${{selectedPoint.k}}</h3>
          <div class="sidebar-field">Źródło: <strong>${{selectedPoint.o}}</strong></div>
          <div class="sidebar-field">Pewność: <strong>${{selectedPoint.conf}}</strong></div>
          <div class="sidebar-field">Zawartość:</div>
          <div class="sidebar-val">${{selectedPoint.v}}</div>
        `;
        draw();
      }} else {{
        selectedPoint = null;
        sidebar.style.display = 'none';
        draw();
      }}
    }});

    searchInput.addEventListener('input', e => {{
      activeFilter = e.target.value;
      draw();
    }});

    document.getElementById('reset-view').addEventListener('click', () => {{
      zoom = 1.0;
      panX = width / 2 - 500;
      panY = height / 2 - 500;
      activeFilter = "";
      searchInput.value = "";
      draw();
    }});

    draw();
  </script>
</body>
</html>"""

    out.write_text(html_content, encoding="utf-8")


# Aliases dla kompatybilności wstecznej z testami i narzędziami v2
tsne = run_tsne
kmeans = run_kmeans
embed = embed_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="ATLAS Memory Map v3")
    parser.add_argument("--out", type=Path, default=Path.home() / "atlas_memory_map_v3.html")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--include-superseded", action="store_true")
    args = parser.parse_args()

    rows = load_rows(DB, include_superseded=args.include_superseded)
    print(f"▶ Załadowano {len(rows)} unikalnych rekordów z bazy ATLAS")

    X = embed_corpus(rows)
    print("▶ Uruchamianie t-SNE z wczesną egzageracją i inicjalizacją PCA...")
    P = run_tsne(X, perplexity=args.perplexity)

    print(f"▶ Dobór optymalnego podziału klastrów (Calinski-Harabasz do k={args.max_k})...")
    rng = np.random.default_rng(42)
    best_k = 4
    best_labels = None
    best_score = -1.0

    max_cand = min(args.max_k, max(4, len(rows) // 6))
    for cand_k in range(4, max_cand + 1):
        lab = run_kmeans(X, cand_k, rng)
        ch_score = calinski_harabasz_score(X, lab, cand_k)
        if ch_score > best_score:
            best_score = ch_score
            best_k = cand_k
            best_labels = lab

    cluster_names = compute_cluster_labels(rows, best_labels, best_k)
    print(f"▶ Wybrano k={best_k} klastrów (Calinski-Harabasz score: {best_score:.2f}):")
    for cid, name in cluster_names.items():
        count = int(np.sum(best_labels == cid))
        print(f"   [{cid:2d}] {name:<35} ({count} rekordów)")

    render_interactive_html(rows, P, best_labels, cluster_names, best_k, args.out)
    print(f"\n Wygenerowano interaktywną mapę wiedzy: {args.out}")
    print("Otwórz plik w przeglądarce, aby badać stan pamięci agenta.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
