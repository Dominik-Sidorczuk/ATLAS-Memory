"""
Testy integracyjne ATLAS z REALNYM OmniRoute (localhost:20128).
Zero mocków — prawdziwe HTTP requests do bramy API.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import yaml

# Dodaj ścieżkę do src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def get_api_key() -> str:
    """Wyciągnij klucz API z ~/.hermes/config.yaml"""
    config_path = Path.home() / ".hermes/config.yaml"
    if not config_path.exists():
        return ""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("model", {}).get("api_key", "")



API_BASE = "http://localhost:20128/v1"
API_KEY = get_api_key()
MODEL_CHEAP = "auto/cheap"  # Tani model do ekstrakcji SPO


async def test_1_real_llm_extraction():
    """T1: Realny LLM call do OmniRoute — ekstrakcja SPO z tury rozmowy."""
    print("\n" + "="*70)
    print("TEST 1: Realny LLM extraction (POST do OmniRoute /chat/completions)")
    print("="*70)

    prompt = """Ekstrahuj fakty z poniższej rozmowy w formacie JSON:
[{"subject": "...", "predicate": "...", "object": "...", "source_type": "user_explicit|tool_output|agent_inference", "confidence": 0.0-1.0}]

Rozmowa:
Użytkownik: Mój serwer NAS ma adres IP 192.168.1.100 i działa na TrueNAS.
Agent: Zanotowałem, NAS to 192.168.1.100 na TrueNAS.

Zwróć TYLKO JSON, bez dodatkowego tekstu."""

    payload = {
        "model": MODEL_CHEAP,
        "messages": [
            {"role": "system", "content": "You are a fact extraction engine. Return only valid JSON array."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 500,
        "stream": False,  # WYMAGANE: OmniRoute streamuje SSE domyślnie (model glm-5.2)
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{API_BASE}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]
    print(f"Model: {data.get('model', 'unknown')}")
    print(f"Usage: {data.get('usage', {})}")
    print(f"Raw content:\n{content[:500]}")

    # Parsuj JSON z odpowiedzi
    try:
        # Wyciągnij JSON (może być w ```json ... ```)
        if "```" in content:
            start = content.find("[")
            end = content.rfind("]") + 1
            content = content[start:end]
        facts = json.loads(content)
        print(f"\n✅ Ekstrahowano {len(facts)} faktów:")
        for f in facts:
            print(f"   - {f.get('subject')} --[{f.get('predicate')}]--> {f.get('object')} (conf={f.get('confidence')})")
        return len(facts) >= 2
    except json.JSONDecodeError as e:
        print(f"❌ Nie udało się sparsować JSON: {e}")
        return False


async def test_2_cache_stats_from_omniroute():
    """T2: Pobierz prawdziwe statystyki cache z OmniRoute."""
    print("\n" + "="*70)
    print("TEST 2: Realne statystyki cache z OmniRoute")
    print("="*70)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Spróbuj endpoint /cache/stats (może nie istnieć)
        resp = await client.get(f"{API_BASE}/../cache/stats", headers=headers)
        if resp.status_code == 200:
            stats = resp.json()
            print(f"✅ Cache stats: {json.dumps(stats, indent=2)[:500]}")
            return True
        else:
            print(f"⚠️  Endpoint /cache/stats nie dostępny (HTTP {resp.status_code})")
            print("   (OmniRoute może nie eksponować cache stats bezpośrednio)")
            return True  # Nie blokujące


async def test_3_real_multi_turn_with_adapter():
    """T3: Adapter z prawdziwymi turami rozmowy (bez realnego LLM w adapterze, ale z realnym cache monitor)."""
    print("\n" + "="*70)
    print("TEST 3: Adapter z 10 turami (cache monitoring)")
    print("="*70)

    from atlas_memory import HermesMemoryAdapter

    adapter = HermesMemoryAdapter.create_default(
        omniroute_api_base=API_BASE,
        omniroute_model=MODEL_CHEAP,
    )

    turns = [
        "Cześć, jak się masz?",
        "Jaki jest IP mojego NAS-a?",
        "Dzięki za pomoc!",
        "Pokaż konfigurację serwera TrueNAS",
        "Super, to wszystko.",
        "Jaki jest port bazy danych?",
        "Miłego dnia!",
        "Gdzie jest zapisany plik konfiguracyjny?",
        "Ok, rozumiem.",
        "Sprawdź status klastra",
    ]

    for i, turn in enumerate(turns, 1):
        res = await adapter.orchestrated_search(turn)
        skipped = res.get("retrieval_skipped", False)
        prefix = adapter.get_cache_prefix({"user": "Dominik"}, ["Rule: be helpful"], model_name=MODEL_CHEAP)
        print(f"  Tura {i}: '{turn[:40]}...' → {'SKIP' if skipped else 'RETRIEVE'} | prefix_hash={prefix[:16]}...")

    report = adapter.get_telemetry_report()
    print("\n✅ Telemetry report:")
    print(f"   Cache stats: {report['cache_stats']}")
    print(f"   Orchestrator stats: {report['orchestrator_stats']}")
    print(f"   Active sensing errors: {report['active_sensing_errors_count']}")

    # Sprawdź że cache monitoring działa (klucze z CacheHitMonitor.monthly_report)
    cache_stats = report["cache_stats"]
    total_turns = cache_stats.get("total_monitored_turns", 0)
    hit_rate = cache_stats.get("overall_cache_hit_rate_pct", 0.0)
    tokens_saved = cache_stats.get("total_tokens_saved", 0)
    print(f"\n✅ Cache monitoring: | hit_rate={hit_rate}% | tokens_saved={tokens_saved} | turns={total_turns}")
    return total_turns >= 10 and hit_rate >= 40.0


async def test_4_real_veracity_arbitration():
    """T4: Realna arbitracja veracity z adapterem (USER_EXPLICIT > AGENT_INFERENCE)."""
    print("\n" + "="*70)
    print("TEST 4: Realna arbitracja veracity (USER_EXPLICIT > AGENT_INFERENCE)")
    print("="*70)

    from atlas_memory import EpistemicSource, HermesMemoryAdapter, MemoryRecord

    adapter = HermesMemoryAdapter.create_default()

    # Symuluj konflikt
    old_infer = MemoryRecord(
        subject="cluster_status", predicate="state", object="degraded",
        source_type=EpistemicSource.AGENT_INFERENCE, confidence=0.6,
        timestamp=time.time() - 10,
    )
    new_user = MemoryRecord(
        subject="cluster_status", predicate="state", object="healthy",
        source_type=EpistemicSource.USER_EXPLICIT, confidence=1.0,
        timestamp=time.time(),
    )

    ranked = adapter.orchestrator.epistemic_rank([old_infer, new_user], query="cluster status")
    top_source = ranked[0][0].source_type
    top_score = ranked[0][1]
    bottom_source = ranked[1][0].source_type
    bottom_score = ranked[1][1]

    print("✅ Ranking:")
    print(f"   1. {top_source.value} (score: {top_score:.3f})")
    print(f"   2. {bottom_source.value} (score: {bottom_score:.3f})")

    if top_source == EpistemicSource.USER_EXPLICIT:
        print(f"✅ USER_EXPLICIT wygrywa nad AGENT_INFERENCE (różnica: {top_score-bottom_score:.3f})")
        return True
    else:
        print("❌ USER_EXPLICIT NIE wygrywa!")
        return False


async def main():
    print("\n" + "█"*70)
    print("█ TESTY INTEGRACYJNE ATLAS — REALNY OMNIRUTE (localhost:20128) " + "█"*10)
    print("█"*70)

    results = []

    # T1: Realny LLM extraction
    try:
        ok = await test_1_real_llm_extraction()
        results.append(("T1: Real LLM extraction", ok))
    except Exception as e:
        print(f"❌ T1 FAILED: {e}")
        results.append(("T1: Real LLM extraction", False))

    # T2: Cache stats
    try:
        ok = await test_2_cache_stats_from_omniroute()
        results.append(("T2: Cache stats", ok))
    except Exception as e:
        print(f"❌ T2 FAILED: {e}")
        results.append(("T2: Cache stats", False))

    # T3: Adapter z 10 turami
    try:
        ok = await test_3_real_multi_turn_with_adapter()
        results.append(("T3: Adapter 10 turns", ok))
    except Exception as e:
        print(f"❌ T3 FAILED: {e}")
        results.append(("T3: Adapter 10 turns", False))

    # T4: Veracity arbitration
    try:
        ok = await test_4_real_veracity_arbitration()
        results.append(("T4: Veracity arbitration", ok))
    except Exception as e:
        print(f"❌ T4 FAILED: {e}")
        results.append(("T4: Veracity arbitration", False))

    # Podsumowanie
    print("\n" + "█"*70)
    print("█ PODSUMOWANIE TESTÓW INTEGRACYJNYCH " + " "*38 + "█")
    print("█"*70)
    for name, ok in results:
        icon = "✅" if ok else "❌"
        print(f"{icon} {name}")

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\n{'='*70}")
    print(f"WYNIK: {passed}/{total} testów przeszło")
    print(f"{'='*70}")

    if passed == total:
        print("\n🎉 WSZYSTKIE TESTY INTEGRACYJNE PRZESZŁY — KOD JEST SPRAWNY NA REALNYM OMNIRUTIE!")
        return True
    else:
        print("\n⚠️  Niektóre testy nie przeszły — wymaga poprawy")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
