"""ATLAS V85 Zero-Domain Core & Linguistic Assets Test Suite.

Verifies:
1. External linguistic asset manager (stopwords, phatic tokens, intent keywords).
2. Zero-shot vector cognitive risk classification (SemanticRiskAssessor) across domains.
3. Declarative domain profile registry (default_core, devops, healthcare).
4. Universal structural extractor and dynamic canonicalizer.
5. Decoupled telemetry and session prefix guard.
"""
from __future__ import annotations

import pytest

from atlas_memory.causal.energy_module import EnergyModule
from atlas_memory.causal.risk_assessor import SemanticRiskAssessor
from atlas_memory.cognitive.lexicon import DEVELOPER_TOOLS, STOP_WORDS, polish_stem
from atlas_memory.cognitive.retrieval.gate import _CONT, _INTENT_KWS, _TRIVIAL, should_retrieve_query
from atlas_memory.core.linguistics import LanguageAssetManager
from atlas_memory.domain.registry import DomainProfileRegistry
from atlas_memory.extensions.canonicalizer import EntityCanonicalizer
from atlas_memory.ingest.domain_rules import DomainRelationExtractor


def test_language_asset_manager_and_lexicon_unification():
    """Verifies that LanguageAssetManager loads external assets and unifies lexicons."""
    lam = LanguageAssetManager.get_instance()

    pl_stops = lam.get_stopwords("pl")
    en_stops = lam.get_stopwords("en")
    combined = lam.get_combined_stopwords(("pl", "en"))

    assert "jest" in pl_stops
    assert "dla" in pl_stops
    assert "the" in en_stops
    assert "with" in en_stops
    assert "jest" in combined and "the" in combined

    # Backward compatibility symbols in lexicon.py
    assert "jest" in STOP_WORDS
    assert "the" in STOP_WORDS
    assert "docker" in DEVELOPER_TOOLS
    assert polish_stem("programowanie") == "program"

    # Phatic and intent tokens
    phatic = lam.get_phatic_tokens("pl")
    assert "cześć" in phatic["trivial"]
    assert "kontynuuj" in phatic["continuation"]
    assert "cześć" in _TRIVIAL
    assert "kontynuuj" in _CONT
    assert "szukaj" in _INTENT_KWS


def test_semantic_risk_assessor_cross_domain():
    """Verifies that SemanticRiskAssessor accurately classifies risk across diverse domains."""
    assessor = SemanticRiskAssessor.get_instance()

    # DevOps action
    level_devops, _ = assessor.assess_action_risk("drop table users delete all data")
    assert assessor.is_volatile("drop table users delete all data") is True

    # Medical action
    level_med, _ = assessor.assess_action_risk("podaj 500mg lethal poison")
    assert level_med == "CRITICAL"
    assert assessor.is_critical("podaj 500mg lethal poison") is True

    # Financial action
    assert assessor.is_volatile("zablokuj linie kredytowa i usun konto") is True

    # Safe inspection
    level_safe, _ = assessor.assess_action_risk("read system status and check logs")
    assert level_safe == "SAFE"
    assert assessor.is_volatile("read system status and check logs") is False


@pytest.mark.asyncio
async def test_energy_module_uses_semantic_risk_assessor():
    """Verifies that EnergyModule delegates action volatility to SemanticRiskAssessor."""
    em = EnergyModule()

    # Destructive action gets high base cost
    cost_drop = await em.intrinsic_cost(state="StateA", action="drop database production")
    assert cost_drop >= 0.45

    # Safe action gets low base cost
    cost_read = await em.intrinsic_cost(state="StateA", action="read system status")
    assert cost_read <= 0.20


def test_domain_profile_registry_and_extractor():
    """Verifies declarative loading of domain profiles from JSON."""
    reg = DomainProfileRegistry.get_instance()
    profiles = reg.list_profiles()

    assert "default_core" in profiles
    assert "devops_infrastructure" in profiles
    assert "healthcare_oncology" in profiles

    rules = reg.get_all_structural_rules()
    assert len(rules) >= 3

    extractor = DomainRelationExtractor.from_registry(reg)
    edges: list[tuple[str, str, str, float]] = []

    def mock_add(u: str, p: str, v: str, c: float = 1.0):
        edges.append((u, p, v, c))

    # Apply extractor to medical content declared in healthcare profile
    extractor.extract_and_apply(
        subject="Pacjent",
        content="Wymaga regularnego screeningu onkologicznego (Morfologia, USG, Marker).",
        add_edge_fn=mock_add,
        stop_words=set(),
    )

    consists_of_targets = [v for u, p, v, _ in edges if p == "consists_of"]
    assert "Morfologia" in consists_of_targets


def test_entity_canonicalizer_loads_from_registry():
    """Verifies EntityCanonicalizer initializes from domain profile entities."""
    canon = EntityCanonicalizer()

    # Registered entities from devops.json profile
    assert canon.canonicalize("mój nas") == "entity_nas_01"
    assert canon.canonicalize("hermes agent") == "entity_agent_core"


def test_policy_gate_with_language_assets():
    """Verifies Policy Gate correctly detects trivial turns and informational intent via LanguageAssetManager."""
    # Trivial turns skipped
    can_ret1, _, reason1 = should_retrieve_query("cześć, co tam?")
    assert can_ret1 is False
    assert reason1 == "conversational_turn_no_entity"

    # Intent query triggered
    can_ret2, ents2, _ = should_retrieve_query("sprawdź konfigurację test:cluster:node1")
    assert can_ret2 is True
    assert "test:cluster:node1" in ents2

