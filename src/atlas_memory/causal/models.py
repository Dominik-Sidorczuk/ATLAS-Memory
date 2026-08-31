from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CausalEdge(BaseModel):
    """Pojedyncza krawędź przyczynowo-skutkowa w grafie zależności."""
    source: str
    predicate: str
    target: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    impact_type: str = Field(default="dependency", description="np. 'dependency', 'triggers', 'hosts', 'configures'")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CausalPath(BaseModel):
    """Wieloskokowa ścieżka propagacji skutków akcji."""
    source_entity: str
    simulated_action: str
    steps: List[CausalEdge]
    affected_target: str
    cumulative_confidence: float = Field(ge=0.0, le=1.0)
    risk_level: str = Field(default="MODERATE", description="LOW, MODERATE, HIGH, CRITICAL")
    jepa_latent_divergence: Optional[float] = None
    description: str = ""


class WhatIfResult(BaseModel):
    """Podsumowanie symulacji ekstrapolacyjnej 'Co się stanie jeśli...'."""
    source_entity: str
    simulated_action: str
    paths: List[CausalPath] = Field(default_factory=list)
    highest_risk_target: Optional[str] = None
    overall_safety_score: float = Field(default=1.0, ge=0.0, le=1.0)
    simulated_at: float = Field(default_factory=time.time)
    jepa_extrapolation_used: bool = False


class DiffusionNode(BaseModel):
    """Pojedynczy węzeł objęty dyfuzją przyczynowo-skutkową."""
    node_id: str = Field(description="Identyfikator węzła")
    impact_score: float = Field(default=0.0, description="Skumulowany wynik wpływu fali dyfuzji")
    path_length: int = Field(default=1, ge=0, description="Długość najkrótszej ścieżki od źródła")
    cumulative_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Łączna pewność ścieżki")


class DiffusionResult(BaseModel):
    """Wynik probabilistycznej analizy dyfuzji przyczynowej."""
    nodes: List[DiffusionNode] = Field(default_factory=list, description="Lista węzłów dotkniętych falą skutków")
    edges: List[CausalEdge] = Field(default_factory=list, description="Krawędzie przyczynowe tworzące drzewo dyfuzji")
    total_impact: float = Field(default=0.0, description="Suma wpływów na wszystkie dotknięte węzły")
    max_depth_reached: int = Field(default=0, ge=0, description="Maksymalna głębokość osiągnięta podczas dyfuzji")


class CPoFNode(BaseModel):
    """Węzeł będący krytycznym punktem awarii (Critical Point of Failure)."""
    node_id: str = Field(description="Identyfikator krytycznego węzła")
    severity: float = Field(ge=0.0, le=1.0, description="Dotkliwość awarii (% utraconych/rozspójnionych węzłów)")
    affected_nodes: List[str] = Field(default_factory=list, description="Lista węzłów tracących osiągalność")
    description: str = Field(default="", description="Opis wpływu rozspójnienia grafu")


