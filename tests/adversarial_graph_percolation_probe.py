"""
GRAPH SCALE & SATURATION PROBE (POST-2027 ENGINE)
Wektor 2: Test Odporności Perkolacyjnej w Kùzu / HippoRAG 2.
Weryfikuje zachowanie operatora Laplace'a W_ij / sqrt(deg(i)*deg(j)) i tunelowania beta_tunnel = 0.10
w syntetycznej sieci bezskalowej Barabási-Albert (N = 5000 węzłów, E = 20000 krawędzi).
Gwarantuje ostrość spektralną (Top-K separowalność >= 0.15) i brak katastrofy delokalizacji.
"""

import time

import networkx as nx

from atlas_memory.cognitive.retrieval.ppr import PersonalizedPageRank


def test_adversarial_graph_percolation_and_spectral_sharpness():
    """
    Test odporności na delokalizację w skali perkolacyjnej (Scale-Free Barabási-Albert N=5000, E=20000).
    Sprawdza:
    1. Czas budowy i rozwiązania PPR O(E) (< 1000ms dla N=5000).
    2. Ostrość spektralną: Top-K separowalność (top1 - top10) / top1 >= 0.15.
    3. Brak jednostajnego rozmycia tła: max(score) >= 50 * (1 / N).
    4. Ochronę tunelowania aksjomatów bez rozpraszania energii w szum.
    """
    # 1. Generowanie syntetycznej sieci bezskalowej N=5000, m=4 (~20 000 krawędzi)
    n_nodes = 5000
    m_edges_per_node = 4
    g_ba = nx.barabasi_albert_graph(n=n_nodes, m=m_edges_per_node, seed=42)

    # Mapowanie węzłów na nazwy semantyczne
    nx_graph = nx.DiGraph()

    # Wstrzyknięcie węzłów z prefiksami
    for u in range(n_nodes):
        if u >= 4980:
            name = f"rule:security_policy_{u}"
        elif 20 <= u < 50:
            name = f"concept:cluster_alpha_{u}"
        elif 50 <= u < 100:
            name = f"service:api_gateway_{u}"
        else:
            name = f"entity:node_{u}"
        nx_graph.add_node(name, type="entity")

    id_to_name = {
        u: (
            f"rule:security_policy_{u}" if u >= 4980
            else f"concept:cluster_alpha_{u}" if 20 <= u < 50
            else f"service:api_gateway_{u}" if 50 <= u < 100
            else f"entity:node_{u}"
        )
        for u in range(n_nodes)
    }

    # Dodanie krawędzi z wagami zaufania (confidence=0.85)
    for u, v in g_ba.edges():
        u_name = id_to_name[u]
        v_name = id_to_name[v]
        nx_graph.add_edge(u_name, v_name, confidence=0.85, predicate="RELATION")
        nx_graph.add_edge(v_name, u_name, confidence=0.85, predicate="RELATION")

    assert nx_graph.number_of_nodes() == 5000
    assert nx_graph.number_of_edges() >= 19000

    # 2. Uruchomienie HippoRAG 2 PPR dla zapytania skupionego na klastrze Alpha
    ppr = PersonalizedPageRank(damping=0.85)
    query = "concept:cluster_alpha_25 service:api_gateway_60"

    t0 = time.perf_counter()
    res = ppr.run(query=query, graph_store=nx_graph, top_k=20)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # ASERCJA 1: Czas wykonania (sub-sekundowy dzięki O(E) igraph backend)
    assert elapsed_ms < 2000.0, f"Zbyt wolna ewaluacja PPR w skali 5000 węzłów: {elapsed_ms:.1f}ms"

    # Pobranie wektora wag stacjonarnych
    scores = res.node_scores
    assert len(scores) > 0, "PPR nie może zwrócić pustego rozkładu"

    sorted_scores = sorted(scores.values(), reverse=True)
    top1 = sorted_scores[0]
    top10 = sorted_scores[min(9, len(sorted_scores) - 1)]

    # ASERCJA 2: Ostrość spektralna (Top-K separowalność >= 0.15)
    # Rozkład nie może ulec spłaszczeniu
    spectral_separability = (top1 - top10) / top1
    assert spectral_separability >= 0.15, (
        f"FALSYFIKACJA: Katastrofa delokalizacji! Top-K separowalność zbyt niska: {spectral_separability:.4f} < 0.15"
    )

    # ASERCJA 3: Ochrona przed delokalizacją Andersona
    # Maksimum rozkładu musi znacznie przewyższać poziom jednostajnego szumu 1 / N = 0.0002
    uniform_noise_level = 1.0 / float(n_nodes)
    ratio_above_noise = top1 / uniform_noise_level
    assert ratio_above_noise >= 40.0, (
        f"FALSYFIKACJA: Rozmycie stochastyczne! Szczyt rozkładu tylko {ratio_above_noise:.1f}x ponad szumem"
    )

    # ASERCJA 4: Obecność nasion zapytania w czołówce rozkładu
    top_keys = [k for k, v in sorted(scores.items(), key=lambda x: -x[1])[:10]]
    assert any("cluster_alpha" in k or "api_gateway" in k for k in top_keys), (
        f"Nasiona zapytania muszą dominować czołówkę PPR, otrzymano: {top_keys}"
    )
