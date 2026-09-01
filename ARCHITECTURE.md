# ATLAS — Technical Architecture & Cognitive Specification

> **Document Version:** 3.0.0
> **Status:** Production-Verified
> **Target Runtime:** Python 3.14 (No-GIL) & Pixi Isolated Workspace
> **Ecosystem Integration:** Hermes Agent Plugin

---

## Table of Contents

1. [Introduction: The Black-Box Paradigm](#1-introduction-the-black-box-paradigm)
2. [Global Data Flow](#2-global-data-flow)
3. [Memory Tiers L0–L3](#3-memory-tiers-l0l3)
4. [Causal Reasoning (Causal Engine)](#4-causal-reasoning-causal-engine)
5. [Multi-Agent Sync (Delta-CRDT & Gossip)](#5-multi-agent-sync-delta-crdt--gossip)
6. [Vector Quantization (MIB 32x & SIMD Popcount)](#6-vector-quantization-mib-32x--simd-popcount)
7. [Active Perception (Active Sensing)](#7-active-perception-active-sensing)
8. [Micro-Sidecar IPC Architecture](#8-micro-sidecar-ipc-architecture)
9. [Hermes Plugin & Memory Contract](#9-hermes-plugin--memory-contract)

---

## 1. Introduction: The Black-Box Paradigm

AI agents running on cloud models (DeepSeek, Qwen, Claude, GPT-4o) face a fundamental constraint: **these models are black-box APIs**. No access to weights, logits, or hidden states makes Test-Time Training or free-energy computation at the tensor level impossible.

**ATLAS** solves this through **externalization of cognitive processes** into a deterministic, probabilistic, and cryptographic memory engine. All retention intelligence, causal reasoning, contradiction arbitration, and prompt-cache optimization are delegated to a dedicated Python 3.14 (No-GIL) environment running as a micro-sidecar alongside the agent.

```
ATLAS PARADIGMS
1. Executable Memory       — procedural knowledge as compiled micro-programs
                             (SKILL.md + handler.py)
2. Causal Topological Graph — structural causal models (SCM) with perturbation
                             diffusion and bottleneck detection (CPoF)
3. Active Predictive Coding — asynchronous prediction error estimation
                             (0 LLM tokens)
4. Veracity-First Epistemic Calibration — truth arbitration:
                             USER_EXPLICIT > TOOL_OUTPUT > EXTERNAL_DOC > AGENT_INFERENCE
5. Decentralized Multi-Agent Sync — encrypted knowledge exchange via
                             Delta-CRDT and Gossip (AES-256-GCM)
6. Hardware SIMD Quantization — 32x float32 → uint64 embedding compression
```

---

## 2. Global Data Flow

```
[ USER TURN ]
        │
        ▼
┌──────────────────────────────────────────────────────┐
│ 1. KV-Cache Prefix Guard & CacheHitMonitor           │
│    • build_immutable_prefix(identity + profile)      │  <-- ~90% Prompt Cache Hit
└────────────────────────┬─────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│ 2. Retrieval Policy Gate & Entity Canonicalizer      │
│    • Chat / greeting → GATE CLOSED (0 tokens)        │  <-- 100% Skip for chat
│    • Entities detected → DUAL-ENGINE RETRIEVAL:      │
│      - Kùzu Cypher Subgraph (1-2 hop + confidence)   │
│      - Qdrant Vector Store (semantic episodes)       │
└────────────────────────┬─────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│ 3. Epistemic Re-Ranker & Token Budget Governor       │
│    • Salience-decay: S(f) = 0.4*Sim + 0.3*I(f) + ... │
│    • Veracity: USER(1.0) > TOOL(0.85) > DOC(0.65)    │
│    • Hard cap to budget (max 1500 tokens)            │
└────────────────────────┬─────────────────────────────┘
        │ Generated context block
        ▼
┌──────────────────────────────────────────────────────┐
│ 4. Main API Inference (black-box LLM)                │
└────────────────────────┬─────────────────────────────┘
        │ Asynchronous event stream
        ▼
┌──────────────────────────────────────────────────────┐
│ 5. Shadow Worker & Active Sensing                    │
│    • Background SPO relation extraction              │
│    • Prediction error detection (Zero-LLM commit)    │
│    • Contradiction resolution (supersede=True)       │
│    • SHA-256 Tamper-Evident Hash-Chain               │
└────────────────────────┬─────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│ 6. Sleep Baker, Skill Compiler & Multi-Agent Sync    │
│    • Trajectory consolidation → SOP                  │
│    • SOP compilation → SKILL.md + handler.py         │
│    • AST Safety Scanner (eval/exec block)            │
│    • Encrypted Gossip Delta-CRDT (AES-256-GCM)       │
└──────────────────────────────────────────────────────┘
```

---

## 3. Memory Tiers L0–L3

### 3.1 Layer 0: Dynamic Latent Memory (`src/atlas_memory/l0_dynamic/`)

Simulates Test-Time Training outside the main model:

$$z = x \cdot (W_{base} + W_{ttt}), \quad \hat{x} = z \cdot W_{recon}$$

- `numba_ttt_adapt_step` compiled with `@njit(fastmath=True, nogil=True)` — adaptation step in **0.01–0.03 ms**
- `adapt_step_async()` delegates computation to the thread pool (no event loop blocking)

### 3.2 Layer 1: Working Memory & JEPA World Model (`src/atlas_memory/l1_working/`, `arrow_buffer/`)

Predictive world model JEPA (*Joint Embedding Predictive Architecture*):

$$s_{t+1} = \tanh(s_t W_s + a_t W_a + b)$$

- `select_best_action_trajectory()` — tree-of-variants simulation to depth k
- `ArrowTrajectoryBuffer` — states in columnar PyArrow memory, NumPy conversion in O(1), Parquet dump

### 3.3 Layer 2: Semantic Memory (`src/atlas_memory/l2_semantic/`)

Hybrid declarative and semantic memory store:

- **Kùzu Graph Store** — Cypher graph DB: `(Entity)-[RELATION {confidence, timestamp}]->(Entity)`
- **Qdrant Vector Store** — vector DB with FastEmbed, in-memory SIMD pre-filtering
- **VerifiedKVStore + SHA-256 Hash-Chain** — tamper-evident audit chain:

$$H_i = \text{SHA-256}(\text{seq}_i \parallel \text{data}_i \parallel H_{i-1})$$

- `verify_chain_integrity()` — O(N) chain integrity verification, immediate substitution detection
- Automatic schema migration (PRAGMA + ALTER TABLE)

### 3.4 Layer 3: Procedural Memory & Sleep Consolidation (`src/atlas_memory/l3_procedural/`)

- **Epistemic Arbitration**: USER_EXPLICIT > TOOL_OUTPUT > AGENT_INFERENCE (supersede=True)
- **SleepBaker**: distillation of repeated tool sequences into SOP:

$$\text{Frequency} \ge f_{min} \land \text{SuccessRate} \ge s_{min} \implies \text{SOP}$$

- **SkillCompiler**: SOP → SKILL.md + deterministic handler.py
- **ASTSafetyScanner**: static block on `eval()`, `exec()`, `os.system()`, `subprocess.Popen()`

---

## 4. Causal Reasoning (Causal Engine)

Module `src/atlas_memory/causal/` — structural causal models (SCM):

- **What-If Simulation (BFS)**: confidence propagation with multiplicative damping:

$$C(v) = C(u) \cdot \text{confidence}(u \to v)$$

  Risk classification: CRITICAL (≥0.9), HIGH (0.7–0.9), MEDIUM (0.4–0.7), LOW (<0.4)

- **Causal Diffusion Wave**: failure propagation with geometric damping, visited-node register (loop-free):

$$\Delta(v) = \Delta(u) \cdot w(u, v) \cdot \alpha^{\text{depth}}$$

- **CPoF Detection** (Critical Points of Failure): nodes whose failure cascades to disable subsystems:

$$\text{SeverityScore}(u) = \frac{|\text{ReachableNodes}(u)|}{|\text{TotalGraphNodes}|} \cdot \bar{w}_{out}(u)$$

---

## 5. Multi-Agent BFT-CRDT Synchronization
Module `src/atlas_memory/sync/` — disconnection-tolerant, cryptographically secure knowledge sync between agent instances:
- **Delta-State CRDT**: VectorClock with Lamport timestamps for causality preservation
- **Byzantine Fault Tolerant LWW-Set** (`BFTLWWSet`): requires $2f+1$ valid threshold signatures (`ThresholdSigner`) before committing state change
- **Epistemic Peer Reputation** (`EpistemicReputationTracker`): tracks accuracy of incoming updates; peers dropping below threshold are quarantined
- **Secure Gossip Transport**: UDP-based peer discovery with AES-256-GCM authenticated payload encryption
- **LWW Tombstone Garbage Collection**: automated cleanup of tombstoned records past configurable TTL

---

## 6. Vector Quantization & Hardware Acceleration
Module `src/atlas_memory/quantization/` — memory footprint reduction and scan acceleration:
- **RaBitQ Asymptotically Optimal 1-Bit Quantization**:
  - Distance estimation with theoretical error bounds
  - 32× RAM compression ($1536 \to 48$ bytes per vector)
  - SIMD Hamming distance popcount with fallback path
- **Matryoshka Representation Learning (MRL)**: Multi-resolution embedding shortlisting ($D=\{64, 128, 256, 512, 1536\}$) for tiered retrieval filtering
- **MIB Median Binarization**: Bitset generation with fastmath popcount

---

## 7. Active Sensing & External Discrepancy Detection
Module `src/atlas_memory/active/` — external active perception loop:

- **Zero-LLM Telemetry Discrepancy Detection**: when observation exceeds tolerance threshold

$$|V_{obs} - V_{exp}| > \text{tolerance}$$

  the engine generates a `PredictionError` and updates state in the Kùzu graph (supersede=True) — **0 overhead tokens**
- **Shadow Worker**: asynchronous SPO relation extraction after each user turn — 0 ms response latency

---

## 8. Micro-Sidecar IPC Architecture

ATLAS runs as a standalone daemon process to avoid library conflicts:

```
┌─────────────────────────────────────────────┐
│  HERMES AGENT CLI (Python 3.11)             │
│  AtlasMemoryProvider (lightweight IPC client)│
└────────────────────┬────────────────────────┘
                     │  Unix Domain Socket RPC (< 0.05 ms)
                     │  ~/.hermes/atlas.sock
                     ▼
┌─────────────────────────────────────────────┐
│  ATLAS MEMORY DAEMON (Python 3.14 No-GIL)   │
│  • Kùzu Graph DB      • MIB SIMD Quantizer  │
│  • Qdrant Vector      • Delta-CRDT Sync     │
│  • VerifiedKVStore    • Skill Baker         │
└─────────────────────────────────────────────┘
```

- **Protocol**: JSON-RPC 2.0 + binary Arrow IPC + Quantized Vectors
- **Performance**: ping/prefetch over UDS in **0.03–0.05 ms**
- **Graceful Fallback**: daemon absent → in-process mode or fallback to base provider

---

## 9. Hermes Plugin & Memory Contract

ATLAS fully complies with the official Hermes plugin system:

- **Plugin Manifest** (`plugin.yaml` v0.2.0): metadata, entry point, tools (`search_memory`, `commit_observation`)
- **Hermes MemoryProvider ABC**: contract implementation:
  - `prefetch(query, session_id) -> str`
  - `on_session_start() -> str`
  - `on_session_end(messages) -> None`
- **PrefixCacheGuard**: stable, immutable identity and system rules block → **≥ 88–92% KV-cache hit-rate**, lower costs and faster time-to-first-token
