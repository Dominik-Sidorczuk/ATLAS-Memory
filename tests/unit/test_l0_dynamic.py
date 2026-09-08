"""
Unit Tests for Layer 0: Dynamic Latent Memory (TTT Layer & Numba Micro-Kernels).
"""
from __future__ import annotations

import numpy as np
import pytest

from atlas_memory.l0_dynamic.ttt_layer import TTTLayer


def test_ttt_initialization():
    layer = TTTLayer(input_dim=32, hidden_dim=16)
    assert layer.input_dim == 32
    assert layer.hidden_dim == 16
    assert layer.step_count == 0


def test_ttt_adapt_step_and_latency():
    layer = TTTLayer(input_dim=64, hidden_dim=32, learning_rate=0.05)
    x = np.random.randn(64).astype(np.float64)

    loss, elapsed_ms = layer.adapt_step(x)
    assert loss >= 0.0
    assert elapsed_ms < 5.0  # Warunek architektury: < 5 ms
    assert layer.step_count == 1

    norm_val = float(np.linalg.norm(layer.w_ttt))
    assert norm_val > 0.0, "Wagi TTT powinny ulec adaptacji gradientowej"


def test_ttt_compress_stream():
    layer = TTTLayer(input_dim=32, hidden_dim=16)
    stream = np.random.randn(10, 32).astype(np.float64).tolist()

    result = layer.compress_stream(stream)
    assert "compressed_latent" in result
    assert len(result["compressed_latent"]) == 16
    assert result["steps_adapted"] == 10
    assert result["w_ttt_norm"] > 0.0


def test_ttt_reset_session():
    layer = TTTLayer(input_dim=32, hidden_dim=16)
    layer.adapt_step(np.random.randn(32).astype(np.float64))
    assert layer.step_count == 1

    layer.reset_session()
    assert layer.step_count == 0
    assert np.all(layer.w_ttt == 0.0)
import time

from atlas_memory.l0_dynamic.ttt_kernels import numba_ttt_adapt_step
from atlas_memory.l1_working.jepa_kernels import numba_jepa_rollout


def test_numba_ttt_microsecond_performance():
    input_dim = 64
    hidden_dim = 32
    w_base = np.random.randn(input_dim, hidden_dim).astype(np.float64) * 0.1
    w_ttt = np.zeros((input_dim, hidden_dim), dtype=np.float64)
    w_recon = np.random.randn(hidden_dim, input_dim).astype(np.float64) * 0.1
    x = np.random.randn(1, input_dim).astype(np.float64)

    # 1. Warmup
    numba_ttt_adapt_step(x, w_base, w_ttt, w_recon, 0.05, 0.001)

    # 2. Pomiar czasu 100 kroków
    start_t = time.perf_counter()
    n_iters = 100
    for _ in range(n_iters):
        loss = numba_ttt_adapt_step(x, w_base, w_ttt, w_recon, 0.05, 0.001)
    total_time_ms = (time.perf_counter() - start_t) * 1000.0
    avg_per_step_ms = total_time_ms / n_iters

    # Numba powinna działać grubo poniżej 0.1 ms (zwykle ~0.01-0.03 ms)
    assert avg_per_step_ms < 0.5
    assert np.linalg.norm(w_ttt) > 0.0
    assert loss >= 0.0


def test_numba_jepa_rollout_compiled():
    state_dim = 32
    action_dim = 16
    n_steps = 5

    s_0 = np.random.randn(1, state_dim).astype(np.float64)
    actions_mat = np.random.randn(n_steps, action_dim).astype(np.float64)
    w_s = np.random.randn(state_dim, state_dim).astype(np.float64) * 0.1
    w_a = np.random.randn(action_dim, state_dim).astype(np.float64) * 0.1
    bias = np.zeros((1, state_dim), dtype=np.float64)
    w_val = np.random.randn(state_dim, 1).astype(np.float64) * 0.1

    states, rewards, uncertainties = numba_jepa_rollout(s_0, actions_mat, w_s, w_a, bias, w_val)

    assert states.shape == (n_steps, state_dim)
    assert rewards.shape == (n_steps,)
    assert uncertainties.shape == (n_steps,)

import asyncio

from atlas_memory.l1_working.jepa_latent import JEPALatentBuffer
from atlas_memory.models import ActionPlan


@pytest.mark.asyncio
async def test_async_ttt_offloading():
    layer = TTTLayer(input_dim=64, hidden_dim=32)
    x = np.random.randn(64).astype(np.float64)

    # Równoległe wywołanie adapt_step_async oraz innego zadania async
    async def dummy_io_task():
        await asyncio.sleep(0.01)
        return "io_done"

    t_ttt = asyncio.create_task(layer.adapt_step_async(x))
    t_io = asyncio.create_task(dummy_io_task())

    loss, elapsed_ms = await t_ttt
    io_res = await t_io

    assert loss >= 0.0
    assert io_res == "io_done"
    assert layer.step_count == 1


@pytest.mark.asyncio
async def test_async_jepa_rollout_offloading():
    buffer = JEPALatentBuffer(state_dim=32, action_dim=16)
    seq = [
        ActionPlan(name="action_a", parameters={"p": 1}),
        ActionPlan(name="action_b", parameters={"p": 2}),
        ActionPlan(name="action_c", parameters={"p": 3}),
    ]

    async def dummy_io_task():
        await asyncio.sleep(0.01)
        return "ok"

    t_rollout = asyncio.create_task(buffer.simulate_rollout_async(seq))
    t_io = asyncio.create_task(dummy_io_task())

    trajectory = await t_rollout
    io_val = await t_io

    assert len(trajectory) == 3
    assert io_val == "ok"

