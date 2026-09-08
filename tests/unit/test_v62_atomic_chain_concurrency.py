"""
V62 Concurrency stress test:
Verifies that interleaved async and sync writes/deletes under high concurrency
never fork the SHA-256 Merkle audit chain.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from atlas_memory.l2_semantic.kv_store import VerifiedKVStore


@pytest.mark.asyncio
async def test_v62_concurrent_async_and_sync_writes_never_fork_chain(tmp_path: Path):
    """
    Stress-tests VerifiedKVStore with concurrent async set_state/delete_state
    and sync set_sync/delete_sync across multiple threads and async tasks.
    Verifies chain integrity deep=True (valid=True, broken=0).
    """
    db_path = str(tmp_path / "concurrent_stress.db")
    kv = VerifiedKVStore(db_path=db_path)

    num_async_tasks = 25
    num_sync_threads = 25

    async def async_worker(worker_id: int):
        for i in range(10):
            key = f"async_key_{worker_id}_{i}"
            await kv.set_state(key, f"val_{worker_id}_{i}", confidence=0.85, reason="async_test")
            if i % 3 == 0:
                await kv.delete_state(key)

    def sync_worker(worker_id: int):
        for i in range(10):
            key = f"sync_key_{worker_id}_{i}"
            kv.set_sync(key, f"val_{worker_id}_{i}", confidence=0.90, reason="sync_test")
            if i % 3 == 0:
                kv.delete_sync(key)

    threads = [
        threading.Thread(target=sync_worker, args=(wid,))
        for wid in range(num_sync_threads)
    ]
    for t in threads:
        t.start()

    async_tasks = [
        asyncio.create_task(async_worker(wid))
        for wid in range(num_async_tasks)
    ]

    await asyncio.gather(*async_tasks)
    for t in threads:
        t.join()

    # Now verify chain integrity across all inserted entries
    valid, broken_seq = await kv.verify_audit_log(deep=True)
    assert valid is True, f"Chain integrity broken at seq {broken_seq}!"
    assert broken_seq == 0

    await kv.close()
