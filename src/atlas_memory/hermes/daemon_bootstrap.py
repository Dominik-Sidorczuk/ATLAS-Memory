"""Daemon Process Bootstrap & UDS Retry Transport for Hermes Provider (V106).

Encapsulates Unix Domain Socket liveness probing, stale socket cleanup,
detached background spawning of `scripts/atlas_daemon_launcher.py`, and
JSON-RPC 2.0 1-retry transport backoff.
"""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from atlas_memory.ipc.uds_client import AtlasTransportError, send_uds_request_sync

logger = logging.getLogger(__name__)


def _get_daemon_paths(sock_path: Path) -> tuple[list[Path], list[Path]]:
    """Zwraca dynamiczne listy kandydatów na pliki PID oraz LOCK dla danego gniazda."""
    pid_cands: list[Path] = []
    lock_cands: list[Path] = []

    # 1. Ścieżki ze zmiennych środowiskowych
    env_pid = os.environ.get("ATLAS_PID_PATH")
    if env_pid:
        pid_cands.append(Path(env_pid).expanduser())
    env_lock = os.environ.get("ATLAS_LOCK_PATH")
    if env_lock:
        lock_cands.append(Path(env_lock).expanduser())

    # 2. Ścieżki wywiedzione z lokalizacji sock_path
    pid_cands.append(sock_path.with_suffix(".pid"))
    pid_cands.append(sock_path.parent / "atlas.pid")
    lock_cands.append(sock_path.with_suffix(".lock"))
    lock_cands.append(sock_path.parent / "atlas.lock")

    # 3. Ścieżki z hermes_home / standardowego ~/.hermes (tylko gdy sock_path leży w katalogu hermes)
    default_home = (Path.home() / ".hermes").resolve()
    target_dir = sock_path.parent.resolve()
    home = os.environ.get("HERMES_HOME")
    h_dir = Path(home).expanduser().resolve() if home else None

    if (h_dir and (target_dir == h_dir or target_dir == (h_dir / "atlas"))) or target_dir == default_home or target_dir == (default_home / "atlas"):
        if h_dir:
            pid_cands.append(h_dir / "atlas.pid")
            pid_cands.append(h_dir / "atlas" / "atlas.pid")
            lock_cands.append(h_dir / "atlas.lock")
            lock_cands.append(h_dir / "atlas" / "atlas.lock")
        pid_cands.append(default_home / "atlas.pid")
        pid_cands.append(default_home / "atlas" / "atlas.pid")
        lock_cands.append(default_home / "atlas.lock")
        lock_cands.append(default_home / "atlas" / "atlas.lock")

    # Deduplikacja z zachowaniem kolejności
    seen_pids: set[str] = set()
    uniq_pids: list[Path] = []
    for p in pid_cands:
        s = str(p.resolve() if p.exists() else p)
        if s not in seen_pids:
            seen_pids.add(s)
            uniq_pids.append(p)

    seen_locks: set[str] = set()
    uniq_locks: list[Path] = []
    for lk in lock_cands:
        s = str(lk.resolve() if lk.exists() else lk)
        if s not in seen_locks:
            seen_locks.add(s)
            uniq_locks.append(lk)

    return uniq_pids, uniq_locks


def _is_daemon_pid_alive(sock_path: Optional[Path] = None) -> bool:
    """Weryfikuje, czy proces demona ATLAS żyje w tablicy procesów OS lub trzyma blokadę."""
    target_sock = (
        sock_path
        if sock_path is not None
        else Path(os.environ.get("ATLAS_SOCKET_PATH", str(Path.home() / ".hermes" / "atlas.sock"))).expanduser()
    )
    pid_cands, lock_cands = _get_daemon_paths(target_sock)

    for cand in pid_cands:
        if cand.exists():
            try:
                content = cand.read_text().strip()
                if not content:
                    continue
                pid = int(content)
                os.kill(pid, 0)
                return True
            except OSError as err:
                if err.errno == errno.ESRCH:
                    continue
                if err.errno == errno.EPERM:
                    return True
            except Exception as exc:
                logger.debug("Daemon pid candidate %s check failed: %s", cand, exc)

    for lock_file in lock_cands:
        if lock_file.exists():
            try:
                fd = os.open(str(lock_file), os.O_RDWR)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except (OSError, BlockingIOError):
                    # Blokada jest trzymana przez aktywny proces -> demon żyje
                    return True
                finally:
                    os.close(fd)
            except Exception:
                pass

    return False


def _can_safely_unlink_socket(sock_path: Path) -> bool:
    """Zero Socket Murder: usunięcie gniazda dozwolone WYŁĄCZNIE, gdy PID jest martwy i brak blokady."""
    if _is_daemon_pid_alive(sock_path):
        return False

    _, lock_cands = _get_daemon_paths(sock_path)
    for lock_file in lock_cands:
        if lock_file.exists():
            try:
                fd = os.open(str(lock_file), os.O_RDWR)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except (OSError, BlockingIOError):
                    # Plik blokady jest trzymany przez inny proces -> nie wolno usuwać
                    return False
                finally:
                    os.close(fd)
            except Exception:
                pass

    return True


def ensure_daemon_process_running(
    provider: Any,
    timeout_s: float = 15.0,
    send_rpc_fn: Optional[Callable[..., Any]] = send_uds_request_sync,
) -> bool:
    """Zapewnia działanie demona UDS ATLAS z transakcyjną blokadą współbieżności i Zero Socket Murder."""
    sock_str = (
        getattr(provider, "_socket_path", None)
        or (str(provider) if isinstance(provider, (str, Path)) else None)
        or os.environ.get("ATLAS_SOCKET_PATH", str(Path.home() / ".hermes" / "atlas.sock"))
    )
    sock_path = Path(sock_str).expanduser()
    if hasattr(provider, "_socket_path"):
        provider._socket_path = str(sock_path)

    initial_sock_mtime = None
    if sock_path.exists():
        try:
            initial_sock_mtime = sock_path.stat().st_mtime_ns
        except OSError:
            initial_sock_mtime = None

    # 1. Szybka ścieżka (Fast Path): gniazdo istnieje i demon odpowiada na RPC
    initial_rpc_failed = False
    if sock_path.exists() and send_rpc_fn is not None:
        try:
            stats = send_rpc_fn(str(sock_path), "get_stats", params={})
            if stats is not None:
                return True
            initial_rpc_failed = True
        except Exception as exc:
            initial_rpc_failed = True
            if _is_daemon_pid_alive(sock_path):
                logger.debug("Daemon socket %s busy/starting (%s), keeping live socket", sock_path, exc)
                return True
            logger.debug("Health check get_stats on existing socket %s failed: %s", sock_path, exc)

    # Guard against spawning a live production daemon on ~/.hermes/atlas during unmocked pytest runs (Rule #2)
    is_popen_mocked = hasattr(subprocess.Popen, "call_count") or hasattr(subprocess.Popen, "assert_called")
    prod_sock = (Path.home() / ".hermes" / "atlas.sock").resolve()
    if (
        "PYTEST_CURRENT_TEST" in os.environ
        and not is_popen_mocked
        and (getattr(provider, "_orchestrator", None) is not None or sock_path.resolve() == prod_sock)
    ):
        logger.warning("daemon spawn skipped in unmocked pytest context, falling back to in-process")
        return False

    # 2. Transakcyjna blokada uruchomieniowa (Atomic Process Lock via fcntl.flock)
    bootstrap_lock_path = sock_path.with_name(f".{sock_path.stem}.bootstrap.lock")
    try:
        bootstrap_lock_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    lock_file = None
    try:
        lock_file = open(bootstrap_lock_path, "a+")
        # Czekaj na blokadę wyłączną
        deadline = time.monotonic() + max(1.0, float(timeout_s))
        acquired = False
        while time.monotonic() < deadline:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except (OSError, BlockingIOError):
                time.sleep(0.05)

        if not acquired:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                acquired = True
            except Exception:
                pass

        # POD BLOKADĄ: Sprawdź ponownie, czy inny proces nie uruchomił już demona
        current_mtime = None
        if sock_path.exists():
            try:
                current_mtime = sock_path.stat().st_mtime_ns
            except OSError:
                current_mtime = None

        socket_changed = (initial_sock_mtime is None) or (current_mtime != initial_sock_mtime)

        if sock_path.exists() and (socket_changed or not initial_rpc_failed or _is_daemon_pid_alive(sock_path)) and send_rpc_fn is not None:
            try:
                stats = send_rpc_fn(str(sock_path), "get_stats", params={})
                if stats is not None:
                    return True
            except Exception:
                pass

        # Weryfikacja martwego gniazda (Zero Socket Murder)
        if sock_path.exists():
            if _can_safely_unlink_socket(sock_path):
                logger.warning("Stale daemon socket detected at %s; removing before respawn", sock_path)
                try:
                    sock_path.unlink()
                except OSError as unlink_err:
                    logger.warning("Failed to unlink stale socket %s: %s", sock_path, unlink_err)
            else:
                logger.debug("Socket %s exists and daemon PID is alive/locked, waiting for readiness", sock_path)

        # 3. Uruchomienie demona via subprocess.Popen
        repo_root = Path(__file__).resolve().parents[3]
        spawn_env = {
            **os.environ,
            "PYTHONPATH": str(repo_root / "src"),
            "ATLAS_SOCKET_PATH": str(sock_path),
        }
        try:
            if sock_path.resolve() != prod_sock:
                sock_path.parent.mkdir(parents=True, exist_ok=True)
                spawn_env.setdefault("ATLAS_DIR", str(sock_path.parent))
                spawn_env.setdefault("ATLAS_PID_PATH", str(sock_path.with_suffix(".pid")))
                spawn_env.setdefault("ATLAS_LOCK_PATH", str(sock_path.with_suffix(".lock")))
        except OSError as exc:
            logger.debug("Non-prod socket dir creation note: %s", exc)

        custom_py = os.environ.get("ATLAS_PYTHON")
        pixi_py = repo_root / ".pixi" / "envs" / "default" / "bin" / "python"
        if custom_py and Path(custom_py).exists():
            daemon_py = str(custom_py)
        elif pixi_py.exists():
            daemon_py = str(pixi_py)
        else:
            daemon_py = sys.executable

        launcher_path = repo_root / "scripts" / "atlas_daemon_launcher.py"
        launcher_cmd = [daemon_py, str(launcher_path)] if launcher_path.exists() else [daemon_py, "-m", "atlas_memory.server"]

        try:
            subprocess.Popen(
                launcher_cmd,
                cwd=str(repo_root) if repo_root.exists() else None,
                env=spawn_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            logger.warning("daemon spawn failed (%s), falling back to in-process", exc)
            return False

        # 4. Polling do momentu gotowości gniazda i RPC get_stats
        poll_deadline = time.monotonic() + max(0.05, float(timeout_s))
        while time.monotonic() < poll_deadline:
            if sock_path.exists():
                if send_rpc_fn is not None:
                    try:
                        resp = send_rpc_fn(str(sock_path), "get_stats", params={})
                        if resp is not None:
                            return True
                    except Exception as exc:
                        logger.debug("Poll get_stats attempt failed: %s", exc)
                else:
                    return True
            time.sleep(0.05)

        logger.warning("daemon spawn timeout, falling back to in-process")
        return False
    finally:
        if lock_file is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            except Exception:
                pass


def call_uds_with_retry(
    provider: Any,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    send_rpc_fn: Optional[Callable[..., Any]] = send_uds_request_sync,
) -> Optional[Any]:
    """Executes sync JSON-RPC 2.0 call over UDS with auto-bootstrap and 1-retry backoff (V106 FIX-B)."""
    sock_str = provider._socket_path or os.environ.get(
        "ATLAS_SOCKET_PATH", str(Path.home() / ".hermes" / "atlas.sock")
    )
    sock = str(Path(sock_str).expanduser())
    provider._socket_path = sock

    if not Path(sock).exists():
        if not provider._ensure_daemon_running():
            logger.warning("daemon unavailable, in-process fallback")
            return None

    params_dict = params or {}
    if send_rpc_fn is not None:
        try:
            return send_rpc_fn(sock, method, params=params_dict)
        except AtlasTransportError as exc:
            logger.warning(
                "UDS sync call %s failed with transport error (%s); attempting daemon respawn & retry",
                method,
                exc,
            )
            if provider._ensure_daemon_running():
                time.sleep(1.0)
                try:
                    return send_rpc_fn(sock, method, params=params_dict)
                except AtlasTransportError as retry_exc:
                    logger.warning(
                        "UDS sync call %s retry failed (%s), falling back to in-process",
                        method,
                        retry_exc,
                    )
                    return None
            logger.warning("daemon unavailable after transport error on %s, in-process fallback", method)
            return None

    client = provider.get_client()
    if client is None or not client.socket_path.exists():
        return None

    import asyncio

    try:
        return asyncio.run(client.call(method, params_dict))
    except Exception as exc:
        logger.debug("UDS call %s failed, falling back to in-process: %s", method, exc)
        return None



def hot_attach_provider_to_live_agents(socket_path: str) -> int:
    """Attaches AtlasMemoryProvider to any live in-memory Hermes agent instances in the current process."""
    attached = 0
    try:
        import gc
        import importlib
        import sys
        if "plugins.user.atlas" in sys.modules:
            try:
                importlib.reload(sys.modules["plugins.user.atlas"])
            except Exception as exc:
                logger.debug("Plugin reload note: %s", exc)
        from agent.memory_manager import MemoryManager, inject_memory_provider_tools

        from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider

        provider = AtlasMemoryProvider(socket_path=socket_path)
        for obj in gc.get_objects():
            try:
                d = getattr(obj, "__dict__", None)
                if isinstance(d, dict) and ("_memory_manager" in d or hasattr(obj, "_build_system_prompt")):
                    mm = d.get("_memory_manager")
                    has_atlas = False
                    if mm is not None and hasattr(mm, "providers"):
                        has_atlas = any(getattr(p, "name", "") == "atlas" for p in mm.providers)
                    if not has_atlas:
                        if mm is None:
                            mm = MemoryManager()
                            obj._memory_manager = mm
                        mm.add_provider(provider)
                        sid = getattr(obj, "session_id", None) or "hermes_default"
                        mm.initialize_all(session_id=sid)
                        try:
                            inject_memory_provider_tools(obj)
                        except Exception as exc:
                            logger.debug("Tools injection note: %s", exc)
                        attached += 1
                        logging.getLogger("agent.memory_manager").info(
                            "Memory provider 'atlas' registered (10 tools) [session=%s]", sid
                        )
                        logging.getLogger("run_agent").info(
                            "Memory provider 'atlas' activated [session=%s]", sid
                        )
            except Exception:
                continue
    except Exception as exc:
        logger.debug("hot_attach_provider_to_live_agents note: %s", exc)
    return attached


def ensure_daemon_running(target: Any = None, timeout_s: float = 8.0) -> bool:
    """Convenience wrapper accepting either an AtlasMemoryProvider instance or a socket path."""
    if hasattr(target, "_socket_path"):
        return ensure_daemon_process_running(target, timeout_s=timeout_s)
    from types import SimpleNamespace
    sock_str = str(target) if target is not None else os.environ.get(
        "ATLAS_SOCKET_PATH", str(Path.home() / ".hermes" / "atlas.sock")
    )
    dummy = SimpleNamespace(_socket_path=sock_str, _orchestrator=None)
    ok = ensure_daemon_process_running(dummy, timeout_s=timeout_s)
    hot_attach_provider_to_live_agents(sock_str)
    return ok
