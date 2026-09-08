# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability in ATLAS, **do not** create a public issue on GitHub. Instead, report it privately by emailing the project maintainers. You will receive an acknowledgment within 48 hours, followed by a detailed remediation plan.

Please provide:

- A description of the vulnerability (type, potential impact, reproduction steps)
- The software version where the issue occurs
- Any suggested fixes

## Project Security Principles

1. **Hermes core is inviolable** — the ATLAS plugin does not modify `~/.hermes/hermes-agent/`.
2. **Self-contained plugin** — all logic runs in an isolated Pixi environment; communication with Hermes happens over UDS IPC.
3. **No `torch` in runtime** — minimizes attack surface in the production layer.
4. **Signature quorum** — BFT-CRDT synchronization requires `2f+1` signatures for operations.
5. **Deterministic hashing** — `zlib.crc32` instead of random `hash()` to avoid unpredictable behavior.

## Supported Versions

| Version | Supported |
|---------|-----------|
| main | ✅ |

## Incident Reporting

Please report vulnerabilities within 24 hours of discovery, following responsible disclosure practices.
