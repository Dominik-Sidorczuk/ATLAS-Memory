#!/usr/bin/env bash
# doctor:fix — full hygiene pass: LoopDoctor --fix + guaranteed pycache cleanup + re-diagnose.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python scripts/loop_doctor.py --fix

# Force-clean any __pycache__ regenerated during the --fix pass (imports recreate them).
find . -name "__pycache__" -not -path "./.pixi/*" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true

python scripts/loop_doctor.py
