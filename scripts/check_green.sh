#!/usr/bin/env bash
# check_green.sh — the test green-gate (audit A1).
#
# Runs the full pytest suite and propagates its exit code, so any RED test
# fails the gate. Wire it into CI and/or a pre-commit hook:
#
#   # .git/hooks/pre-commit  (chmod +x)
#   #!/usr/bin/env bash
#   exec scripts/check_green.sh
#
#   # CI step
#   - run: scripts/check_green.sh
#
# Usage:
#   scripts/check_green.sh                 # run tests/ quietly
#   scripts/check_green.sh -k transport    # forward extra args to pytest
#
# Honors $PYTEST (default: "python -m pytest") so a venv/tox wrapper can
# override the interpreter. Activates ./.venv automatically when present.
set -euo pipefail

# Resolve repo root from this script's location so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Best-effort venv activation (no-op if already active or absent).
if [[ -z "${VIRTUAL_ENV:-}" && -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

PYTEST="${PYTEST:-python -m pytest}"

echo "[check_green] running: ${PYTEST} tests/ -q $*"
# `exec` so the gate's exit status IS pytest's exit status (0 green, non-zero red).
exec ${PYTEST} tests/ -q "$@"
