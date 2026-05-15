#!/usr/bin/env bash
# [P1-LIVE-2 · 2026-05-12] Wrapper local del CI gate (bash).
#
# Reproduce los 3 jobs de .github/workflows/ci.yml en el entorno local:
#   1. pytest del bundle parser-based + funcional (excluyendo e2e).
#   2. vitest del frontend.
#   3. vite build production.
#
# Uso:
#   ./scripts/run_ci.sh
#   SKIP_BACKEND=1 ./scripts/run_ci.sh       # solo frontend
#   SKIP_FRONTEND=1 ./scripts/run_ci.sh      # solo backend
#   SKIP_BUILD=1 ./scripts/run_ci.sh         # tests sin build prod
#
# Exit code: 0 si todos los jobs (no-skipped) pasan, 1 si alguno falla.
#
# Recomendado: invocar antes de cada `git push` (manual o vía hook
# pre-push tras `git init`).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAILED=()

run_step() {
    local label="$1"
    shift
    echo ""
    echo "==> $label"
    if "$@"; then
        echo "    PASS ($label)"
    else
        local rc=$?
        FAILED+=("$label")
        echo "    FAIL ($label) exit=$rc"
    fi
}

run_backend() {
    cd "$REPO_ROOT/backend"
    local py="python"
    if [ -x "venv/bin/python" ]; then py="venv/bin/python";
    elif [ -x "venv/bin/python.exe" ]; then py="venv/bin/python.exe";
    elif [ -x "venv/Scripts/python.exe" ]; then py="venv/Scripts/python.exe";
    fi
    "$py" -m pytest tests/ -v --tb=short -m "not e2e" -x
}

run_frontend_test() {
    cd "$REPO_ROOT/frontend"
    npm test
}

run_frontend_build() {
    cd "$REPO_ROOT/frontend"
    npm run build
}

[ "${SKIP_BACKEND:-0}"  = "1" ] || run_step "Backend pytest"      run_backend
[ "${SKIP_FRONTEND:-0}" = "1" ] || run_step "Frontend vitest"     run_frontend_test
[ "${SKIP_BUILD:-0}"    = "1" ] || run_step "Frontend vite build" run_frontend_build

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All CI jobs PASS"
    exit 0
else
    echo "CI FAIL on: ${FAILED[*]}"
    exit 1
fi
