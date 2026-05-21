#!/usr/bin/env bash
# [I6 / P3-COVERAGE-HEATMAP · 2026-05-20] Wrapper para generar el coverage
# heatmap de los 770 tests pytest del backend.
#
# Por qué existe:
#   El audit `docs/gaps-audit-2026-05.md` I6 flageó que el repo tiene
#   ~770 tests pero CERO observabilidad de cobertura. No sabes qué
#   zonas calientes (graph_orchestrator 14k líneas, cron_tasks 27k)
#   tienen 80% test coverage vs 5%. Sin heatmap, decisiones de
#   "qué refactor es seguro" se toman a ciegas.
#
# Diseño minimalista (consistente con MVP <100 MAU + 1 dev):
#   - NO añadir pytest-cov a requirements.txt prod (es solo dev/local).
#   - NO subir el reporte a CI por ahora (sube el cost de cada run
#     ~20-40s; reabrir cuando crucemos 1k MAU o tengamos 2do dev).
#   - Output a `htmlcov/index.html` (gitignored) + summary terminal.
#
# Uso:
#   ./scripts/run_coverage.sh                # html + terminal summary
#   ./scripts/run_coverage.sh --term         # solo terminal (sin html)
#   ./scripts/run_coverage.sh -m "not e2e"   # filtros pytest extra
#
# Tras correr: abrir `htmlcov/index.html` en el browser para el heatmap
# por archivo. Click un archivo para ver line-by-line coverage.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Detección de pytest-cov. Si no instalado, instalarlo (best-effort).
if ! python -c "import pytest_cov" 2>/dev/null; then
    echo "==> pytest-cov no instalado, instalando..."
    pip install --quiet pytest-cov || {
        echo "FALLO: no pude instalar pytest-cov. Intenta:"
        echo "  pip install pytest-cov"
        exit 1
    }
fi

# Default: html + terminal. Toggleable con --term.
COV_REPORT_FLAGS=("--cov-report=term-missing:skip-covered" "--cov-report=html:htmlcov")
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --term)
            # Solo terminal, omitir html (más rápido).
            COV_REPORT_FLAGS=("--cov-report=term-missing")
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

# Filtro `not e2e` por default (consistente con CI). Override pasando
# `-m "<expr>"` como extra arg.
if ! printf '%s\n' "${EXTRA_ARGS[@]}" | grep -q '^-m'; then
    EXTRA_ARGS+=("-m" "not e2e")
fi

echo "==> pytest --cov=. ${COV_REPORT_FLAGS[*]} ${EXTRA_ARGS[*]}"
pytest \
    --cov=. \
    --cov-config=.coveragerc \
    "${COV_REPORT_FLAGS[@]}" \
    "${EXTRA_ARGS[@]}"

rc=$?
echo ""
if [ $rc -eq 0 ]; then
    if [[ " ${COV_REPORT_FLAGS[*]} " == *html* ]]; then
        echo "==> Coverage report HTML: file://$REPO_ROOT/htmlcov/index.html"
    fi
fi
exit $rc
