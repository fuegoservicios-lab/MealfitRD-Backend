#!/usr/bin/env bash
# [P1-LIVE-2 · 2026-05-12] Wrapper local del CI gate (bash) — DEPRECADO.
#
# [P3-I18N-RUN-CI-SH-FOSIL · 2026-08-22 · alineado P0-CI-VERDICT · 2026-09-04] Este fichero
# era una copia del wrapper de tres jobs del monorepo muerto y reproducía un CI que ya no
# existe (pedía `backend/` y `frontend/` como subdirectorios). Un wrapper deprecado que corre
# y devuelve 0 es el peor de los dos mundos: quien lo invoque —una costumbre, un alias, un
# runbook viejo— ve verde sin haber ejecutado ni un test. Por eso este muñón FALLA
# ruidosamente y dice cuál es su reemplazo.
#
# Reemplazo: scripts/run_ci.ps1 (PowerShell 7, multiplataforma):
#   pwsh -File scripts/run_ci.ps1                 # pytest + vitest + build
#   pwsh -File scripts/run_ci.ps1 -SkipFrontend   # solo backend
#   pwsh -File scripts/run_ci.ps1 -SkipBackend    # solo frontend
#
# Equivalente sin pwsh, desde la raíz del backend:
#   python -m pytest tests/ -q -m "not e2e" -rs
#   (cd ../frontend && npm test && npm run build)
echo "run_ci.sh está DEPRECADO: usa scripts/run_ci.ps1 (pwsh -File scripts/run_ci.ps1)." >&2
exit 2
