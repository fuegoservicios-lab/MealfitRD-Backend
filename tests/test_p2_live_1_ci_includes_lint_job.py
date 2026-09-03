"""[P2-LIVE-1 · 2026-05-12] Tests parser-based: el CI gate (P1-LIVE-2)
DEBE incluir un job de lint frontend, aún sea no-bloqueante.

Contexto del gap:
    El cierre P1-LIVE-2 introdujo el CI gate con 3 jobs (backend-tests +
    frontend-tests + frontend-build) pero NO incluyó `npm run lint`. El
    frontend tiene `eslint` configurado (`frontend/package.json` script
    `lint`) y al ejecutarlo localmente reporta 245 errores + 13 warnings
    pre-existentes. Sin el job lint en CI, esos counts pueden crecer sin
    freno en cada PR — el gate no protege contra introducción de nuevos
    errores eslint.

Estrategia del cierre:
    Añadir `frontend-lint` job al workflow con `continue-on-error: true`.
    Esto:
      - EJECUTA `npm run lint` en cada push/PR (visibilidad real).
      - NO bloquea el merge si falla (no paraliza el flujo sobre la
        baseline 245 pre-existente).
      - Genera status check visible en GitHub UI: PRs que aumentan el
        count se ven, aunque el rojo no impida merge.
      - Migration path: tras cleanup incremental que baje el count a 0,
        flippear `continue-on-error: false` para convertirlo en gate
        bloqueante real.

Este test bloquea regresión del cierre:
    - alguien remueve el job `frontend-lint` del workflow.
    - alguien remueve la flag `continue-on-error` (rompería todos los
      merges hasta cleanup).
    - alguien remueve el script `lint` de package.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

P2_LIVE_1_ANCHOR = "P2-LIVE-1"

REPO_ROOT = Path(__file__).resolve().parents[2]
# [G31 · 2026-08-23] El ci.yml de la raíz fue retirado (jamás ejecutó un test:
# el workspace-root no ve los repos hermanos). El lint de CI vive ahora en el
# workflow del REPO FRONTEND, dentro del job `quality`, y ya no es no-bloqueante:
# la baseline de 245 errores se limpió y el gate es el tope `--max-warnings`.
WORKFLOW = REPO_ROOT / "frontend" / ".github" / "workflows" / "ci.yml"
PACKAGE_JSON = REPO_ROOT / "frontend" / "package.json"


def test_a_workflow_has_frontend_lint_job():
    """El workflow del frontend corre eslint en CI (dentro del job quality)."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "eslint" in text, (
        "[P2-LIVE-1] El workflow del frontend dejó de correr eslint: los avisos "
        "pueden crecer sin freno en cada PR."
    )


def test_b_lint_job_runs_npm_lint():
    """El gate de lint es el tope de avisos: `eslint . --max-warnings <N>`."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"eslint \. --max-warnings \d+", text), (
        "[P2-LIVE-1] El paso de eslint perdió el tope `--max-warnings`: sin cap, "
        "eslint con 66 avisos históricos sale 0 y el lint deja de ser un gate."
    )


def test_c_lint_job_is_non_blocking_initially():
    """INVERTIDO a propósito (la deuda se pagó): el paso de eslint ya NO puede
    llevar `continue-on-error` — la baseline de 245 errores se limpió y volver
    a hacerlo no-bloqueante resucitaría el modo de fallo original en silencio."""
    text = WORKFLOW.read_text(encoding="utf-8")
    i = text.find("eslint . --max-warnings")
    assert i > 0, "no encontré el paso de eslint para inspeccionar su bloque"
    ventana = text[max(0, i - 400):i]
    assert "continue-on-error" not in ventana, (
        "[P2-LIVE-1] El paso de eslint volvió a ser no-bloqueante "
        "(continue-on-error): el gate del tope de avisos queda decorativo."
    )


def test_d_package_json_has_lint_script():
    """El script `lint` debe existir en package.json — contrato con el job CI."""
    data = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    scripts = data.get("scripts", {})
    assert "lint" in scripts, (
        "[P2-LIVE-1] frontend/package.json sin script `lint`. El job "
        "frontend-lint del CI invoca `npm run lint` — sin este script el "
        "job falla con 'Missing script'."
    )
    assert "eslint" in scripts["lint"].lower(), (
        f"[P2-LIVE-1] El script `lint` no invoca eslint. Valor actual: "
        f"{scripts['lint']!r}."
    )


def test_e_anchor_marker_present():
    """Anchor P2-LIVE-1 preservado en ci.yml para que un futuro audit
    pueda rastrear el contexto del cierre."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert P2_LIVE_1_ANCHOR in text, (
        "[P2-LIVE-1] Anchor `P2-LIVE-1` removido de ci.yml. Restaurar el "
        "comment header del job frontend-lint."
    )
