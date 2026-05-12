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
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PACKAGE_JSON = REPO_ROOT / "frontend" / "package.json"


def test_a_workflow_has_frontend_lint_job():
    """El workflow debe definir el job `frontend-lint`."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"^\s*frontend-lint\s*:", text, re.MULTILINE), (
        "[P2-LIVE-1] Job `frontend-lint` no encontrado en .github/workflows/"
        "ci.yml. Sin este job, eslint nunca corre en CI y los 245 errores "
        "pre-existentes pueden crecer sin freno en cada PR."
    )


def test_b_lint_job_runs_npm_lint():
    """El job `frontend-lint` debe invocar `npm run lint`."""
    text = WORKFLOW.read_text(encoding="utf-8")
    # Localizar el bloque del job hasta el próximo job o EOF.
    # Aislamiento robusto: matchear desde `frontend-lint:` (indent 2 = top
    # level dentro de jobs) hasta el próximo job al mismo indent o EOF. Los
    # sub-keys dentro del job (steps, name, runs-on) tienen indent ≥4, así
    # que `^  \S` solo matchea otro job o EOF.
    m = re.search(
        r"^  frontend-lint\s*:.*?(?=^  \S|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert m, "[P2-LIVE-1] No pude aislar el bloque del job frontend-lint."
    body = m.group(0)
    assert "npm run lint" in body, (
        "[P2-LIVE-1] El job frontend-lint no invoca `npm run lint`. El "
        "comando es el contrato con package.json scripts.lint — sin él el "
        "job no ejecuta eslint."
    )


def test_c_lint_job_is_non_blocking_initially():
    """El job debe tener `continue-on-error: true` mientras la baseline
    de 245 errores no se haya limpiado. Sin esta flag, todos los merges
    se bloquean. Tras cleanup (count=0), flippear a false."""
    text = WORKFLOW.read_text(encoding="utf-8")
    # Aislamiento robusto: matchear desde `frontend-lint:` (indent 2 = top
    # level dentro de jobs) hasta el próximo job al mismo indent o EOF. Los
    # sub-keys dentro del job (steps, name, runs-on) tienen indent ≥4, así
    # que `^  \S` solo matchea otro job o EOF.
    m = re.search(
        r"^  frontend-lint\s*:.*?(?=^  \S|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert m, "[P2-LIVE-1] No pude aislar el bloque del job frontend-lint."
    body = m.group(0)
    assert re.search(r"continue-on-error\s*:\s*true", body), (
        "[P2-LIVE-1] El job frontend-lint debe tener `continue-on-error: "
        "true` hasta que la baseline de 245 errores eslint se haya limpiado. "
        "Sin esta flag, todos los merges quedarían bloqueados sobre la "
        "deuda técnica pre-existente. Tras cleanup, flippear a `false` y "
        "actualizar este test para que enforce `continue-on-error: false`."
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
