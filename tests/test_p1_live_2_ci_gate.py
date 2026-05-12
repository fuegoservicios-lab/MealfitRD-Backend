"""[P1-LIVE-2 · 2026-05-12] Tests parser-based para el CI gate.

Contexto:
    Hasta P1-LIVE-2, el repo NO tenía un gate automático de pre-merge:
    los ~120+ tests parser-based + Vitest del frontend solo corrían
    cuando el desarrollador los invocaba manualmente. Regresiones
    laterales (e.g. el incidente P1-SCHEDULER-1 donde una línea reformateada
    rompió 5 tests SOP) no se detectaban hasta el siguiente audit manual.

    El cierre P1-LIVE-2 introduce:
      1) `.github/workflows/ci.yml` — 3 jobs (backend-tests/frontend-tests/
         frontend-build) que disparan en push + pull_request a main.
      2) `scripts/run_ci.ps1` y `scripts/run_ci.sh` — wrappers locales
         cross-platform que reproducen los mismos 3 jobs antes de push.
      3) `frontend/package.json` script `"test": "vitest run"` — el CI lo
         invoca; sin este script `npm test` falla con "Missing script".

Este test escanea cada artefacto del cierre y bloquea regresión:
    - alguien remueve el workflow file (deshabilita CI silenciosamente).
    - alguien remueve el script `test` (rompe el job frontend-tests).
    - alguien remueve los wrappers locales (pierde paridad CI↔local).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

P1_LIVE_2_ANCHOR = "P1-LIVE-2"

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PACKAGE_JSON = REPO_ROOT / "frontend" / "package.json"
SCRIPT_PS1 = REPO_ROOT / "scripts" / "run_ci.ps1"
SCRIPT_SH = REPO_ROOT / "scripts" / "run_ci.sh"


def test_a_workflow_file_exists():
    """`.github/workflows/ci.yml` debe existir con anchor P1-LIVE-2."""
    assert WORKFLOW.exists(), (
        "[P1-LIVE-2] `.github/workflows/ci.yml` no existe. Restaurar para "
        "que CI corra pytest+vitest+build automáticamente en push/PR. Sin "
        "este workflow, regresiones laterales se detectan solo en audits "
        "manuales — el modo de fallo que P1-LIVE-2 cierra."
    )
    text = WORKFLOW.read_text(encoding="utf-8")
    assert P1_LIVE_2_ANCHOR in text, (
        "[P1-LIVE-2] Anchor `P1-LIVE-2` removido de ci.yml. Restaurar el "
        "comentario para que un futuro audit pueda rastrear el contexto."
    )


def test_b_workflow_has_three_jobs():
    """ci.yml debe definir los 3 jobs canónicos: backend-tests,
    frontend-tests, frontend-build. Cualquiera faltante deja un gap en
    la red de seguridad."""
    text = WORKFLOW.read_text(encoding="utf-8")
    for job in ("backend-tests", "frontend-tests", "frontend-build"):
        assert re.search(rf"^\s*{re.escape(job)}\s*:", text, re.MULTILINE), (
            f"[P1-LIVE-2] Job `{job}` no encontrado en ci.yml. Los 3 jobs "
            f"(backend-tests, frontend-tests, frontend-build) son requeridos: "
            f"backend-tests cubre el bundle parser-based, frontend-tests cubre "
            f"Vitest, frontend-build valida tree-shaking + import resolution + "
            f"bundle size en prod."
        )


def test_c_workflow_triggers_on_push_and_pr():
    """ci.yml debe disparar en `push` + `pull_request` (gate pre-merge)."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"^\s*push\s*:", text, re.MULTILINE), (
        "[P1-LIVE-2] `on.push` trigger missing en ci.yml. Sin push trigger, "
        "el CI no corre en branches feature → bugs solo se detectan en PR "
        "(demasiado tarde si la feature toma 1 semana)."
    )
    assert re.search(r"^\s*pull_request\s*:", text, re.MULTILINE), (
        "[P1-LIVE-2] `on.pull_request` trigger missing. Sin PR trigger, "
        "main puede recibir merges sin verificación."
    )


def test_d_workflow_runs_pytest_with_correct_marker():
    """El job backend-tests debe ejecutar pytest con el filtro `not e2e`
    (los tests E2E necesitan DB live, no corren en CI runners)."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "pytest" in text, "[P1-LIVE-2] ci.yml no invoca pytest."
    assert '"not e2e"' in text or "'not e2e'" in text or "not e2e" in text, (
        "[P1-LIVE-2] El job backend-tests no filtra `-m \"not e2e\"`. Sin "
        "el filtro, los tests E2E intentan conectarse a Supabase y fallan "
        "en CI runners (sin DB). Mantener el filtro para que CI corra solo "
        "los parser-based + funcionales sin DB."
    )


def test_e_workflow_runs_vitest_and_build():
    """frontend-tests debe invocar `npm test` y frontend-build `npm run build`."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "npm test" in text, (
        "[P1-LIVE-2] ci.yml no invoca `npm test` en frontend-tests. Sin "
        "Vitest en CI, regresiones en helpers frontend (renderCoherenceWarnings, "
        "etc.) no se detectan."
    )
    assert "npm run build" in text, (
        "[P1-LIVE-2] ci.yml no invoca `npm run build`. El build es el único "
        "gate que cacha errores de tree-shaking + import resolution + bundle "
        "size que Vitest no atrapa."
    )


def test_f_package_json_has_test_script():
    """frontend/package.json debe tener script `test` invocando vitest."""
    assert PACKAGE_JSON.exists(), "[P1-LIVE-2] frontend/package.json missing."
    data = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    scripts = data.get("scripts", {})
    assert "test" in scripts, (
        "[P1-LIVE-2] frontend/package.json sin script `test`. El job "
        "frontend-tests del CI invoca `npm test` — sin este script, el job "
        "falla con 'Missing script'. Restaurar `\"test\": \"vitest run\"`."
    )
    assert "vitest" in scripts["test"].lower(), (
        f"[P1-LIVE-2] El script `test` no invoca vitest. Valor actual: "
        f"{scripts['test']!r}. Esperado algo como `vitest run` (modo "
        f"no-watch para CI)."
    )


def test_g_local_wrappers_exist():
    """Los wrappers locales (PS1 + SH) deben existir para paridad CI↔local."""
    assert SCRIPT_PS1.exists(), (
        "[P1-LIVE-2] scripts/run_ci.ps1 missing. Wrapper PowerShell permite "
        "a desarrolladores Windows correr el mismo CI localmente antes de "
        "push, evitando el ciclo lento push→CI-rojo→fix→push."
    )
    assert SCRIPT_SH.exists(), (
        "[P1-LIVE-2] scripts/run_ci.sh missing. Wrapper bash para Linux/macOS."
    )
    for f in (SCRIPT_PS1, SCRIPT_SH):
        text = f.read_text(encoding="utf-8")
        assert P1_LIVE_2_ANCHOR in text, (
            f"[P1-LIVE-2] Anchor removido de {f.name}. Restaurar comment "
            f"header para que un futuro refactor entienda el propósito."
        )


def test_h_local_wrappers_run_three_steps():
    """Wrappers locales deben cubrir los 3 mismos pasos del CI: backend
    pytest, frontend vitest, frontend build."""
    for f in (SCRIPT_PS1, SCRIPT_SH):
        text = f.read_text(encoding="utf-8")
        assert "pytest" in text, (
            f"[P1-LIVE-2] {f.name} no invoca pytest — paridad con CI rota."
        )
        # `npm test` y `npm run build` deben aparecer ambos.
        assert "npm test" in text, (
            f"[P1-LIVE-2] {f.name} no invoca `npm test`."
        )
        assert "npm run build" in text, (
            f"[P1-LIVE-2] {f.name} no invoca `npm run build`."
        )
