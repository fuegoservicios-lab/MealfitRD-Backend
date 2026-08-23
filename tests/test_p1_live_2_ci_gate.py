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
# [P2-CI-RAIZ-CERO-TESTS · G31 · 2026-08-23] El ci.yml de la RAÍZ ya no existe, a
# propósito: el workspace-root excluye backend/ y frontend/ (repos hermanos con
# remotes propios), así que ese workflow corrió 66/66 veces en ROJO sin ejecutar
# UN test — un gate perpetuamente rojo entrena a ignorar el CI entero. El
# contrato de P1-LIVE-2 (pytest+vitest+build automáticos en push/PR) NO murió:
# vive repartido en los workflows de los repos hermanos, y estos tests ahora
# anclan ESO — la razón de cada assert es la misma, cambió el fichero que la
# satisface.
WORKFLOW_RAIZ_RETIRADO = REPO_ROOT / ".github" / "workflows" / "ci.yml"
WORKFLOW_BACKEND = REPO_ROOT / "backend" / ".github" / "workflows" / "ci.yml"
WORKFLOW_FRONTEND = REPO_ROOT / "frontend" / ".github" / "workflows" / "ci.yml"
PACKAGE_JSON = REPO_ROOT / "frontend" / "package.json"
SCRIPT_PS1 = REPO_ROOT / "scripts" / "run_ci.ps1"
SCRIPT_SH = REPO_ROOT / "scripts" / "run_ci.sh"


def test_a_workflow_file_exists():
    """G31: la raíz NO debe tener ci.yml (jamás pudo ejecutar un test) y los DOS
    repos hermanos SÍ deben tener el suyo — ahí vive ahora el gate real."""
    assert not WORKFLOW_RAIZ_RETIRADO.exists(), (
        "[G31] Reapareció .github/workflows/ci.yml en la raíz. Ese workflow no "
        "puede ver backend/ ni frontend/ (repos hermanos): corrió 66 veces en rojo "
        "sin ejecutar un test. Si hace falta CI cross-repo, va en los hermanos."
    )
    assert WORKFLOW_BACKEND.exists(), "[P1-LIVE-2] falta backend/.github/workflows/ci.yml"
    assert WORKFLOW_FRONTEND.exists(), "[P1-LIVE-2] falta frontend/.github/workflows/ci.yml"


def test_b_workflow_has_three_jobs():
    """Los 3 jobs canónicos de P1-LIVE-2 siguen existiendo, repartidos: pytest en
    el workflow del backend; vitest y build en el del frontend."""
    be = WORKFLOW_BACKEND.read_text(encoding="utf-8")
    fe = WORKFLOW_FRONTEND.read_text(encoding="utf-8")
    assert "pytest" in be, "[P1-LIVE-2] el workflow del backend perdió el job de pytest"
    assert "npm test" in fe, "[P1-LIVE-2] el workflow del frontend perdió vitest (npm test)"
    assert "npm run build" in fe, "[P1-LIVE-2] el workflow del frontend perdió el build"


def test_c_workflow_triggers_on_push_and_pr():
    """Ambos workflows disparan en push + pull_request (gate pre-merge)."""
    for etiqueta, ruta in (("backend", WORKFLOW_BACKEND), ("frontend", WORKFLOW_FRONTEND)):
        text = ruta.read_text(encoding="utf-8")
        assert re.search(r"^\s*push\s*:", text, re.MULTILINE), (
            f"[P1-LIVE-2] `on.push` falta en el ci.yml de {etiqueta}"
        )
        assert re.search(r"^\s*pull_request\s*:", text, re.MULTILINE), (
            f"[P1-LIVE-2] `on.pull_request` falta en el ci.yml de {etiqueta}"
        )


def test_d_workflow_runs_pytest_with_correct_marker():
    """El pytest del CI backend filtra `-m "not e2e"` (los e2e piden DB viva)."""
    text = WORKFLOW_BACKEND.read_text(encoding="utf-8")
    assert "pytest" in text, "[P1-LIVE-2] el ci.yml del backend no invoca pytest."
    assert "not e2e" in text, (
        "[P1-LIVE-2] El pytest del CI backend perdió el filtro `-m \"not e2e\"`: "
        "los e2e intentarían conectar a la DB desde el runner y fallarían siempre."
    )


def test_e_workflow_runs_vitest_and_build():
    """El workflow del frontend invoca `npm test` y `npm run build`."""
    text = WORKFLOW_FRONTEND.read_text(encoding="utf-8")
    assert "npm test" in text, "[P1-LIVE-2] el ci.yml del frontend no invoca `npm test`."
    assert "npm run build" in text, (
        "[P1-LIVE-2] el ci.yml del frontend no invoca `npm run build` — el build es "
        "el único gate de tree-shaking/imports/bundle-size que vitest no atrapa."
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


def test_h_local_wrapper_vivo_corre_los_tres_pasos():
    """El wrapper VIVO debe cubrir los 3 mismos pasos del CI: backend pytest,
    frontend vitest, frontend build.

    [P3-CI-WRAPPER-FOSIL-GUARD · 2026-08-22] Este caso exigía los tres pasos a
    los DOS wrappers. `P3-I18N-RUN-CI-SH-FOSIL` (2026-08-21) convirtió
    `run_ci.sh` en un muñón de deprecación que ya no corre nada y no reconvirtió
    este guard, así que llevaba rojo desde entonces: el guard seguía exigiéndole
    trabajo a un fichero cuyo trabajo es no hacer nada.

    La paridad que este fichero defiende sigue viva — sólo que ahora la sostiene
    un único wrapper. El muñón tiene su propio caso abajo, que es el que impide
    que «deprecado» degenere en «silenciosamente inútil».
    """
    text = SCRIPT_PS1.read_text(encoding="utf-8")
    assert "pytest" in text, (
        f"[P1-LIVE-2] {SCRIPT_PS1.name} no invoca pytest — paridad con CI rota."
    )
    assert "npm test" in text, (
        f"[P1-LIVE-2] {SCRIPT_PS1.name} no invoca `npm test`."
    )
    assert "npm run build" in text, (
        f"[P1-LIVE-2] {SCRIPT_PS1.name} no invoca `npm run build`."
    )


def test_h2_el_wrapper_fosil_falla_ruidosamente():
    """El muñón no puede salir con 0.

    Un wrapper deprecado que no corre nada y devuelve éxito es la peor de las
    dos opciones: quien lo invoque —una costumbre, un alias, un runbook viejo—
    verá verde sin haber ejecutado ni un test. Es el mismo falso verde que este
    repo ha pagado con un marcador deseleccionado y con un filtro que no casa
    con nada. Deprecar obliga a fallar RUIDOSAMENTE y a decir cuál es el
    reemplazo.
    """
    text = SCRIPT_SH.read_text(encoding="utf-8")
    assert re.search(r"exit\s+[1-9]", text), (
        f"[P3-CI-WRAPPER-FOSIL-GUARD] {SCRIPT_SH.name} está deprecado y sale "
        f"con 0: quien lo invoque verá verde sin haber corrido nada."
    )
    assert "run_ci.ps1" in text, (
        f"[P3-CI-WRAPPER-FOSIL-GUARD] {SCRIPT_SH.name} no dice cuál es su "
        f"reemplazo. Un muñón que sólo dice «no» manda a leer el git log."
    )
