"""[P3-LIVE-1 · 2026-05-12] Tests parser-based: la documentación del CI
gate existe y referencia los contratos críticos.

Contexto:
    Los cierres P1-LIVE-2 (CI workflow) y P2-LIVE-1 (lint no-bloqueante)
    introdujeron artefactos cuyo uso correcto no es obvio sin lectura del
    código (`continue-on-error: true` es invariante load-bearing; `-m
    "not e2e"` es load-bearing; los wrappers locales tienen flags skip
    no-obvios). Sin docs, un nuevo desarrollador no sabe:
      - Qué wrapper invocar antes de push.
      - Cómo activar branch protection en GitHub.
      - Por qué `frontend-lint` no bloquea y cuándo flippearlo.

P3-LIVE-1 cierra el gap creando:
    - scripts/README.md — uso de run_ci.ps1/sh + cuándo invocar + hook
      pre-push opcional.
    - .github/README.md — workflow jobs + activación branch protection
      + SOPs operacionales.

Este test bloquea regresión documental:
    - Alguien remueve un README → onboarding silenciosamente regresa.
    - Alguien remueve referencia al invariante `continue-on-error: true`
      en docs → futuro reader no entiende por qué el lint no bloquea y
      lo "arregla" rompiendo merges.
"""

from __future__ import annotations

import re
from pathlib import Path

P3_LIVE_1_ANCHOR = "P3-LIVE-1"

# [P0-CI-VERDICT · 2026-09-04] Se leen las copias VERSIONADAS del backend (antes: la raíz del
# workspace, que ningún repo versiona → el guard no corría en CI y su README describía un
# workflow monorepo que murió el 2026-08-22).
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
SCRIPTS_README = BACKEND / "scripts" / "README.md"
GITHUB_README = BACKEND / ".github" / "README.md"


def test_a_scripts_readme_exists_with_anchor():
    """`scripts/README.md` debe existir con anchor P3-LIVE-1."""
    assert SCRIPTS_README.exists(), (
        "[P3-LIVE-1] `scripts/README.md` missing. Documenta uso de los "
        "wrappers locales run_ci.ps1/sh introducidos por P1-LIVE-2. Sin "
        "este README, un nuevo desarrollador no descubre los wrappers ni "
        "sabe cuándo invocarlos."
    )
    text = SCRIPTS_README.read_text(encoding="utf-8")
    assert P3_LIVE_1_ANCHOR in text, (
        "[P3-LIVE-1] Anchor removido de scripts/README.md."
    )


def test_b_scripts_readme_documents_both_wrappers():
    """El README debe documentar ambos wrappers (PS1 + SH) — cross-platform
    es parte del contrato del cierre P1-LIVE-2."""
    text = SCRIPTS_README.read_text(encoding="utf-8")
    assert "run_ci.ps1" in text, (
        "[P3-LIVE-1] scripts/README.md no menciona run_ci.ps1 (wrapper "
        "PowerShell para Windows)."
    )
    assert "run_ci.sh" in text, (
        "[P3-LIVE-1] scripts/README.md no menciona run_ci.sh (wrapper "
        "bash para Linux/macOS)."
    )


def test_c_scripts_readme_documents_skip_flags():
    """Las flags skip son no-obvias y deben estar documentadas.

    [P3-CI-SKIP-FLAGS-FOSIL · 2026-08-22] Este caso exigía además las grafías `SKIP_BACKEND`,
    `SKIP_FRONTEND` y `SKIP_BUILD`: variables de entorno del wrapper **bash**, que
    `P3-I18N-RUN-CI-SH-FOSIL` deprecó. Documentar una flag que ya no existe no es inocuo — manda
    al usuario a exportar una variable que nadie lee, y el wrapper corre los tres jobs igual
    mientras él cree haber saltado uno. Un guard que EXIGE documentar lo inexistente es peor que
    no tener guard.

    Lo que se conserva es la propiedad de verdad: **las flags que el wrapper VIVO soporta tienen
    que estar documentadas**, y se leen de su propio `param(...)` en vez de enumerarlas a mano —
    así una flag nueva sin documentar falla aquí, que es justo lo que este fichero existe para
    conseguir."""
    text = SCRIPTS_README.read_text(encoding="utf-8")
    ps1 = (SCRIPTS_README.parent / "run_ci.ps1").read_text(encoding="utf-8", errors="replace")

    declaradas = re.findall(r"\[switch\]\$(\w+)", ps1)
    assert declaradas, (
        "[P3-LIVE-1] no encuentro switches en run_ci.ps1: sin ellos este test pasaría por "
        "vacuidad y una flag nueva sin documentar se colaría"
    )
    for flag in declaradas:
        assert f"-{flag}" in text, (
            f"[P3-LIVE-1] scripts/README.md no documenta la flag `-{flag}`. El wrapper la "
            f"soporta; si no está en docs, los usuarios corren los 3 jobs siempre."
        )


def test_d_github_readme_exists_with_anchor():
    """`.github/README.md` debe existir con anchor P3-LIVE-1."""
    assert GITHUB_README.exists(), (
        "[P3-LIVE-1] `.github/README.md` missing. Documenta los 4 jobs del "
        "workflow + activación branch protection + SOPs."
    )
    text = GITHUB_README.read_text(encoding="utf-8")
    assert P3_LIVE_1_ANCHOR in text, (
        "[P3-LIVE-1] Anchor removido de .github/README.md."
    )


def test_e_github_readme_documents_lint_gate_history():
    """**Critical, invertido en P0-CI-VERDICT**: el README DEBE explicar que el lint del
    frontend es BLOQUEANTE (techo global + techo por regla, P2-LINT-RATCHET-POR-REGLA) y que
    el `continue-on-error: true` de P2-LIVE-1 está SUPERSEDED. Sin la historia escrita, un
    futuro reader "restaura" la flag creyendo que falta — y desarma el gate."""
    text = GITHUB_README.read_text(encoding="utf-8")
    assert "continue-on-error" in text and "P2-LIVE-1" in text, (
        "[P3-LIVE-1] .github/README.md debe contar la historia del lint no-bloqueante "
        "(P2-LIVE-1, `continue-on-error: true`) para que nadie la resucite."
    )
    assert "P2-LINT-RATCHET-POR-REGLA" in text and "lint-count.mjs" in text, (
        "[P3-LIVE-1] .github/README.md no documenta el gate vivo (techo por regla en "
        "scripts/lint-count.mjs, P2-LINT-RATCHET-POR-REGLA)."
    )
    for wf in (BACKEND / ".github" / "workflows").glob("*.yml"):
        assert "continue-on-error" not in wf.read_text(encoding="utf-8"), (
            f"[P0-CI-VERDICT] {wf.name} volvió a llevar `continue-on-error`: un job que no "
            "bloquea es telemetría, no gate."
        )


def test_f_github_readme_documents_branch_protection_setup():
    """El README debe explicar cómo activar branch protection — el
    workflow sin protection es solo telemetría, no gate real."""
    text = GITHUB_README.read_text(encoding="utf-8")
    assert "branch protection" in text.lower(), (
        "[P3-LIVE-1] .github/README.md no documenta branch protection. "
        "Sin esta config en GitHub UI, el workflow corre pero NO bloquea "
        "merges → gate no funcional."
    )
    # Y debe nombrar explícitamente los status checks bloqueantes REALES de los dos repos.
    for job_label in ("Backend pytest", "quality", "audit", "suministro", "e2e (webkit)"):
        assert job_label in text, (
            f"[P3-LIVE-1] .github/README.md no menciona el status check "
            f"`{job_label}` en la sección de branch protection setup."
        )


def test_g_github_readme_documents_not_e2e_filter():
    """El filtro `-m "not e2e"` es load-bearing — runners sin DB live
    crashean sin el filtro. Debe estar documentado."""
    text = GITHUB_README.read_text(encoding="utf-8")
    assert "not e2e" in text, (
        "[P3-LIVE-1] .github/README.md no documenta el filtro `not e2e`. "
        "Si alguien remueve el filtro 'para correr todos los tests', los "
        "runners sin Supabase live crashean."
    )
