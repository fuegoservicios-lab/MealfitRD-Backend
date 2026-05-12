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

from pathlib import Path

P3_LIVE_1_ANCHOR = "P3-LIVE-1"

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_README = REPO_ROOT / "scripts" / "README.md"
GITHUB_README = REPO_ROOT / ".github" / "README.md"


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
    """Las flags skip son no-obvias y deben estar documentadas."""
    text = SCRIPTS_README.read_text(encoding="utf-8")
    for flag in ("-SkipBackend", "-SkipFrontend", "-SkipBuild",
                 "SKIP_BACKEND", "SKIP_FRONTEND", "SKIP_BUILD"):
        assert flag in text, (
            f"[P3-LIVE-1] scripts/README.md no documenta la flag `{flag}`. "
            f"Los wrappers la soportan; si no está en docs, los usuarios "
            f"corren los 3 jobs siempre (perdiendo tiempo)."
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


def test_e_github_readme_documents_continue_on_error_invariant():
    """**Critical**: el README DEBE explicar por qué `frontend-lint` tiene
    `continue-on-error: true`. Sin esta nota, un futuro reader puede
    "arreglar" la flag removiéndola, bloqueando todos los merges sobre
    la baseline 245 eslint errores."""
    text = GITHUB_README.read_text(encoding="utf-8")
    assert "continue-on-error" in text, (
        "[P3-LIVE-1] .github/README.md no menciona `continue-on-error`. "
        "Esta flag es invariante load-bearing del cierre P2-LIVE-1. Si no "
        "está documentada, alguien puede removerla 'arreglando' el CI y "
        "bloquear todos los merges sobre la baseline 245 eslint errores."
    )
    # Y debe mencionar P2-LIVE-1 como contexto del invariante.
    assert "P2-LIVE-1" in text, (
        "[P3-LIVE-1] .github/README.md no referencia P2-LIVE-1 como origen "
        "del invariante `continue-on-error: true`. Referencia necesaria "
        "para que un futuro reader encuentre el contexto en memoria."
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
    # Y debe nombrar explícitamente los 3 status checks bloqueantes.
    for job_label in ("Backend pytest", "Frontend Vitest", "Frontend build"):
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
