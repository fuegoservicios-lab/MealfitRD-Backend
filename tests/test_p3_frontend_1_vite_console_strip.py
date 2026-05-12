"""[P3-FRONTEND-1 · 2026-05-12] `frontend/vite.config.js` strip-ea
`console.log/warn/debug/info` y `debugger` en builds production.

Este test es el "marker-test link" del backend (cumple
`test_p2_hist_audit_14_marker_test_link` que exige
`backend/tests/test_<slug>*.py` con slug del marker
`_LAST_KNOWN_PFIX`). El test funcional equivalente vive en
`frontend/src/__tests__/vite_config_p3_frontend_1_console_strip.test.js`.

Lo que enforza este side:
    A) `frontend/vite.config.js` existe.
    B) `defineConfig(({ mode }) =>` para conditional dev vs prod.
    C) `esbuild` block gated por `mode === 'production'`.
    D) `pure` array contiene `console.log`, `console.warn`,
       `console.debug`, `console.info`.
    E) `pure` array NO contiene `console.error` (preservación).
    F) `drop` array contiene `'debugger'`.
    G) Anchor `P3-FRONTEND-1` presente.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_VITE_CONFIG = _REPO_ROOT / "frontend" / "vite.config.js"


@pytest.fixture(scope="module")
def vite_src() -> str:
    assert _VITE_CONFIG.exists(), (
        "P3-FRONTEND-1: frontend/vite.config.js no encontrado. ¿Movido?"
    )
    return _VITE_CONFIG.read_text(encoding="utf-8")


def test_a_define_config_uses_mode_function(vite_src: str):
    """`defineConfig(({ mode }) => ({...}))` para conditional dev vs prod."""
    pattern = re.compile(r"defineConfig\(\s*\(\s*\{\s*mode\s*\}\s*\)\s*=>")
    assert pattern.search(vite_src), (
        "P3-FRONTEND-1: defineConfig no usa función con `({ mode })`. "
        "Sin esto, conditional `mode === 'production'` no puede leer "
        "el mode efectivo y el strip aplicaría TAMBIÉN en dev/test."
    )


def test_b_esbuild_block_gated_on_production_mode(vite_src: str):
    """`esbuild: mode === 'production' ? { ... } : {}` o spread conditional."""
    pattern = re.compile(
        r"esbuild\s*:\s*mode\s*===\s*['\"]production['\"]"
        r"|mode\s*===\s*['\"]production['\"][\s\S]{0,200}esbuild"
    )
    assert pattern.search(vite_src), (
        "P3-FRONTEND-1: bloque esbuild no gated por production. "
        "El strip aplicaría a dev/test rompiendo Vitest specs que "
        "verifican console output."
    )


def test_c_pure_contains_log_warn_debug_info(vite_src: str):
    """`pure: ['console.log', 'console.warn', 'console.debug', 'console.info']`."""
    expected = ("console.log", "console.warn", "console.debug", "console.info")
    # Aislar el array `pure: [...]`.
    pure_match = re.search(r"pure\s*:\s*\[([^\]]+)\]", vite_src)
    assert pure_match, "P3-FRONTEND-1: array `pure` no declarado en esbuild."
    block = pure_match.group(1)
    for m in expected:
        assert m in block, (
            f"P3-FRONTEND-1: `{m}` ausente en `pure`. Tree-shaking no "
            f"eliminará ese console call del bundle prod."
        )


def test_d_pure_excludes_console_error(vite_src: str):
    """`console.error` NO debe estar en `pure` — preservación explícita
    para post-mortem de bugs reportados por usuario."""
    pure_match = re.search(r"pure\s*:\s*\[([^\]]+)\]", vite_src)
    assert pure_match
    block = pure_match.group(1)
    assert "console.error" not in block, (
        "P3-FRONTEND-1: `console.error` en `pure` strip-earía errores "
        "genuinos del bundle prod — perdemos post-mortem capability."
    )


def test_e_drop_contains_debugger(vite_src: str):
    pattern = re.compile(
        r"drop\s*:\s*\[[^\]]*['\"`]debugger['\"`]",
        re.MULTILINE,
    )
    assert pattern.search(vite_src), (
        "P3-FRONTEND-1: `drop: [..., 'debugger', ...]` ausente. "
        "Sentences `debugger;` accidentales sobrevivirían en prod."
    )


def test_f_anchor_present(vite_src: str):
    assert "P3-FRONTEND-1" in vite_src, (
        "P3-FRONTEND-1: anchor desapareció del vite.config.js."
    )
