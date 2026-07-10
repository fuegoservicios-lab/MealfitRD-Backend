"""[P2-UNIT-CONV-1 · 2026-05-11] Flip de default canary OFF → ON del knob
`MEALFIT_COHERENCE_UNIT_CONVERTER_ENABLED`.

Bug original (audit 2026-05-11):
    El knob `MEALFIT_COHERENCE_UNIT_CONVERTER_ENABLED` introducido en
    P1-NEW-10 era CANARY default False. Sin él, el coherence guard
    reportaba false drift cuando expected y aggregated usaban distintos
    aliases del mismo sistema físico (e.g., `{Arroz: {kg: 1.0}}` vs
    `{Arroz: {g: 1000.0}}` — ambos correctos, pero el guard reportaba
    `unit_mismatch`). Esto contaminaba `_shopping_coherence_block_history`
    y métricas P3-B con noise post-mortem.

Cierre:
    1. Flip default a True (audit MCP del 2026-05-11 confirmó 0 entries
       en prod history y 0 prod data; sin risk medible).
    2. Knob queda como kill-switch invertido: setear
       `MEALFIT_COHERENCE_UNIT_CONVERTER_ENABLED=false` revierte sin
       redeploy si en el futuro el converter genera retry storms.
    3. Tests `test_p1_new_10_*` siguen verde (cubren matemática).
    4. Test parser-based en `test_p1_new_10_*::test_knob_default_is_true_post_p2_unit_conv_1`
       enforza el contrato del default.

Este archivo es el **marker anchor** para el cross-link bidireccional
`P2-HIST-AUDIT-14` (slug `p2_unit_conv_1` derivado del marker
`P2-UNIT-CONV-1 · 2026-05-11`). Sin él, `test_p2_hist_audit_14_marker_test_link`
falla porque no encuentra archivo `tests/test_p2_unit_conv_1*.py`.

Tooltip-anchor: P2-UNIT-CONV-1-MARKER-ANCHOR
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHOPCALC_FP = _REPO_ROOT / "backend" / "shopping_calculator.py"
_APP_FP = _REPO_ROOT / "backend" / "app.py"


@pytest.fixture(scope="module")
def shopcalc_src() -> str:
    return _SHOPCALC_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_src() -> str:
    return _APP_FP.read_text(encoding="utf-8")


def test_p2_unit_conv_1_default_flipped(shopcalc_src: str):
    """El default del knob es True post-P2-UNIT-CONV-1."""
    m = re.search(
        r'_knob_env_bool\(\s*"MEALFIT_COHERENCE_UNIT_CONVERTER_ENABLED"\s*,\s*(\w+)',
        shopcalc_src,
    )
    assert m, (
        "P2-UNIT-CONV-1 regresión: el knob "
        "MEALFIT_COHERENCE_UNIT_CONVERTER_ENABLED ya no se lee vía "
        "`_knob_env_bool`. Si se removió, el converter pierde su kill switch."
    )
    assert m.group(1) == "True", (
        f"P2-UNIT-CONV-1 regresión: default flippeado de True a {m.group(1)}. "
        f"El default debe ser True post-P2-UNIT-CONV-1. Si genuinamente "
        f"necesitas restaurar canary OFF, primero documentar la razón "
        f"(retry storm? regresión real?) en memoria + bumpear marker."
    )


def test_p2_unit_conv_1_docstring_documents_history(shopcalc_src: str):
    """El docstring del helper menciona el flip y el rationale."""
    fn_idx = shopcalc_src.find("def _get_coherence_unit_converter_enabled(")
    assert fn_idx > 0, "_get_coherence_unit_converter_enabled no encontrado"
    body = shopcalc_src[fn_idx: fn_idx + 3000]
    assert "P2-UNIT-CONV-1" in body, (
        "P2-UNIT-CONV-1 regresión: el docstring del helper no menciona el "
        "marker P2-UNIT-CONV-1. Sin trazabilidad, un revisor futuro no "
        "entiende por qué el default es True (parece arbitrario)."
    )
    assert "kill switch" in body or "kill-switch" in body, (
        "P2-UNIT-CONV-1: el docstring debe documentar que el knob queda como "
        "kill switch (=false revierte sin redeploy)."
    )


def test_p2_unit_conv_1_marker_active(app_src: str):
    """`_LAST_KNOWN_PFIX` apunta a P2-UNIT-CONV-1 (marker activo)."""
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', app_src)
    assert m, "_LAST_KNOWN_PFIX no parsea en app.py"
    # No exigimos el marker exacto (otros P-fixes posteriores pueden
    # bumpear), solo que el marker tenga la forma canónica.
    marker = m.group(1)
    # [P0-1-PAIRING-PLAUSIBILITY-GATE · 2026-07-10] fix: el segmento tras el primer guion puede
    # empezar con un DÍGITO ("P0-1-FINAL-BAND-CLOSER", "P0-1-PAIRING-PLAUSIBILITY-GATE" — convención
    # activa desde hace meses, ver `_MARKER_PATTERN` canónico en test_p3_1_last_known_pfix_freshness.py:
    # `P\d+(?:-[A-Z0-9]+)+`). El regex previo (`[A-Z][A-Z0-9-]*`) exigía letra inicial y quedaba
    # incorrectamente ROJO cada vez que el marker activo usaba esa convención estándar.
    assert re.match(r"^P[0-3]-[A-Z0-9][A-Z0-9-]*\s+·\s+\d{4}-\d{2}-\d{2}$", marker), (
        f"P2-UNIT-CONV-1: _LAST_KNOWN_PFIX={marker!r} no matchea formato "
        f"canónico `Pn-X · YYYY-MM-DD`. Test P3-1 cubre formato general; "
        f"acá solo lo replicamos como sanity check del marker activo."
    )


def test_anchor_token_present_for_grep(shopcalc_src: str):
    """El tooltip-anchor `P2-UNIT-CONV-1-DEFAULT` debe estar en el helper
    para grep cross-codebase desde futuros audits / docs."""
    fn_idx = shopcalc_src.find("def _get_coherence_unit_converter_enabled(")
    body = shopcalc_src[fn_idx: fn_idx + 3000]
    assert "P2-UNIT-CONV-1-DEFAULT" in body, (
        "P2-UNIT-CONV-1: tooltip-anchor `P2-UNIT-CONV-1-DEFAULT` desaparecido. "
        "Sin él, un grep desde docs externos / audits futuros no encuentra "
        "este punto de configuración."
    )
