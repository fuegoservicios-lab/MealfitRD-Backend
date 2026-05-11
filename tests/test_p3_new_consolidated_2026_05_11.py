"""[P3-NEW-1/2/3/4 · 2026-05-11] Tests consolidados para los 4 P3
restantes. P3-NEW-5 tiene archivo propio (slug del marker).

Cubre:
    - P3-NEW-1: lifecycle plan_id documentado en CLAUDE.md.
    - P3-NEW-2: sentinels review tracker en sentinels.js.
    - P3-NEW-3: Pantry recalc tiene try/catch + guards de success.
    - P3-NEW-4: dietType drift documentado inline en formValidation.js.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# P3-NEW-1: lifecycle plan_id docstring en CLAUDE.md
# ---------------------------------------------------------------------------
def test_p3_new_1_claude_md_documents_plan_id_lifecycle():
    """CLAUDE.md debe tener una sección "Lifecycle de plan_id"
    documentando los 6 sistemas + invariantes I1-I5."""
    claude_md = (_REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "Lifecycle de `plan_id`" in claude_md, (
        "P3-NEW-1 regresión: sección 'Lifecycle de plan_id' ausente "
        "en CLAUDE.md. Sin la documentación, los devs no saben dónde "
        "se asigna ni dónde valida ownership el sistema."
    )
    # 6 stages: form → orchestrator → chunks → shopping → historial → recipe
    expected_stages = [
        "FORMULARIO",
        "ORQUESTADOR",
        "CHUNKS",
        "SHOPPING LIST",
        "HISTORIAL",
        "RECIPE EXPAND",
    ]
    for stage in expected_stages:
        assert stage in claude_md, (
            f"P3-NEW-1 regresión: stage `{stage}` ausente del flow "
            "diagram. El lifecycle de plan_id queda incompleto."
        )


def test_p3_new_1_invariants_enumerated():
    """La tabla de invariantes debe enumerar al menos I1-I5."""
    claude_md = (_REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    # Localizar la sección.
    lifecycle_idx = claude_md.find("Lifecycle de `plan_id`")
    next_section_idx = claude_md.find("## Flujo de coherencia", lifecycle_idx)
    section = claude_md[lifecycle_idx:next_section_idx]
    for inv in ["I1", "I2", "I3", "I4", "I5"]:
        assert inv in section, (
            f"P3-NEW-1 regresión: invariante {inv} no enumerada. "
            "La tabla de invariantes debe cubrir las 5 propiedades del "
            "lifecycle (assignment, ownership filter, client-side guard, "
            "cache invalidation, quality alert)."
        )


# ---------------------------------------------------------------------------
# P3-NEW-2: sentinels review tracker
# ---------------------------------------------------------------------------
def test_p3_new_2_sentinels_review_block_present():
    """`sentinels.js` debe tener un bloque de "Review tracker" con
    fecha de última verificación + lista de invariantes confirmadas."""
    sentinels = (_REPO_ROOT / "frontend" / "src" / "config" / "sentinels.js").read_text(encoding="utf-8")
    assert "P3-NEW-2" in sentinels, (
        "P3-NEW-2 regresión: marker no presente en sentinels.js. "
        "Sin marker, no hay trazabilidad de cuándo se hizo el último "
        "review."
    )
    assert "Última review" in sentinels or "Ultima review" in sentinels, (
        "P3-NEW-2 regresión: bloque 'Última review' ausente."
    )


# ---------------------------------------------------------------------------
# P3-NEW-3: Pantry recalc error paths
# ---------------------------------------------------------------------------
def test_p3_new_3_pantry_recalc_has_try_catch():
    """`_recalcShoppingListAfterPantryChange` debe estar wrapped en
    try/catch — un fallo del recalc NO debe abortar el flujo principal
    de pantry (el cambio ya se persistió via increment_inventory_quantity)."""
    pantry_fp = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"
    src = pantry_fp.read_text(encoding="utf-8")
    fn_idx = src.find("_recalcShoppingListAfterPantryChange = async")
    assert fn_idx > 0, "función no encontrada"
    # Buscar el closing brace del scope `};` correspondiente.
    end_idx = src.find("\n    };", fn_idx)
    assert end_idx > fn_idx, "fin del scope no encontrado"
    body = src[fn_idx:end_idx]
    assert "try {" in body and "catch" in body, (
        "P3-NEW-3 regresión: `_recalcShoppingListAfterPantryChange` "
        "sin try/catch. Un fallo del fetch propagaría al caller "
        "(handleDeleteItem, handleUpdateQuantity) y rompería la UX."
    )


def test_p3_new_3_pantry_recalc_guards_success():
    """El bloque post-fetch debe validar `result.success &&
    result.plan_data` antes de invocar `setPlanData`. Sin guard,
    una response con `success: false` pisaría planData con junk."""
    pantry_fp = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"
    src = pantry_fp.read_text(encoding="utf-8")
    # Patrón: `if (result.success && result.plan_data)`.
    assert re.search(
        r"if\s*\(\s*result\.success\s*&&\s*result\.plan_data\s*\)",
        src,
    ), (
        "P3-NEW-3 regresión: guard `result.success && result.plan_data` "
        "removido pre-setPlanData. Sin guard, response degradada (e.g., "
        "backend rate-limited) pisaría planData con junk."
    )


# ---------------------------------------------------------------------------
# P3-NEW-4: dietType drift inline doc
# ---------------------------------------------------------------------------
def test_p3_new_4_diet_type_drift_documented():
    """`formValidation.js` debe documentar inline POR QUÉ `dietType`
    está en `REQUIRED_FORM_FIELDS` frontend pero NO en backend."""
    fp = _REPO_ROOT / "frontend" / "src" / "config" / "formValidation.js"
    src = fp.read_text(encoding="utf-8")
    # Buscar bloque P3-NEW-4 cerca de 'dietType'.
    diet_idx = src.find("'dietType'")
    assert diet_idx > 0
    # Ventana de 800 chars hacia atrás para encontrar el bloque P3-NEW-4.
    window = src[max(0, diet_idx - 1200):diet_idx + 200]
    assert "P3-NEW-4" in window, (
        "P3-NEW-4 regresión: marker P3-NEW-4 ausente cerca de "
        "`'dietType'`. Sin el comentario, alguien podría 'arreglar' "
        "la asimetría agregando dietType al backend, rompiendo la "
        "compat legacy."
    )
    assert "legacy" in window.lower() or "compat" in window.lower(), (
        "P3-NEW-4 regresión: la justificación de la asimetría no "
        "menciona 'legacy' / 'compat'. Sin razón, el comentario es "
        "ornamental."
    )
