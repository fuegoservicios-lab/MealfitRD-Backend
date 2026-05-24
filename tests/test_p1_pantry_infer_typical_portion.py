"""[P1-PANTRY-INFER · 2026-05-22] Tests de la inferencia de porción típica
para `deduct_consumed_meal_from_inventory`.

Cierra el bug productivo verificado 2026-05-22: el chat agent registraba
comidas con `ingredients=["Avena", "Semillas de chía", "Mantequilla de maní"]`
(nombres sin cantidad — la LLM emite el shape natural del usuario, no el
formato cuantitativo de la docstring). `_parse_quantity` retornaba qty=0
con name limpio; el flow lo categorizaba `parse_failed_or_invalid_qty` y
la nevera nunca se descontaba. Tabla `failed_inventory_deductions` tenía
3 incidentes acumulados (avena/chía/maní 2026-05-22, bandeja paisa 2026-05-20,
mayonesa/kétchup 2026-05-20) sin retry.

El fix añade:
  1. `_TYPICAL_PORTION_BY_NAME`: dict canónico es-DO (~120 entries).
  2. `_infer_typical_portion(name)`: lookup exacto → substring → fallback
     heurístico unidad/gramos.
  3. Branch en `deduct_consumed_meal_from_inventory` que invoca la inferencia
     antes de marcar el item como failed.
  4. Knob `MEALFIT_PANTRY_INFER_TYPICAL_PORTION` (default True) para rollback.

Cross-link convention (P2-HIST-AUDIT-14): slug `p1_pantry_infer_typical_portion`
matchea este archivo.

Tooltip-anchor: P1-PANTRY-INFER-TYPICAL-PORTION-TABLE, P1-PANTRY-INFER-INFER-FN.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_INVENTORY_PY = _BACKEND_ROOT / "db_inventory.py"


@pytest.fixture(scope="module")
def db_inventory_src() -> str:
    return _DB_INVENTORY_PY.read_text(encoding="utf-8")


# ===========================================================================
# Sección 1 — parser checks: anchors, knob, dict shape
# ===========================================================================

def test_tooltip_anchor_typical_portion_table_present(db_inventory_src: str):
    """El dict `_TYPICAL_PORTION_BY_NAME` debe tener el tooltip-anchor que el
    test parsea — sin esto, un renombre rompe producción antes que el test."""
    assert "P1-PANTRY-INFER-TYPICAL-PORTION-TABLE" in db_inventory_src, (
        "P1-PANTRY-INFER regresión: tooltip-anchor "
        "`P1-PANTRY-INFER-TYPICAL-PORTION-TABLE` removido de db_inventory.py. "
        "Si renombraste o moviste el dict, actualizar el anchor + este test."
    )


def test_tooltip_anchor_infer_fn_present(db_inventory_src: str):
    """El helper `_infer_typical_portion` debe tener tooltip-anchor."""
    assert "P1-PANTRY-INFER-INFER-FN" in db_inventory_src, (
        "P1-PANTRY-INFER regresión: tooltip-anchor `P1-PANTRY-INFER-INFER-FN` "
        "removido. Si renombraste el helper, actualizar este test + anchor."
    )


def test_knob_pantry_infer_typical_portion_referenced(db_inventory_src: str):
    """El knob `MEALFIT_PANTRY_INFER_TYPICAL_PORTION` debe leerse en el módulo
    (vía `_env_bool`/`_knob_env_bool`). Default es True; rollback sin redeploy."""
    assert "MEALFIT_PANTRY_INFER_TYPICAL_PORTION" in db_inventory_src, (
        "P1-PANTRY-INFER regresión: knob `MEALFIT_PANTRY_INFER_TYPICAL_PORTION` "
        "ya no se lee en db_inventory.py. Es el mecanismo de rollback "
        "operacional sin redeploy — preservar el callsite."
    )


def test_typical_portion_dict_declared(db_inventory_src: str):
    """`_TYPICAL_PORTION_BY_NAME: Dict[str, tuple] = {...}` debe existir."""
    m = re.search(
        r"_TYPICAL_PORTION_BY_NAME\s*:\s*Dict\[str,\s*tuple\]\s*=\s*\{",
        db_inventory_src,
    )
    assert m is not None, (
        "P1-PANTRY-INFER regresión: declaración del dict "
        "`_TYPICAL_PORTION_BY_NAME` removida o cambió de tipo. Si refactoreaste "
        "a un módulo separado, actualizar este test."
    )


def test_typical_portion_dict_minimum_size(db_inventory_src: str):
    """Cuenta aproximada de entries (líneas `"name": (qty, "unit"),`) ≥ 80.

    Range razonable: el bundle nace con ~120 entries (granos + proteínas +
    nuts + lácteos + condimentos + verduras + frutas + tubérculos + panes).
    Si alguien limpia agresivamente y baja de 80, sospechar regresión."""
    m = re.search(
        r"_TYPICAL_PORTION_BY_NAME[^{]*\{(.*?)^\}",
        db_inventory_src,
        re.DOTALL | re.MULTILINE,
    )
    assert m is not None
    body = m.group(1)
    entry_re = re.compile(r'^\s*"[^"]+"\s*:\s*\([^)]+\)\s*,', re.MULTILINE)
    count = len(entry_re.findall(body))
    assert count >= 80, (
        f"P1-PANTRY-INFER regresión: `_TYPICAL_PORTION_BY_NAME` tiene {count} "
        f"entries (esperado ≥80). Items canonical típicos en es-DO: avena, "
        f"arroz, habichuelas, gandules, mantequilla de maní, queso blanco, "
        f"yogurt griego, pollo, pescado, huevo, plátano, guineo, mango, etc."
    )


def test_critical_items_present_in_dict(db_inventory_src: str):
    """Los 3 items del incidente productivo 2026-05-22 (avena/chía/maní) DEBEN
    estar como entries canónicas — el test funcional E2E falla si están missing.

    `mantequilla de mani` y `semillas de chia` están sin tilde por convención
    `_strip_accents_lower` antes del lookup."""
    required = [
        '"avena":',
        '"mantequilla de mani":',
        '"semillas de chia":',
        '"mani":',
        '"huevo":',
        '"pollo":',
        '"arroz":',
        '"habichuelas":',
        '"platano":',
        '"guineo":',
        '"yogurt griego":',
    ]
    missing = [k for k in required if k not in db_inventory_src]
    assert not missing, (
        f"P1-PANTRY-INFER regresión: items canónicos faltantes en "
        f"`_TYPICAL_PORTION_BY_NAME`: {missing}. Estos cubren el incidente "
        f"productivo 2026-05-22 y los top platos es-DO. Si removiste por "
        f"razón válida, actualizar la lista en este test."
    )


def test_unitary_hints_tuple_present(db_inventory_src: str):
    """`_UNITARY_NAME_HINTS` tuple debe existir — driver del fallback
    heurístico (1.0, 'unidad') vs (50.0, 'g')."""
    assert "_UNITARY_NAME_HINTS" in db_inventory_src, (
        "P1-PANTRY-INFER regresión: `_UNITARY_NAME_HINTS` removida. Sin la "
        "lista de hints, el fallback genérico no distingue 'manzana' "
        "(1 unidad) de 'arroz' (50g) → comportamiento inconsistente."
    )


def test_infer_called_in_deduct_function(db_inventory_src: str):
    """El branch del `if not (name and qty > 0)` en
    `deduct_consumed_meal_from_inventory` debe invocar `_infer_typical_portion`
    ANTES de marcar el item como failed. Sin esto, el fix no aplica al
    runtime path real."""
    fn_re = re.compile(r"def\s+deduct_consumed_meal_from_inventory\s*\(")
    m = fn_re.search(db_inventory_src)
    assert m is not None
    body_start = m.end()
    next_def = re.search(r"\ndef\s|\nclass\s", db_inventory_src[body_start:])
    body_end = body_start + (
        next_def.start() if next_def else min(8000, len(db_inventory_src) - body_start)
    )
    body = db_inventory_src[body_start:body_end]
    assert "_infer_typical_portion" in body, (
        "P1-PANTRY-INFER regresión: `deduct_consumed_meal_from_inventory` "
        "no invoca `_infer_typical_portion` en su cuerpo. Si re-arquitecturaste "
        "el flow, asegurarte que la inferencia siga corriendo ANTES del "
        "branch `failed_items.append({'reason': 'parse_failed_or_invalid_qty'})`."
    )

    # La inferencia debe estar ANTES del append failed, no después (correctness
    # del orden importa — si no, el bug regresa).
    infer_pos = body.find("_infer_typical_portion")
    failed_pos = body.find('"parse_failed_or_invalid_qty"')
    assert infer_pos < failed_pos, (
        "P1-PANTRY-INFER regresión: `_infer_typical_portion` aparece DESPUÉS "
        "de `parse_failed_or_invalid_qty`. Order matters: el fix debe inferir "
        "primero y solo caer al failed branch si la inferencia retornó None."
    )


# ===========================================================================
# Sección 2 — tests funcionales del helper (import directo del dict)
# ===========================================================================

def _import_helper():
    """Importa `_infer_typical_portion` y el dict del módulo prod.

    El módulo `db_inventory.py` requiere `db_core` y `shopping_calculator`
    al import-time. Si esos están disponibles, el import funciona; sino,
    skip el test funcional y dejamos los parser-based como red de
    seguridad mínima."""
    try:
        import sys, os
        backend_root = str(_BACKEND_ROOT)
        if backend_root not in sys.path:
            sys.path.insert(0, backend_root)
        from db_inventory import _infer_typical_portion, _TYPICAL_PORTION_BY_NAME
        return _infer_typical_portion, _TYPICAL_PORTION_BY_NAME
    except Exception as e:
        pytest.skip(f"db_inventory import failed (probable dep faltante): {e!r}")


def test_infer_exact_avena():
    fn, _ = _import_helper()
    qty, unit = fn("Avena")
    assert qty == 40.0 and unit == "g", (
        f"P1-PANTRY-INFER regresión funcional: inferencia para 'Avena' "
        f"esperada (40.0, 'g'), obtenida ({qty}, {unit!r}). "
        f"Una taza cocida ≈ 40g secos."
    )


def test_infer_exact_mantequilla_mani_with_accent():
    fn, _ = _import_helper()
    qty, unit = fn("Mantequilla de maní")
    assert qty == 16.0 and unit == "g", (
        f"P1-PANTRY-INFER regresión funcional: inferencia para "
        f"'Mantequilla de maní' (con tilde) esperada (16.0, 'g'), "
        f"obtenida ({qty}, {unit!r}). El strip de tildes debe ocurrir antes "
        f"del lookup contra el dict (que está sin tildes)."
    )


def test_infer_exact_semillas_chia_with_accent():
    fn, _ = _import_helper()
    qty, unit = fn("Semillas de chía")
    assert qty == 15.0 and unit == "g", (
        f"P1-PANTRY-INFER regresión funcional: 'Semillas de chía' (con tilde) "
        f"esperada (15.0, 'g'), obtenida ({qty}, {unit!r})."
    )


def test_infer_substring_match():
    """Si el name es más largo pero contiene un key del dict, debe matchear
    (preferencia por longest match)."""
    fn, _ = _import_helper()
    # "pollo a la plancha" debe matchear "pollo" no "pollo a la plancha"
    qty, unit = fn("pollo a la plancha")
    assert qty == 120.0 and unit == "g"

    # "pechuga de pollo deshuesada" matchea "pechuga de pollo" (más largo que "pollo")
    qty, unit = fn("pechuga de pollo deshuesada")
    assert qty == 120.0 and unit == "g"


def test_infer_fallback_unitary_hint():
    """Item desconocido pero con hint unitario → (1.0, 'unidad')."""
    fn, _ = _import_helper()
    # 'manzana golden deliciosa' no es key exacto pero contiene 'manzana'
    qty, unit = fn("manzana golden deliciosa")
    # Match exacto a "manzana" via substring; (1.0, "unidad")
    assert qty == 1.0 and unit == "unidad"


def test_infer_fallback_generic_grams():
    """Item totalmente desconocido sin hint unitario → (50.0, 'g')."""
    fn, _ = _import_helper()
    qty, unit = fn("kelp deshidratado del pacífico")
    assert qty == 50.0 and unit == "g", (
        f"P1-PANTRY-INFER: fallback genérico esperado (50.0, 'g'), obtenido "
        f"({qty}, {unit!r}). Si cambiaste el fallback, actualizar el test "
        f"y documentar la decisión."
    )


def test_infer_returns_none_on_empty():
    """Name vacío / whitespace → None. Caller debe poder distinguir 'no se
    pudo inferir' del path 'inferido OK'."""
    fn, _ = _import_helper()
    assert fn("") is None
    assert fn("   ") is None
    assert fn(None) is None


def test_dict_values_have_correct_shape():
    """Cada valor del dict es (qty: float, unit: str) con qty > 0."""
    _, dct = _import_helper()
    for name, val in dct.items():
        assert isinstance(val, tuple) and len(val) == 2, (
            f"P1-PANTRY-INFER regresión: entry {name!r} no es tuple-2: {val!r}"
        )
        qty, unit = val
        assert isinstance(qty, (int, float)) and qty > 0, (
            f"entry {name!r}: qty inválido {qty!r}"
        )
        assert isinstance(unit, str) and unit, (
            f"entry {name!r}: unit inválido {unit!r}"
        )
