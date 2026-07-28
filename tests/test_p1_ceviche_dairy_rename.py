"""[P1-CEVICHE-DAIRY-RENAME · 2026-07-28] El ceviche cura pescado crudo; el lácteo no.

## El caso vivo

Plan ab2b0a16 (16:24, plan del owner): "Ceviche de Queso Gouda Marinado en Cítricos".
El queso no se "cura" en cítrico — el plato real es queso marinado. El sistema raw-safety
NO lo veía porque 'queso' no es proteína animal cruda (_RAW_ANIMAL_PROTEIN_TERMS), y el
ceviche VEGETAL es plato RD legítimo (P2-RAW-SEAFOOD-FALSE-POSITIVE: "Ceviche de yuca") —
así que el nombre deshonesto sobrevivía todos los guards.

## El contrato

- "Ceviche/Cebiche de <lácteo>" → renombre a "<Lácteo> Marinado/a en Cítricos"
  (género por cabeza; si ya menciona marinado/cítrico/limón, solo se quita la cabeza).
- El PROTAGONISTA se corta en conectores (con/sobre/y/en/al/a): "Ceviche de yuca con
  queso rallado" queda INTACTO (queso de guarnición ≠ protagonista).
- "Leche de coco/almendras/soya/avena" es vegetal → intacto.
- Pescado/marisco ("Ceviche de Mero") → intacto (dominio del sistema raw-safety).
- La desc del meal renombrado cambia "ceviche"→"marinado".
- Knob MEALFIT_CEVICHE_DAIRY_RENAME (default True); OFF → no-op.

tooltip-anchor: P1-CEVICHE-DAIRY-RENAME
"""
from __future__ import annotations

from unittest.mock import patch

import graph_orchestrator as go


def _run(meals):
    days = [{"meals": meals}]
    n = go._ceviche_dairy_rename_pass(days)
    return n, days[0]["meals"]


def test_caso_vivo_gouda():
    n, meals = _run([{"name": "Ceviche de Queso Gouda Marinado en Cítricos",
                      "desc": "Un ceviche cremoso de gouda."}])
    assert n == 1
    assert meals[0]["name"] == "Queso Gouda Marinado en Cítricos"
    assert "ceviche" not in meals[0]["desc"].lower()
    assert "marinado" in meals[0]["desc"].lower()


def test_lacteo_sin_hint_gana_sufijo_con_genero():
    n, meals = _run([{"name": "Ceviche de Mozzarella fresca", "desc": "Un cebiche lácteo."}])
    assert n == 1
    assert meals[0]["name"] == "Mozzarella fresca Marinada en Cítricos"
    n2, meals2 = _run([{"name": "Ceviche de Queso de Freír", "desc": ""}])
    assert n2 == 1
    assert meals2[0]["name"] == "Queso de Freír Marinado en Cítricos"


def test_vegetal_intacto_incluso_con_queso_de_guarnicion():
    n, meals = _run([
        {"name": "Ceviche de yuca con edamame", "desc": "Clásico vegetal."},
        {"name": "Ceviche de yuca con queso rallado", "desc": "Vegetal con garnish."},
    ])
    assert n == 0
    assert meals[0]["name"] == "Ceviche de yuca con edamame"
    assert meals[1]["name"] == "Ceviche de yuca con queso rallado"


def test_pescado_y_leches_vegetales_intactos():
    n, meals = _run([
        {"name": "Ceviche de Mero al limón", "desc": "Pescado fresco."},
        {"name": "Cebiche de Leche de coco", "desc": "vegano"},
    ])
    assert n == 0
    assert meals[0]["name"] == "Ceviche de Mero al limón"
    assert meals[1]["name"] == "Cebiche de Leche de coco"


def test_knob_off_no_op():
    with patch.object(go, "CEVICHE_DAIRY_RENAME_ENABLED", False):
        n, meals = _run([{"name": "Ceviche de Queso Gouda", "desc": "x"}])
    assert n == 0
    assert meals[0]["name"] == "Ceviche de Queso Gouda"


def test_wired_en_finalize():
    """El pase corre en finalize_plan_data_coherence (counter `ceviche_dairy=`)."""
    import pathlib
    src = pathlib.Path(go.__file__).with_suffix(".py").read_text(encoding="utf-8")
    i = src.find("def finalize_plan_data_coherence")
    _nxt = src.find("\ndef ", i + 10)
    body = src[i:_nxt if _nxt > 0 else len(src)]
    assert "_ceviche_dairy_rename_pass(days)" in body, "el pase no está wireado en finalize"
    assert "ceviche_dairy=" in body, "falta el counter en el log de finalize"
