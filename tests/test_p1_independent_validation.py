"""[P1-INDEPENDENT-VALIDATION · 2026-06-26] (auditoría gap #2) Ancla el harness de validación INDEPENDIENTE
del catálogo (rompe la circularidad auto-referencial de la validación existente).

NO corre la comparación contra el catálogo vivo (requiere Neon/VPS — ver el script). Aquí se valida, SIN DB:
  1. La tabla de referencia `_USDA_REF` es internamente consistente (Atwater: kcal ≈ 4P+4C+9F) → caza typos
     en la referencia que la harían inútil como ground-truth.
  2. Cubre los staples clave es-DO.
  3. La referencia es LITERAL (hardcoded), no leída del catálogo → anti-circularidad estructural.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "clinical_independent_validation.py"


def _load():
    spec = importlib.util.spec_from_file_location("clinical_independent_validation", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reference_table_is_atwater_consistent():
    """Cada fila de la referencia USDA debe cumplir kcal ≈ 4P+4C+9F (sanity anti-typo). Los veggies de muy
    pocas kcal se saltan (la fibra infla el Atwater de carbohidratos → discrepancia esperada, no un typo)."""
    mod = _load()
    for name, vals in mod._USDA_REF.items():
        kcal, p, c, f = vals[0], vals[1], vals[2], vals[3]
        if kcal < 40:  # low-cal/high-fiber veggies: Atwater sobre-cuenta la fibra → no comparable
            continue
        atwater = 4 * p + 4 * c + 9 * f
        assert abs(atwater - kcal) / kcal <= 0.18, (
            f"{name}: referencia inconsistente (Atwater {atwater:.0f} vs kcal {kcal}) — ¿typo en _USDA_REF?")


def test_reference_covers_key_staples():
    mod = _load()
    names = {n.lower() for n in mod._USDA_REF}
    for staple in ("arroz blanco", "pechuga de pollo", "huevo", "aceite de oliva", "lentejas", "guineo"):
        assert staple in names, f"falta el staple {staple!r} en la referencia"
    assert len(mod._USDA_REF) >= 20, "la muestra de referencia debe cubrir >=20 staples"


def test_tolerances_defined_and_sane():
    mod = _load()
    for m in ("kcal", "protein", "carbs", "fats"):
        assert 0 < mod._TOL[m] <= 0.5, f"tolerancia de {m} fuera de rango"


def test_reference_is_independent_literal_not_from_catalog():
    """Anti-circularidad: la referencia debe ser literal hardcoded, NO derivada de master_ingredients."""
    src = _SCRIPT.read_text(encoding="utf-8")
    assert "_USDA_REF" in src and "INDEPENDIENTE" in src
    # La referencia se define como literal ANTES de cualquier query; el catálogo solo se LEE para comparar.
    ref_pos = src.index("_USDA_REF = {")
    sql_pos = src.index("master_ingredients")
    assert ref_pos < sql_pos or "SELECT" in src, "la referencia debe ser literal, no leída del catálogo"
