"""[P1-TORTILLA-PERECEDERO · 2026-07-06] Fix de categorización de tortilla/wrap
integral en la lista de compras.

Bug observado en la review visual del plan de 30 días:
  "Tortilla integral 2 paquetes (Wraps Wholesome Wheat 6 unid 11 Oz · Toufayan)"
  aparecía en la sección "DESPENSA DEL MES — ESTABLES (COMPRA UNA SOLA VEZ)"
  junto a arroz/aceite/sal. Pero un wrap de trigo fresco dura ~1 semana
  refrigerado; 12 wraps comprados el día 1 para un ciclo de 30 días se
  enmohecen. El pan integral SÍ estaba correcto en perecederos (P1-PAN-PERECEDERO)
  → inconsistencia: la tortilla es el mismo tipo de pan blando.

Fix:
  1. `'tortilla'` añadido a `_DESPENSA_PERISHABLE_EXCEPTIONS` (cubre "tortilla
     integral"/"tortilla de trigo"/"tortilla de maíz" por substring).
  2. `is_perishable_category` (Clasificador B, SSOT del flag `is_perishable`
     del path weekly/PDF) gana un parámetro `name` con el mismo override —
     antes era category-only y no tenía hook por nombre.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SHOPPING_PY = _BACKEND_ROOT / "shopping_calculator.py"


# ---------------------------------------------------------------------------
# 1. Parser-based: constante + firma
# ---------------------------------------------------------------------------
def test_tortilla_in_exceptions():
    """`'tortilla'` DEBE estar en `_DESPENSA_PERISHABLE_EXCEPTIONS`."""
    src = _SHOPPING_PY.read_text(encoding="utf-8")
    block = re.search(
        r"_DESPENSA_PERISHABLE_EXCEPTIONS = frozenset\(\{.*?\}\)",
        src,
        re.DOTALL,
    )
    assert block, "No se pudo aislar `_DESPENSA_PERISHABLE_EXCEPTIONS`."
    body_no_comments = re.sub(r"#[^\n]*", "", block.group(0))
    assert "'tortilla'" in body_no_comments, (
        "'tortilla' ausente de _DESPENSA_PERISHABLE_EXCEPTIONS — sin ella el "
        "wrap integral cae en 'estables — compra una sola vez' y se enmohece."
    )


def test_is_perishable_category_has_name_param():
    """`is_perishable_category` DEBE aceptar `name` para el override por nombre
    (espejo del que `_classify_perishability` ya aplica)."""
    src = _SHOPPING_PY.read_text(encoding="utf-8")
    assert re.search(
        r"def is_perishable_category\([^)]*\bname\b", src, re.DOTALL
    ), "is_perishable_category no expone el parámetro `name`."


def test_callsites_pass_name():
    """Los callsites del aggregator DEBEN pasar `name=name` al clasificador —
    sin eso el override nunca se activa en producción."""
    src = _SHOPPING_PY.read_text(encoding="utf-8")
    n = len(re.findall(
        r'_cat_for_perish,\s*market_obj\.get\("shelf_life_days"\),\s*name=name', src
    ))
    assert n >= 2, (
        f"Solo {n} callsite(s) del aggregator pasan name=name (esperado 2). "
        f"El flag is_perishable no reflejaría el override de tortilla."
    )


# ---------------------------------------------------------------------------
# 2. Funcional
# ---------------------------------------------------------------------------
def _load():
    os.environ.setdefault("GEMINI_API_KEY", "dummy")
    os.environ.setdefault("SUPABASE_URL", "https://dummy.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "dummy")
    os.environ.setdefault("CRON_SECRET", "dummy")
    sys.path.insert(0, str(_BACKEND_ROOT))
    from shopping_calculator import _classify_perishability, is_perishable_category
    return _classify_perishability, is_perishable_category


def test_tortilla_classify_perishable():
    """Clasificador A (biweekly/monthly): tortilla integral con category=Despensa
    + shelf=14 default → perishable."""
    classify, _ = _load()
    for nm in ("Tortilla integral", "Tortilla de trigo", "Tortilla de maíz", "Wrap integral de trigo"):
        # "Wrap" no contiene 'tortilla'; sólo verificamos las tortillas nombradas.
        if "tortilla" not in nm.lower():
            continue
        result = classify(nm, {"category": "Despensa", "shelf_life_days": 14})
        assert result == "perishable", (
            f"{nm!r} clasificado como {result!r} (esperado 'perishable')."
        )


def test_tortilla_is_perishable_category_flag():
    """Clasificador B (weekly/PDF flag): con name pasado, la tortilla integral
    catalogada Despensa → True (perecedero)."""
    _, is_perish = _load()
    assert is_perish("Despensa", 14, name="Tortilla integral") is True, (
        "El flag is_perishable de la tortilla integral debe ser True; sin el "
        "override quedaría en 'estables' en el weekly."
    )


def test_tortilla_without_name_still_stable_backcompat():
    """Sin `name` (back-compat), la categoría Despensa sigue devolviendo False —
    el override es aditivo, no cambia el comportamiento previo."""
    _, is_perish = _load()
    assert is_perish("Despensa", 14) is False


def test_no_regression_casabe_arroz_still_staple():
    """El fix de tortilla NO debe contaminar staples reales."""
    classify, is_perish = _load()
    for nm in ("Casabe", "Arroz integral", "Avena"):
        assert classify(nm, {"category": "Despensa", "shelf_life_days": 365}) == "staple"
        assert is_perish("Despensa", 365, name=nm) is False
