"""[P3-OLIVE-RENDER-FIX · 2026-05-16] Bug del renderer de aceitunas:
"X aceitunas" (unidades individuales) se mostraba como "X frascos" en lugar
de "N frascos" (con N = ceil(X×density/container_weight)).

Síntoma observado plan 4cc91584 (PDFs 2026-05-16):
- Ciclo 7d × 1 persona: "Aceitunas: 24 frascos (12 oz c/u)" = 18 lbs
- Ciclo 15d × 1 persona: "Aceitunas: 47 frascos (12 oz c/u)" = 35 lbs
- Ciclo 30d × 1 persona: "Aceitunas: 68 frascos (12 oz c/u)" = 51 lbs

Realidad: 1 frasco (340g, 12 oz) basta para 4 semanas × 1 persona.

Root cause: en `apply_smart_market_units` BLOQUE 4 línea ~2093, cuando
`unit_str == 'unidad'` Y `db_container` existe, asumía "X unidades = X
envases" sin chequear si density_g_per_unit indica "unidades pequeñas"
(aceituna 5g, almendra 1.2g, etc.) que caben en el container (frasco 340g,
bolsa 113g, etc.).

Fix: heurística `_small_unit_in_big_container` — si density_per_u < 50g
Y container_weight_g >= density × 5, convertir unidades a gramos y dividir
por container para obtener N envases reales. Items beneficiados: aceitunas,
almendras, nueces, semillas, pasas. Items NO afectados (siguen comportamiento
legacy): yogurt, leche, huevos (density por container).
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SHOPCALC = (_BACKEND_ROOT / "shopping_calculator.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Anchor parser-based: fix presente en apply_smart_market_units BLOQUE 4
# ---------------------------------------------------------------------------


def test_marker_present():
    assert "P3-OLIVE-RENDER-FIX" in _SHOPCALC, (
        "Marker P3-OLIVE-RENDER-FIX ausente — un refactor cosmético podría "
        "borrar el por qué del fix sin signal."
    )


def test_heuristic_constants_correct():
    """`density < 50g` Y `container >= density × 5` — anchor de los
    thresholds. Bajar density threshold pierde items legítimos (aceitunas
    5g, almendras 1.2g); subirlo afecta items con density media."""
    # Locate the fix block by marker
    idx = _SHOPCALC.find("P3-OLIVE-RENDER-FIX")
    assert idx > 0
    block = _SHOPCALC[idx:idx + 2000]
    assert "density_per_u < 50.0" in block, (
        "Threshold density (50g) cambió o se removió. Items afectados: "
        "aceitunas (5g), almendras (1.2g), nueces (3g), semillas (0.5g)."
    )
    assert "density_per_u * 5.0" in block, (
        "Threshold container (density × 5) cambió. Sin esto, items con "
        "container apenas mayor que la unidad (e.g., 1 unidad = 1 envase) "
        "se mal-clasificarían."
    )


def test_q_rounded_overwritten_to_container_count():
    """CRÍTICO: el fix DEBE reescribir `q_rounded = str(_container_count)`
    porque downstream `market_qty = float(q_rounded)` (línea ~2110)
    sobrescribiría el valor en container_count con el raw count."""
    # Slice generoso (~5000 chars): el bloque completo del try+except+fallback
    # legacy queda dentro.
    idx = _SHOPCALC.find("P3-OLIVE-RENDER-FIX")
    block = _SHOPCALC[idx:idx + 5000]
    assert "q_rounded = str(_container_count)" in block, (
        "Sin `q_rounded = str(_container_count)`, el assignment downstream "
        "`market_qty = float(q_rounded)` revierte el fix — market_qty queda "
        "en raw count (68 unidades) en lugar de container_count (1 frasco). "
        "Bug observable: PDF muestra '1 frasco' pero plan_data tiene "
        "market_qty=68 → cost calc y restock rotos."
    )


def test_fallback_to_legacy_on_exception():
    """Si algo falla (q_rounded no parsea), DEBE caer al comportamiento
    legacy en lugar de crashear."""
    idx = _SHOPCALC.find("P3-OLIVE-RENDER-FIX")
    block = _SHOPCALC[idx:idx + 5000]
    assert "_small_unit_in_big_container = False" in block, (
        "Fix no tiene fallback al legacy en except. Si q_rounded no parsea, "
        "el fix abandona pero deja `_small_unit_in_big_container=True` → "
        "el `if not _small_unit_in_big_container:` downstream no fires → "
        "display_qty queda en estado inconsistente."
    )


# ---------------------------------------------------------------------------
# Funcional: import apply_smart_market_units e invocar con master_items
# realistas. Si el import falla (deps no instaladas en CI), skip.
# ---------------------------------------------------------------------------


def _try_import_apply_smart():
    """Best-effort import. Si el módulo no carga (DB deps, etc.), skip."""
    try:
        # Add backend dir to path
        sys.path.insert(0, str(_BACKEND_ROOT))
        from shopping_calculator import apply_smart_market_units
        return apply_smart_market_units
    except Exception as e:
        pytest.skip(f"No se pudo importar apply_smart_market_units: {e}")


def test_aceitunas_30d_renders_as_1_frasco():
    """68 aceitunas (cap 30d 1p) DEBE renderizar como 1 frasco (340g),
    NO como 68 frascos (23 kg)."""
    apply_smart = _try_import_apply_smart()
    master_item = {
        "category": "Despensa",
        "density_g_per_unit": 5.0,  # 1 oliva = 5g
        "market_container": "frasco",
        "container_weight_g": 340,  # 1 frasco = 340g = 12oz
    }
    result = apply_smart("Aceitunas verdes", 0.0, "unidad", 68, master_item)
    assert "1 frasco" in result["display_qty"] or "1 Frasco" in result["display_qty"], (
        f"display_qty = {result['display_qty']!r}, esperado contener '1 frasco'. "
        f"Bug del rendering NO arreglado — 68 unidades sigue mostrándose como "
        f"68 frascos en lugar de 1."
    )
    assert result["market_qty"] == 1 or result.get("market_qty_numeric") == 1, (
        f"market_qty = {result.get('market_qty')!r} (numeric={result.get('market_qty_numeric')!r}), "
        "esperado 1. Si queda en 68, downstream cost calc y restock están rotos."
    )


def test_aceitunas_7d_few_units_also_correct():
    """24 aceitunas (raw 7d, sin cap) también debe renderizar como 1 frasco
    (24×5g=120g = 1 frasco a 340g, rounded up)."""
    apply_smart = _try_import_apply_smart()
    master_item = {
        "category": "Despensa",
        "density_g_per_unit": 5.0,
        "market_container": "frasco",
        "container_weight_g": 340,
    }
    result = apply_smart("Aceitunas verdes", 0.0, "unidad", 24, master_item)
    assert "1 frasco" in result["display_qty"] or "1 Frasco" in result["display_qty"], (
        f"24 aceitunas → {result['display_qty']!r}, esperado '1 frasco'. "
        f"Sin cap previo, raw count 24 debe convertirse correctamente."
    )


def test_yogurt_pote_NOT_affected():
    """Yogurt: density por pote (200g), container=pote 200g. 3 unidades
    = 3 potes (NO se debe convertir). El threshold `container >= density × 5`
    NO matchea (200 < 200×5=1000), así que cae al legacy."""
    apply_smart = _try_import_apply_smart()
    master_item = {
        "category": "Lacteos",
        "density_g_per_unit": 200.0,  # 1 pote = 200g
        "market_container": "pote",
        "container_weight_g": 200,
    }
    result = apply_smart("Yogurt griego", 0.0, "unidad", 3, master_item)
    # Esperamos "3 potes (...)" — comportamiento legacy preservado
    assert ("3 potes" in result["display_qty"]
            or "3 Potes" in result["display_qty"]), (
        f"Yogurt rendering ROTO post-fix: {result['display_qty']!r}, "
        f"esperado contener '3 potes'. El fix debería NO afectar items donde "
        "1 unidad = 1 envase (density == container)."
    )


def test_almendras_small_units_in_bag():
    """Almendras: density ~1.2g/almendra, bolsa 113g (¼ lb). 100 almendras
    debe ser 2 bolsas (100 × 1.2g = 120g → ceil(120/113) = 2)."""
    apply_smart = _try_import_apply_smart()
    master_item = {
        "category": "Despensa",
        "density_g_per_unit": 1.2,
        "market_container": "bolsa",
        "container_weight_g": 113,  # ¼ lb
    }
    result = apply_smart("Almendras fileteadas", 0.0, "unidad", 100, master_item)
    # 100 × 1.2g = 120g → 2 bolsas (113g cada)
    expected_count = math.ceil((100 * 1.2) / 113)  # = 2
    assert f"{expected_count} bolsa" in result["display_qty"].lower(), (
        f"Almendras: {result['display_qty']!r}, esperado contener "
        f"'{expected_count} bolsa'. Heurística no aplicó a almendras "
        f"(density 1.2 < 50 ✓, container 113 >= 1.2×5=6 ✓)."
    )


def test_no_density_unchanged_behavior():
    """Si master_item NO tiene density_g_per_unit, el fix NO debe disparar
    (no hay forma de convertir unidades a gramos). Legacy preservado."""
    apply_smart = _try_import_apply_smart()
    master_item = {
        "category": "Despensa",
        "market_container": "lata",
        "container_weight_g": 400,
        # density_g_per_unit ausente
    }
    result = apply_smart("Atún en lata", 0.0, "unidad", 5, master_item)
    # Esperamos comportamiento legacy: 5 unidades → 5 latas
    assert "5 lata" in result["display_qty"].lower(), (
        f"Atún sin density: {result['display_qty']!r}, esperado '5 latas' "
        "(comportamiento legacy preservado cuando density ausente)."
    )
