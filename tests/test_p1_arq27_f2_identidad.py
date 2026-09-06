# -*- coding: utf-8 -*-
"""[ARQ27-P1-05 · 2026-09-06] Identidad ≠ categoría comercial.

Cinco filas de nombre vegetal viven en la categoría **Lácteos** del catálogo: Leche de soya, de coco,
de avena, de almendras y Yogur de coco. La categoría es de TIENDA —dice en qué pasillo está— y el gap
pregunta si el motor la lee como verdad dietaria.

**Medido antes de tocar nada, sobre las 347 filas del catálogo vivo**: casi todo estaba ya bien. El
guard de dieta acertaba en las nueve pruebas, la durabilidad resolvía por nombre y no por categoría,
ninguna plantilla vegana ofrecía un lácteo real, y los alias no inflaban el recuento de identidades.

Quedaba **una** fila: `Yogur de coco` se declaraba `['lacteos', 'lactosa']`. La lista de excepciones
de `P1-PLANT-MILK-NOT-DAIRY` (06-sep) traía «yogur vegetal» y no el nombre real de la fila. Efecto:
un plato con yogur de coco caía fuera del pool de un alérgico a la leche por un alérgeno que no
tiene — el mismo daño en las dos direcciones que aquel arreglo describía.

Y era la **única discrepancia entre las dos capas**: lo que el registry DECLARA
(`allergen_classes_for`, que alimenta `intrinsic_risk_attributes` y por tanto el filtro
`exclude_allergens` del selector) y lo que el guard DECIDE (`_scan_allergen_violations`). Que
divergieran en silencio es el defecto de fondo; `test_las_dos_capas_de_lacteo_coinciden` es la parte
que impide la próxima.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import dish_registry as DR  # noqa: E402

VEGETALES_EN_LACTEOS = ["Leche de soya", "Leche de coco", "Leche de avena", "Leche de almendras",
                        "Yogur de coco"]
LACTEOS_DE_VERDAD = ["Leche", "Queso mozzarella", "Yogurt griego entero", "Mantequilla",
                     "Queso de hoja", "Crema agria", "Kéfir"]


def _viola_alergia_lacteos(nombre: str) -> bool:
    from graph_orchestrator import _scan_allergen_violations
    return bool(_scan_allergen_violations(
        {"days": [{"meals": [{"name": "_x", "ingredients": [nombre]}]}]}, ["lácteos"]))


# ── la fila que faltaba ───────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("nombre", VEGETALES_EN_LACTEOS)
def test_un_vegetal_en_el_pasillo_de_lacteos_no_es_lacteo(nombre):
    cls = DR.allergen_classes_for([nombre])
    assert "lacteos" not in cls and "lactosa" not in cls, f"{nombre} → {cls}"


@pytest.mark.parametrize("nombre", LACTEOS_DE_VERDAD)
def test_un_lacteo_de_verdad_sigue_siendolo(nombre):
    """La otra mitad, y la que de verdad importa: una excepción demasiado ancha convertiría a un
    alérgico en su propio guard."""
    assert "lacteos" in DR.allergen_classes_for([nombre]), nombre


@pytest.mark.parametrize("nombre", VEGETALES_EN_LACTEOS)
def test_la_bebida_vegetal_conserva_SUS_alergenos(nombre):
    """Quitar el lácteo no puede quitar lo demás: una bebida de almendras SIGUE siendo frutos secos y
    la de avena, gluten. Excusar la clase equivocada sería el mismo bug con otro signo."""
    esperado = {"Leche de soya": "soya", "Leche de avena": "gluten",
                "Leche de almendras": "frutos secos"}.get(nombre)
    cls = DR.allergen_classes_for([nombre])
    if esperado:
        assert esperado in cls, f"{nombre} perdió {esperado}: {cls}"


# ── el ancla que impide la próxima ────────────────────────────────────────────────────────────
def test_las_dos_capas_de_lacteo_coinciden():
    """Paridad entre lo que el registry DECLARA y lo que el guard DECIDE, sobre TODO el catálogo.

    Son dos caminos distintos hacia la misma pregunta: `allergen_classes_for` alimenta
    `intrinsic_risk_attributes` y con él el `exclude_allergens` del selector; `_scan_allergen_violations`
    decide en el backstop final. Divergieron en una fila y nadie se enteró, porque cada capa por
    separado parecía razonable. Éste es el test que las obliga a estar de acuerdo."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        import db_core
        db_core.connection_pool.open()
        from shopping_calculator import get_master_ingredients
        filas = list(get_master_ingredients() or [])
    except Exception:
        pytest.skip("catálogo no disponible en este entorno")
    if not filas:
        pytest.skip("catálogo vacío")
    discrepancias = []
    for r in filas:
        n = r.get("name")
        if not n:
            continue
        declara = "lacteos" in DR.allergen_classes_for([n])
        decide = _viola_alergia_lacteos(n)
        if declara != decide:
            discrepancias.append((n, "registry dice lácteo" if declara else "el scan dice lácteo"))
    assert not discrepancias, f"las dos capas no coinciden: {discrepancias}"


# ── lo que la medición confirmó que YA estaba bien ────────────────────────────────────────────
@pytest.mark.parametrize("nombre", VEGETALES_EN_LACTEOS)
def test_la_dieta_vegana_no_los_prohibe(nombre):
    from graph_orchestrator import _diet_pool_item_banned
    assert _diet_pool_item_banned(nombre, "vegan") is False, nombre


@pytest.mark.parametrize("nombre", LACTEOS_DE_VERDAD)
def test_la_dieta_vegana_si_prohibe_el_lacteo_real(nombre):
    from graph_orchestrator import _diet_pool_item_banned
    assert _diet_pool_item_banned(nombre, "vegan") is True, nombre


@pytest.mark.parametrize("nombre", ["Leche de soya", "Leche de coco", "Leche de avena", "Leche de almendras"])
def test_la_durabilidad_resuelve_por_nombre_no_por_categoria(nombre):
    """Si cayeran al default de la categoría `lacteos` serían frescos de 10 días; son envases estables
    de 365. La regla por nombre gana, que es lo correcto."""
    import pantry_durability as PD
    d = PD.classify(nombre, "Lácteos")
    assert d["cls"] == "pantry" and d["days_fresh"] >= 180, f"{nombre} → {d}"


def test_un_alias_no_aumenta_los_alimentos_unicos():
    """«Alias no aumenta alimentos únicos» — criterio de cierre del gap. 347 filas y ~1.300 alias
    siguen siendo 347 identidades."""
    from catalog_capability import catalog_capability
    s = catalog_capability("DO")
    if s is None:
        pytest.skip("catálogo no disponible")
    assert s["count"] == len(s["names"])
    assert len(s["aliases"]) > len(s["names"]), "los alias existen y NO se cuentan como alimentos"
