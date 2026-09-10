# -*- coding: utf-8 -*-
"""[P1-RETINOL-PREFORMADO · 2026-09-09] Un techo que sólo vigila a quien sube no es un techo.

## De dónde sale

El dueño preguntó por qué había dejado cosas abiertas. Al medir la variedad sobre el ciclo REAL de
30 días (había medido 14) apareció que **un plato que yo mismo metí esta mañana** salía 8 veces en
30 días: «Hígado de res encebollado con plátano hervido».

```
Hígado de res .............. 4.970 mcg RAE / 100 g
la ración de la plantilla ... 120 g  →  5.964 mcg
UL de vitamina A (IOM) ...... 3.000 mcg/día
                              => una sola ración = 2× el techo DIARIO, ocho veces al mes
```

Ni el escáner culinario ni el backstop clínico lo vieron: `verifica_comida` devolvía 0 violaciones.

## El techo ya existía, y miraba en una sola dirección

`graph_orchestrator._MICRO_CLOSER_UL` tiene `"vit_a_mcg": 3000.0` desde julio, y su propio
comentario dice para qué: *«defensa contra escalar hígado/vísceras al cerrar hierro»*. Es un techo
sobre el **cerrador de micronutrientes** — impide que un pase SUBA un ingrediente. Un plato que nace
por encima pasa por debajo del radar.

## La distinción que hace correcto este guard, y que casi me como

El UL es de **retinol PREFORMADO**. El beta-caroteno de la auyama o la zanahoria **no intoxica** —
el cuerpo regula su conversión— y el catálogo guarda ambos en la misma columna, que es RAE total.

Medido sobre 18 días de planes vivos: **2 pasaban el UL, y los dos por AUYAMA** (uno sumó 4.622 mcg
entre dos líneas de 600 y 485 g; el otro 3.004 entre auyama y melón). Un techo
sobre RAE total los habría rechazado: habría roto cocina dominicana legítima para «arreglar» algo
que no está roto, y encima habría dado la sensación de estar protegiendo.

*Un guard que no distingue la fuente no mide el riesgo: mide una columna.*

## Lo que se hizo con el plato

Se **retiró**, no se le bajó la ración. Incluso ~60 g agotan el techo del día entero en un solo
plato, y el selector no tiene memoria entre días: sirve ocho veces al mes lo que mejor puntúa, y el
hígado puntúa altísimo en proteína-por-caloría (15,1 g/100 kcal) **y** en precio (RD$119/lb).
«Barato + proteico + tóxico en exceso» es justo la combinación que peor lleva un selector ciego a la
frecuencia. Si vuelve algún día, vuelve CON esa maquinaria.
"""
import pytest

import deterministic_day as dd


def _cat():
    return {
        "Hígado de res": {"vitamin_a_mcg_rae_per_100g": 4970, "kcal_per_100g": 135,
                          "protein_g_per_100g": 20.4, "carbs_g_per_100g": 3.9,
                          "fats_g_per_100g": 3.6, "fiber_g_per_100g": 0},
        "Auyama": {"vitamin_a_mcg_rae_per_100g": 426, "kcal_per_100g": 26,
                   "protein_g_per_100g": 1, "carbs_g_per_100g": 6.5,
                   "fats_g_per_100g": 0.1, "fiber_g_per_100g": 0.5},
        "Zanahoria": {"vitamin_a_mcg_rae_per_100g": 835, "kcal_per_100g": 41,
                      "protein_g_per_100g": 0.9, "carbs_g_per_100g": 9.6,
                      "fats_g_per_100g": 0.2, "fiber_g_per_100g": 2.8},
        "Pechuga de pollo": {"vitamin_a_mcg_rae_per_100g": 9, "kcal_per_100g": 107,
                             "protein_g_per_100g": 22.5, "carbs_g_per_100g": 0,
                             "fats_g_per_100g": 2.6, "fiber_g_per_100g": 0},
    }


def _meal(*lineas):
    return {"meal": "Cena", "name": "prueba", "ingredients": [f"{g:g} g de {n}" for g, n in lineas],
            "ingredients_raw": [f"{g:g} g de {n}" for g, n in lineas], "recipe": ["paso"]}


# ── El caso que lo originó ───────────────────────────────────────────────────
def test_la_racion_de_higado_que_meti_hoy_se_rechaza():
    ret = dd._retinol_preformado_mcg(_meal((120, "Hígado de res")), _cat())
    assert ret == pytest.approx(5964, abs=1)
    assert ret > dd._UL_RETINOL_MCG * 1.9, "el escenario dejó de demostrar el exceso; revísalo"
    viol = dd.verifica_comida(_meal((120, "Hígado de res")), {}, _cat())
    assert any("retinol" in str(v).lower() for v in viol), (
        "una ración de 2× el UL diario volvió a pasar sin que nadie la mire")


def test_una_racion_pequena_de_higado_pasa():
    """El guard acota la ración, no prohíbe el alimento: 50 g están por debajo del techo."""
    assert not any("retinol" in str(v).lower()
                   for v in dd.verifica_comida(_meal((50, "Hígado de res")), {}, _cat()))


# ── LA frontera: la fuente, no la columna ────────────────────────────────────
def test_la_auyama_NO_dispara_el_techo_aunque_su_RAE_lo_pase():
    """Los 2 días vivos que pasaban el UL eran auyama. Beta-caroteno no intoxica.

    Si esto cae, el guard volvió a mirar la columna en vez de la fuente y está rechazando cocina
    dominicana legítima.
    """
    plato = _meal((800, "Auyama"))
    rae = 426 * 800 / 100
    assert rae > dd._UL_RETINOL_MCG, "el escenario ya no reproduce el falso positivo que evitamos"
    assert dd._retinol_preformado_mcg(plato, _cat()) == 0.0
    assert not any("retinol" in str(v).lower() for v in dd.verifica_comida(plato, {}, _cat()))


@pytest.mark.parametrize("nombre,gramos", [("Auyama", 900), ("Zanahoria", 500),
                                           ("Pechuga de pollo", 300)])
def test_ninguna_fuente_vegetal_ni_carne_magra_dispara(nombre, gramos):
    assert dd._retinol_preformado_mcg(_meal((gramos, nombre)), _cat()) == 0.0


def test_el_vocabulario_cubre_las_visceras_y_se_queda_corto_a_proposito():
    """La lista nombra vísceras; ampliarla a «carne» o «res» convertiría el guard en ruido."""
    for v in ("higado", "viscera", "mondongo", "molleja"):
        assert any(v in t for t in dd._RETINOL_ANIMAL), f"se perdió {v!r} del vocabulario"
    for falso in ("carne", "res", "pollo", "auyama", "zanahoria", "batata"):
        assert not any(t == falso for t in dd._RETINOL_ANIMAL), (
            f"{falso!r} entró en la lista de vísceras: el guard pasa a rechazar comida normal")


# ── El techo no es un cuarto número ──────────────────────────────────────────
def test_el_UL_es_el_MISMO_que_ya_tenia_el_repo():
    """`_MICRO_CLOSER_UL["vit_a_mcg"]` existe desde julio y dice 3000. Escribir otro número aquí
    es la lección de `P1-DIET-CANON-SSOT`: tres tablas, drifearon, y la del filtro servía pollo a
    vegetarianas."""
    import graph_orchestrator as go

    assert dd._UL_RETINOL_MCG == pytest.approx(float(go._MICRO_CLOSER_UL["vit_a_mcg"])), (
        "el techo del plato se separó del techo del cerrador de micros")


def test_el_plato_retirado_no_volvio_al_catalogo():
    """Se retiró a sabiendas: incluso ~60 g agotan el techo del día y el selector no tiene memoria
    entre días. Si alguien lo repone, que sea con la maquinaria de frecuencia."""
    import json
    import os

    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "registry", "dish_registry_do_v1.json")
    if not os.path.exists(p):
        pytest.skip("sin snapshot DO")
    with open(p, encoding="utf-8") as f:
        nombres = {t["name"] for t in (json.load(f).get("templates") or [])}
    assert "Hígado de res encebollado con plátano hervido" not in nombres, (
        "volvió el plato de hígado: medido, el selector lo servía 8 veces en 30 días a 2× el UL")


def test_ninguna_plantilla_viva_pasa_el_techo_en_una_racion():
    """Barrido del catálogo entero: ninguna plantilla nace por encima del UL de retinol."""
    import json
    import os

    _b = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(_b, "data", "registry", "dish_registry_do_v1.json")
    if not os.path.exists(p):
        pytest.skip("sin snapshot DO")
    with open(p, encoding="utf-8") as f:
        snap = json.load(f)
    malas = []
    for t in snap.get("templates") or []:
        if t.get("status") != "ok":
            continue
        meal = _meal(*[(float(c.get("grams") or 0), str(c.get("name") or ""))
                       for c in (t.get("constituents") or [])])
        # el catálogo de prueba sólo trae hígado: basta, es la única fuente animal del corpus DO
        if dd._retinol_preformado_mcg(meal, _cat()) > dd._UL_RETINOL_MCG:
            malas.append(t["name"])
    assert not malas, f"plantillas que nacen por encima del UL de retinol: {malas}"
