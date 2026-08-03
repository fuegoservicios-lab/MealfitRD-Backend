"""[P1-VEG-BACKFILL-HONESTY · 2026-08-02] Planes VIVOS cuya receta (texto) contradice su propia
lista de compras, sin ninguna nota — residuo de datos YA PERSISTIDOS que Task 7
(P1-RECONCILE-CDA-DENSITY) no toca (esa arregla el lado GENERADOR para planes NUEVOS).

Evidencia medida en producción (SELECT, sin escritura):
  - plan `5f4bb17e`: receta dice "600 g de espárragos" en la cena; la lista SEMANAL compra
    583.33 g — una sola cena agota el 103% de la compra de la semana. `capped_by=null`, sin aviso:
    ningún cap conocido (`_CAPS_APPLIED_LAST_RUN`) explica el déficit porque espárragos no está en
    `_VEG_PER_WEEK_PER_PERSON` (P5-VEG-CAP) ni en ningún otro cap por categoría — el número
    simplemente llegó corto y nadie lo dijo.
  - plan `8d3f246a`: "470 g de tayota" vs lista 891 g (ratio 0.635 lado guard).
  - plan `cf3a81fb`: vainitas 933 g (=400×7/3, inflación de solver entregada completa) + calabacín
    1372 g/persona.

## El mecanismo (dos entregables, decisión ya tomada — ver task-8-brief.md)

(a) Mecanismo PERMANENTE en el agregador: cuando la cantidad final que `apply_smart_market_units`
    resuelve para un ítem (`base_qty`, en gramos) queda por debajo de `MEALFIT_QTY_SHORTFALL_
    NOTE_MIN` (default 0.9) veces la demanda que las RECETAS piden (`text_demand_g`, mismo parse
    que usa el guard — threaded desde `get_shopping_list_delta` vía `expected_sum_from_recipes` +
    `_normalize_food_units_to_base`), se estampa un `capped_by="qty_reconcile_v7"` SINTÉTICO. Ese
    `capped_by` alimenta el MISMO `_cap_hit` que P1-CAPPED-STAPLE-HONESTY (2026-07-26) ya consume
    para componer la nota "alcanza ~N de 30 días — recompra" — no se reimplementa el copy.

(b) Script one-shot `scripts/backfill_veg_lines_v7.py` (dry-run por defecto, NO ejecutado en esta
    sesión — DB de prod, ver header del script) para los planes YA persistidos con el defecto.

## Composición con P1-SKU-COVER-HONESTY (Task 5, mismo día)

Task 5 ya excluye de SU sufijo los ítems que traen `capped_by` (para no duplicar "alcanza...alcanza").
Estampar el `capped_by` sintético ANTES de que corra el bloque de Task 5 (mismo orden: el lookup de
`_cap_hit` corre completo, INCLUYENDO nuestro fallback, antes de que el bloque de
`pkg_cover_ratio` decida si sufija) cierra el círculo: la nota que se muestra es SIEMPRE la de
P1-CAPPED-STAPLE-HONESTY (el "dueño" histórico del copy), nunca duplicada.

## Riesgos verificados explícitamente (ver pedido de la tarea)

  1. Un ítem que YA tiene `capped_by` REAL (post<pre estricto) no debe ser pisado por el sintético
     — `_cap_hit is None` guardia la rama nueva (`test_no_pisa_un_cap_real_ya_registrado`).
  2. Nunca dos sufijos "alcanza" en el mismo `display_qty` (`test_no_duplica_sufijo_con_sku_cover_honesty`).
  3. Un ítem cuya compra SÍ cubre el texto (>=90%) no recibe nada (`test_compra_suficiente_no_recibe_nota`).
"""
from __future__ import annotations

import pytest

import shopping_calculator as sc
from knobs import _env_float, get_knobs_registry_snapshot

CARTON = {"market_container": "cartón", "container_weight_g": 946.0}


def _por_gramos(name, grams, master_item=None, text_demand_g=None, cycle_days=7):
    """Adapta la firma real `apply_smart_market_units(name, weight_in_lbs, unit_str, raw_qty,
    master_item, cycle_days, text_demand_g)` a una necesidad expresada en gramos — mismo patrón
    que `test_p1_sku_cover_honesty.py::_por_gramos`."""
    lbs = grams / 453.592
    return sc.apply_smart_market_units(
        name, lbs, "g", grams, master_item=master_item, cycle_days=cycle_days,
        text_demand_g=text_demand_g,
    )


# ───────────── 1. el mecanismo dispara cuando NINGÚN cap real explica el déficit ─────────────

def test_dispara_cuando_compra_queda_bajo_90pct_del_texto():
    """Espejo del caso medido en prod (espárragos 600 g × 7 días = 4200 g de demanda; la lista
    entrega solo 3000 g, 71% — muy por debajo del 90%). Sin ningún cap registrado
    (`_CAPS_APPLIED_LAST_RUN` vacío), el mecanismo nuevo debe ser el ÚNICO que explique el
    faltante."""
    sc.reset_caps_applied_last_run()
    item = _por_gramos("Espárragos", 3000.0, text_demand_g=4200.0)
    assert item.get("capped_by") == "qty_reconcile_v7"
    assert "alcanza" in item.get("display_qty", ""), item.get("display_qty")
    assert "alcanza" in item.get("display_string", "")


def test_compra_suficiente_no_recibe_nota():
    """[Riesgo 3] Si lo comprado YA cubre el 90% (o más) de lo que el texto exige, no hay nada
    que avisar — silencio es la respuesta correcta, no un bug."""
    sc.reset_caps_applied_last_run()
    item = _por_gramos("Espárragos", 4000.0, text_demand_g=4200.0)  # 95.2%
    assert item.get("capped_by") is None
    assert "alcanza" not in item.get("display_qty", "")


def test_sin_text_demand_g_es_no_op():
    """Callers que no pasan `text_demand_g` (la mayoría hoy — el plumbing es nuevo) deben ver
    comportamiento previo byte-idéntico: sin el dato de referencia, no hay base para comparar."""
    sc.reset_caps_applied_last_run()
    item = _por_gramos("Espárragos", 100.0, text_demand_g=None)
    assert item.get("capped_by") is None
    assert "alcanza" not in item.get("display_qty", "")


def test_umbral_respeta_el_knob(monkeypatch):
    """Con el knob relajado a 0.5, un déficit del 71% (0.71 >= 0.5) ya NO dispara."""
    monkeypatch.setattr(sc, "QTY_SHORTFALL_NOTE_MIN", 0.5)
    sc.reset_caps_applied_last_run()
    item = _por_gramos("Espárragos", 3000.0, text_demand_g=4200.0)
    assert item.get("capped_by") is None


# ───────────── 2. [Riesgo 1] no pisa un cap real ya registrado ─────────────

def test_no_pisa_un_cap_real_ya_registrado():
    """Si `_CAPS_APPLIED_LAST_RUN` ya explica el déficit con post<pre estricto (cap real de
    almacenaje, ej. P5-VEG-CAP), el `capped_by` debe seguir siendo la razón REAL — el fallback
    sintético sólo debe activarse cuando NINGÚN cap real lo explicó (`_cap_hit is None`)."""
    sc.reset_caps_applied_last_run()
    try:
        sc._record_cap_applied("Cebolla", 2000.0, 600.0, "P5-VEG-CAP")
        item = _por_gramos("Cebolla", 600.0, text_demand_g=4200.0)  # dispararía el sintético solo
        assert item.get("capped_by") == "P5-VEG-CAP", (
            f"el cap sintético piso un cap REAL: capped_by={item.get('capped_by')!r}")
    finally:
        sc.reset_caps_applied_last_run()


# ───────────── 3. [Riesgo 2] composición con P1-SKU-COVER-HONESTY (Task 5) ─────────────

def test_no_duplica_sufijo_con_sku_cover_honesty():
    """Un ítem con envase (`market_packages`/`container_weight_g`) que ADEMÁS calificaría para el
    aviso de Task 5 (`pkg_cover_ratio < PKG_COVER_NOTE_MIN`) no debe terminar con DOS sufijos
    "alcanza" — Task 5 ya se abstiene cuando `capped_by` viene seteado (decisión #3); nuestro
    synthetic cap debe ser justamente lo que dispara esa abstención."""
    sc.reset_caps_applied_last_run()
    # Envase que deliberadamente sub-cubre (cover bajo) PARA que Task 5 evaluaría sufijar si no
    # fuera por nuestro capped_by sintético.
    master = {"market_container": "cartón", "container_weight_g": 200.0}
    item = _por_gramos("Espárragos", 3000.0, master_item=master, text_demand_g=4200.0)
    assert item.get("capped_by") == "qty_reconcile_v7"
    assert item.get("display_qty", "").count("alcanza") == 1, item.get("display_qty")
    assert item.get("display_string", "").count("alcanza") == 1, item.get("display_string")


# ───────────── 4. knob ─────────────

def test_knob_registrado_con_default_09():
    assert sc.QTY_SHORTFALL_NOTE_MIN == pytest.approx(0.9)
    reg = get_knobs_registry_snapshot()
    assert "MEALFIT_QTY_SHORTFALL_NOTE_MIN" in reg
    assert reg["MEALFIT_QTY_SHORTFALL_NOTE_MIN"]["default"] == pytest.approx(0.9)


def test_knob_clamp_rechaza_fuera_de_rango():
    import os
    os.environ["MEALFIT_QTY_SHORTFALL_NOTE_MIN_TEST_OOR"] = "1.5"
    try:
        val = _env_float("MEALFIT_QTY_SHORTFALL_NOTE_MIN_TEST_OOR", 0.9, lambda v: 0.0 < v <= 1.0)
        assert val == pytest.approx(0.9), "1.5 excede el clamp (0,1] -> debe caer al default"
    finally:
        del os.environ["MEALFIT_QTY_SHORTFALL_NOTE_MIN_TEST_OOR"]


# ───────────── 5. plumbing (source-level, offline) ─────────────

def test_plumbing_text_demand_g_parametro_en_apply_smart_market_units():
    import inspect
    sig = inspect.signature(sc.apply_smart_market_units)
    assert "text_demand_g" in sig.parameters
    assert sig.parameters["text_demand_g"].default is None


def test_plumbing_text_demand_g_threading_source():
    """`text_demand_g` se propaga desde `get_shopping_list_delta` -> `aggregate_and_deduct_
    shopping_list` -> los 2 call-sites de `apply_smart_market_units`. Parser-based (espejo de
    `test_cycle_days_plumbing_local` en test_p1_sku_cover_honesty.py) — anclado a texto para que
    un rename futuro falle el test antes que producción."""
    from pathlib import Path
    src = Path(sc.__file__).resolve().read_text(encoding="utf-8")

    i = src.index("def aggregate_and_deduct_shopping_list(")
    sig_end = src.index(")", i)
    assert "text_demand_g_map" in src[i:sig_end + 1]

    assert "apply_smart_market_units(name, weight_in_lbs, 'lb', 0.0, master_item, cycle_days=_cycle_days_for_note, text_demand_g=" in src
    assert "apply_smart_market_units(name, 0.0, u, q, master_item, cycle_days=_cycle_days_for_note, text_demand_g=" in src

    j = src.index("def get_shopping_list_delta(")
    j_end = src.index("\ndef ", j + 10)
    assert "expected_sum_from_recipes(" in src[j:j_end]
    assert "text_demand_g_map=" in src[j:j_end]


# ───────────── 6. end-to-end dinámico (get_shopping_list_delta), wiring real ─────────────

def test_e2e_shopping_list_delta_dispara_nota_cuando_texto_infla(monkeypatch):
    """Prueba el WIRING real (no sólo el mecanismo unitario): un plan de 1 día con "600 g de
    espárragos" en una cena, proyectado ×7 (`get_shopping_list_delta` con 1 día materializado
    proyecta a semana). Sin catálogo (worktree sin .env → `get_master_ingredients()` devuelve
    []), la aritmética real del aggregator reproduce fielmente el texto (sin caps ni pantry) — así
    que forzamos la divergencia monkeypencheando `expected_sum_from_recipes` para simular el caso
    real de producción donde el texto exige más de lo que la lista, por la razón que sea, terminó
    comprando. Esto ejercita: (a) `get_shopping_list_delta` arma `text_demand_g_map` desde
    `expected_sum_from_recipes` + `_normalize_food_units_to_base`, (b) lo threadea a través de
    `aggregate_and_deduct_shopping_list`, (c) `apply_smart_market_units` lo consume y estampa la
    nota."""
    plan = {"days": [{"meals": [{
        "ingredients": ["600 g de espárragos"],
        "ingredients_raw": ["600 g de espárragos"],
    }]}]}

    def _fake_expected(plan_data, *, apply_yield=False, multiplier=1.0):
        return {"Espárragos": {"g": 999999.0 * multiplier}}

    monkeypatch.setattr(sc, "expected_sum_from_recipes", _fake_expected)
    sc.reset_caps_applied_last_run()
    items = sc.get_shopping_list_delta(None, plan, True, False, True, 1.0)
    esp = next(i for i in items if "esp" in str(i.get("name", "")).lower())
    assert esp.get("capped_by") == "qty_reconcile_v7", esp
    assert "alcanza" in esp.get("display_qty", ""), esp.get("display_qty")


def test_e2e_shopping_list_delta_sin_divergencia_no_lleva_nota(monkeypatch):
    """Contrapartida honesta del test anterior: si `expected_sum_from_recipes` (mismo parse del
    guard) NO diverge de lo que el aggregator resolvió, no debe aparecer ninguna nota — el
    mecanismo nuevo no es ruidoso por default."""
    plan = {"days": [{"meals": [{
        "ingredients": ["600 g de espárragos"],
        "ingredients_raw": ["600 g de espárragos"],
    }]}]}
    sc.reset_caps_applied_last_run()
    items = sc.get_shopping_list_delta(None, plan, True, False, True, 1.0)
    esp = next(i for i in items if "esp" in str(i.get("name", "")).lower())
    assert esp.get("capped_by") is None, esp
    assert "alcanza" not in esp.get("display_qty", "")
