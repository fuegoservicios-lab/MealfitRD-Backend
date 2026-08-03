"""[P1-TRIP-WINDOWED-PERISHABLES · 2026-08-02] Perecederos del viaje activo, no promedio del plan.

Contexto: `get_shopping_list_delta` promedia TODOS los dias materializados y proyecta a 7
(`base_duration_scale = 7.0 / num_days`). En planes de 15/30 dias las semanas son
DELIBERADAMENTE distintas (freq-tracking cross-chunk), asi que la lista del viaje 1 traia
~1/N del pescado de la semana 3 (se dana en la nevera) y solo una fraccion del pollo que la
semana 1 SI cocina. El guard de coherencia era estructuralmente ciego: espeja la MISMA
formula (`P1-COHERENCE-DAY-BASIS`), asi que la divergencia se cancelaba a ambos lados.

El fix ventanea SOLO los PERECEDEROS a los 7 dias del viaje activo (los estables siguen
saliendo del agregado del periodo completo: se compran UNA vez para todo el ciclo) y espeja
el mismo ventaneo en el lado esperado del guard.

Los tests son 100% OFFLINE: `get_master_ingredients` se stubea a `[]` (el `.env` apunta a
produccion y el worktree no tiene `.env`; jamas dependemos de `master_ingredients` vivo).
Con catalogo vacio la clasificacion cae en `_PERISHABLE_NAME_HINTS`/`_STAPLE_NAME_HINTS`
(pollo/pescado -> perecedero, arroz/aceite -> estable), que es exactamente el camino que
corre en produccion cuando el item no resuelve a master.
"""
from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

import shopping_calculator as sc


# ---------------------------------------------------------------------------
# Fixtures de plan
# ---------------------------------------------------------------------------
def _day(protein: str, i: int, extra: list | None = None) -> dict:
    ings = [f"200 g de {protein}", "100 g de arroz"] + list(extra or [])
    return {
        "day": i,
        "day_name": f"Dia {i}",
        "date": f"2026-08-{2 + i:02d}",
        "meals": [{
            "meal": "Almuerzo",
            "name": f"Plato {i}",
            "ingredients": list(ings),
            "ingredients_raw": list(ings),
        }],
    }


def _plan_14d() -> dict:
    """14 dias materializados: semana 1 = pollo, semana 2 = pescado + aceite.

    `aceite` es ESTABLE y vive SOLO en la semana 2: es el caso que distingue
    "ventanear los perecederos" de "ventanear la lista entera".
    """
    dias = [_day("pollo", i) for i in range(1, 8)]
    dias += [_day("pescado", i, extra=["10 ml de aceite"]) for i in range(8, 15)]
    return {"days": dias, "grocery_start_date": "2026-08-03"}


def _by_name(items) -> dict:
    return {str(i.get("name", "")).lower(): i for i in items if isinstance(i, dict)}


def _delta(plan, **kw):
    with patch.object(sc, "get_master_ingredients", return_value=[]):
        return sc.get_shopping_list_delta(
            None, plan, True, False, True, 1.0,
            inventory_override=[], consumed_override=[], **kw
        )


# ---------------------------------------------------------------------------
# 1. El bug: el viaje 1 debe traer la semana 1, no el promedio del plan
# ---------------------------------------------------------------------------
def test_viaje_1_trae_la_semana_1_no_el_promedio():
    plan = _plan_14d()
    items = _delta(plan, window_days=plan["days"][:7])
    by_name = _by_name(items)

    pollo = by_name.get("pollo")
    assert pollo is not None, "el pollo de la semana 1 debe estar en el viaje 1"
    # 7 dias x 200 g = 1400 g. Sin ventana: 2800/14*7 = 700 g (el promedio del plan).
    assert pollo["base_qty"] >= 1300, (
        f"el viaje 1 debe traer ~1400 g de pollo, trajo {pollo['base_qty']} "
        f"(700 = promedio del plan)"
    )
    assert "pescado" not in by_name, (
        "el pescado de la semana 2 NO va en el viaje 1 (se dana antes de cocinarse)"
    )


def test_estable_de_la_semana_2_sobrevive_la_ventana():
    """Los ESTABLES se compran UNA vez para todo el ciclo: el aceite de la semana 2
    debe seguir en la lista aunque no se cocine esta semana."""
    plan = _plan_14d()
    items = _delta(plan, window_days=plan["days"][:7])
    by_name = _by_name(items)
    assert "aceite" in by_name, "el aceite (estable) NO se ventanea"
    # Arroz esta en los 14 dias: base del PERIODO (1400 g x 7/14 = 700), no de la ventana.
    arroz = by_name.get("arroz")
    assert arroz is not None
    assert arroz["base_qty"] == pytest.approx(700.0, rel=0.05), (
        f"el arroz (estable) conserva la base del plan completo, no la de la ventana "
        f"(base_qty={arroz['base_qty']})"
    )


def test_sin_window_days_el_comportamiento_es_el_de_hoy():
    """Default `None` = promedio del plan (los callers que no la pasan no cambian)."""
    plan = _plan_14d()
    by_name = _by_name(_delta(plan))
    assert "pescado" in by_name, "sin ventana, el pescado del promedio sigue apareciendo"
    assert by_name["pollo"]["base_qty"] == pytest.approx(700.0, rel=0.05)


def test_knob_off_es_byte_identico_al_promedio(monkeypatch):
    """Red de rollback: con el knob en False, pasar `window_days` no cambia NADA."""
    plan = _plan_14d()
    base = _delta(plan)
    monkeypatch.setenv("MEALFIT_TRIP_WINDOWED_PERISHABLES", "false")
    con_ventana = _delta(plan, window_days=plan["days"][:7])
    assert con_ventana == base


def test_plan_mas_corto_que_la_ventana_no_regresiona():
    """3 dias materializados (el caso de produccion del viaje 1): la ventana ES el
    plan completo -> mismo resultado que sin ventana."""
    plan = {"days": [_day("pollo", i) for i in range(1, 4)], "grocery_start_date": "2026-08-03"}
    base = _delta(plan)
    con_ventana = _delta(plan, window_days=plan["days"])
    assert con_ventana == base


def test_la_suma_de_los_viajes_cubre_el_plan_completo():
    """El riesgo del ventaneo seria convertir un problema de TIMING en uno de CANTIDAD
    total (el viaje 1 trae menos y ningun viaje posterior trae el resto). No ocurre: el
    plan es una ventana RODANTE — el shift poda los dias consumidos y deja `days[0]=hoy`,
    asi que el viaje k ve la semana k. Medido, el ventaneo cubre MEJOR que el promedio:
    el promedio deja el pollo de la semana 1 al 50% y compra 150% del pescado.
    """
    semana1 = [_day("pollo", i) for i in range(1, 8)]
    semana2 = [_day("pescado", i) for i in range(8, 15)]

    def _viajes(ventanear: bool):
        # Viaje 1: 14 dias materializados.
        p1 = {"days": semana1 + semana2, "grocery_start_date": "2026-08-03"}
        w1 = sc.active_trip_window_days(p1) if ventanear else None
        t1 = _delta(p1, window_days=w1)
        # Viaje 2: tras el shift solo queda la semana 2 (los dias consumidos se podan).
        p2 = {"days": semana2, "grocery_start_date": "2026-08-10"}
        w2 = sc.active_trip_window_days(p2) if ventanear else None
        t2 = _delta(p2, window_days=w2)
        return _by_name(t1), _by_name(t2)

    def _q(d, food):
        return float(d.get(food, {}).get("base_qty", 0.0)) if food in d else 0.0

    v1, v2 = _viajes(True)
    assert _q(v1, "pollo") + _q(v2, "pollo") == pytest.approx(1400.0, rel=0.05)
    assert _q(v1, "pescado") + _q(v2, "pescado") == pytest.approx(1400.0, rel=0.05)
    # Estable: los dos viajes lo compran con la base del periodo (sin cambio).
    assert _q(v1, "arroz") + _q(v2, "arroz") == pytest.approx(1400.0, rel=0.05)

    # Contraste con el comportamiento previo: infra-compra la semana 1 y sobre-compra la 2.
    o1, o2 = _viajes(False)
    assert _q(o1, "pollo") + _q(o2, "pollo") == pytest.approx(700.0, rel=0.05)
    assert _q(o1, "pescado") + _q(o2, "pescado") == pytest.approx(2100.0, rel=0.05)


def test_el_hibrido_no_resucita_el_perecedero_de_la_semana_2():
    """Razon por la que la ventana va TAMBIEN a las llamadas de 15/30 dias:
    `_build_hybrid_shopping_list` incluye los items que estan SOLO en la lista de
    periodo. Si el periodo no se ventanea, el pescado de la semana 2 vuelve a la lista
    del viaje 1 con la cantidad del CICLO ENTERO — peor que el bug original.
    """
    plan = _plan_14d()
    win = plan["days"][:7]
    s7 = _delta(plan, window_days=win)
    with patch.object(sc, "get_master_ingredients", return_value=[]):
        s15 = sc.get_shopping_list_delta(
            None, plan, True, False, True, sc.cycle_qty_multiplier("biweekly"),
            inventory_override=[], consumed_override=[],
            cycle_days=sc.cycle_days_for_duration("biweekly"), window_days=win,
        )
    hybrid = sc._build_hybrid_shopping_list(s7, s15)
    nombres = {str(i.get("name", "")).lower() for i in hybrid}
    assert "pescado" not in nombres, f"el hibrido resucito el pescado de la semana 2: {nombres}"
    assert "pollo" in nombres and "aceite" in nombres


# ---------------------------------------------------------------------------
# 2. Helper de derivacion de la ventana
# ---------------------------------------------------------------------------
def test_helper_deriva_la_ventana_del_viaje_activo():
    plan = _plan_14d()
    win = sc.active_trip_window_days(plan)
    assert win is not None
    assert len(win) == 7
    assert win[0]["day"] == 1 and win[-1]["day"] == 7


def test_helper_none_cuando_el_plan_cabe_en_la_ventana():
    """<=7 dias materializados -> None (no-op explicito: ventana == plan completo)."""
    plan = {"days": [_day("pollo", i) for i in range(1, 4)]}
    assert sc.active_trip_window_days(plan) is None
    assert sc.active_trip_window_days({"days": []}) is None
    assert sc.active_trip_window_days({}) is None


def test_helper_respeta_el_knob(monkeypatch):
    monkeypatch.setenv("MEALFIT_TRIP_WINDOWED_PERISHABLES", "false")
    assert sc.active_trip_window_days(_plan_14d()) is None


def test_helper_ancla_en_grocery_start_date_cuando_days0_ya_paso():
    """El shift reescribe `grocery_start_date` a HOY siguiendo a `days[0]`. Si las
    fechas del plan arrancan despues del ancla, la ventana sigue las FECHAS."""
    plan = _plan_14d()
    plan["grocery_start_date"] = "2026-08-06"  # arranca en el dia 4
    win = sc.active_trip_window_days(plan)
    assert win is not None
    assert win[0]["date"] == "2026-08-06"
    assert len(win) == 7
    assert win[-1]["date"] == "2026-08-12"


# ---------------------------------------------------------------------------
# 3. Espejo del guard (obligatorio: sin el, divergencias falsas masivas)
# ---------------------------------------------------------------------------
def test_guard_no_fabrica_divergencias_con_lista_ventaneada():
    plan = _plan_14d()
    items = _delta(plan, window_days=plan["days"][:7])
    plan_result = dict(plan)
    plan_result["aggregated_shopping_list_weekly"] = items
    plan_result["aggregated_shopping_list"] = items
    plan_result["calc_household_multiplier"] = 1.0

    with patch.object(sc, "get_master_ingredients", return_value=[]):
        divs = sc.run_shopping_coherence_guard(plan_result, mode_override="warn")

    faltantes = {d["food"] for d in divs if d.get("side") == "expected_only"}
    assert not any("pescado" in f.lower() for f in faltantes), (
        f"el guard NO debe pedir el pescado de la semana 2 en una lista ventaneada: {divs}"
    )
    magnitudes = {d["food"]: d for d in divs if d.get("magnitude")}
    assert not any("pollo" in f.lower() for f in magnitudes), (
        f"el pollo ventaneado (1400 g) debe casar con el esperado ventaneado: {magnitudes}"
    )


def test_el_espejo_del_guard_es_load_bearing_no_cosmetico():
    """Prueba de que el espejo HACE algo (no un parser test): con el espejo
    neutralizado, el guard fabrica `Pescado expected_only` (hipotesis
    `cap_swallowed_modifier`, una de las SEVERAS que escalan warn->block en
    P2-COHERENCE-1 -> retry forzado) + `Pollo` con magnitud al 100% de delta."""
    plan = _plan_14d()
    items = _delta(plan, window_days=plan["days"][:7])
    plan_result = dict(plan)
    plan_result["aggregated_shopping_list_weekly"] = items
    plan_result["calc_household_multiplier"] = 1.0

    with patch.object(sc, "get_master_ingredients", return_value=[]):
        con_espejo = sc.run_shopping_coherence_guard(plan_result, mode_override="warn")
        with patch.object(sc, "_mirror_trip_window_expected", lambda p, e, **k: e):
            sin_espejo = sc.run_shopping_coherence_guard(plan_result, mode_override="warn")

    assert con_espejo == [], f"con espejo no debe haber divergencias: {con_espejo}"
    hipotesis = {d.get("food"): d.get("hypothesis") for d in sin_espejo}
    assert hipotesis.get("Pescado") == "cap_swallowed_modifier", (
        f"sin espejo el guard DEBE fabricar la divergencia (si no, el espejo es "
        f"decorativo y este test no prueba nada): {sin_espejo}"
    )
    assert any(d.get("magnitude") and "pollo" in str(d.get("food")).lower()
               for d in sin_espejo), sin_espejo


def test_guard_sigue_ciego_sin_ventana():
    """Sin ventana la lista NO lleva el sello -> el guard usa la base de siempre."""
    plan = _plan_14d()
    items = _delta(plan)
    plan_result = dict(plan)
    plan_result["aggregated_shopping_list_weekly"] = items
    plan_result["calc_household_multiplier"] = 1.0
    with patch.object(sc, "get_master_ingredients", return_value=[]):
        divs = sc.run_shopping_coherence_guard(plan_result, mode_override="warn")
    magnitudes = {d["food"].lower() for d in divs if d.get("magnitude")}
    assert not any("pollo" in f for f in magnitudes), (
        f"regresion: sin ventana el guard ya cuadraba y debe seguir cuadrando: {divs}"
    )


def test_lista_ventaneada_lleva_el_sello():
    """El sello `trip_window_days` es lo que permite al guard (y a un lector futuro)
    saber que esta lista NO es el promedio del plan."""
    plan = _plan_14d()
    items = _delta(plan, window_days=plan["days"][:7])
    assert items, "la lista no puede salir vacia"
    assert all(i.get("trip_window_days") == 7 for i in items), (
        "todos los items de una lista ventaneada llevan el sello"
    )
    sin_ventana = _delta(plan)
    assert not any("trip_window_days" in i for i in sin_ventana)


# ---------------------------------------------------------------------------
# 4. Contrato de firma + knob
# ---------------------------------------------------------------------------
def test_window_days_es_keyword_only_y_default_none():
    p = inspect.signature(sc.get_shopping_list_delta).parameters["window_days"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY
    assert p.default is None


def test_knob_registrado_en_el_registry():
    from knobs import get_knobs_registry_snapshot
    sc._trip_windowed_perishables_enabled()
    assert "MEALFIT_TRIP_WINDOWED_PERISHABLES" in get_knobs_registry_snapshot()


def test_marker_inline_presente():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "shopping_calculator.py").read_text(encoding="utf-8")
    assert "P1-TRIP-WINDOWED-PERISHABLES" in src
