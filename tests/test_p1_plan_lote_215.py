# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-215 · 2026-09-24] Compra única de 30 días: la lista del día 1 alcanza para TODO el ciclo.

Medido en el plan real del dueño (6594aae1, 30 días, sin reposición de frescos ni congelador): la lista extrapolaba ×10
los 3 días generados —2 lb de pechuga y 32 oz de pescado que aguantan 3 días, 60 huevos «alcanza ~13 de 30 días»— y no
traía ninguna proteína de despensa para los días 8-30. Ahora la fuente de días de la lista (`shopping_source_days`)
proyecta el ciclo: los días reales + los que faltan, copiados en rueda y pasados por la sustitución de duraderos. La
lista, el lado esperado del guard y su base de días leen el mismo mes, y la híbrida no parte la compra en semanas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import compra_unica as cu  # noqa: E402
import shopping_calculator as sc  # noqa: E402

SINGLE = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none", "batch_cooking": "never"},
          "diet": {"type": "balanced", "allergies": []}}
WEEKLY = {"shopping": {"main_cycle_days": 7, "fresh_topup_days": None, "freezer_mode": "limited"},
          "diet": {"type": "balanced", "allergies": []}}


@pytest.fixture(autouse=True)
def _sin_rastro_de_compra_unica():
    """`set_single_trip_notes` es estado de módulo: que la última lista de compra única de este fichero no le cambie
    la cola de las notas («recompra») a los tests de otros ficheros que corran después en el mismo proceso."""
    yield
    sc.set_single_trip_notes(False)
    cu._MEMO.clear()


def _dia(n, comidas):
    return {"day": n, "meals": [{"meal": m, "name": nombre, "ingredients": list(ings), "ingredients_raw": list(ings)}
                                for m, nombre, ings in comidas]}


def _plan(politica=SINGLE):
    return {
        "total_days_requested": 30,
        "_plan_policy": {"effective": politica},
        "days": [
            _dia(1, [("Desayuno", "Avena con lechosa", ["40 g de avena", "85 g de lechosa"]),
                     ("Almuerzo", "Pollo guisado", ["200 g de pechuga de pollo", "1 taza de arroz blanco"]),
                     ("Cena", "Habichuelas con batata", ["¾ taza de habichuelas negras", "100 g de batata"])]),
            _dia(2, [("Desayuno", "Huevos con casabe", ["3 huevos", "1 torta pequeña de casabe"]),
                     ("Almuerzo", "Pescado a la criolla", ["150 g de filete de pescado blanco", "1 taza de arroz blanco"]),
                     ("Cena", "Mangú", ["130 g de plátano verde", "½ cebolla"])]),
            _dia(3, [("Desayuno", "Avena con fresas", ["40 g de avena", "50 g de fresas"]),
                     ("Almuerzo", "Arroz con lentejas", ["½ taza de lentejas", "1 taza de arroz blanco"]),
                     ("Cena", "Yuca con huevo", ["200 g de yuca", "2 huevos"])]),
        ],
    }


def _todas(dias):
    return [x for d in dias for m in d["meals"] for x in m["ingredients_raw"]]


def test_proyecta_el_ciclo_entero_con_duraderos_despues_de_la_semana_de_frescos():
    p = _plan()
    dias = sc.shopping_source_days(p)
    assert len(dias) == 30
    assert dias[:3] == p["days"], "los días reales van primero y sin tocar"
    semana = _todas(dias[3:7])
    resto = _todas(dias[7:])
    assert any("pechuga de pollo" in x for x in semana) and any("fresas" in x for x in semana), "lo fresco, en su semana"
    for fresco in ("pollo", "pescado", "lechosa", "fresas"):
        assert not any(fresco in cu._sa(x) for x in resto), (fresco, [x for x in resto if fresco in cu._sa(x)])
    # el plátano verde aguanta 10 días; después, batata
    assert not any("platano" in cu._sa(x) for x in _todas(dias[10:]))
    # la rueda de la proteína reparte los tres duraderos (con «día + comida» cada día copiado recibía siempre el mismo)
    assert any("atun en agua" in x for x in resto) and any("sardinas en lata" in x for x in resto)
    assert any("garbanzos cocidos" in x for x in resto)
    assert any("batata" in x for x in resto) and any("manzana" in x for x in resto)
    # la yuca aguanta hasta el día 21 y después pasa a batata
    assert any("yuca" in x for x in _todas(dias[7:21])) and not any("yuca" in x for x in _todas(dias[21:]))
    assert all(d.get("_proyectado") for d in dias[3:])


def test_sin_compra_unica_o_con_el_ciclo_completo_no_proyecta():
    semanal = _plan(WEEKLY)
    assert sc.shopping_source_days(semanal) == semanal["days"]
    completo = _plan()
    completo["days"] = (completo["days"] * 10)[:30]
    assert len(sc.shopping_source_days(completo)) == 30 and not any(
        d.get("_proyectado") for d in sc.shopping_source_days(completo))
    sin_politica = _plan()
    sin_politica.pop("_plan_policy")
    assert sc.shopping_source_days(sin_politica) == sin_politica["days"]


def test_un_bloque_suelto_no_se_proyecta_y_una_corrida_nueva_si():
    import nevera_exigida as ne
    corrida = _plan()
    corrida.pop("_plan_policy")
    tok = ne.fijar({"_plan_policy_effective": SINGLE, "_days_offset": 0})
    try:
        assert len(sc.shopping_source_days(corrida)) == 30, "el result del bloque 1 aún no lleva el sello de política"
    finally:
        ne.soltar(tok)
    tok = ne.fijar({"_plan_policy_effective": SINGLE, "_days_offset": 6})
    try:
        assert len(sc.shopping_source_days(corrida)) == 3, "el result de un bloque 2+ es transitorio"
    finally:
        ne.soltar(tok)


def test_alergia_de_la_politica_llega_a_la_proyeccion():
    pol = {**SINGLE, "diet": {"type": "balanced", "allergies": ["Pescado"]}}
    resto = _todas(sc.shopping_source_days(_plan(pol))[7:])
    assert not any("atun" in x or "sardina" in x for x in resto), [x for x in resto if "atun" in x or "sardina" in x]
    assert any("garbanzos cocidos" in x for x in resto)


def test_el_guard_y_la_lista_leen_el_mismo_mes():
    p = _plan()
    esperado = sc.expected_sum_from_recipes(p)
    nombres = {k.lower() for k in esperado}
    assert any("atún en agua" in n or "atun en agua" in n for n in nombres), nombres
    assert sc.ingredient_demand_days(p) == sc.shopping_source_days(p)


def test_la_lista_del_ciclo_es_la_suma_del_mes_y_no_el_x10():
    p = _plan()
    mes = sc.get_shopping_list_delta(None, p, True, False, True, sc.cycle_qty_multiplier("monthly"),
                                     inventory_override=[], consumed_override=[],
                                     cycle_days=sc.cycle_days_for_duration("monthly"))
    por_nombre = {cu._sa(i.get("name")): i for i in mes if isinstance(i, dict)}
    assert any("atun" in n for n in por_nombre) and any("sardina" in n for n in por_nombre), sorted(por_nombre)
    assert all(i.get("_compra_unica") == 30 for i in mes if isinstance(i, dict)), "sello del ciclo en cada ítem"
    # El pollo solo se cocina en la semana de frescos (días 1, 4 y 7 → 3 × 200 g), no «3 días × 10» = 10 × 200 g.
    esperado = sc.expected_sum_from_recipes(p)
    g_pollo = sum((u.get("g") or 0) for n, u in esperado.items() if "pollo" in cu._sa(n))
    assert g_pollo == pytest.approx(600), esperado


def test_la_hibrida_no_parte_una_compra_unica_en_semanas():
    semanal = [{"name": "Pechuga de pollo", "display_qty": "0.4 lb", "is_perishable": True, "category": "Proteínas",
                "shelf_life_days": 3, "_compra_unica": 30}]
    ciclo = [{"name": "Pechuga de pollo", "display_qty": "0.9 lb", "is_perishable": True, "category": "Proteínas",
              "shelf_life_days": 3, "_compra_unica": 30}]
    out = sc._build_hybrid_shopping_list(semanal, ciclo)
    assert out[0]["display_qty"] == "0.9 lb", out
    sin_sello = [dict(ciclo[0])]
    sin_sello[0].pop("_compra_unica")
    semanal_ss = [dict(semanal[0])]
    semanal_ss[0].pop("_compra_unica")
    assert sc._build_hybrid_shopping_list(semanal_ss, sin_sello)[0]["display_qty"] == "0.4 lb", "semanal: sin cambios"


def test_lo_comprado_no_se_vuelve_a_pedir_en_todo_el_ciclo():
    from datetime import datetime, timedelta, timezone
    hace_12 = (datetime.now(timezone.utc) - timedelta(days=12)).isoformat()
    ciclo = [{"name": "Arroz blanco", "display_qty": "3 fundas", "is_perishable": False, "_compra_unica": 30}]
    assert sc._build_hybrid_shopping_list(ciclo, ciclo, restocked_items={"arroz blanco": hace_12}) == []
    semanal = [{"name": "Arroz blanco", "display_qty": "1 funda", "is_perishable": False}]
    assert sc._build_hybrid_shopping_list(semanal, semanal, restocked_items={"arroz blanco": hace_12}), \
        "en compra semanal, a los 12 días se vuelve a pedir"


def test_memoiza_la_proyeccion(monkeypatch):
    p = _plan()
    cu._MEMO.clear()
    llamadas = []
    real = cu._proyectar
    monkeypatch.setattr(cu, "_proyectar", lambda *a, **k: llamadas.append(1) or real(*a, **k))
    a = sc.shopping_source_days(p)
    b = sc.shopping_source_days(p)
    assert a == b and len(llamadas) == 1


def test_cableado():
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    i = src.index("def shopping_source_days(plan_data) -> list:")
    cuerpo = src[i:src.index("\ndef expected_sum_from_recipes(", i)]
    assert cuerpo.count("return _con_ciclo_de_compra_unica(plan_data, ") == 3
    assert 'res = __import__("compra_unica").sellar_lista(res, plan_result)' in src
    assert "tooltip-anchor: P1-PLAN-LOTE-215-HIBRIDA" in src
    assert "tooltip-anchor: P1-PLAN-LOTE-215-PROYECCION" in (_BACKEND / "compra_unica.py").read_text(encoding="utf-8")


def test_la_despensa_de_proteina_no_se_recorta_con_topes_semanales():
    """El plan real de 30 días compraba 736 g de atún para 1.266 g de recetas y 60 huevos para 140: los topes de
    «realismo» (1 lata/semana, 2 huevos/día) son de quien repone. En compra única la lata y el huevo son la proteína de
    los días sin frescos."""
    p = _plan()
    mes = sc.get_shopping_list_delta(None, p, True, False, True, sc.cycle_qty_multiplier("monthly"),
                                     inventory_override=[], consumed_override=[],
                                     cycle_days=sc.cycle_days_for_duration("monthly"))
    for it in mes:
        n = cu._sa(it.get("name"))
        if "atun" in n or "sardina" in n or "huevo" in n:
            assert it.get("capped_by") not in ("P6-CANNED-PROTEIN-CAP", "P6-EGGS-AGGREGATE-CAP"), it
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert 'MEALFIT_SINGLE_TRIP_CANNED_PER_PW", 3.0' in src and 'MEALFIT_SINGLE_TRIP_EGGS_PER_DAY", 5.0' in src
    assert "compra_unica=bool(_ciclo_cu)" in src

