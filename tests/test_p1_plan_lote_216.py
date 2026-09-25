# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-216 · 2026-09-24] Compra única: los bloques 2+ cocinan con lo COMPRADO.

La lista del día 1 alcanza para el ciclo (215), pero sin Nevera (vacía porque el usuario no marcó «Ya compré», o
apagada) el bloque se generaba libre y podía traer alimentos que no se compraron. En una compra única lo comprado ES la
Nevera: el bloque recibe los nombres de la lista del ciclo como Nevera y el revisor la exige; las guardas de Nevera lo
eximen (no hay cantidades que medir). Con la Nevera real en uso, manda la real.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import compra_unica as cu  # noqa: E402
import nevera_exigida as ne  # noqa: E402

SINGLE = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none"},
          "diet": {"type": "balanced", "allergies": []}}
LISTA = [{"name": n, "_compra_unica": 30} for n in (
    "Atún en agua", "Sardinas en lata", "Garbanzos", "Huevo", "Arroz blanco", "Batata", "Casabe", "Cebolla", "Ajo",
    "Avena", "Lentejas", "Habichuelas negras", "Aceite de oliva", "Manzana", "Pechuga de pollo")] + [
    {"name": "Leche", "_compra_unica": 30, "category": "🚨 Compra Urgente"}]


def _consultar(fila):
    llamadas = []

    def _q(sql, params=(), **k):
        llamadas.append((sql, params))
        return fila
    return _q, llamadas


def _fd(**extra):
    fd = {"_plan_policy_effective": SINGLE, "_days_offset": 6, "current_pantry_ingredients": []}
    fd.update(extra)
    return fd


def test_sin_nevera_el_bloque_recibe_la_compra_del_ciclo():
    q, llamadas = _consultar({"activa": LISTA, "mensual": LISTA})
    fd = cu.nevera_virtual(_fd(), task_id=7, user_id="u1", consultar=q)
    assert fd["_nevera_virtual"] is True and fd["_fresh_pantry_source"] == "compra_unica_virtual"
    assert "Atún en agua" in fd["current_pantry_ingredients"] and "Leche" not in fd["current_pantry_ingredients"]
    # [P1-PLAN-LOTE-221] 14, no 15: la pechuga fresca del día 1 no llega al bloque del día 7 sin congelador
    assert len(fd["current_pantry_ingredients"]) == 14
    assert "Pechuga de pollo" not in fd["current_pantry_ingredients"]
    sql, params = llamadas[0]
    assert "q.id = %s AND mp.user_id = %s" in sql and params == (7, "u1")
    # y el revisor la EXIGE como a cualquier Nevera (la regla b del sembrador también manda con ella)
    assert ne.lista(fd) is not None and ne.nevera_manda(fd) is True


def test_la_nevera_apagada_en_compra_unica_tambien():
    q, _ = _consultar({"mensual": LISTA})
    fd = cu.nevera_virtual(_fd(_nevera_apagada=True, _pantry_advisory_only=True,
                               current_pantry_ingredients=[]), task_id=7, user_id="u1", consultar=q)
    assert fd["_nevera_virtual"] is True and "_pantry_advisory_only" not in fd


def test_con_la_nevera_real_manda_la_real():
    q, llamadas = _consultar({"mensual": LISTA})
    fd = cu.nevera_virtual(_fd(current_pantry_ingredients=["20 Huevo", "Pollo"]), task_id=7, user_id="u1", consultar=q)
    assert fd["current_pantry_ingredients"] == ["20 Huevo", "Pollo"] and not fd.get("_nevera_virtual")
    assert llamadas == []


def test_no_aplica_fuera_de_la_compra_unica_ni_al_bloque_1():
    q, llamadas = _consultar({"mensual": LISTA})
    semanal = {"shopping": {"main_cycle_days": 7}}
    assert not cu.nevera_virtual(_fd(_plan_policy_effective=semanal), task_id=7, user_id="u1", consultar=q).get("_nevera_virtual")
    assert not cu.nevera_virtual(_fd(_days_offset=0), task_id=7, user_id="u1", consultar=q).get("_nevera_virtual")
    assert not cu.nevera_virtual(_fd(), task_id=7, user_id="guest", consultar=q).get("_nevera_virtual")
    assert llamadas == []


def test_lista_corta_o_error_no_cambia_nada():
    q, _ = _consultar({"mensual": LISTA[:2]})
    assert not cu.nevera_virtual(_fd(), task_id=7, user_id="u1", consultar=q).get("_nevera_virtual")

    def _revienta(*a, **k):
        raise RuntimeError("db")
    fd = cu.nevera_virtual(_fd(), task_id=7, user_id="u1", consultar=_revienta)
    assert not fd.get("_nevera_virtual") and fd["current_pantry_ingredients"] == []


def test_ninguna_guarda_de_nevera_la_mide_por_cantidades():
    import cron_tasks as ct
    fd = {"_nevera_virtual": True, "current_pantry_ingredients": ["Atún en agua"] * 3}
    assert ct._pantry_gate_waiver_reason(chunk_kind="rolling_refill", form_data=fd) == "compra_unica_virtual"
    assert ct._should_pause_for_empty_pantry("compra_unica_virtual", fd["current_pantry_ingredients"], {}, fd) is False


def test_cableado():
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    i = src.index("def _refresh_chunk_pantry(")
    cuerpo = src[i:src.index("\ndef ", i + 10)]
    assert cuerpo.count('__import__("compra_unica").nevera_virtual(form_data, task_id, user_id)') == 2
    assert "tooltip-anchor: P1-PLAN-LOTE-216-NEVERA-VIRTUAL" in (_BACKEND / "compra_unica.py").read_text(encoding="utf-8")


def test_la_rueda_salta_lo_que_el_dia_ya_lleva(monkeypatch):
    """Batería real: el día que ya traía atún al desayuno recibía MÁS atún de la rueda (17 líneas de atún frente a 8 de
    sardinas y 7 de garbanzos en la proyección de 30 días)."""
    import graph_orchestrator as go
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)

    class _NoopDB:
        def macros_from_ingredient_string(self, s):
            return None

    days = [{"day": i + 1, "meals": [{"meal": "Cena", "name": "x", "ingredients": ["1 taza de arroz"],
                                      "ingredients_raw": ["1 taza de arroz"]}]} for i in range(12)]
    # semilla 9 + 0 ⇒ la rueda empieza por el atún: sin «evitar» esta cena recibiría atún
    days[9]["meals"] = [
        {"meal": "Cena", "name": "Pollo", "ingredients": ["150 g de pechuga de pollo"], "ingredients_raw": ["150 g de pechuga de pollo"]},
        {"meal": "Desayuno", "name": "Tostada", "ingredients": ["160 g de atún en agua"], "ingredients_raw": ["160 g de atún en agua"]},
    ]
    go._single_trip_fresh_substitute(days, db=_NoopDB(), effective=SINGLE, diet="balanced")
    cena = days[9]["meals"][0]["ingredients"][0]
    assert "atun" not in cena and cena.startswith("150 g de "), cena
    assert cu.duraderos_del_dia(["160 g de atún en agua", "1 lata de sardinas"]) == {"atun en agua", "sardinas en lata"}
    # sin otra opción segura, el que ya está vale antes que dejar el fresco
    assert cu.sustituto_seguro(cu.PROTEINA_TABLA, 0, False, evitar={"atun en agua", "sardinas en lata",
                                                                      "garbanzos cocidos"}) == "atun en agua"

