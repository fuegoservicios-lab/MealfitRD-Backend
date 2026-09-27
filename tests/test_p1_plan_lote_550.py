# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-550 · 2026-09-27] La Nevera APAGADA no se lee al cambiar platos.

Auditoría del formulario + verificación de la otra sesión: con `nevera_enabled=false` (puesto por el sistema tras 48 h
vacía) y 0 filas, `evaluate_pantry_sufficiency` daba proteína 0 → soft-fail «Agrega más ítems a tu Nevera» en «Cambiar
plato» y «Actualizar platos», y la Nevera ni sale en el menú (cuenta c7b90ca3, 27-sep).
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import nevera_opcional  # noqa: E402


def _sin_inventario(*_a, **_k):
    raise AssertionError("con la Nevera apagada no se lee user_inventory")


def test_el_gate_de_suficiencia_no_evalua_una_nevera_apagada(monkeypatch):
    import db
    import inventory_sufficiency as inv
    monkeypatch.setattr(nevera_opcional, "nevera_activa", lambda uid: False)
    monkeypatch.setattr(db, "get_raw_user_inventory", _sin_inventario)
    r = inv.evaluate_pantry_sufficiency("u-550", {"mainGoal": "lose_fat"}, scope="day",
                                        meal_target={"kcal": 1800, "protein_g": 120}, nutrition_db=object())
    assert r["sufficient"] is True and r.get("skipped") == "nevera_apagada", r


def test_encendida_sigue_evaluando(monkeypatch):
    import db
    import inventory_sufficiency as inv
    monkeypatch.setattr(nevera_opcional, "nevera_activa", lambda uid: True)
    llamado = []
    monkeypatch.setattr(db, "get_raw_user_inventory", lambda uid: llamado.append(uid) or [])
    inv.evaluate_pantry_sufficiency("u-550", {"mainGoal": "lose_fat"}, scope="day",
                                    meal_target={"kcal": 1800, "protein_g": 120}, nutrition_db=object())
    assert llamado == ["u-550"]


def test_el_universo_del_swap_con_la_nevera_apagada_es_vacio(monkeypatch):
    import agent
    import db
    monkeypatch.setattr(nevera_opcional, "nevera_activa", lambda uid: False)
    leidas = []                                    # la función se traga las excepciones: se cuenta la LECTURA
    monkeypatch.setattr(db, "get_raw_user_inventory", lambda uid: leidas.append(uid) or [])
    assert agent._swap_real_pantry_ledger_lines("u-550") == []
    assert leidas == [], "con la Nevera apagada no se lee user_inventory"


def test_regenerar_dia_y_coach_no_leen_la_nevera_apagada():
    plans = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    assert ('get_raw_user_inventory(user_id) if __import__("nevera_opcional").nevera_activa(user_id) else [], _db)'
            in plans)
    tools = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    assert tools.count("if nevera_activa(user_id) else") >= 3
