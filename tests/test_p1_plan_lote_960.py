# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-960 · 2026-10-01] El coach usa la Nevera con INICIATIVA: proteína por porción, cruzada con lo que le
falta hoy, reglas de cuándo proponerla, y el aviso de comida que propone con lo que tiene.

El dueño: «tengo una proteína en la Nevera y a veces me gustaría que tomara la iniciativa, o si digo algo, que pueda
complementar con lo que ya tengo».
"""
from __future__ import annotations

import os
from datetime import date

import pytest

import nevera_del_coach as n

_CATALOGO = [
    {"id": 1, "name": "Pechuga de pollo", "category": "Proteínas", "protein_g_per_100g": 22.5, "kcal_per_100g": 107.4,
     "shelf_life_days": 14, "default_unit": "lb"},
    {"id": 2, "name": "Huevo", "category": "Proteínas", "protein_g_per_100g": 12.6, "kcal_per_100g": 138.9,
     "density_g_per_unit": 50, "shelf_life_days": 35, "default_unit": "cartón"},
    {"id": 3, "name": "Queso cottage", "category": "Lácteos", "protein_g_per_100g": 10.5, "kcal_per_100g": 81,
     "shelf_life_days": 14},
    {"id": 4, "name": "Arroz blanco", "category": "Despensa", "protein_g_per_100g": 7, "kcal_per_100g": 360},
    {"id": 5, "name": "Lentejas", "category": "Despensa", "protein_g_per_100g": 24.6, "kcal_per_100g": 361.5,
     "shelf_life_days": 180},
    {"id": 6, "name": "Salami", "category": "Proteínas", "protein_g_per_100g": 16, "kcal_per_100g": 323,
     "ready_to_eat": True, "shelf_life_days": 30},
]
_HOY = date(2026, 10, 1)


def _fila(nombre, qty, unit, creada="2026-09-30", mid=None):
    return {"ingredient_name": nombre, "quantity": qty, "unit": unit, "created_at": creada, "updated_at": None,
            "master_ingredient_id": mid}


@pytest.fixture(autouse=True)
def _sin_backstop(monkeypatch):
    # El backstop real se prueba aparte (test_vegana_*); aquí, todo apto salvo que el test diga otra cosa.
    monkeypatch.setattr(n, "_violaciones", lambda nombre, perfil: [])


def test_solo_las_fuentes_de_proteina_con_su_porcion():
    pr = n.proteinas_de([_fila("Pechuga de pollo", 908, "g"), _fila("Arroz blanco", 2000, "g"),
                         _fila("Lentejas", 400, "g")], _CATALOGO, hoy=_HOY)
    nombres = [x["nombre"] for x in pr]
    assert "Arroz blanco" not in nombres and "Pechuga de pollo" in nombres and "Lentejas" in nombres
    pollo = next(x for x in pr if x["nombre"] == "Pechuga de pollo")
    assert pollo["porcion_g"] == 150 and round(pollo["porcion_p"]) == 34 and pollo["porciones"] == 6
    lent = next(x for x in pr if x["nombre"] == "Lentejas")
    assert lent["clase"] == "legumbre" and lent["porcion_g"] == 80   # en seco, no 150 g de lentejas crudas


def test_el_embutido_va_en_porcion_chica_y_el_huevo_en_unidades():
    pr = n.proteinas_de([_fila("Salami", 200, "g"), _fila("Huevo", 12, "unidad")], _CATALOGO, hoy=_HOY)
    sal = next(x for x in pr if x["nombre"] == "Salami")
    hue = next(x for x in pr if x["nombre"] == "Huevo")
    assert sal["clase"] == "embutido" and sal["porcion_g"] == 50
    assert hue["porcion_g"] == 100 and "2 huevos" in n._linea(hue)


def test_cruza_lo_que_le_falta_hoy_con_lo_que_tiene():
    pr = n.proteinas_de([_fila("Pechuga de pollo", 908, "g")], _CATALOGO, hoy=_HOY)
    t = n.bloque_chat(pr, 45.0)
    assert "PROTEÍNA QUE YA TIENE EN SU NEVERA" in t and "INICIATIVA CON SU NEVERA" in t
    assert "CON LO QUE LE FALTA HOY (~45 g de proteína): 200 g en crudo de Pechuga de pollo (de su Nevera)" in t
    assert "lo cierra sin comprar nada" in t


def test_no_propone_mas_de_lo_que_tiene():
    pr = n.proteinas_de([_fila("Pechuga de pollo", 120, "g")], _CATALOGO, hoy=_HOY)
    prop = n.propuesta_para_falta(pr, 60.0)
    assert prop["porcion"].startswith("100 g") and not prop["cubre"]


def test_poca_falta_no_dispara_propuesta():
    pr = n.proteinas_de([_fila("Pechuga de pollo", 908, "g")], _CATALOGO, hoy=_HOY)
    assert n.propuesta_para_falta(pr, 8.0) is None and n.propuesta_para_falta(pr, None) is None
    assert "CON LO QUE LE FALTA HOY" not in n.bloque_chat(pr, 8.0)


def test_en_el_desayuno_lo_ligero_antes_que_la_carne():
    pr = n.proteinas_de([_fila("Pechuga de pollo", 908, "g"), _fila("Huevo", 12, "unidad")], _CATALOGO, hoy=_HOY)
    assert n.propuesta_para_falta(pr, 25.0, "desayuno")["nombre"] == "Huevo"
    assert n.propuesta_para_falta(pr, 25.0, "cena")["nombre"] == "Pechuga de pollo"


def test_lo_que_caduca_pronto_va_primero_y_se_marca():
    pr = n.proteinas_de([_fila("Queso cottage", 450, "g", creada="2026-09-18"),
                         _fila("Pechuga de pollo", 908, "g")], _CATALOGO, hoy=_HOY)
    assert pr[0]["nombre"] == "Queso cottage" and pr[0]["urgente"]
    assert "⏳ úsala PRIMERO" in n._linea(pr[0])
    # Muy pasada de fecha (fila olvidada o congelada): ni se marca ni se afirma que caducó
    viejo = n.proteinas_de([_fila("Pechuga de pollo", 908, "g", creada="2026-08-01")], _CATALOGO, hoy=_HOY)
    assert not viejo[0]["urgente"] and "⏳" not in n._linea(viejo[0])


def test_lo_vetado_no_se_propone_nunca(monkeypatch):
    monkeypatch.setattr(n, "_violaciones",
                        lambda nombre, perfil: ["carne (no apto)"] if "pollo" in nombre.lower() else [])
    pr = n.proteinas_de([_fila("Pechuga de pollo", 908, "g"), _fila("Lentejas", 400, "g")], _CATALOGO,
                        perfil={"dietType": "vegana"}, hoy=_HOY)
    assert n.propuesta_para_falta(pr, 40.0)["nombre"] == "Lentejas"
    t = n.bloque_chat(pr, 40.0)
    assert "NO la propongas" in t and "Pechuga de pollo" in t.split("NO la propongas")[1]
    assert "Pechuga de pollo" not in n.bloque_aviso(pr, 40.0)


def test_sin_proteina_no_finge():
    t = n.bloque_chat([], 40.0)
    assert "ninguna registrada" in t and "no finjas que tiene" in t
    assert n.bloque_aviso([], 40.0) == ""


def test_el_aviso_propone_para_esa_comida_no_para_el_dia(monkeypatch):
    monkeypatch.setattr(n, "_proteinas_del_usuario",
                        lambda uid, perfil: n.proteinas_de([_fila("Pechuga de pollo", 908, "g")], _CATALOGO, hoy=_HOY))
    import aviso_del_dia
    import nevera_opcional
    monkeypatch.setattr(nevera_opcional, "nevera_activa", lambda uid: True)
    monkeypatch.setattr(aviso_del_dia, "_metas", lambda health: (2400, 160))
    t = n.para_aviso("u1", {}, [], "Almuerzo")
    assert "Para esta comida: 225 g en crudo de Pechuga de pollo" in t   # 30 % de 160 = 48 g, no los 160 del día
    assert "Toma la iniciativa" in t


def test_la_nevera_apagada_o_el_knob_apagado_callan(monkeypatch):
    import nevera_opcional
    monkeypatch.setattr(nevera_opcional, "nevera_activa", lambda uid: False)
    assert n.para_aviso("u1", {}, [], "cena") == ""
    assert n.para_chat("u1", False, {}, None, []) == ""
    assert n.para_chat("guest", True, {}, None, []) == ""
    monkeypatch.setenv("MEALFIT_COACH_NEVERA_INICIATIVA", "false")
    assert not n.activo()


def test_va_cableado_en_los_dos_caminos_del_chat_y_en_el_aviso():
    raiz = os.path.dirname(os.path.abspath(n.__file__))
    agent = open(os.path.join(raiz, "agent.py"), encoding="utf-8").read()
    assert agent.count('__import__("nevera_del_coach").para_chat(') == 2
    pro = open(os.path.join(raiz, "proactive_agent.py"), encoding="utf-8").read()
    assert '__import__("nevera_del_coach").para_aviso(user_id, health, consumed, meal_to_check)' in pro
    assert "prompt += _nevera_aviso" in pro
    assert 'salvo los de SU NEVERA listados abajo.' in pro


def test_vegana_el_backstop_real_veta_el_pollo(monkeypatch):
    monkeypatch.undo()
    assert n._violaciones("Pechuga de pollo", {"dietType": "vegana", "allergies": []})
    assert not n._violaciones("Lentejas", {"dietType": "vegana", "allergies": []})
    assert n._violaciones("Camarones", {"dietType": "balanced", "allergies": ["Mariscos"]})
