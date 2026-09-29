# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-913 · 2026-09-29] Medidor de punto fijo del escudo pre-INSERT.

Medido en el VPS con el entorno real sobre el `final_plan` ENTREGADO de las 5 baterías del 28/29-sep (60 comidas): la 2.ª
pasada del escudo cambia 34 comidas y la 3.ª todavía 3 — el edamame de una cena sube 105 → 115 → 125 g, el arroz integral
baja 45 → 40 g en la 3.ª y «10 g de avena» se dropea. Los medidores del tablero miran la salida de UNA pasada: ninguno lo ve.
"""
from __future__ import annotations

import copy
import importlib.util
import pathlib

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("medir_punto_fijo_escudo", _BACKEND / "scripts" / "medir_punto_fijo_escudo.py")
mpf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mpf)


def _plan(edamame=105):
    return {"calories": 1700, "delivered_calories": 1692, "days": [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Pollo guisado con arroz integral", "cals": 581, "protein": 73, "carbs": 50, "fats": 10,
         "ingredients": ["2 pechugas de pollo (≈300 g)", "45 g de arroz integral crudo"],
         "ingredients_raw": ["300 g de pechuga de pollo", "45 g de arroz integral crudo"],
         "recipe": ["Mise en place: corta 300 g de pechuga en trozos.", "Montaje: sirve el pollo sobre el arroz."]},
        {"meal": "Cena", "name": "Yuca guisada con queso blanco fresco, aguacate y edamame", "cals": 576, "protein": 20,
         "carbs": 52, "fats": 32, "ingredients": ["60 g de yuca", f"{edamame} g de edamame cocido"],
         "ingredients_raw": ["60 g de yuca", f"{edamame} g de edamame cocido"],
         "recipe": ["El Toque de Fuego: hierve la yuca 10-12 min.", "Montaje: sirve la yuca. Acompaña con edamame."]}]}]}


def _sube_10(plan, form, uid):
    """Un cierre que no converge: 10 g más de edamame en cada pasada."""
    cena = plan["days"][0]["meals"][1]
    g = int(cena["ingredients"][1].split()[0]) + 10
    cena["ingredients"][1] = cena["ingredients_raw"][1] = f"{g} g de edamame cocido"
    cena["cals"] += 17
    return plan


def test_dos_planes_iguales_no_cambian():
    r = mpf.comparar(_plan(), _plan())
    assert r["comidas"] == 2 and r["cambiadas"] == 0 and r["campos"] == {} and r["detalle"] == []


def test_cuenta_la_comida_y_los_campos_que_cambian():
    r = mpf.comparar(_plan(105), _sube_10(_plan(105), {}, "u"))
    assert r["comidas"] == 2 and r["cambiadas"] == 1
    assert r["campos"] == {"ingredients": 1, "ingredients_raw": 1, "cals": 1}
    d = r["detalle"][0]
    assert d["dia"] == 0 and d["comida"] == 1 and d["nombre"].startswith("Yuca guisada")
    assert d["campo"] == "ingredients" and d["antes"] == ["105 g de edamame cocido"] and d["despues"] == ["115 g de edamame cocido"]


def test_una_comida_de_mas_o_de_menos_cuenta_como_cambio():
    b = _plan()
    b["days"][0]["meals"].pop()
    r = mpf.comparar(_plan(), b)
    assert r["cambiadas"] == 1 and r["campos"] == {"comida_ausente": 1}


def test_un_cierre_que_sube_10_g_por_pasada_no_es_punto_fijo():
    r = mpf.medir([("rdv868", _plan(105), {}, "u")], _sube_10, pasadas=2)
    assert r["planes"] == 1 and r["comidas"] == 2
    assert [p["pasada"] for p in r["pasadas"]] == [2, 3]
    assert [p["cambiadas"] for p in r["pasadas"]] == [1, 1]
    assert r["punto_fijo"] is False
    assert r["pasadas"][1]["detalle"][0]["despues"] == ["125 g de edamame cocido"]
    assert r["pasadas"][1]["detalle"][0]["plan"] == "rdv868"


def test_un_escudo_que_no_toca_lo_entregado_es_punto_fijo():
    r = mpf.medir([("a", _plan(), {}, "u"), ("b", _plan(115), {}, "u")], lambda plan, form, uid: plan, pasadas=2)
    assert r["planes"] == 2 and r["comidas"] == 4 and [p["cambiadas"] for p in r["pasadas"]] == [0, 0]
    assert r["punto_fijo"] is True


def test_el_plan_de_entrada_no_se_toca():
    entregado = _plan(105)
    copia = copy.deepcopy(entregado)
    mpf.medir([("rdv868", entregado, {}, "u")], _sube_10, pasadas=2)
    assert entregado == copia


def test_un_escudo_que_revienta_se_cuenta_y_no_para_la_medicion():
    def roto(plan, form, uid):
        raise RuntimeError("sin catálogo")
    r = mpf.medir([("a", _plan(), {}, "u")], roto, pasadas=2)
    assert r["errores"] == [{"plan": "a", "pasada": 2, "error": "RuntimeError: sin catálogo"}]
    assert r["punto_fijo"] is None, "sin medición no hay veredicto"


def test_la_guarda_de_escritura_se_activa_antes_de_importar_el_backend(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    mpf.activar_guarda()
    import os
    assert os.environ.get("PYTEST_CURRENT_TEST", "").startswith("medir_punto_fijo_escudo")
    src = (_BACKEND / "scripts" / "medir_punto_fijo_escudo.py").read_text(encoding="utf-8")
    cuerpo = src[src.index("def _escudo_real():"):src.index("def render(")]
    assert 0 < cuerpo.index("activar_guarda()") < cuerpo.index("import db_core") < cuerpo.index("import db_plans"),         "la guarda va ANTES de importar el backend"
    assert "tooltip-anchor: P1-PLAN-LOTE-913" in src


def test_el_backend_se_puede_senalar_por_entorno(monkeypatch, tmp_path):
    """En el VPS el script vive fuera del árbol vivo (/tmp): `RD_BACKEND`, como en rd_battery.py."""
    monkeypatch.setenv("RD_BACKEND", str(tmp_path))
    spec = importlib.util.spec_from_file_location("mpf_env", _BACKEND / "scripts" / "medir_punto_fijo_escudo.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m._BACKEND == tmp_path
    monkeypatch.delenv("RD_BACKEND")
    spec.loader.exec_module(m)
    assert m._BACKEND == _BACKEND
