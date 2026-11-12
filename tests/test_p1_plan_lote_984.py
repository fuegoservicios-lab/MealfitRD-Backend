"""[P1-PLAN-LOTE-984 · 2026-11-11] Los tres flecos de la conversación del dueño del 11-nov.

1. El almuerzo (arroz 240 g, pechuga 175 g, maduro frito 100 g, lentejas 200 g) quedó en 940 kcal / 22 g de grasa; los
   platos COCINADOS del catálogo dan ≈1.227 / 36 ⇒ `contraste_con_catalogo` rechaza una vez con la cifra.
2. «ensalada de pollo» como renglón de las verduras ⇒ la descripción de `log_consumed_meal` lo prohíbe.
3. El aviso de las 8:30 «avena cremosa con pavo y aguacate» ⇒ `combinacion_del_aviso` regenera una vez.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import combinacion_del_aviso as ca  # noqa: E402
import contraste_con_catalogo as cc  # noqa: E402

_ALMUERZO = ["arroz blanco cocido (1.5 tazas, ≈240 g)", "pechuga de pollo cocinada (3.5 pedazos, porción estimada ≈175 g)",
             "plátano maduro frito (6 pedacitos, ≈100 g)", "lentejas cocidas (1 taza, ≈200 g)"]


def setup_function(_):
    cc._YA_AVISADO.clear()


# ───────────── 1. contraste con el catálogo ─────────────

def test_el_almuerzo_del_dueno_se_queda_corto():
    c = cc.se_queda_corto(940, _ALMUERZO)
    assert c is not None
    assert 1150 < c["kcal"] < 1300 and 30 < c["grasa"] < 42
    assert [label for _t, label, _g, _k in c["lineas"]] == [
        "Arroz blanco", "Pollo al horno (pechuga)", "Plátanos maduros fritos", "Lentejas guisadas"]


def test_un_estimado_razonable_pasa():
    assert cc.se_queda_corto(1180, _ALMUERZO) is None


def test_la_cabeza_de_la_linea_manda():
    # «sancocho … con plátano verde» no es plátano verde hervido; «mangú de auyama» no es el mangú de plátano
    assert cc.plato_de("Sancocho criollo con plátano verde, víveres y carne (≈500 g)") is None
    assert cc.plato_de("mangú de auyama o batata (≈200 g)") is None
    assert cc.plato_de("6 rodajas de plátano verde frito (≈110 g)")[0] == "Tostones (plátano verde frito)"
    assert cc.plato_de("mangú de plátano verde majado (350 g)")[0] == "Mangú (plátano verde majado)"


def test_sin_gramos_no_cuenta():
    assert cc.contraste(["arroz blanco", "lentejas"])["lineas"] == []


def test_una_vez_por_usuario_y_knob(monkeypatch):
    txt = cc.rechazo("u", "Almuerzo", 940, 22, _ALMUERZO, ahora=1000.0)
    assert txt and txt.startswith("⚠️ NO REGISTRADO TODAVÍA") and "Lentejas guisadas" in txt
    assert cc.rechazo("u", "Almuerzo", 940, 22, _ALMUERZO, ahora=1010.0) is None   # el reintento pasa
    assert cc.rechazo("v", "Almuerzo", 940, 22, _ALMUERZO, ahora=1010.0) is not None
    monkeypatch.setenv("MEALFIT_CATALOG_FLOOR_GUARD", "0")
    assert cc.rechazo("w", "Almuerzo", 940, 22, _ALMUERZO, ahora=1010.0) is None


def test_no_se_confunde_con_aviso_de_duplicado():
    from agent import _AVISOS_DE_DUPLICADO
    txt = cc.rechazo("x", "Almuerzo", 940, 22, _ALMUERZO, ahora=5.0)
    assert not any(a in txt for a in _AVISOS_DE_DUPLICADO)


def test_cableado_en_log_consumed_meal():
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    i_qty = src.index("_icc.rechazo(user_id, meal_name, ingredients, es_foto=_es_foto)")
    i_cat = src.index("_ccc.rechazo(user_id, meal_name, calories, healthy_fats, ingredients)")
    i_insert = src.index("# [P1-CONSUMED-BACKDATE · 2026-07-12] Fecha real de consumo + guard dup.")
    assert i_qty < i_cat < i_insert


# ───────────── 2. el renglón nombra lo que es ─────────────

def test_regla_de_nombres_en_la_descripcion():
    import tools
    assert "nunca un renglón \"ensalada de pollo\" para las verduras" in (tools.log_consumed_meal.description or "")


# ───────────── 3. combinación del aviso ─────────────

def test_detecta_los_avisos_reales():
    assert ca.mezcla_rara("Una avena cremosa con pavo y aguacate te caería buenísima para desayunar ahora") == \
        "avena cremosa con pavo"
    assert ca.mezcla_rara("Una avena cremosa con pavo te vendría buenísima para desayunar ahora; es sabrosa")


def test_combinaciones_normales_pasan():
    for t in ("Un sándwich de pavo con batido de lechosa", "Un sándwich de pavo con un batido",
              "Avena con canela o un sándwich de pollo", "Yogur griego con fresas y almendras",
              "Pollo empanizado con avena y ensalada", "Pan con pavo y aguacate, y de postre un yogur"):
        assert ca.mezcla_rara(t) is None, t


def test_regenera_una_vez_y_si_no_el_aviso_fijo(monkeypatch):
    import proactive_agent as pa

    class _Resp:
        def __init__(self, c):
            self.content = c

    class _LLM:
        def __init__(self, salidas):
            self.salidas, self.prompts = list(salidas), []

        def invoke(self, p):
            self.prompts.append(p)
            return _Resp(self.salidas.pop(0))

    bueno = _LLM(["Un pan tostado con pavo y aguacate te cae bien ahora."])
    assert pa._sin_mezcla_rara(bueno, "P", "Avena cremosa con pavo", "u", "Desayuno", "es-DO").startswith("Un pan")
    assert ca.CORRECCION in bueno.prompts[0]

    terco = _LLM(["Avena con pavo, de verdad"])
    out = pa._sin_mezcla_rara(terco, "P", "Avena cremosa con pavo", "u", "Desayuno", "es-DO")
    from meal_reminders import texto_del_aviso
    assert out == texto_del_aviso("Desayuno", "es-DO")[1]

    nada = _LLM([])
    assert pa._sin_mezcla_rara(nada, "P", "Huevos con casabe", "u", "Desayuno", "es-DO") == "Huevos con casabe"
    assert nada.prompts == []


def test_regla_en_el_prompt_del_aviso():
    src = (_BACKEND / "proactive_agent.py").read_text(encoding="utf-8")
    assert '__import__("combinacion_del_aviso").REGLA' in src
    assert "content = _sin_mezcla_rara(chat_llm, prompt, content, user_id, meal_to_check, _nudge_locale)" in src


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · ', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 984
