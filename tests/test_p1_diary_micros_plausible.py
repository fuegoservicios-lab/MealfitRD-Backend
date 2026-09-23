"""[P1-DIARY-MICROS-PLAUSIBLE · 2026-09-23] «8 rodajas de plátano maduro hervido» se contaba como 8 plátanos
enteros (2.240 g): el lector de unidades no conoce «rodajas», devuelve 'unidad' y pega la palabra al nombre, así
que la guarda «unidad desconocida ⇒ no adivinar» nunca salta. La captura del dueño decía potasio 9.815 mg, vit C
455 mg y fibra 49,7 g para dos comidas de 1.205 kcal. El arreglo del lector es del generador; AQUÍ se cierra la
consecuencia en el diario: si los renglones resueltos suman más de 1,5× las kcal guardadas de la comida, esa comida
no aporta micros (cuenta como «sin datos»: honesto, nunca inventado)."""
from __future__ import annotations

from pathlib import Path

import diary_micros as dm

_BACKEND = Path(__file__).resolve().parents[1]


class _DBConKcal:
    """Resolutor falso: 100 g por renglón salvo el plátano inflado (2.240 g)."""
    _TABLA = {
        "8 rodajas de plátano maduro hervido": (2240.0, 1.22, 3.96),   # g, kcal/g, K mg/g
        "170 g de pollo al horno": (170.0, 1.65, 3.3),
        "80 g de zanahoria": (80.0, 0.41, 3.2),
    }

    def micros_from_ingredient_string(self, s):
        g, _kcal, k = self._TABLA[s]
        return {"grams": g, "potassium_mg": g * k, "fiber": 0.0, "sodium_mg": 0.0, "calcium_mg": 0.0,
                "iron_mg": 0.0, "vit_c_mg": 0.0, "vit_a_mcg": 0.0, "vit_d_mcg": 0.0}

    def macros_from_ingredient_string(self, s):
        g, kcal, _k = self._TABLA[s]
        return {"grams": g, "kcal": g * kcal}


class _DBSinMacros:
    def micros_from_ingredient_string(self, s):
        return {"grams": 100.0, "potassium_mg": 300.0}


def test_micros_de_ingredientes_suma_las_kcal_resueltas():
    m = dm.micros_de_ingredientes(["170 g de pollo al horno", "80 g de zanahoria"], _DBConKcal())
    assert m["kcal"] == round(170 * 1.65 + 80 * 0.41, 1)


def test_sin_macros_en_el_resolutor_no_hay_kcal_ni_guarda():
    m = dm.micros_de_ingredientes(["cualquier cosa"], _DBSinMacros())
    assert "kcal" not in m
    assert dm.micros_plausibles(m, 500) is m


def test_la_comida_inflada_pierde_sus_micros():
    m = dm.micros_de_ingredientes(
        ["8 rodajas de plátano maduro hervido", "170 g de pollo al horno", "80 g de zanahoria"], _DBConKcal())
    assert m["kcal"] > 3000
    assert dm.micros_plausibles(m, 695) is None


def test_la_comida_plausible_conserva_sus_micros():
    m = dm.micros_de_ingredientes(["170 g de pollo al horno", "80 g de zanahoria"], _DBConKcal())
    assert dm.micros_plausibles(m, 350) is m


def test_sin_kcal_de_la_comida_no_se_juzga():
    m = dm.micros_de_ingredientes(["8 rodajas de plátano maduro hervido"], _DBConKcal())
    assert dm.micros_plausibles(m, 0) is m
    assert dm.micros_plausibles(m, None) is m
    assert dm.micros_plausibles(None, 695) is None


def test_el_umbral_es_un_knob(monkeypatch):
    m = dm.micros_de_ingredientes(["170 g de pollo al horno"], _DBConKcal())   # 280,5 kcal
    monkeypatch.setenv("MEALFIT_DIARY_MICROS_KCAL_RATIO_MAX", "1.2")
    assert dm.micros_plausibles(m, 200) is None      # 280,5 > 1,2 × 200
    monkeypatch.setenv("MEALFIT_DIARY_MICROS_KCAL_RATIO_MAX", "2.0")
    assert dm.micros_plausibles(m, 200) is m          # 280,5 < 2,0 × 200


def test_el_endpoint_del_dia_aplica_la_guarda():
    diary = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    ancla = 'm["micros"] = micros_de_ingredientes(m.pop("ingredients", None), _ndb)'
    i = diary.index(ancla)
    assert 'm["micros"] = micros_plausibles(m["micros"], m.get("calories"))' in diary[i:i + 300]


def test_el_coach_pide_gramos_por_renglon():
    tools = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    assert "(≈180 g)" in tools, "la descripción de log_consumed_meal debe pedir el peso de cada renglón"
