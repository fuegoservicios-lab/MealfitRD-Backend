# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-933 · 2026-09-29] El cerrador de proteína cuenta la línea que ESCRIBE, no la fila del catálogo.

El cerrador calcula los gramos con la densidad de la FILA («Lentejas»: 24,6 g de proteína por 100 g, en SECO) y escribe
la línea con su participio, «124 g de lentejas cocidas», que la base mide cocidas: 100 g de lentejas cocidas son 35 g
secas y 8,6 g de proteína. Cree que añadió 31 g y añadió 10,7.

Simulado sobre 566 planes del VPS (1.340 cierres): 64 difieren en más de 3 g; en la dirección que ABRE déficit, las
legumbres (habichuelas blancas 10, lentejas 4, gandules 2, habichuelas negras 1), la soya texturizada (2) y el yogurt
natural del plato escalado con la densidad del griego (17). Con el lote 932 el cerrador de la compra única vuelve a tener
legumbres entre sus candidatos: sin esto, las elegiría y cerraría un tercio.

Sólo se corrige la dirección que entrega MENOS de lo contado. Las carnes y los mariscos van al revés (100 g de pechuga
cocida son 135 g crudos: el cerrador cuenta 22,5 g y entrega 30,4) y no se tocan aquí: cambia los gramos de casi todos
los cierres con carne y pide una batería.
"""
from __future__ import annotations

import pathlib
import re

import cierre_medido as cm
import graph_orchestrator as go

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


class _Info:
    def __init__(self, name, protein, kcal, carbs=0.0, fats=0.0):
        self.name, self.protein, self.kcal, self.carbs, self.fats = name, protein, kcal, carbs, fats
        self.fiber = 7.0


_LENTEJAS = _Info("Lentejas", 24.6, 362.0, 63.0, 1.1)
_POLLO = _Info("Pechuga de pollo", 22.5, 107.0, 0.0, 1.9)
_GRIEGO = _Info("Yogurt griego entero", 8.8, 94.0, 4.0, 4.4)
_FILAS = {"lentejas": _LENTEJAS, "pechuga de pollo": _POLLO, "yogurt griego": _GRIEGO,
          "yogurt": _Info("Yogurt", 3.5, 61.0, 4.7, 3.3)}
_FACTOR = {"lentejas": 0.3513, "pechuga de pollo": 1.35}       # gramos de la fila por gramo «cocido»


class _DB:
    """Como la base real: «cocidas» convierte al peso de la fila; «secas» no."""

    def _fila(self, s):
        n = go._norm_text(str(s))
        return next((k for k in sorted(_FILAS, key=len, reverse=True) if k in n), None)

    def grams_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\s+de\s+", str(s))
        return float(m.group(1).replace(",", ".")) if m else None

    def macros_from_ingredient_string(self, s):
        g, k = self.grams_from_ingredient_string(s), self._fila(s)
        if g is None or k is None:
            return None
        n = go._norm_text(str(s))
        if re.search(r"\bcocid[oa]s?\b", n) and not re.search(r"\bsec[oa]s?\b", n):
            g *= _FACTOR.get(k, 1.0)
        f = _FILAS[k]
        return {"grams": g, "kcal": f.kcal * g / 100, "protein": f.protein * g / 100, "carbs": f.carbs * g / 100,
                "fats": f.fats * g / 100}

    def __getattr__(self, _n):
        return lambda *a, **k: None


def _cena(*extra):
    return {"meal": "Cena", "name": "Yuca hervida con vegetales salteados",
            "protein": 6, "cals": 320, "carbs": 60, "fats": 6,
            "ingredients": ["200 g de yuca", "1 zanahoria", *extra],
            "recipe": ["El Toque de Fuego: hierve la yuca 20 minutos y saltea la zanahoria.", "Montaje: sirve."]}


def _cerrar(meal, info, objetivo=30.0):
    return go._close_protein_gap_for_meal(meal, objetivo, _DB(), [(0.0, info.name, info)], allergies=None,
                                          fill_pct=1.0, max_add_g=300, enforce_min_threshold=False,
                                          day_used_proteins=set(), diet="vegana", country="DO", goal="lose_fat")


def _medida(meal):
    return sum((_DB().macros_from_ingredient_string(x) or {}).get("protein", 0.0) for x in meal["ingredients"])


def test_la_legumbre_cocida_cierra_lo_que_el_cerrador_cuenta():
    m = _cena()
    g = _cerrar(m, _LENTEJAS, objetivo=20.0)             # faltan 14 g
    linea = m["ingredients"][-1]
    assert re.match(r"^\d+ g de lentejas cocidas$", linea), linea
    assert 150 <= g <= 170, (g, "14 g de proteína son ~162 g de lentejas cocidas, no 57")
    assert abs((m["protein"] - 6) - _medida(m)) <= 1.5, (m["protein"], _medida(m))
    assert abs(m["cals"] - 320 - _DB().macros_from_ingredient_string(linea)["kcal"]) <= 3


def test_la_linea_seca_que_el_plato_ya_tiene_se_escala_en_seco():
    m = _cena("30 g de lentejas secas")
    m["name"], m["protein"] = "Yuca hervida con lentejas guisadas", 13
    antes = _medida(m)
    g = _cerrar(m, _LENTEJAS, objetivo=25.0)             # faltan 12 g: ~49 g secos más
    linea = next(x for x in m["ingredients"] if "lentejas" in x)
    assert re.match(r"^\d+ g de lentejas secas$", linea), linea
    assert 70 <= float(linea.split()[0]) <= 85, linea
    assert abs((_medida(m) - antes) - (m["protein"] - 13)) <= 1.5, (linea, m["protein"])
    assert g > 0


def test_la_carne_no_cambia_aqui():
    m = _cena()
    m["meal"] = "Almuerzo"
    _cerrar(m, _POLLO, objetivo=26.0)                    # faltan 20 g
    linea = m["ingredients"][-1]
    assert re.match(r"^\d+ g de pechuga de pollo cocid[oa]$", linea), linea
    assert 85 <= float(linea.split()[0]) <= 92, (linea, "20 / 0,225 = 89 g, como siempre")
    assert m["protein"] == 26


def test_el_yogurt_natural_del_plato_no_se_cuenta_como_griego():
    m = {"meal": "Merienda", "name": "Vasito de yogurt con lechosa", "protein": 6, "cals": 150, "carbs": 20, "fats": 4,
         "ingredients": ["150 g de yogurt natural", "100 g de lechosa"],
         "recipe": ["Montaje: sirve el yogurt con la lechosa."]}
    antes = _medida(m)
    _cerrar(m, _GRIEGO, objetivo=9.0)                    # faltan 3 g
    assert abs((_medida(m) - antes) - (m["protein"] - 6)) <= 1.0, (m["ingredients"], m["protein"])


def test_como_se_mide():
    db = _DB()
    assert cm.como_se_mide(_cena(), _POLLO, False, db) is _POLLO, "la carne entrega MÁS de lo contado: no se toca"
    assert cm.como_se_mide(_cena(), _GRIEGO, False, db) is _GRIEGO
    p = cm.como_se_mide(_cena(), _LENTEJAS, False, db)
    assert p is not _LENTEJAS and p.name == "Lentejas" and 8.4 < p.protein < 8.8 and 125 < p.kcal < 129
    assert p.fiber == 7.0, "lo demás, de la fila"
    seco = cm.como_se_mide(_cena("30 g de lentejas secas"), _LENTEJAS, False, db)
    assert seco is _LENTEJAS, "la línea del plato está en seco, como la fila"
    assert cm.como_se_mide(_cena(), _LENTEJAS, False, None) is _LENTEJAS
    assert cm.como_se_mide(None, _LENTEJAS, False, db) is _LENTEJAS


def test_el_sufijo_es_el_que_escribe_el_cerrador():
    """Paridad: si el cerrador cambia cómo escribe la línea, este test lo dice."""
    for info in (_LENTEJAS, _POLLO, _GRIEGO):
        m = _cena()
        m["meal"] = "Almuerzo"
        _cerrar(m, info, objetivo=20.0)
        nm = info.name.lower()
        assert m["ingredients"][-1].endswith(" g de " + nm + cm.sufijo(nm, False)), m["ingredients"][-1]


def test_con_el_knob_apagado_la_cuenta_de_antes(monkeypatch):
    monkeypatch.setenv("MEALFIT_CLOSER_COUNTS_WRITTEN_LINE", "false")
    m = _cena()
    g = _cerrar(m, _LENTEJAS, objetivo=20.0)
    assert 55 <= g <= 60 and m["protein"] == 20, (g, m["protein"])


def test_anclas():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    cerrador = src[src.index("def _close_protein_gap_for_meal("):src.index("def _ingredient_is_protein_dominant(")]
    gancho = 'chosen = __import__("cierre_medido").como_se_mide(meal, chosen, no_cook, db)  # [P1-PLAN-LOTE-933]'
    assert gancho in cerrador and cerrador.index(gancho) < cerrador.index("gap = target - cur_p")
    assert len(src.splitlines()) <= 52_240
    modulo = _BACKEND / "cierre_medido.py"
    assert "tooltip-anchor: P1-PLAN-LOTE-933" in modulo.read_text(encoding="utf-8")
    assert b"\x08" not in modulo.read_bytes() and b"\r" not in modulo.read_bytes()
