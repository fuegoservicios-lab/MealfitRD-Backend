# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-931 · 2026-09-29] Los guisantes SECOS que pone el cerrador se cuecen, y el cerrador los deja para el final.

Corpus del VPS (566 planes): 15 platos con «N g de guisantes secos cocidos» añadidos por el cerrador y un único texto que
los nombra, «Acompaña con guisantes secos.» — guisantes partidos SIN cocer de guarnición, también en una merienda
(«Casabe tostado con mantequilla de maní, canela, nueces y guisantes secos», 80 g). Todos del perfil de compra mensual
con alergia al pescado: desde el día 4 el cerrador sólo puede añadir lo que aguanta el mes, y «Guisantes secos» es la
legumbre más magra de la lista. La base los mide en SECO (la fila dice «secos») aunque la línea diga «cocidos».

El detector de «seco sin cocción» (V7c), la nota de cocción previa (lote 375) y la legumbre lista (lote 545) no conocían
el guisante: `habichuela|frijol|lenteja|garbanzo|gandul|haba`. *Un hueco del vocabulario no lo ve NINGUNA capa.*
"""
from __future__ import annotations

import pathlib
import re

import graph_orchestrator as go
import pasos_cantidades as pc
import seco_al_final as saf
from culinary_coherence import _v7c_seco_sin_coccion, build_culinary_index

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_IDX = build_culinary_index([
    {"name": "Guisantes secos", "aliases": ["guisantes partidos", "arvejas secas"], "category": "Legumbres"},
    {"name": "Batata", "aliases": [], "category": "Víveres"},
    {"name": "Queso blanco", "aliases": ["queso"], "category": "Lácteos"},
])


def _cena():
    return {"meal": "Cena", "name": "Batata majada con queso blanco fresco, zanahoria salteada y guisantes secos",
            "ingredients": ["½ batata mediana", "40 g de queso blanco fresco", "40 g de guisantes secos cocidos"],
            "recipe": ["Mise en place: pela la batata y córtala en cubos.",
                       "El Toque de Fuego: hierve la batata 15-18 minutos y májala.",
                       "Montaje: sirve la batata majada con el queso blanco. Acompaña con guisantes secos y agua."]}


def test_el_detector_ve_los_guisantes_secos_que_nadie_cuece():
    hallazgos = _v7c_seco_sin_coccion({"day": 0}, _cena(), _IDX)
    assert [v.get("food") for v in hallazgos] == ["Guisantes secos"], hallazgos


def test_la_nota_de_coccion_previa_dice_como_cocerlos():
    m = _cena()
    assert pc.coccion_previa(m, _IDX) == 1
    nota = m["recipe"][1]
    assert nota.startswith("💡 Cocción previa: hierve los guisantes secos 40-60 min"), nota
    assert "tanda de varios días" in nota and m["recipe"][0].startswith("Mise en place")
    assert not _v7c_seco_sin_coccion({"day": 0}, m, _IDX)
    assert pc.coccion_previa(m, _IDX) == 0, "idempotente"
    assert [k for k, _t in pc._COCCION_PREVIA_375 if k in ("guisante", "arveja", "chicharo")] == ["guisante", "arveja", "chicharo"]


def test_los_guisantes_que_un_paso_ya_hierve_no_llevan_nota():
    m = _cena()
    m["recipe"][1] = "El Toque de Fuego: hierve los guisantes secos 45 minutos; hierve la batata 15-18 minutos y májala."
    assert not _v7c_seco_sin_coccion({"day": 0}, m, _IDX) and pc.coccion_previa(m, _IDX) == 0


class _Info:
    def __init__(self, name, protein, kcal, carbs=0.0, fats=0.0):
        self.name, self.protein, self.kcal, self.carbs, self.fats = name, protein, kcal, carbs, fats


_GUISANTES = _Info("Guisantes secos", 24.6, 341.0, 60.0, 1.2)
_HABICHUELAS = _Info("Habichuelas blancas", 23.4, 333.0, 60.0, 0.8)
_HUEVO = _Info("Huevo", 12.6, 139.0, 0.7, 9.5)


def test_lo_seco_va_al_final_del_pool():
    pool = [(_GUISANTES, "guisantes secos"), (_HABICHUELAS, "habichuelas blancas"), (_HUEVO, "huevo")]
    assert [n for _i, n in saf.ordenar(pool)] == ["habichuelas blancas", "huevo", "guisantes secos"]
    assert saf.ordenar(pool[:1]) == pool[:1], "sin otro candidato, se queda"
    assert saf.ordenar([]) == [] and saf.ordenar(None) is None
    limpio = [(_HABICHUELAS, "habichuelas blancas"), (_HUEVO, "huevo")]
    assert saf.ordenar(limpio) == limpio


class _DB:
    def grams_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\s+de\s+", str(s))
        return float(m.group(1).replace(",", ".")) if m else None

    def macros_from_ingredient_string(self, s):
        g = self.grams_from_ingredient_string(s)
        return None if g is None else {"grams": g, "kcal": g * 1.2, "protein": g * 0.1, "carbs": g * 0.1, "fats": g * 0.03}

    def __getattr__(self, _n):
        return lambda *a, **k: None


def test_el_cerrador_elige_la_legumbre_que_no_es_seca():
    cena = {"meal": "Cena", "name": "Batata majada con queso blanco fresco y zanahoria salteada",
            "protein": 12, "cals": 380, "carbs": 60, "fats": 9,
            "ingredients": ["½ batata mediana", "40 g de queso blanco fresco", "½ zanahoria"],
            "recipe": ["El Toque de Fuego: hierve la batata 15-18 minutos y májala.", "Montaje: sirve la batata con el queso."]}
    base = list(cena["ingredients"])
    g = go._close_protein_gap_for_meal(cena, 30.0, _DB(), [(0.0, i.name, i) for i in (_GUISANTES, _HABICHUELAS)],
                                       allergies=None, fill_pct=0.92, max_add_g=120, enforce_min_threshold=False,
                                       day_used_proteins=set(), diet=None, country="DO", goal="lose_fat")
    nuevas = [x for x in cena["ingredients"] if x not in base]
    assert g > 0 and nuevas and "habichuelas blancas" in nuevas[0] and "guisantes" not in nuevas[0], nuevas


def test_con_el_knob_apagado_el_orden_de_siempre(monkeypatch):
    monkeypatch.setenv("MEALFIT_CLOSER_DRY_LAST", "false")
    pool = [(_GUISANTES, "guisantes secos"), (_HABICHUELAS, "habichuelas blancas")]
    assert saf.ordenar(pool) == pool


def test_anclas():
    go_src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert ('_pool = __import__("seco_al_final").ordenar(__import__("topes_por_linea").caben(meal, _pool, db, '
            'CLOSER_COOKABLE_MIN_G))  # [P1-PLAN-LOTE-889] tope por alimento · [P1-PLAN-LOTE-931]') in go_src
    assert len(go_src.splitlines()) <= 52_240
    assert "tooltip-anchor: P1-PLAN-LOTE-931" in (_BACKEND / "seco_al_final.py").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-931" in (_BACKEND / "pasos_cantidades.py").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-931" in (_BACKEND / "culinary_coherence.py").read_text(encoding="utf-8")
    for f in ("seco_al_final.py", "pasos_cantidades.py", "culinary_coherence.py", "graph_orchestrator.py"):
        assert b"\x08" not in (_BACKEND / f).read_bytes(), f
