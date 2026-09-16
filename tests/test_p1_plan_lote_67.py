# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-67 · 2026-09-16] Los dos hallazgos de las generaciones reales del 16-sep, cerrados.

(A) **Mezclar no es cocer.** Un batido del plan vegano REAL disparó cinco V1 de golpe —Maní, Linaza, Vainilla, Canela
    en polvo y Sal— por la rama «listo-para-comer»: licuar es MEZCLA, y el propio V7f ya lo clasifica así
    (`_V7F_MEZCLA_RE`). Se exime esa rama y NADA más: a un alimento que no es listo-para-comer se le sigue exigiendo
    que `prep_methods` admita el licuado, por eso «licúa las habas» del mismo plan SIGUE acusando — ahí la cadena de
    reparación metió habas crudas a la licuadora. `Avena` y `Leche de avena` se cierran por catálogo (migración de
    alcance mínimo, con las dos recetas reales citadas), no silenciando la regla.

(B) **Mencionar no es trabajar.** `_append_closer_protein_step` se saltaba su paso de cocción si CUALQUIER paso
    nombraba el alimento — y un «Montaje: … Acompaña con pechuga de pollo» bastaba. Reproducido sobre el plato real
    (día 1 cena «Ñame Guisado…»): devolvía False y el plato entregaba 222,75 g de pechuga CRUDA con 45 g de proteína
    contados. Ahora una cláusula que sólo emplata no cuenta; las que cuecen o manipulan siguen contando (el yogurt
    frío que esta guarda protege).
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_MIGRACION = _BACKEND / "migrations" / "p1_plan_lote_67_licuar_prep_method_2026_09_16.sql"


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _fila(nombre, prep, rte, cat="Despensa"):
    return {"name": nombre, "prep_methods": prep, "ready_to_eat": rte, "category": cat, "aliases": []}


def _plan(paso, ingredientes):
    return {"days": [{"day": 1, "meals": [{"meal": "Desayuno", "name": "Batido", "recipe": [paso],
                                           "ingredients": ingredientes}]}]}


# ─────────────────────────────── (A) mezclar no es cocer

def test_la_clausula_que_licua_tres_o_mas_alimentos_no_acusa_a_sus_componentes():
    """El batido del plan real: lo que se licúa es la MEZCLA, no la canela por su cuenta."""
    import culinary_coherence as cc
    filas = [_fila("Avena", ["hervir", "ninguno", "tostar"], False),
             _fila("Leche de avena", ["hervir", "ninguno"], False, cat="Lácteos"),
             _fila("Maní", ["ninguno", "tostar"], True), _fila("Linaza", ["ninguno", "tostar"], True),
             _fila("Canela en polvo", ["ninguno"], True), _fila("Sal", ["ninguno"], True)]
    plan = _plan("El Toque de Fuego: licúa la avena con la leche de avena, la mitad del maní, la linaza, "
                 "la canela y la pizca de sal hasta obtener una masa fluida.",
                 ["40 g de avena", "5 ml de leche de avena", "20 g de maní", "10 g de linaza",
                  "¼ cdta de canela en polvo", "½ pizca de sal"])
    v1 = [v for v in (cc.culinary_contract_scan(plan, filas) or []) if v["check"] == "V1"]
    assert v1 == [], v1


def test_hervir_sobre_un_listo_para_comer_sigue_acusando():
    """La exención es del método, no de la rama: cocer un listo-para-comer sigue siendo un hallazgo."""
    import culinary_coherence as cc
    filas = [_fila("Canela en polvo", ["ninguno"], True)]
    plan = _plan("El Toque de Fuego: hierve la canela en polvo 10 minutos.", ["1 cdta de canela en polvo"])
    v1 = [v for v in (cc.culinary_contract_scan(plan, filas) or []) if v["check"] == "V1"]
    assert len(v1) == 1 and v1[0]["food"] == "Canela en polvo", v1


def test_licuar_un_alimento_que_no_es_listo_para_comer_sigue_acusando():
    """El caso `Habas` del plan real: crudas a la licuadora, y `prep_methods` no admite licuar."""
    import culinary_coherence as cc
    filas = [_fila("Habas", ["hervir", "guisar"], False, cat="Legumbres")]
    plan = _plan("El Toque de Fuego: agrega habas a la licuadora y licúa hasta integrar.", ["100 g de habas"])
    v1 = [v for v in (cc.culinary_contract_scan(plan, filas) or []) if v["check"] == "V1"]
    assert len(v1) == 1 and v1[0]["food"] == "Habas", v1


def test_la_clausula_de_mezcla_sigue_exigiendo_tres_alimentos():
    """El umbral es la defensa: con DOS alimentos no hay mezcla que juzgar y la regla sigue acusando."""
    import culinary_coherence as cc
    filas = [_fila("Jamón serrano", ["ninguno"], True, cat="Proteínas"),
             _fila("Casabe", ["plancha", "guisar"], True, cat="Víveres")]
    plan = _plan("El Toque de Fuego: licúa el jamón serrano con el casabe.",
                 ["30 g de jamón serrano", "1 casabe"])
    v1 = [v for v in (cc.culinary_contract_scan(plan, filas) or []) if v["check"] == "V1"]
    assert len(v1) == 2, v1


def test_no_se_toco_el_catalogo():
    """No hay migración en este lote, a propósito: con la regla arreglada no queda falso positivo vivo que la
    justifique, y tocar `prep_methods` por intuición es lo que P1-LIBRARY-V3-SIN-SUSTITUTO prohibió por escrito."""
    assert not list((_BACKEND / "migrations").glob("*lote_67*")), "este lote no toca el catálogo"


# ─────────────────────────────── (B) mencionar no es trabajar

@pytest.fixture
def plato_del_plan_real():
    """El plato tal cual salió: guiso de ñame cuyo ÚNICO rastro del pollo es «Acompaña con pechuga de pollo»."""
    return {"meal": "Cena", "name": "Ñame Guisado en Salsa Criolla con Espinacas, Maní Tostado y Pechuga de pollo",
            "ingredients": ["5.27 g de ñame pelado en cubos", "222.75 g de pechuga de pollo"],
            "recipe": ["Mise en place: pela el ñame y córtalo en cubos pequeños de 2 cm; pica cebolla, ajo y tomate.",
                       "El Toque de Fuego: en una sartén profunda, calienta el aceite y sofríe cebolla, ajo y ají 4 minutos.",
                       "Montaje: sirve el guiso en un plato hondo, corona con el maní tostado. Acompaña con pechuga de pollo."]}


def test_el_montaje_que_solo_sirve_ya_no_silencia_al_cerrador(plato_del_plan_real):
    from graph_orchestrator import _append_closer_protein_step
    m = copy.deepcopy(plato_del_plan_real)
    assert _append_closer_protein_step(m, "Pechuga de pollo", False) is True, \
        "un plato que sólo MENCIONA la pechuga al servir necesita su paso de cocción"
    pasos = " ".join(str(s) for s in m["recipe"])
    assert "💪" in pasos and re.search(r"(?i)(cocina|plancha|hervid|guiso)", pasos), pasos


def test_si_la_receta_ya_lo_cuece_no_se_repite_el_paso():
    from graph_orchestrator import _append_closer_protein_step
    m = {"meal": "Almuerzo", "name": "Pollo guisado", "ingredients": ["200 g de pechuga de pollo"],
         "recipe": ["El Toque de Fuego: sofríe la cebolla y cocina la pechuga de pollo 12 minutos hasta dorarla.",
                    "Montaje: sirve caliente."]}
    assert _append_closer_protein_step(m, "Pechuga de pollo", False) is False, \
        "la receta ya la cuece: el paso genérico sólo ensuciaría"


def test_si_la_receta_lo_incorpora_en_frio_tampoco_se_repite():
    """La guarda del yogurt/cottage: incorporar y mezclar SIGUEN contando como trabajar el alimento."""
    from graph_orchestrator import _append_closer_protein_step
    m = {"meal": "Merienda", "name": "Lechosa con yogurt", "ingredients": ["150 g de yogurt griego"],
         "recipe": ["Mise en place: corta la lechosa en cubos.",
                    "Montaje: incorpora el yogurt griego y mézclalo con la lechosa antes de servir."]}
    assert _append_closer_protein_step(m, "Yogurt griego", True) is False


# ─────────────────────────────── docs y marker

def test_docs_y_marker():
    doc = _src(_BACKEND / "docs" / "culinary_coherence.md")
    assert "P1-PLAN-LOTE-67" in doc and "licuar" in doc
    m = re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src(_BACKEND / "app.py"), re.M)
    assert m and int(m.group(1)) >= 67 and m.group(2) >= "2026-09-16"
