# -*- coding: utf-8 -*-
"""[P1-OATS-NOT-A-DINNER · 2026-09-06] «Comida fuera de horario» era el motivo de rechazo MÁS frecuente del
revisor: 16 veces en cuatro días, cada una un plan regenerado entero. Contados en el journal, casi todos eran
avena en almuerzo o cena — «Tortitas de avena y atún», «Bowl cremoso de avena salada», «Arepitas de avena».

La causa no era el modelo inventando: era el prompt ORDENÁNDOSELO. La regla de bases dice «el Almuerzo y la
Cena deben llevar bases DISTINTAS: una con X y la otra con Y», y cuando el pool del día trae avena, X o Y ES la
avena. El sembrador ya la excluye de sus parejas (`ai_helpers._base_carbs_for_pairs`), pero sus parejas solo
COMPLETAN pools cortos: cuando el planificador llenó el pool con dos bases y una era avena, esa limpieza nunca
llegaba a aplicarse.

Y un detalle que se comió el primer intento de arreglo: la regla se asignaba con `=`, así que sobrescribía
cualquier aviso añadido antes."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

from prompts.day_generator import build_day_assignment_context  # noqa: E402


def _ctx(pool):
    sk = {"protein_pool": ["Pollo"], "carb_pool": list(pool), "fruit_pool": ["Guineo"],
          "meal_types": ["Desayuno", "Almuerzo", "Merienda", "Cena"],
          "assigned_technique": "guisado", "brief_concept": "Criollo"}
    return build_day_assignment_context(sk, 2)


def _regla_de_bases(ctx):
    return next((l.strip() for l in ctx.splitlines()
                 if "NO REPITAS LA BASE" in l or "base del Almuerzo" in l), "")


@pytest.mark.parametrize("pool", [["Avena", "Yuca"], ["Avena", "Yuca", "Arroz"], ["Granola", "Papa", "Yuca"]])
def test_el_cereal_nunca_se_nombra_como_base_de_comida_fuerte(pool):
    regla = _regla_de_bases(_ctx(pool)).lower()
    assert regla, f"sin regla de bases para {pool}"
    assert "avena" not in regla or "va al desayuno" in regla, regla
    assert "granola" not in regla or "va al desayuno" in regla, regla


@pytest.mark.parametrize("pool", [["Avena", "Yuca"], ["Avena", "Granola"], ["Granola", "Papa", "Yuca"]])
def test_el_aviso_sale_siempre_que_haya_cereal_en_el_pool(pool):
    """El aviso se perdía: la regla siguiente se asignaba con `=` en vez de `+=`."""
    ctx = _ctx(pool)
    assert "base de DESAYUNO o MERIENDA" in ctx, pool


def test_con_una_sola_base_fuerte_se_reparte_en_vez_de_prohibir():
    """El caso del plan vivo: pool ['Avena','Yuca']. Pedir «dos bases distintas» con una sola fuerte sería
    insatisfacible, y una restricción imposible es como el gate de la fruta acabó forzando el 67 % de
    reintentos. Se reparte: la fuerte a las dos comidas fuertes, el cereal al desayuno."""
    regla = _regla_de_bases(_ctx(["Avena", "Yuca"]))
    assert "La base del Almuerzo y de la Cena es 'Yuca'" in regla, regla


def test_sin_ninguna_base_fuerte_no_se_inventa_una_regla():
    """Pool solo de cereales: nombrar avena y granola para almuerzo y cena es peor que no decir nada."""
    ctx = _ctx(["Avena", "Granola"])
    assert not _regla_de_bases(ctx), _regla_de_bases(ctx)
    assert "base de DESAYUNO o MERIENDA" in ctx, "el aviso sí se queda"


def test_el_pool_sin_cereales_conserva_la_regla_de_siempre():
    regla = _regla_de_bases(_ctx(["Yuca", "Arroz"]))
    assert "NO REPITAS LA BASE" in regla and "Yuca" in regla and "Arroz" in regla


def test_la_lista_de_cereales_es_la_del_sembrador():
    """SSOT: una segunda lista aquí se desincronizaría con `_base_carbs_for_pairs` — la lección de
    P1-DIET-CANON-SSOT, que empezó con tres tablas de dieta escritas a mano."""
    src = (_BACKEND / "prompts" / "day_generator.py").read_text(encoding="utf-8")
    assert "from ai_helpers import _BREAKFAST_ONLY_BASES" in src
