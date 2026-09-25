# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-249 · 2026-09-25] Las notas de pescado/mariscos para el revisor miran la CLASE declarada.

Batería real rd248, perfil «sin gluten y sin huevo»: el revisor rechazó «pescado blanco… excluido por la alergia al
pescado de aleta». El usuario no declaró pescado: «bacalaítos» vive en la clase gluten (la masa) Y en la de pescado, y
el lote 227 decidía la nota por intersección de sinónimos. Con 240 en producción, todo alérgico al gluten recibía la
nota falsa (y el revisor le quitaba el pescado).
"""
from __future__ import annotations

import copy
import itertools
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import etiquetas_clinicas as ec  # noqa: E402

# Chips del formulario (QAllergies.jsx) salvo las dos clases del mar.
_OTROS_CHIPS = ["Lacteos", "Gluten", "Huevo", "Frutos Secos", "Mani", "Soya", "Sesamo", "Lactosa"]


def test_gluten_no_declara_pescado():
    assert ec._pescado_sin_mariscos({"allergies": ["Gluten"]}) == ""
    assert ec.notas_para_el_revisor({"allergies": ["Gluten"]}) == ("", "")
    assert ec.notas_para_el_revisor({"allergies": ["Gluten", "Huevo"]}) == ("", "")


@pytest.mark.parametrize("combo", [c for n in (1, 2) for c in itertools.combinations(_OTROS_CHIPS, n)])
def test_ningun_otro_chip_dispara_notas_del_mar(combo):
    fd = {"allergies": list(combo)}
    assert ec._pescado_sin_mariscos(fd) == ""
    assert ec._solo_mariscos(fd) is False
    assert ec.notas_para_el_revisor(fd) == ("", "")


def test_mariscos_con_gluten_sigue_siendo_solo_mariscos():
    # Antes: «bacalaítos» (del gluten) contaba como pescado declarado y la nota de mariscos se perdía.
    assert ec._solo_mariscos({"allergies": ["Mariscos", "Gluten"]}) is True
    mariscos, pescado = ec.notas_para_el_revisor({"allergies": ["Mariscos", "Gluten"]})
    assert "pescado está PERMITIDO" in mariscos and pescado == ""


def test_la_nota_al_plato_tambien_con_gluten():
    plan = {"days": [{"day": 1, "meals": [{"meal": "Almuerzo", "name": "Tilapia al horno",
                                            "ingredients": ["150 g de filete de tilapia", "1 taza de arroz"],
                                            "recipe": ["Hornea la tilapia 15 min."]}]}]}
    p = copy.deepcopy(plan)
    assert ec._nota_mariscos_no_pescado(p, {"allergies": ["Mariscos", "Gluten"]}) == 1


@pytest.mark.parametrize("fd,esperado", [
    ({"allergies": ["Pescado"]}, "alergia"),
    ({"allergies": ["Pescado", "Gluten"]}, "alergia"),
    ({"otherAllergies": "tilapia"}, "alergia"),
    ({"dislikes": ["Pescado"]}, "rechazo"),
    ({"allergies": ["Gluten"], "dislikes": ["Pescado"]}, "rechazo"),
    ({"allergies": ["Pescado", "Mariscos"]}, ""),
    ({"allergies": ["Pescado"], "dislikes": ["Mariscos"]}, ""),
    ({"otherAllergies": "seafood"}, ""),
    ({}, ""),
])
def test_pescado_sin_mariscos_por_clase(fd, esperado):
    assert ec._pescado_sin_mariscos(fd) == esperado


@pytest.mark.parametrize("fd,esperado", [
    ({"allergies": ["Mariscos"]}, True),
    ({"allergies": ["Mariscos", "Pescado"]}, False),
    ({"otherAllergies": "seafood"}, False),
    ({"allergies": ["Pescado"]}, False),
    ({}, False),
])
def test_solo_mariscos_por_clase(fd, esperado):
    assert ec._solo_mariscos(fd) is esperado


def test_ancla():
    src = (_BACKEND / "etiquetas_clinicas.py").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-249-CLASE-DECLARADA" in src
    assert "exp & pes" not in src and "& mar:" not in src


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 249
