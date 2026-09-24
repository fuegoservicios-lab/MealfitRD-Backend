# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-210 · 2026-09-24] Mariscos ≠ pescado, escrito — y «seafood» protege de los dos.

El revisor médico preguntó «confirmar si incluye pescado» en 3 de 4 corridas del perfil con alergia a mariscos: el plan
sirve pescado porque el usuario marcó sólo el chip «Mariscos» (distinto del chip «Pescado»), pero nada lo decía. Y el chip
«Mariscos» usaba el MISMO icono de pez que «Pescado». Además, «seafood» (inglés: pez y marisco) sólo cubría el pescado.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator
    return graph_orchestrator


def _plan():
    return {"days": [{"meals": [
        {"meal": "Almuerzo", "name": "Tilapia al horno", "ingredients": ["150 g de Tilapia", "1 taza de arroz"],
         "recipe": ["Hornea."]},
        {"meal": "Cena", "name": "Pollo guisado", "ingredients": ["150 g de pechuga de pollo"], "recipe": ["Cocina."]},
        {"meal": "Merienda", "name": "Ensalada de atún", "ingredients": ["1 lata de Atún en agua"], "recipe": ["Mezcla."]},
    ]}]}


def test_solo_mariscos_escribe_la_nota_en_los_platos_con_pescado():
    import etiquetas_clinicas as ec
    plan = _plan()
    assert ec._nota_mariscos_no_pescado(plan, {"allergies": ["Mariscos", "Lacteos"]}) == 2
    notas = [m["recipe"][-1] for m in plan["days"][0]["meals"]]
    assert notas[0].startswith("⚕️ Alergia a mariscos") and notas[2].startswith("⚕️ Alergia a mariscos")
    assert notas[1] == "Cocina.", "sin pescado, sin nota"
    assert ec._nota_mariscos_no_pescado(plan, {"allergies": ["Mariscos"]}) == 0, "idempotente"


@pytest.mark.parametrize("alergias", [["Mariscos", "Pescado"], ["Pescado"], ["seafood"], ["Ninguna"], []])
def test_sin_nota_si_el_pescado_tambien_esta_declarado_o_no_hay_mariscos(alergias):
    import etiquetas_clinicas as ec
    assert ec._nota_mariscos_no_pescado(_plan(), {"allergies": alergias}) == 0, alergias


def test_la_nota_entra_por_la_puerta_de_etiquetas(go):
    import etiquetas_clinicas as ec
    plan = _plan()
    assert ec.etiquetar(plan, {"allergies": ["Mariscos"]}) >= 2
    src = (_BACKEND / "etiquetas_clinicas.py").read_text(encoding="utf-8")
    assert "n += _nota_mariscos_no_pescado(plan, form_data)" in src


def test_seafood_protege_de_pescado_y_de_mariscos(go):
    for ingrediente in ("Atún", "Almejas", "Camarones"):
        assert go.clinical_backstop_for_meal({"meal": "Almuerzo", "name": "x", "ingredients": [f"100 g de {ingrediente}"]},
                                             allergies=["seafood"]), ingrediente


def test_mariscos_sigue_sin_quitar_el_pescado(go):
    """La distinción del formulario se respeta: el chip «Mariscos» no excluye la tilapia."""
    assert not go.clinical_backstop_for_meal({"meal": "Almuerzo", "name": "x", "ingredients": ["150 g de Tilapia"]},
                                             allergies=["Mariscos"])


def test_el_chip_de_mariscos_ya_no_es_un_pez():
    ui = _BACKEND.parent / "frontend" / "src" / "components" / "assessment" / "questions" / "QAllergies.jsx"
    if not ui.exists():
        pytest.skip("árbol del frontend ausente")
    src = ui.read_text(encoding="utf-8")
    assert '{ val: "Mariscos", label: t(\'Mariscos\'), icon: Shrimp }' in src
    assert '{ val: "Pescado", label: t(\'Pescado\'), icon: Fish }' in src


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 210 and m.group(2) >= "2026-09-24"
