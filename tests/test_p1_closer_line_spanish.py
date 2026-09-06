# -*- coding: utf-8 -*-
"""[P1-CLOSER-LINE-SPANISH · 2026-09-06] La línea del cerrador delataba a la máquina.

En el plan vivo `c5880d96` conviven las dos manos, a una comida de distancia:

    35g de soya texturizada cocido      ← el cerrador de proteína
    55 g de soya texturizada seca       ← el modelo

Dos diferencias, y ninguna es del modelo: **la cifra pegada a la unidad** y **el participio
hardcodeado en masculino** sobre un sustantivo femenino. Medido sobre 70 planes vivos de 14 días:
**113 líneas sin espacio en 53 planes**, 32 de ellas con el género equivocado (soya, avena, quinoa,
lentejas, habichuelas…).

No es cosmético del todo: el usuario lee la receta, y una línea que no se parece a las demás dice
«esto lo pegó un programa». La lista de compras y los pasos sí las tratan igual — el parser de
cantidades acepta las dos formas—, así que el daño es de lectura, no de cálculo.

**Por qué este caso sí y el general no.** En julio se descartó la concordancia de género general
«por falsos positivos: hace falta el núcleo del sintagma». Aquí el núcleo no hay que adivinarlo: el
cerrador acaba de elegir el alimento y su primera palabra ES el núcleo. `participio_concordado`
reusa `_NAME_ADJ_FEM` y `_NAME_FEM_FOODS`, que ya existían para el título — no es una tabla nueva.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

from graph_orchestrator import participio_concordado as pc  # noqa: E402

_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_NAMING = (_BACKEND / "dish_naming.py").read_text(encoding="utf-8")


# ── concordancia ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("alimento,esperado", [
    ("soya texturizada", "cocida"),      # el caso vivo
    ("quinoa", "cocida"),
    ("avena", "cocida"),
    ("pechuga de pollo", "cocida"),      # el núcleo es «pechuga», no «pollo»
    ("carne molida", "cocida"),
    ("leche descremada", "cocida"),      # termina en -e: la morfología no la ve, la tabla sí
    ("lentejas", "cocidas"),
    ("habichuelas rojas", "cocidas"),
    ("claras de huevo", "cocidas"),
    ("garbanzos", "cocidos"),
    ("frijoles pintos", "cocidos"),
    ("arroz blanco", "cocido"),
    ("atún en agua", "cocido"),
    ("tofu firme", "cocido"),
    ("edamame", "cocido"),
    ("huevo", "cocido"),
    ("queso ricotta", "cocido"),         # el núcleo es «queso», masculino
])
def test_el_participio_concuerda_con_el_nucleo(alimento, esperado):
    assert pc(alimento) == esperado, f"«{alimento} {pc(alimento)}»"


def test_el_nucleo_es_la_PRIMERA_palabra_no_la_ultima():
    """La razón por la que este caso se puede cerrar y el general no. «Pechuga de pollo» concuerda
    con pechuga; leerlo al revés daría «pechuga de pollo cocido», que es el error de siempre."""
    assert pc("pechuga de pollo") == "cocida"
    assert pc("filete de tilapia") == "cocido"   # núcleo «filete», masculino


@pytest.mark.parametrize("participio", ["cocido", "asado", "salteado", "molido", "picado"])
def test_vale_para_los_demas_participios_de_la_tabla(participio):
    """Reusa `_NAME_ADJ_FEM`, así que no hay que tocar nada si mañana se emite otro participio."""
    from graph_orchestrator import _NAME_ADJ_FEM
    assert pc("soya texturizada", participio) == _NAME_ADJ_FEM[participio]
    assert pc("arroz blanco", participio) == participio


def test_fail_safe_ante_basura():
    for basura in (None, "", "   ", 123):
        assert pc(basura) == "cocido"


def test_no_inventa_participios_fuera_de_la_tabla():
    """Un participio que la tabla no conoce vuelve tal cual: mejor sin concordar que inventado."""
    assert pc("soya texturizada", "flambeado") == "flambeado"


# ── formato de la línea ───────────────────────────────────────────────────────────────────────
def _lineas_de_ingrediente() -> list:
    """Los f-strings del orquestador que ESCRIBEN una línea de ingrediente (`line = f"…"`)."""
    return re.findall(r'line = f"\{[^"]*\}\s*g de [^"]*"', _SRC)


def test_ninguna_linea_pega_la_cifra_a_la_unidad():
    """El modelo escribe «55 g de …» y el cerrador escribía «55g de …». Una línea que no se parece a
    las demás delata al añadido dentro de la propia receta."""
    malas = [l for l in _lineas_de_ingrediente() if re.search(r"\}g de ", l)]
    assert not malas, f"líneas con la cifra pegada a la unidad: {malas}"


def test_hay_lineas_que_comprobar():
    """Si el regex deja de encontrar nada, el test de arriba pasaría vacío y no probaría nada."""
    assert len(_lineas_de_ingrediente()) >= 3, _lineas_de_ingrediente()


def test_los_dos_cerradores_usan_el_helper():
    """El participio ya no se escribe a mano en ninguno de los dos sitios que lo emitían."""
    assert 'else " cocido"' not in _SRC, "volvió el participio hardcodeado en masculino"
    assert _SRC.count("participio_concordado(nm)") >= 2


def test_el_helper_no_es_una_tabla_nueva():
    """La lección del repo: una cuarta tabla deriva. Este helper reusa las dos que ya existían."""
    i = _NAMING.find("def participio_concordado(")
    assert i > 0
    cuerpo = _NAMING[i:i + 1800]
    assert "_NAME_ADJ_FEM" in cuerpo and "_NAME_FEM_FOODS" in cuerpo
