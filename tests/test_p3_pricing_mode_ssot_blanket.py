"""[P3-PRICING-MODE-SSOT-BLANKET · 2026-08-22] El comentario del SSOT prohibía el segundo chequeo
a mano por escrito, y había uno.

`constants.pricing_mode_for_country` es la única puerta que decide si un país suprime montos en
RD$, y su comentario lo dice con estas palabras: «Un 2º chequeo `has_native_prices` a mano en
cualquiera de esos sitios sería la 2ª tabla que `P1-DIET-CANON-SSOT` ya pagó una vez (3 tablas de
dieta a mano, driftaron, una sirvió Pollo a vegetarianas)».

Medido: `agent.py` hacía exactamente eso —

    if not COUNTRY_PROFILES.get(_consent_cc, {}).get("has_native_prices", True):

— en el camino del consentimiento de la Nevera, para vaciar `est_price_rd` de los ingredientes que
faltan. Hoy es equivalente al SSOT, así que no hay bug de conducta; lo que hay es la segunda tabla
que el comentario pedía no escribir, y la lección de este repo es que las segundas tablas no nacen
divergiendo: divergen después, y en silencio.

POR QUÉ UN BLANKET Y NO UN ARREGLO PUNTUAL. Cambiar esa línea cierra el caso de hoy y no impide el
de mañana — el propio comentario ya existía y no lo impidió. Un lector que necesite la decisión
volverá a tener `COUNTRY_PROFILES` a mano y la respuesta le parecerá una línea. Este fichero hace
que esa línea falle en CI.

CUIDADO CON LA PROSA. La zona está llena de comentarios que NOMBRAN `has_native_prices` para
explicar por qué no hay que tocarlo — incluidos los del propio SSOT y dos de
`graph_orchestrator`. Un guard textual ingenuo los acusaría: comentario-vence-guard, que esta ola
ha pagado once veces. Por eso se escanea con AST y no con regex: un comentario no es un nodo del
árbol, así que no puede poner en rojo al texto que defiende la regla.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent

#: `constants.py` es el SSOT: ahí el acceso es legítimo (lo define). El resto del backend, no.
_EXENTOS = {"constants.py"}

#: Módulos productivos que se escanean. Se listan en vez de barrer el árbol para no arrastrar
#: `venv/`, `test_venv/` ni scripts one-shot.
_DIRS = (".", "routers", "prompts")


def _ficheros():
    for d in _DIRS:
        base = _BACKEND / d
        if not base.is_dir():
            continue
        for p in sorted(base.glob("*.py")):
            if p.name in _EXENTOS:
                continue
            yield p


class _Visitante(ast.NodeVisitor):
    """Busca cualquier acceso al literal `has_native_prices` fuera del SSOT."""

    def __init__(self):
        self.hits: list[int] = []

    def visit_Constant(self, node: ast.Constant):  # noqa: N802
        if isinstance(node.value, str) and node.value == "has_native_prices":
            self.hits.append(node.lineno)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):  # noqa: N802
        if node.attr == "has_native_prices":
            self.hits.append(node.lineno)
        self.generic_visit(node)


def test_nadie_consulta_has_native_prices_a_mano():
    """EL CASO. La decisión «¿este país suprime montos?» tiene UNA puerta:
    `pricing_mode_for_country`. Cualquier otro sitio que mire el flag es la segunda tabla."""
    violaciones = []
    for p in _ficheros():
        try:
            arbol = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:  # pragma: no cover
            continue
        v = _Visitante()
        v.visit(arbol)
        violaciones += [f"{p.name}:{n}" for n in v.hits]

    assert not violaciones, (
        "estos sitios consultan `has_native_prices` a mano en vez de llamar a "
        f"`constants.pricing_mode_for_country`: {violaciones}. Es la 2ª tabla que el comentario "
        f"del SSOT prohíbe por escrito, y la lección de P1-DIET-CANON-SSOT es que las segundas "
        f"tablas no nacen divergiendo — divergen después, en silencio"
    )


def test_el_ssot_sigue_siendo_el_unico_que_lo_define():
    """Guard del guard: si el flag desapareciera de `constants.py`, el test de arriba pasaría por
    vacuidad sobre un sistema que ya no tiene la decisión en ningún sitio."""
    src = (_BACKEND / "constants.py").read_text(encoding="utf-8", errors="replace")
    assert "has_native_prices" in src
    assert "def pricing_mode_for_country" in src


def test_la_puerta_sigue_respondiendo_lo_que_debe():
    """Funcional, no textual: mover el call site a la puerta no puede cambiar la conducta."""
    from constants import COUNTRY_PROFILES, pricing_mode_for_country

    assert pricing_mode_for_country("DO") is None
    for code, perfil in COUNTRY_PROFILES.items():
        esperado = None if perfil.get("has_native_prices", True) else "beta_no_prices"
        assert pricing_mode_for_country(code) == esperado, code
    # País desconocido: fail-safe a «no inventes un modo», igual que el chequeo a mano que
    # sustituye (`.get(cc, {}).get(..., True)` no suprimía nada para un código no registrado).
    assert pricing_mode_for_country("XX") is None


@pytest.mark.parametrize("modulo", ["agent.py"])
def test_el_call_site_del_consentimiento_usa_la_puerta(modulo):
    """El sitio concreto que se movió, anclado: si alguien lo revierte, que falle aquí con nombre
    y apellidos y no sólo en el blanket."""
    src = (_BACKEND / modulo).read_text(encoding="utf-8", errors="replace")
    assert "pricing_mode_for_country" in src, (
        f"{modulo} dejó de usar la puerta del SSOT para decidir si suprime montos"
    )
