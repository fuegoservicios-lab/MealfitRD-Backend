"""[P1-CHATMODIFY-DISH-COUNTRY · 2026-08-23] El chat-modify pedía inspiración a la biblioteca de
platos SIN país.

`P1-DISH-LIBRARY-COUNTRY` se declaró cerrado con «3 superficies (day-gen, swap, chat-modify)». La
tercera no estaba: `tools.execute_modify_single_meal` llamaba a `build_swap_inspiration_context`
sin `country=`, aunque el país ya venía derivado en esa misma función (`_modify_country`, por la
ÚNICA puerta) y se usaba en cinco sitios más.

Medido antes del arreglo (knob de países encendido):

    build_swap_inspiration_context('Almuerzo', seed=3)              (la llamada de tools.py)
      → «Pastelón de yuca con pollo desmenuzado; La bandera: arroz, habichuelas rojas…»
    build_swap_inspiration_context('Almuerzo', seed=3, country='ES')
      → «Empanada gallega de sardinas; Tortilla española con patata y cebolla»
    …country='MX' → «Enchiladas verdes de pollo al horno; Huachinango al horno con hierbas»

O sea: al usuario de Madrid que pedía «cámbiame la cena» se le inyectaban platos dominicanos en el
tramo MÁS concreto y MÁS cercano a la generación, que es justo donde P1-DIET-BLIND-DIRECTIVES midió
que el modelo obedece al ejemplo y no a la directiva de cabecera.

El guard es AST (no texto): resuelve el alias real del import dentro de `tools.py` y exige que
TODA llamada a la biblioteca de inspiración lleve `country=`. Un comentario no puede satisfacerlo
porque el árbol sintáctico no tiene comentarios; y no se ancla a la grafía del alias (`_bsi_cm`),
que es exactamente lo que un renombre rompería sin que el defecto volviera.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_TOOLS = _BACKEND / "tools.py"

_LIBRARY_FN = "build_swap_inspiration_context"


def _aliases_de_la_biblioteca(tree: ast.AST) -> set:
    """Nombres locales con los que `tools.py` invoca `build_swap_inspiration_context`."""
    alias = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "dish_library":
            for nm in node.names:
                if nm.name == _LIBRARY_FN:
                    alias.add(nm.asname or nm.name)
    return alias


def _llamadas(tree: ast.AST, nombres: set) -> list:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in nombres:
            out.append(node)
    return out


def test_tools_importa_la_biblioteca_de_inspiracion():
    """Ancla del guard: si el import desaparece, el resto de asserts sería vacuo."""
    tree = ast.parse(_TOOLS.read_text(encoding="utf-8"))
    assert _aliases_de_la_biblioteca(tree), (
        f"`tools.py` ya no importa `{_LIBRARY_FN}` de dish_library — este guard quedaría inerte; "
        "si la superficie se movió, mueve el guard con ella."
    )


def test_toda_llamada_de_chat_modify_a_la_biblioteca_lleva_pais():
    tree = ast.parse(_TOOLS.read_text(encoding="utf-8"))
    llamadas = _llamadas(tree, _aliases_de_la_biblioteca(tree))
    assert llamadas, "no se encontró ninguna llamada a la biblioteca de inspiración en tools.py"
    sin_pais = [c.lineno for c in llamadas
                if not any(kw.arg == "country" for kw in c.keywords)]
    assert not sin_pais, (
        f"llamadas a `{_LIBRARY_FN}` sin `country=` en tools.py, líneas {sin_pais}: el chat-modify "
        "volvería a inyectar platos dominicanos a un usuario de España (medido: 'Pastelón de yuca' "
        "vs 'Empanada gallega de sardinas')."
    )


def test_el_pais_que_se_pasa_es_el_derivado_una_sola_vez():
    """No vale re-derivar el país aquí: `_modify_country` es la derivación única de la función."""
    tree = ast.parse(_TOOLS.read_text(encoding="utf-8"))
    llamadas = _llamadas(tree, _aliases_de_la_biblioteca(tree))
    for call in llamadas:
        kw = next(k for k in call.keywords if k.arg == "country")
        assert isinstance(kw.value, ast.Name), (
            f"línea {call.lineno}: `country=` debería ser la variable ya derivada, no una "
            f"expresión nueva ({ast.dump(kw.value)[:80]}) — una segunda derivación es la tabla "
            "que P1-DIET-CANON-SSOT ya pagó una vez."
        )


@pytest.mark.parametrize("cc", ["ES", "MX"])
def test_la_biblioteca_de_verdad_cambia_con_el_pais(monkeypatch, cc):
    """Si la biblioteca fuese ciega al país, el `country=` de arriba sería decorativo."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    from dish_library import build_swap_inspiration_context as bsi

    do = bsi("Almuerzo", seed=3)
    beta = bsi("Almuerzo", seed=3, country=cc)
    if not do or not beta:
        pytest.skip("biblioteca de platos deshabilitada en este entorno")
    assert do != beta, (
        f"la inspiración de {cc} es byte-idéntica a la dominicana: pasar `country=` no arregla nada."
    )
