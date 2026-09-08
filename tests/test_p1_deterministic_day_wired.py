# -*- coding: utf-8 -*-
"""[P1-DETERMINISTIC-DAY-WIRED · 2026-09-08] El enganche al pipeline, anclado aparte.

`test_p1_deterministic_day.py` prueba que el ensamblador FUNCIONA. Este prueba que está
ENCHUFADO, y son dos contratos distintos: hoy mismo encontramos `recipe_for_dish_name` con
cero call sites — una función correcta, probada y desplegada que no servía a nadie. Separarlos
hace que borrar el enganche falle el test del enganche, no el del algoritmo.
"""
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def test_el_ensamblador_esta_enchufado_al_generador_de_dias():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "from deterministic_day import build_day_for_skeleton as _det_day" in src, (
        "sin el import, el ensamblador vuelve a ser una feature inerte")
    assert ("_det_day(nutrition, form_data, skel_day, day_num) or "
            "await _generate_day_hedged") in src, (
        "el determinista va PRIMERO y el LLM es el `or`. Invertirlo deja el día determinista "
        "inalcanzable — inerte con toda la apariencia de estar enchufado")


def test_el_enganche_no_engorda_el_god_file():
    """`graph_orchestrator.py` tiene techo duro y superarlo NO se arregla subiendo el número: se
    arregla extrayendo. Este enganche cabe en 0 líneas netas — import colgado de una línea que ya
    existía y una línea sustituida por otra."""
    n = len((_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8").splitlines())
    assert n <= 53100, f"el god-file subió a {n} líneas: extraer, no subir el cap"


def test_el_modulo_no_importa_el_grafo():
    """`deterministic_day` no puede depender de `graph_orchestrator`: sería un ciclo, y además lo
    haría imposible de probar sin montar el grafo entero.

    Se mira el AST, no el texto. Las menciones narrativas en comentarios son historia legítima —
    el módulo explica en su docstring que un día que sale de ahí bypasea `assemble_plan_node`, y
    eso es exactamente lo que el lector necesita saber. Es la misma distinción que el blanket
    anti-Gemini ya hace entre usar un API y contar por qué no se usa.
    """
    import ast

    arbol = ast.parse((_BACKEND / "deterministic_day.py").read_text(encoding="utf-8"))
    modulos = set()
    for n in ast.walk(arbol):
        if isinstance(n, ast.Import):
            modulos.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            modulos.add(n.module.split(".")[0])
    assert "graph_orchestrator" not in modulos, (
        f"importa el grafo — ciclo y test imposible. Importa: {sorted(modulos)}")
