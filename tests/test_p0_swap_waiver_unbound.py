# [P0-SWAP-WAIVER-UNBOUND · 2026-08-09] El waiver de nevera-vacía (P1-SWAP-EMPTY-PANTRY-WAIVER)
# se insertó ~400 líneas ANTES de que `strict_pantry` naciera: su asignación convirtió el nombre
# en LOCAL para toda la función → UnboundLocalError en el primer read → **500 en el 100% de los
# swaps** (guest Y autenticados) durante ~1 día, hasta que la corrida 31304538636 lo midió
# (swap ok_pct=0.0, latency p50=0.2s — un fallo instantáneo no es un timeout de LLM).
#
# Doble lección de instrumental:
#   1. El test original era parser-based y validaba PROXIMIDAD de texto («cerca de un
#      `if clean_ingredients:`») — había VARIOS y ancló al equivocado. Certificaba texto,
#      no ejecutabilidad (clase P1-G).
#   2. «Cero regresiones» de la suite era verdad: ningún test EJECUTA el path del swap.
#      Este test verifica el ORDEN por AST — la propiedad que de verdad rompió producción.
import ast
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_AGENT_PATH = os.path.join(os.path.dirname(__file__), "..", "agent.py")
_SRC = open(_AGENT_PATH, encoding="utf-8").read()


def _swap_meal_func():
    tree = ast.parse(_SRC)
    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    # la función que contiene el waiver (la asignación del knob puede partirse en 2 líneas)
    marker_line = next(i + 1 for i, l in enumerate(_SRC.splitlines())
                       if "_swap_empty_waiver_on = os.environ.get(" in l)
    cands = [f for f in funcs if f.lineno <= marker_line <= f.end_lineno]
    assert cands, "el waiver debe vivir dentro de una función (swap_meal)"
    return max(cands, key=lambda f: f.lineno), marker_line


def test_ningun_read_de_strict_pantry_antes_de_su_primera_asignacion():
    # La propiedad EXACTA que rompió prod: en la función del waiver, ninguna LECTURA de
    # `strict_pantry` puede preceder a su primera ASIGNACIÓN. AST, no proximidad de texto.
    func, _ = _swap_meal_func()
    first_store = None
    first_load = None
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and node.id == "strict_pantry":
            if isinstance(node.ctx, ast.Store):
                if first_store is None or node.lineno < first_store:
                    first_store = node.lineno
            else:
                if first_load is None or node.lineno < first_load:
                    first_load = node.lineno
    assert first_store is not None, "strict_pantry debe asignarse en swap_meal"
    assert first_load is not None, "strict_pantry debe leerse en swap_meal"
    assert first_store < first_load, (
        f"LECTURA de strict_pantry (línea {first_load}) ANTES de su primera asignación "
        f"(línea {first_store}) → UnboundLocalError → 500 en TODOS los swaps (incidente "
        f"P0-SWAP-WAIVER-UNBOUND, ~1 día de swaps rotos en prod)"
    )


def test_waiver_despues_de_la_resolucion_de_strictness():
    # El waiver debe correr DESPUÉS de la asignación canónica (P4-UPDATE-DISHES-STRICT-ALL)
    # y del override de discovery, y ANTES del raise honesto — orden por posición en el
    # fuente (misma función, flujo lineal).
    i_assign = _SRC.index("strict_pantry = True if _strict_all")
    i_waiver = _SRC.index("if strict_pantry and not clean_ingredients and _swap_empty_waiver_on:")
    i_raise = _SRC.index("SWAP_STRICT_PANTRY_NO_INVENTORY: el usuario eligió una razón")
    assert i_assign < i_waiver < i_raise, (
        "orden requerido: nace strict_pantry → waiver → raise honesto"
    )


def test_waiver_sigue_completo():
    # El contenido del waiver (kill switch + desactivación real) sigue vivo tras la reubicación.
    i = _SRC.index("if strict_pantry and not clean_ingredients and _swap_empty_waiver_on:")
    blk = _SRC[i:i + 500]
    assert "strict_pantry = False" in blk
    assert "MEALFIT_SWAP_EMPTY_PANTRY_WAIVER" in _SRC
