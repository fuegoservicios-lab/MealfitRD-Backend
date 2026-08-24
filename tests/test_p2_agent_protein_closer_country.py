"""[P2-AGENT-PROTEIN-CLOSER-COUNTRY · 2026-08-23] Los call sites del closer de proteína dentro de
`agent.swap_meal` no pasaban el país.

`graph_orchestrator` sí se lo pasa en sus cuatro call sites; `agent.py` en ninguno de los suyos.
Medido antes del arreglo (knob de países encendido, catálogo vivo):

    _safe_high_density_proteins([], IngredientNutritionDB(), min_protein=18.0)   → 36 candidatos
    …lo mismo con country='ES'                                                   → 43

    Lo que el español perdía HOY en el swap: ['Pechuga de pollo', 'Percebes', 'Anchoas',
    'Jamón ibérico', 'Lomo embuchado', 'Jamón serrano', 'Chorizo español'].
    Lo que DO perdía: ninguno (el pool dominicano es subconjunto del beta).

El daño es pérdida de opciones, no comida equivocada —los que quedan a ≥18 g/100 g son
universales—, por eso P2 y no P1.

SON SEIS CALL SITES, NO TRES: `_close_protein_gap_for_meal` acepta `country` y **reconstruye
candidatos por dentro** (rama lácteo-dulce, `P1-CLOSER-SWEET-DAIRY`). Pasarle el país sólo al pool
deja media fuga abierta, así que el guard exige el kwarg en las DOS funciones.

Guard AST: resuelve los alias reales de los imports perezosos dentro de `swap_meal` y exige
`country=` en cada llamada. No mira texto (un comentario no puede satisfacerlo) ni se ancla a la
grafía de los alias (`_pc_pool`, `_safe_pc`, `_close_pc`…), que cambian sin que el defecto vuelva.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_AGENT = _BACKEND / "agent.py"

_HELPERS = ("_safe_high_density_proteins", "_close_protein_gap_for_meal")


def _swap_meal_node() -> ast.FunctionDef:
    tree = ast.parse(_AGENT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "swap_meal":
            return node
    raise AssertionError("`swap_meal` no existe en agent.py — el guard perdió su sujeto")


def _alias_por_helper(fn: ast.FunctionDef) -> dict:
    """{helper canónico: {nombres locales con que se invoca dentro de swap_meal}}"""
    out = {h: set() for h in _HELPERS}
    for node in ast.walk(fn):
        if isinstance(node, ast.ImportFrom) and node.module == "graph_orchestrator":
            for nm in node.names:
                if nm.name in out:
                    out[nm.name].add(nm.asname or nm.name)
    return out


def test_swap_meal_deriva_el_pais_una_sola_vez():
    """El guard asume que hay una variable de país ya derivada; si desaparece, avisa."""
    fn = _swap_meal_node()
    nombres = {t.id for node in ast.walk(fn) if isinstance(node, ast.Assign)
               for t in node.targets if isinstance(t, ast.Name)}
    assert "_swap_country" in nombres, (
        "`swap_meal` ya no deriva `_swap_country`: los call sites de abajo no tendrían de dónde "
        "sacar el país y este guard estaría pidiendo algo imposible."
    )


def test_los_dos_helpers_aceptan_country():
    """Sin el parámetro real, pasar `country=` sería un TypeError y el guard AST, decorativo."""
    import graph_orchestrator as go

    for h in _HELPERS:
        params = inspect.signature(getattr(go, h)).parameters
        assert "country" in params, f"`{h}` ya no acepta `country=` — el arreglo nació inerte"


def test_los_seis_call_sites_del_closer_pasan_el_pais():
    fn = _swap_meal_node()
    alias = _alias_por_helper(fn)
    for h in _HELPERS:
        assert alias[h], (
            f"`swap_meal` ya no importa `{h}` de graph_orchestrator — si la superficie se movió, "
            "mueve el guard con ella en vez de dejarlo verde por vacío."
        )

    todos = {a: h for h, s in alias.items() for a in s}
    llamadas = [n for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in todos]
    assert len(llamadas) >= 6, (
        f"se esperaban al menos los 6 call sites medidos y se ven {len(llamadas)}: si se borró "
        "alguno, revisa que no quedara una ruta sin país en vez de bajar el número."
    )
    sin_pais = [(todos[c.func.id], c.lineno) for c in llamadas
                if not any(kw.arg == "country" for kw in c.keywords)]
    assert not sin_pais, (
        f"call sites del closer de proteína sin `country=` en swap_meal: {sin_pais}. El usuario de "
        "España vuelve a elegir entre 36 candidatos dominicanos en vez de sus 43."
    )
