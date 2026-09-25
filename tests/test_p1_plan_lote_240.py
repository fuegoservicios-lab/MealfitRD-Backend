# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-240 · 2026-09-25] Aire en el god-file: la lógica nueva de 227-239 y cuatro tablas de datos fuera del grafo.

El gate de 227-230 falló por el tope de `graph_orchestrator.py` (52 397 > 52 240; «extrae, no subas el tope»). Con los
lotes 231-239 el grafo iba a 52 599. La lógica nueva vive en módulos propios (`revisor_no_defectos`, `critico_no_agudo`,
`rechazos`, `emergencia_segura`, `relleno_listo`, `desayuno_por_alergia`) y cuatro tablas de datos puros se movieron
verbatim (`notas_clinicas_datos`, `tecnicas_basicos`, `fallback_pools`). En el grafo quedan los knobs (registro), los
guards cableados, los tooltip-anchors y el re-export con el mismo nombre.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402

_MODULOS = ("revisor_no_defectos", "critico_no_agudo", "rechazos", "emergencia_segura", "relleno_listo",
            "desayuno_por_alergia", "restricciones_finales", "yemas_colesterol")
_DATOS = {"notas_clinicas_datos": ("_PREGNANCY_SAFETY_CLAUSES", "_CONDITION_SAFETY_CLAUSES"),
          "tecnicas_basicos": ("_STAPLE_TECHNIQUE_CANONICAL",),
          "fallback_pools": ("_FALLBACK_ALLERGEN_KEYWORDS",)}


def test_el_grafo_queda_bajo_el_tope():
    n = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8").count("\n") + 1
    assert n <= 52_240, n


def test_los_datos_son_el_mismo_objeto_que_ve_el_grafo():
    import importlib
    for mod, nombres in _DATOS.items():
        m = importlib.import_module(mod)
        for nombre in nombres:
            assert getattr(go, nombre) is getattr(m, nombre), (mod, nombre)


def test_los_modulos_de_datos_son_solo_literales():
    for mod in ("notas_clinicas_datos", "tecnicas_basicos"):
        arbol = ast.parse((_BACKEND / f"{mod}.py").read_text(encoding="utf-8"))
        for n in arbol.body:
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
                continue   # docstring
            assert isinstance(n, ast.Assign), (mod, ast.dump(n)[:80])
            ast.literal_eval(n.value)


def test_ningun_modulo_importa_el_grafo_al_cargar():
    """El grafo los importa MIENTRAS se carga: un `import graph_orchestrator` de nivel superior sería circular."""
    for mod in _MODULOS:
        arbol = ast.parse((_BACKEND / f"{mod}.py").read_text(encoding="utf-8"))
        for n in arbol.body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                nombres = [a.name for a in n.names] + [getattr(n, "module", "") or ""]
                assert "graph_orchestrator" not in nombres, mod


def test_el_grafo_reexporta_con_el_mismo_nombre():
    for nombre in ("_downgrade_reviewer_non_issues", "_critical_is_non_acute", "_dislike_declarations",
                   "_scan_dislike_violations", "_fallback_template_violations", "_gm_line_violates",
                   "_gm_ready_carb_for", "_night_rice_sub_for"):
        assert callable(getattr(go, nombre)), nombre
    for knob in ("REVIEWER_NON_ISSUES_ADVISORY", "NON_ACUTE_CRITICAL_SOFT_REJECT", "DISLIKE_HARD_GUARD",
                 "FALLBACK_SSOT_SCAN"):
        assert getattr(go, knob) is True, knob


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 240
