# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-198 · 2026-09-24] Aire en `graph_orchestrator.py` sin cambiar conducta.

El god-file estaba en 52.239 líneas con tope 52.240: el siguiente arreglo del generador no cabía. Las plantillas del plan
de EMERGENCIA (`_FALLBACK_MEAL_POOLS`, genérico de 3 comidas, y `_FALLBACK_MEAL_POOLS_BARIATRIC`, curado de 6) son datos
puros —literales y `frozenset`, sin leer ningún global— y se mudan TAL CUAL a `fallback_pools.py`; el grafo las
re-exporta con el mismo nombre (el mismo objeto), así que quien lee o parchea `go._FALLBACK_MEAL_POOLS` no nota nada.
Se quedan en el grafo los helpers que las usan (detección bariátrica, ratios por slot, `_build_fallback_day`).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_NOMBRES = ("_FALLBACK_MEAL_POOLS", "_FALLBACK_MEAL_POOLS_BARIATRIC")


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _definidos(src: str) -> set:
    return {t.id for n in ast.parse(src).body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}


def test_los_pools_viven_en_su_modulo_y_el_grafo_ya_no_los_define():
    assert set(_NOMBRES) <= _definidos(_src("fallback_pools.py"))
    assert not set(_NOMBRES) & _definidos(_src("graph_orchestrator.py")), "una segunda copia en el grafo divergiría"


def test_el_grafo_reexporta_el_mismo_objeto():
    import fallback_pools as fp
    import graph_orchestrator as go
    for nombre in _NOMBRES:
        assert getattr(go, nombre) is getattr(fp, nombre), nombre


def test_el_modulo_es_datos_puros_sin_imports():
    """Sin imports ni lógica: si algún día necesita leer un knob o el catálogo, ya no es una plantilla (y un import
    desde aquí hacia el grafo sería un ciclo)."""
    arbol = ast.parse(_src("fallback_pools.py"))
    tipos = {type(n).__name__ for n in arbol.body}
    assert tipos <= {"Expr", "Assign"}, tipos


def test_las_plantillas_conservan_su_forma():
    import fallback_pools as fp
    assert set(fp._FALLBACK_MEAL_POOLS) == {"Desayuno", "Almuerzo", "Cena"}
    assert len(fp._FALLBACK_MEAL_POOLS_BARIATRIC) == 6
    for pool in (fp._FALLBACK_MEAL_POOLS, fp._FALLBACK_MEAL_POOLS_BARIATRIC):
        for slot, items in pool.items():
            for nombre, tokens, desc, ingredientes in items:
                assert isinstance(tokens, frozenset) and nombre and desc and ingredientes, (slot, nombre)
            assert items[-1][1] == frozenset(), f"{slot}: la última plantilla debe ser neutral (fail-safe multi-alergia)"


def test_el_dia_de_emergencia_se_sigue_construyendo():
    import graph_orchestrator as go
    nutr = {"target_calories": 2000, "macros": {"protein_g": 130, "carbs_g": 220, "fats_g": 65}}
    dia = go._build_fallback_day(nutr, 1)
    assert len(dia.get("meals") or []) == 3, dia


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 198 and m.group(2) >= "2026-09-24"
