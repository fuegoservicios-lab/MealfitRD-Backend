"""[P1-ASSEMBLE-PASS-MAP · 2026-07-25] Saber QUÉ tramo del ensamblador cuesta, antes de rescopearlo.

`assemble_plan_node` es el tramo caro del pipeline y un rechazo lo re-ejecuta **entero**: ése es el
motivo real de que los chunks expiren a los 600 s, no el LLM. Medido en el chunk del 17-jul:

    planificador   20 s
    día-gen        39 s      ← el LLM es ~10% del tiempo
    self-critique  41 s
    ENSAMBLADOR   222 s
    ❌ rechazo por variedad → ENSAMBLADOR otra vez, desde cero → timeout

Y en 30 días de `pipeline_metrics`: 400 pasadas, mediana **83 s**, máximo **447 s**.

## Por qué instrumentar antes de refactorizar

La única descomposición que existía salía de restar marcas de tiempo entre líneas de log que
*casualmente* logean — eso atribuye a un pase todo lo que ocurre entre dos mensajes, incluidos los
pases mudos que hay en medio. Yo mismo escribí "COHERENCE-FINALIZE: 98 s" a partir de esa resta;
no es una medición de esa función, es el hueco entre dos logs.

Rescopear un nodo de ~1900 líneas sobre una atribución así es adivinar. Y el aviso ya estaba
escrito en el propio código (P1-ENGINE-RESCOPE-POST-REGEN, 2026-07-10): *"no se fuerza el refactor
sin esto — muchas pasadas son legítimamente cross-día: shopping aggregation, cuota cross-día de
proteína, fruit-dedup"*.

Este pase no cambia el plan: sólo anota `(etiqueta, tiempo)` en los bordes de los tramos caros y
emite los deltas con la métrica que ya existía.
"""
import ast
from pathlib import Path

import pytest

import graph_orchestrator as go


_SRC = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
_TREE = ast.parse(_SRC)


def _assemble_fn():
    for n in ast.walk(_TREE):
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name == "assemble_plan_node":
            return n
    raise AssertionError("assemble_plan_node no encontrado")


def _ck_labels() -> list[str]:
    return [c.args[0].value
            for c in ast.walk(_assemble_fn())
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "_ck"
            and c.args and isinstance(c.args[0], ast.Constant)]


# ───────────── 1. el instrumento cubre los tramos caros ─────────────

def test_marca_los_bordes_de_los_pases_caros():
    labels = set(_ck_labels())
    for esperado in ("inicio", "pre_macro_engine", "pre_micro_closer", "pre_shopping_list",
                     "pre_coherence_guard", "fin"):
        assert esperado in labels, f"falta el checkpoint {esperado!r}: {sorted(labels)}"


def test_hay_suficientes_tramos_para_discriminar():
    """Con 2-3 marcas el mapa no distingue nada; el objetivo es señalar UN tramo culpable."""
    assert len(_ck_labels()) >= 8


def test_arranca_y_cierra():
    labels = _ck_labels()
    assert labels[0] == "inicio" if labels else False
    assert "fin" in labels


# ───────────── 2. es un INSTRUMENTO: no puede cambiar el plan ─────────────

def test_ck_solo_anota():
    """Si `_ck` hiciera algo más que append, dejaría de ser gratis y podría alterar el resultado."""
    fn = _assemble_fn()
    ck_def = next((n for n in ast.walk(fn)
                   if isinstance(n, ast.FunctionDef) and n.name == "_ck"), None)
    assert ck_def is not None, "_ck debe estar definida dentro de assemble_plan_node"
    llamadas = {n.func.attr for n in ast.walk(ck_def)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert llamadas <= {"append", "time"}, f"_ck hace algo más que anotar: {llamadas}"


def test_ck_es_fail_safe():
    """Un fallo del instrumento jamás puede tumbar el ensamblado de un plan ya generado."""
    fn = _assemble_fn()
    ck_def = next(n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef) and n.name == "_ck")
    assert any(isinstance(n, ast.Try) for n in ast.walk(ck_def))


def test_el_emisor_tambien_es_fail_safe():
    i = _SRC.index("P1-ASSEMBLE-PASS-MAP] assemble=")
    bloque = _SRC[i - 900:i + 400]
    assert "except Exception" in bloque


# ───────────── 3. el mapa llega a la telemetría ─────────────

def test_el_mapa_viaja_en_la_metrica():
    assert '"pass_map_s": _tramos or None' in _SRC, \
        "sin esto el mapa sólo vive en el log y no se puede agregar sobre muchas corridas"


def test_convive_con_la_metrica_de_reentry():
    """El mapa y el flag de re-entry se leen JUNTOS: 'qué tramo cuesta' sólo decide el rescope si
    se sabe además cuántos días tocó el regen."""
    i = _SRC.index('"pass_map_s"')
    bloque = _SRC[i - 700:i + 200]
    assert "is_marker_regen_reentry" in bloque
    assert "marker_regen_touched_days" in bloque
