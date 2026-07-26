"""[P1-CONFTEST-EAGER-GO · 2026-07-26] Un stub instalado primero secuestra el módulo real.

## El patrón que lo causa

Varios archivos de test instalan stubs así, EN TIEMPO DE IMPORT:

    if "X" not in sys.modules:
        sys.modules["X"] = <stub parcial>

Ese guard no distingue *"no está cargado"* de *"está cargado el real"*. Si el archivo que lo
ejecuta se colecta ANTES de que nadie importe el módulo de verdad, deja un stub parcial
instalado para **toda la sesión**, y cualquier test posterior revienta con:

    AttributeError: module 'graph_orchestrator' has no attribute '<lo que sea>'

Este `conftest` YA documenta y cura exactamente esta trampa para `langgraph` (P0-5) y para
`langchain_openai`. Faltaba el módulo del repo que más se stubea.

## Lo medido (2026-07-26)

Glob `polish/display/finalize`, 32 archivos:

    sin el eager-import:  la coleccion se INTERRUMPE — 2 errores, no corre nada
    con el eager-import:  21 failed, 255 passed, 0 errores de coleccion

Y sobre una porcion grande e independiente (`tests/test_p1_*.py`, 6557 tests) el resultado es
**identico con y sin el arreglo**: 68 failed / 6489 passed. Es decir: no-op en el caso comun
—alguien importa el modulo real a tiempo— y la diferencia entre "no corre nada" y "corren 255"
cuando el instalador de stubs se colecta primero. Quita una mina sin mover nada mas.

tooltip-anchor: P1-CONFTEST-EAGER-GO
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def test_graph_orchestrator_esta_cargado_antes_de_los_tests():
    """El conftest lo pre-importa, así que para cuando corre cualquier test ya está."""
    assert "graph_orchestrator" in sys.modules


def test_y_es_el_modulo_REAL_no_un_stub():
    """El sintoma exacto: un stub parcial no tiene los simbolos del modulo real."""
    go = sys.modules["graph_orchestrator"]
    assert isinstance(go, types.ModuleType)
    for simbolo in ("_closer_protein_step_text", "_name_suggests_blended",
                    "get_knobs_registry_snapshot"):
        assert hasattr(go, simbolo), (
            f"'graph_orchestrator' no tiene {simbolo!r} → hay un stub instalado por otro "
            "archivo de test que se colecto antes"
        )


def test_el_guard_de_los_stubs_es_ahora_no_op():
    """Reproduce el patron culpable: con el modulo real ya cargado, el `if not in sys.modules`
    no instala nada. Ese es todo el mecanismo del fix."""
    antes = sys.modules["graph_orchestrator"]
    if "graph_orchestrator" not in sys.modules:          # el guard de los otros archivos
        sys.modules["graph_orchestrator"] = types.ModuleType("graph_orchestrator")
    assert sys.modules["graph_orchestrator"] is antes, "el guard no debe poder sustituirlo"


# ───────────── ancla de la clase ─────────────

def _conftest_src() -> str:
    return (Path(__file__).resolve().parent / "conftest.py").read_text(encoding="utf-8")


def test_el_conftest_lo_preimporta():
    src = _conftest_src()
    assert "import graph_orchestrator" in src
    assert "P1-CONFTEST-EAGER-GO" in src


def test_el_preimport_es_fail_open():
    """Este conftest NO debe ser quien tumbe la suite: si el import real falla (entorno sin
    deps), se deja pasar con un aviso y cada test se apana como hasta ahora."""
    src = _conftest_src()
    i = src.index("P1-CONFTEST-EAGER-GO")
    bloque = src[i:i + 1400]
    assert "try:" in bloque and "except Exception" in bloque, "debe ser fail-open"
    assert "raise" not in bloque


def test_corre_DESPUES_de_los_otros_eager_imports():
    """`graph_orchestrator` importa langgraph y langchain_openai transitivamente. Si su
    pre-import se colara ANTES de los de P0-5, un entorno sin esas deps instalaria los stubs
    tarde y volveria el ModuleNotFoundError que P0-5 cerro."""
    src = _conftest_src()
    assert src.index("import langgraph") < src.index("P1-CONFTEST-EAGER-GO")
    assert src.index("import langchain_openai") < src.index("P1-CONFTEST-EAGER-GO")


@pytest.mark.parametrize("modulo", ["langgraph", "langchain_openai"])
def test_los_eager_imports_previos_siguen(modulo):
    """No se toco lo que ya funcionaba."""
    assert modulo in sys.modules
