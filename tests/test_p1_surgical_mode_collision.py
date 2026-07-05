"""[P1-SURGICAL-MODE-COLLISION · 2026-07-05] Unión de modos en el nodo quirúrgico.

Caso vivo corr=200c69f3: el router atribuyó el rechazo al Día [1] (proteína repetida) pero el
nodo corrigió el Día [3] — un `_critique_unresolved` STALE secuestraba la pasada (el marker
mode tenía prioridad absoluta: `if not marker_day_nums and not review_passed`). La repetición
del Día 1 sobrevivió a la re-review, y solo un re-enrutamiento fortuito (el flag
`_surgical_reject_attempted` nunca se seteó en marker mode) dio la segunda pasada que cerró
el ciclo (aprobado, primera vez con proteína-repetida).

Fix: con review_passed=False los dos conjuntos se corrigen en la MISMA pasada
(`marker_day_nums = markers ∪ reject_days`), y cada día recibe su prompt apto:
issues de reject (con regla de pool permisiva) SOLO si ese día los tiene; los días-marker
conservan su issue del critique y la precedencia inviolable original.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_union_not_priority():
    i = _GO.index("[P1-SURGICAL-MODE-COLLISION · 2026-07-05] UNIÓN de modos")
    win = _GO[i:i + 1600]
    assert 'sorted(set(marker_day_nums) | set(_srj["days"]))' in win, \
        "los días-marker y los días-reject se corrigen en la MISMA pasada"
    # la derivación reject ya NO está condicionada a 'not marker_day_nums'.
    assert "if not state.get(\"review_passed\", False) and SURGICAL_REJECT_RETRY_ENABLED:" in win


def test_reject_prompt_only_for_reject_days():
    assert _GO.count("if _reject_mode and _reject_issues.get(day_num):") >= 2, \
        "el issue de reject Y la precedencia permisiva aplican SOLO al día que tiene issues " \
        "de reject; los días-marker de la unión conservan su prompt original"


def test_marker_anchored_in_source():
    assert "P1-SURGICAL-MODE-COLLISION" in _GO
