"""[P1-RESCOPE-METRIC-BLIND · 2026-07-25] Una métrica que sólo podía dar False.

`surgical_marker_regen_node` devuelve `_marker_regen_touched_days` (los días que el regen
quirúrgico tocó de verdad) y `assemble_plan_node` lo lee para tagear su métrica de duración como
re-entry. Esa era la evidencia con la que P1-ENGINE-RESCOPE-POST-REGEN (2026-07-10) iba a decidir
si rescopear el engine a día-local — el propio comentario dice *"no se fuerza el refactor sin
esto"*.

**La clave no estaba declarada en `PlanState`**, así que LangGraph la descartaba y assemble leía
siempre `None`. Consultando 30 días de `pipeline_metrics`:

    assemble_plan · primera pasada:  n=400  mediana 83 s  max 447 s
    assemble_plan · RE-ENTRY:        sin datos     ← con re-entries ocurriendo de verdad

Los re-entries son visibles en el journal (17-jul, chunk 2: dos `ENSAMBLADOR` en la misma corrida
antes del timeout). La instrumentación llevaba 15 días midiendo un booleano que no podía ser True.

⚠️ Es el mismo modo de fallo de P1-HARDEN-POOLS-CANARY-GATING (2026-07-24): cohortes sin declarar
hicieron que 80/80 filas de telemetría mintieran. **Segunda vez en dos días.** Por eso este test
no ancla una clave: ancla la CLASE. Un nodo de LangGraph sólo puede propagar lo que `PlanState`
declara; todo lo demás se pierde en silencio y produce datos que parecen medir algo.
"""
import ast
from pathlib import Path

import pytest

import graph_orchestrator as go


_SRC = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
_TREE = ast.parse(_SRC)


def _planstate_fields() -> set[str]:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.ClassDef) and node.name == "PlanState":
            return {s.target.id for s in node.body
                    if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}
    raise AssertionError("PlanState no encontrado en graph_orchestrator.py")


def _node_functions():
    """Funciones-nodo del grafo (convención del repo: sufijo `_node`)."""
    for node in ast.walk(_TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.endswith("_node"):
            yield node


def _state_update_keys(fn) -> set[str]:
    """Claves literales que la función devuelve como actualización de state.

    Cubre `return {...}` y `state_update = {...}` (el patrón de este archivo). Ignora claves
    dinámicas y `**spread`: no se puede afirmar sobre lo que no es literal.
    """
    keys: set[str] = set()

    def _collect(d: ast.Dict):
        for k in d.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.add(k.value)

    for sub in ast.walk(fn):
        if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
            _collect(sub.value)
        elif isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Dict):
            if any(isinstance(t, ast.Name) and "state_update" in t.id for t in sub.targets):
                _collect(sub.value)
    return keys


# ───────────── 1. la clave que estaba ciega ─────────────

def test_marker_regen_touched_days_declarado():
    assert "_marker_regen_touched_days" in _planstate_fields(), (
        "sin esta declaración LangGraph descarta la clave y la métrica de re-entry "
        "de assemble_plan sólo puede dar False")


def test_assemble_sigue_leyendola():
    """Si el consumidor desaparece, la declaración sobra y este test debe avisar."""
    assert 'state.get("_marker_regen_touched_days")' in _SRC


def test_el_productor_sigue_emitiendola():
    assert '"_marker_regen_touched_days": list(marker_day_nums)' in _SRC


# ───────────── 2. la CLASE entera ─────────────

def test_todo_lo_que_un_nodo_devuelve_vive_en_planstate():
    """Invariante de LangGraph: un nodo sólo propaga lo que el schema declara.

    Cualquier clave devuelta y no declarada se pierde SIN ERROR — el síntoma no es una excepción,
    es telemetría que miente o un guard que nunca dispara.
    """
    campos = _planstate_fields()
    huerfanas = []
    for fn in _node_functions():
        for k in _state_update_keys(fn):
            if k not in campos:
                huerfanas.append(f"{fn.name} → {k}")
    assert not huerfanas, (
        "claves devueltas por nodos y NO declaradas en PlanState (LangGraph las descarta "
        "en silencio):\n  " + "\n  ".join(sorted(huerfanas)))


def test_el_escaner_de_verdad_mira_nodos():
    """Guarda contra el test decorativo: si el escáner no encuentra nodos ni claves, el test de
    arriba pasaría vacío y no vigilaría nada."""
    nodos = list(_node_functions())
    assert len(nodos) >= 3, [n.name for n in nodos]
    total = sum(len(_state_update_keys(fn)) for fn in nodos)
    assert total >= 10, f"sólo {total} claves detectadas: el parser dejó de ver los state updates"


def test_planstate_no_esta_vacio():
    assert len(_planstate_fields()) >= 20
