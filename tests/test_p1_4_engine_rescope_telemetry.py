"""[P1-ENGINE-RESCOPE-POST-REGEN · 2026-07-10] Forensic corr=d57ffe04 (2026-07-10): el engine
determinista de `assemble_plan_node` (~18 pases, ~2360 líneas) corre COMPLETO (todos los días) tanto en
el primer pase como tras `surgical_marker_regen` — que solo reescribe 1-2 días. La corrida medida gastó
196s de 358s totales (55%) en DOS pasadas completas del engine para regenerar apenas 1 día. Rescopear el
engine a día-local es un refactor de alto riesgo sobre un archivo de ~35k líneas que genera planes
CLÍNICOS (muchas pasadas son legítimamente cross-día: shopping aggregation, cuota cross-día de proteína,
fruit-dedup a nivel plan) — no se fuerza sin evidencia agregada de flota primero (mismo principio
"evidence-first" ya aplicado a P1-SOLVER-SATURATION-RELIEF). Este batch cierra la MITAD segura: hace
observable, por plan, si el pase de `assemble_plan` fue un re-entry post marker-regen y cuántos días
tocó el regen vs el total — sin esto, medir el % de latencia atribuible a re-runs innecesarios requería
reprocesar logs a mano (como en este forensic). Sienta la base de datos agregada para decidir si/cómo
rescopear con confianza.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO_SRC = f.read()


def test_marker_present():
    assert "P1-ENGINE-RESCOPE-POST-REGEN" in _GO_SRC


def test_surgical_marker_regen_persists_touched_days_reject_mode():
    i = _GO_SRC.index('if _reject_mode:\n        state_update = {')
    window = _GO_SRC[i:i + 800]
    assert "_marker_regen_touched_days" in window, \
        "el modo rechazo debe persistir qué días tocó el regen quirúrgico"


def test_surgical_marker_regen_persists_touched_days_approved_mode():
    i = _GO_SRC.index('if _reject_mode:\n        state_update = {')
    j = _GO_SRC.index('\n    state_update = {', i)  # el bloque approved-mode, tras el early-return
    window = _GO_SRC[j:j + 400]
    assert "_marker_regen_touched_days" in window, \
        "el modo aprobado-con-markers debe persistir qué días tocó el regen quirúrgico"


def test_assemble_plan_metric_tags_reentry_and_touched_days():
    i = _GO_SRC.index('"node": "assemble_plan",')
    window = _GO_SRC[i:i + 700]
    assert "is_marker_regen_reentry" in window
    assert "marker_regen_touched_days" in window
    assert "total_days" in window
