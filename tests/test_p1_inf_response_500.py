"""[P1-INF-RESPONSE-500 · 2026-07-04] Hotfix del incidente vivo 2026-07-04 21:57-22:07.

Cadena del incidente (3 planes generados en loop para el mismo usuario en ~10 min):
  1. `compute_shopping_coherence_divergences` reporta `delta_pct=float("inf")` por
     CONTRATO cuando `expected == 0 and actual > 0` (divergencia "infinita").
  2. El builder de `_swap_coherence_warnings.summary` (P1-SWAP-COHERENCE-ESCALATE)
     copiaba ese inf tal cual al plan_result.
  3. El INSERT sobrevivía gracias a P2-PERSIST-NAN-GUARD (DB limpia)… pero la
     RESPONSE de `/analyze` (Starlette json.dumps con allow_nan=False) tiraba
     `ValueError: Out of range float values are not JSON compliant: inf` → 500
     DESPUÉS de persistir el plan y encolar 7 chunks; y el payload del SSE llegaba
     imparseable → el frontend caía al endpoint sync → 500 → reintento → OTRA
     generación completa. Loop de cuota.

Fix en dos capas:
  A. FUENTE: el builder del summary coacciona delta_pct no-finito → None.
  B. DEFENSA: `/analyze` retorna `_sanitize_floats_for_json(result)` (mismo helper
     P3-NAN-INF-SANITIZE que ya protegía /recalculate-shopping-list).
"""
import math
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_GO = _read("graph_orchestrator.py")
_PL = _read(os.path.join("routers", "plans.py"))


# ---------------------------------------------------------------------------
# A · fuente: el summary jamás lleva inf
# ---------------------------------------------------------------------------

def test_summary_builder_coerces_nonfinite_delta_pct():
    i = _GO.index('plan_result["_swap_coherence_warnings"] = {')
    block = _GO[max(0, i - 2500):i]
    assert "_dp_finite" in block, "el builder debe chequear finitud de delta_pct"
    assert 'round(_dp_raw, 3) if _dp_finite else None' in block, \
        "delta_pct no-finito debe coaccionarse a None (JSON-compliant)"
    # el chequeo cubre NaN (x != x) e Inf (abs == inf) sin depender de import math.
    assert '_dp_raw == _dp_raw and abs(_dp_raw) != float("inf")' in block


def test_source_contract_inf_still_documented_in_detector():
    """El DETECTOR conserva su contrato (inf interno para ordenar/bloquear) — el fix
    es en el payload user-facing, no en la semántica del guard."""
    sc = _read("shopping_calculator.py")
    assert '"delta_pct": float("inf")' in sc, \
        "el contrato interno del detector no debe cambiar (ordenamiento/bloqueo lo usan)"


# ---------------------------------------------------------------------------
# B · defensa: /analyze retorna sanitizado
# ---------------------------------------------------------------------------

def test_analyze_sync_returns_sanitized():
    i = _PL.index('@router.post("/analyze")')
    j = _PL.index('@router.post("/analyze/stream")')
    body = _PL[i:j]
    assert "return _sanitize_floats_for_json(result)" in body, \
        "/analyze debe sanear NaN/Inf antes de serializar (Starlette allow_nan=False)"
    assert "P1-INF-RESPONSE-500" in body


def test_sanitizer_helper_kills_inf():
    import importlib.util
    # helper puro — testeable sin levantar el router completo: lo replicamos del
    # contrato (idempotente, inf/nan → None) usando el sanitizador real importado.
    import routers.plans as rp
    out = rp._sanitize_floats_for_json({
        "a": float("inf"), "b": float("-inf"), "c": float("nan"),
        "d": [1.5, float("inf")], "e": "txt",
    })
    assert out["a"] is None and out["b"] is None and out["c"] is None
    assert out["d"] == [1.5, None] and out["e"] == "txt"
    import json
    json.dumps(out, allow_nan=False)  # no debe tirar
