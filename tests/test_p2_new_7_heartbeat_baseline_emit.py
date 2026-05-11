"""[P2-NEW-7 · 2026-05-10] Baseline emit always para heartbeat stats:
añade una INSERT separada bajo node `_chunk_heartbeat_baseline` que se
ejecuta cuando hubo updates (no solo cuando anómalo).

Bug original (audit 2026-05-10):
  El heartbeat thread emitía pipeline_metrics SOLO cuando
  `_is_anomalous and _total > 0` (cron_tasks.py:~17034). SRE no podía
  medir distribución p50/p95 de heartbeat lag — sin baseline, p95 de
  solo-anomalías está fuertemente sesgado y no detectaba degradación
  gradual.

Fix:
  Añadir segundo INSERT bajo node `_chunk_heartbeat_baseline` que se
  ejecuta `if _total > 0` (siempre que hubo updates). Confidence=1.0
  cuando saludable, 0.0 cuando anómalo. Knob
  `MEALFIT_HEARTBEAT_BASELINE_EMIT` (default True) permite drop si el
  volumen se vuelve problemático.

  Node separado preserva dashboards existentes que filtran por
  `_chunk_heartbeat_lag` (siguen viendo solo anomalías).

Cobertura:
  1. La INSERT original con `_chunk_heartbeat_lag` sigue presente
     (no removida, preserva backward compat).
  2. Existe segunda INSERT con `_chunk_heartbeat_baseline`.
  3. La gate de baseline es `if _total > 0` (no `_is_anomalous`).
  4. Knob `MEALFIT_HEARTBEAT_BASELINE_EMIT` se consulta antes del INSERT.
  5. La metadata incluye `is_anomalous` flag para que dashboards puedan
     filtrar.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"


def _read_source() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


def test_anomaly_emit_path_preserved():
    """La INSERT original (`_chunk_heartbeat_lag` gateada por _is_anomalous)
    debe seguir presente — dashboards existentes la consultan."""
    src = _read_source()
    assert '_chunk_heartbeat_lag' in src, (
        "El node `_chunk_heartbeat_lag` (path original P3-3) desapareció — "
        "rompería dashboards que filtran anomalías por ese nombre."
    )


def test_baseline_emit_path_added():
    """Nuevo node `_chunk_heartbeat_baseline` debe estar presente bajo
    una gate `if _total > 0`."""
    src = _read_source()
    assert '_chunk_heartbeat_baseline' in src, (
        "Falta el node `_chunk_heartbeat_baseline` — sin él SRE no tiene "
        "baseline para medir p50/p95 del lag."
    )


def test_baseline_emit_uses_knob_with_default_true():
    """El emit del baseline debe consultar `MEALFIT_HEARTBEAT_BASELINE_EMIT`
    con default True (knob para drop si volumen problemático)."""
    src = _read_source()
    pattern = re.compile(
        r"_(?:knob_)?env_bool\(\s*[\"']MEALFIT_HEARTBEAT_BASELINE_EMIT[\"']\s*,\s*True\s*\)",
        re.DOTALL,
    )
    assert pattern.search(src) is not None, (
        "Knob `MEALFIT_HEARTBEAT_BASELINE_EMIT` no se consulta con default "
        "True para el baseline emit. Sin el knob, no hay vía de rollback en "
        "caso de explosion de volumen."
    )


def test_baseline_metadata_has_is_anomalous_flag():
    """Metadata del baseline debe incluir `is_anomalous` para que
    dashboards puedan diferenciar entre healthy y anomaly."""
    src = _read_source()
    # Buscar la región alrededor de _chunk_heartbeat_baseline y verificar
    # que `is_anomalous` aparece en su contexto cercano (within ~30 lines).
    # Hay 2 ocurrencias del literal `_chunk_heartbeat_baseline` (test name + INSERT)
    # — buscamos la del INSERT yendo desde el final hacia atrás.
    idx = src.rfind("_chunk_heartbeat_baseline")
    assert idx > -1
    # Ventana amplia para captar el metadata jsonb que sigue a la línea del node.
    window = src[max(0, idx - 100):idx + 2500]
    assert "is_anomalous" in window, (
        "Metadata del baseline INSERT no incluye `is_anomalous` flag. Sin él "
        "los consumidores no pueden filtrar healthy vs anomaly al hacer p95."
    )


def test_baseline_gate_is_total_gt_zero_not_anomalous():
    """La gate del baseline emit debe ser `if _total > 0` (no `_is_anomalous`).
    Si fuera anomalous-only, sería duplicado del path original."""
    src = _read_source()
    # Buscar el bloque P2-NEW-7 (marcado con comment textual) y verificar
    # que su gate contiene `_total > 0` antes del INSERT, no _is_anomalous.
    p2_marker = src.find("[P2-NEW-7")
    assert p2_marker > -1, "Marker `[P2-NEW-7` no presente en el bloque."
    window = src[p2_marker:p2_marker + 1500]
    # La estructura esperada: comment block + `if _total > 0:` + try + emit.
    pattern = re.compile(r"if\s+_total\s*>\s*0\s*:")
    assert pattern.search(window) is not None, (
        "Gate `if _total > 0:` no presente en el bloque P2-NEW-7. La "
        "gate `if _is_anomalous` sería redundante (path original)."
    )
