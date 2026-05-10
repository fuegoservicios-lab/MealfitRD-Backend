"""[P2-HIST-AUDIT-4 · 2026-05-09] Tests del helper SSOT
`_apply_coherence_history_cap` + knob
`MEALFIT_COHERENCE_BLOCK_HISTORY_CAP`.

Bug original (audit historial 2026-05-08):
    El cap del `_shopping_coherence_block_history` estaba hardcoded
    `if len(new_history) > 20` en DOS sites del orchestrator
    (assemble_plan_node, _recompute_aggregates_after_swap). Si un
    plan superaba 20 entries, las viejas se descartaban silenciosa-
    mente — el chip "X ajustes" subnumeraba sin señal al operador.

Fix:
    - Helper SSOT `_apply_coherence_history_cap(prior, entry,
      plan_id_hint)` que aplica el cap leído del knob
      `MEALFIT_COHERENCE_BLOCK_HISTORY_CAP` (default 20).
    - Telemetría `logger.warning` cuando trunca, con plan_id_hint y
      truncated_count → SRE puede detectar cuándo el cap empieza a
      doler vía grep agregado.
    - Guard cap >= 1 (cap inválido fallback al default 20).
    - Drift detection: ambos call sites legacy migrados al helper.

Cobertura:
    - Helper preserva entries cuando len <= cap (no-op funcional).
    - Helper trunca cuando len > cap, mantiene la entry recién
      insertada (no descarta la nueva, descarta las viejas).
    - Knob bumps el cap (env var override).
    - Cap inválido (0, negativo) cae al default 20.
    - Telemetría: log warning emitido al truncar.
    - prior_history no-list normaliza a [] (defensa).
    - Drift: ambos call sites NO usan `> 20` literal hardcoded.
"""
from __future__ import annotations

import inspect
import logging
import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GRAPH_ORCH_PATH = _BACKEND_ROOT / "graph_orchestrator.py"


# ---------------------------------------------------------------------------
# 1. Helper existe y tiene marker
# ---------------------------------------------------------------------------
def test_helper_exists():
    from graph_orchestrator import _apply_coherence_history_cap
    assert callable(_apply_coherence_history_cap)


def test_helper_marker_present():
    from graph_orchestrator import _apply_coherence_history_cap
    src = inspect.getsource(_apply_coherence_history_cap)
    # El marker puede estar en docstring O en el comment del bloque
    # arriba — buscamos en el archivo completo cerca de la def.
    text = _GRAPH_ORCH_PATH.read_text(encoding="utf-8")
    assert "P2-HIST-AUDIT-4" in text, (
        "El marker debe estar en graph_orchestrator.py (helper o "
        "comments cercanos)."
    )


# ---------------------------------------------------------------------------
# 2. Comportamiento: preservar / truncar
# ---------------------------------------------------------------------------
def test_helper_appends_entry_when_below_cap():
    """Si prior_history < cap, no se trunca: la nueva entry queda al
    final de la lista."""
    from graph_orchestrator import _apply_coherence_history_cap
    prior = [{"action_taken": "reject_minor"}] * 5
    entry = {"action_taken": "degrade", "marker": "new"}
    result = _apply_coherence_history_cap(prior, entry)
    assert len(result) == 6
    assert result[-1] == entry
    # Las prior intactas.
    assert result[:5] == prior


def test_helper_truncates_when_above_cap_preserves_new_entry():
    """Si prior_history > cap-1, truncamos pero PRESERVAMOS la nueva
    entry (descartamos viejas, no la recién insertada)."""
    from graph_orchestrator import _apply_coherence_history_cap
    # Default cap = 20. Pasamos 25 prior + 1 nueva = 26 → trunca a 20.
    prior = [{"action_taken": f"old_{i}"} for i in range(25)]
    entry = {"action_taken": "new_entry", "marker": "fresh"}
    result = _apply_coherence_history_cap(prior, entry)
    assert len(result) == 20
    # La nueva entry SIEMPRE está al final.
    assert result[-1] == entry
    # Las primeras 5 prior se descartaron (índices 0-4 de 25).
    # Las que sobrevivieron: prior[6:25] + entry = 19 + 1 = 20.
    assert result[0] == prior[6]


def test_helper_normalizes_non_list_prior_to_empty():
    """Defensa contra prior_history=None / dict / str → tratamos como
    []. Sin esto, un append() sobre None crashea."""
    from graph_orchestrator import _apply_coherence_history_cap
    entry = {"action_taken": "degrade"}
    for bad in [None, {"x": 1}, "not-a-list", 42]:
        result = _apply_coherence_history_cap(bad, entry)
        assert result == [entry], (
            f"prior={bad!r} debió normalizarse a [] y devolver [entry]; "
            f"got {result!r}"
        )


# ---------------------------------------------------------------------------
# 3. Knob MEALFIT_COHERENCE_BLOCK_HISTORY_CAP
# ---------------------------------------------------------------------------
def test_knob_bumps_cap_via_env():
    """Setear el knob a 50 cambia el cap → el helper preserva 50
    entries en lugar de 20."""
    from graph_orchestrator import _apply_coherence_history_cap
    prior = [{"action_taken": f"old_{i}"} for i in range(60)]
    entry = {"action_taken": "fresh"}
    with patch.dict(os.environ, {"MEALFIT_COHERENCE_BLOCK_HISTORY_CAP": "50"}):
        result = _apply_coherence_history_cap(prior, entry)
    assert len(result) == 50
    assert result[-1] == entry


def test_knob_invalid_value_falls_back_to_default(caplog):
    """cap=0 o negativo es patológico → fallback al default 20 +
    log warning."""
    from graph_orchestrator import _apply_coherence_history_cap
    prior = [{"action_taken": f"old_{i}"} for i in range(25)]
    entry = {"action_taken": "fresh"}
    with patch.dict(os.environ, {"MEALFIT_COHERENCE_BLOCK_HISTORY_CAP": "0"}):
        with caplog.at_level(logging.WARNING, logger="graph_orchestrator"):
            result = _apply_coherence_history_cap(prior, entry)
    # Default 20 aplicado.
    assert len(result) == 20
    # Warning emitido.
    assert any(
        "P2-HIST-AUDIT-4" in record.message and "inválido" in record.message
        for record in caplog.records
    ), (
        f"Esperaba log warning de fallback. Got: "
        f"{[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# 4. Telemetría: log warning cuando trunca
# ---------------------------------------------------------------------------
def test_helper_logs_warning_when_truncating(caplog):
    from graph_orchestrator import _apply_coherence_history_cap
    prior = [{"action_taken": f"old_{i}"} for i in range(22)]
    entry = {"action_taken": "fresh"}
    plan_id = "aaaa-bbbb-cccc"
    with caplog.at_level(logging.WARNING, logger="graph_orchestrator"):
        _apply_coherence_history_cap(prior, entry, plan_id_hint=plan_id)
    matching = [
        r for r in caplog.records
        if "COH-HISTORY-TRUNCATED" in r.message
    ]
    assert matching, (
        f"Esperaba log warning con `COH-HISTORY-TRUNCATED`. Got: "
        f"{[r.message for r in caplog.records]}"
    )
    # plan_id_hint en el log para diagnóstico SRE.
    assert plan_id in matching[0].message


def test_helper_does_not_log_when_not_truncating(caplog):
    """Si len(new_history) <= cap, NO emitimos log (evitar spam)."""
    from graph_orchestrator import _apply_coherence_history_cap
    prior = [{"action_taken": f"old_{i}"} for i in range(5)]
    entry = {"action_taken": "fresh"}
    with caplog.at_level(logging.WARNING, logger="graph_orchestrator"):
        _apply_coherence_history_cap(prior, entry)
    truncate_logs = [
        r for r in caplog.records if "COH-HISTORY-TRUNCATED" in r.message
    ]
    assert not truncate_logs, (
        f"NO esperaba log de truncate. Got: "
        f"{[r.message for r in truncate_logs]}"
    )


# ---------------------------------------------------------------------------
# 5. Drift detection: call sites legacy migrados al helper
# ---------------------------------------------------------------------------
def test_no_hardcoded_cap_20_in_orchestrator():
    """Ningún site del archivo debe tener `if len(...) > 20` con
    referencia a coherence_block_history. Si alguien revierte al
    literal hardcoded, el knob queda inactivo y la regresión vuelve.
    """
    text = _GRAPH_ORCH_PATH.read_text(encoding="utf-8")
    # Patrón legacy: `if len(new_history) > 20:` o variante.
    legacy_pattern = re.compile(
        r"if\s+len\(\s*new_history\s*\)\s*>\s*20\s*:",
        re.IGNORECASE,
    )
    matches = legacy_pattern.findall(text)
    assert not matches, (
        f"Encontradas {len(matches)} referencias al cap hardcoded "
        f"`if len(new_history) > 20`. Migrar a "
        f"`_apply_coherence_history_cap()` para usar el knob."
    )


def test_two_call_sites_use_helper():
    """Ambos sites históricos (assemble_plan_node + post_swap_revalidation)
    deben llamar al helper SSOT.
    """
    text = _GRAPH_ORCH_PATH.read_text(encoding="utf-8")
    helper_calls = re.findall(
        r"_apply_coherence_history_cap\s*\(",
        text,
    )
    assert len(helper_calls) >= 2, (
        f"Esperaba >= 2 llamadas al helper (assemble_plan_node + "
        f"post_swap_revalidation), got {len(helper_calls)}. ¿Algún "
        f"call site quedó sin migrar?"
    )


def test_helper_signature_includes_plan_id_hint():
    """El helper acepta `plan_id_hint` keyword para incluirlo en el
    log warning. Sin esto, el log es demasiado vago para ser útil."""
    from graph_orchestrator import _apply_coherence_history_cap
    sig = inspect.signature(_apply_coherence_history_cap)
    assert "plan_id_hint" in sig.parameters
    assert sig.parameters["plan_id_hint"].kind == inspect.Parameter.KEYWORD_ONLY


# ---------------------------------------------------------------------------
# 6. Knob registrado en _KNOBS_REGISTRY (tras primera resolución)
# ---------------------------------------------------------------------------
def test_knob_registered_after_first_call():
    """`_env_int` registra cada knob en `_KNOBS_REGISTRY` la primera
    vez que se invoca. Después de un call al helper, el knob debe
    estar en el snapshot."""
    from graph_orchestrator import (
        _coherence_block_history_cap,
        get_knobs_registry_snapshot,
    )
    _coherence_block_history_cap()  # fuerza el registro
    snapshot = get_knobs_registry_snapshot()
    assert "MEALFIT_COHERENCE_BLOCK_HISTORY_CAP" in snapshot, (
        "Knob no aparece en el registry tras invocar el helper. "
        "¿Error en _env_int o el helper no usa el wrapper?"
    )
