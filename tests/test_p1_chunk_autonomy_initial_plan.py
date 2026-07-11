"""[P1-CHUNK-AUTONOMY · 2026-07-10] Los chunks `initial_plan` (días 4-30 de un plan RECIÉN
generado) no se pausan por nevera vacía ni por reservas 0/N — la nevera pre-compra es el
estado NORMAL del día 4, no un error.

Evidencia (dry-run 2026-07-10 + journalctl 7 días): TODAS las reservas de producción =
0/N (0/62…0/88) con `user_inventory` VACÍO al momento del chunk → RECONCILE-EXHAUSTED →
`pending_user_action` → un plan de 30 días jamás llegaba solo al día 30. El SSE ya
saltaba el guard (MEALFIT_INITIAL_CHUNK_PANTRY_GUARD=False, P0-2/RENEWAL-PANTRY-IGNORE);
el worker no — esta es la paridad. `rolling_refill`/`catchup` CONSERVAN pausa y gate
(a mitad de plan sí prometemos cocinar con lo que hay).

Validador funcional E2E: tests/test_chunked_7days_thursday_e2e.py (chunk 2 initial_plan
debe llegar a 'completed' con nevera de fixture no-matching).

tooltip-anchor: P1-CHUNK-AUTONOMY
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_CRON = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")

_KNOB = 'os.environ' if False else '_env_bool("MEALFIT_INITIAL_CHUNK_PANTRY_AUTONOMY", True)'


def test_knob_read_at_three_callsites():
    # pause pre-pipeline + gate de reservas (branch normal) + branch de excepción
    n = _CRON.count('_env_bool("MEALFIT_INITIAL_CHUNK_PANTRY_AUTONOMY", True)')
    assert n >= 3, (
        f"P1-CHUNK-AUTONOMY: esperaba el knob en ≥3 callsites (pause + gate + except), hay {n}. "
        "Sin los 3, algún path sigue pausando initial_plan con nevera pre-compra."
    )


def test_pause_pre_pipeline_is_kind_aware():
    i = _CRON.find("[P1-CHUNK-AUTONOMY] Chunk {week_number}")
    assert i > 0, "el skip del pause pre-pipeline desapareció"
    blk = _CRON[i - 2000: i + 2500]
    assert '_should_pause_for_empty_pantry' in blk, "el skip vive DENTRO del if should_pause"
    assert 'chunk_kind == "initial_plan"' in blk
    # el else conserva la pausa para refill/catchup
    assert "_pause_chunk_for_pantry_refresh(task_id, user_id, week_number, fresh_inventory)" in blk
    assert "[P1-1/PANTRY-EMPTY]" in blk, "la pausa legacy (refill/catchup) debe seguir viva"


def test_reservation_gate_best_effort_before_partial():
    i_be = _CRON.find("reservation_status = 'best_effort'")
    i_partial = _CRON.find("Marcando reservation_status='partial'")
    assert i_be > 0, "el branch best_effort del gate desapareció"
    assert i_partial > 0, "el branch partial (refill/catchup) debe seguir vivo"
    assert i_be < i_partial, (
        "el elif initial_plan debe evaluarse ANTES del else partial — invertirlo re-activa "
        "el RECONCILE-EXHAUSTED para initial_plan"
    )
    blk = _CRON[i_be - 1600: i_be]
    assert 'elif chunk_kind == "initial_plan"' in blk


def test_refill_catchup_keep_full_gate():
    # El contrato de refill/catchup NO cambia: reconcile + pause + return siguen presentes.
    assert "_reconcile_chunk_reservations(user_id, str(task_id), new_days)" in _CRON
    assert "_handle_reservation_reconciliation_exhausted(" in _CRON
    i = _CRON.find("[P1-CHUNKS-2/RECONCILE-EXHAUSTED] Pausando chunk")
    assert i > 0
    assert "return" in _CRON[i: i + 1200], "el return post-pausa protege contra overbooking en refill"


def test_exception_path_is_kind_aware():
    i = _CRON.find("Reserva lanzó excepción en chunk initial_plan")
    assert i > 0, "el branch best-effort del except desapareció"
    blk = _CRON[i - 1200: i + 2600]
    assert "except Exception as reserve_err" in blk
    assert "reservation_status = 'best_effort'" in blk


def test_marker_anchored_in_source():
    # [P1-REBALANCE-LINE-CLAMP · 2026-07-10] durable: anclar en el CÓDIGO, no en el
    # _LAST_KNOWN_PFIX vigente (pinnear el marker actual rota con cada bump posterior).
    assert _CRON.count("P1-CHUNK-AUTONOMY") >= 3, "los anchors del skip desaparecieron de cron_tasks"
