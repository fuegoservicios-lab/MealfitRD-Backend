"""[P1-CHUNK-AUTONOMY · 2026-07-10] Los chunks `initial_plan` (días 4-30 de un plan RECIÉN
generado) no se pausan por nevera vacía ni por reservas 0/N — la nevera pre-compra es el
estado NORMAL del día 4, no un error.

Evidencia (dry-run 2026-07-10 + journalctl 7 días): TODAS las reservas de producción =
0/N (0/62…0/88) con `user_inventory` VACÍO al momento del chunk → RECONCILE-EXHAUSTED →
`pending_user_action` → un plan de 30 días jamás llegaba solo al día 30. El SSE ya
saltaba el guard (MEALFIT_INITIAL_CHUNK_PANTRY_GUARD=False, P0-2/RENEWAL-PANTRY-IGNORE);
el worker no — esta es la paridad.

[P1-PANTRY-GATE-SSOT · 2026-07-26] **La condición se generalizó, la garantía no cambia.**
Los 3 gates ya no preguntan `chunk_kind == "initial_plan"` cada uno por su cuenta: llaman
a `_pantry_gate_waiver_reason`, que responde `"initial_plan_autonomy"` para exactamente los
mismos casos de antes — más `flexible_mode`/`advisory_only`, que el gate de reservas
ignoraba y por eso reabría el lazo (ver ese docstring). `rolling_refill`/`catchup` SIN
waiver CONSERVAN pausa y gate (a mitad de plan sí prometemos cocinar con lo que hay).

Los asserts de este archivo pasaron de literal-string a comportamiento: la versión previa
casaba `'elif chunk_kind == "initial_plan"'` textualmente y habría bloqueado la SSOT sin
que la garantía se rompiera en absoluto.

Validador funcional E2E: tests/test_chunked_7days_thursday_e2e.py (chunk 2 initial_plan
debe llegar a 'completed' con nevera de fixture no-matching).

tooltip-anchor: P1-CHUNK-AUTONOMY
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_CRON = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")

_KNOB = 'MEALFIT_INITIAL_CHUNK_PANTRY_AUTONOMY'


def _waiver():
    from cron_tasks import _pantry_gate_waiver_reason
    return _pantry_gate_waiver_reason


# ───────────── la garantía original, ahora medida por comportamiento ─────────────

def test_initial_plan_sigue_exento():
    """El corazón de P1-CHUNK-AUTONOMY: un initial_plan nunca es gateado por la nevera."""
    assert _waiver()(chunk_kind="initial_plan") == "initial_plan_autonomy"


def test_refill_y_catchup_conservan_el_gate():
    """Sin waiver, refill/catchup siguen sujetos al gate completo."""
    for kind in ("rolling_refill", "catchup"):
        assert _waiver()(chunk_kind=kind) is None, kind


def test_el_knob_apaga_la_autonomia(monkeypatch):
    monkeypatch.setenv(_KNOB, "false")
    assert _waiver()(chunk_kind="initial_plan") is None, (
        "el knob debe seguir siendo el rollback sin redeploy"
    )


# ───────────── el knob tiene UN solo lector ─────────────

def test_knob_leido_en_un_solo_sitio():
    """Antes se leía en 3 callsites y cada uno podía divergir — divergieron (P1-PANTRY-GATE-SSOT).

    Ahora los 3 gates consultan la SSOT y el knob se lee una vez.
    """
    n = _CRON.count(f'_env_bool("{_KNOB}", True)')
    assert n == 1, (
        f"esperaba el knob leído 1 vez (dentro de _pantry_gate_waiver_reason), hay {n}. "
        "Un segundo lector es exactamente cómo el gate de reservas se desincronizó."
    )


def test_los_tres_gates_consultan_la_ssot():
    """pause pre-pipeline + gate de reservas (normal) + branch del except."""
    n = _CRON.count("_pantry_gate_waiver_reason(")
    assert n >= 4, (  # 1 def + 3 callsites
        f"esperaba la SSOT definida y llamada por los 3 gates, aparece {n} veces"
    )


# ───────────── estructura de los branches (orden y supervivencia) ─────────────

def test_reservation_gate_best_effort_before_partial():
    i_be = _CRON.find("reservation_status = 'best_effort'")
    i_partial = _CRON.find("Marcando reservation_status='partial'")
    assert i_be > 0, "el branch best_effort del gate desapareció"
    assert i_partial > 0, "el branch partial (refill/catchup) debe seguir vivo"
    assert i_be < i_partial, (
        "el elif del waiver debe evaluarse ANTES del else partial — invertirlo re-activa "
        "el RECONCILE-EXHAUSTED para initial_plan"
    )
    assert "elif _res_waiver:" in _CRON[i_be - 1800: i_be]


def test_pause_pre_pipeline_is_kind_aware():
    i = _CRON.find("[P1-CHUNK-AUTONOMY] Chunk {week_number}")
    assert i > 0, "el skip del pause pre-pipeline desapareció"
    blk = _CRON[i - 2500: i + 2500]
    assert "_should_pause_for_empty_pantry" in blk, "el skip vive DENTRO del if should_pause"
    assert "_pause_waiver" in blk, "el skip debe decidir por la SSOT, no por su propia condición"
    # el else conserva la pausa para refill/catchup
    assert "_pause_chunk_for_pantry_refresh(task_id, user_id, week_number, fresh_inventory)" in blk
    assert "[P1-1/PANTRY-EMPTY]" in blk, "la pausa legacy (refill/catchup) debe seguir viva"


def test_refill_catchup_keep_full_gate():
    # El contrato de refill/catchup NO cambia: reconcile + pause + return siguen presentes.
    assert "_reconcile_chunk_reservations(user_id, str(task_id), new_days)" in _CRON
    assert "_handle_reservation_reconciliation_exhausted(" in _CRON
    i = _CRON.find("[P1-CHUNKS-2/RECONCILE-EXHAUSTED] Pausando chunk")
    assert i > 0
    assert "return" in _CRON[i: i + 1200], "el return post-pausa protege contra overbooking en refill"


def test_exception_path_is_kind_aware():
    i = _CRON.find("Reserva lanzó excepción en chunk")
    assert i > 0, "el branch best-effort del except desapareció"
    blk = _CRON[i - 1400: i + 2600]
    assert "except Exception as reserve_err" in blk
    assert "reservation_status = 'best_effort'" in blk
    assert "if _res_waiver:" in blk, "el except debe usar el MISMO waiver que el branch normal"


def test_marker_anchored_in_source():
    # [P1-REBALANCE-LINE-CLAMP · 2026-07-10] durable: anclar en el CÓDIGO, no en el
    # _LAST_KNOWN_PFIX vigente (pinnear el marker actual rota con cada bump posterior).
    assert _CRON.count("P1-CHUNK-AUTONOMY") >= 3, "los anchors del skip desaparecieron de cron_tasks"
