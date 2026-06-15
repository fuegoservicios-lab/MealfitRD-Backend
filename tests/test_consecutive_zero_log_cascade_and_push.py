"""[P0-4 zero-log cascade · reescrito 2026-06-14] Cobertura de backend de la feature "zero-log
consecutivo → degrade + push + delay 24h".

Cuando un chunk corre con learning zero-log tras `CHUNK_LEARNING_READY_MAX_DEFERRALS` deferrals y ya
hubo 2 zero-logs consecutivos, el cron debe: (1) bumpear `plan_data._consecutive_zero_log_chunks` a 3,
(2) flippear `generation_status` a `degraded_pending_engagement`, (3) disparar push notification, (4)
delay de futuros chunks pending 24h.

POR QUÉ ESTE ENFOQUE (no driving del monolito):
    El test original mockeaba módulos a nivel de `sys.modules[...] = MagicMock()` (contaminaba la
    sesión + `AttributeError: __path__` con el cron refactorizado → rompía la colección de TODA la
    suite) y llamaba a `process_plan_chunk(...)`, un per-chunk worker SYNC que ya NO existe. Tras el
    refactor la cascade vive en una closure DOBLEMENTE anidada — `process_plan_chunk_queue` (~L24177)
    → `_chunk_worker(task)` (~L24594) → la rama zero-log (~L26120) — no aislable ni invocable, y
    driving el monolito de ~2000 líneas a esa rama vía mocks sería brittle (re-rompería con cada
    refactor del cron). El bump pasó a `update_plan_data_atomic(meal_plan_id, _bump_zero_log, ...)` y el
    push a `_build_zero_log_push_payload(...)` (módulo-level).

    Esta reescritura cubre lo que SÍ es testeable de forma mantenible:
      - FUNCIONAL: `_build_zero_log_push_payload` (el copy/título/url del push por contador +
        logging_preference) — la lógica de mensaje al usuario, módulo-level.
      - PARSER-ANCHOR: la cascade no-aislable (bump→degrade en `_bump_zero_log`, vía
        `update_plan_data_atomic`, el dispatch del push, el delay 24h con `week_number > %s`) — ancla
        la feature contra borrado/refactor silencioso (el intent del test original).
    CERO mocks de módulo → no contamina `sys.modules`. Reemplaza al test stale que estaba skipped.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_CRON = Path(__file__).resolve().parent.parent / "cron_tasks.py"


# ════════════════════════════════════════════════════════════════════════════════════════════════
# FUNCIONAL — `_build_zero_log_push_payload` (módulo-level): el push por contador + logging_preference
# ════════════════════════════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def build_payload():
    from cron_tasks import _build_zero_log_push_payload
    return _build_zero_log_push_payload


def test_push_3plus_manual_alarm_title_and_optout(build_payload):
    """≥3 zero-logs + logging manual → título de alarma + CTA 'Continuar sin registrar' + deeplink."""
    p = build_payload(3, "manual")
    assert p["title"] == "Tu plan se está generando sin tu feedback"
    assert "varios bloques sin registrar" in p["body"]
    assert "Continuar sin registrar" in p["body"]
    assert p["url"] != "/dashboard"  # deeplink al banner del diario (CHUNK_ZERO_LOG_DEEPLINK)


def test_push_3plus_autoproxy_no_optout_cta(build_payload):
    """≥3 + auto_proxy → mismo título de alarma pero SIN el CTA de opt-out (ya optó) + url default."""
    p = build_payload(5, "auto_proxy")
    assert p["title"] == "Tu plan se está generando sin tu feedback"
    assert "Continuar sin registrar" not in p["body"]
    assert p["url"] == "/dashboard"


def test_push_below3_softer_title(build_payload):
    """<3 → título suave (no alarma); manual sigue ofreciendo el opt-out."""
    p = build_payload(2, "manual")
    assert p["title"] == "Loguea tus comidas para continuar"
    assert p["title"] != "Tu plan se está generando sin tu feedback"
    assert "Continuar sin registrar" in p["body"]


def test_push_below3_autoproxy(build_payload):
    p = build_payload(1, "auto_proxy")
    assert p["title"] == "Loguea tus comidas para continuar"
    assert "Continuar sin registrar" not in p["body"]
    assert p["url"] == "/dashboard"


def test_push_payload_shape(build_payload):
    """Contrato: siempre devuelve {title, body, url} (se pasa a _dispatch_push_notification(**payload))."""
    p = build_payload(0, "manual")
    assert set(p.keys()) == {"title", "body", "url"}
    assert all(isinstance(p[k], str) and p[k] for k in p)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# PARSER-ANCHOR — la cascade vive en una closure doblemente anidada (no aislable); anclamos su lógica
# en el source para que un refactor que la rompa/borre falle ESTE test ANTES de tocar producción.
# ════════════════════════════════════════════════════════════════════════════════════════════════
def _src() -> str:
    return _CRON.read_text(encoding="utf-8")


def test_anchor_bump_closure_degrades_at_3():
    """`_bump_zero_log`: incrementa el contador y a ≥3 flippea generation_status a degraded."""
    src = _src()
    assert "def _bump_zero_log(" in src
    assert 'pd["_consecutive_zero_log_chunks"] = n' in src
    assert "if n >= 3:" in src
    assert 'pd["generation_status"] = "degraded_pending_engagement"' in src


def test_anchor_bump_via_atomic_rmw():
    """El bump va por `update_plan_data_atomic` (SELECT FOR UPDATE + callback), NO por un overwrite
    raw de `meal_plans SET plan_data` (que además fallaba: generation_status no es columna)."""
    src = _src()
    import re
    assert re.search(r"update_plan_data_atomic\(\s*meal_plan_id,\s*_bump_zero_log", src), \
        "el bump del contador debe pasar por update_plan_data_atomic(meal_plan_id, _bump_zero_log, ...)"


def test_anchor_push_dispatched_with_consecutive_count():
    """El push se construye con `_build_zero_log_push_payload(consecutive_zero_log_chunks=...)` y se
    despacha vía `_dispatch_push_notification`."""
    src = _src()
    assert "_build_zero_log_push_payload(" in src
    assert "consecutive_zero_log_chunks=" in src
    assert "_dispatch_push_notification(" in src


def test_anchor_future_chunks_delayed_24h():
    """Delay de los chunks pending FUTUROS del mismo plan por 24h."""
    src = _src()
    import re
    assert re.search(r"execute_after\s*=\s*NOW\(\)\s*\+\s*interval '24 hours'", src)
    assert "week_number > %s" in src
