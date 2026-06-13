"""[P1-NEW-2 · 2026-05-10] Listener APScheduler DEBE escuchar
`EVENT_JOB_EXECUTED` y auto-resolver alerts `scheduler_missed_<job>` y
`scheduler_error_<job>` cuando el mismo job ejecuta exitosamente.

Bug original (audit 2026-05-10):
    Listener solo recibía `EVENT_JOB_MISSED|EVENT_JOB_ERROR`. Las alerts
    insertadas vivían con `resolved_at=NULL` hasta resolución manual.
    Producción acumuló 25 `scheduler_missed_*` unresolved en últimas 24h —
    imposible distinguir "job sigue fallando" vs "ya recuperó". Triage
    cost crecía linealmente.

Fix:
    1. Import `EVENT_JOB_EXECUTED` y stub fallback si APScheduler ausente.
    2. Handler `elif code == EVENT_JOB_EXECUTED:` que UPDATEea
       `system_alerts SET resolved_at=NOW() WHERE alert_key IN
       (scheduler_missed_<job>, scheduler_error_<job>) AND
       resolved_at IS NULL`.
    3. `scheduler.add_listener(..., MISSED | ERROR | EXECUTED)`.

Estrategia del test (parser estático sobre app.py):
    1. Verificar import de `EVENT_JOB_EXECUTED`.
    2. Verificar branch `elif code == EVENT_JOB_EXECUTED:` en el listener.
    3. Verificar UPDATE `resolved_at=...` con `alert_key in [missed, error]`.
    4. Verificar mask con `EVENT_JOB_EXECUTED` en `add_listener`.
    5. Verificar idempotencia: filtro `resolved_at is null` para no
       tocar alerts ya cerradas manualmente.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_APP_PY = _REPO_ROOT / "backend" / "app.py"


@pytest.fixture(scope="module")
def app_src() -> str:
    return _APP_PY.read_text(encoding="utf-8")


def test_event_job_executed_imported(app_src: str):
    """`EVENT_JOB_EXECUTED` debe importarse de apscheduler.events."""
    pattern = re.compile(
        r"from\s+apscheduler\.events\s+import\s+[^\n]*EVENT_JOB_EXECUTED",
    )
    assert pattern.search(app_src), (
        "P1-NEW-2 regresión: `EVENT_JOB_EXECUTED` ya no se importa de "
        "`apscheduler.events`. Sin el import, el handler de auto-resolve "
        "no puede correr (NameError al primer EXECUTED event)."
    )


def test_event_job_executed_stub_on_missing_apscheduler(app_src: str):
    """En el fallback `except ImportError`, `EVENT_JOB_EXECUTED` debe
    tener un valor stub (0) — sin esto, el listener referencia una
    variable indefinida cuando APScheduler no está instalado."""
    # Patrón: stub que define EVENT_JOB_EXECUTED = ... = 0 dentro del
    # except ImportError block.
    pattern = re.compile(
        r"except\s+ImportError.*?EVENT_JOB_EXECUTED\s*=",
        re.DOTALL,
    )
    assert pattern.search(app_src), (
        "P1-NEW-2 regresión: el `except ImportError` ya no stub-ea "
        "`EVENT_JOB_EXECUTED`. Si APScheduler no está instalado, el "
        "código del listener falla con NameError al evaluar "
        "`code == EVENT_JOB_EXECUTED`."
    )


def test_listener_handles_executed_branch(app_src: str):
    """El listener debe tener una rama explícita para EVENT_JOB_EXECUTED
    que UPDATEea `system_alerts.resolved_at`. Defense-in-depth: si la
    rama está pero no hace UPDATE, los alerts siguen huérfanos."""
    # Localizar `_scheduler_alert_listener` body hasta el siguiente def.
    listener_match = re.search(
        r"def\s+_scheduler_alert_listener\s*\([^)]*\)\s*:(.*?)(?=\n(?:def\s|@app\.|@asynccontextmanager))",
        app_src,
        re.DOTALL,
    )
    assert listener_match, (
        "P1-NEW-2 regresión: no se encontró la función "
        "`_scheduler_alert_listener` en app.py."
    )
    body = listener_match.group(1)

    branch_pattern = re.compile(
        r"elif\s+code\s*==\s*EVENT_JOB_EXECUTED\s*:",
    )
    assert branch_pattern.search(body), (
        "P1-NEW-2 regresión: el listener no tiene rama "
        "`elif code == EVENT_JOB_EXECUTED:`. Sin ella, los EXECUTED "
        "events caen al `else: return` y las alerts misseadas previas "
        "nunca se resuelven."
    )

    # Dentro del listener body, debe aparecer la actualización de
    # resolved_at con los alert_keys del scheduler.
    # [P1-NEON-DB-MIGRATION · 2026-06-12] Re-anclado del builder PostgREST
    # (`.update({'resolved_at': ...})`) al SQL directo ejecutable:
    # `UPDATE system_alerts SET resolved_at = %s`.
    resolve_pattern = re.compile(
        r"UPDATE\s+system_alerts\s+SET\s+resolved_at\s*=\s*%s",
    )
    assert resolve_pattern.search(body), (
        "P1-NEW-2 regresión: el listener no ejecuta "
        "`UPDATE system_alerts SET resolved_at = %s` para auto-resolver "
        "alerts. Sin esto, las filas en system_alerts viven indefinidamente."
    )


def test_auto_resolve_filters_only_unresolved(app_src: str):
    """El UPDATE de resolved_at debe filtrar por `resolved_at IS NULL`.
    [P1-NEON-DB-MIGRATION · 2026-06-12] Re-anclado del builder PostgREST
    (`.is_('resolved_at', 'null')`) al predicado SQL ejecutable dentro del
    MISMO statement UPDATE del listener (no un comentario). Sin esto,
    pisamos timestamps de alerts ya cerradas manualmente.
    """
    listener_match = re.search(
        r"def\s+_scheduler_alert_listener\s*\([^)]*\)\s*:(.*?)(?=\n(?:def\s|@app\.|@asynccontextmanager))",
        app_src,
        re.DOTALL,
    )
    assert listener_match, (
        "P1-NEW-2 regresión: no se encontró la función "
        "`_scheduler_alert_listener` en app.py."
    )
    body = listener_match.group(1)
    # El predicado debe vivir DENTRO del mismo string SQL del UPDATE
    # ([^"] impide saltar a otro statement/comentario fuera del literal).
    pattern = re.compile(
        r'UPDATE\s+system_alerts\s+SET\s+resolved_at\s*=\s*%s'
        r'[^"]*?WHERE[^"]*?resolved_at\s+IS\s+NULL',
    )
    assert pattern.search(body), (
        "P1-NEW-2 regresión: el UPDATE de resolved_at NO filtra por "
        "`resolved_at IS NULL` en el mismo statement. Sin este predicado, "
        "una alert que ops resolvió hace días se le sobrescribe el "
        "timestamp cada vez que el job ejecuta — perdemos el historial "
        "real de resolución."
    )


def test_auto_resolve_covers_both_missed_and_error(app_src: str):
    """El UPDATE debe cubrir ambos alert_keys: `scheduler_missed_<job>`
    y `scheduler_error_<job>`. Cubrir solo missed deja errores
    persistentes sin auto-cleanup."""
    listener_match = re.search(
        r"def\s+_scheduler_alert_listener\s*\([^)]*\)\s*:(.*?)(?=\n(?:def\s|@app\.|@asynccontextmanager))",
        app_src,
        re.DOTALL,
    )
    assert listener_match
    body = listener_match.group(1)

    assert "scheduler_missed_" in body and "scheduler_error_" in body, (
        "P1-NEW-2 regresión: el auto-resolve no cubre ambos alert_keys "
        "(`scheduler_missed_<job>` y `scheduler_error_<job>`). "
        "Solo cubrir missed deja errores persistentes huérfanos."
    )


def test_add_listener_mask_includes_executed(app_src: str):
    """`scheduler.add_listener(handler, MASK)` debe usar
    `EVENT_JOB_MISSED | EVENT_JOB_ERROR | EVENT_JOB_EXECUTED`."""
    pattern = re.compile(
        r"add_listener\(\s*_scheduler_alert_listener\s*,\s*"
        r"EVENT_JOB_MISSED\s*\|\s*EVENT_JOB_ERROR\s*\|\s*EVENT_JOB_EXECUTED",
    )
    assert pattern.search(app_src), (
        "P1-NEW-2 regresión: la mask del `add_listener` no incluye "
        "`EVENT_JOB_EXECUTED`. Sin él, APScheduler no notifica al listener "
        "cuando los jobs ejecutan, y el branch de auto-resolve nunca "
        "se ejecuta — alerts misseadas se acumulan otra vez."
    )
