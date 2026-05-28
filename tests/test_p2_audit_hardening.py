"""[P2-AUDIT-HARDENING · 2026-05-25] Regression tests parser-based del bundle.

Cubre 3 hardenings del cierre del audit prod-readiness 2026-05-25:

  P2-1: endpoint `POST /admin/deploy-lag/check` gana body opt-in
        `{auto_bump: bool, expected_marker: str}`. Si auto_bump=true Y
        live_marker==expected_marker (param), UPSERT KV. Defensa:
        expected_marker OBLIGATORIO si auto_bump=true (sin él, 400).

  P2-2: cron `_alert_stuck_chunks` con triple filtro
        `pending+attempts=0+execute_after<NOW()-N h`. Cierra false positive
        del audit P0 (JIT-future chunks). Auto-resuelve cuando condición
        desaparece.

  P2-3: cron `_alert_stranded_partial_plans` cierra hueco simétrico de I8
        (`partial+days=[]+age>72h`). Auto-resuelve.

Cross-link: slug `p2_audit_hardening` → este test, exigido por
test_p2_hist_audit_14_marker_test_link.py.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_TASKS = _BACKEND_ROOT / "cron_tasks.py"
_SYSTEM_ROUTER = _BACKEND_ROOT / "routers" / "system.py"
_ALERTS_DOC = _BACKEND_ROOT / "docs" / "system_alerts_resolution_table.md"


@pytest.fixture(scope="module")
def cron_source() -> str:
    return _CRON_TASKS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def system_source() -> str:
    return _SYSTEM_ROUTER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def alerts_doc() -> str:
    return _ALERTS_DOC.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# P2-1: endpoint /admin/deploy-lag/check auto_bump
# ---------------------------------------------------------------------------

def test_p2_1_body_model_present(system_source: str) -> None:
    """`_DeployLagCheckBody` Pydantic model con `auto_bump` + `expected_marker`."""
    assert "class _DeployLagCheckBody" in system_source, (
        "Falta `_DeployLagCheckBody`. Si renombras, actualizar este test."
    )
    assert "auto_bump: bool" in system_source, (
        "`_DeployLagCheckBody.auto_bump: bool` ausente."
    )
    assert "expected_marker: Optional[str]" in system_source, (
        "`_DeployLagCheckBody.expected_marker: Optional[str]` ausente."
    )


def test_p2_1_endpoint_accepts_body(system_source: str) -> None:
    """El endpoint debe aceptar el body opcional via `Body(default=None)`."""
    pattern = re.compile(
        r"def admin_force_deploy_lag_check\(\s*"
        r"request:\s*Request,\s*"
        r"body:\s*Optional\[_DeployLagCheckBody\]"
        r"\s*=\s*Body\(default=None\)"
    )
    assert pattern.search(system_source), (
        "Signature del endpoint cambió. Body opcional Pydantic ausente."
    )


def test_p2_1_requires_expected_marker_when_auto_bump(system_source: str) -> None:
    """Defensa crítica: auto_bump=true sin expected_marker → HTTPException 400."""
    # Hay un raise HTTPException(status_code=400, ...) gatado por
    # `if auto_bump and not expected_marker_param`
    pattern = re.compile(
        r"if\s+auto_bump\s+and\s+not\s+expected_marker_param"
        r".*?HTTPException\(\s*status_code=400",
        re.DOTALL,
    )
    assert pattern.search(system_source), (
        "Guard `if auto_bump and not expected_marker_param: raise HTTPException(400)` "
        "ausente. Sin él, auto_bump ciego podría sincronizar KV con binario "
        "incorrecto (e.g., EasyPanel deployó commit equivocado)."
    )


def test_p2_1_bumps_only_when_live_matches_expected(system_source: str) -> None:
    """UPSERT del KV solo cuando expected_marker_param == live_marker."""
    pattern = re.compile(
        r"if\s+expected_marker_param\s*==\s*live_clean.*?"
        r"INSERT INTO app_kv_store",
        re.DOTALL,
    )
    assert pattern.search(system_source), (
        "El UPSERT del KV en auto_bump path debe estar gateado por "
        "`if expected_marker_param == live_clean`. Sin esa comparación, "
        "auto_bump aceptaría cualquier binario."
    )


def test_p2_1_response_has_kv_bumped_field(system_source: str) -> None:
    """Response debe incluir `kv_bumped: bool`."""
    assert '"kv_bumped": kv_bumped' in system_source, (
        "Response del endpoint debe incluir `kv_bumped` (contrato con SOP)."
    )


def test_p2_1_marker_anchor_in_router(system_source: str) -> None:
    """Tooltip-anchor presente."""
    assert "[P2-DEPLOY-LAG-AUTO-BUMP · 2026-05-25]" in system_source, (
        "Marker anchor `[P2-DEPLOY-LAG-AUTO-BUMP · 2026-05-25]` desapareció."
    )


# ---------------------------------------------------------------------------
# P2-2: cron _alert_stuck_chunks
# ---------------------------------------------------------------------------

def test_p2_2_function_present(cron_source: str) -> None:
    """`def _alert_stuck_chunks` definido."""
    assert "def _alert_stuck_chunks(" in cron_source, (
        "Cron `_alert_stuck_chunks` no encontrado. Si renombras, actualizar."
    )


def test_p2_2_triple_filter_present(cron_source: str) -> None:
    """Triple-filtro CRÍTICO: `pending + attempts=0 + execute_after<NOW()`.

    El filtro `execute_after < NOW() - make_interval(...)` es lo que cierra
    el false positive del audit P0 (JIT-future chunks de planes rolling
    multi-semana). Sin él, la alert es ruidosa cada vez que se crea un plan.
    """
    pattern = re.compile(
        r"q\.status\s*=\s*'pending'.*?"
        r"q\.attempts\s*=\s*0.*?"
        r"q\.execute_after\s*<\s*NOW\(\)\s*-\s*make_interval",
        re.DOTALL,
    )
    assert pattern.search(cron_source), (
        "Falta el triple-filtro `status='pending' AND attempts=0 AND "
        "execute_after<NOW()-N h` en `_alert_stuck_chunks`. Si quitas "
        "execute_after<NOW(), la alert volverá a flagear JIT-future "
        "chunks legítimos como zombies (regression del audit P0)."
    )


def test_p2_2_auto_resolves_when_clear(cron_source: str) -> None:
    """Cuando no hay zombies, cron UPDATE resolved_at de previas."""
    # Patrón: if not rows: ... UPDATE system_alerts SET resolved_at = NOW()
    # WHERE alert_key LIKE 'plan_chunk_zombie:%'
    body_match = re.search(
        r"def _alert_stuck_chunks\(.*?def _alert_stranded_partial_plans",
        cron_source,
        re.DOTALL,
    )
    assert body_match, "No pude extraer el body de _alert_stuck_chunks"
    body = body_match.group(0)
    auto_resolve = re.search(
        r"if not rows:.*?UPDATE system_alerts.*?SET resolved_at\s*=\s*NOW\(\).*?"
        r"alert_key LIKE",
        body,
        re.DOTALL,
    )
    assert auto_resolve, (
        "Auto-resolve sweep ausente en `_alert_stuck_chunks`. Sin él, "
        "el modelo declarado en docs/system_alerts_resolution_table.md "
        "(Auto-explicit) miente — alerts quedarían abiertas para siempre."
    )


def test_p2_2_registered_in_scheduler(cron_source: str) -> None:
    """Cron registrado en `register_plan_chunk_scheduler` con job id estable."""
    assert "scheduler.get_job(\"alert_stuck_chunks\")" in cron_source, (
        "Cron no registrado vía `if not scheduler.get_job('alert_stuck_chunks')`."
    )
    assert "id=\"alert_stuck_chunks\"" in cron_source, (
        "Job id `alert_stuck_chunks` ausente al registrar el cron."
    )


# ---------------------------------------------------------------------------
# P2-3: cron _alert_stranded_partial_plans
# ---------------------------------------------------------------------------

def test_p2_3_function_present(cron_source: str) -> None:
    """`def _alert_stranded_partial_plans` definido."""
    assert "def _alert_stranded_partial_plans(" in cron_source, (
        "Cron `_alert_stranded_partial_plans` no encontrado."
    )


def test_p2_3_filter_partial_zero_days(cron_source: str) -> None:
    """Filtro debe ser `partial + days=[] + age>N`."""
    pattern = re.compile(
        r"plan_data->>'generation_status'\s*=\s*'partial'.*?"
        r"jsonb_array_length\(.*?plan_data->'days'.*?\)\s*=\s*0.*?"
        r"created_at\s*<\s*NOW\(\)\s*-\s*make_interval",
        re.DOTALL,
    )
    assert pattern.search(cron_source), (
        "Falta el filtro `plan_data->>'generation_status'='partial' AND "
        "jsonb_array_length(days)=0 AND created_at<NOW()-N h` en "
        "`_alert_stranded_partial_plans`. Esta es la query que cierra el "
        "hueco simétrico de I8."
    )


def test_p2_3_auto_resolves_when_clear(cron_source: str) -> None:
    """Sin stranded, cron UPDATE resolved_at de previas (Auto-explicit)."""
    body_match = re.search(
        r"def _alert_stranded_partial_plans\(.*?def _alert_chunk_pantry_snapshots_stale",
        cron_source,
        re.DOTALL,
    )
    assert body_match, "No pude extraer body de _alert_stranded_partial_plans"
    body = body_match.group(0)
    auto_resolve = re.search(
        r"if not rows:.*?UPDATE system_alerts.*?SET resolved_at\s*=\s*NOW\(\).*?"
        r"alert_key LIKE",
        body,
        re.DOTALL,
    )
    assert auto_resolve, (
        "Auto-resolve sweep ausente en `_alert_stranded_partial_plans`."
    )


def test_p2_3_registered_in_scheduler(cron_source: str) -> None:
    """Cron registrado con job id `alert_stranded_partial_plans`."""
    assert (
        "scheduler.get_job(\"alert_stranded_partial_plans\")" in cron_source
    ), "Cron P2-3 no registrado."
    assert "id=\"alert_stranded_partial_plans\"" in cron_source, (
        "Job id `alert_stranded_partial_plans` ausente."
    )


# ---------------------------------------------------------------------------
# Cross-link: alert_keys documentados en system_alerts_resolution_table.md
# (test P2-AUDIT-4 también lo verifica, pero asserts directos aquí cierran
# riesgo de que alguien añada el cron pero olvide el doc — y el sibling
# test corre con timing diferente).
# ---------------------------------------------------------------------------

def test_p2_alert_keys_documented(alerts_doc: str) -> None:
    """Los 2 nuevos alert_keys deben tener fila en docs/system_alerts_resolution_table.md."""
    assert "plan_chunk_zombie:<plan_id>" in alerts_doc, (
        "Alert key `plan_chunk_zombie:<plan_id>` no documentado en la tabla."
    )
    assert "plan_stranded_partial:<plan_id>" in alerts_doc, (
        "Alert key `plan_stranded_partial:<plan_id>` no documentado en la tabla."
    )


# ---------------------------------------------------------------------------
# Marker bump verificado
# ---------------------------------------------------------------------------

def test_marker_bumped_to_p2_audit_hardening() -> None:
    """`_LAST_KNOWN_PFIX` debe estar en o despues de P2-AUDIT-HARDENING
    (2026-05-25). Cuando supersede a este bundle un P-fix posterior, el
    test se relaja a date-floor (patron sibling documentado: exact-match
    al cierre + relajacion al supersede)."""
    import re
    from datetime import date, datetime
    app_py = (_BACKEND_ROOT / "app.py").read_text(encoding="utf-8")
    m = re.search(
        r'^_LAST_KNOWN_PFIX\s*=\s*"(?P<val>[^"]+)"',
        app_py,
        re.MULTILINE,
    )
    assert m is not None, "_LAST_KNOWN_PFIX no encontrado en app.py"
    marker = m.group("val")
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})$", marker)
    assert date_m is not None, f"marker sin fecha ISO: {marker!r}"
    marker_date = datetime.strptime(date_m.group(1), "%Y-%m-%d").date()
    assert marker_date >= date(2026, 5, 25), (
        f"_LAST_KNOWN_PFIX={marker!r} con fecha {marker_date} < "
        f"floor 2026-05-25 (P2-AUDIT-HARDENING). Bumpear al ultimo "
        f"P-fix cerrado."
    )
