"""[P1-AUDIT-1 · 2026-05-10] Escalación de `_chunk_learning_stub_count >= 2`
a `system_alerts`.

Bug original (audit 2026-05-10):
    `cron_tasks.py:21543` solo loggeaba a stderr cuando un plan acumulaba
    2+ stubs puros consecutivos (aprendizaje sistemático roto). SRE no
    podía detectar el problema vía dashboard de `system_alerts` — debía
    inspeccionar logs. Inconsistencia con sibling alerts del mismo
    archivo (failed_inventory_deductions, dead_lettered_*) que SÍ
    insertaban en system_alerts.

Fix:
    INSERT/UPSERT a `system_alerts` con:
      - alert_key: `chunk_learning_stub_overload:<plan_id>` (dedup per-plan).
      - alert_type: `chunk_learning_stub_overload`.
      - severity: `critical`.
      - ON CONFLICT DO UPDATE: re-bumpea triggered_at + metadata + message.
    Best-effort: try/except envuelve el INSERT — un fallo del alert NO
    debe pausar el worker.

Estrategia del test (parser estático sobre cron_tasks.py):
    1. INSERT existe inmediatamente después del logger.error STUB-ALERT.
    2. alert_type='chunk_learning_stub_overload'.
    3. severity='critical'.
    4. alert_key contiene `meal_plan_id` (dedup per-plan).
    5. ON CONFLICT DO UPDATE re-bumpea triggered_at.
    6. Try/except defensivo — no propaga al worker.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


def _extract_stub_alert_window(src: str) -> str:
    """Localiza el bloque desde el `logger.error("[P0-3/STUB-ALERT]")`
    hasta el siguiente `_new_lesson = {` (~30 líneas)."""
    m = re.search(
        r"logger\.error\(\s*f?[\"']\[P0-3/STUB-ALERT\][^\"']*[\"'][^)]*\)"
        r"(?P<after>.*?)_new_lesson\s*=\s*\{",
        src,
        re.DOTALL,
    )
    assert m, (
        "P1-AUDIT-1: no se encontró el bloque después del logger.error "
        "STUB-ALERT — ¿se refactorizó el path stub?"
    )
    return m.group("after")


def test_stub_overload_inserts_to_system_alerts(cron_src: str):
    """Tras el logger.error, debe haber INSERT INTO system_alerts."""
    window = _extract_stub_alert_window(cron_src)
    pattern = re.compile(
        r"INSERT\s+INTO\s+system_alerts",
        re.IGNORECASE,
    )
    assert pattern.search(window), (
        "P1-AUDIT-1 regresión: tras el logger.error STUB-ALERT NO hay "
        "INSERT a system_alerts. El evento crítico (≥2 stubs puros) "
        "solo va a stderr → SRE no lo ve en dashboard. Restaurar el "
        "INSERT siguiendo el patrón de las otras alerts del archivo."
    )


def test_alert_type_is_chunk_learning_stub_overload(cron_src: str):
    """`alert_type` debe ser `chunk_learning_stub_overload` (consistente
    con la entrada en CLAUDE.md tabla `system_alerts`)."""
    window = _extract_stub_alert_window(cron_src)
    assert "'chunk_learning_stub_overload'" in window, (
        "P1-AUDIT-1 regresión: alert_type no es "
        "`chunk_learning_stub_overload`. Cambiar el nombre rompe el "
        "linkage con CLAUDE.md y el resolver documentado."
    )


def test_severity_is_critical(cron_src: str):
    """`severity` debe ser `critical` — ≥2 stubs es señal sistémica,
    no warning."""
    window = _extract_stub_alert_window(cron_src)
    # Bind del valor literal en el VALUES.
    pattern = re.compile(r"VALUES\s*\(\s*%s\s*,\s*'chunk_learning_stub_overload'\s*,\s*'critical'")
    assert pattern.search(window), (
        "P1-AUDIT-1 regresión: severity ya no es 'critical'. Downgrade "
        "a 'warning' enmascara la urgencia — el aprendizaje sistemático "
        "roto en 2+ chunks consecutivos requiere intervención humana."
    )


def test_alert_key_dedups_per_plan(cron_src: str):
    """alert_key debe incluir `meal_plan_id` para dedup per-plan
    (formato `chunk_learning_stub_overload:<plan_id>`). Un alert_key
    constante haría que solo el primer plan con el problema sea visible."""
    window = _extract_stub_alert_window(cron_src)
    pattern = re.compile(
        r"chunk_learning_stub_overload:\{meal_plan_id\}",
    )
    assert pattern.search(window), (
        "P1-AUDIT-1 regresión: alert_key no incluye `{meal_plan_id}`. "
        "Sin dedup per-plan, un único upsert sobreescribe los alerts "
        "de OTROS planes con el mismo problema — pierdes visibilidad "
        "de escala."
    )


def test_on_conflict_re_bumps_triggered_at(cron_src: str):
    """ON CONFLICT DO UPDATE debe re-bumpear triggered_at (refresh
    cuando el mismo plan acumula MÁS stubs en el mismo run del cron)."""
    window = _extract_stub_alert_window(cron_src)
    pattern = re.compile(
        r"ON\s+CONFLICT\s*\(\s*alert_key\s*\)\s*DO\s+UPDATE\s+SET\s+triggered_at\s*=\s*NOW\(\)",
        re.IGNORECASE,
    )
    assert pattern.search(window), (
        "P1-AUDIT-1 regresión: ON CONFLICT DO UPDATE no re-bumpea "
        "triggered_at. Sin esto, el segundo stub del mismo plan no "
        "actualiza la fecha del alert — el dashboard mostraría una "
        "fecha vieja aunque el problema persista."
    )


def test_on_conflict_resets_resolved_at(cron_src: str):
    """ON CONFLICT debe poner `resolved_at = NULL`. Si ops resolvió el
    alert manualmente y el problema vuelve, el alert debe re-abrirse."""
    window = _extract_stub_alert_window(cron_src)
    pattern = re.compile(
        r"resolved_at\s*=\s*NULL",
        re.IGNORECASE,
    )
    assert pattern.search(window), (
        "P1-AUDIT-1 regresión: ON CONFLICT no resetea `resolved_at = NULL`. "
        "Si ops cerró manualmente y el problema reaparece, el alert "
        "permanece 'resolved' — silent miss."
    )


def test_insert_wrapped_in_try_except(cron_src: str):
    """INSERT debe estar dentro de try/except — un fallo de DB no
    debe propagar al worker."""
    window = _extract_stub_alert_window(cron_src)
    # Buscar try: cerca del INSERT.
    pattern = re.compile(
        r"try\s*:[^}]*?INSERT\s+INTO\s+system_alerts",
        re.DOTALL | re.IGNORECASE,
    )
    assert pattern.search(window), (
        "P1-AUDIT-1 regresión: INSERT no está envuelto en try/except. "
        "Un blip de DB durante el merge del chunk haría propagar al "
        "worker → chunk falla → escalation_reason='alert_insert_failed'. "
        "Best-effort obligatorio para alerts secundarios."
    )
