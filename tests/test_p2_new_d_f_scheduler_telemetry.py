"""[P2-NEW-D + P2-NEW-F · 2026-05-08] Tests del scheduler config + listener.

Bug original (audit 2026-05-07):
  - P2-NEW-D: `BackgroundScheduler()` se instanciaba sin args. Default
    `ThreadPoolExecutor(max_workers=10)` + `misfire_grace_time=1s`. Con ~23
    cron jobs registrados un burst en minuto 0 podía saturar el pool de 10
    threads y APScheduler skipeaba silenciosamente runs que no llegaban en
    1s — sin métrica.
  - P2-NEW-F: ningún listener `EVENT_JOB_MISSED`/`EVENT_JOB_ERROR` registrado.
    Skips/errores solo aparecían en log warning sin alerta consultable.

Fix:
  1. `BackgroundScheduler(executors={"default": ThreadPoolExecutor(N)},
     job_defaults={"misfire_grace_time": M, "coalesce": True, "max_instances": 1})`.
  2. Knobs `MEALFIT_SCHEDULER_MAX_WORKERS` (default 20),
     `MEALFIT_SCHEDULER_MISFIRE_GRACE_S` (default 60),
     `MEALFIT_SCHEDULER_TELEMETRY_ENABLED` (default `on`, runtime kill switch).
  3. Listener `_scheduler_alert_listener` registrado ANTES de `start()` con
     mask `EVENT_JOB_MISSED | EVENT_JOB_ERROR` que upserta `system_alerts`
     con `alert_key=scheduler_<event_type>_<job_id>` (idempotente).
  4. Try/except defensivo en el listener: errores no crashean el scheduler.

Cobertura (defensa textual, mismo patrón que test_p1_2_yield_multiplier_symmetry
y otros que validan invariantes de archivos pesados de importar):
  - test_scheduler_explicit_executors_and_job_defaults
  - test_scheduler_max_workers_knob_present
  - test_scheduler_misfire_grace_knob_present
  - test_scheduler_telemetry_enabled_knob_runtime
  - test_listener_function_defined
  - test_listener_handles_missed_and_error_codes
  - test_listener_upserts_system_alerts_with_alert_key
  - test_listener_has_defensive_try_except
  - test_listener_registered_before_scheduler_start
"""
import re
from pathlib import Path

import pytest


_APP_PATH = Path(__file__).parent / "app.py"


@pytest.fixture(scope="module")
def app_source() -> str:
    return _APP_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# P2-NEW-D: BackgroundScheduler con executors + job_defaults explícitos
# ---------------------------------------------------------------------------
def test_scheduler_explicit_executors_and_job_defaults(app_source):
    """`BackgroundScheduler(...)` debe pasar `executors=` y `job_defaults=`
    para no caer en defaults de APScheduler (10 workers / 1s grace)."""
    # Aislar el bloque del constructor (paréntesis balanceados). Anchor en
    # la asignación `scheduler = BackgroundScheduler(` para saltar la
    # mención en el comentario explicativo del bug original.
    start = app_source.find("scheduler = BackgroundScheduler(")
    assert start != -1, "`scheduler = BackgroundScheduler(...)` no encontrado."
    start = app_source.find("BackgroundScheduler(", start)
    depth = 0
    end = -1
    for i, ch in enumerate(app_source[start:], start=start):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    assert end != -1, "Constructor BackgroundScheduler sin cierre balanceado."
    ctor = app_source[start:end + 1]
    assert "executors=" in ctor, (
        "BackgroundScheduler debe instanciarse con `executors=` explícito. "
        "Sin esto, APScheduler usa ThreadPoolExecutor(max_workers=10) por "
        "default, lo que satura bajo 23+ crons activos."
    )
    assert "job_defaults=" in ctor, (
        "BackgroundScheduler debe pasar `job_defaults=` con `misfire_grace_time` "
        "explícito. El default de 1s es demasiado agresivo para crons que "
        "compiten por un thread pool."
    )
    # `misfire_grace_time`, `coalesce`, `max_instances` dentro de job_defaults.
    assert "misfire_grace_time" in ctor
    assert '"coalesce": True' in ctor or "'coalesce': True" in ctor
    assert '"max_instances": 1' in ctor or "'max_instances': 1" in ctor


def test_scheduler_max_workers_knob_present(app_source):
    """Knob `MEALFIT_SCHEDULER_MAX_WORKERS` debe leerse del env con default 20."""
    assert "MEALFIT_SCHEDULER_MAX_WORKERS" in app_source
    # Default 20: cubre el burst esperado de ~23 jobs sin sobre-provisionar.
    m = re.search(
        r'MEALFIT_SCHEDULER_MAX_WORKERS["\s,)]+["\s]*"?(\d+)"?',
        app_source,
    )
    assert m, "Default explícito de MEALFIT_SCHEDULER_MAX_WORKERS no detectado."
    default = int(m.group(1))
    assert 15 <= default <= 50, (
        f"Default MEALFIT_SCHEDULER_MAX_WORKERS={default} fuera de rango "
        f"razonable [15, 50]. Bajo subprovisiona; alto desperdicia memoria."
    )


def test_scheduler_misfire_grace_knob_present(app_source):
    """Knob `MEALFIT_SCHEDULER_MISFIRE_GRACE_S` con default ≥ 30s."""
    assert "MEALFIT_SCHEDULER_MISFIRE_GRACE_S" in app_source
    m = re.search(
        r'MEALFIT_SCHEDULER_MISFIRE_GRACE_S["\s,)]+["\s]*"?(\d+)"?',
        app_source,
    )
    assert m, "Default explícito de MEALFIT_SCHEDULER_MISFIRE_GRACE_S no detectado."
    default = int(m.group(1))
    assert default >= 30, (
        f"Default MEALFIT_SCHEDULER_MISFIRE_GRACE_S={default}s es muy bajo. "
        f"Mínimo 30s para absorber GC pauses/lock contention transitorios. "
        f"El valor original (1s default de APScheduler) era el bug."
    )


# ---------------------------------------------------------------------------
# P2-NEW-F: listener EVENT_JOB_MISSED|EVENT_JOB_ERROR → system_alerts
# ---------------------------------------------------------------------------
def test_scheduler_telemetry_enabled_knob_runtime(app_source):
    """`_is_scheduler_telemetry_enabled` debe ser FUNCIÓN (no constante) para
    que el kill switch sea runtime (sin restart). Si fuera constante, togglear
    el env var requiere redeploy — perdiendo el valor del knob."""
    assert "_is_scheduler_telemetry_enabled" in app_source
    # Debe ser función (def), no asignación a una constante leída al import.
    assert re.search(
        r"def\s+_is_scheduler_telemetry_enabled\s*\(", app_source
    ), (
        "_is_scheduler_telemetry_enabled debe ser función (lee env fresh "
        "cada llamada). Una constante captura el valor al import y rompe "
        "el kill switch operacional."
    )
    # La función debe leer el knob — buscar la siguiente declaración def
    # después para acotar el cuerpo, sin asumir formato del return.
    func_start = app_source.find("def _is_scheduler_telemetry_enabled")
    assert func_start != -1
    next_def = app_source.find("\ndef ", func_start + 1)
    next_class = app_source.find("\n@", func_start + 1)
    boundary = min(p for p in (next_def, next_class) if p != -1)
    func_body = app_source[func_start:boundary]
    assert "MEALFIT_SCHEDULER_TELEMETRY_ENABLED" in func_body, (
        "_is_scheduler_telemetry_enabled debe leer el env "
        "MEALFIT_SCHEDULER_TELEMETRY_ENABLED en cada llamada."
    )


def test_listener_function_defined(app_source):
    """`_scheduler_alert_listener(event)` debe existir como función."""
    assert re.search(
        r"def\s+_scheduler_alert_listener\s*\(\s*event\s*\)", app_source
    ), "Falta la función _scheduler_alert_listener(event)."


def test_listener_handles_missed_and_error_codes(app_source):
    """El listener debe manejar AMBOS códigos `EVENT_JOB_MISSED` y `EVENT_JOB_ERROR`."""
    # Aislamos el cuerpo del listener para que el assert no se cumpla por
    # accidente con otras menciones del código.
    listener_match = re.search(
        r"def\s+_scheduler_alert_listener.*?(?=\n(?:def\s|@asynccontextmanager))",
        app_source,
        re.DOTALL,
    )
    assert listener_match, "No se pudo aislar el cuerpo de _scheduler_alert_listener."
    body = listener_match.group(0)
    assert "EVENT_JOB_MISSED" in body, "Listener no maneja EVENT_JOB_MISSED."
    assert "EVENT_JOB_ERROR" in body, "Listener no maneja EVENT_JOB_ERROR."


def test_listener_upserts_system_alerts_with_alert_key(app_source):
    """El listener debe upsertar a `system_alerts` con `on_conflict='alert_key'`
    para idempotencia (cada nuevo evento del mismo job actualiza la fila)."""
    listener_match = re.search(
        r"def\s+_scheduler_alert_listener.*?(?=\n(?:def\s|@asynccontextmanager))",
        app_source,
        re.DOTALL,
    )
    assert listener_match
    body = listener_match.group(0)
    assert 'table("system_alerts")' in body or "table('system_alerts')" in body, (
        "Listener debe escribir a la tabla system_alerts."
    )
    assert "upsert" in body, "Listener debe usar UPSERT (no INSERT plano)."
    assert (
        'on_conflict="alert_key"' in body
        or "on_conflict='alert_key'" in body
    ), (
        "UPSERT debe especificar on_conflict='alert_key' para que multiples "
        "eventos del mismo job actualicen la misma fila (idempotencia)."
    )
    # alert_key debe codificar el event_type + job_id para diferenciar runs.
    assert "scheduler_" in body and "job_id" in body, (
        "alert_key debe contener el job_id y el tipo de evento."
    )


def test_listener_has_defensive_try_except(app_source):
    """El listener debe envolver su lógica en try/except para no crashear el
    scheduler si supabase falla (por ejemplo durante un DB outage)."""
    listener_match = re.search(
        r"def\s+_scheduler_alert_listener.*?(?=\n(?:def\s|@asynccontextmanager))",
        app_source,
        re.DOTALL,
    )
    assert listener_match
    body = listener_match.group(0)
    assert "try:" in body and "except" in body, (
        "Listener debe tener try/except defensivo. Si un fallo en supabase "
        "propaga, APScheduler puede des-registrar el listener (peor)."
    )
    # El except debe loguear, no silenciar completamente.
    assert "logger." in body, (
        "El except debe loguear la falla (warning/error), no silenciarla."
    )


def test_listener_registered_before_scheduler_start(app_source):
    """`scheduler.add_listener(...)` DEBE estar antes de `scheduler.start()`
    para no perder los primeros eventos."""
    add_listener_pos = app_source.find("scheduler.add_listener")
    start_pos = app_source.find("scheduler.start()")
    assert add_listener_pos != -1, "scheduler.add_listener no encontrado."
    assert start_pos != -1, "scheduler.start() no encontrado."
    assert add_listener_pos < start_pos, (
        "scheduler.add_listener debe llamarse ANTES de scheduler.start(). "
        "Si va después, los eventos disparados durante el arranque (incluido "
        "el primer batch de jobs si llegan rápido) se pierden."
    )


def test_listener_uses_combined_event_mask(app_source):
    """El add_listener debe pasar mask combinado MISSED|ERROR (no llamar dos veces)."""
    m = re.search(
        r"scheduler\.add_listener\s*\(\s*_scheduler_alert_listener\s*,\s*"
        r"EVENT_JOB_MISSED\s*\|\s*EVENT_JOB_ERROR",
        app_source,
    )
    assert m, (
        "add_listener debe pasar `EVENT_JOB_MISSED | EVENT_JOB_ERROR` como "
        "mask combinado. Dos add_listener por separado funcionan pero son "
        "más ruido en logs y diferencian incorrectamente."
    )
