"""[P0-NEW-1-AUTOHEAL + P0-NEW-2-AUTOHEAL · 2026-05-11] Autohealer del
scheduler cascade y startup-time sweep de scheduler alerts huérfanas.

Bug original (audit 2026-05-10):
    P0-AUDIT-1 añadió `_resolve_stale_scheduler_alerts` (sweep cron
    horario, TTL=24h) para cerrar la long-tail de alerts
    `scheduler_missed_*` huérfanas que P1-NEW-2 (listener auto-resolve)
    no cubre (jobs UUID one-off, jobs renombrados, deploy lag).

    PERO el sweep está DENTRO del scheduler que limpia. En audit
    2026-05-10 vimos:
      - 27 alerts `scheduler_missed_*` no resueltas, una de 31.75h
        pese a TTL=24h.
      - El propio `aggregate_coherence_block_history_metrics` y
        `proactive_refresh_pantry_snapshots` MISSED — el scheduler
        estaba saturado.
      - Cuando el scheduler está saturado, el sweep periódico
        también está MISSED → chicken-and-egg: la herramienta que
        limpia el síntoma no corre cuando el síntoma existe.

    Tampoco el detector `_alert_scheduler_cascade_missed` escalaba
    OOB: solo escribía a DB, igualmente susceptible al scheduler
    saturado para entregar la alerta.

Fix (P0-NEW-1-AUTOHEAL + P0-NEW-2-AUTOHEAL):
    1. **Startup-run del sweep** en `lifespan` (app.py) ANTES de
       `scheduler.start()`. Asegura UNA limpieza por deploy/restart
       independiente del estado del pool. El cron periódico
       (P0-AUDIT-1) sigue como defense-in-depth.
    2. **Cascade autohealer** en `_alert_scheduler_cascade_missed`
       (cron_tasks.py): tras emitir la alert DB, escalar a Sentry
       (out-of-band del scheduler), invocar el sweep directamente
       (auto-limpieza local), y emitir
       `pipeline_metrics._scheduler_cascade_autoheal` para
       post-mortem.

Estrategia del test (parser estático):
    1. `app.py:lifespan` importa e invoca `_resolve_stale_scheduler_alerts`
       antes de `scheduler.start()`.
    2. `_alert_scheduler_cascade_missed` invoca `sentry_sdk.capture_message`.
    3. `_alert_scheduler_cascade_missed` invoca `_resolve_stale_scheduler_alerts`.
    4. `_alert_scheduler_cascade_missed` emite `pipeline_metrics`
       con node=`_scheduler_cascade_autoheal`.
    5. El marker `_LAST_KNOWN_PFIX` refleja el P-fix.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"


@pytest.fixture(scope="module")
def app_src() -> str:
    return _APP_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


def _extract_function_body(src: str, name: str) -> str:
    """Extrae el cuerpo de una función top-level (sync o async) hasta
    el siguiente `def`/`async def` top-level."""
    m = re.search(
        rf"^(?:async\s+)?def\s+{re.escape(name)}\s*\([^)]*\)[^:]*:(.*?)(?=^(?:async\s+)?def\s)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, f"No se encontró función top-level `{name}`."
    return m.group(1)


# ---------------------------------------------------------------------------
# P0-NEW-1-AUTOHEAL: startup-time sweep en lifespan
# ---------------------------------------------------------------------------
def test_lifespan_imports_resolve_stale_scheduler_alerts(app_src: str):
    """`lifespan` debe importar `_resolve_stale_scheduler_alerts` para
    el startup-run. Sin import no hay forma de invocarlo."""
    body = _extract_function_body(app_src, "lifespan")
    pattern = re.compile(
        r"from\s+cron_tasks\s+import\s+_resolve_stale_scheduler_alerts"
    )
    assert pattern.search(body), (
        "P0-NEW-1-AUTOHEAL regresión: `lifespan` no importa "
        "`_resolve_stale_scheduler_alerts`. Sin import, el startup "
        "sweep no se puede invocar — la cascada se autosostiene "
        "(chicken-and-egg de P0-AUDIT-1)."
    )


def test_lifespan_invokes_sweep_at_startup(app_src: str):
    """`lifespan` debe invocar `_resolve_stale_scheduler_alerts()`
    explícitamente. Verificación más fuerte que solo importar."""
    body = _extract_function_body(app_src, "lifespan")
    # Permitir whitespace flexible alrededor de los paréntesis.
    pattern = re.compile(r"_resolve_stale_scheduler_alerts\s*\(\s*\)")
    assert pattern.search(body), (
        "P0-NEW-1-AUTOHEAL regresión: `lifespan` ya no invoca "
        "`_resolve_stale_scheduler_alerts()` al startup. Sin esto, "
        "el sweep depende del scheduler periódico — pero ese mismo "
        "scheduler puede estar saturado (cascade) y MISSEAR el sweep."
    )


def test_startup_sweep_inside_try_except(app_src: str):
    """El startup sweep DEBE estar wrapped en try/except — un fallo
    NO puede tirar el startup completo (un sweep fallido es menor
    que tener el app down)."""
    body = _extract_function_body(app_src, "lifespan")
    # Buscar `_resolve_stale_scheduler_alerts()` y verificar que
    # cae dentro de un bloque try/except cercano.
    invoke_idx = body.find("_resolve_stale_scheduler_alerts()")
    assert invoke_idx >= 0, "invoke no encontrado (cubierto por otro test)"
    # Ventana de 500 chars antes y 500 después para validar try/except.
    window_start = max(0, invoke_idx - 500)
    window_end = invoke_idx + 500
    window = body[window_start:window_end]
    assert "try:" in window and "except" in window, (
        "P0-NEW-1-AUTOHEAL regresión: startup sweep sin try/except. "
        "Un fallo del sweep (DB blip, lock contention) tumbaría el "
        "startup completo — pero el sweep NO es startup-critical."
    )


def test_startup_sweep_runs_before_scheduler_start(app_src: str):
    """El startup sweep debe ejecutarse ANTES de `scheduler.start()`.
    Si corre después, el scheduler ya entró en cola de jobs y el
    sweep compite con la cascada que pretende romper."""
    body = _extract_function_body(app_src, "lifespan")
    sweep_idx = body.find("_resolve_stale_scheduler_alerts()")
    start_idx = body.find("scheduler.start()")
    # Si scheduler.start() no aparece textualmente (HAS_SCHEDULER false
    # branches), el test se relaja (no es regresión real).
    if start_idx < 0:
        pytest.skip("scheduler.start() no aparece textualmente en lifespan")
    assert sweep_idx < start_idx, (
        "P0-NEW-1-AUTOHEAL regresión: startup sweep corre DESPUÉS de "
        "`scheduler.start()`. Debería correr ANTES — sino el sweep "
        "compite con la cascada de scheduler_missed que pretende "
        "limpiar."
    )


# ---------------------------------------------------------------------------
# P0-NEW-2-AUTOHEAL: cascade autohealer
# ---------------------------------------------------------------------------
def test_cascade_detector_captures_to_sentry(cron_src: str):
    """`_alert_scheduler_cascade_missed` debe invocar
    `sentry_sdk.capture_message` — escalación OOB del scheduler
    saturado (que puede atrasar la entrega de la alert DB)."""
    body = _extract_function_body(cron_src, "_alert_scheduler_cascade_missed")
    pattern = re.compile(r"sentry_sdk\.capture_message\s*\(")
    assert pattern.search(body), (
        "P0-NEW-2-AUTOHEAL regresión: cascade detector no escala a "
        "Sentry. Sin OOB, la alerta solo vive en DB y depende del "
        "mismo scheduler saturado para que un consumer (cron, "
        "dashboard) la descubra."
    )


def test_cascade_detector_invokes_sweep(cron_src: str):
    """Tras detectar cascade, debe invocar `_resolve_stale_scheduler_alerts()`
    directamente. La cascada es exactamente el momento donde hay
    alerts viejas acumuladas — auto-cleanup local cierra el loop."""
    body = _extract_function_body(cron_src, "_alert_scheduler_cascade_missed")
    pattern = re.compile(r"_resolve_stale_scheduler_alerts\s*\(\s*\)")
    assert pattern.search(body), (
        "P0-NEW-2-AUTOHEAL regresión: cascade detector no invoca el "
        "sweep al final. El sweep periódico puede estar MISSED — "
        "la cascada que se acaba de detectar es nuestra mejor "
        "oportunidad de limpiar."
    )


def test_cascade_detector_emits_pipeline_metrics(cron_src: str):
    """`_alert_scheduler_cascade_missed` debe persistir
    `pipeline_metrics._scheduler_cascade_autoheal` para que el
    post-mortem correlacione cascada → autosweep."""
    body = _extract_function_body(cron_src, "_alert_scheduler_cascade_missed")
    # Cualquiera de los dos patrones es suficiente.
    assert "_scheduler_cascade_autoheal" in body, (
        "P0-NEW-2-AUTOHEAL regresión: cascade detector no emite "
        "`pipeline_metrics._scheduler_cascade_autoheal`. Sin esto, "
        "no hay forma de saber post-mortem si la cascada se "
        "auto-curó vs. si se perdió en el ruido."
    )


def test_cascade_detector_inserts_into_pipeline_metrics(cron_src: str):
    """El emit de autoheal debe ser un INSERT real a `pipeline_metrics`
    — no un log. Diferenciar log de tabla persistente."""
    body = _extract_function_body(cron_src, "_alert_scheduler_cascade_missed")
    pattern = re.compile(r"INSERT\s+INTO\s+pipeline_metrics", re.IGNORECASE)
    assert pattern.search(body), (
        "P0-NEW-2-AUTOHEAL regresión: autoheal signal no se persiste "
        "como INSERT en pipeline_metrics. Sin persistencia, queda "
        "solo en logs efímeros del contenedor."
    )


def test_cascade_autoheal_steps_are_independent(cron_src: str):
    """Cada paso del autohealer (Sentry, sweep, metrics) debe estar
    en su propio try/except — un fallo aislado NO debe abortar los
    otros. Defensa-en-profundidad."""
    body = _extract_function_body(cron_src, "_alert_scheduler_cascade_missed")
    # Contar `try:` después del primer mention de "AUTOHEAL".
    autoheal_idx = body.find("AUTOHEAL")
    if autoheal_idx < 0:
        pytest.fail(
            "P0-NEW-2-AUTOHEAL regresión: marker `AUTOHEAL` no "
            "presente en cascade detector — bloque del autohealer "
            "ausente."
        )
    autoheal_section = body[autoheal_idx:]
    try_count = autoheal_section.count("try:")
    assert try_count >= 3, (
        "P0-NEW-2-AUTOHEAL regresión: el autohealer no tiene 3 "
        "try/except independientes (Sentry, sweep, metrics). "
        "Un fallo (ej. Sentry SDK ausente) NO debe abortar los "
        "siguientes pasos del recovery — found "
        f"{try_count} try blocks, expected >=3."
    )


# ---------------------------------------------------------------------------
# Marker freshness
# ---------------------------------------------------------------------------
def test_last_known_pfix_marker_not_pre_autoheal(app_src: str):
    """`_LAST_KNOWN_PFIX` debe estar bumpeado a P0-NEW-2-AUTOHEAL **o
    posterior** — defensa contra rollback accidental del bump del audit
    2026-05-11.

    Diseño inicial (P0-NEW-2-AUTOHEAL · 2026-05-11):
        El test exigía match exacto `P0-NEW-2-AUTOHEAL`. Funcionó al
        momento del cierre pero quedó stale cuando P-fixes posteriores
        (P2-NEXT-1, P3-NEXT-1, P3-NEXT-2, P3-FINAL-1) bumpearon el
        marker. Patrón conocido del repo: tests "per-PFIX bump
        enforcement" se vuelven stale al siguiente bump.

    Refactor (P2-B-OBS · 2026-05-11):
        Relajado a "marker presente Y fecha ≥ 2026-05-11". El espíritu
        del test (impedir rollback del bump asociado al autohealer)
        se preserva sin requerir actualización por cada P-fix futuro.
        La defensa estricta de formato + freshness vive en
        `test_p3_1_last_known_pfix_freshness.py` (SSOT global).
    """
    from datetime import date, datetime
    marker_re = re.compile(
        r'_LAST_KNOWN_PFIX\s*=\s*[\'"](?P<val>(?P<prefix>P\d+(?:-[A-Z0-9]+)+)\s+·\s+(?P<date>\d{4}-\d{2}-\d{2}))[\'"]'
    )
    m = marker_re.search(app_src)
    assert m, (
        "P0-NEW-2-AUTOHEAL regresión: `_LAST_KNOWN_PFIX` no presente o "
        "con formato inválido. Esperado `Pn-X · YYYY-MM-DD` en app.py:32."
    )
    marker_date = datetime.strptime(m.group("date"), "%Y-%m-%d").date()
    # 2026-05-11 = fecha de cierre del audit que cerró P0-NEW-2-AUTOHEAL.
    # Bump posteriores son aceptables (esa es la intención del repo);
    # rollback ANTES de esa fecha implicaría revertir el autohealer.
    autoheal_floor = date(2026, 5, 11)
    assert marker_date >= autoheal_floor, (
        f"P0-NEW-2-AUTOHEAL regresión: `_LAST_KNOWN_PFIX={m.group('val')!r}` "
        f"tiene fecha {marker_date} < floor {autoheal_floor}. Esto implica "
        f"que el bump asociado al autohealer fue revertido — el cierre del "
        f"audit 2026-05-11 (P0-NEW-1-AUTOHEAL + P0-NEW-2-AUTOHEAL + posteriores) "
        f"debe quedar referenciado en HEAD."
    )


# ---------------------------------------------------------------------------
# [P2-B-OBS · 2026-05-11] Tick observability — cierra el gap donde el sweep
# y el detector de cascada solo emitían pipeline_metrics en path "hot"
# (cuando había alerts a barrer o cascada detectada). En path "cold"
# (sweep healthy con 0 alerts, detector healthy sin cascada) producían 0
# filas en pipeline_metrics — indistinguible de "el cron está MISSED".
# El tick observable cierra ese gap con 1 INSERT por invocación.
# ---------------------------------------------------------------------------
def test_sweep_emits_tick_metric_always(cron_src: str):
    """`_resolve_stale_scheduler_alerts` debe emitir `_scheduler_alerts_sweep_tick`
    en cada invocación — independiente de si barrió 0 o N alerts. Sin esto,
    no hay forma de distinguir "sweep healthy con 0 alerts" de "sweep MISSED"."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    assert "_scheduler_alerts_sweep_tick" in body, (
        "P2-B-OBS regresión: `_resolve_stale_scheduler_alerts` ya no emite "
        "`pipeline_metrics._scheduler_alerts_sweep_tick`. Sin el tick, "
        "el cron es invisible cuando la backlog está limpia — el operador "
        "no puede confirmar liveness en ausencia de alerts huérfanas."
    )


def test_sweep_tick_inserts_into_pipeline_metrics(cron_src: str):
    """El tick del sweep debe ser un INSERT real a pipeline_metrics, no
    un log. Diferenciar log efímero de tabla persistente."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    # El INSERT debe aparecer cerca del nombre del node (mismo bloque).
    pattern = re.compile(
        r"INSERT\s+INTO\s+pipeline_metrics.*?_scheduler_alerts_sweep_tick",
        re.DOTALL | re.IGNORECASE,
    )
    assert pattern.search(body), (
        "P2-B-OBS regresión: tick del sweep no se persiste vía INSERT en "
        "pipeline_metrics. Si solo está como log, queda en stdout efímero "
        "del contenedor — no consultable post-mortem."
    )


def test_sweep_tick_emit_inside_try_except(cron_src: str):
    """El tick del sweep debe estar wrapped en try/except — un fallo de
    persistencia NO puede tirar al caller (lifespan, cascade autohealer,
    cron periódico todos lo invocan)."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    tick_idx = body.find("_scheduler_alerts_sweep_tick")
    assert tick_idx >= 0, "tick emit no encontrado (cubierto por otro test)"
    # Ventana: 800 chars antes (para capturar `try:`) y 2400 después
    # (para capturar `except` posterior al INSERT).
    # [stale-parser fix 2026-06-16] El dict de metadata del tick creció
    # (P3-CLEANUP + P2-NEW-E + P1-CRON-CONSECUTIVE-FAIL añadieron ~25 keys),
    # empujando el `except Exception as _tick_err:` a ~1700 chars tras el
    # nombre del node. El INSERT sigue íntegramente envuelto en try/except
    # (best-effort intacto); solo el span del bloque creció. Ampliamos la
    # ventana forward para alcanzar el `except`.
    window_start = max(0, tick_idx - 800)
    window = body[window_start:tick_idx + 2400]
    assert "try:" in window and "except" in window, (
        "P2-B-OBS regresión: tick emit del sweep no está dentro de "
        "try/except. Best-effort: una excepción al insert (DB blip, "
        "RLS policy change) NO debe abortar el caller — el tick es "
        "observabilidad, no path crítico."
    )


def test_sweep_tick_metadata_includes_swept_count(cron_src: str):
    """El tick metadata debe incluir `swept_count` para que el post-mortem
    correlacione "cuántas alerts se barrieron" con "cuándo corrió"."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    # `swept_count` debe aparecer en el bloque del tick (no en el log INFO
    # original que usaba `n`).
    tick_idx = body.find("_scheduler_alerts_sweep_tick")
    assert tick_idx >= 0
    tick_block = body[tick_idx:tick_idx + 800]
    assert "swept_count" in tick_block, (
        "P2-B-OBS regresión: tick metadata sin `swept_count`. Sin este "
        "campo, post-mortem ve 'el cron corrió' pero no 'cuántas alerts "
        "limpió' — pierde la mitad de la señal."
    )


def test_cascade_detector_emits_check_tick_always(cron_src: str):
    """`_alert_scheduler_cascade_missed` debe emitir
    `_scheduler_cascade_check_tick` SIEMPRE — incluso cuando NO detecta
    cascada. Sin esto, el detector es invisible en path healthy
    (`cascade_detected=False`), indistinguible de "detector MISSED"."""
    body = _extract_function_body(cron_src, "_alert_scheduler_cascade_missed")
    assert "_scheduler_cascade_check_tick" in body, (
        "P2-B-OBS regresión: `_alert_scheduler_cascade_missed` ya no emite "
        "`pipeline_metrics._scheduler_cascade_check_tick`. Sin el tick, "
        "el detector solo aparece en pipeline_metrics cuando detecta cascada "
        "(via `_scheduler_cascade_autoheal`) — el path healthy es invisible "
        "y un detector caído queda enmascarado."
    )


def test_cascade_check_tick_inside_try_except(cron_src: str):
    """El check tick debe estar wrapped en try/except — falla al insertar
    NO debe abortar el resto del flujo (early-return cuando no hay cascada,
    o el bloque de autoheal cuando sí)."""
    body = _extract_function_body(cron_src, "_alert_scheduler_cascade_missed")
    tick_idx = body.find("_scheduler_cascade_check_tick")
    assert tick_idx >= 0, "check tick no encontrado (cubierto por otro test)"
    # Ventana: 800 chars antes y después para capturar try: + except completos.
    window_start = max(0, tick_idx - 800)
    window = body[window_start:tick_idx + 800]
    assert "try:" in window and "except" in window, (
        "P2-B-OBS regresión: check tick emit sin try/except. El tick es "
        "best-effort observabilidad; una excepción aquí NO debe pausar el "
        "flujo de detección de cascada (el autoheal viene después)."
    )


def test_cascade_check_tick_metadata_includes_detection_state(cron_src: str):
    """El metadata debe incluir `cascade_detected` (bool) — campo crítico
    para que el post-mortem distinga "el detector corrió y NO había cascada"
    vs "el detector corrió y la cascada se autoheal-eó"."""
    body = _extract_function_body(cron_src, "_alert_scheduler_cascade_missed")
    tick_idx = body.find("_scheduler_cascade_check_tick")
    assert tick_idx >= 0
    tick_block = body[tick_idx:tick_idx + 800]
    assert "cascade_detected" in tick_block, (
        "P2-B-OBS regresión: check tick metadata sin `cascade_detected`. "
        "Sin este flag, post-mortem ve 'el cron corrió N veces' pero no "
        "puede separar invocaciones healthy de invocaciones con cascada."
    )


def test_cascade_check_tick_runs_before_early_return(cron_src: str):
    """El check tick debe emitirse ANTES del `return` cuando no hay cascada.
    Si va después, en path healthy (cascade_detected=False) el cron retorna
    sin emitir nada — defeats the purpose del tick."""
    body = _extract_function_body(cron_src, "_alert_scheduler_cascade_missed")
    tick_idx = body.find("_scheduler_cascade_check_tick")
    assert tick_idx >= 0
    # Buscar `if not cascade_detected:` y su return posterior.
    no_cascade_return_pattern = re.compile(
        r"if\s+not\s+cascade_detected\s*:\s*\n\s*return"
    )
    m = no_cascade_return_pattern.search(body)
    assert m, (
        "P2-B-OBS regresión: no se encontró early-return basado en "
        "`cascade_detected`. ¿Refactor cambió el control flow? "
        "Verificar que el tick sigue emitiéndose antes de retornar."
    )
    early_return_idx = m.start()
    assert tick_idx < early_return_idx, (
        "P2-B-OBS regresión: check tick emit ocurre DESPUÉS del "
        "`if not cascade_detected: return`. En path healthy (sin cascada), "
        "el return aborta ANTES del tick → cron invisible en path normal."
    )
