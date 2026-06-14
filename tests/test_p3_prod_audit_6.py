"""[P3-PROD-AUDIT-6 · 2026-05-12] Bundle test parser-based para los 6 P3
del audit production-readiness post-P2-PROD-AUDIT-3:

  1. P3-LOOSE-SQL-FILES — `supabase/` raíz debe contener SOLO la subcarpeta
     `migrations/`. Cualquier `*.sql` suelto rompe la convención SSOT
     (CLAUDE.md "DDL en runtime: prohibido").

  2. P3-DEPRECATED-UTCNOW — `datetime.utcnow()` (deprecated en Python 3.12+)
     no debe aparecer en código de producción. Tests con marker
     `naive a propósito` están exentos.

  3. P3-PASSWORD-MIN-LENGTH — Register.jsx y ResetPassword.jsx deben
     exigir mínimo 8 caracteres (OWASP), no 6.

  4. P3-PREVIEW-MODEL-KNOB — `proactive_agent.py` debe usar el helper
     `_proactive_model_name()` (knob-driven) en lugar de hardcodear el
     model ID. Permite swap inmediato si Google deprecar el modelo
     preview sin redeploy.

  5. P3-FULL-TABLE-SCAN-HEALTH — `/api/system/health` debe usar
     `INTERVAL '<N> days'` lookback en nudge_outcomes/abandoned_meal_reasons
     y `LIMIT` en user_profiles. Knobs:
       - MEALFIT_SYSTEM_HEALTH_NUDGE_DAYS (default 30, clamp [1, 365]).
       - MEALFIT_SYSTEM_HEALTH_PROFILE_LIMIT (default 10000, clamp [100, 100000]).

  6. P3-NOTIFICATIONS-RATE-LIMIT — `/api/notifications/subscribe` y
     `/unsubscribe` deben usar `RateLimiter` en lugar de
     `Depends(get_verified_user_id)` directo (sin throttle).

Tooltip-anchor: P3-PROD-AUDIT-6-TESTS
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ROOT = _REPO_ROOT / "backend"
_FRONTEND_ROOT = _REPO_ROOT / "frontend"
_MIGRATIONS_ROOT = _REPO_ROOT / "migrations"

_DIARY_PY = _BACKEND_ROOT / "routers" / "diary.py"
_SYSTEM_PY = _BACKEND_ROOT / "routers" / "system.py"
_NOTIFICATIONS_PY = _BACKEND_ROOT / "routers" / "notifications.py"
_PROACTIVE_PY = _BACKEND_ROOT / "proactive_agent.py"
_REGISTER_JSX = _FRONTEND_ROOT / "src" / "pages" / "Register.jsx"
_RESET_PASSWORD_JSX = _FRONTEND_ROOT / "src" / "pages" / "ResetPassword.jsx"


@pytest.fixture(scope="module")
def diary_src() -> str:
    return _DIARY_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def system_src() -> str:
    return _SYSTEM_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def notifications_src() -> str:
    return _NOTIFICATIONS_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def proactive_src() -> str:
    return _PROACTIVE_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def register_src() -> str:
    return _REGISTER_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def reset_password_src() -> str:
    return _RESET_PASSWORD_JSX.read_text(encoding="utf-8")


# ============================================================================
# Section 1: P3-LOOSE-SQL-FILES
# ============================================================================

def test_migrations_dir_is_flat_ssot():
    """[P1-NEON-DB-MIGRATION] El SSOT del DDL vive en `migrations/` (estructura
    PLANA: los `.sql` directamente en el dir, sin subcarpeta). Guard de
    regresión del rename `supabase/`→`migrations/`: `migrations/` existe con
    `.sql` y NO quedó un `supabase/` legacy."""
    assert _MIGRATIONS_ROOT.is_dir(), (
        f"P1-NEON-DB-MIGRATION: falta el dir SSOT `migrations/` en {_REPO_ROOT}."
    )
    sql_files = sorted(p.name for p in _MIGRATIONS_ROOT.glob("*.sql"))
    assert sql_files, (
        "P1-NEON-DB-MIGRATION: `migrations/` no contiene `.sql` — "
        "¿el rename movió mal los archivos?"
    )
    legacy = _REPO_ROOT / "supabase"
    assert not legacy.exists(), (
        "P1-NEON-DB-MIGRATION: el dir legacy `supabase/` reapareció. El SSOT del "
        "DDL se renombró a `migrations/` (plano); mover los `.sql` y borrar `supabase/`."
    )


# ============================================================================
# Section 2: P3-DEPRECATED-UTCNOW
# ============================================================================

def test_no_datetime_utcnow_in_diary(diary_src: str):
    """`datetime.utcnow()` deprecated. Reemplazar con
    `datetime.now(timezone.utc)`."""
    assert "datetime.utcnow()" not in diary_src, (
        "P3-DEPRECATED-UTCNOW regresión: `datetime.utcnow()` reintroducido "
        "en diary.py. Python 3.12+ emite DeprecationWarning; en 3.14 "
        "(esperado) será removed. Usar `datetime.now(timezone.utc)` que "
        "produce tz-aware datetime equivalente."
    )


def test_no_datetime_utcnow_in_production_paths():
    """Escaneo global de uso prod (excluyendo tests + comentarios)."""
    # Whitelist: tests pueden usar utcnow con comment explícito
    # `# naive a propósito` para testear contratos legacy.
    prod_files = list((_BACKEND_ROOT).rglob("*.py"))
    violations: list[tuple[str, int, str]] = []
    for f in prod_files:
        rel = f.relative_to(_REPO_ROOT)
        parts = set(rel.parts)
        # Skip tests, scratch, refactor scripts y venvs locales (deps
        # externas en site-packages NO son código de producción nuestro —
        # e.g. requests_toolbelt usa utcnow() y no podemos arreglarlo).
        if any(p in parts for p in (
            "tests", "scratch",
            "venv", ".venv", "venv-test", "test_venv",
            "site-packages", "node_modules",
        )):
            continue
        if f.name.startswith(("test_", "refactor", "scratch_")):
            continue
        # [P3-DEBUG-TIME-CLEANUP · 2026-05-20] `recalc_now.py` movido a
        # `backend/scratch/legacy_root_helpers/` — el filtro `scratch in parts`
        # arriba ya lo excluye. Dejado este comentario por trazabilidad si
        # alguien reintroduce un script one-shot CLI en raíz.
        try:
            for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if "datetime.utcnow()" in line and not line.strip().startswith("#"):
                    violations.append((str(rel), lineno, line.strip()))
        except Exception:
            continue
    assert violations == [], (
        f"P3-DEPRECATED-UTCNOW: `datetime.utcnow()` detectado en código "
        f"de producción ({len(violations)} ocurrencias):\n"
        + "\n".join(f"  {f}:{ln}: {line}" for f, ln, line in violations[:10])
        + "\n\nReemplazar con `datetime.now(timezone.utc)` (tz-aware) o "
        f"`datetime.now(timezone.utc).replace(tzinfo=None)` (naive) según "
        f"lo que el caller espere."
    )


# ============================================================================
# Section 3: P3-PASSWORD-MIN-LENGTH
# ============================================================================

def test_register_password_min_8(register_src: str):
    """Register.jsx debe exigir mínimo 8 caracteres."""
    # Patrón legacy prohibido
    assert "password.length < 6" not in register_src, (
        "P3-PASSWORD-MIN-LENGTH regresión: Register.jsx volvió a aceptar "
        "passwords de 6 caracteres. OWASP recomienda ≥ 8."
    )
    assert "password.length < 8" in register_src, (
        "P3-PASSWORD-MIN-LENGTH: Register.jsx no exige mínimo 8 caracteres."
    )
    # Error message coherente
    assert "al menos 8 caracteres" in register_src, (
        "P3-PASSWORD-MIN-LENGTH: el error message de Register.jsx aún "
        "menciona 6 caracteres. Sync con el check `< 8`."
    )


def test_reset_password_min_8(reset_password_src: str):
    """ResetPassword.jsx también debe exigir mínimo 8."""
    assert "password.length < 6" not in reset_password_src, (
        "P3-PASSWORD-MIN-LENGTH regresión: ResetPassword.jsx volvió a "
        "aceptar 6 caracteres. Drift con Register.jsx."
    )
    assert "password.length < 8" in reset_password_src, (
        "P3-PASSWORD-MIN-LENGTH: ResetPassword.jsx no exige mínimo 8 "
        "caracteres."
    )


# ============================================================================
# Section 4: P3-PREVIEW-MODEL-KNOB
# ============================================================================

def test_proactive_model_knob_helper_defined(proactive_src: str):
    """Helper `_proactive_model_name()` debe estar definido y leer
    `MEALFIT_PROACTIVE_SENTIMENT_MODEL` con default explícito."""
    assert "def _proactive_model_name" in proactive_src, (
        "P3-PREVIEW-MODEL-KNOB: helper `_proactive_model_name` no definido. "
        "Sin el helper, las dos callsites del cron quedan hardcoded — sin "
        "escape hatch si Google deprecat el modelo preview."
    )
    assert "MEALFIT_PROACTIVE_SENTIMENT_MODEL" in proactive_src, (
        "P3-PREVIEW-MODEL-KNOB: knob "
        "`MEALFIT_PROACTIVE_SENTIMENT_MODEL` no presente en "
        "proactive_agent.py."
    )


def test_proactive_callsites_use_helper(proactive_src: str):
    """Ambas callsites (`classify_nudge_sentiment` y
    `_compose_proactive_message`) deben invocar el helper, no hardcodear
    el model ID."""
    # Buscar al menos 2 invocaciones del helper en ChatDeepSeek
    # ([P0-DEEPSEEK-MIGRATION · 2026-06-12] constructor renombrado).
    invocations = re.findall(
        r"ChatDeepSeek\(\s*\n?\s*model\s*=\s*_proactive_model_name\(\)",
        proactive_src,
        re.MULTILINE,
    )
    assert len(invocations) >= 2, (
        f"P3-PREVIEW-MODEL-KNOB: encontradas {len(invocations)} invocaciones "
        f"de `model=_proactive_model_name()`, esperaban ≥ 2 (callsites en "
        f"`classify_nudge_sentiment` + `_compose_proactive_message`). "
        f"Una callsite legacy probablemente sigue hardcoded."
    )


# ============================================================================
# Section 5: P3-FULL-TABLE-SCAN-HEALTH
# ============================================================================

def test_system_health_uses_rolling_window(system_src: str):
    """`/system/health` debe usar `INTERVAL '<N> days'` (lookback rolling)
    en lugar de full table scan."""
    # Patrón nuevo presente
    assert "MEALFIT_SYSTEM_HEALTH_NUDGE_DAYS" in system_src, (
        "P3-FULL-TABLE-SCAN-HEALTH: knob "
        "`MEALFIT_SYSTEM_HEALTH_NUDGE_DAYS` no presente. Sin el knob, no "
        "podemos ajustar el lookback sin redeploy."
    )
    assert "MEALFIT_SYSTEM_HEALTH_PROFILE_LIMIT" in system_src, (
        "P3-FULL-TABLE-SCAN-HEALTH: knob "
        "`MEALFIT_SYSTEM_HEALTH_PROFILE_LIMIT` no presente."
    )
    # Query de nudge_outcomes debe ser rolling, no full scan
    assert "INTERVAL '" in system_src and "days'" in system_src, (
        "P3-FULL-TABLE-SCAN-HEALTH: las queries del health endpoint deben "
        "usar `INTERVAL '<N> days'` para evitar full table scan."
    )


def test_system_health_profile_query_has_limit(system_src: str):
    """La query de `user_profiles` debe tener LIMIT (no scan completo)."""
    # Aislar la sección del handler get_system_health
    handler_re = re.compile(
        r'def\s+get_system_health\([^)]*\)[\s\S]+?(?=^def\s|\Z)',
        re.MULTILINE,
    )
    m = handler_re.search(system_src)
    assert m is not None, "Handler get_system_health no encontrado."
    body = m.group(0)
    # Patrón legacy prohibido: SELECT health_profile ... FROM user_profiles
    # sin LIMIT.
    assert "FROM user_profiles" in body
    # El LIMIT debe estar presente cerca de la query
    has_limit = re.search(
        r"FROM\s+user_profiles[\s\S]{0,500}LIMIT\s+\{profile_sample_limit\}",
        body,
    )
    assert has_limit is not None, (
        "P3-FULL-TABLE-SCAN-HEALTH: la query de `user_profiles` en "
        "`/system/health` no tiene `LIMIT {profile_sample_limit}`. Sin "
        "límite, dashboards polleando consumen DB CPU lineal en N usuarios."
    )


# ============================================================================
# Section 6: P3-NOTIFICATIONS-RATE-LIMIT
# ============================================================================

def test_notifications_uses_rate_limiter(notifications_src: str):
    """`/subscribe` y `/unsubscribe` deben usar `RateLimiter`, no
    `Depends(get_verified_user_id)` directo."""
    assert "from rate_limiter import RateLimiter" in notifications_src, (
        "P3-NOTIFICATIONS-RATE-LIMIT: `RateLimiter` no importado en "
        "notifications.py."
    )
    assert "_PUSH_SUBSCRIBE_LIMITER" in notifications_src, (
        "P3-NOTIFICATIONS-RATE-LIMIT: instancia "
        "`_PUSH_SUBSCRIBE_LIMITER` no definida."
    )
    assert "_PUSH_UNSUBSCRIBE_LIMITER" in notifications_src, (
        "P3-NOTIFICATIONS-RATE-LIMIT: instancia "
        "`_PUSH_UNSUBSCRIBE_LIMITER` no definida."
    )


def test_notifications_handlers_use_limiters(notifications_src: str):
    """Los dos handlers deben llevar `Depends(_PUSH_*_LIMITER)`, no
    `Depends(get_verified_user_id)`."""
    # /subscribe handler. [P3-BACKEND-AUDIT · 2026-06-01] `subscribe_push`
    # pasó a `def` plano (cuerpo psycopg síncrono — async bloquearía el
    # event loop); el `async` es opcional en el regex porque la propiedad
    # protegida es el LIMITER en la signature, no la corrutina.
    subscribe_match = re.search(
        r'@router\.post\("/subscribe"\)\s*\n\s*(?:async\s+)?def\s+subscribe_push\([^)]*\)',
        notifications_src,
        re.DOTALL,
    )
    assert subscribe_match is not None, "Handler /subscribe no encontrado."
    assert "_PUSH_SUBSCRIBE_LIMITER" in subscribe_match.group(0), (
        "P3-NOTIFICATIONS-RATE-LIMIT: handler /subscribe no usa "
        "`_PUSH_SUBSCRIBE_LIMITER`. Sin throttle, atacante autenticado "
        "puede llenar `push_subscriptions`."
    )

    # /unsubscribe handler
    unsubscribe_match = re.search(
        r'@router\.delete\("/unsubscribe"\)\s*\n\s*(?:async\s+)?def\s+unsubscribe_push\([^)]*\)',
        notifications_src,
        re.DOTALL,
    )
    assert unsubscribe_match is not None, "Handler /unsubscribe no encontrado."
    assert "_PUSH_UNSUBSCRIBE_LIMITER" in unsubscribe_match.group(0), (
        "P3-NOTIFICATIONS-RATE-LIMIT: handler /unsubscribe no usa "
        "`_PUSH_UNSUBSCRIBE_LIMITER`."
    )
