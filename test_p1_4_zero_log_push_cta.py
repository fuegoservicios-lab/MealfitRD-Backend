"""
Tests P1-4: Push notification del zero-log pause incluye CTA "Continuar sin registrar".

Antes: el push solo invitaba a loguear comidas → URL=/dashboard. El usuario que
no quería/podía loguear quedaba esperando 4-24h sin saber que existía la opción
auto_proxy (PUT /api/diary/preferences/logging) ya expuesta en el banner del
frontend vía /api/blocked-reasons.

Cambio P1-4: cuando logging_preference='manual', el push ahora menciona
explícitamente la alternativa "Continuar sin registrar" y deeplinkea al diario
(CHUNK_ZERO_LOG_DEEPLINK) donde el banner muestra el toggle. Si el usuario ya
está en auto_proxy, el CTA no aparece.

La lógica vive en `_build_zero_log_push_payload(consecutive_zero_log_chunks,
logging_preference)`. Tests unitarios directos del helper.
"""
import pytest
from cron_tasks import _build_zero_log_push_payload


def test_p1_4_manual_normal_includes_optout_cta():
    """logging_preference='manual' + zero-log normal (<3 consec) → CTA en body, deeplink al diario."""
    import constants
    payload = _build_zero_log_push_payload(
        consecutive_zero_log_chunks=1,
        logging_preference="manual",
    )
    assert payload["title"] == "Loguea tus comidas para continuar"
    assert "continuar sin registrar" in payload["body"].lower(), (
        f"Body debe mencionar 'Continuar sin registrar'; got {payload['body']!r}"
    )
    assert payload["url"] == constants.CHUNK_ZERO_LOG_DEEPLINK


def test_p1_4_manual_degraded_includes_optout_cta():
    """logging_preference='manual' + degraded (≥3 consec) → mensaje degraded con CTA secundario."""
    import constants
    payload = _build_zero_log_push_payload(
        consecutive_zero_log_chunks=3,
        logging_preference="manual",
    )
    assert payload["title"] == "Tu plan se está generando sin tu feedback"
    assert "continuar sin registrar" in payload["body"].lower()
    assert payload["url"] == constants.CHUNK_ZERO_LOG_DEEPLINK


def test_p1_4_auto_proxy_normal_omits_cta():
    """logging_preference='auto_proxy' → NO menciona la opción, URL=/dashboard."""
    payload = _build_zero_log_push_payload(
        consecutive_zero_log_chunks=1,
        logging_preference="auto_proxy",
    )
    assert payload["title"] == "Loguea tus comidas para continuar"
    assert "continuar sin registrar" not in payload["body"].lower()
    assert payload["url"] == "/dashboard"


def test_p1_4_auto_proxy_degraded_omits_cta():
    """logging_preference='auto_proxy' + degraded → tampoco CTA, URL=/dashboard."""
    payload = _build_zero_log_push_payload(
        consecutive_zero_log_chunks=5,
        logging_preference="auto_proxy",
    )
    assert payload["title"] == "Tu plan se está generando sin tu feedback"
    assert "continuar sin registrar" not in payload["body"].lower()
    assert payload["url"] == "/dashboard"


def test_p1_4_unknown_preference_treated_as_non_manual():
    """logging_preference desconocido (e.g., None, 'other') → tratado como NO manual,
    sin CTA, URL=/dashboard. Solo 'manual' explícito ofrece el CTA."""
    payload = _build_zero_log_push_payload(
        consecutive_zero_log_chunks=1,
        logging_preference="unknown_pref",
    )
    assert "continuar sin registrar" not in payload["body"].lower()
    assert payload["url"] == "/dashboard"


def test_p1_4_zero_consecutive_falls_in_normal_branch():
    """consecutive_zero_log_chunks=0 cae en mensaje normal (no degraded)."""
    payload = _build_zero_log_push_payload(
        consecutive_zero_log_chunks=0,
        logging_preference="manual",
    )
    assert payload["title"] == "Loguea tus comidas para continuar"


def test_p1_4_consecutive_field_handles_none_gracefully():
    """Si consecutive_zero_log_chunks=None (defensivo), no crashea."""
    payload = _build_zero_log_push_payload(
        consecutive_zero_log_chunks=None,
        logging_preference="manual",
    )
    # None → 0 → normal branch
    assert payload["title"] == "Loguea tus comidas para continuar"


def test_p1_4_default_logging_preference_is_manual():
    """El default de logging_preference es 'manual' (CTA presente)."""
    payload = _build_zero_log_push_payload(
        consecutive_zero_log_chunks=1,
        # sin logging_preference → default
    )
    assert "continuar sin registrar" in payload["body"].lower()


def test_p1_4_deeplink_honors_env_override(monkeypatch):
    """CHUNK_ZERO_LOG_DEEPLINK debe respetar override de env var."""
    monkeypatch.setenv("CHUNK_ZERO_LOG_DEEPLINK", "mealfit://diario/zero-log")
    import importlib
    import constants
    importlib.reload(constants)
    assert constants.CHUNK_ZERO_LOG_DEEPLINK == "mealfit://diario/zero-log"
    # Restaurar default
    monkeypatch.delenv("CHUNK_ZERO_LOG_DEEPLINK", raising=False)
    importlib.reload(constants)


def test_p1_4_constant_default_points_to_diary():
    """Default de CHUNK_ZERO_LOG_DEEPLINK apunta a una ruta del diario."""
    import constants
    assert hasattr(constants, "CHUNK_ZERO_LOG_DEEPLINK")
    val = constants.CHUNK_ZERO_LOG_DEEPLINK
    assert "diario" in val.lower() or val.startswith("mealfit://"), (
        f"Default debe deeplinkar al diario; got {val!r}"
    )
