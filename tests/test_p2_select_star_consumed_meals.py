"""[P2-SELECT-STAR-CONSUMED-MEALS · 2026-05-15] Test bundle marker (último
alfabético de la Fase 2 audit 2026-05-15 noche). Cubre los 10 P2 fixes:

    - P2-AGENTPAGE-ERROR-SENTRY: AgentPage error catches → Sentry.captureException
    - P2-COHERENCE-GUARD-PERF: duration_ms + alert si > slow threshold
    - P2-DIARY-NO-PYDANTIC: Pydantic models en endpoints diary
    - P2-DIARY-PROGRESS-RATELIMIT: RateLimiter en /api/diary/progress
    - P2-INVENTORY-STATEMENT-TIMEOUT: SET LOCAL statement_timeout en tx inventory
    - P2-LLM-TIMEOUT-PIPELINE-METRICS: helper aplicado a self-critique + surgical-regen
    - P2-LOCALSTORAGE-GETITEM-DEFENSIVE: safeLocalStorageGet en initializer
    - P2-LOCALSTORAGE-REMOVEITEM: safeLocalStorageRemove en logout/reset
    - P2-RATELIMITER-BUCKET-METRICS: emit cardinalidad + alert saturation
    - P2-SELECT-STAR-CONSUMED-MEALS: columnas explícitas en consumed_meals

Cada bloque es un test parser-based contra source-de-prod. El bundle entero
recibe un marker único (alfabético-último); este archivo cierra el cross-link
P2-HIST-AUDIT-14.

Tooltip-anchor: P2-SELECT-STAR-CONSUMED-MEALS-BUNDLE
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_FRONTEND = _BACKEND.parent / "frontend"


# ===========================================================================
# P2-SELECT-STAR-CONSUMED-MEALS (marker bundle)
# ===========================================================================
class TestSelectStarConsumedMeals:
    @pytest.fixture(scope="class")
    def db_facts_src(self) -> str:
        return (_BACKEND / "db_facts.py").read_text(encoding="utf-8")

    def test_consumed_meals_select_explicit_columns(self, db_facts_src: str):
        """`get_consumed_meals_today` y `get_consumed_meals_since` deben
        especificar columnas explícitas (no `*`)."""
        # No deben quedar `SELECT * FROM consumed_meals` en el módulo.
        assert "SELECT * FROM consumed_meals" not in db_facts_src, (
            "P2-SELECT-STAR-CONSUMED-MEALS regresión: `SELECT *` reapareció."
        )
        # Tampoco `.select("*").eq("user_id"... gte("consumed_at"...)`.
        assert not re.search(
            r'\.select\("\*"\)[\s\S]{0,200}consumed_at',
            db_facts_src,
        ), (
            "P2-SELECT-STAR-CONSUMED-MEALS regresión: `.select(\"*\")` en path "
            "Supabase de consumed_meals — fallback no-pool perdió la lista de columnas."
        )

    def test_marker_present(self, db_facts_src: str):
        assert "P2-SELECT-STAR-CONSUMED-MEALS" in db_facts_src, (
            "Marker bundle ausente — si renombras, actualizar test cross-link."
        )


# ===========================================================================
# P2-DIARY-NO-PYDANTIC + P2-DIARY-PROGRESS-RATELIMIT
# ===========================================================================
class TestDiaryPydanticAndRateLimit:
    @pytest.fixture(scope="class")
    def diary_src(self) -> str:
        return (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")

    def test_pydantic_models_defined(self, diary_src: str):
        assert re.search(r"class ConsumedMealRequest\(BaseModel\)", diary_src), (
            "P2-DIARY-NO-PYDANTIC: `ConsumedMealRequest(BaseModel)` no definido."
        )
        assert re.search(r"class ProgressRequest\(BaseModel\)", diary_src), (
            "P2-DIARY-NO-PYDANTIC: `ProgressRequest(BaseModel)` no definido."
        )

    def test_consumed_endpoint_uses_pydantic(self, diary_src: str):
        assert re.search(
            r"api_log_consumed_meal\([\s\S]+?payload:\s*ConsumedMealRequest",
            diary_src,
        ), (
            "P2-DIARY-NO-PYDANTIC: `api_log_consumed_meal` ya no acepta "
            "`ConsumedMealRequest` — habrá vuelto a `data: dict = Body(...)`."
        )

    def test_progress_endpoint_uses_pydantic_and_limiter(self, diary_src: str):
        assert re.search(
            r"api_log_progress\([\s\S]+?payload:\s*ProgressRequest",
            diary_src,
        ), (
            "P2-DIARY-NO-PYDANTIC: `api_log_progress` ya no acepta `ProgressRequest`."
        )
        assert re.search(
            r"Depends\(\s*_PROGRESS_LIMITER\s*\)",
            diary_src,
        ), (
            "P2-DIARY-PROGRESS-RATELIMIT: el endpoint /progress ya no usa "
            "`Depends(_PROGRESS_LIMITER)` — rate-limit removido."
        )

    def test_progress_limiter_instantiated(self, diary_src: str):
        assert re.search(
            r"_PROGRESS_LIMITER\s*=\s*RateLimiter\(\s*max_calls\s*=\s*\d+,\s*period_seconds\s*=\s*\d+\s*\)",
            diary_src,
        ), (
            "P2-DIARY-PROGRESS-RATELIMIT: `_PROGRESS_LIMITER` no se instancia "
            "con `RateLimiter(max_calls=..., period_seconds=...)`."
        )

    def test_finite_validator_present(self, diary_src: str):
        """Los validadores de NaN/Infinity deben estar en ambos models."""
        assert "math.isfinite" in diary_src, (
            "P2-DIARY-NO-PYDANTIC: validator para finitud (math.isfinite) ausente."
        )


# ===========================================================================
# P2-LOCALSTORAGE-REMOVEITEM + P2-LOCALSTORAGE-GETITEM-DEFENSIVE
# ===========================================================================
class TestLocalStorageDefensive:
    @pytest.fixture(scope="class")
    def assessment_src(self) -> str:
        return (_FRONTEND / "src" / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")

    @pytest.fixture(scope="class")
    def settings_src(self) -> str:
        return (_FRONTEND / "src" / "pages" / "Settings.jsx").read_text(encoding="utf-8")

    @pytest.fixture(scope="class")
    def pantry_src(self) -> str:
        return (_FRONTEND / "src" / "pages" / "Pantry.jsx").read_text(encoding="utf-8")

    def test_assessment_imports_remove_and_get(self, assessment_src: str):
        assert re.search(
            r"import\s*\{[^}]*safeLocalStorageRemove[^}]*\}\s*from\s*'\.\./utils/safeLocalStorage'",
            assessment_src,
        ), (
            "P2-LOCALSTORAGE-REMOVEITEM: `safeLocalStorageRemove` no se importa "
            "en AssessmentContext.jsx."
        )
        assert re.search(
            r"import\s*\{[^}]*safeLocalStorageGet[^}]*\}\s*from\s*'\.\./utils/safeLocalStorage'",
            assessment_src,
        ), (
            "P2-LOCALSTORAGE-GETITEM-DEFENSIVE: `safeLocalStorageGet` no se importa."
        )

    def test_assessment_initializer_uses_safe_get(self, assessment_src: str):
        """Las 3 lecturas críticas del initializer del provider (savedPlan,
        savedForm, savedLikes) deben usar safeLocalStorageGet."""
        for key in ("mealfit_plan", "mealfit_form", "mealfit_likes"):
            assert re.search(
                rf"safeLocalStorageGet\(\s*['\"]{key}['\"]",
                assessment_src,
            ), (
                f"P2-LOCALSTORAGE-GETITEM-DEFENSIVE: `safeLocalStorageGet('{key}', ...)` "
                f"no aparece — el initializer aún hace `localStorage.getItem` raw."
            )

    def test_assessment_logout_uses_safe_remove(self, assessment_src: str):
        """Las 8 llaves del logout/reset flow deben usar safeLocalStorageRemove."""
        critical_keys = (
            "mealfit_form_secure", "mealfit_plan", "mealfit_likes",
            "mealfit_user_id", "mealfit_guest_session",
            "mealfit_guest_sessions_list", "mealfit_current_session",
            "mealfit_dislikes",
        )
        for key in critical_keys:
            assert re.search(
                rf"safeLocalStorageRemove\(\s*['\"]{key}['\"]",
                assessment_src,
            ), (
                f"P2-LOCALSTORAGE-REMOVEITEM: `safeLocalStorageRemove('{key}')` "
                f"ausente — flow logout/reset no protegido contra iOS Private Mode."
            )

    def test_settings_uses_safe_remove(self, settings_src: str):
        """El handler resetPreferences debe limpiar via safeLocalStorageRemove."""
        for key in ("mealfit_disabled_ingredients", "mealfit_plan", "mealfit_likes", "mealfit_dislikes"):
            assert re.search(
                rf"safeLocalStorageRemove\(\s*['\"]{key}['\"]",
                settings_src,
            ), (
                f"P2-LOCALSTORAGE-REMOVEITEM: Settings.jsx::handleResetPreferences "
                f"no usa safeLocalStorageRemove para `{key}`."
            )

    def test_pantry_uses_safe_get(self, pantry_src: str):
        assert re.search(
            r"safeLocalStorageGet\(\s*['\"]mealfit_disabled_ingredients['\"]",
            pantry_src,
        ), (
            "P2-LOCALSTORAGE-GETITEM-DEFENSIVE: Pantry.jsx mount handler "
            "no usa safeLocalStorageGet para `mealfit_disabled_ingredients`."
        )


# ===========================================================================
# P2-AGENTPAGE-ERROR-SENTRY
# ===========================================================================
class TestAgentPageSentryCapture:
    @pytest.fixture(scope="class")
    def agent_page_src(self) -> str:
        return (_FRONTEND / "src" / "pages" / "AgentPage.jsx").read_text(encoding="utf-8")

    def test_sentry_imported(self, agent_page_src: str):
        assert re.search(
            r"import\s+\*\s+as\s+Sentry\s+from\s+['\"]@sentry/react['\"]",
            agent_page_src,
        ), (
            "P2-AGENTPAGE-ERROR-SENTRY: `import * as Sentry from '@sentry/react'` no aparece."
        )

    def test_capture_helper_defined(self, agent_page_src: str):
        assert "_captureAgentPageException" in agent_page_src, (
            "P2-AGENTPAGE-ERROR-SENTRY: helper `_captureAgentPageException` no definido."
        )
        # Helper debe llamar a Sentry.captureException con tags.
        assert re.search(
            r"Sentry\.captureException\([\s\S]{0,200}tags:\s*\{\s*component:\s*['\"]AgentPage['\"]",
            agent_page_src,
        ), (
            "P2-AGENTPAGE-ERROR-SENTRY: el helper no llama a "
            "`Sentry.captureException(err, { tags: { component: 'AgentPage', ... } })`."
        )

    def test_three_error_handlers_use_capture(self, agent_page_src: str):
        """fetchSessions, fetchSessionMessages, deleteChat son los 3 paths
        que el audit identificó. Cada uno debe invocar el helper."""
        for action in ("fetchSessions", "fetchSessionMessages", "deleteChat"):
            assert re.search(
                rf"_captureAgentPageException\([\s\S]{{0,200}}action:\s*['\"]{action}['\"]",
                agent_page_src,
            ), (
                f"P2-AGENTPAGE-ERROR-SENTRY: action `{action}` no invoca al helper."
            )


# ===========================================================================
# P2-RATELIMITER-BUCKET-METRICS
# ===========================================================================
class TestRateLimiterBucketMetrics:
    @pytest.fixture(scope="class")
    def rl_src(self) -> str:
        return (_BACKEND / "rate_limiter.py").read_text(encoding="utf-8")

    def test_cleanup_emits_metric(self, rl_src: str):
        """El bloque cleanup debe invocar el helper de telemetría."""
        assert "_emit_rl_cleanup_metric" in rl_src, (
            "P2-RATELIMITER-BUCKET-METRICS: helper `_emit_rl_cleanup_metric` ausente."
        )
        assert "rate_limiter_cleanup" in rl_src, (
            "P2-RATELIMITER-BUCKET-METRICS: node `rate_limiter_cleanup` ausente."
        )

    def test_saturation_alert_emits(self, rl_src: str):
        assert "_emit_rl_saturation_alert" in rl_src, (
            "P2-RATELIMITER-BUCKET-METRICS: helper `_emit_rl_saturation_alert` ausente."
        )
        assert "rate_limiter_bucket_saturation" in rl_src, (
            "P2-RATELIMITER-BUCKET-METRICS: alert_key canónica ausente."
        )

    def test_knob_referenced(self, rl_src: str):
        assert "MEALFIT_RATE_LIMITER_BUCKET_LIMIT_WARN" in rl_src, (
            "P2-RATELIMITER-BUCKET-METRICS: knob no se lee desde env."
        )


# ===========================================================================
# P2-INVENTORY-STATEMENT-TIMEOUT
# ===========================================================================
class TestInventoryStatementTimeout:
    @pytest.fixture(scope="class")
    def db_inv_src(self) -> str:
        return (_BACKEND / "db_inventory.py").read_text(encoding="utf-8")

    def test_set_local_statement_timeout_present(self, db_inv_src: str):
        """La transacción de liberación de reservas debe prepend `SET LOCAL
        statement_timeout`."""
        assert re.search(
            r"SET LOCAL statement_timeout",
            db_inv_src,
        ), (
            "P2-INVENTORY-STATEMENT-TIMEOUT: `SET LOCAL statement_timeout` "
            "no aparece — los UPDATEs caen al timeout global Supavisor (~60s)."
        )

    def test_knob_referenced(self, db_inv_src: str):
        assert "MEALFIT_INVENTORY_UPDATE_STMT_TIMEOUT_MS" in db_inv_src, (
            "P2-INVENTORY-STATEMENT-TIMEOUT: knob no se lee."
        )

    def test_env_helper_defined(self, db_inv_src: str):
        assert "_env_int_safe_inventory" in db_inv_src, (
            "P2-INVENTORY-STATEMENT-TIMEOUT: helper `_env_int_safe_inventory` no definido."
        )


# ===========================================================================
# P2-COHERENCE-GUARD-PERF
# ===========================================================================
class TestCoherenceGuardPerf:
    @pytest.fixture(scope="class")
    def sc_src(self) -> str:
        return (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")

    def test_emit_helper_defined(self, sc_src: str):
        assert "_emit_coherence_guard_metric" in sc_src, (
            "P2-COHERENCE-GUARD-PERF: helper `_emit_coherence_guard_metric` no definido."
        )

    def test_helper_inserts_pipeline_metric(self, sc_src: str):
        anchor = re.search(
            r"def _emit_coherence_guard_metric\([\s\S]+?(?=\ndef |\nclass )",
            sc_src,
        )
        assert anchor is not None, "helper body not found"
        body = anchor.group(0)
        assert "INSERT INTO pipeline_metrics" in body, (
            "P2-COHERENCE-GUARD-PERF: helper no inserta a pipeline_metrics."
        )
        assert "coherence_guard_validation" in body, (
            "P2-COHERENCE-GUARD-PERF: node canónico ausente."
        )

    def test_slow_threshold_knob(self, sc_src: str):
        assert "MEALFIT_COHERENCE_GUARD_SLOW_MS" in sc_src, (
            "P2-COHERENCE-GUARD-PERF: knob de slow-threshold no se lee."
        )

    def test_guard_calls_emit_on_exit(self, sc_src: str):
        """`run_shopping_coherence_guard` debe llamar al emit helper en al
        menos 2 paths (off-mode + normal exit)."""
        guard_match = re.search(
            r"def run_shopping_coherence_guard\([\s\S]+?(?=\ndef |\nclass )",
            sc_src,
        )
        assert guard_match is not None
        body = guard_match.group(0)
        emit_count = body.count("_emit_coherence_guard_metric(")
        assert emit_count >= 2, (
            f"P2-COHERENCE-GUARD-PERF: el guard solo invoca _emit en {emit_count} "
            f"paths — esperados ≥2 (off-mode + exit normal)."
        )


# ===========================================================================
# P2-LLM-TIMEOUT-PIPELINE-METRICS
# ===========================================================================
class TestLLMTimeoutPipelineMetrics:
    @pytest.fixture(scope="class")
    def orch_src(self) -> str:
        return (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")

    def test_self_critique_emits_metric(self, orch_src: str):
        assert "self_critique_correction_timeout" in orch_src, (
            "P2-LLM-TIMEOUT-PIPELINE-METRICS: node `self_critique_correction_timeout` "
            "ausente — el TimeoutError de self-critique correction no se telemetría."
        )

    def test_surgical_regen_emits_metric(self, orch_src: str):
        assert "surgical_regen_timeout" in orch_src, (
            "P2-LLM-TIMEOUT-PIPELINE-METRICS: node `surgical_regen_timeout` "
            "ausente — el segundo TimeoutError (marker-regen) no se discrimina."
        )

    def test_helper_invoked_with_attempt_field(self, orch_src: str):
        """El surgical-regen debe enviar `attempt=2` en metadata para
        distinguirlo del primer pass."""
        assert re.search(
            r"\"attempt\":\s*2",
            orch_src,
        ), (
            "P2-LLM-TIMEOUT-PIPELINE-METRICS: attempt=2 en surgical_regen_timeout "
            "metadata ausente — sin él no se diferencia first-pass vs retry."
        )
