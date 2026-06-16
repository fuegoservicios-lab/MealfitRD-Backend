"""[P1-RETRY-CLASSIFY] Tests para la clasificación de HIGH severity en
`should_retry` y la decisión de permitir retry vs abortar.

Bug pre-fix (corrida 2026-05-05 01:49):
  Plan rechazado HIGH al primer intento por:
    - "Día 1 omitió múltiples proteínas clave asignadas: ['atún', 'huevos']"
    - "Repetición excesiva de pechuga de pavo procesada"
    - "Falta de variedad en las fuentes de proteína"
  Lógica original `if severity == "high": return "end"` abortaba SIEMPRE
  sin retry. Resultado: plan entregado con disclaimer (transparency banner)
  cuando un retry probablemente lo habría arreglado (skeleton fidelity es
  fallo de adherencia del LLM al prompt, no contextual).

Fix:
  Helper `_classify_high_severity(reasons)` distingue:
  - **contextual**: despensa, alergia, condición médica, intolerancia →
    regenerar NO ayuda (las restricciones del usuario no cambian).
    `should_retry` aborta como antes.
  - **regenerable**: skeleton fidelity, repetición, variedad, recipe
    coherence → otro intento del LLM puede arreglarlo. `should_retry`
    cae al check de attempts/budget como cualquier minor.

Cobertura:
  - Clasificador con casos canónicos (regenerable + contextual)
  - Edge cases (vacío, mixed, keywords parciales)
  - should_retry con HIGH regenerable → retry permitido
  - should_retry con HIGH contextual → end (sin cambio vs antes)
  - should_retry con HIGH regenerable pero attempt cap → end
  - should_retry con CRITICAL → end (no cambio)
  - Sinergia con P0-PIPE-1: snapshot del intento 1 sigue funcionando
"""
import time

import pytest

from graph_orchestrator import (
    _HIGH_SEVERITY_CONTEXTUAL_KEYWORDS,
    _classify_high_severity,
    should_retry,
    MAX_ATTEMPTS,
)


# ---------------------------------------------------------------------------
# 1. Clasificador
# ---------------------------------------------------------------------------
class TestClassifyHighSeverity:
    @pytest.mark.parametrize("reason", [
        "Día 1 omitió múltiples proteínas clave asignadas: ['atún', 'huevos']",
        "Repetición excesiva de pechuga de pavo procesada",
        "Falta de variedad en las fuentes de proteína",
        "Día 2: receta indica pollo pero no hay ingrediente equivalente",
        "Elevada cantidad de huevos (9 unidades) en el plan",
        "Coherencia receta-ingredientes: pavo en instrucciones sin listar",
    ])
    def test_regenerable_classification(self, reason):
        """Estos rechazos son fallos de calidad del LLM — retry puede ayudar."""
        assert _classify_high_severity([reason]) == "regenerable"

    @pytest.mark.parametrize("reason", [
        "Violación de despensa estricta: el plan usa pavo pero la despensa no lo tiene",
        "Plan contiene gluten pero el usuario es celíaco",
        "Ingrediente no disponible en inventario del usuario",
        "Alergia a maní detectada en plato 3",
        "Condición médica diabetes incompatible con plan alto en azúcar",
        "Intolerancia a la lactosa: plan contiene queso fresco",
    ])
    def test_contextual_classification(self, reason):
        """Estos rechazos son contextuales — retry NO ayuda (mismo contexto)."""
        assert _classify_high_severity([reason]) == "contextual"

    def test_empty_reasons_default_regenerable(self):
        """Sin razones explícitas, asumimos regenerable (más permisivo)."""
        assert _classify_high_severity([]) == "regenerable"
        assert _classify_high_severity([""]) == "regenerable"
        assert _classify_high_severity(None or []) == "regenerable"

    def test_mixed_reasons_contextual_wins(self):
        """Si AL MENOS una razón es contextual, clasificar como contextual.
        Razonamiento: si el plan viola despensa AND tiene problema de
        variedad, regenerar puede arreglar la variedad pero NO la despensa.
        Mejor abortar y entregar plan marcado."""
        reasons = [
            "Falta de variedad en las fuentes de proteína",
            "Plan usa ingredientes fuera del inventario del usuario",
        ]
        assert _classify_high_severity(reasons) == "contextual"

    def test_keyword_substring_match(self):
        """El match es por substring case-insensitive — variantes con tilde,
        plurales, mayúsculas todas matchean."""
        for variant in ["DESPENSA ESTRICTA",
                        "el plan viola la despensa estricta del usuario",
                        "Alergia detectada", "ALÉRGENO presente"]:
            assert _classify_high_severity([variant]) == "contextual"

    def test_keyword_set_no_empty(self):
        """Sanity check: el set de keywords no debe estar vacío
        (regression guard si alguien lo limpia accidentalmente)."""
        assert len(_HIGH_SEVERITY_CONTEXTUAL_KEYWORDS) >= 5


# ---------------------------------------------------------------------------
# 2. should_retry decisions con la nueva clasificación
# ---------------------------------------------------------------------------
class TestShouldRetryHighRegenerable:
    def _base_state(self, **overrides):
        """Estado base con tiempo fresco para que el budget guard no aborte."""
        state = {
            "_rejection_severity": "high",
            "rejection_reasons": [],
            "review_passed": False,
            "attempt": 1,
            "pipeline_start": time.time(),  # acaba de empezar → mucho budget
            "form_data": {},
        }
        state.update(overrides)
        return state

    def test_high_regenerable_attempt1_permite_retry(self):
        """Caso del incidente 2026-05-05 01:49: HIGH por skeleton fidelity
        en intento #1 → debe permitir retry (no abortar)."""
        state = self._base_state(
            rejection_reasons=["Día 1 omitió múltiples proteínas clave asignadas"]
        )
        assert should_retry(state) == "retry"

    def test_high_contextual_attempt1_aborta(self):
        """HIGH contextual (despensa) NO debe retry — restricción inmutable."""
        state = self._base_state(
            rejection_reasons=["Violación de despensa estricta: ingrediente fuera de inventario"]
        )
        assert should_retry(state) == "end"

    def test_high_regenerable_attempt_max_aborta(self):
        """Aún si HIGH es regenerable, al alcanzar MAX_ATTEMPTS abortamos.

        [stale-parser fix · P1-LOW-SIGNAL-FALLBACK · 2026-05-21] MAX_ATTEMPTS
        subió de 2 → 3 (más intentos para usuarios de señal baja + banner
        explícito cuando se agotan). El test ahora ancla `attempt=MAX_ATTEMPTS`
        (el valor real, no el literal 2) para no volver a driftear con el knob.
        En `attempt < MAX_ATTEMPTS` el retry es correcto (quedan intentos);
        sólo en el cap se aborta — que es lo que este test verifica."""
        state = self._base_state(
            attempt=MAX_ATTEMPTS,
            rejection_reasons=["Falta de variedad en proteínas"],
        )
        assert should_retry(state) == "end"

    def test_high_alergia_aborta(self):
        """Alergia siempre = contextual. Aunque se clasifique como HIGH (no
        critical), no retry porque el usuario es alérgico."""
        state = self._base_state(
            rejection_reasons=["Plan contiene maní pero usuario tiene alergia"]
        )
        assert should_retry(state) == "end"

    def test_high_mixed_contextual_aborta(self):
        """Mixed reasons con AL MENOS UNA contextual → no retry."""
        state = self._base_state(rejection_reasons=[
            "Falta de variedad en proteínas",
            "Ingrediente fuera de inventario",
        ])
        assert should_retry(state) == "end"


class TestShouldRetryUnaffected:
    """Verificar que mi cambio NO regresa el comportamiento de otros paths."""

    def test_critical_aborta_siempre(self):
        """CRITICAL aborta sin importar regenerabilidad — política inmutable."""
        state = {
            "_rejection_severity": "critical",
            "rejection_reasons": ["Repetición excesiva (clasificado crítico por context)"],
            "review_passed": False,
            "attempt": 1,
            "pipeline_start": time.time(),
        }
        assert should_retry(state) == "end"

    def test_minor_attempt1_retry(self):
        """MINOR sigue permitiendo retry (sin cambio)."""
        state = {
            "_rejection_severity": "minor",
            "rejection_reasons": ["Pequeña observación"],
            "review_passed": False,
            "attempt": 1,
            "pipeline_start": time.time(),
        }
        assert should_retry(state) == "retry"

    def test_approved_end(self):
        """Si review_passed=True, end (sin cambio)."""
        state = {
            "_rejection_severity": None,
            "rejection_reasons": [],
            "review_passed": True,
            "attempt": 1,
            "pipeline_start": time.time(),
        }
        assert should_retry(state) == "end"


# ---------------------------------------------------------------------------
# 3. Repro del incidente 2026-05-05 01:49
# ---------------------------------------------------------------------------
def test_repro_incident_2026_05_05_skeleton_fidelity_now_retries():
    """Reproduce exactamente el escenario:
       - Intento #1 falla con HIGH:
         * Skeleton fidelity violation (Día 1 omitió atún + huevos)
         * Repetición excesiva pavo procesado
         * Falta de variedad
         * Elevada cantidad de huevos
       - Pre-fix: should_retry → "end" (sin retry, plan entregado roto)
       - Post-fix: should_retry → "retry" (LLM tiene 2da oportunidad)
    """
    state = {
        "_rejection_severity": "high",
        "rejection_reasons": [
            "Repetición excesiva de pechuga de pavo procesada (lonjas) en casi todas las comidas",
            "Falta de variedad en las fuentes de proteína",
            "Elevada cantidad de huevos (9 unidades) en el plan de 3 días",
            "Día 1 omitió múltiples proteínas clave asignadas: ['atún', 'huevos']",
        ],
        "review_passed": False,
        "attempt": 1,
        "pipeline_start": time.time(),  # fresh → budget OK
    }
    decision = should_retry(state)
    assert decision == "retry", (
        f"Pre-fix abortaba; post-fix debe permitir retry. Decisión: {decision}"
    )


def test_repro_high_contextual_unchanged():
    """Repro: HIGH causado por despensa estricta — esto SIGUE no haciendo
    retry (correctamente). El usuario debería actualizar su despensa."""
    state = {
        "_rejection_severity": "high",
        "rejection_reasons": [
            "Plan usa habichuelas rojas pero la despensa estricta no las tiene",
            "Ingrediente fuera de inventario disponible",
        ],
        "review_passed": False,
        "attempt": 1,
        "pipeline_start": time.time(),
    }
    assert should_retry(state) == "end"


# ---------------------------------------------------------------------------
# 4. Sinergia con P0-PIPE-1
# ---------------------------------------------------------------------------
def test_sinergia_p0_pipe_1_snapshot_sigue_funcionando_post_retry():
    """Si HIGH regenerable retry y el intento #2 termina peor:
    P0-PIPE-1 swap restaura intento #1 (snapshot ya guardado por
    review_plan_node). Verificación de invariante de integración."""
    from graph_orchestrator import _swap_to_best_attempt_if_better

    final_state = {
        "plan_result": {"days": [{"day": 1, "broken": True}]},
        "review_passed": False,
        "_rejection_severity": "high",
        "rejection_reasons": ["intento #2 falló otra vez"],
        "attempt": 2,
        # Snapshot del intento #1 (que también era HIGH pero distintos issues)
        "_best_attempt_plan": {"days": [{"day": 1, "best": True}]},
        "_best_attempt_severity": "minor",  # supongamos #1 fue minor
        "_best_attempt_reasons": ["intento #1 minor issue"],
        "_best_attempt_review_passed": False,
        "_best_attempt_number": 1,
    }
    swapped = _swap_to_best_attempt_if_better(final_state)
    assert swapped is True
    # El plan final debe ser el del intento #1 (mejor severidad)
    assert final_state["plan_result"]["days"][0].get("best") is True
