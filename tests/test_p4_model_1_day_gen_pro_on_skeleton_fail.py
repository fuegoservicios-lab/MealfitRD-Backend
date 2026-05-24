"""[P4-MODEL-1 / P4-MODEL-2] Tests para la escalación targeted del day
generator a Pro en retry cuando el rechazo previo es señal de baja
adherencia del modelo al skeleton.

Bug observable (corridas 2026-05-05 múltiples):
  El planner asigna proteínas distintas, pero Gemini Flash en day generator
  ignora el skeleton ~30% de las veces. Patrón cascading:
    1. Intento #1 Flash: omite proteínas asignadas → HIGH skeleton fidelity
    2. P1-RETRY-CLASSIFY permite retry (HIGH es regenerable)
    3. Intento #2 Flash: omite OTRAS proteínas → HIGH otra vez
    4. P0-PIPE-1 swap restaura intento #1 (mejor de los dos)
  Resultado: usuario recibe plan menos malo + disclaimer minor. Sub-óptimo.

Fix P4-MODEL-1:
  Helper `_route_model_for_day_generator(form_data, attempt, prev_rejection_reasons)`
  que escala al modelo Pro (`gemini-3.1-pro-preview`) cuando:
    - attempt > 1 (es un retry)
    - knob `MEALFIT_DAY_GEN_RETRY_USE_PRO` está habilitado (default True)
    - prev_rejection_reasons mencionan skeleton fidelity / "omitió proteínas"

  Pro tiene mejor adherencia (~95% vs ~70% Flash) a costo de 2.5× latencia
  y 17× costo por token. Targeted (solo cuando aplica) controla costo total.

Expansión P4-MODEL-2:
  El clasificador ahora también detecta "repetición excesiva" y "falta de
  variedad" como señales de adherencia. Son síntomas correlacionados del
  mismo modo de fallo (model laziness): cuando Flash no respeta el
  skeleton, frecuentemente también colapsa la variedad. Si el reviewer
  los reporta solos (sin mencionar skeleton fidelity literal) ANTES habrían
  caído a default Flash en retry — ahora escalan a Pro como deben.
  Rechazos ortogonales (despensa, alergia, recipe-ingredient mismatch)
  siguen quedando fuera — esos no se arreglan con un modelo más grande.

Cobertura:
  - `_is_skeleton_fidelity_rejection`: detección por keywords (literal + síntomas)
  - `_route_model_for_day_generator`: escenarios canónicos
  - Knob disable preserva Flash en retry
  - Sinergia con _route_model legacy (clinical complex sigue Pro)
  - Edge cases: razones vacías, mixed
"""
import os

import pytest

from graph_orchestrator import (
    _FLASH_MODEL_NAME,
    _PRO_MODEL_NAME,
    _is_skeleton_fidelity_rejection,
    _route_model_for_day_generator,
)


# ---------------------------------------------------------------------------
# 1. Detección de skeleton fidelity rejection
# ---------------------------------------------------------------------------
class TestIsSkeletonFidelityRejection:
    @pytest.mark.parametrize("reason", [
        "Día 1 omitió múltiples proteínas clave asignadas: ['atún', 'huevos']",
        "Día 3 omitió múltiples proteínas clave asignadas: ['queso mozzarella fresco', 'pavo fresco']",
        "Skeleton fidelity violation detectada",
        "skeleton fidelity issue en día 2",
        "El plan omitió proteínas asignadas por el planificador",
    ])
    def test_detects_skeleton_fidelity_literal(self, reason):
        """[P4-MODEL-1] Razones literales de skeleton fidelity."""
        assert _is_skeleton_fidelity_rejection([reason]) is True

    @pytest.mark.parametrize("reason", [
        "Repetición excesiva de pechuga de pavo procesada",
        "Repetición excesiva de huevos en almuerzos",
        "Falta de variedad en las fuentes de proteína",
        "Falta de variedad: 4 días con pollo",
        # Case-insensitive y con/sin tilde
        "REPETICIÓN EXCESIVA de carbohidratos",
        "repeticion excesiva de almidones",
    ])
    def test_detects_model_laziness_symptoms(self, reason):
        """[P4-MODEL-2] Síntomas correlacionados (model laziness) que también
        mejoran al escalar a Pro: repetición excesiva, falta de variedad.
        Antes estos rechazos quedaban en Flash en retry; ahora escalan."""
        assert _is_skeleton_fidelity_rejection([reason]) is True

    @pytest.mark.parametrize("reason", [
        "Día 2: receta indica pollo pero no hay ingrediente equivalente listado",
        "Violación de despensa estricta",
        "Plan contiene gluten para usuario celíaco",
        "Elevada cantidad de huevos (9 unidades)",
        # Edge: mención de variedad SIN ser señal de laziness
        "Buena variedad pero falta proteína magra el día 3",
    ])
    def test_does_not_detect_orthogonal_rejections(self, reason):
        """Rechazos ortogonales NO escalan a Pro — un modelo más grande
        no arregla violaciones de despensa ni alergias ni recipe-ingredient
        mismatch (esos son problemas de prompt o de datos)."""
        assert _is_skeleton_fidelity_rejection([reason]) is False

    def test_empty_reasons(self):
        assert _is_skeleton_fidelity_rejection([]) is False
        assert _is_skeleton_fidelity_rejection(None) is False

    def test_mixed_reasons_finds_skeleton(self):
        """Si AL MENOS una razón es skeleton fidelity, detecta True."""
        reasons = [
            "Repetición de pavo",
            "Día 1 omitió múltiples proteínas clave asignadas",
        ]
        assert _is_skeleton_fidelity_rejection(reasons) is True


# ---------------------------------------------------------------------------
# 2. Routing del day generator
# ---------------------------------------------------------------------------
class TestRouteModelForDayGenerator:
    def test_attempt_1_uses_default_routing(self):
        """En attempt #1 SIEMPRE usa default routing (Flash o Pro según
        clinical complexity), NUNCA upgrade por skeleton fidelity."""
        form = {"mainGoal": "gain_muscle"}
        # Aún si por alguna razón hay rejection_reasons (no debería en #1):
        result = _route_model_for_day_generator(
            form, attempt=1,
            prev_rejection_reasons=["skeleton fidelity violation"],
        )
        assert result == _FLASH_MODEL_NAME

    def test_retry_with_skeleton_fidelity_escalates_to_pro(self):
        """[P4-MODEL-1] Caso clave: retry + skeleton fidelity literal → Pro."""
        form = {"mainGoal": "gain_muscle"}
        result = _route_model_for_day_generator(
            form, attempt=2,
            prev_rejection_reasons=[
                "Día 1 omitió múltiples proteínas clave asignadas: ['atún', 'huevos']"
            ],
        )
        assert result == _PRO_MODEL_NAME

    def test_retry_with_repetition_excessive_escalates_to_pro(self):
        """[P4-MODEL-2] Síntoma de laziness solo (sin skeleton fidelity
        literal) también escala a Pro — antes quedaba en Flash."""
        form = {"mainGoal": "gain_muscle"}
        result = _route_model_for_day_generator(
            form, attempt=2,
            prev_rejection_reasons=[
                "Repetición excesiva de pechuga de pavo procesada en 4 días"
            ],
        )
        assert result == _PRO_MODEL_NAME

    def test_retry_with_lack_of_variety_escalates_to_pro(self):
        """[P4-MODEL-2] 'Falta de variedad' como única razón también escala."""
        form = {"mainGoal": "gain_muscle"}
        result = _route_model_for_day_generator(
            form, attempt=2,
            prev_rejection_reasons=[
                "Falta de variedad en las fuentes de proteína"
            ],
        )
        assert result == _PRO_MODEL_NAME

    def test_retry_without_adherence_signal_stays_default(self):
        """Retry pero NO por señal de adherencia (skeleton fidelity literal
        ni síntoma de laziness) → default routing. Razones ortogonales
        como despensa o gluten NO ameritan escalar a Pro."""
        form = {"mainGoal": "gain_muscle"}
        result = _route_model_for_day_generator(
            form, attempt=2,
            prev_rejection_reasons=["Violación de despensa estricta en día 2"],
        )
        # Default = Flash para perfil easy gain_muscle
        assert result == _FLASH_MODEL_NAME

    def test_retry_no_reasons_stays_default(self):
        form = {"mainGoal": "gain_muscle"}
        result = _route_model_for_day_generator(
            form, attempt=2, prev_rejection_reasons=[],
        )
        assert result == _FLASH_MODEL_NAME

    def test_clinical_complex_attempt_1_uses_pro(self):
        """Sanity: el routing default escala a Pro cuando hay condiciones
        médicas. Esta lógica no es del fix nuevo; verificamos no regresión."""
        form = {
            "mainGoal": "gain_muscle",
            "medicalConditions": ["diabetes tipo 2"],
        }
        result = _route_model_for_day_generator(
            form, attempt=1, prev_rejection_reasons=[],
        )
        assert result == _PRO_MODEL_NAME


# ---------------------------------------------------------------------------
# 3. Knob env override
# ---------------------------------------------------------------------------
class TestEnvKnob:
    def test_knob_disabled_no_escalation(self, monkeypatch):
        """Si MEALFIT_DAY_GEN_RETRY_USE_PRO=0, el retry NUNCA escala a Pro
        aunque haya skeleton fidelity. Operadores pueden desactivar para
        controlar costos a riesgo de más fallos cascading."""
        # El knob se evalúa al import. Para test efectivo necesitamos
        # patchear el módulo-level constant.
        import graph_orchestrator
        original = graph_orchestrator.DAY_GEN_RETRY_USE_PRO
        monkeypatch.setattr(graph_orchestrator, "DAY_GEN_RETRY_USE_PRO", False)
        try:
            form = {"mainGoal": "gain_muscle"}
            result = graph_orchestrator._route_model_for_day_generator(
                form, attempt=2,
                prev_rejection_reasons=[
                    "Día 1 omitió múltiples proteínas clave asignadas"
                ],
            )
            # Sin knob, default routing (Flash for easy)
            assert result == _FLASH_MODEL_NAME
        finally:
            monkeypatch.setattr(graph_orchestrator, "DAY_GEN_RETRY_USE_PRO", original)


# ---------------------------------------------------------------------------
# 4. Repro corrida 2026-05-05 02:49 (cascading skeleton fidelity)
# ---------------------------------------------------------------------------
def test_repro_corrida_2026_05_05_02_49_cascading_failure():
    """Reproduce el escenario donde Flash falla skeleton fidelity 2 intentos
    seguidos:
       - Intento #1: Flash → minor (papas en bowl + pavo procesado)
       - Intento #2: Flash → HIGH ("Día 3 omitió queso mozzarella fresco, pavo fresco")
       - P0-PIPE-1 swap a #1, holistic 0.769

    Post-fix (P4-MODEL-1):
       - Intento #1: Flash (sin cambio)
       - Intento #2: AHORA Pro (skeleton fidelity es la causa) → adherencia
         mayor → probablemente APROBADO o al menos minor
       - P0-PIPE-1 swap NO necesario, holistic >0.9 esperado
    """
    form = {"mainGoal": "gain_muscle"}

    # Intento #1: sin razones previas, default routing
    model_attempt_1 = _route_model_for_day_generator(form, 1, [])
    assert model_attempt_1 == _FLASH_MODEL_NAME, (
        f"Intento #1 debe usar Flash default, recibido {model_attempt_1}"
    )

    # Intento #2: skeleton fidelity del intento #1 → Pro
    intento_1_rejection_reasons = [
        "Inconsistencia en la merienda del Día 2: papas en bowl de yogurt",
        "Día 1 omitió múltiples proteínas clave asignadas: ['atún', 'huevos']",
    ]
    model_attempt_2 = _route_model_for_day_generator(
        form, 2, intento_1_rejection_reasons,
    )
    assert model_attempt_2 == _PRO_MODEL_NAME, (
        f"Intento #2 con skeleton fidelity debe escalar a Pro, recibido {model_attempt_2}"
    )


# ---------------------------------------------------------------------------
# 5. Verificación de constantes de modelo
# ---------------------------------------------------------------------------
def test_model_constants_match_production():
    """Las constantes de nombre de modelo deben coincidir con los IDs
    reales que el resto del codebase usa (CB tracking, logs).

    [P1-FLASH-MODEL-GA · 2026-05-21] Flash actualizado a `gemini-3.5-flash` (GA).
    [P1-ALL-MODELS-GA · 2026-05-21] Pro también migrado a `gemini-3.5-flash`
    (eliminación total de modelos `*-preview` por riesgo deprecation).
    """
    assert _FLASH_MODEL_NAME == "gemini-3.5-flash"
    assert _PRO_MODEL_NAME == "gemini-3.5-flash"
