"""[P5-PROMPT-D] Tests para `build_minimal_correction_context`: versión
recortada del nutrition context para el corrector LLM.

Bug observable (corrida 2026-05-05 04:14):
  Día 3 corrigió en 143s con full nutrition_context — peligrosamente
  cerca del cap de 150s (P4-TIMEOUT-3). El bloque kinematics +
  metabolismo evolutivo era ruido para el corrector (que solo necesita
  preservar los targets duros).

Fix:
  Nuevo helper `build_minimal_correction_context(nutrition)` que emite
  solo calorías + macros. Aplicado en `self_critique_node` correction
  prompt y en `surgical_marker_regen_node`.

Cobertura:
  - Reducción real de tokens vs full context
  - Targets duros preservados (calories, protein/carbs/fats)
  - Ruido removido (kinematics, metabolismo evolutivo, body_fat_trend)
  - Tolerancia a `nutrition` sin kinematics (caso common: usuario nuevo)
  - Wiring: `_build_shared_context` expone `nutrition_context_minimal`
  - Wiring: ambos correctores usan el bloque mínimo
"""
import pytest


def _build_nutrition_with_kinematics():
    """Nutrition payload completo (caso usuario con historial)."""
    return {
        "bmr": 1499,
        "tdee": 1799,
        "target_calories": 2050,
        "goal_label": "Ganancia Muscular (Superávit 15%)",
        "macros": {
            "protein_g": 154,
            "carbs_g": 231,
            "fats_g": 57,
            "protein_str": "154g",
            "carbs_str": "231g",
            "fats_str": "57g",
        },
        "calculation_details": (
            "BMR Mifflin-St Jeor: 1499 kcal\n"
            "⚠️ [METABOLISMO EVOLUTIVO] Ajuste -50 kcal por adaptación\n"
        ),
        "kinematics": {
            "velocity_current": 0.05,
            "acceleration": 0.002,
            "body_fat_trend": -0.15,
            "is_losing_decelerating": False,
            "is_losing_accelerating": False,
            "is_gaining_decelerating": False,
            "is_gaining_accelerating": True,
        },
    }


def _build_nutrition_minimal():
    """Nutrition sin kinematics (caso usuario nuevo / guest)."""
    return {
        "bmr": 1499,
        "tdee": 1799,
        "target_calories": 2050,
        "goal_label": "Ganancia Muscular",
        "macros": {
            "protein_g": 154,
            "carbs_g": 231,
            "fats_g": 57,
            "protein_str": "154g",
            "carbs_str": "231g",
            "fats_str": "57g",
        },
        "calculation_details": "BMR Mifflin-St Jeor: 1499 kcal\n",
        # Sin "kinematics" key
    }


# ---------------------------------------------------------------------------
# 1. Reducción real de tokens
# ---------------------------------------------------------------------------
class TestReductionRatio:
    def test_minimal_context_shorter_than_full_with_kinematics(self):
        """Caso típico del bug: usuario con kinematics pobladas → bloque
        completo es ~2x más largo. Mínimo debe ser menor."""
        from prompts.plan_generator import (
            build_minimal_correction_context,
            build_nutrition_context,
        )
        nutrition = _build_nutrition_with_kinematics()
        full = build_nutrition_context(nutrition)
        minimal = build_minimal_correction_context(nutrition)
        assert len(minimal) < len(full), (
            f"Minimal ({len(minimal)} chars) debe ser menor que "
            f"full ({len(full)} chars)"
        )
        # Esperamos al menos 30% de reducción cuando hay kinematics
        ratio = len(minimal) / len(full)
        assert ratio <= 0.7, (
            f"Reducción esperada: ≥30%. Actual: {(1 - ratio) * 100:.0f}% "
            f"(minimal={len(minimal)}, full={len(full)})"
        )

    def test_minimal_context_shorter_when_no_kinematics(self):
        """Aun sin kinematics, el mínimo debe ser MÁS CORTO porque
        omite headers verbose y la nota IMPORTANTE doble del full."""
        from prompts.plan_generator import (
            build_minimal_correction_context,
            build_nutrition_context,
        )
        nutrition = _build_nutrition_minimal()
        full = build_nutrition_context(nutrition)
        minimal = build_minimal_correction_context(nutrition)
        assert len(minimal) <= len(full)


# ---------------------------------------------------------------------------
# 2. Targets duros preservados
# ---------------------------------------------------------------------------
class TestHardTargetsPreserved:
    def test_calories_present(self):
        from prompts.plan_generator import build_minimal_correction_context
        nutrition = _build_nutrition_with_kinematics()
        out = build_minimal_correction_context(nutrition)
        assert "2050" in out, "Target calories debe aparecer"
        assert "kcal" in out

    def test_macros_present_in_grams(self):
        from prompts.plan_generator import build_minimal_correction_context
        nutrition = _build_nutrition_with_kinematics()
        out = build_minimal_correction_context(nutrition)
        assert "154g" in out, "Proteína g debe aparecer"
        assert "231g" in out, "Carbos g debe aparecer"
        assert "57g" in out, "Grasas g debe aparecer"

    def test_macros_string_form_present_for_schema_match(self):
        """Las macros como string ('154g') son lo que SingleDayPlanModel
        espera; el corrector las re-emite tal cual. Sin esta paridad,
        el structured output puede fallar parseo."""
        from prompts.plan_generator import build_minimal_correction_context
        nutrition = _build_nutrition_with_kinematics()
        out = build_minimal_correction_context(nutrition)
        assert "protein='154g'" in out
        assert "carbs='231g'" in out
        assert "fats='57g'" in out


# ---------------------------------------------------------------------------
# 3. Ruido removido
# ---------------------------------------------------------------------------
class TestNoiseRemoved:
    def test_no_kinematics_block(self):
        """Velocity/acceleration/body_fat_trend son señales para el
        planner, no para el corrector que solo modifica un día."""
        from prompts.plan_generator import build_minimal_correction_context
        nutrition = _build_nutrition_with_kinematics()
        out = build_minimal_correction_context(nutrition)
        assert "Velocidad actual" not in out
        assert "Aceleración" not in out
        assert "body_fat" not in out.lower()
        assert "Cinética" not in out
        assert "CINÉTICA METABÓLICA" not in out

    def test_no_metabolismo_evolutivo_block(self):
        """La nota de adaptación evolutiva instruye al planner sobre el
        TONO general — irrelevante cuando re-emites un día existente."""
        from prompts.plan_generator import build_minimal_correction_context
        nutrition = _build_nutrition_with_kinematics()
        out = build_minimal_correction_context(nutrition)
        assert "METABOLISMO EVOLUTIVO" not in out
        assert "INSTRUCCIÓN IA" not in out

    def test_no_warning_blocks(self):
        """Las advertencias 'perdiendo más rápido / más lento' van al
        planner — el corrector no debe alterar tono."""
        from prompts.plan_generator import build_minimal_correction_context
        nutrition = _build_nutrition_with_kinematics()
        out = build_minimal_correction_context(nutrition)
        assert "ADVERTENCIA" not in out
        assert "estancamiento" not in out

    def test_no_bmr_tdee_in_minimal(self):
        """BMR y TDEE son para entender la fórmula — el corrector solo
        necesita el target final."""
        from prompts.plan_generator import build_minimal_correction_context
        nutrition = _build_nutrition_with_kinematics()
        out = build_minimal_correction_context(nutrition)
        # Target calories sí aparece (2050) — verificamos que NO aparezcan
        # los intermedios (1499, 1799) que solo confunden al corrector.
        assert "BMR" not in out, "BMR no es necesario para corregir un día"
        assert "TDEE" not in out, "TDEE tampoco — solo target_calories"
        assert "1499" not in out
        assert "1799" not in out


# ---------------------------------------------------------------------------
# 4. Wiring: _build_shared_context expone el campo nuevo
# ---------------------------------------------------------------------------
class TestSharedContextExposesMinimal:
    def test_shared_context_includes_nutrition_context_minimal_key(self):
        """`_build_shared_context` debe exponer `nutrition_context_minimal`
        además del full. Sin la key, los correctores fallan con KeyError."""
        import inspect
        import graph_orchestrator as go
        src = inspect.getsource(go._build_shared_context)
        assert '"nutrition_context_minimal"' in src

    def test_self_critique_correction_uses_minimal_context(self):
        """[P5-PROMPT-D] El correction_prompt en self_critique_node debe
        usar `nutrition_context_minimal`, no `nutrition_context`."""
        import inspect
        import graph_orchestrator as go
        src = inspect.getsource(go.self_critique_node)
        # Encuentra el correction_prompt
        assert "ctx['nutrition_context_minimal']" in src, (
            "self_critique_node correction_prompt debe usar la versión "
            "mínima del contexto para reducir tokens"
        )

    def test_surgical_marker_regen_uses_minimal_context(self):
        """[P5-PROMPT-D] Mismo bloque mínimo en surgical_marker_regen_node
        — paridad entre las dos rutas de corrección."""
        import inspect
        import graph_orchestrator as go
        src = inspect.getsource(go.surgical_marker_regen_node)
        assert "ctx['nutrition_context_minimal']" in src


# ---------------------------------------------------------------------------
# 5. Edge cases del builder
# ---------------------------------------------------------------------------
class TestBuilderEdgeCases:
    def test_handles_missing_kinematics_key(self):
        """No debe lanzar si nutrition['kinematics'] no existe."""
        from prompts.plan_generator import build_minimal_correction_context
        nutrition = _build_nutrition_minimal()
        out = build_minimal_correction_context(nutrition)
        assert "2050" in out  # build OK

    def test_returns_string(self):
        from prompts.plan_generator import build_minimal_correction_context
        nutrition = _build_nutrition_with_kinematics()
        out = build_minimal_correction_context(nutrition)
        assert isinstance(out, str)
        assert len(out) > 0
