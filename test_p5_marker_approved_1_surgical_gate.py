"""[P5-MARKER-APPROVED-1] Tests para el surgical gate post-approval.

Bug observable (corrida 2026-05-05 03:54-04:00):
  Self-critique detectó repetición almuerzo↔cena en Días 1, 2, 3.
  Día 1 corrigió OK (43s), Día 3 corrigió OK (90s), Día 2 timeoutó al
  cap de 150s (P4-TIMEOUT-3) → marker `_critique_unresolved`.
  Reviewer médico aprobó porque su lente (alergias, sodio, despensa,
  skeleton fidelity) es ortogonal a slot coherence.
  Plan llegó al usuario con Día 2 mostrando la repetición intacta.

Causa raíz:
  `should_retry` retornaba "end" cuando `review_passed=True`, sin
  consultar markers `_critique_unresolved` que pudieran indicar problemas
  detectados-pero-no-corregidos por el self-critique.

Fix:
  Nuevo nodo `surgical_marker_regen_node` y rama "marker_regen" en
  `should_retry`. Cuando reviewer aprueba PERO hay markers pendientes,
  re-corremos el corrector LLM con budget fresco para esos días.
  Una sola pasada (flag `_marker_regen_attempted`).

Cobertura:
  - `_collect_unresolved_marker_days`: detección
  - `should_retry`: nueva rama "marker_regen" + flag de re-entrada
  - `surgical_marker_regen_node`: fast-path sin markers
  - `initial_state`: flag inicializado a False
  - `retry_reflection_node`: flag reseteado en nuevos attempts
  - Graph wiring: nodo + edge condicional + edge surgical→assemble
"""
import asyncio
import importlib

import pytest


# ---------------------------------------------------------------------------
# 1. _collect_unresolved_marker_days — heurística pura
# ---------------------------------------------------------------------------
class TestCollectUnresolvedMarkerDays:
    def setup_method(self):
        import graph_orchestrator
        self.go = graph_orchestrator

    def test_empty_plan_returns_empty(self):
        assert self.go._collect_unresolved_marker_days({}) == []
        assert self.go._collect_unresolved_marker_days(None) == []

    def test_no_marker_returns_empty(self):
        plan = {"days": [{"day": 1, "meals": []}, {"day": 2, "meals": []}]}
        assert self.go._collect_unresolved_marker_days(plan) == []

    def test_marker_dict_present_includes_day(self):
        plan = {
            "days": [
                {"day": 1, "meals": []},
                {
                    "day": 2,
                    "meals": [],
                    "_critique_unresolved": {
                        "reason": "timeout",
                        "issue": "slot repetition",
                        "attempt": 1,
                    },
                },
            ]
        }
        assert self.go._collect_unresolved_marker_days(plan) == [2]

    def test_multiple_markers_sorted(self):
        plan = {
            "days": [
                {"day": 3, "_critique_unresolved": {"reason": "cb_open"}},
                {"day": 1, "_critique_unresolved": {"reason": "timeout"}},
                {"day": 2},  # sin marker
            ]
        }
        assert self.go._collect_unresolved_marker_days(plan) == [1, 3]

    def test_marker_falsy_ignored(self):
        """Markers None / {} no cuentan — defensa ante state corrupto."""
        plan = {
            "days": [
                {"day": 1, "_critique_unresolved": None},
                {"day": 2, "_critique_unresolved": {}},
                {"day": 3, "_critique_unresolved": {"reason": "timeout"}},
            ]
        }
        assert self.go._collect_unresolved_marker_days(plan) == [3]

    def test_invalid_day_num_skipped(self):
        plan = {
            "days": [
                {"day": "two", "_critique_unresolved": {"reason": "timeout"}},
                {"day": 2, "_critique_unresolved": {"reason": "timeout"}},
            ]
        }
        assert self.go._collect_unresolved_marker_days(plan) == [2]

    def test_non_dict_day_skipped(self):
        plan = {"days": [None, "garbage", {"day": 2, "_critique_unresolved": {"reason": "timeout"}}]}
        assert self.go._collect_unresolved_marker_days(plan) == [2]


# ---------------------------------------------------------------------------
# 2. should_retry — rama nueva "marker_regen"
# ---------------------------------------------------------------------------
class TestShouldRetryMarkerRegenBranch:
    def setup_method(self):
        import graph_orchestrator
        self.go = graph_orchestrator

    def _base_state_approved(self, **overrides):
        """State mínimo para que `should_retry` con review_passed=True
        no caiga en otras ramas (critical / high)."""
        state = {
            "review_passed": True,
            "attempt": 1,
            "pipeline_start": 0.0,  # mucho budget
            "_rejection_severity": None,
            "_marker_regen_attempted": False,
            "plan_result": {"days": []},
        }
        state.update(overrides)
        return state

    def test_approved_no_markers_routes_to_end(self):
        """Comportamiento previo preservado: aprobado sin markers → end."""
        state = self._base_state_approved()
        assert self.go.should_retry(state) == "end"

    def test_approved_with_markers_and_flag_false_routes_to_marker_regen(self):
        """Caso clave del fix: aprobado + markers + sin gate previo → regen."""
        state = self._base_state_approved(
            plan_result={
                "days": [
                    {"day": 1, "meals": []},
                    {
                        "day": 2,
                        "meals": [],
                        "_critique_unresolved": {"reason": "timeout"},
                    },
                ]
            }
        )
        assert self.go.should_retry(state) == "marker_regen"

    def test_approved_with_markers_but_already_attempted_routes_to_end(self):
        """Flag previene loop: 2da pasada va a end aunque haya markers."""
        state = self._base_state_approved(
            _marker_regen_attempted=True,
            plan_result={
                "days": [
                    {
                        "day": 2,
                        "_critique_unresolved": {"reason": "timeout"},
                    },
                ]
            },
        )
        assert self.go.should_retry(state) == "end"

    def test_rejected_path_unaffected_by_markers(self):
        """Si reviewer rechaza, la rama de retry sigue intacta — markers
        son consultados solo en el branch aprobado. El path de rechazo
        usa `_augment_affected_days_with_critique_markers` (P1-SURGICAL-1)
        en `plan_skeleton_node`, ortogonal a este fix."""
        state = self._base_state_approved(
            review_passed=False,
            _rejection_severity="minor",
            rejection_reasons=["alguna razón"],
            plan_result={
                "days": [
                    {"day": 1, "_critique_unresolved": {"reason": "timeout"}},
                ]
            },
        )
        # No debe ser "marker_regen" — debe ser "retry" (rejected, minor).
        assert self.go.should_retry(state) != "marker_regen"


# ---------------------------------------------------------------------------
# 3. surgical_marker_regen_node — fast-path sin markers
# ---------------------------------------------------------------------------
class TestSurgicalMarkerRegenNodeFastPath:
    def setup_method(self):
        import graph_orchestrator
        self.go = graph_orchestrator

    def test_no_markers_returns_flag_only(self):
        """Defensa: si por alguna razón se invoca el nodo sin markers,
        debe retornar solo el flag (sin tocar plan_result, sin llamar LLM).

        En la práctica `should_retry` ya filtra este caso, pero el nodo
        debe ser resiliente."""
        state = {
            "plan_result": {"days": [{"day": 1}, {"day": 2}]},
            "form_data": {},
        }
        result = asyncio.run(self.go.surgical_marker_regen_node(state))
        assert result == {"_marker_regen_attempted": True}

    def test_invalid_plan_result_returns_flag_only(self):
        state = {"plan_result": None, "form_data": {}}
        result = asyncio.run(self.go.surgical_marker_regen_node(state))
        assert result == {"_marker_regen_attempted": True}

    def test_plan_result_not_dict_returns_flag_only(self):
        state = {"plan_result": "not_a_dict", "form_data": {}}
        result = asyncio.run(self.go.surgical_marker_regen_node(state))
        assert result == {"_marker_regen_attempted": True}


# ---------------------------------------------------------------------------
# 4. retry_reflection_node — reset del flag
# ---------------------------------------------------------------------------
class TestRetryReflectionResetsFlag:
    def setup_method(self):
        import graph_orchestrator
        self.go = graph_orchestrator

    def test_retry_reflection_resets_marker_regen_flag(self):
        """Cada attempt nuevo debe tener su propia oportunidad de surgical
        regen. Sin reset, attempt #2 aprobado-con-markers iría directo a
        `end` heredando el flag `True` del attempt #1."""
        state = {
            "attempt": 1,
            "_marker_regen_attempted": True,
            "rejection_reasons": ["alguna razón"],
        }
        result = asyncio.run(self.go.retry_reflection_node(state))
        assert result.get("_marker_regen_attempted") is False
        assert result.get("attempt") == 2


# ---------------------------------------------------------------------------
# 5. Graph wiring — nodo + edges
# ---------------------------------------------------------------------------
class TestGraphWiring:
    def setup_method(self):
        import graph_orchestrator
        self.go = graph_orchestrator

    def test_surgical_marker_regen_node_referenced_in_build(self):
        """[P5-MARKER-APPROVED-1] El nodo debe estar añadido al grafo y
        conectado a assemble_plan. Verificamos por substring en build_plan_graph."""
        import inspect
        src = inspect.getsource(self.go.build_plan_graph)
        assert 'surgical_marker_regen' in src, (
            "build_plan_graph debe registrar 'surgical_marker_regen'"
        )
        assert 'add_edge("surgical_marker_regen", "assemble_plan")' in src, (
            "Tras surgical regen debe ir a assemble_plan para re-aggregate"
        )
        assert '"marker_regen": "surgical_marker_regen"' in src, (
            "El conditional edge debe mapear 'marker_regen' al nodo nuevo"
        )

    def test_should_retry_returns_marker_regen_branch(self):
        """`should_retry` debe tener una rama que retorne 'marker_regen'."""
        import inspect
        src = inspect.getsource(self.go.should_retry)
        assert "'marker_regen'" in src or '"marker_regen"' in src, (
            "should_retry debe retornar la nueva rama 'marker_regen' "
            "para activar el surgical gate"
        )

    def test_compiled_graph_has_surgical_node(self):
        """Sanity: build_plan_graph compila sin errores con el nodo nuevo
        y el grafo compilado expone el nodo."""
        compiled = self.go.build_plan_graph()
        # LangGraph guarda los nodos en `nodes` o similar — solo verificamos
        # que la compilación no haya lanzado.
        assert compiled is not None


# ---------------------------------------------------------------------------
# 6. PlanState + initial_state
# ---------------------------------------------------------------------------
class TestStateInitialization:
    def test_planstate_has_marker_regen_flag(self):
        """Verifica que el campo está declarado en PlanState (TypedDict)."""
        import graph_orchestrator
        # PlanState es TypedDict — usamos __annotations__ para verificar.
        assert "_marker_regen_attempted" in graph_orchestrator.PlanState.__annotations__

    def test_initial_state_includes_marker_regen_false(self):
        """El initial_state de arun_plan_pipeline debe inicializar el flag.
        Verificamos por substring para evitar setup pesado del pipeline."""
        import inspect
        import graph_orchestrator
        src = inspect.getsource(graph_orchestrator.arun_plan_pipeline)
        assert '"_marker_regen_attempted": False' in src, (
            "initial_state en arun_plan_pipeline debe inicializar "
            "_marker_regen_attempted=False explícitamente"
        )


# ---------------------------------------------------------------------------
# 7. Repro corrida 2026-05-05 03:54-04:00 — gate atrapa el caso real
# ---------------------------------------------------------------------------
def test_repro_corrida_2026_05_05_03_54_marker_gate_activates():
    """Reproduce el state al final de review_plan_node (approved=True) con
    Día 2 marcado por timeout. should_retry debe enrutar a marker_regen."""
    import graph_orchestrator as go

    state_at_review_end = {
        "review_passed": True,
        "attempt": 1,
        "pipeline_start": 0.0,  # mucho budget
        "_rejection_severity": None,
        "_marker_regen_attempted": False,
        "plan_result": {
            "days": [
                {"day": 1, "meals": []},  # corregido OK
                {
                    "day": 2,
                    "meals": [],
                    "_critique_unresolved": {
                        "reason": "timeout",
                        "issue": "Día 2: Sustituir los gandules de la cena por otra proteína.",
                        "attempt": 1,
                    },
                },
                {"day": 3, "meals": []},  # corregido OK
            ]
        },
    }

    decision = go.should_retry(state_at_review_end)
    assert decision == "marker_regen", (
        f"Pre-fix: 'end' (plan al usuario con bug). "
        f"Post-fix: 'marker_regen' (surgical regen primero). "
        f"Recibido: {decision!r}"
    )

    # Detección por el helper
    assert go._collect_unresolved_marker_days(state_at_review_end["plan_result"]) == [2]
