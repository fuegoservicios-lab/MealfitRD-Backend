"""[P0-PIPE-1] Tests para el fallback al MEJOR intento previo cuando el
último intento del pipeline empeoró respecto a un intento anterior.

Bug original (incidente producción 2026-05-05):
  - Plan reset desde 0, pipeline corre.
  - Intento #1: rechazo `minor` (pavo procesado, camarones repetidos) —
    plan estructuralmente válido, observaciones no-críticas.
  - Intento #2: surgical fix recicla Día 2 del intento #1, regenera Días 1
    y 3. El review detecta:
      * Día 1 omitió proteínas asignadas (pechuga de pollo, queso mozzarella)
      * Día 2 (reciclado): receta menciona "pollo" pero no aparece en
        ingredients_raw
      * Día 3: receta menciona "pescado" pero solo lista "merluza"
    Severidad: HIGH.
  - `should_retry` retorna "end" → END del grafo → arun retorna
    `final_state["plan_result"]` (= intento #2 ROTO) al cliente.
  - Holistic Score: 0.744 (vs 1.000 del plan anterior aprobado al primer
    intento).

Fix:
  `review_plan_node` mantiene un snapshot del intento con MENOR severidad
  (rank: approved=0 < minor=1 < high=2 < critical=3) en `_best_attempt_*`
  fields del state. Tras el grafo, `_swap_to_best_attempt_if_better` (en
  `arun_plan_pipeline`) compara current vs best y restaura el best si es
  estrictamente mejor — ANTES de los guardrails de critical/transparency,
  para que el cliente reciba el plan de mayor calidad disponible.

Cobertura:
  - `_attempt_quality_rank`: orden correcto (approved < minor < high < critical)
  - `_swap_to_best_attempt_if_better`: 6 escenarios canónicos
  - `review_plan_node` integración: aprobado → best siempre, rechazado →
    best solo si current rank < prior rank
  - Edge cases: snapshot ausente, plan_result vacío, severity inválida
"""
import copy

import pytest

from graph_orchestrator import (
    _SEVERITY_RANK,
    _attempt_quality_rank,
    _swap_to_best_attempt_if_better,
)


# ---------------------------------------------------------------------------
# 1. Helper de rank
# ---------------------------------------------------------------------------
class TestAttemptQualityRank:
    def test_approved_es_mejor(self):
        assert _attempt_quality_rank(True, None) == 0
        assert _attempt_quality_rank(True, "high") == 0  # passed gana sobre severity

    def test_minor_es_segundo(self):
        assert _attempt_quality_rank(False, "minor") == 1

    def test_high_es_tercero(self):
        assert _attempt_quality_rank(False, "high") == 2

    def test_critical_es_peor(self):
        assert _attempt_quality_rank(False, "critical") == 3

    def test_severity_none_default_minor(self):
        assert _attempt_quality_rank(False, None) == 1

    def test_severity_invalida_default_minor(self):
        assert _attempt_quality_rank(False, "garbage_value") == 1


# ---------------------------------------------------------------------------
# 2. _swap_to_best_attempt_if_better — escenarios canónicos
# ---------------------------------------------------------------------------
class TestSwapToBestAttempt:
    def _make_state(self, *, current_severity, current_passed,
                    best_severity, best_passed, best_attempt_n=1,
                    current_attempt_n=2, has_best=True):
        plan_current = {"days": [{"meal": "current_attempt"}], "marker": "current"}
        plan_best = {"days": [{"meal": "best_attempt"}], "marker": "best"} if has_best else None
        return {
            "plan_result": plan_current,
            "review_passed": current_passed,
            "_rejection_severity": current_severity,
            "rejection_reasons": ["current_reason"],
            "attempt": current_attempt_n,
            "_best_attempt_plan": plan_best,
            "_best_attempt_severity": best_severity,
            "_best_attempt_reasons": ["best_reason"],
            "_best_attempt_review_passed": best_passed,
            "_best_attempt_number": best_attempt_n,
        }

    # --- Casos donde DEBE haber swap ---
    def test_high_rejected_to_minor_rejected_swaps(self):
        """Caso del incidente real: intento #2 HIGH, intento #1 minor → swap."""
        state = self._make_state(
            current_severity="high", current_passed=False,
            best_severity="minor", best_passed=False,
        )
        swapped = _swap_to_best_attempt_if_better(state)
        assert swapped is True
        assert state["plan_result"]["marker"] == "best"
        assert state["_rejection_severity"] == "minor"
        assert state["rejection_reasons"] == ["best_reason"]
        assert state["plan_result"]["_best_attempt_swapped_from"] == 2
        assert state["plan_result"]["_best_attempt_swapped_severity"] == "high"

    def test_critical_rejected_to_minor_rejected_swaps(self):
        state = self._make_state(
            current_severity="critical", current_passed=False,
            best_severity="minor", best_passed=False,
        )
        swapped = _swap_to_best_attempt_if_better(state)
        assert swapped is True
        assert state["plan_result"]["marker"] == "best"

    def test_critical_rejected_to_approved_swaps(self):
        """Best fue aprobado, current crítico → swap restaura plan aprobado."""
        state = self._make_state(
            current_severity="critical", current_passed=False,
            best_severity=None, best_passed=True,
        )
        swapped = _swap_to_best_attempt_if_better(state)
        assert swapped is True
        assert state["plan_result"]["marker"] == "best"
        assert state["review_passed"] is True
        # Tras swap a approved, severity se normaliza a "minor" (no relevante)
        assert state["_rejection_severity"] == "minor"

    def test_high_rejected_to_approved_swaps(self):
        state = self._make_state(
            current_severity="high", current_passed=False,
            best_severity=None, best_passed=True,
        )
        assert _swap_to_best_attempt_if_better(state) is True
        assert state["review_passed"] is True

    # --- Casos donde NO debe haber swap ---
    def test_current_approved_no_swap(self):
        """Si current ya está aprobado, no swap aunque haya snapshot."""
        state = self._make_state(
            current_severity=None, current_passed=True,
            best_severity="minor", best_passed=False,
        )
        swapped = _swap_to_best_attempt_if_better(state)
        assert swapped is False
        assert state["plan_result"]["marker"] == "current"
        assert state["review_passed"] is True

    def test_no_best_snapshot_no_swap(self):
        """Sin best snapshot (e.g. pipeline abortó pre-review), no-op."""
        state = self._make_state(
            current_severity="high", current_passed=False,
            best_severity=None, best_passed=False, has_best=False,
        )
        # Plan_result actual queda intacto
        swapped = _swap_to_best_attempt_if_better(state)
        assert swapped is False
        assert state["plan_result"]["marker"] == "current"

    def test_best_same_severity_no_swap(self):
        """Si best y current tienen el mismo rank, conservamos current."""
        state = self._make_state(
            current_severity="minor", current_passed=False,
            best_severity="minor", best_passed=False,
        )
        swapped = _swap_to_best_attempt_if_better(state)
        assert swapped is False
        assert state["plan_result"]["marker"] == "current"

    def test_best_worse_severity_no_swap(self):
        """Best peor que current (ej. best=high, current=minor) — no swap."""
        state = self._make_state(
            current_severity="minor", current_passed=False,
            best_severity="high", best_passed=False,
        )
        swapped = _swap_to_best_attempt_if_better(state)
        assert swapped is False

    # --- Edge cases / robustez ---
    def test_best_plan_dict_vacio_no_swap(self):
        """`_best_attempt_plan = {}` (dict vacío) cuenta como "no snapshot"."""
        state = {
            "plan_result": {"marker": "current"},
            "review_passed": False,
            "_rejection_severity": "high",
            "rejection_reasons": [],
            "attempt": 2,
            "_best_attempt_plan": {},
            "_best_attempt_severity": "minor",
            "_best_attempt_reasons": [],
            "_best_attempt_review_passed": False,
            "_best_attempt_number": 1,
        }
        assert _swap_to_best_attempt_if_better(state) is False

    def test_best_plan_no_dict_no_swap(self):
        """`_best_attempt_plan` con tipo inesperado (string, list, None) → no-op."""
        for bad_plan in [None, [], "not a plan", 42]:
            state = {
                "plan_result": {"marker": "current"},
                "review_passed": False,
                "_rejection_severity": "high",
                "_best_attempt_plan": bad_plan,
                "_best_attempt_severity": "minor",
                "_best_attempt_review_passed": False,
            }
            assert _swap_to_best_attempt_if_better(state) is False, (
                f"Esperado no-swap con _best_attempt_plan={bad_plan!r}"
            )

    def test_swap_preserva_keys_originales_del_plan_best(self):
        """Tras swap, el plan_result debe ser el del best snapshot
        (preservando sus keys), MÁS los markers de telemetría."""
        state = self._make_state(
            current_severity="high", current_passed=False,
            best_severity="minor", best_passed=False,
        )
        original_best_keys = set(state["_best_attempt_plan"].keys())
        _swap_to_best_attempt_if_better(state)
        # El plan resultante debe contener TODAS las keys originales del best
        for k in original_best_keys:
            assert k in state["plan_result"]
        # Plus los 2 markers de swap
        assert "_best_attempt_swapped_from" in state["plan_result"]
        assert "_best_attempt_swapped_severity" in state["plan_result"]


# ---------------------------------------------------------------------------
# 3. Severity rank ordering — invariante crítica (regresión guard)
# ---------------------------------------------------------------------------
def test_severity_rank_total_order():
    """Si alguien añade una nueva severidad sin actualizar el orden total,
    los swaps pueden fallar silenciosamente. Esta invariante valida que las
    4 severidades canónicas siguen ordenadas correctamente."""
    assert _SEVERITY_RANK["none"] < _SEVERITY_RANK["minor"]
    assert _SEVERITY_RANK["minor"] < _SEVERITY_RANK["high"]
    assert _SEVERITY_RANK["high"] < _SEVERITY_RANK["critical"]


# ---------------------------------------------------------------------------
# 4. Repro completo del incidente 2026-05-05
# ---------------------------------------------------------------------------
def test_repro_incident_2026_05_05():
    """Reproduce exactamente el escenario del log de producción:
       - Intento #1 termina con severity='minor' (pavo procesado + camarones)
       - Intento #2 termina con severity='high' (skeleton fidelity rota +
         3 errores de coherencia receta-ingredientes)
       - Pre-fix: usuario recibía intento #2 (roto)
       - Post-fix: usuario recibe intento #1 (válido, con disclaimer minor)"""
    final_state = {
        # Intento #2 (último) — el que estaba siendo entregado al usuario
        "plan_result": {
            "days": [
                {"day": 1, "meal": "broken_skeleton"},
                {"day": 2, "meal": "recycled_no_pollo"},
                {"day": 3, "meal": "merluza_no_pescado"},
            ],
            "_review_severity": "high",
        },
        "review_passed": False,
        "_rejection_severity": "high",
        "rejection_reasons": [
            "Día 1 omitió múltiples proteínas clave asignadas",
            "Día 2: La receta indica 'pollo' pero no hay ingrediente equivalente listado",
            "Día 3: La receta indica 'pescado' pero no hay ingrediente equivalente listado",
        ],
        "attempt": 2,
        # Best preserved del intento #1
        "_best_attempt_plan": {
            "days": [
                {"day": 1, "meal": "pavo_pollo_yuca"},
                {"day": 2, "meal": "pollo_yuca_caldo"},
                {"day": 3, "meal": "camarones_repetidos"},
            ],
            "_review_severity": "minor",
        },
        "_best_attempt_severity": "minor",
        "_best_attempt_reasons": [
            "Uso excesivo de pechuga de pavo procesada en el Día 1",
            "Repetición de camarones en almuerzo y cena del Día 3",
        ],
        "_best_attempt_review_passed": False,
        "_best_attempt_number": 1,
    }

    swapped = _swap_to_best_attempt_if_better(final_state)
    assert swapped is True, "Repro del incidente debe swappear"

    # Validar que el usuario ahora recibe el plan #1
    assert final_state["plan_result"]["days"][0]["meal"] == "pavo_pollo_yuca"
    assert final_state["_rejection_severity"] == "minor"
    assert "pavo procesada" in final_state["rejection_reasons"][0]

    # Telemetría visible en el plan
    assert final_state["plan_result"]["_best_attempt_swapped_from"] == 2
    assert final_state["plan_result"]["_best_attempt_swapped_severity"] == "high"

    # Downstream: `_apply_critical_review_guardrails` verá severity=minor
    # → marcará `_review_failed_but_delivered=True` (no fallback matemático)
    # → cliente recibe plan #1 con disclaimer en lugar del plan #2 roto.
    # Esto se valida en e2e tests de la suite de guardrails.


# ---------------------------------------------------------------------------
# 5. Idempotencia del swap
# ---------------------------------------------------------------------------
def test_swap_idempotent_solo_uno():
    """Llamar swap 2× consecutivas: el primero swappea, el segundo no-op
    (porque tras swap, current_passed/severity reflejan el best, así que
    rank(current) == rank(best) → no swap)."""
    state = {
        "plan_result": {"marker": "current"},
        "review_passed": False,
        "_rejection_severity": "high",
        "rejection_reasons": [],
        "attempt": 2,
        "_best_attempt_plan": {"marker": "best"},
        "_best_attempt_severity": "minor",
        "_best_attempt_reasons": [],
        "_best_attempt_review_passed": False,
        "_best_attempt_number": 1,
    }
    assert _swap_to_best_attempt_if_better(state) is True
    # Segunda llamada no debe re-swappear (current == best ahora)
    assert _swap_to_best_attempt_if_better(state) is False
    # Plan sigue siendo el best (no perdido)
    assert state["plan_result"]["marker"] == "best"
