"""[P1-SURGICAL-1] Tests para la augmentación de `_affected_days` con días
que arrastran un `_critique_unresolved` marker del intento previo.

Bug original (incidente producción 2026-05-05):
  - Intento #1: self_critique detecta "Día 2 almuerzo y cena comparten
    proteína (pollo)" pero el corrector LLM hace TIMEOUT (90s) sin
    completar la corrección. El día queda inalterado con el bug latente.
  - El revisor médico del intento #1 NO flagea Día 2 (su lente es
    sodio/alergia/repetición proteína de TIPO distinto, no incoherencia
    receta-ingredientes). Solo flagea Día 1 y Día 3.
  - `_affected_days = [1, 3]`.
  - Intento #2: surgical fix recicla Día 2 como "válido" porque no está en
    affected_days. Regenera Día 1 y Día 3.
  - Revisor médico del intento #2 ahora SÍ detecta el problema en Día 2:
    "Croquetas de Pollo: la receta indica 'pollo' pero no hay ingrediente
    equivalente listado". Severity HIGH → plan entregado roto al usuario.

Fix:
  `_correct_single_day` marca el día con `_critique_unresolved` cuando la
  corrección no se completa (timeout / CB-open / exception / LLM-None).
  `_augment_affected_days_with_critique_markers` (llamado por
  `plan_skeleton_node` en surgical_mode) AÑADE esos días a `affected_days`
  antes de decidir reciclaje. Política: forzar regen aunque medical no lo
  flagee. Costo: ~30-40s extras por día regenerado.

Cobertura:
  - Helper aislado con escenarios canónicos
  - Repro exacto del incidente 2026-05-05
  - Edge cases: previous_days malformado, marker malformado, día ya en affected
"""
import pytest

from graph_orchestrator import _augment_affected_days_with_critique_markers


# ---------------------------------------------------------------------------
# 1. Escenarios canónicos
# ---------------------------------------------------------------------------
class TestAugmentAffectedDays:
    def test_marker_promueve_dia_no_flageado(self):
        """Día 2 tiene marker pero medical solo flageó Día 1 → augment promueve."""
        affected = [1]
        prev_days = [
            {"day": 1, "meals": []},
            {"day": 2, "meals": [], "_critique_unresolved": {
                "reason": "timeout", "issue": "pollo repetido", "attempt": 1
            }},
            {"day": 3, "meals": []},
        ]
        result = _augment_affected_days_with_critique_markers(affected, prev_days)
        assert result == [1, 2], f"Esperado [1,2], recibido {result}"

    def test_multiple_markers_todos_promovidos(self):
        """Si 2 días tienen marker, ambos se añaden a affected."""
        result = _augment_affected_days_with_critique_markers(
            affected_days=[],
            previous_days=[
                {"day": 1, "_critique_unresolved": {"reason": "timeout"}},
                {"day": 2},
                {"day": 3, "_critique_unresolved": {"reason": "cb_open"}},
            ],
        )
        assert result == [1, 3]

    def test_sin_markers_devuelve_affected_intacto(self):
        """Sin markers, no cambia affected_days."""
        affected = [1, 3]
        prev_days = [{"day": 1}, {"day": 2}, {"day": 3}]
        result = _augment_affected_days_with_critique_markers(affected, prev_days)
        assert sorted(result) == sorted(affected)

    def test_dia_ya_en_affected_no_dobla(self):
        """Día con marker que ya está en affected: no se duplica (es set)."""
        result = _augment_affected_days_with_critique_markers(
            [1],
            [{"day": 1, "_critique_unresolved": {"reason": "timeout"}}],
        )
        assert result == [1]

    def test_orden_ascendente_garantizado(self):
        """El resultado debe estar ordenado para consistencia downstream."""
        result = _augment_affected_days_with_critique_markers(
            [3],
            [{"day": 1, "_critique_unresolved": {"reason": "timeout"}}],
        )
        assert result == [1, 3]


# ---------------------------------------------------------------------------
# 2. Edge cases / robustez
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_previous_days_none_no_crash(self):
        result = _augment_affected_days_with_critique_markers([1], None)
        assert result == [1]

    def test_previous_days_vacio_no_crash(self):
        result = _augment_affected_days_with_critique_markers([2], [])
        assert result == [2]

    def test_affected_days_none_no_crash(self):
        result = _augment_affected_days_with_critique_markers(None, [
            {"day": 1, "_critique_unresolved": {"reason": "timeout"}}
        ])
        assert result == [1]

    def test_dia_no_dict_se_ignora(self):
        """Día tipo no-dict (str, list, None) en previous_days → ignorar."""
        result = _augment_affected_days_with_critique_markers(
            [],
            [None, "not a day", 42, {"day": 1, "_critique_unresolved": {"reason": "timeout"}}],
        )
        assert result == [1]

    def test_day_no_int_se_ignora(self):
        """`day` debe ser int — strings/floats no cuentan."""
        result = _augment_affected_days_with_critique_markers(
            [],
            [
                {"day": "1", "_critique_unresolved": {"reason": "timeout"}},   # str
                {"day": 1.5, "_critique_unresolved": {"reason": "timeout"}},   # float
                {"day": 2, "_critique_unresolved": {"reason": "timeout"}},     # int OK
            ],
        )
        assert result == [2]

    def test_marker_no_dict_se_ignora(self):
        """`_critique_unresolved` con tipo inválido (str, list, True) → ignorar."""
        for bad_marker in ["timeout", ["timeout"], True, 42, None]:
            result = _augment_affected_days_with_critique_markers(
                [],
                [{"day": 1, "_critique_unresolved": bad_marker}],
            )
            assert result == [], (
                f"Marker inválido {bad_marker!r} debió ser ignorado, "
                f"pero promovió Día 1: {result}"
            )

    def test_marker_dict_vacio_promueve(self):
        """`_critique_unresolved = {}` (dict vacío) cuenta como marker presente."""
        result = _augment_affected_days_with_critique_markers(
            [],
            [{"day": 1, "_critique_unresolved": {}}],
        )
        assert result == [1]


# ---------------------------------------------------------------------------
# 3. Reasons soportadas (timeout, cb_open, error, llm_returned_none)
# ---------------------------------------------------------------------------
class TestReasonsSupportadas:
    """Cualquier reason en el marker dispara la promoción — el helper no
    discrimina por causa, todas representan un día con problema sin resolver."""

    def test_timeout_promueve(self):
        result = _augment_affected_days_with_critique_markers(
            [], [{"day": 1, "_critique_unresolved": {"reason": "timeout"}}]
        )
        assert result == [1]

    def test_cb_open_promueve(self):
        result = _augment_affected_days_with_critique_markers(
            [], [{"day": 1, "_critique_unresolved": {"reason": "cb_open"}}]
        )
        assert result == [1]

    def test_llm_returned_none_promueve(self):
        result = _augment_affected_days_with_critique_markers(
            [], [{"day": 1, "_critique_unresolved": {"reason": "llm_returned_none"}}]
        )
        assert result == [1]

    def test_error_promueve(self):
        result = _augment_affected_days_with_critique_markers(
            [], [{"day": 1, "_critique_unresolved": {"reason": "error:TimeoutError"}}]
        )
        assert result == [1]


# ---------------------------------------------------------------------------
# 4. Repro exacto del incidente 2026-05-05
# ---------------------------------------------------------------------------
def test_repro_incident_2026_05_05_day2_recycled_with_unresolved_critique():
    """Reproduce exactamente el escenario:
       - self_critique detectó pollo+pollo en Día 2 → timeout corrigiéndolo
       - medical reviewer del intento #1 solo flageó Días 1 y 3
       - Pre-fix: surgical fix reciclaba Día 2 (BUG)
       - Post-fix: helper añade Día 2 a affected_days → surgical regenera"""
    affected_days_from_medical = [1, 3]
    plan_result_days_intento_1 = [
        {
            "day": 1,
            "meals": [{"meal": "Almuerzo", "name": "Pavo procesado"}],
        },
        {
            "day": 2,
            "meals": [{"meal": "Almuerzo", "name": "Pollo airfryer"},
                      {"meal": "Cena", "name": "Pollo guisado"}],
            "_critique_unresolved": {
                "reason": "timeout",
                "issue": "Día 2: almuerzo y cena comparten proteína principal (pollo). Cambia la proteína de la cena.",
                "attempt": 1,
            },
        },
        {
            "day": 3,
            "meals": [{"meal": "Almuerzo", "name": "Camarones"},
                      {"meal": "Cena", "name": "Camarones"}],
        },
    ]

    result = _augment_affected_days_with_critique_markers(
        affected_days_from_medical, plan_result_days_intento_1
    )

    # Pre-fix: result == [1, 3] → Día 2 reciclado → bug latente entra a intento #2
    # Post-fix: result == [1, 2, 3] → todos los días con problema se regeneran
    assert result == [1, 2, 3], (
        f"Esperado que el helper promueva Día 2 (con _critique_unresolved) "
        f"a affected_days, recibido {result}"
    )

    # Confirmar: con [1, 2, 3] y len(skeleton_days)=3, el surgical_mode evalúa
    # `len(affected_days) < len(skeleton_days)` → False → FULL regen del plan,
    # no surgical. Comportamiento correcto: si TODOS los días tienen problema,
    # regenerar el plan entero es la mejor estrategia.
    assert len(result) == 3
