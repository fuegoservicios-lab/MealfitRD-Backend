"""[P3-LSQ-ERROR-PER-CALL · 2026-08-04] `_LAST_LSQ_ERROR` era un global de módulo
(`portion_solver.py`) que solo se ESCRIBÍA en el `except` de `_compute_scale_factors` y JAMÁS se
limpiaba. Se estampaba en el dict de retorno como `lsq_error` cuando `method == "greedy"` — pero
`method == "greedy"` tiene TRES causas: el knob `MEALFIT_SOLVER_LSQ` apagado, `A_rows` vacío (sin
targets positivos), o el LSQ crasheando de verdad. Tras UN crash en el proceso (el solver corre
~28 veces por plan), cualquier comida POSTERIOR que cayera a greedy por OTRA razón heredaba el
TypeError viejo de OTRA comida — la ambigüedad que el campo existe para resolver. Además el global
se comparte entre threads sin lock.

Fix: el error viaja como 5º elemento del tuple de `_compute_scale_factors` (`None` si el greedy no
vino de un crash) — cada llamada reporta SU PROPIO resultado, sin estado compartido entre llamadas
ni entre threads. `_LSQ_ERR_SEEN` (dedup del WARNING por tipo) se conserva intacto.

100% OFFLINE: `StubDB` sustituye a `IngredientNutritionDB` (el .env apunta a PRODUCCIÓN).
"""
import portion_solver as ps


POLLO_ENTRY = {"macros": {"kcal": 165, "protein": 31, "carbs": 0, "fats": 3.6}, "group": "protein"}


def _boom(*_a, **_k):
    raise TypeError("matriz singular de prueba (P3-LSQ-ERROR-PER-CALL)")


# ═════════════ 1 · el crash se reporta EN esa llamada ═════════════

def test_el_crash_del_lsq_llega_como_5o_elemento_del_tuple(monkeypatch):
    monkeypatch.setattr(ps, "_box_lsq", _boom)
    monkeypatch.setattr(ps, "SOLVER_LSQ", True)
    ps._LSQ_ERR_SEEN.clear()

    result = ps._compute_scale_factors(
        [dict(POLLO_ENTRY)], {"kcal": 300, "protein": 60, "carbs": 0, "fats": 0}, 0.5, 3.5, 5.0)

    assert len(result) == 5, f"el tuple debe tener 5 elementos (factors, method, sat_hi, sat_lo, lsq_error): {result!r}"
    factors, method, _sat_hi, _sat_lo, lsq_error = result
    assert method == "greedy", "el valor de `method` NO cambia: el caller lo compara exacto"
    assert lsq_error and "TypeError" in lsq_error


# ═════════════ 2 · el bug real: una llamada POSTERIOR sin crash no hereda el error ═════════════

def test_greedy_por_a_rows_vacio_tras_un_crash_previo_no_hereda_el_error_stale(monkeypatch):
    """Reproduce el escenario del audit: comida A crashea el LSQ; comida B (targets todos ≤0 →
    `A_rows` vacío, greedy SIN excepción) no debe estampar el TypeError de la comida A."""
    monkeypatch.setattr(ps, "_box_lsq", _boom)
    monkeypatch.setattr(ps, "SOLVER_LSQ", True)
    ps._LSQ_ERR_SEEN.clear()

    # Comida A: crash real del LSQ.
    _f_a, method_a, _hi_a, _lo_a, lsq_error_a = ps._compute_scale_factors(
        [dict(POLLO_ENTRY)], {"kcal": 300, "protein": 60, "carbs": 0, "fats": 0}, 0.5, 3.5, 5.0)
    assert method_a == "greedy" and lsq_error_a, "sanity: la comida A sí crasheó"

    # Comida B: todos los targets ≤ 0 → ninguna fila entra a `A_rows` → greedy SIN excepción.
    _f_b, method_b, _hi_b, _lo_b, lsq_error_b = ps._compute_scale_factors(
        [dict(POLLO_ENTRY)], {"kcal": 0, "protein": 0, "carbs": 0, "fats": 0}, 0.5, 3.5, 5.0)

    assert method_b == "greedy"
    assert lsq_error_b is None, (
        f"greedy por A_rows vacío no vino de un crash — no debe heredar el error de otra llamada "
        f"(hoy con el global stale devolvería: {lsq_error_b!r})")


# ═════════════ 3 · knob OFF: greedy directo, sin haber intentado el LSQ ═════════════

def test_knob_off_no_reporta_error_aunque_haya_habido_un_crash_previo(monkeypatch):
    ps._LSQ_ERR_SEEN.clear()
    # Un crash previo con el knob ON (contaminaría el global si siguiera existiendo).
    monkeypatch.setattr(ps, "_box_lsq", _boom)
    monkeypatch.setattr(ps, "SOLVER_LSQ", True)
    ps._compute_scale_factors(
        [dict(POLLO_ENTRY)], {"kcal": 300, "protein": 60, "carbs": 0, "fats": 0}, 0.5, 3.5, 5.0)

    # Ahora el knob se apaga: greedy directo, ni se intenta el LSQ.
    monkeypatch.setattr(ps, "SOLVER_LSQ", False)
    _f, method, _hi, _lo, lsq_error = ps._compute_scale_factors(
        [dict(POLLO_ENTRY)], {"kcal": 300, "protein": 60, "carbs": 0, "fats": 0}, 0.5, 3.5, 5.0)

    assert method == "greedy"
    assert lsq_error is None


# ═════════════ 4 · end-to-end: `solve_meal_macros` no filtra el stale al dict público ═════════════

class _StubDB:
    """Catálogo offline: cero DB, cero red."""

    def macros_from_ingredient_string(self, s):
        return dict(POLLO_ENTRY["macros"]) if "pollo" in s.lower() else None


def test_solve_meal_macros_no_propaga_lsq_error_stale_a_una_comida_sin_crash(monkeypatch):
    monkeypatch.setattr(ps, "_box_lsq", _boom)
    monkeypatch.setattr(ps, "SOLVER_LSQ", True)
    ps._LSQ_ERR_SEEN.clear()

    # Comida A: crash real.
    out_a = ps.solve_meal_macros(["100 g de pollo"], {"protein": 60, "kcal": 300, "carbs": 0, "fats": 0},
                                  db=_StubDB())
    assert out_a["method"] == "greedy" and out_a["lsq_error"], "sanity: la comida A sí crasheó"

    # Comida B: targets todos ≤0 → A_rows vacío → greedy sin excepción. NO debe heredar el error de A.
    out_b = ps.solve_meal_macros(["100 g de pollo"], {"protein": 0, "kcal": 0, "carbs": 0, "fats": 0},
                                  db=_StubDB())
    assert out_b["method"] == "greedy"
    assert out_b["lsq_error"] is None, (
        f"el dict público heredó el error de otra comida: {out_b['lsq_error']!r}")
