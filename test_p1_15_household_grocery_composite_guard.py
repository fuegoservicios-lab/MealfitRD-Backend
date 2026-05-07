"""[P1-15] Tests para el guard composicional `householdSize > 6 +
groceryDuration='monthly'` en `_validate_form_data_ranges`.

Bug original (audit P1-15):
  El cap individual de `householdSize` es 12 (familias extendidas
  legítimas) y el wizard expone chips 1..6 + 3 ciclos
  (weekly/biweekly/monthly). Pero un cliente legacy/scraper enviando
  `householdSize=12 + groceryDuration='monthly'` producía escalado
  de hasta ~360× el plato base. Riesgos:
    - Lista de compras absurda (cientos de libras de cada ingrediente).
    - Posible OOM en `aggregate_and_deduct_shopping_list` con miles
      de líneas a humanizar/clasificar.
    - Chunk timeouts del LLM por tamaño desproporcionado del prompt.
  Sin guard composicional, el backend aceptaba la combinación
  silenciosamente.

Fix:
  Guard agregado tras la validación individual de `householdSize`. Si
  `household > 6 AND groceryDuration='monthly'` → rechazar con 422
  accionable. El error incluye `accepted_range=[1, 6]` específico para
  esta combinación + `reason` con la lógica del cap.

Cobertura:
  - test_household_8_monthly_rejected_with_422
  - test_household_12_monthly_rejected_with_422
  - test_household_6_monthly_accepted (límite inferior pasa)
  - test_household_8_weekly_accepted (otro grocery_duration pasa)
  - test_household_12_biweekly_accepted (mismo)
  - test_household_3_monthly_accepted (caso normal)
  - test_household_above_cap_individual_validation_runs_first
  - test_groceryDuration_case_insensitive_for_guard
  - test_error_payload_includes_p1_15_marker_in_reason
"""
import pytest

from routers.plans import _validate_form_data_ranges


def _base_form(**overrides):
    """Form con todos los campos requeridos válidos como base; los tests
    sobrescriben solo lo que necesitan."""
    base = {
        "age": 30,
        "gender": "male",
        "height": 175,
        "weight": 75,
        "weightUnit": "kg",
        "activityLevel": "moderate",
    }
    base.update(overrides)
    return base


def _household_errors(errors):
    return [e for e in errors if e.get("field") == "householdSize"]


# ---------------------------------------------------------------------------
# 1. Combinaciones rechazadas (P1-15 guard activo).
# ---------------------------------------------------------------------------
def test_household_8_monthly_rejected_with_422():
    """householdSize=8 con monthly → debe rechazarse con error específico."""
    ok, errors = _validate_form_data_ranges(_base_form(
        householdSize=8, groceryDuration="monthly",
    ))
    assert ok is False
    h_errors = _household_errors(errors)
    assert h_errors, "P1-15: debe haber error específico para householdSize"
    # Algún error debe tener accepted_range=[1, 6].
    assert any(e.get("accepted_range") == [1, 6] for e in h_errors), \
        f"P1-15: error debe indicar [1,6] como rango aceptado para esta combo. Got: {h_errors}"


def test_household_12_monthly_rejected_with_422():
    """householdSize=12 (cap individual) con monthly → 422 por composite."""
    ok, errors = _validate_form_data_ranges(_base_form(
        householdSize=12, groceryDuration="monthly",
    ))
    assert ok is False
    h_errors = _household_errors(errors)
    assert any(e.get("accepted_range") == [1, 6] for e in h_errors)


def test_household_7_monthly_rejected_minimum_excess():
    """householdSize=7 (mínimo por encima de 6) con monthly → rechazado."""
    ok, errors = _validate_form_data_ranges(_base_form(
        householdSize=7, groceryDuration="monthly",
    ))
    assert ok is False
    h_errors = _household_errors(errors)
    assert any(e.get("accepted_range") == [1, 6] for e in h_errors)


# ---------------------------------------------------------------------------
# 2. Combinaciones aceptadas.
# ---------------------------------------------------------------------------
def test_household_6_monthly_accepted_boundary():
    """householdSize=6 + monthly → pasa (límite inferior del cap del wizard)."""
    ok, errors = _validate_form_data_ranges(_base_form(
        householdSize=6, groceryDuration="monthly",
    ))
    h_errors = _household_errors(errors)
    # NO debe haber error de tipo composite (accepted_range=[1,6]).
    composite = [e for e in h_errors if e.get("accepted_range") == [1, 6]]
    assert not composite, f"P1-15: 6 + monthly NO debe rechazarse. Got: {composite}"


def test_household_8_weekly_accepted():
    """householdSize=8 + weekly → pasa (otro grocery_duration)."""
    ok, errors = _validate_form_data_ranges(_base_form(
        householdSize=8, groceryDuration="weekly",
    ))
    h_errors = _household_errors(errors)
    composite = [e for e in h_errors if e.get("accepted_range") == [1, 6]]
    assert not composite


def test_household_12_biweekly_accepted():
    """householdSize=12 + biweekly → pasa (escalado ~12×2 = 24× tolerable)."""
    ok, errors = _validate_form_data_ranges(_base_form(
        householdSize=12, groceryDuration="biweekly",
    ))
    h_errors = _household_errors(errors)
    composite = [e for e in h_errors if e.get("accepted_range") == [1, 6]]
    assert not composite


def test_household_3_monthly_accepted_normal_case():
    """householdSize=3 + monthly → caso normal (3×4 = 12× razonable)."""
    ok, errors = _validate_form_data_ranges(_base_form(
        householdSize=3, groceryDuration="monthly",
    ))
    assert ok is True or _household_errors(errors) == []


def test_household_1_monthly_accepted():
    """householdSize=1 + monthly → ok (1×4 = 4× muy razonable)."""
    ok, errors = _validate_form_data_ranges(_base_form(
        householdSize=1, groceryDuration="monthly",
    ))
    assert ok is True or _household_errors(errors) == []


# ---------------------------------------------------------------------------
# 3. Validación individual sigue funcionando primero.
# ---------------------------------------------------------------------------
def test_household_above_individual_cap_rejected_with_full_range():
    """householdSize=999 (excede cap individual de 12) → rechazo con
    `accepted_range=[1, 12]`, NO con el composite [1, 6]. La validación
    individual sigue activa."""
    ok, errors = _validate_form_data_ranges(_base_form(
        householdSize=999, groceryDuration="monthly",
    ))
    assert ok is False
    h_errors = _household_errors(errors)
    # Debe haber al menos un error con accepted_range=[1, 12] (el del cap individual).
    assert any(e.get("accepted_range") == [1, 12] for e in h_errors), \
        f"P1-15: cap individual debe rechazar 999 antes que el composite. Got: {h_errors}"


# ---------------------------------------------------------------------------
# 4. Edge cases en groceryDuration.
# ---------------------------------------------------------------------------
def test_groceryDuration_case_insensitive_for_guard():
    """`MONTHLY` mayúsculas también dispara el guard (defensa por
    `.strip().lower()` interno)."""
    ok, errors = _validate_form_data_ranges(_base_form(
        householdSize=8, groceryDuration="MONTHLY",
    ))
    assert ok is False
    h_errors = _household_errors(errors)
    assert any(e.get("accepted_range") == [1, 6] for e in h_errors)


def test_groceryDuration_with_whitespace_for_guard():
    """`'  monthly  '` con whitespace → strip + lower → dispara guard."""
    ok, errors = _validate_form_data_ranges(_base_form(
        householdSize=8, groceryDuration="  monthly  ",
    ))
    assert ok is False


def test_groceryDuration_missing_does_not_trigger_guard():
    """Sin groceryDuration en el form → guard composite NO dispara
    (validación individual de presencia se maneja en otro lugar)."""
    form = _base_form(householdSize=8)
    # Sin groceryDuration explícito.
    form.pop("groceryDuration", None)
    ok, errors = _validate_form_data_ranges(form)
    h_errors = _household_errors(errors)
    composite = [e for e in h_errors if e.get("accepted_range") == [1, 6]]
    assert not composite, (
        f"P1-15: sin groceryDuration el guard NO debe disparar. Got: {composite}"
    )


# ---------------------------------------------------------------------------
# 5. Payload del error: marker + reason.
# ---------------------------------------------------------------------------
def test_error_payload_includes_p1_15_marker_in_reason():
    """El error debe incluir un campo `reason` con el marker `P1-15` para
    que SRE pueda alertar específicamente y los frontend devs entiendan
    el rationale."""
    ok, errors = _validate_form_data_ranges(_base_form(
        householdSize=10, groceryDuration="monthly",
    ))
    h_errors = _household_errors(errors)
    composite = [e for e in h_errors if e.get("accepted_range") == [1, 6]]
    assert composite
    err = composite[0]
    assert "reason" in err
    assert "P1-15" in err["reason"]


def test_documentation_p1_15_in_source():
    """Comentario `[P1-15]` debe estar presente en el source de
    `_validate_form_data_ranges` para que un futuro maintainer entienda
    el rationale del guard composicional."""
    import inspect
    from routers import plans as plans_module
    src = inspect.getsource(plans_module._validate_form_data_ranges)
    assert "[P1-15]" in src or "P1-15" in src
