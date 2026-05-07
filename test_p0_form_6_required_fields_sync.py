"""
Tests P0-FORM-6: sincronía de `REQUIRED_FORM_FIELDS` entre frontend y backend.

Bug original:
  El frontend (`frontend/src/config/formValidation.js REQUIRED_FORM_FIELDS`)
  gateaba 19 campos como required (botón "Siguiente" bloqueado, asterisco rojo,
  chips), pero el backend (`_REQUIRED_FORM_FIELDS` en `routers/plans.py`) solo
  validaba ~12. Inconsistencia real:
    - Cliente legacy / hidratación rota / scraper que omitía `scheduleType` /
      `cookingTime` / `budget` / `sleepHours` / `stressLevel` /
      `dislikes` / `struggles` entraba al pipeline silenciosamente.
    - Pero un cliente que enviara esos mismos campos con un valor bogus era
      rechazado por `_NON_CRITICAL_ENUM_VALIDATIONS` con 422.
  Resultado: "ausente pasa silencioso, presente-bogus rechaza" → plan
  generado con señales vacías de timing/conducta, LLM defaulteando sin
  telemetría, calidad degradada para cualquier path no-wizard.

Fix:
  7 campos añadidos a `_REQUIRED_FORM_FIELDS` en `routers/plans.py`:
    `scheduleType`, `cookingTime`, `budget`, `sleepHours`, `stressLevel`,
    `dislikes`, `struggles`.
  `dietType` permanece OUT del backend por compat con perfiles legacy
  (`_DIET_TYPE_LEGACY_ACCEPTED` en plans.py); decisión documentada.
"""
import pytest

from routers.plans import _REQUIRED_FORM_FIELDS, _validate_form_data_min


# ---------------------------------------------------------------------------
# 1. Snapshot del set canónico — guard de regresión contra removals accidentales.
# ---------------------------------------------------------------------------
# Espejo del SSOT del frontend (`REQUIRED_FORM_FIELDS` en formValidation.js)
# MENOS `dietType` (deliberadamente fuera del backend; ver comentario del tuple
# en plans.py). Si el frontend añade un campo, este test revienta intencional-
# mente para forzar el sync (o documentar la exención como `dietType`).
_BACKEND_EXPECTED_REQUIRED = frozenset({
    "gender", "age", "height", "weight", "weightUnit", "activityLevel",
    "scheduleType", "sleepHours", "stressLevel", "cookingTime", "budget",
    "householdSize", "groceryDuration",
    "allergies", "dislikes", "medicalConditions",
    "mainGoal", "struggles", "motivation",
})


def test_required_fields_set_es_exactamente_el_esperado():
    """Si esto falla, alguien añadió/quitó un campo sin actualizar el SSOT
    del frontend o este test. Sincronizar AMBOS lados antes de mergear."""
    actual = frozenset(_REQUIRED_FORM_FIELDS)
    extra = actual - _BACKEND_EXPECTED_REQUIRED
    missing = _BACKEND_EXPECTED_REQUIRED - actual
    assert not extra, f"Campo nuevo en backend sin sync con frontend: {extra}"
    assert not missing, f"Campo del frontend sin enforcement backend: {missing}"


def test_diet_type_NO_es_required_por_compat_legacy():
    """`dietType` queda fuera deliberadamente. `_DIET_TYPE_LEGACY_ACCEPTED`
    documenta variantes ES históricas en `health_profile.dietType`
    ("Omnívora", "vegetariana", etc.). Hacerlo required-presence rompería
    rehidratación de perfiles legacy sin beneficio de safety: downstream
    `_get_fast_filtered_catalogs` defaultea a balanced (catálogo completo,
    benigno). Si en el futuro se decide subirlo a required, eliminar este
    test Y el set legacy."""
    assert "dietType" not in _REQUIRED_FORM_FIELDS


# ---------------------------------------------------------------------------
# 2. Cada uno de los 7 campos nuevos rechaza con 422 si está ausente.
# ---------------------------------------------------------------------------
def _build_complete_payload() -> dict:
    """Payload completo que pasa `_validate_form_data_min`. Cada test
    individual omite UN campo a la vez para verificar el rechazo aislado."""
    return {
        "age": 30, "mainGoal": "lose_fat", "weight": 70, "height": 170,
        "gender": "male", "activityLevel": "moderate", "weightUnit": "kg",
        "householdSize": 1, "groceryDuration": "weekly",
        "motivation": "Recuperar mi energía diaria.",
        "allergies": ["Ninguna"], "medicalConditions": ["Ninguna"],
        "scheduleType": "standard", "cookingTime": "30min",
        "budget": "medium", "sleepHours": "7-8 horas",
        "stressLevel": "Moderado",
        "dislikes": ["Ninguno"], "struggles": ["Ninguno"],
    }


def test_payload_completo_pasa_validacion():
    """Sanity check: el payload base sin omisiones pasa. Si esto falla, los
    tests siguientes son inválidos (omiten encima de un payload ya inválido)."""
    ok, missing = _validate_form_data_min(_build_complete_payload())
    assert ok is True, f"payload base debe pasar; missing={missing}"


_NEW_REQUIRED_P0_FORM_6 = (
    "scheduleType", "cookingTime", "budget", "sleepHours", "stressLevel",
    "dislikes", "struggles",
)


@pytest.mark.parametrize("field", _NEW_REQUIRED_P0_FORM_6)
def test_omitir_campo_nuevo_dispara_missing_required_fields(field):
    """Cada campo individual ausente → `_validate_form_data_min` lo reporta
    en `missing`. Cubre el modo de fallo "cliente no oficial / hidratación
    rota omite UN campo y el resto está bien"."""
    payload = _build_complete_payload()
    payload.pop(field)
    ok, missing = _validate_form_data_min(payload)
    assert ok is False, f"campo {field!r} ausente debió rechazarse"
    assert field in missing, f"missing={missing} no incluye {field!r}"


@pytest.mark.parametrize("field", _NEW_REQUIRED_P0_FORM_6)
def test_string_vacio_se_trata_como_ausente(field):
    """Para los campos string (no array), `""` y whitespace-only equivalen a
    ausente — bug observado en clientes que mandaban "" tras un reset parcial
    del wizard."""
    if field in ("dislikes", "struggles"):
        pytest.skip("array fields — el caso vacío se cubre en otro test")
    payload = _build_complete_payload()
    payload[field] = ""
    ok, missing = _validate_form_data_min(payload)
    assert ok is False
    assert field in missing


@pytest.mark.parametrize("field", _NEW_REQUIRED_P0_FORM_6)
def test_none_se_trata_como_ausente(field):
    """`None` explícito (cliente que envía `null` en JSON) debe rechazarse
    igual que ausencia."""
    payload = _build_complete_payload()
    payload[field] = None
    ok, missing = _validate_form_data_min(payload)
    assert ok is False
    assert field in missing


# ---------------------------------------------------------------------------
# 3. Comportamiento específico de arrays (`dislikes`, `struggles`).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field", ("dislikes", "struggles"))
def test_array_vacio_pasa_validacion(field):
    """A diferencia de `allergies`/`medicalConditions` (safety-críticos,
    en `_REQUIRED_NONEMPTY_ARRAY_FIELDS`), `dislikes` y `struggles` aceptan
    `[]` como answer válida. Downstream el LLM trata `[]` como "sin
    preferencias declaradas" — benigno. Solo PRESENCIA del key se exige."""
    payload = _build_complete_payload()
    payload[field] = []
    ok, missing = _validate_form_data_min(payload)
    assert ok is True, f"{field}=[] debe pasar; missing={missing}"


# ---------------------------------------------------------------------------
# 4. Bug original reproducido: payload sin scheduleType pasa silencioso ANTES
#    del fix; ahora rechaza.
# ---------------------------------------------------------------------------
def test_escenario_bug_original_payload_sin_scheduleType_se_rechaza():
    """
    PRE-FIX: cliente legacy enviaba payload completo de biometría pero omitía
    `scheduleType` → `_validate_form_data_min` pasaba → pipeline corría →
    `build_time_context` recibía None → LLM generaba plan con horario
    genérico sin reflejar el ritmo del usuario. Sin telemetría.

    POST-FIX: rechazo en <1ms con 422 accionable. Frontend redirige al step
    QSchedule via `findFirstIncompleteField`.
    """
    payload = _build_complete_payload()
    payload.pop("scheduleType")
    ok, missing = _validate_form_data_min(payload)
    assert ok is False
    assert "scheduleType" in missing


def test_escenario_bug_original_payload_sin_dislikes_se_rechaza():
    """Mismo modo de fallo pero con array: cliente que omitía `dislikes`
    pasaba silenciosamente. Ahora rechaza."""
    payload = _build_complete_payload()
    payload.pop("dislikes")
    ok, missing = _validate_form_data_min(payload)
    assert ok is False
    assert "dislikes" in missing
