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

[P3-B · 2026-05-07] Extensión: además del set hardcoded canónico, este módulo
ahora parsea `REQUIRED_FORM_FIELDS` directamente del archivo JS y verifica
la simetría cross-language. El set hardcoded sigue actuando como guard contra
removals accidentales del backend (forzando review del PR), pero el parser
captura drift introducido SOLO en el frontend — el caso que el set hardcoded
no veía. Mismo patrón que `test_p1_form_14_supplements_sync.py`.
"""
import re
from pathlib import Path

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


# ===========================================================================
# [P3-B · 2026-05-07] Drift cross-language: parsea JS, compara contra backend
# ---------------------------------------------------------------------------
# El set `_BACKEND_EXPECTED_REQUIRED` arriba es un anchor hardcoded — captura
# si el backend pierde un campo, pero NO ve si el frontend añade uno sin
# sincronizar. Estos tests adicionales parsean `formValidation.js` con regex
# (mismo patrón que test_p1_form_14_supplements_sync.py) y validan parity
# real cross-language.
#
# Excepción documentada: `dietType` está en el array JS pero deliberadamente
# fuera del backend (`_DIET_TYPE_LEGACY_ACCEPTED` cubre variantes ES legacy).
# El test la trata explícitamente como exclusión esperada.
# ===========================================================================

# [P3-3 · 2026-05-10] Tras migración tests root → tests/ (P3-CANDIDATE-B),
# `__file__.parent` es `backend/tests/`. Necesitamos subir DOS niveles para
# llegar al root del monorepo (`MealfitRD.IA/`) donde vive `frontend/` sibling.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
_FORM_VALIDATION_JS = _REPO_ROOT / "frontend" / "src" / "config" / "formValidation.js"

# Match: `export const REQUIRED_FORM_FIELDS = [ ... ];` (multi-línea, tolerando
# whitespace y comentarios entre entries).
_REQUIRED_FORM_FIELDS_BLOCK_PATTERN = re.compile(
    r"export\s+const\s+REQUIRED_FORM_FIELDS\s*=\s*\[(?P<body>.*?)\]\s*;",
    re.DOTALL,
)
_QUOTED_STRING = re.compile(r"'([^']+)'")

# Campos del frontend SSOT cuya ausencia en backend es intencional y documentada.
# Si en el futuro se decide subir alguno a required-presence backend, eliminar
# del set Y eliminar el test específico que protege la decisión (ej.
# `test_diet_type_NO_es_required_por_compat_legacy`).
# [P1-PLANSOURCE-REQUIRED · 2026-08-12] planSource: obligatoria en el wizard por
# decisión del owner (elegir libre vs Nevera es la primera decisión del plan),
# pero el backend sigue tratando ausente como generación libre — perfiles
# legacy y payloads de regeneración directa no traen el campo. Mismo patrón
# que dietType.
# [P1-APPMODE-REQUIRED · 2026-08-12] appMode: el paso 0 del wizard (¿plan o
# contador?) también obligatorio por el owner. Puro ruteo del wizard: el
# backend JAMÁS lee appMode del payload — el modo se conmuta por el endpoint
# /api/profile/plan-mode, no por el formulario.
# [P1-ARQ25-F4-FORM · 2026-09-03] mealOrganization: el perfil global de recurrencia (rutina /
# equilibrio / exploración) es obligatorio en el wizard (roadmap §6.7: «solo el perfil global y el
# ciclo principal son obligatorios»), pero el backend lo defaultea a `balanced` en
# `plan_policy.policy_from_form`: un cliente viejo (app nativa sin actualizar) sigue generando.
_FRONTEND_ONLY_BY_DESIGN = frozenset({"dietType", "planSource", "appMode", "mealOrganization"})


def _read_form_validation_js() -> str:
    if not _FORM_VALIDATION_JS.exists():
        pytest.skip(f"formValidation.js no existe en {_FORM_VALIDATION_JS}")
    return _FORM_VALIDATION_JS.read_text(encoding="utf-8")


def _parse_frontend_required_fields(text: str) -> list[str]:
    """Extrae la lista ordenada de strings dentro de
    `export const REQUIRED_FORM_FIELDS = [...]`."""
    block = _REQUIRED_FORM_FIELDS_BLOCK_PATTERN.search(text)
    if not block:
        raise AssertionError(
            "No se encontró el bloque `export const REQUIRED_FORM_FIELDS = [...]` "
            "en formValidation.js. Si el formato cambió (ej. migración a TS, "
            "rename a `REQUIRED_FIELDS`), actualiza `_REQUIRED_FORM_FIELDS_BLOCK_PATTERN`."
        )
    body = block.group("body")
    return _QUOTED_STRING.findall(body)


def test_form_validation_js_exists():
    """Sanity: el SSOT del frontend está donde lo apunta el comentario del
    backend. Si el frontend mueve archivos, esta ruta debe ajustarse."""
    assert _FORM_VALIDATION_JS.exists(), (
        f"formValidation.js no existe en {_FORM_VALIDATION_JS}. "
        f"Si la estructura del repo cambió, actualiza `_FORM_VALIDATION_JS`."
    )


def test_parser_extracts_minimum_count():
    """Sanity del parser: extrae al menos los 19 campos canónicos.
    Si el regex falla, este test falla antes de los de drift cross-language
    para que el output diga claramente "el parser no funciona" en vez de
    "drift detectado" sobre un set vacío."""
    items = _parse_frontend_required_fields(_read_form_validation_js())
    assert len(items) >= 19, (
        f"Parser sólo extrajo {len(items)} entries (esperaba ≥19). "
        f"Si el formato del archivo cambió, ajusta `_REQUIRED_FORM_FIELDS_BLOCK_PATTERN` "
        f"o `_QUOTED_STRING`. Items vistos: {items!r}"
    )


def test_parser_detects_synthetic_drift():
    """Sanity inverso: el parser captura entries añadidas en input sintético.
    Si esto falla, el parser podría no detectar drift real (false negative)."""
    fake_js = (
        "export const REQUIRED_FORM_FIELDS = [\n"
        "    'gender', 'age', 'syntheticNewField',\n"
        "];\n"
    )
    items = _parse_frontend_required_fields(fake_js)
    assert "syntheticNewField" in items, (
        f"Parser falló al detectar entry sintética; vio {items!r}"
    )


def test_parser_raises_on_missing_block():
    """Si el archivo perdió el bloque, el parser falla explícito (no devuelve
    set vacío que pasaría tests de subset trivialmente)."""
    with pytest.raises(AssertionError, match="No se encontró"):
        _parse_frontend_required_fields("// archivo sin REQUIRED_FORM_FIELDS\n")


def test_frontend_required_minus_design_exclusions_equals_backend():
    """Drift cross-language: parsea JS, descuenta exclusiones documentadas
    (`dietType` por legacy compat), y compara contra `_REQUIRED_FORM_FIELDS`
    del backend. Asimétrico: cualquier desalineación falla con mensaje
    accionable indicando qué lado tiene el surplus.

    Casos cubiertos:
      - PR añade campo SOLO al backend → `extra_in_backend` no vacío.
      - PR añade campo SOLO al frontend → `missing_in_backend` no vacío.
      - PR añade campo a AMBOS lados → ✅ pasa (ideal).
      - PR añade `dietType` al backend → falla con "exclusión `dietType`
        movida al backend; revisar test_diet_type_NO_es_required_por_compat_legacy
        antes de mergear".
    """
    frontend_text = _read_form_validation_js()
    frontend_set = frozenset(_parse_frontend_required_fields(frontend_text))
    backend_set = frozenset(_REQUIRED_FORM_FIELDS)

    # 1. Frontend ∖ exclusiones-de-diseño debe ser ⊆ backend.
    frontend_minus_excl = frontend_set - _FRONTEND_ONLY_BY_DESIGN
    missing_in_backend = frontend_minus_excl - backend_set
    assert not missing_in_backend, (
        f"Drift detectado: el frontend gateaba {sorted(missing_in_backend)} pero "
        f"el backend NO los valida.\n"
        f"Acciones:\n"
        f"  1. Si el campo debe ser required: añadir a `_REQUIRED_FORM_FIELDS` "
        f"     en `backend/routers/plans.py` Y al set `_BACKEND_EXPECTED_REQUIRED` "
        f"     arriba en este test.\n"
        f"  2. Si es exclusión intencional (compat legacy, downstream defaultea "
        f"     benigno): añadir a `_FRONTEND_ONLY_BY_DESIGN` arriba con razón en\n"
        f"     comentario."
    )

    # 2. Backend ⊆ frontend (todo campo backend debe estar en el wizard, sin
    # excepciones — un campo backend-only sin entry en el frontend significa
    # que el wizard no lo captura → 422 garantizado al primer submit).
    extra_in_backend = backend_set - frontend_set
    assert not extra_in_backend, (
        f"Drift detectado: el backend exige {sorted(extra_in_backend)} pero el "
        f"wizard frontend no los gateaba en `REQUIRED_FORM_FIELDS`.\n"
        f"Resultado en producción: el wizard permite avanzar sin estos campos, "
        f"el usuario llega al final, el POST falla con 422 — UX rota.\n"
        f"Acciones:\n"
        f"  1. Añadir el campo a `REQUIRED_FORM_FIELDS` en\n"
        f"     `frontend/src/config/formValidation.js` Y a `FIELD_LABELS`.\n"
        f"  2. Asegurar que algún step del wizard (`InteractiveAssessmentFlow.jsx`)\n"
        f"     declare `fields: ['<field>']` para que la nav-a-faltante apunte al\n"
        f"     step correcto."
    )


def test_exclusiones_intencionales_son_exactamente_las_documentadas():
    """Anchor: el set de `_FRONTEND_ONLY_BY_DESIGN` es EXACTO — crece solo con
    review explícito del trade-off (silently default downstream vs 422 estricto).

    Historial de reviews:
      - dietType (P3-NEW-4 · 2026-05-11): perfiles antiguos sin dietType siguen
        funcionando con default "balanced" en graph_orchestrator.
      - planSource (P1-PLANSOURCE-REQUIRED · 2026-08-12): obligatoria en el
        wizard por decisión del owner (libre vs Nevera es la primera decisión
        del plan); el backend trata ausente como generación libre — perfiles
        legacy y regeneración directa (useRegeneratePlan) no traen el campo y
        el downstream es benigno.
      - appMode (P1-APPMODE-REQUIRED · 2026-08-12): paso 0 del wizard, puro
        ruteo de rama (plan vs contador). El backend jamás lo lee del payload:
        el modo se conmuta vía /api/profile/plan-mode. Exigirlo backend-side
        sería exigir un campo que ningún caller no-wizard puede conocer.
    """
    assert _FRONTEND_ONLY_BY_DESIGN == frozenset({"dietType", "planSource", "appMode", "mealOrganization"}), (
        f"Conjunto de exclusiones cambió: {sorted(_FRONTEND_ONLY_BY_DESIGN)}.\n"
        f"Cada exclusión bypasses la red de safety cross-language. Documenta "
        f"la razón en el comentario de `_FRONTEND_ONLY_BY_DESIGN` Y en el "
        f"historial del docstring antes de actualizar este test."
    )
