"""[P3-NEW-A · 2026-05-08] Meta-test de paridad form ↔ backend.

Cierra la asimetría documentada en el plan de auditoría 2026-05-08: existían
3 tests separados (test_p0_form_6, test_p1_form_14, test_p1_form_13 +
test_p3_form_sentinel) que verifican drift cross-language para `REQUIRED_FORM_FIELDS`,
`SUPPLEMENTS` y sentinels respectivamente. Pero NO había cobertura para los
otros enums exportados por `formValidation.js`:
  - `DIET_TYPES` ↔ `_DIET_TYPE_CANONICAL` (con `_DIET_TYPE_LEGACY_ACCEPTED` documentado)
  - `ACTIVITY_LEVELS` ↔ `_ACTIVITY_LEVEL_ENUM`
  - `MAIN_GOALS` ↔ `_MAIN_GOAL_ENUM`
  - `BIO_RANGES` ↔ `_BIO_RANGES`
  - `FIELD_LABELS` cubre `REQUIRED_FORM_FIELDS` (self-consistency)

Cada test falla con mensaje accionable indicando qué archivo + variable
sincronizar en cada lado.

Convenciones (siguiendo el patrón de test_p0_form_6 y test_p1_form_14):
  - Lectura del JS con regex tolerante a whitespace/comentarios.
  - Skip si el archivo no existe (no asumimos checkout completo del frontend).
  - Falla explícito si el regex no encuentra el bloque (no devuelve set vacío
    que pasaría tests de subset trivialmente).
"""
import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# File locator
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
_FORM_VALIDATION_JS = _REPO_ROOT / "frontend" / "src" / "config" / "formValidation.js"


def _read_form_validation_js() -> str:
    if not _FORM_VALIDATION_JS.exists():
        pytest.skip(f"formValidation.js no existe en {_FORM_VALIDATION_JS}")
    return _FORM_VALIDATION_JS.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Regex parsers (mismo patrón que test_p1_form_14)
# ---------------------------------------------------------------------------
_QUOTED_STRING = re.compile(r"'([^']+)'")

# `export const NAME = Object.freeze([ ... ]);` con DOTALL para multilínea.
def _make_freeze_array_pattern(name: str) -> re.Pattern:
    return re.compile(
        rf"export\s+const\s+{re.escape(name)}\s*=\s*Object\.freeze\(\s*\[(?P<body>.*?)\]\s*\)\s*;",
        re.DOTALL,
    )


_DIET_TYPES_PATTERN = _make_freeze_array_pattern("DIET_TYPES")
_ACTIVITY_LEVELS_PATTERN = _make_freeze_array_pattern("ACTIVITY_LEVELS")
_MAIN_GOALS_PATTERN = _make_freeze_array_pattern("MAIN_GOALS")

# `export const BIO_RANGES = { age: { min: ..., max: ... }, ... };`
_BIO_RANGES_BLOCK_PATTERN = re.compile(
    r"export\s+const\s+BIO_RANGES\s*=\s*\{(?P<body>.*?)\}\s*;",
    re.DOTALL,
)
# Captura `key: { min: <num>, max: <num>, ...}` dentro del bloque.
_BIO_RANGE_ENTRY_PATTERN = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\{[^}]*?min\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*max\s*:\s*(-?\d+(?:\.\d+)?)",
    re.MULTILINE,
)

# `export const FIELD_LABELS = { gender: 'Sexo biológico', ... };`
_FIELD_LABELS_BLOCK_PATTERN = re.compile(
    r"export\s+const\s+FIELD_LABELS\s*=\s*\{(?P<body>.*?)\}\s*;",
    re.DOTALL,
)
# Match `key: 'value',` — keys son bare identifiers.
_FIELD_LABEL_KEY_PATTERN = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*'",
    re.MULTILINE,
)


def _parse_freeze_array(text: str, pattern: re.Pattern, var_name: str) -> list[str]:
    block = pattern.search(text)
    if not block:
        raise AssertionError(
            f"No se encontró el bloque `export const {var_name} = Object.freeze([...])` "
            f"en formValidation.js. Si el formato cambió, actualiza el regex."
        )
    return _QUOTED_STRING.findall(block.group("body"))


def _parse_bio_ranges(text: str) -> dict[str, tuple[float, float]]:
    block = _BIO_RANGES_BLOCK_PATTERN.search(text)
    if not block:
        raise AssertionError(
            "No se encontró el bloque `export const BIO_RANGES = {...}` en "
            "formValidation.js. Si el formato cambió, actualiza el regex."
        )
    out: dict[str, tuple[float, float]] = {}
    for match in _BIO_RANGE_ENTRY_PATTERN.finditer(block.group("body")):
        key, lo, hi = match.group(1), match.group(2), match.group(3)
        # Preservar tipo (int si no tiene decimal, float si sí). Mismo criterio
        # que el backend para que la comparación sea simétrica.
        out[key] = (
            float(lo) if "." in lo else int(lo),
            float(hi) if "." in hi else int(hi),
        )
    return out


def _parse_field_labels_keys(text: str) -> set[str]:
    block = _FIELD_LABELS_BLOCK_PATTERN.search(text)
    if not block:
        raise AssertionError(
            "No se encontró el bloque `export const FIELD_LABELS = {...}` en "
            "formValidation.js. Si el formato cambió, actualiza el regex."
        )
    return set(_FIELD_LABEL_KEY_PATTERN.findall(block.group("body")))


# ===========================================================================
# 1. DIET_TYPES (frontend) ↔ _DIET_TYPE_CANONICAL (backend)
# ===========================================================================
def test_diet_types_frontend_matches_backend_canonical():
    """`DIET_TYPES` (frontend, lower-case canonical) debe ser exactamente igual
    a `_DIET_TYPE_CANONICAL` (backend). El backend acepta también
    `_DIET_TYPE_LEGACY_ACCEPTED` (variantes ES históricas) por compat con
    perfiles pre-migración, pero el wizard solo emite los canónicos.

    Si difieren:
      - Frontend solo: el wizard ofrece un chip que el backend no reconoce
        como canónico → cae a fallback legacy o `balanced`. UX confusa.
      - Backend solo: nuevo canónico añadido al backend pero el wizard no lo
        ofrece → usuarios pre-existentes con ese valor ven un chip "ninguno
        seleccionado" tras hidratación.
    """
    from routers.plans import _DIET_TYPE_CANONICAL

    frontend = set(_parse_freeze_array(
        _read_form_validation_js(), _DIET_TYPES_PATTERN, "DIET_TYPES",
    ))
    backend = set(_DIET_TYPE_CANONICAL)
    missing_in_backend = frontend - backend
    extra_in_backend = backend - frontend
    assert not missing_in_backend and not extra_in_backend, (
        f"Drift entre frontend DIET_TYPES y backend _DIET_TYPE_CANONICAL:\n"
        f"  Solo en frontend: {sorted(missing_in_backend)}\n"
        f"  Solo en backend:  {sorted(extra_in_backend)}\n"
        f"Acciones:\n"
        f"  1. Si añadiste un tipo de dieta canónico, actualiza:\n"
        f"     - DIET_TYPES en frontend/src/config/formValidation.js\n"
        f"     - _DIET_TYPE_CANONICAL en backend/routers/plans.py\n"
        f"     - _get_fast_filtered_catalogs en backend/constants.py\n"
        f"  2. Si es variante ES legacy (ej. 'omnivora'), agrégalo a\n"
        f"     _DIET_TYPE_LEGACY_ACCEPTED en backend (no a este test)."
    )


def test_diet_type_legacy_accepted_documented():
    """Anchor: el backend extiende el enum con `_DIET_TYPE_LEGACY_ACCEPTED`
    (variantes ES + diets adicionales como 'pescatarian'). Si en el futuro se
    añade un valor nuevo aquí, debe documentarse en el comentario de
    `_DIET_TYPE_ENUM`. Este test ancla el set actual; falla intencionalmente
    si crece para forzar review del trade-off (silently default downstream
    vs 422 estricto)."""
    from routers.plans import _DIET_TYPE_LEGACY_ACCEPTED

    expected = frozenset({
        "omnivora", "vegetariana", "vegana",
        "vegetariano", "vegano",
        "pescatarian", "pescetariano",
    })
    assert _DIET_TYPE_LEGACY_ACCEPTED == expected, (
        f"_DIET_TYPE_LEGACY_ACCEPTED cambió: {sorted(_DIET_TYPE_LEGACY_ACCEPTED)}.\n"
        f"Cada legacy bypasses el match canónico. Documenta la razón en el "
        f"comentario de plans.py:341 antes de actualizar este test."
    )


# ===========================================================================
# 2. ACTIVITY_LEVELS (frontend) ↔ _ACTIVITY_LEVEL_ENUM (backend)
# ===========================================================================
def test_activity_levels_frontend_matches_backend():
    """`ACTIVITY_LEVELS` (frontend) debe ser EXACTAMENTE `_ACTIVITY_LEVEL_ENUM`
    (backend). A diferencia de `dietType`, este enum es estricto en backend
    (sin capa legacy). Cualquier desalineación produce 422 al final del wizard."""
    from routers.plans import _ACTIVITY_LEVEL_ENUM

    frontend = set(_parse_freeze_array(
        _read_form_validation_js(), _ACTIVITY_LEVELS_PATTERN, "ACTIVITY_LEVELS",
    ))
    backend = set(_ACTIVITY_LEVEL_ENUM)
    assert frontend == backend, (
        f"Drift entre frontend ACTIVITY_LEVELS y backend _ACTIVITY_LEVEL_ENUM:\n"
        f"  Solo en frontend: {sorted(frontend - backend)}\n"
        f"  Solo en backend:  {sorted(backend - frontend)}\n"
        f"Acciones (si añadiste un nivel):\n"
        f"  1. ACTIVITY_LEVELS en frontend/src/config/formValidation.js\n"
        f"  2. _ACTIVITY_LEVEL_ENUM en backend/routers/plans.py\n"
        f"  3. ACTIVITY_MULTIPLIERS en backend/nutrition_calculator.py\n"
        f"     (sin esto el calculador defaultea a 1.55 silenciosamente)\n"
        f"  4. Chip + label en QActivityLevel (InteractiveQuestions.jsx)"
    )


# ===========================================================================
# 3. MAIN_GOALS (frontend) ↔ _MAIN_GOAL_ENUM (backend)
# ===========================================================================
def test_main_goals_frontend_matches_backend():
    """`MAIN_GOALS` (frontend) ↔ `_MAIN_GOAL_ENUM` (backend). Mismo patrón
    estricto que ACTIVITY_LEVELS."""
    from routers.plans import _MAIN_GOAL_ENUM

    frontend = set(_parse_freeze_array(
        _read_form_validation_js(), _MAIN_GOALS_PATTERN, "MAIN_GOALS",
    ))
    backend = set(_MAIN_GOAL_ENUM)
    assert frontend == backend, (
        f"Drift entre frontend MAIN_GOALS y backend _MAIN_GOAL_ENUM:\n"
        f"  Solo en frontend: {sorted(frontend - backend)}\n"
        f"  Solo en backend:  {sorted(backend - frontend)}\n"
        f"Acciones (si añadiste un goal):\n"
        f"  1. MAIN_GOALS en frontend/src/config/formValidation.js\n"
        f"  2. _MAIN_GOAL_ENUM en backend/routers/plans.py\n"
        f"  3. GOAL_ADJUSTMENTS y MACRO_SPLITS en backend/nutrition_calculator.py\n"
        f"  4. Chip + label en QMainGoal (InteractiveQuestions.jsx)"
    )


# ===========================================================================
# 4. BIO_RANGES (frontend) ↔ _BIO_RANGES (backend)
# ===========================================================================
# Mapping deliberado: el frontend usa camelCase + UI-only keys (heightFt,
# heightIn, weightLb) que el backend no necesita porque normaliza a cm/kg
# antes de validar. Este dict marca qué keys del frontend tienen un análogo
# numérico exacto en el backend.
_FRONTEND_TO_BACKEND_BIO_KEY = {
    "age": "age",
    "heightCm": "height_cm",
    "weightKg": "weight_kg",
    "bodyFat": "bodyFat",
    "household": "household",
}
# Keys del frontend que NO tienen contraparte en backend (UI-only conversions).
_FRONTEND_BIO_KEYS_UI_ONLY = frozenset({"heightFt", "heightIn", "weightLb"})


def test_bio_ranges_canonical_keys_match_backend():
    """Para cada par (frontend, backend) con la misma semántica, el rango
    numérico debe coincidir exactamente. Drift produce gating UI distinto al
    server-side: el botón "Siguiente" deja pasar valores que el backend
    rechaza, o viceversa."""
    from routers.plans import _BIO_RANGES

    frontend = _parse_bio_ranges(_read_form_validation_js())

    mismatches = []
    for f_key, b_key in _FRONTEND_TO_BACKEND_BIO_KEY.items():
        if f_key not in frontend:
            mismatches.append(f"  - frontend BIO_RANGES no tiene `{f_key}`")
            continue
        if b_key not in _BIO_RANGES:
            mismatches.append(f"  - backend _BIO_RANGES no tiene `{b_key}`")
            continue
        f_lo, f_hi = frontend[f_key]
        b_lo, b_hi = _BIO_RANGES[b_key]
        if (float(f_lo), float(f_hi)) != (float(b_lo), float(b_hi)):
            mismatches.append(
                f"  - {f_key} (frontend) = ({f_lo}, {f_hi}) ≠ "
                f"{b_key} (backend) = ({b_lo}, {b_hi})"
            )

    assert not mismatches, (
        "Drift en rangos biométricos frontend ↔ backend:\n"
        + "\n".join(mismatches)
        + "\n\nEl backend es source of truth (defense-in-depth contra clientes "
        + "no oficiales). Si ajustaste un rango, sincronizá:\n"
        + "  1. _BIO_RANGES en backend/routers/plans.py\n"
        + "  2. BIO_RANGES en frontend/src/config/formValidation.js\n"
        + "Mantener AMBOS lados igual para que el gating UI no sea inconsistente "
        + "con la validación server-side (UX rota: 'el wizard me deja avanzar y "
        + "luego el backend rechaza con 422')."
    )


def test_bio_ranges_ui_only_keys_present_and_sane():
    """Las keys UI-only (heightFt/heightIn/weightLb) no tienen análogo backend
    pero deben (a) existir en el frontend y (b) tener `min < max` no vacío.
    Sanity para evitar que un PR rompa el componente QMeasurements."""
    frontend = _parse_bio_ranges(_read_form_validation_js())
    for key in _FRONTEND_BIO_KEYS_UI_ONLY:
        assert key in frontend, (
            f"BIO_RANGES no exporta `{key}` (UI-only). QMeasurements rompe "
            f"al renderear si está ausente."
        )
        lo, hi = frontend[key]
        assert lo < hi, f"BIO_RANGES.{key} tiene min >= max ({lo} >= {hi})"


# ===========================================================================
# 5. FIELD_LABELS cubre REQUIRED_FORM_FIELDS (self-consistency frontend)
# ===========================================================================
def test_field_labels_covers_all_required_form_fields():
    """Cada entry de `REQUIRED_FORM_FIELDS` DEBE tener su label en `FIELD_LABELS`
    o el toast del wizard muestra el nombre técnico del field ('scheduleType'
    en vez de 'Tu horario cotidiano').

    Documentado en formValidation.js:138-140; este test lo enforcea para que
    un PR que añada un required sin label rompa CI."""
    from test_p0_form_6_required_fields_sync import _parse_frontend_required_fields

    text = _read_form_validation_js()
    required = set(_parse_frontend_required_fields(text))
    label_keys = _parse_field_labels_keys(text)

    missing_labels = required - label_keys
    assert not missing_labels, (
        f"REQUIRED_FORM_FIELDS sin label en FIELD_LABELS: {sorted(missing_labels)}\n"
        f"Sin labels, el toast 'Completa <FIELD>' muestra el snake_case técnico.\n"
        f"Acción: añadir entry a FIELD_LABELS en formValidation.js:140 con\n"
        f"un texto user-facing en español."
    )


# ===========================================================================
# 6. _REQUIRED_NONEMPTY_ARRAY_FIELDS ⊆ _REQUIRED_FORM_FIELDS (backend invariante)
# ===========================================================================
def test_required_nonempty_array_subset_of_required():
    """`_REQUIRED_NONEMPTY_ARRAY_FIELDS` (donde `[]` cuenta como ausente) debe
    ser subconjunto de `_REQUIRED_FORM_FIELDS` (donde "ausente" se chequea).
    Sin esto, un campo "no empty array" pero ni siquiera presence-required
    queda sin enforcement real."""
    from routers.plans import _REQUIRED_FORM_FIELDS, _REQUIRED_NONEMPTY_ARRAY_FIELDS

    extra = _REQUIRED_NONEMPTY_ARRAY_FIELDS - frozenset(_REQUIRED_FORM_FIELDS)
    assert not extra, (
        f"_REQUIRED_NONEMPTY_ARRAY_FIELDS contiene campos que NO están en "
        f"_REQUIRED_FORM_FIELDS: {sorted(extra)}\n"
        f"Esos campos no son enforced para presencia, así que el chequeo "
        f"'array vacío rechaza' nunca se ejecuta. Añade el campo a "
        f"_REQUIRED_FORM_FIELDS en routers/plans.py o quítalo del subset."
    )


# ===========================================================================
# 7. Sanity tests del parser (paralelos a los de test_p0_form_6 / test_p1_form_14)
# ===========================================================================
def test_parser_extracts_diet_types_minimum():
    """Sanity: el parser extrae al menos los 3 canónicos. Defensa contra
    regex roto que devolvería [] silenciosamente."""
    items = _parse_freeze_array(
        _read_form_validation_js(), _DIET_TYPES_PATTERN, "DIET_TYPES",
    )
    assert len(items) >= 3, (
        f"Parser de DIET_TYPES extrajo {len(items)} entries (esperaba ≥3). "
        f"Items: {items!r}"
    )


def test_parser_extracts_activity_levels_minimum():
    items = _parse_freeze_array(
        _read_form_validation_js(), _ACTIVITY_LEVELS_PATTERN, "ACTIVITY_LEVELS",
    )
    assert len(items) >= 5, (
        f"Parser de ACTIVITY_LEVELS extrajo {len(items)} entries (esperaba ≥5). "
        f"Items: {items!r}"
    )


def test_parser_extracts_main_goals_minimum():
    items = _parse_freeze_array(
        _read_form_validation_js(), _MAIN_GOALS_PATTERN, "MAIN_GOALS",
    )
    assert len(items) >= 4, (
        f"Parser de MAIN_GOALS extrajo {len(items)} entries (esperaba ≥4). "
        f"Items: {items!r}"
    )


def test_parser_bio_ranges_extracts_canonical_keys():
    """El parser de BIO_RANGES debe extraer al menos las 5 keys canónicas
    del frontend (las UI-only inclusive, total 8)."""
    parsed = _parse_bio_ranges(_read_form_validation_js())
    expected_keys = (
        set(_FRONTEND_TO_BACKEND_BIO_KEY.keys()) | _FRONTEND_BIO_KEYS_UI_ONLY
    )
    missing = expected_keys - set(parsed.keys())
    assert not missing, (
        f"Parser de BIO_RANGES no extrajo keys canónicas: {sorted(missing)}\n"
        f"Vio: {sorted(parsed.keys())}\n"
        f"Si el formato del archivo cambió, ajustar `_BIO_RANGE_ENTRY_PATTERN`."
    )


def test_parser_detects_synthetic_drift_diet_types():
    """Inverso: el parser captura entries añadidas en input sintético."""
    fake_js = (
        "export const DIET_TYPES = Object.freeze(['balanced', 'syntheticKeto']);\n"
    )
    items = _parse_freeze_array(fake_js, _DIET_TYPES_PATTERN, "DIET_TYPES")
    assert "syntheticKeto" in items, (
        f"Parser falló al detectar entry sintética; vio {items!r}"
    )


def test_parser_raises_on_missing_diet_types_block():
    with pytest.raises(AssertionError, match="No se encontró"):
        _parse_freeze_array(
            "// archivo sin DIET_TYPES\n", _DIET_TYPES_PATTERN, "DIET_TYPES",
        )


def test_parser_raises_on_missing_bio_ranges_block():
    with pytest.raises(AssertionError, match="No se encontró"):
        _parse_bio_ranges("// archivo sin BIO_RANGES\n")


def test_parser_raises_on_missing_field_labels_block():
    with pytest.raises(AssertionError, match="No se encontró"):
        _parse_field_labels_keys("// archivo sin FIELD_LABELS\n")
