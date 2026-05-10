"""[P1-FORM-13] Drift test: `SENTINELS` (frontend) ↔ `_SENTINEL_NONE_VALUES` (backend).

Cierra el último drift de safety médica del formulario.

Bug latente cubierto:
  El frontend (`frontend/src/config/sentinels.js`) declara los strings exclusivos
  que el wizard usa para "no aplica" en los 4 multi-select chip-based:
    SENTINELS = {
      allergies: 'Ninguna',
      medicalConditions: 'Ninguna',
      dislikes: 'Ninguno',
      struggles: 'Ninguno',
    }
  El backend (`graph_orchestrator._SENTINEL_NONE_VALUES`) reconoce esos mismos
  valores en lowercase para descartar `other*` cuando el array contiene un
  sentinel exclusivo (ver `_merge_other_text_fields`, P0-FORM-1).

  Si un futuro PR cambia el copy en frontend (ej. "Ninguna" → "Sin alergia") sin
  actualizar `_SENTINEL_NONE_VALUES`, la detección de exclusividad se rompe
  silenciosamente y reaparece la contradicción de P0-FORM-1: el array llega al
  LLM como `["Sin alergia", "Maní"]` con AMBOS verdaderos. Riesgo de safety
  médica directa para alergias y condiciones médicas.

  El test `test_p0_form_6_required_fields_sync.py` cubre el drift de campos
  requeridos pero NO los sentinels; este archivo cierra ese hueco.

Cobertura:
  1. Cada `SENTINELS.<field>` (frontend) `.lower()` ∈ `_SENTINEL_NONE_VALUES`.
  2. Los extras del backend están limitados a un allowlist documentado
     (`"none"` para clientes legacy/no oficiales en inglés).
  3. Los 4 fields multi-select chip-based esperados están presentes en SENTINELS.
  4. `SENTINEL_VALUES` (export derivado) es consistente con `set(SENTINELS.values())`.
  5. Sanity tests del parser sintetizan drift y verifican detección.

NOTA: el parser usa regex sobre el texto del archivo .js — no requiere intérprete
JS. Si el formato del archivo cambia (e.g., se añade tipado TypeScript, se mueve
a JSON, se cambia la sintaxis a una clase), actualizar los regex.
"""
import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Locator del archivo frontend
# ---------------------------------------------------------------------------
# [P3-3 · 2026-05-10] Tras migración tests root → tests/ (P3-CANDIDATE-B),
# `__file__.parent` es `backend/tests/`. Necesitamos subir DOS niveles para
# llegar al root del monorepo (`MealfitRD.IA/`) donde vive `frontend/` sibling.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
_SENTINELS_JS_PATH = _REPO_ROOT / "frontend" / "src" / "config" / "sentinels.js"

# ---------------------------------------------------------------------------
# Allowlist de extras legítimos en backend pero no en frontend.
# El backend acepta `"none"` (inglés) por defensa para clientes legacy / no
# oficiales que no pasaron por el wizard español. Está documentado al lado
# del frozenset (`graph_orchestrator.py:_SENTINEL_NONE_VALUES`).
# ---------------------------------------------------------------------------
_BACKEND_LEGACY_ALIASES = frozenset({"none"})

# Fields que el wizard espera con sentinel exclusivo. Mapping debe estar completo.
_EXPECTED_FIELDS_WITH_SENTINEL = frozenset({
    "allergies", "medicalConditions", "dislikes", "struggles",
})


# ---------------------------------------------------------------------------
# Parser: extrae los pares `key: 'value'` del bloque `SENTINELS` en sentinels.js
# ---------------------------------------------------------------------------
_SENTINELS_BLOCK_PATTERN = re.compile(
    r"export\s+const\s+SENTINELS\s*=\s*Object\.freeze\(\s*\{(?P<body>.*?)\}\s*\)\s*;",
    re.DOTALL,
)
# Match de pares key: 'value' tolerando comentarios entre líneas.
_KEY_VALUE_PATTERN = re.compile(
    r"\b(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*'(?P<value>[^']+)'",
)
_SENTINEL_VALUES_BLOCK_PATTERN = re.compile(
    r"export\s+const\s+SENTINEL_VALUES\s*=\s*Object\.freeze\(",
)


def _parse_sentinels_js(text: str) -> dict:
    """Devuelve `{field: sentinel_string}` extraído del bloque SENTINELS."""
    block = _SENTINELS_BLOCK_PATTERN.search(text)
    if not block:
        raise AssertionError(
            "No se encontró el bloque `export const SENTINELS = Object.freeze({...})` "
            "en sentinels.js. Si el formato cambió, actualizar el regex."
        )
    body = block.group("body")
    pairs: dict = {}
    for m in _KEY_VALUE_PATTERN.finditer(body):
        pairs[m.group("key")] = m.group("value")
    return pairs


def _read_sentinels_js() -> str:
    if not _SENTINELS_JS_PATH.exists():
        pytest.skip(f"sentinels.js no encontrado en {_SENTINELS_JS_PATH}")
    return _SENTINELS_JS_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_sentinels_js_file_exists():
    """Sanity: el archivo SSOT del frontend está donde esperamos."""
    assert _SENTINELS_JS_PATH.exists(), (
        f"sentinels.js no existe en {_SENTINELS_JS_PATH}. Si la estructura del "
        f"repo cambió, actualizar `_SENTINELS_JS_PATH`."
    )


def test_parser_extracts_all_expected_field_keys():
    """`SENTINELS` declara una entrada por cada uno de los 4 fields multi-select
    chip-based del wizard. Si falta uno, hay un campo cuyo sentinel quedará
    huérfano de detección backend."""
    parsed = _parse_sentinels_js(_read_sentinels_js())
    actual_fields = set(parsed.keys())
    missing = _EXPECTED_FIELDS_WITH_SENTINEL - actual_fields
    extra = actual_fields - _EXPECTED_FIELDS_WITH_SENTINEL
    assert not missing, (
        f"SENTINELS no cubre fields esperados: {sorted(missing)}. "
        f"Si añadiste un nuevo multi-select con sentinel exclusivo, agregalo "
        f"a SENTINELS y actualiza este test."
    )
    assert not extra, (
        f"SENTINELS tiene fields no documentados: {sorted(extra)}. Si retiraste "
        f"un field, también actualiza este test y el wizard."
    )


def test_frontend_sentinels_subset_of_backend_lowercased():
    """Cada valor de `SENTINELS` (case-insensitive) DEBE existir en
    `_SENTINEL_NONE_VALUES` del backend.

    Si el frontend renombra el copy ("Ninguna" → "Sin alergia") sin actualizar
    el backend, este test falla y previene la regresión de P0-FORM-1."""
    from graph_orchestrator import _SENTINEL_NONE_VALUES

    parsed = _parse_sentinels_js(_read_sentinels_js())
    frontend_values = {v.strip().lower() for v in parsed.values()}
    drift = frontend_values - set(_SENTINEL_NONE_VALUES)
    assert not drift, (
        f"Sentinels del frontend (lowercased) no reconocidos por el backend: "
        f"{sorted(drift)}.\n"
        f"Frontend SSOT: {sorted(frontend_values)}\n"
        f"Backend SSOT:  {sorted(_SENTINEL_NONE_VALUES)}\n"
        f"Si cambiaste el copy en sentinels.js, añade el valor en lowercase "
        f"al frozenset `_SENTINEL_NONE_VALUES` en graph_orchestrator.py "
        f"o el sentinel deja de descartar `other*` y reaparece P0-FORM-1."
    )


def test_backend_extras_only_legacy_aliases():
    """Los extras del backend (no presentes en frontend SSOT) deben estar en
    el allowlist documentado de aliases legacy. Cualquier otra divergencia
    indica drift bidireccional (backend declarando un sentinel que el wizard
    nunca emite)."""
    from graph_orchestrator import _SENTINEL_NONE_VALUES

    parsed = _parse_sentinels_js(_read_sentinels_js())
    frontend_values = {v.strip().lower() for v in parsed.values()}
    backend_extras = set(_SENTINEL_NONE_VALUES) - frontend_values
    unexpected = backend_extras - _BACKEND_LEGACY_ALIASES
    assert not unexpected, (
        f"Backend declara sentinels que el frontend nunca emite y no están "
        f"documentados como legacy: {sorted(unexpected)}.\n"
        f"Allowlist actual: {sorted(_BACKEND_LEGACY_ALIASES)}.\n"
        f"Si añadiste un alias legacy nuevo, actualiza `_BACKEND_LEGACY_ALIASES` "
        f"en este test con la justificación."
    )


def test_sentinel_values_export_is_dict_values_dedup():
    """El export `SENTINEL_VALUES` debe ser el set deduplicado de `Object.values(SENTINELS)`.

    Sanity sobre la consistencia interna del archivo: si alguien añade un
    valor crudo a `SENTINEL_VALUES` (en vez de derivarlo de SENTINELS) o lo
    reemplaza por una lista hardcoded, el invariante se rompe. Solo verificamos
    que el bloque exista con la forma `Array.from(new Set(Object.values(SENTINELS)))`.
    """
    text = _read_sentinels_js()
    assert _SENTINEL_VALUES_BLOCK_PATTERN.search(text), (
        "No se encontró el export `SENTINEL_VALUES = Object.freeze(...)` en sentinels.js. "
        "Si la implementación cambió, actualiza el regex."
    )
    # Verificamos también que la derivación textualmente use Object.values(SENTINELS):
    assert "Object.values(SENTINELS)" in text, (
        "`SENTINEL_VALUES` ya no se deriva de `Object.values(SENTINELS)`. "
        "Si la derivación cambió, actualiza este test (riesgo de drift dentro "
        "del mismo archivo: SENTINELS y SENTINEL_VALUES desincronizados)."
    )


def test_backend_set_is_immutable_frozenset():
    """`_SENTINEL_NONE_VALUES` debe ser frozenset. Garantiza que el invariante
    no se pueda evadir con `_SENTINEL_NONE_VALUES.add(...)` runtime."""
    from graph_orchestrator import _SENTINEL_NONE_VALUES
    assert isinstance(_SENTINEL_NONE_VALUES, frozenset), (
        f"_SENTINEL_NONE_VALUES debe ser frozenset (es {type(_SENTINEL_NONE_VALUES).__name__}) "
        f"para que la mutación runtime no evada el invariante."
    )


# ---------------------------------------------------------------------------
# Sanity del parser: drift sintético.
# ---------------------------------------------------------------------------
def test_parser_extracts_correctly_from_canonical_format():
    """Sanity: el parser extrae los 4 pares actuales correctamente.

    Si este test falla pero los anteriores pasan, el problema está en el
    parser (regex desactualizado), no en drift real."""
    parsed = _parse_sentinels_js(_read_sentinels_js())
    assert parsed.get("allergies"), "Esperaba `allergies` en SENTINELS"
    assert parsed.get("medicalConditions"), "Esperaba `medicalConditions` en SENTINELS"
    assert parsed.get("dislikes"), "Esperaba `dislikes` en SENTINELS"
    assert parsed.get("struggles"), "Esperaba `struggles` en SENTINELS"
    # Los valores actuales esperados (snapshot del SSOT en la fecha del fix).
    # Si cambian deliberadamente, este snapshot debe actualizarse — y el test
    # de drift (test_frontend_sentinels_subset_of_backend_lowercased) verificará
    # que el backend siga aceptándolos.
    assert parsed["allergies"] == "Ninguna"
    assert parsed["medicalConditions"] == "Ninguna"
    assert parsed["dislikes"] == "Ninguno"
    assert parsed["struggles"] == "Ninguno"


def test_parser_detects_synthetic_drift():
    """Sanity inverso: si el parser recibe contenido con un nuevo sentinel
    inventado, debe extraerlo. Garantiza que no devuelva dict vacío
    silenciosamente bajo input malformado."""
    fake_js = (
        "export const SENTINELS = Object.freeze({\n"
        "    allergies: 'Sin alergia',\n"
        "    medicalConditions: 'Ninguna',\n"
        "    dislikes: 'Ninguno',\n"
        "    struggles: 'Ninguno',\n"
        "});\n"
    )
    parsed = _parse_sentinels_js(fake_js)
    assert parsed["allergies"] == "Sin alergia", (
        f"Parser debió capturar el sentinel sintético; vio {parsed!r}"
    )


def test_parser_raises_on_missing_block():
    """Si el archivo perdió el bloque SENTINELS (renombrado, movido, eliminado),
    el parser debe fallar explícitamente — no devolver dict vacío que pasaría
    el test de subset trivialmente."""
    with pytest.raises(AssertionError, match="No se encontró"):
        _parse_sentinels_js("// archivo sin bloque SENTINELS\n")
