"""[P3-5 · 2026-05-08] Test de paridad cross-language para BIO_RANGES.

Bug original (audit 2026-05-07):
  Los rangos biométricos (`age`, `weight_kg`, `height_cm`, `bodyFat`,
  `household`) estaban duplicados en:
    - Backend: `_BIO_RANGES` en `routers/plans.py:539`
    - Frontend: `BIO_RANGES` en `config/formValidation.js:198`
  Sin guardrail que detecte drift. Si alguien bumpea un lado:
    - Frontend stricter que backend → UX bloquea valores válidos del backend.
    - Backend stricter que frontend → form acepta y backend rechaza tras
      30-90s del pipeline (UX terrible).

  La premisa específica del audit ("UI chips 1..6 vs BIO_RANGES 1..12") era
  FALSO POSITIVO post-P0-12 — los chips de householdSize en QHousehold ya
  no existen (verificado: `householdSize` no aparece en
  `InteractiveQuestions.jsx`). El comentario stale fue actualizado.

  El gap accionable real: paridad entre BIO_RANGES de ambos lados.

Fix:
  1. Comentarios stale en frontend `formValidation.js:206-211` y backend
     `routers/plans.py:544-549` actualizados — referencian este test.
  2. Test parsea `formValidation.js` (regex) y `routers/plans.py` (regex)
     para extraer los rangos y comparar. Mismo patrón que `P3-NEW-A`
     (form_backend_parity_meta_test) y `P0-FORM-6` (required_fields).

Mapeo backend → frontend (los nombres difieren):
  - `age` ↔ `age`
  - `weight_kg` ↔ `weightKg`
  - `height_cm` ↔ `heightCm`
  - `bodyFat` ↔ `bodyFat`
  - `household` ↔ `household`

(weightLb/heightFt/heightIn son derivados solo-frontend para conversión
de unidades; el backend convierte lb→kg en `_coerce_numeric` y solo valida
el rango canónico kg.)

Cobertura:
  - Backend `_BIO_RANGES` se parsea correctamente.
  - Frontend `BIO_RANGES` se parsea correctamente.
  - Cada par de rangos (age/weight_kg/height_cm/bodyFat/household) tiene
    `min` y `max` numéricamente iguales.
  - El test pita con mensaje claro si hay drift, indicando AMBOS valores
    para diagnóstico inmediato.
"""
import pathlib
import re

import pytest


_BACKEND_PLANS = pathlib.Path(__file__).parent / "routers" / "plans.py"
_FRONTEND_FV = (
    pathlib.Path(__file__).parent.parent
    / "frontend" / "src" / "config" / "formValidation.js"
)


def _parse_backend_bio_ranges() -> dict[str, tuple[float, float]]:
    """Parsea `_BIO_RANGES = { "key": (min, max), ... }` de backend.

    Regex permisivo: acepta int o float, espacios variables, comentarios
    al final de cada línea. No usa AST porque la edición humana frecuente
    del bloque haría ast frágil ante un trailing comma o parens extras."""
    src = _BACKEND_PLANS.read_text(encoding="utf-8")
    # Aislar el dict.
    block_match = re.search(
        r"_BIO_RANGES\s*=\s*\{(.*?)\}",
        src,
        re.DOTALL,
    )
    assert block_match is not None, "_BIO_RANGES dict no encontrado en plans.py"
    block = block_match.group(1)
    # Cada entry: "key": (min, max),
    entry_re = re.compile(
        r'"(?P<key>[a-zA-Z_]+)"\s*:\s*\(\s*(?P<min>-?\d+(?:\.\d+)?)\s*,\s*(?P<max>-?\d+(?:\.\d+)?)\s*\)',
    )
    ranges: dict[str, tuple[float, float]] = {}
    for m in entry_re.finditer(block):
        ranges[m.group("key")] = (float(m.group("min")), float(m.group("max")))
    return ranges


def _parse_frontend_bio_ranges() -> dict[str, tuple[float, float]]:
    """Parsea `export const BIO_RANGES = { key: { min, max, ... }, ... }`.

    Regex extrae `key: { min: N, max: M, ... }`. Permisivo con whitespace
    y campos extra (step, unit) que ignoramos."""
    src = _FRONTEND_FV.read_text(encoding="utf-8")
    block_match = re.search(
        r"export\s+const\s+BIO_RANGES\s*=\s*\{(.*?)\n\};",
        src,
        re.DOTALL,
    )
    assert block_match is not None, (
        "BIO_RANGES dict no encontrado en formValidation.js. "
        "Si se renombró o movió, actualizar el regex de _parse_frontend_bio_ranges."
    )
    block = block_match.group(1)
    entry_re = re.compile(
        r"(?P<key>[a-zA-Z]+)\s*:\s*\{\s*"
        r"min\s*:\s*(?P<min>-?\d+(?:\.\d+)?)\s*,\s*"
        r"max\s*:\s*(?P<max>-?\d+(?:\.\d+)?)",
    )
    ranges: dict[str, tuple[float, float]] = {}
    for m in entry_re.finditer(block):
        ranges[m.group("key")] = (float(m.group("min")), float(m.group("max")))
    return ranges


# Mapeo entre nombres backend (snake_case) y frontend (camelCase).
# Solo los que existen en AMBOS lados — weightLb/heightFt/heightIn son
# derivaciones de unidad solo-frontend.
_PAIRS = [
    ("age", "age"),
    ("weight_kg", "weightKg"),
    ("height_cm", "heightCm"),
    ("bodyFat", "bodyFat"),
    ("household", "household"),
]


# ---------------------------------------------------------------------------
# 1. Parsers funcionan
# ---------------------------------------------------------------------------
def test_backend_bio_ranges_parses_all_keys():
    ranges = _parse_backend_bio_ranges()
    expected_keys = {"age", "weight_kg", "height_cm", "bodyFat", "household"}
    assert set(ranges.keys()) == expected_keys, (
        f"Backend _BIO_RANGES tiene keys {set(ranges.keys())}, esperado {expected_keys}. "
        f"Si se añadió un campo, actualizar `_PAIRS` en este test para incluirlo "
        f"en la auditoría de paridad."
    )


def test_frontend_bio_ranges_parses_all_keys():
    ranges = _parse_frontend_bio_ranges()
    # Frontend incluye derivaciones de unidad. Las 5 canónicas deben estar.
    canonical_keys = {pair[1] for pair in _PAIRS}
    missing = canonical_keys - set(ranges.keys())
    assert not missing, (
        f"Frontend BIO_RANGES no tiene keys {missing}. "
        f"Cualquier campo en _PAIRS debe existir en formValidation.js."
    )


# ---------------------------------------------------------------------------
# 2. Paridad de cada rango
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend_key,frontend_key", _PAIRS)
def test_bio_range_parity(backend_key, frontend_key):
    """Cada par (backend/frontend) debe tener min y max numéricamente
    iguales. Drift produce: form acepta y backend rechaza (UX terrible)
    o form bloquea valores válidos del backend."""
    backend_ranges = _parse_backend_bio_ranges()
    frontend_ranges = _parse_frontend_bio_ranges()
    b_min, b_max = backend_ranges[backend_key]
    f_min, f_max = frontend_ranges[frontend_key]
    assert b_min == f_min, (
        f"DRIFT en `min` de {backend_key}↔{frontend_key}: "
        f"backend={b_min}, frontend={f_min}. "
        f"Subir/bajar AMBOS lados simultáneamente. Backend es SSOT pero "
        f"el frontend gateaba UX para feedback inmediato — sin paridad la "
        f"validación cliente desincroniza con la del servidor."
    )
    assert b_max == f_max, (
        f"DRIFT en `max` de {backend_key}↔{frontend_key}: "
        f"backend={b_max}, frontend={f_max}. "
        f"Si se quería bumpear el cap (ej. household 12→20 para hogares "
        f"masivos), actualizar AMBOS lados. Si solo se bumpeó el frontend, "
        f"el backend rechazará valores válidos del UI tras 30-90s del "
        f"pipeline. Si solo el backend, el form bloqueará valores válidos."
    )


# ---------------------------------------------------------------------------
# 3. Smoke: comentarios stale removidos
# ---------------------------------------------------------------------------
def test_stale_chips_comment_removed_from_frontend():
    """Pre-P3-5, formValidation.js:207 decía 'wizard solo expone chips 1..6
    (QHousehold)' — pero post-P0-12 esos chips ya no existen. El comentario
    stale fue actualizado para reflejar la realidad."""
    src = _FRONTEND_FV.read_text(encoding="utf-8")
    assert "El wizard solo expone chips 1..6" not in src, (
        "Comentario stale `El wizard solo expone chips 1..6 (QHousehold)` "
        "todavía presente en formValidation.js. P0-12 eliminó esos chips; "
        "actualizar el comentario para no confundir al próximo dev."
    )


def test_stale_chips_comment_removed_from_backend():
    """Análogo en el lado backend."""
    src = _BACKEND_PLANS.read_text(encoding="utf-8")
    assert "Wizard ofrece chips 1..6" not in src, (
        "Comentario stale `Wizard ofrece chips 1..6` todavía presente en "
        "routers/plans.py. P0-12 eliminó esos chips."
    )


def test_parity_test_referenced_in_both_sources():
    """Los comentarios actualizados deben referenciar este test para que
    el próximo dev sepa cómo verificar paridad."""
    backend_src = _BACKEND_PLANS.read_text(encoding="utf-8")
    frontend_src = _FRONTEND_FV.read_text(encoding="utf-8")
    assert "test_p3_5_bio_ranges_parity" in backend_src, (
        "El comentario en routers/plans.py debería mencionar el test de paridad."
    )
    assert "test_p3_5_bio_ranges_parity" in frontend_src, (
        "El comentario en formValidation.js debería mencionar el test de paridad."
    )
