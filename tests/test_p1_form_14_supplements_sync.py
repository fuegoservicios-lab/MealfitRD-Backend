"""[P1-FORM-14] Drift sync test: SSOT de `selectedSupplements` cross-language.

Cuatro sites del enum de suplementos deben mantenerse alineados:

  1. Frontend SSOT       : `SUPPLEMENTS` en `frontend/src/config/formValidation.js`.
  2. Frontend metadata UI: `SUPPLEMENT_META` en
                            `frontend/src/components/assessment/questions/InteractiveQuestions.jsx`.
  3. Backend API gate    : `_SUPPLEMENT_ENUM` en `backend/routers/plans.py`.
  4. Backend prompt      : `SUPPLEMENT_NAMES` en `backend/constants.py` (keys).

Bug latente cubierto:
  Antes, `QSupplements` (componente React) hardcodeaba 12 entries `{val, label,
  emoji}` literalmente y el backend tenía sus 12 keys repetidas en 2 sets. Si
  un PR añadía `'ashwagandha'` al frontend o renombraba `'vegan_protein' →
  'plant_protein'` SIN actualizar los otros 3 sites, el wizard producía un
  array que el backend rechazaba con 422 silenciosamente al final del flow,
  perdiendo la cuota de generación. Ortogonal al test P0-FORM-6 (campos
  requeridos) y P1-FORM-13 (sentinels).

Cobertura:
  - test_frontend_ssot_matches_backend_api_gate
  - test_frontend_ssot_matches_backend_prompt_dict_keys
  - test_frontend_meta_matches_frontend_ssot
  - test_all_four_sites_have_identical_set
  - sanity tests del parser

NOTA: el parser usa regex sobre los archivos JS. Si el formato cambia (ej.
migración a TypeScript con tipado, cambio a un objeto literal en vez de
array), actualizar los regex.
"""
import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Locator de archivos
# ---------------------------------------------------------------------------
# [P3-3 · 2026-05-10] Tras migración tests root → tests/ (P3-CANDIDATE-B),
# `__file__.parent` es `backend/tests/`. Necesitamos subir DOS niveles para
# llegar al root del monorepo (`MealfitRD.IA/`) donde vive `frontend/` sibling.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
_FORM_VALIDATION_JS = _REPO_ROOT / "frontend" / "src" / "config" / "formValidation.js"
_INTERACTIVE_QUESTIONS_JSX = (
    _REPO_ROOT
    / "frontend"
    / "src"
    / "components"
    / "assessment"
    / "questions"
    / "InteractiveQuestions.jsx"
)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
_SUPPLEMENTS_BLOCK_PATTERN = re.compile(
    r"export\s+const\s+SUPPLEMENTS\s*=\s*Object\.freeze\(\s*\[(?P<body>.*?)\]\s*\)\s*;",
    re.DOTALL,
)
_SUPPLEMENT_META_BLOCK_PATTERN = re.compile(
    r"const\s+SUPPLEMENT_META\s*=\s*\{(?P<body>.*?)\}\s*;",
    re.DOTALL,
)
_QUOTED_STRING = re.compile(r"'([^']+)'")
# Match de claves del mapping `key: { ... }`. La key puede ser bare identifier
# o quoted string. Limitamos a líneas que arrancan con identifier seguido de `:`
# para no capturar claves nested ({ label: '...' } es level 1+).
_META_KEY_PATTERN = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\{",
    re.MULTILINE,
)


def _parse_frontend_supplements(text: str) -> list[str]:
    """Extrae la lista ordenada de strings dentro de `SUPPLEMENTS = Object.freeze([...])`."""
    block = _SUPPLEMENTS_BLOCK_PATTERN.search(text)
    if not block:
        raise AssertionError(
            "No se encontró el bloque `export const SUPPLEMENTS = Object.freeze([...])` "
            "en formValidation.js. Si el formato cambió, actualiza el regex."
        )
    body = block.group("body")
    return _QUOTED_STRING.findall(body)


def _parse_supplement_meta_keys(text: str) -> list[str]:
    """Extrae las claves del mapping `SUPPLEMENT_META = { key: { label, emoji }, ... }`."""
    block = _SUPPLEMENT_META_BLOCK_PATTERN.search(text)
    if not block:
        raise AssertionError(
            "No se encontró el bloque `const SUPPLEMENT_META = { ... };` en "
            "InteractiveQuestions.jsx. Si el formato cambió, actualiza el regex."
        )
    return _META_KEY_PATTERN.findall(block.group("body"))


def _read_file(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"Archivo no encontrado: {path}")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests de drift cross-language
# ---------------------------------------------------------------------------
def test_frontend_ssot_files_exist():
    """Sanity: los dos archivos del frontend están en su ubicación esperada."""
    assert _FORM_VALIDATION_JS.exists(), (
        f"formValidation.js no existe en {_FORM_VALIDATION_JS}. "
        f"Si la estructura del repo cambió, actualiza `_FORM_VALIDATION_JS`."
    )
    assert _INTERACTIVE_QUESTIONS_JSX.exists(), (
        f"InteractiveQuestions.jsx no existe en {_INTERACTIVE_QUESTIONS_JSX}. "
        f"Si la estructura del repo cambió, actualiza `_INTERACTIVE_QUESTIONS_JSX`."
    )


def test_frontend_ssot_matches_backend_api_gate():
    """`SUPPLEMENTS` (frontend) debe ser exactamente igual a `_SUPPLEMENT_ENUM` (backend).

    Si difieren, el wizard puede emitir un array que el validador rechazará
    con 422 al final del flow, perdiendo cuota LLM y degradando UX.
    """
    from routers.plans import _SUPPLEMENT_ENUM

    frontend = set(_parse_frontend_supplements(_read_file(_FORM_VALIDATION_JS)))
    backend = set(_SUPPLEMENT_ENUM)
    missing_in_backend = frontend - backend
    extra_in_backend = backend - frontend
    assert not missing_in_backend and not extra_in_backend, (
        f"Drift entre frontend SUPPLEMENTS y backend _SUPPLEMENT_ENUM:\n"
        f"  Frontend → Backend (sólo en frontend): {sorted(missing_in_backend)}\n"
        f"  Backend → Frontend (sólo en backend):  {sorted(extra_in_backend)}\n"
        f"Si añadiste un suplemento, actualiza:\n"
        f"  1. SUPPLEMENTS en formValidation.js\n"
        f"  2. SUPPLEMENT_META en InteractiveQuestions.jsx\n"
        f"  3. _SUPPLEMENT_ENUM en routers/plans.py\n"
        f"  4. SUPPLEMENT_NAMES en constants.py"
    )


def test_frontend_ssot_matches_backend_prompt_dict_keys():
    """`SUPPLEMENTS` (frontend) debe ser exactamente las keys de `SUPPLEMENT_NAMES` (backend).

    `SUPPLEMENT_NAMES` traduce cada key a un nombre legible que se inyecta al
    prompt LLM. Si el frontend manda un valor sin entry aquí, `build_supplements_context`
    devuelve la key snake_case cruda al LLM ("DEBES incluir: vegan_protein") en
    vez del nombre humano. Ya hay un filtro defensivo (P1-FORM-11) pero el
    invariante correcto es que NUNCA haya divergencia.
    """
    from constants import SUPPLEMENT_NAMES

    frontend = set(_parse_frontend_supplements(_read_file(_FORM_VALIDATION_JS)))
    backend_keys = set(SUPPLEMENT_NAMES.keys())
    missing_translation = frontend - backend_keys
    extra_translation = backend_keys - frontend
    assert not missing_translation and not extra_translation, (
        f"Drift entre frontend SUPPLEMENTS y backend SUPPLEMENT_NAMES.keys():\n"
        f"  Sin traducción humana en backend: {sorted(missing_translation)}\n"
        f"  En backend pero no en frontend:  {sorted(extra_translation)}\n"
        f"Cada key debe tener un nombre legible en SUPPLEMENT_NAMES para que "
        f"el prompt LLM no reciba la key cruda."
    )


def test_frontend_meta_matches_frontend_ssot():
    """`SUPPLEMENT_META` (InteractiveQuestions.jsx) debe cubrir 1:1 a `SUPPLEMENTS`.

    El componente itera `SUPPLEMENTS.map(val => SUPPLEMENT_META[val])`; un val
    sin entry en META renderea null (el `if (!meta) return null` es safety net,
    no UX intencional). Drift visible solo si el usuario entra a QSupplements;
    este test lo cataloga en CI.
    """
    frontend = set(_parse_frontend_supplements(_read_file(_FORM_VALIDATION_JS)))
    meta_keys = set(_parse_supplement_meta_keys(_read_file(_INTERACTIVE_QUESTIONS_JSX)))
    missing_meta = frontend - meta_keys
    extra_meta = meta_keys - frontend
    assert not missing_meta and not extra_meta, (
        f"Drift entre SUPPLEMENTS (formValidation.js) y SUPPLEMENT_META "
        f"(InteractiveQuestions.jsx):\n"
        f"  Sin metadata UI: {sorted(missing_meta)}\n"
        f"  Metadata huérfana: {sorted(extra_meta)}\n"
        f"El invariante runtime en dev-mode también lo avisa, pero este test "
        f"corre en CI (incluyendo prod builds)."
    )


def test_all_four_sites_have_identical_set():
    """Invariante final: los 4 sites tienen exactamente el mismo set.

    Si los 3 tests anteriores pasan, este es trivial — pero lo dejamos como
    aserción explícita del contrato P1-FORM-14 para que sea fácil leer el
    intent en el output de pytest."""
    from routers.plans import _SUPPLEMENT_ENUM
    from constants import SUPPLEMENT_NAMES

    fv_text = _read_file(_FORM_VALIDATION_JS)
    iq_text = _read_file(_INTERACTIVE_QUESTIONS_JSX)

    set1 = set(_parse_frontend_supplements(fv_text))           # SSOT frontend
    set2 = set(_parse_supplement_meta_keys(iq_text))            # metadata UI
    set3 = set(_SUPPLEMENT_ENUM)                                 # API gate backend
    set4 = set(SUPPLEMENT_NAMES.keys())                          # prompt translator backend

    sites = {
        "frontend SUPPLEMENTS (formValidation.js)": set1,
        "frontend SUPPLEMENT_META (InteractiveQuestions.jsx)": set2,
        "backend _SUPPLEMENT_ENUM (routers/plans.py)": set3,
        "backend SUPPLEMENT_NAMES (constants.py)": set4,
    }
    union = set1 | set2 | set3 | set4
    intersection = set1 & set2 & set3 & set4
    if union != intersection:
        diffs = {
            name: sorted(union - members) for name, members in sites.items()
            if members != union
        }
        raise AssertionError(
            f"Drift entre 4 sites del SSOT P1-FORM-14.\n"
            f"Union: {sorted(union)}\n"
            f"Faltantes por site: {diffs}\n"
            f"Si añadiste un suplemento nuevo, asegúrate de actualizarlo en TODOS los sites."
        )


def test_supplement_count_matches_documented_12():
    """ANCHOR: el wizard ofrece exactamente 12 suplementos (per docs).

    Si en el futuro se decide ampliar a 13+, este test debe actualizarse
    junto con la doc. Acta como recordatorio explícito de revisar UX
    (grid layout, spacing) cuando crezca la lista.
    """
    from routers.plans import _SUPPLEMENT_ENUM
    assert len(_SUPPLEMENT_ENUM) == 12, (
        f"Esperaba 12 suplementos en _SUPPLEMENT_ENUM (cantidad documentada); "
        f"encontró {len(_SUPPLEMENT_ENUM)}. Si añadiste/quitaste opciones, "
        f"actualiza este test y revisa el grid CSS de QSupplements."
    )


def test_backend_set_is_immutable_frozenset():
    """`_SUPPLEMENT_ENUM` debe ser frozenset; mutación runtime evadiría el invariante."""
    from routers.plans import _SUPPLEMENT_ENUM
    assert isinstance(_SUPPLEMENT_ENUM, frozenset), (
        f"_SUPPLEMENT_ENUM debe ser frozenset (es {type(_SUPPLEMENT_ENUM).__name__}) "
        f"para que `.add(...)` no compile y el drift no se pueda evadir runtime."
    )


# ---------------------------------------------------------------------------
# Sanity del parser
# ---------------------------------------------------------------------------
def test_parser_extracts_canonical_count():
    """Sanity: el parser de `SUPPLEMENTS` extrae los 12 elementos actuales."""
    items = _parse_frontend_supplements(_read_file(_FORM_VALIDATION_JS))
    assert len(items) == 12, (
        f"Parser sólo extrajo {len(items)} suplementos (esperaba 12). "
        f"Si el formato del archivo cambió, ajustar el regex. Items vistos: {items!r}"
    )


def test_meta_parser_extracts_canonical_count():
    """Sanity: el parser de `SUPPLEMENT_META` extrae las 12 keys actuales."""
    keys = _parse_supplement_meta_keys(_read_file(_INTERACTIVE_QUESTIONS_JSX))
    assert len(keys) == 12, (
        f"Parser de SUPPLEMENT_META sólo extrajo {len(keys)} keys "
        f"(esperaba 12). Keys vistas: {keys!r}"
    )


def test_parser_detects_synthetic_drift():
    """Sanity inverso: el parser detecta entries nuevos en input sintético."""
    fake_js = (
        "export const SUPPLEMENTS = Object.freeze([\n"
        "    'whey_protein',\n"
        "    'ashwagandha',\n"
        "    'creatine',\n"
        "]);\n"
    )
    items = _parse_frontend_supplements(fake_js)
    assert "ashwagandha" in items, (
        f"Parser falló al detectar entry sintética; vio {items!r}"
    )


def test_parser_raises_on_missing_block():
    """Si el archivo perdió el bloque, el parser falla explícito (no devuelve
    set vacío que pasaría el subset trivialmente)."""
    with pytest.raises(AssertionError, match="No se encontró"):
        _parse_frontend_supplements("// archivo sin SUPPLEMENTS\n")


def test_meta_parser_raises_on_missing_block():
    """Idem para SUPPLEMENT_META."""
    with pytest.raises(AssertionError, match="No se encontró"):
        _parse_supplement_meta_keys("// archivo sin SUPPLEMENT_META\n")
