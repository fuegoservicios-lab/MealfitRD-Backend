"""[P3-AGG-CLEAN-LEADING-PUNCT · 2026-05-23] Root-cause cleanup del
bucle infinito de retries del swap-meal verificado en log productivo
2026-05-23 00:33-00:35.

## Caso productivo

El plan_data del user `bf6f1383-...` (plan `884bd00a-...`) tenía un
ingrediente con `/` corrupto:

```json
"ingredients": ["8 lonjas/pedazos de queso", ...]
```

El LLM emitió "lonjas/pedazos" como contracción de "lonjas o pedazos".
``_parse_quantity()`` extrajo el ``name`` del item como
``"/pedazos de queso"`` (preservando el slash). Eso fluyó al aggregator
y se almacenó en ``aggregated_shopping_list`` como
``"½ lb de /pedazos de queso"``.

Modo de fallo recurrente: cuando el user hacía "Cambiar Plato" en
cualquier slot, el LLM veía la pantry list con el item corrupto
``/pedazos de queso`` y trataba de usar ``queso`` en su receta. El
pantry guard hace exact-match contra ``/pedazos de queso`` y NO
matchea ``queso`` → ``unauthorized`` → retry. 3 retries en el mismo
loop → caída al fallback (engañoso pre-P3-SWAP-LLM-RETRIES-422,
422 explícito post-fix).

## Fix

1. **Helper** ``_clean_leading_punct_from_name(name)`` que strip-ea
   ``/``, ``-``, ``*``, ``•``, ``·``, ``▪``, ``▫``, ``◦``, ``‣``, ``⁃``,
   ``▸``, ``◾``, ``◽``, ``■``, ``□`` al inicio del name.
2. **Aplicado en ambos loops** del aggregator (``plan_ingredients`` y
   ``consumed_ingredients``) DESPUÉS de ``_parse_quantity`` y ANTES
   de añadir a ``aggregated[name][unit]``.
3. **Log warning** cuando se aplica el strip — visibilidad operacional
   para detectar upstream bugs sin romper el flujo runtime.
4. **One-shot SQL** (manual via Supabase MCP, no test) limpió el
   ingrediente corrupto en BD para el plan afectado + invalidó las 4
   caches ``aggregated_shopping_list*`` para forzar recalc.

Cross-link con ``test_p2_hist_audit_14_marker_test_link``: slug
``p3_agg_clean_leading_punct`` ↔ filename
``test_p3_agg_clean_leading_punct.py``.
"""
import pathlib

import pytest

BACKEND_ROOT = pathlib.Path(__file__).parent.parent
SHOPPING_PY = (BACKEND_ROOT / "shopping_calculator.py").read_text(encoding="utf-8")
APP_PY = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Section A — Helper unit tests
# ---------------------------------------------------------------------------

def _import_helper():
    pytest.importorskip("langchain_google_genai", reason="shopping_calculator requires langchain")
    from shopping_calculator import _clean_leading_punct_from_name
    return _clean_leading_punct_from_name


@pytest.mark.parametrize("raw,expected", [
    # Caso productivo verificado
    ("/pedazos de queso", "pedazos de queso"),
    # Otros punct/bullets que el LLM podría emitir como list-item markers
    ("-arroz blanco", "arroz blanco"),
    ("* cebolla", "cebolla"),
    ("• tomate", "tomate"),
    ("· papa", "papa"),
    ("▪ brócoli", "brócoli"),
    ("▫ lechuga", "lechuga"),
    # Múltiples chars de punct (defense-in-depth)
    ("//pedazos de queso", "pedazos de queso"),
    ("- *cebolla", "cebolla"),
    # Spaces leading también
    ("  arroz", "arroz"),
    ("\t\tpollo", "pollo"),
    # Inputs limpios — idempotencia
    ("queso", "queso"),
    ("Aceite de oliva", "Aceite de oliva"),
    ("Pollo a la plancha", "Pollo a la plancha"),
    # Edge cases defensivos
    ("", ""),
    ("   ", ""),
])
def test_clean_leading_punct_from_name(raw, expected):
    """[FUNCIONAL] El helper debe stripear leading punct/bullets/spaces
    y ser idempotente para inputs limpios."""
    helper = _import_helper()
    assert helper(raw) == expected, (
        f"Expected {expected!r} from {raw!r}, got {helper(raw)!r}"
    )


def test_clean_leading_punct_non_string_returns_input():
    """[FUNCIONAL] Non-string inputs (None, int, dict) deben retornarse
    as-is (defensivo, no raise)."""
    helper = _import_helper()
    assert helper(None) is None
    assert helper(123) == 123
    assert helper([]) == []


def test_punct_in_middle_of_name_preserved():
    """[FUNCIONAL] Solo el LEADING punct se strip-ea. Punct EN MEDIO del
    name se preserva — algunos ingredient names legítimos contienen
    puntuación (ej. nombres de marcas, hyphenation)."""
    helper = _import_helper()
    # Si el name legitimamente tiene "/" en medio, NO debe stripearse
    assert helper("agua-mineral") == "agua-mineral"
    assert helper("salsa thai/asiática") == "salsa thai/asiática"


# ---------------------------------------------------------------------------
# Section B — Wiring: aplicado en ambos loops del aggregator
# ---------------------------------------------------------------------------

def test_helper_applied_in_plan_loop():
    """El cleanup debe estar invocado en el loop de ``plan_ingredients``
    DESPUÉS de ``_parse_quantity`` y ANTES de ``aggregated[name][unit]``."""
    # Buscar el patrón canónico del plan loop
    assert "_clean_leading_punct_from_name(name)" in SHOPPING_PY, (
        "Falta invocación de `_clean_leading_punct_from_name(name)` en "
        "el aggregator. Sin esto, el helper queda definido pero no se "
        "aplica en runtime."
    )
    # Mínimo 2 invocaciones esperadas (plan loop + consumed loop)
    assert SHOPPING_PY.count("_clean_leading_punct_from_name(name)") >= 2, (
        "Esperaba al menos 2 invocaciones del helper (plan loop + "
        "consumed loop). Si solo está en uno, la asimetría puede causar "
        "que el delta entre plan vs consumed quede inconsistente."
    )


def test_helper_defined_with_warning_log():
    """El helper debe emitir log warning cuando aplica el strip —
    sin esto, los bugs upstream que inyectan punct quedan invisibles
    al operador."""
    # Match sobre la definición del helper
    helper_def_start = SHOPPING_PY.find("def _clean_leading_punct_from_name")
    assert helper_def_start > 0, "Helper no definido"
    # Tomar ~500 chars después de la def para inspeccionar el body
    helper_body = SHOPPING_PY[helper_def_start:helper_def_start + 800]
    assert "logging.warning" in helper_body, (
        "Helper debe llamar logging.warning cuando aplica el strip "
        "para visibilidad operacional (detectar upstream bugs)."
    )
    assert "P3-AGG-CLEAN-LEADING-PUNCT" in helper_body, (
        "El log warning debe incluir el marker P3-AGG-CLEAN-LEADING-PUNCT "
        "para que un grep lo encuentre."
    )


# ---------------------------------------------------------------------------
# Section C — Marker anchor
# ---------------------------------------------------------------------------
# Pin removido siguiendo política establecida — pin-tests se rompen cada
# P-fix siguiente. El contract "marker fresco" lo cubre
# `test_p3_1_last_known_pfix_freshness` (floor check). Las secciones A-B-D-E
# anclan el CONTENIDO del fix (parser-based, no temporales).


# ---------------------------------------------------------------------------
# Section D — Regex pattern matches all known bullet/punct chars
# ---------------------------------------------------------------------------

def test_regex_pattern_covers_known_bullets():
    """El regex ``_LEADING_PUNCT_RE`` debe cubrir al menos los bullets
    Unicode comunes que un LLM podría emitir."""
    pytest.importorskip("langchain_google_genai", reason="shopping_calculator requires langchain")
    from shopping_calculator import _LEADING_PUNCT_RE
    # Sanity: bullets canónicos deben matchear como leading
    for char in ['/', '-', '*', '•', '·', '▪']:
        assert _LEADING_PUNCT_RE.match(f"{char}item"), (
            f"Regex no matchea `{char}` como leading punct."
        )
    # Idempotencia: caracteres normales NO deben matchear como leading punct
    for char in ['a', 'q', '1', '2', '8']:
        assert not _LEADING_PUNCT_RE.match(f"{char}item"), (
            f"Regex match incorrecto sobre `{char}` (no es punct)."
        )


# ---------------------------------------------------------------------------
# Section E — End-to-end con _parse_quantity + cleanup
# ---------------------------------------------------------------------------

def test_aggregator_e2e_strips_slash_from_real_input():
    """[FUNCIONAL] Pasar el input real verificado en log productivo
    (`8 lonjas/pedazos de queso`) al aggregator y verificar que el name
    canonicalizado NO contiene leading slash."""
    pytest.importorskip("langchain_google_genai", reason="shopping_calculator requires langchain")
    from shopping_calculator import aggregate_and_deduct_shopping_list

    # Input minimalista que reproduce el bug del log
    plan_ingredients = ["8 lonjas/pedazos de queso", "30 g de nueces"]
    result = aggregate_and_deduct_shopping_list(plan_ingredients, [], multiplier=1.0)
    # Result es lista de display strings; verificamos que NINGUNO empiece
    # con `/` o contenga `/pedazos` como name canónico
    for item in result:
        assert not item.startswith("/"), (
            f"Output item empieza con `/`: {item!r}. El helper no se "
            f"aplicó en el aggregator."
        )
        assert "/pedazos" not in item.lower(), (
            f"Output item contiene `/pedazos` (root cause original): "
            f"{item!r}"
        )
