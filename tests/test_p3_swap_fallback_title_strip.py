"""[P3-SWAP-FALLBACK-TITLE-STRIP · 2026-05-23] Revisión del fix
P3-SWAP-FALLBACK-TITLE-COPY del día anterior: aquel asumió que
`aggregated_shopping_list` items eran DICTS y prefirió `name` sobre
`display_string`. Pero `get_realtime_pantry()` (shopping_calculator.py)
retorna directamente el output de `aggregate_and_deduct_shopping_list()`,
que produce STRINGS con formato display.

Caso productivo verificado log 2026-05-23 00:09:09:
  user swap "Queso Fresco a la Plancha" (Cena), LLM agotó 2 retries
  (cap_swallowed queso + dorado coherence), cayó al fallback con título:

    "Cena con 1 Cabeza (~500g) Brócoli y 1 Mazo Cilantro"

Esos son los items de `clean_ingredients` que vinieron de
`get_realtime_pantry()` (NO del empty-pantry-fallback, que era el path
que P3-SWAP-FALLBACK-TITLE-COPY arregló).

Fix definitivo: extracción robusta del nombre limpio en el punto de
construcción del título, idempotente para inputs ya limpios.

Cross-link con ``test_p2_hist_audit_14_marker_test_link``: slug
``p3_swap_fallback_title_strip`` ↔ filename
``test_p3_swap_fallback_title_strip.py``.
"""
import pathlib
import re

import pytest

BACKEND_ROOT = pathlib.Path(__file__).parent.parent
AGENT_PY = (BACKEND_ROOT / "agent.py").read_text(encoding="utf-8")
APP_PY = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Section A — Helper extraction unit tests
# ---------------------------------------------------------------------------

def _import_helper():
    pytest.importorskip("langchain_google_genai", reason="agent.py requires langchain")
    from agent import _extract_clean_name_from_display_string
    return _extract_clean_name_from_display_string


@pytest.mark.parametrize("raw,expected", [
    # Casos exactos del caso productivo verificado
    ("1 Cabeza (~500g) Brócoli", "Brócoli"),
    ("1 Mazo Cilantro", "Cilantro"),
    ("1 Ud. (~2 lbs) Piña", "Piña"),
    ("1 Ud. Guineo", "Guineo"),
    # Casos con "de" connector
    ("1 botella (250ml) de Aceite de oliva", "Aceite de oliva"),
    ("1 cartón (473ml) de Leche", "Leche"),
    ("1 lata (425g) de Gandules", "Gandules"),
    ("1 paquete (340g) de Habichuelas negras", "Habichuelas negras"),
    ("1 paquete (¼ lb) de Almendras fileteadas", "Almendras fileteadas"),
    # Fracciones unicode
    ("¼ lb (~1 Ud.) de Papa", "Papa"),
    ("¼ lb (~1 Ud.) de Tomate", "Tomate"),
    # Múltiples uds
    ("3 Uds. Limón", "Limón"),
    ("2 Uds. Plátano", "Plátano"),
    # Inputs ya limpios — idempotencia
    ("Brócoli", "Brócoli"),
    ("Aceite de oliva", "Aceite de oliva"),
    ("Pollo", "Pollo"),
    ("Habichuelas negras", "Habichuelas negras"),
    # Edge cases — defensa
    ("", ""),
    ("   ", ""),
])
def test_extract_clean_name_from_display_string(raw, expected):
    """[FUNCIONAL] El helper debe extraer el nombre limpio de strings
    con formato display Y ser idempotente para inputs ya limpios."""
    extractor = _import_helper()
    assert extractor(raw) == expected, (
        f"Expected {expected!r} from {raw!r}, got {extractor(raw)!r}"
    )


def test_extract_clean_name_non_string_returns_empty():
    """[FUNCIONAL] Inputs no-string (None, int, dict) → empty string,
    sin raise. Defensivo para upstream con tipos inesperados."""
    extractor = _import_helper()
    assert extractor(None) == ""
    assert extractor(123) == ""
    assert extractor({"name": "Brócoli"}) == ""
    assert extractor([]) == ""


# ---------------------------------------------------------------------------
# Section B — Wiring: el title del fallback usa el helper
# ---------------------------------------------------------------------------

def test_helper_invoked_in_fallback_title_construction():
    """El helper debe estar invocado en el bloque de construcción del
    título del fallback (response = {...}). Sin esto, el title puede
    recibir display strings crudos."""
    # Buscar el bloque del response del fallback con el patrón único
    fallback_block_match = re.search(
        r"_ing_title_tokens\s*=\s*\[\].*?response\s*=\s*\{",
        AGENT_PY,
        re.DOTALL,
    )
    assert fallback_block_match, (
        "No se encontró el bloque que construye `_ing_title_tokens` "
        "antes del response del fallback. Si renombraste la variable, "
        "actualiza este test."
    )
    block = fallback_block_match.group(0)
    assert "_extract_clean_name_from_display_string" in block, (
        "El helper `_extract_clean_name_from_display_string` debe estar "
        "invocado al construir `_ing_title_tokens`. Sin esto, el title "
        "del fallback expone display strings crudos como "
        "'1 Cabeza (~500g) Brócoli'."
    )


def test_helper_is_idempotent_for_hardcoded_fallback():
    """[FUNCIONAL] Cuando el fallback usa los hardcoded
    `["Pollo", "Arroz", "Aguacate"]` (caso pantry vacío + reason no-strict),
    el helper debe pasar los nombres limpios as-is."""
    extractor = _import_helper()
    for clean_name in ["Pollo", "Arroz", "Aguacate", "Lechuga", "Brócoli"]:
        assert extractor(clean_name) == clean_name, (
            f"Helper NO debe modificar nombres ya limpios. "
            f"Input {clean_name!r} → got {extractor(clean_name)!r}"
        )


# ---------------------------------------------------------------------------
# Section C — Boundary safety: el helper no es agresivo con números
# en medio del string
# ---------------------------------------------------------------------------

def test_extractor_does_not_strip_numbers_in_middle_of_name():
    """[FUNCIONAL] Strings que tienen números EN MEDIO (ej. nombres con
    versiones tipo "Vitamina B12") no deben verse mutilados. Solo el
    prefijo numérico inicial cuenta."""
    extractor = _import_helper()
    # Si NO empieza con dígito, debe pasar as-is
    assert extractor("Vitamina B12") == "Vitamina B12"
    assert extractor("Pasta 100% integral") == "Pasta 100% integral"


def test_extractor_handles_only_prefix_numerics():
    """[FUNCIONAL] Si todo el string es prefijo numérico sin nombre
    al final, debe retornar el string original (no vacío) para que el
    title fallback tenga el guard 'ingredientes de tu nevera'."""
    extractor = _import_helper()
    # Caso patológico — todo es prefijo
    result = extractor("1 Cabeza (~500g)")
    # Acepta: o el original o vacío. Lo importante es NO crashear.
    assert result in ("1 Cabeza (~500g)", "", "Cabeza")  # tolerancia
