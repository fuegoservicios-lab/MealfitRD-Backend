"""[P3-NEW-6 · 2026-05-11] Anchor surgical de la decisión P3-NEW-1:
detecta ASIGNACIONES explícitas a `_plan_modified_at` en
`api_expand_recipe`, complementario al test broad de
`test_p3_new_1_recipe_expand_no_modified_at_bump.py`.

Bug detection coverage:
    `test_p3_new_1_no_plan_modified_at_string_in_handler` (broad):
        Falla si CUALQUIER mención del string `_plan_modified_at`
        aparece fuera del bloque comment P3-NEW-1 scrubeado. Brittle:
        cualquier explicación legítima que mencione el string en
        commentary falla aunque el código sea correcto. P1-NEW-7
        demostró esto — el comentario nuevo del fix P0 tuvo que
        rephrase para evitar el literal en su explicación del bug
        histórico.

    P3-NEW-6 (este test, surgical):
        Falla SOLO si aparece un patrón de ASIGNACIÓN a
        `_plan_modified_at` (mutación real del dict, no mención
        explicativa). Más robusto a documentación legítima.

Coverage complementaria: ambos tests + buen anchor de la decisión.
El broad puede aceptar más rephrases en commentary (vía scrubbing);
el surgical es contundente contra mutaciones reales.

Patrones cubiertos:
    1. `target_plan_data["_plan_modified_at"] = ...`   (subscript)
    2. `target_plan_data._plan_modified_at = ...`      (attr)
    3. `jsonb_set(..., '_plan_modified_at', ...)`       (raw SQL)
    4. `{"_plan_modified_at": ...}`                     (dict literal)
    5. `'_plan_modified_at': ...`                       (dict literal alt)

Tooltip-anchor: P3-NEW-6-NO-MODIFIED-AT-ASSIGN | surgical P3-NEW-1 anchor
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"


def _extract_function_body(src: str, fn_name: str) -> str:
    """Mismo helper que test_p1_new_7_*: extrae body desde `def fn_name(`
    hasta el siguiente top-level def/router decorator."""
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    if not m:
        raise AssertionError(f"No se encontró def {fn_name}.")
    start = m.start()
    next_def = re.search(r"\n(?:@router\.|@app\.|def\s)", src[start + 1:])
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def expand_body() -> str:
    src = _PLANS_PY.read_text(encoding="utf-8")
    return _extract_function_body(src, "api_expand_recipe")


# Patrones de ASIGNACIÓN específicos. Diseñados para NO matchear
# menciones explicativas en comentarios (que el test broad de P3-NEW-1
# puede aceptar via scrubbing).
_ASSIGNMENT_PATTERNS = {
    "subscript_assign": r"\w+\s*\[\s*[\"']_plan_modified_at[\"']\s*\]\s*=(?!=)",
    "attr_assign": r"\.\s*_plan_modified_at\s*=(?!=)",
    "jsonb_set_raw_sql": r"jsonb_set\([^)]*_plan_modified_at",
    "dict_literal_double_quote": r"[\"']_plan_modified_at[\"']\s*:",
}


@pytest.mark.parametrize("pattern_name,pattern", list(_ASSIGNMENT_PATTERNS.items()))
def test_no_modified_at_assignment_in_handler(
    expand_body: str, pattern_name: str, pattern: str,
):
    """Cada patrón de ASIGNACIÓN a `_plan_modified_at` debe estar
    AUSENTE del cuerpo de `api_expand_recipe`. Permite menciones en
    commentary (e.g., el bloque P3-NEW-1 que documenta la decisión, o
    el bloque P1-NEW-7 que documenta el bug histórico) pero rechaza
    cualquier mutación real del dict.
    """
    match = re.search(pattern, expand_body)
    if match:
        # Contexto alrededor del match para debug.
        ctx_start = max(0, match.start() - 40)
        ctx_end = min(len(expand_body), match.end() + 40)
        context = expand_body[ctx_start:ctx_end].replace("\n", "\\n")
        pytest.fail(
            f"P3-NEW-6 regresión: el handler `api_expand_recipe` contiene "
            f"un patrón de asignación a `_plan_modified_at` "
            f"(`{pattern_name}`). Contexto: `...{context}...`.\n\n"
            f"La decisión P3-NEW-1 prohíbe bumpear `_plan_modified_at` "
            f"desde este handler — cook-clicks NO deben reordenar el "
            f"Historial (el sort semántico se reserva para mutaciones "
            f"reales del plan, no para interacciones de visualización).\n\n"
            f"Si la decisión cambió:\n"
            f"  1. Actualizar el bloque comment P3-NEW-1 en plans.py "
            f"     explicando el nuevo razonamiento.\n"
            f"  2. Retirar este test (junto con "
            f"     `test_p3_new_1_no_plan_modified_at_string_in_handler`).\n"
            f"  3. Actualizar `frontend` si el sort del Historial depende "
            f"     de este path."
        )


def test_patterns_dictionary_is_complete():
    """Sanity: el dict `_ASSIGNMENT_PATTERNS` tiene los 4 patterns
    enumerados en el docstring. Si alguien borra uno, este test alerta
    independientemente de los parametrized tests."""
    expected_patterns = {
        "subscript_assign",
        "attr_assign",
        "jsonb_set_raw_sql",
        "dict_literal_double_quote",
    }
    assert set(_ASSIGNMENT_PATTERNS.keys()) == expected_patterns, (
        f"P3-NEW-6: dict `_ASSIGNMENT_PATTERNS` cambió. Esperado "
        f"{expected_patterns}, encontrado {set(_ASSIGNMENT_PATTERNS.keys())}. "
        f"Si añadiste/borraste un pattern intencionalmente, actualizar "
        f"este test + el docstring del archivo para mantener el "
        f"contrato visible."
    )


def test_marker_anchor_present():
    """Slug del filename matchea el marker `P3-NEW-6`."""
    expected_slug = "p3_new_6"
    assert expected_slug in __file__.replace("\\", "/").lower()
