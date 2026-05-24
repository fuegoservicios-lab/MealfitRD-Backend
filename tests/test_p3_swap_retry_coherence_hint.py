"""[P3-SWAP-RETRY-COHERENCE-HINT · 2026-05-22] Cuando el coherence
validator rechaza un attempt del LLM, el retry prompt original (P1-SWAP-
RECIPE-COHERENCE) inyectaba un summary genérico que solo mencionaba el
nombre CANÓNICO de la proteína. El LLM no encontraba ese canónico en su
propia receta (porque había escrito un alias) y reintentaba con el mismo
alias — verificado en log productivo 2026-05-22 23:04-23:05:

  attempt 1: 42g Queso unauthorized → retry
  attempt 2: "dorado" mentioned, "pescado" not listed → retry
  attempt 3: SAME "dorado" → fallback (Plato Fallback feo, P3-SWAP-FALLBACK-TITLE-COPY)

Dos fixes coordinados para subir señal al LLM:

  1. ``validate_meal_recipe_ingredients_coherence`` (nutrition_calculator.py)
     ahora cita el ALIAS verbatim en el summary (no solo el canónico) +
     estructura "elige UNA opción" con rutas (a)/(b) explícitas.
  2. ``agent.py::swap_meal`` y ``tools.py::execute_modify_single_meal``
     appendean una "REGLA INVARIANTE" self-check al retry prompt
     instruyendo al LLM verificar la consistencia recipe↔ingredients
     ANTES de devolver respuesta.

Cross-link con ``test_p2_hist_audit_14_marker_test_link``: slug
``p3_swap_retry_coherence_hint`` ↔ filename
``test_p3_swap_retry_coherence_hint.py``.
"""
import pathlib
import re

import pytest

BACKEND_ROOT = pathlib.Path(__file__).parent.parent
AGENT_PY = (BACKEND_ROOT / "agent.py").read_text(encoding="utf-8")
TOOLS_PY = (BACKEND_ROOT / "tools.py").read_text(encoding="utf-8")
NUTR_PY = (BACKEND_ROOT / "nutrition_calculator.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Section A — Summary del validator cita el alias verbatim
# ---------------------------------------------------------------------------

def test_validator_summary_includes_mentioned_alias_not_just_canonical():
    """[FUNCIONAL] Cuando ``divergences = {'pescado': {'mentioned_alias':
    'dorado', 'listed': False}}``, el summary debe contener ``"dorado"``
    (el alias que el LLM escribió) en addition al canónico ``"pescado"``."""
    pytest.importorskip("langchain_google_genai", reason="nutrition_calculator requiere langchain")
    from nutrition_calculator import validate_meal_recipe_ingredients_coherence

    # Mock meal con recipe que menciona "dorado" pero ingredients listan otra cosa
    meal = {
        "name": "Plato con dorado",
        "ingredients": ["100g arroz", "50g lechuga"],
        "recipe": [
            "Marina el dorado en limón.",
            "Cocina el dorado a la plancha 5 min por lado.",
            "Sirve con arroz y lechuga.",
        ],
    }
    passed, divs, summary = validate_meal_recipe_ingredients_coherence(meal)
    assert passed is False, f"Validator debió detectar divergence pero passed={passed}"
    assert "pescado" in divs, f"Esperaba canonical 'pescado' en divs, got {list(divs.keys())}"
    assert divs["pescado"].get("mentioned_alias") == "dorado", (
        f"Esperaba mentioned_alias='dorado', got {divs['pescado'].get('mentioned_alias')!r}"
    )

    # El summary nuevo DEBE citar el alias 'dorado'
    assert "dorado" in summary, (
        f"Summary debe mencionar el alias 'dorado' que el LLM escribió. "
        f"Si solo dice 'pescado' (canónico), el LLM no encuentra qué corregir "
        f"y reintenta con el mismo alias. Summary: {summary!r}"
    )
    # También debe seguir mencionando el canónico (para contexto)
    assert "pescado" in summary, (
        f"Summary debe mencionar canónico 'pescado' como contexto. Summary: {summary!r}"
    )


def test_validator_summary_offers_two_explicit_correction_paths():
    """[FUNCIONAL] El summary debe ofrecer 2 rutas explícitas (a)/(b)
    para corregir — pre-fix era ambiguo ('Añade el ingrediente O reescribe
    la receta') y el LLM no elegía ninguno consistentemente."""
    pytest.importorskip("langchain_google_genai", reason="nutrition_calculator requiere langchain")
    from nutrition_calculator import validate_meal_recipe_ingredients_coherence

    meal = {
        "name": "Plato con pollo",
        "ingredients": ["100g arroz"],
        "recipe": ["Cocina el pollo a la plancha."],
    }
    passed, divs, summary = validate_meal_recipe_ingredients_coherence(meal)
    assert passed is False, "Debió detectar divergence con 'pollo' no listado"

    # Las dos rutas (a) y (b)
    assert "(a)" in summary, (
        f"Falta ruta (a) explícita en summary. Summary: {summary!r}"
    )
    assert "(b)" in summary, (
        f"Falta ruta (b) explícita en summary. Summary: {summary!r}"
    )
    assert "elige UNA" in summary or "CORRECCIÓN OBLIGATORIA" in summary, (
        f"Summary debe ser directivo (no sugerencia tibia). Summary: {summary!r}"
    )


def test_validator_summary_handles_multiple_divergences():
    """[FUNCIONAL] Cuando hay múltiples canonicals divergentes, el summary
    debe listar los aliases de TODOS, no solo el primero."""
    pytest.importorskip("langchain_google_genai", reason="nutrition_calculator requiere langchain")
    from nutrition_calculator import validate_meal_recipe_ingredients_coherence

    meal = {
        "name": "Plato mixto",
        "ingredients": ["100g arroz"],
        "recipe": [
            "Marina el dorado.",
            "Saltea el cerdo con cebolla.",
            "Sirve con arroz.",
        ],
    }
    passed, divs, summary = validate_meal_recipe_ingredients_coherence(meal)
    assert passed is False
    # Al menos dorado o cerdo debe estar mencionado
    aliases_in_summary = sum(1 for alias in ["dorado", "cerdo"] if alias in summary)
    assert aliases_in_summary >= 1, (
        f"Summary debe citar al menos uno de los aliases ('dorado', 'cerdo'). "
        f"Summary: {summary!r}"
    )


# ---------------------------------------------------------------------------
# Section B — agent.py::swap_meal inyecta self-check directive
# ---------------------------------------------------------------------------

def test_agent_py_appends_invariant_self_check_to_retry_prompt():
    """``agent.py::swap_meal`` debe appendear una "REGLA INVARIANTE" al
    retry prompt en el branch del coherence validator. Pre-fix solo
    inyectaba el summary; el self-check directive sube señal."""
    # Buscar el bloque dentro del branch de coherence (post ⚠️ divergence detected)
    coherence_branch_match = re.search(
        r"P1-SWAP-RECIPE-COHERENCE\] divergence detected.*?raise ValueError\(coh_summary\)",
        AGENT_PY,
        re.DOTALL,
    )
    assert coherence_branch_match, (
        "No se encontró el branch del coherence validator en agent.py — "
        "puede que el marker P1-SWAP-RECIPE-COHERENCE haya sido renombrado."
    )
    branch_body = coherence_branch_match.group(0)
    assert "REGLA INVARIANTE" in branch_body, (
        "Falta `REGLA INVARIANTE` self-check en el retry prompt del "
        "coherence branch de agent.py."
    )
    assert "ANTES de devolver" in branch_body, (
        "Self-check debe instruir al LLM verificar ANTES de devolver respuesta."
    )


def test_tools_py_mirrors_self_check_in_execute_modify_single_meal():
    """``tools.py::execute_modify_single_meal`` (chat-agent) debe espejar
    el mismo self-check directive (paridad con swap_meal)."""
    coherence_branch_match = re.search(
        r"P1-SWAP-RECIPE-COHERENCE\] divergence in modify_meal.*?raise ValueError\(coh_summary\)",
        TOOLS_PY,
        re.DOTALL,
    )
    assert coherence_branch_match, (
        "No se encontró el branch del coherence validator en tools.py "
        "(execute_modify_single_meal)."
    )
    branch_body = coherence_branch_match.group(0)
    assert "REGLA INVARIANTE" in branch_body, (
        "Falta `REGLA INVARIANTE` self-check en tools.py — paridad rota "
        "con swap_meal."
    )


# ---------------------------------------------------------------------------
# Section C — Marker anchor
# ---------------------------------------------------------------------------
# Pin removido siguiendo política establecida (test_p2_swap_422_ux_copy,
# test_p3_swap_pantry_default, test_p3_swap_fallback_title_copy): pin-tests
# se rompen cada P-fix siguiente cuando el marker avanza. El contract
# "marker fresco a nivel codebase" lo cubre `test_p3_1_last_known_pfix_freshness`
# (floor check). Las secciones A-B-D anclan el CONTENIDO del fix.


# ---------------------------------------------------------------------------
# Section D — Back-compat: shape de divergences sin cambios
# ---------------------------------------------------------------------------

def test_divergences_dict_shape_unchanged_for_back_compat():
    """[FUNCIONAL] El shape de ``divergences`` (segundo elemento del tuple
    retornado) debe seguir siendo ``dict[canonical, {mentioned_alias, listed}]``
    sin cambios. Solo el ``summary`` (tercer elemento) cambió. Esto preserva
    cualquier caller que inspeccione el dict programáticamente (logs,
    métricas, etc.)."""
    pytest.importorskip("langchain_google_genai", reason="nutrition_calculator requiere langchain")
    from nutrition_calculator import validate_meal_recipe_ingredients_coherence

    meal = {
        "name": "Plato con dorado",
        "ingredients": ["100g arroz"],
        "recipe": ["Cocina el dorado a la plancha."],
    }
    passed, divs, summary = validate_meal_recipe_ingredients_coherence(meal)
    assert isinstance(divs, dict)
    for canonical, info in divs.items():
        assert isinstance(info, dict), f"Divergence info debe ser dict, got {type(info)}"
        assert "mentioned_alias" in info, (
            f"Shape de divergences[{canonical!r}] perdió 'mentioned_alias'. "
            f"Callers (logs, métricas) podrían romperse."
        )
        assert "listed" in info, (
            f"Shape de divergences[{canonical!r}] perdió 'listed'."
        )
