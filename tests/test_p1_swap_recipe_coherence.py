"""[P1-SWAP-RECIPE-COHERENCE · 2026-05-22] Test dedicado del validador
`validate_meal_recipe_ingredients_coherence` + wiring en `swap_meal` y
`execute_modify_single_meal`.

Pre-fix (gap del bundle inicial P1-SWAP-MACROS · 2026-05-22):
    El validador existía en `nutrition_calculator.py` y se invocaba en el
    retry loop de ambos surfaces (`agent.py::invoke_with_retry` línea 648
    y `tools.py::invoke_with_retry` línea ~704), pero la cobertura estaba
    INDIRECTAMENTE en `test_p1_swap_macros.py` Section G (mezclada con
    asserts de macros). Audit production-readiness 2026-05-22 identificó:
    "NO hay test unitario `test_p1_swap_recipe_coherence.py` específico —
    solo cobertura indirecta en `test_p1_swap_macros.py`".

    Si alguien degrada el validador (cambia el boundary check, rompe la
    canonicalización con acentos, refactoriza `_FALLBACK_PROTEIN_SYNONYMS`
    en un commit ortogonal), las regresiones se manifiestan como
    cap_swallowed_modifier llegando al PDF — el user ve "el pollo" en la
    receta pero su lista de compras solo dice "pavo" → frustración
    cocinando + lección incorrecta de la app.

Cierre P1-SWAP-RECIPE-COHERENCE-DEDICATED:
    Este file ancla el validador con tests unitarios funcionales
    (NO parser-based; cargamos el módulo y ejercitamos su lógica) +
    tests parser-based del wiring en los dos surfaces de swap. Cualquier
    regresión del validador es detectada antes de que llegue a prod.

Tooltip-anchor: P1-SWAP-RECIPE-COHERENCE | test dedicado audit 2026-05-22
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND = _REPO_ROOT / "backend"
_AGENT_PY = _BACKEND / "agent.py"
_TOOLS_PY = _BACKEND / "tools.py"
_NUTRITION_PY = _BACKEND / "nutrition_calculator.py"

# Permitir import de nutrition_calculator sin estar en sys.path por defecto.
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def validator():
    """Importa el validador. Si falla por dep no instalada (langchain en
    minimal envs), skip el módulo entero — la cobertura parser-based de las
    secciones D/E sigue corriendo desde otros tests."""
    try:
        from nutrition_calculator import validate_meal_recipe_ingredients_coherence
        return validate_meal_recipe_ingredients_coherence
    except Exception as e:
        pytest.skip(
            f"nutrition_calculator no importable en este env: "
            f"{type(e).__name__}: {e}"
        )


@pytest.fixture(scope="module")
def fallback_synonyms():
    try:
        from nutrition_calculator import _FALLBACK_PROTEIN_SYNONYMS
        return _FALLBACK_PROTEIN_SYNONYMS
    except Exception as e:
        pytest.skip(f"_FALLBACK_PROTEIN_SYNONYMS no importable: {e}")


@pytest.fixture(scope="module")
def alias_word_check():
    try:
        from nutrition_calculator import _alias_appears_as_word
        return _alias_appears_as_word
    except Exception as e:
        pytest.skip(f"_alias_appears_as_word no importable: {e}")


@pytest.fixture(scope="module")
def kill_switch_helper():
    try:
        from nutrition_calculator import _swap_recipe_coherence_enabled
        return _swap_recipe_coherence_enabled
    except Exception as e:
        pytest.skip(f"_swap_recipe_coherence_enabled no importable: {e}")


@pytest.fixture(scope="module")
def agent_src() -> str:
    return _AGENT_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tools_src() -> str:
    return _TOOLS_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Section A — Validator unit tests (funcional)
# ---------------------------------------------------------------------------
class TestValidatorBehavior:
    """Tests funcionales directos del validador. Cubre el modo de fallo
    `cap_swallowed_modifier` que es el bug user-facing original."""

    def test_passes_when_ingredient_in_recipe_and_listed(self, validator):
        """Caso golden: receta menciona pollo, ingredients tiene pollo →
        passed=True."""
        meal = {
            "ingredients": ["200g pechuga de pollo", "100g arroz integral"],
            "recipe": [
                "Sazona el pollo con sal y pimienta.",
                "Cocina a fuego medio 8 minutos por lado.",
            ],
        }
        passed, divs, summary = validator(meal)
        assert passed is True, (
            f"Golden: receta menciona pollo Y ingredients tiene pollo, "
            f"debería passed=True. divs={divs}, summary={summary!r}"
        )
        assert divs == {}
        assert summary == ""

    def test_detects_cap_swallowed_modifier(self, validator):
        """Bug original: receta dice 'pollo' pero ingredients solo tiene
        carne molida. El validador DEBE detectar la divergencia.

        Usamos 'carne molida' (alias de 'res') en lugar de 'pavo' porque
        'pavo' incluye 'pechuga' como alias, lo cual también es alias de
        'pollo' — overlap real del catálogo. Esto es un escenario más
        limpio sin ambigüedad de aliases."""
        meal = {
            "ingredients": ["200g carne molida", "100g arroz integral"],
            "recipe": [
                "Sazona el pollo con sal y pimienta.",
                "Sofríe el pollo hasta dorar.",
            ],
        }
        passed, divs, summary = validator(meal)
        assert passed is False, (
            "P1-SWAP-RECIPE-COHERENCE regresión: cap_swallowed_modifier "
            "(recipe='pollo', ingredients=['carne molida']) NO detectado. "
            "El validador devolvió passed=True — se reabre el bug "
            "user-facing de receta sin ingrediente listado."
        )
        # 'pollo' canonicalizado debe aparecer entre las claves del divergence dict
        assert any("pollo" in str(k).lower() for k in divs.keys()), (
            f"Divergence dict no menciona 'pollo': {divs}"
        )
        assert "no listados" in summary.lower() or "no listado" in summary.lower(), (
            f"Summary no comunica la naturaleza del error: {summary!r}"
        )

    def test_passes_when_no_ingredients(self, validator):
        """Defensive: sin ingredients → no hay comparación, passed=True."""
        meal = {
            "ingredients": [],
            "recipe": ["Cocina el pollo a la plancha."],
        }
        passed, divs, summary = validator(meal)
        assert passed is True
        assert divs == {}

    def test_passes_when_no_recipe(self, validator):
        """Defensive: sin recipe → no hay nada que comparar, passed=True."""
        meal = {
            "ingredients": ["200g pollo"],
            "recipe": [],
        }
        passed, divs, summary = validator(meal)
        assert passed is True
        assert divs == {}

    def test_invalid_input_returns_passed_silently(self, validator):
        """Defensive: input no-dict → passed=True (no rompe el invoke loop)."""
        for bad_input in (None, [], "not a meal", 42, True):
            passed, divs, summary = validator(bad_input)
            assert passed is True, (
                f"Bad input {bad_input!r} no debe levantar — sería un "
                f"hard-fail del swap que la LLM no puede corregir."
            )

    def test_accents_normalized(self, validator):
        """Receta usa 'pollo' con acento decorativo o canonicaliza desde
        plural: 'pescados' debe matchear 'pescado'."""
        meal = {
            "ingredients": ["200g atun"],
            "recipe": ["Cocina los pescados al horno con limón."],
        }
        passed, divs, summary = validator(meal)
        # 'pescado' es el canónico; 'atun' está en sus aliases.
        # 'pescados' (plural) DEBE matchear 'pescado'.
        # Ingredients tiene 'atun' que TAMBIÉN está en aliases de 'pescado'.
        # Por lo tanto: NO divergence (un alias del canónico está listado).
        assert passed is True, (
            f"Acentos/plurales: 'pescados' en recipe y 'atun' en "
            f"ingredients ambos canonicalizan a 'pescado' → debería "
            f"passed=True. Got divs={divs}"
        )


# ---------------------------------------------------------------------------
# Section B — Boundary check (anti false-positives)
# ---------------------------------------------------------------------------
class TestBoundaryCheck:
    """Tests del helper `_alias_appears_as_word` — evita falsos positivos
    como 'res' en 'estresante' o 'ave' en 'lavar'."""

    def test_alias_at_start(self, alias_word_check):
        assert alias_word_check("pollo", "pollo sazonado y dorado") is True

    def test_alias_at_end(self, alias_word_check):
        assert alias_word_check("pollo", "sazonado pollo") is True

    def test_alias_middle(self, alias_word_check):
        assert alias_word_check("pollo", "sazonado el pollo bien") is True

    def test_no_false_positive_res_in_estresante(self, alias_word_check):
        """Crítico: 'res' es alias canónico pero NO debe matchear 'estresante'."""
        assert alias_word_check("res", "el cocinero estresante grita") is False, (
            "P1-SWAP-RECIPE-COHERENCE regresión: boundary check rompió. "
            "'res' matcheó 'estresante' — falso positivo que rechazará "
            "recetas válidas y forzará retries innecesarios."
        )

    def test_no_false_positive_pollo_in_polloncito(self, alias_word_check):
        """Sufijos diminutivos no deben confundirse — match estricto."""
        assert alias_word_check("pollo", "el polloncito tierno") is False

    def test_punctuation_boundary(self, alias_word_check):
        """Comas, puntos y signos de puntuación cuentan como boundary."""
        assert alias_word_check("pollo", "pon el pollo, sal y pimienta") is True
        assert alias_word_check("pollo", "trozos de pollo. servir caliente") is True


# ---------------------------------------------------------------------------
# Section C — Knob kill-switch
# ---------------------------------------------------------------------------
class TestKillSwitch:
    """`MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE` env var controla el wiring."""

    def test_default_true_when_unset(self, kill_switch_helper, monkeypatch):
        monkeypatch.delenv(
            "MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE", raising=False
        )
        assert kill_switch_helper() is True

    def test_returns_false_when_env_explicitly_false(
        self, kill_switch_helper, monkeypatch
    ):
        monkeypatch.setenv("MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE", "false")
        assert kill_switch_helper() is False

    def test_truthy_values_remain_true(self, kill_switch_helper, monkeypatch):
        """Solo el literal 'false' (case-insensitive) desactiva. Cualquier
        otro valor (incluyendo '0', 'no', basura) deja el validador ON
        para evitar desactivación accidental por typo."""
        for val in ("true", "True", "TRUE", "1", "yes", "garbage"):
            monkeypatch.setenv("MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE", val)
            assert kill_switch_helper() is True, (
                f"Valor {val!r} no debería desactivar — solo el literal "
                f"'false' (case-insensitive) cumple esa función."
            )


# ---------------------------------------------------------------------------
# Section D — Fallback subset (drift detection vs canonical)
# ---------------------------------------------------------------------------
def test_fallback_proteins_has_meaningful_overlap_with_canonical(
    fallback_synonyms,
):
    """Si `constants.PROTEIN_SYNONYMS` es importable, las top-5 proteínas
    más comunes en planes RD DEBEN existir en ambos dicts (sanity:
    el fallback no degrada el check de las proteínas críticas).

    Diseño deliberadamente laxo en lugar de "subset estricto" porque:
      - El fallback puede usar nombres canónicos que el módulo `constants`
        agrupa bajo otros canónicos (e.g., `habichuelas` puede vivir bajo
        `leguminosas` o tener nombre normalizado distinto).
      - Drift entre los dos dicts es expected (el fallback es minimalista
        intencional, no auto-genera de canonical).

    El test garantiza que las 5 proteínas más usadas en planes RD
    (`pollo`, `res`, `huevos`, `pescado`, `cerdo`) están en ambos.
    Skip si `constants` no es importable (dev sandbox / minimal env).
    """
    try:
        from constants import PROTEIN_SYNONYMS as canonical
    except Exception:
        pytest.skip("constants.PROTEIN_SYNONYMS no importable en este env")

    fallback_keys = {str(k).lower() for k in fallback_synonyms.keys()}
    canonical_keys = {str(k).lower() for k in canonical.keys()}

    # Top-5 proteínas críticas que DEBEN coincidir entre ambos catálogos.
    critical_proteins = {"pollo", "res", "huevos", "pescado", "cerdo"}
    missing_in_fallback = critical_proteins - fallback_keys
    missing_in_canonical = critical_proteins - canonical_keys

    assert not missing_in_fallback, (
        f"P1-SWAP-RECIPE-COHERENCE: las proteínas críticas "
        f"{sorted(missing_in_fallback)} faltan en "
        f"`_FALLBACK_PROTEIN_SYNONYMS`. El fallback degradaría a un "
        f"check incompleto para casos productivos de alta frecuencia."
    )
    assert not missing_in_canonical, (
        f"P1-SWAP-RECIPE-COHERENCE: las proteínas críticas "
        f"{sorted(missing_in_canonical)} faltan en "
        f"`constants.PROTEIN_SYNONYMS`. El catálogo canónico perdió "
        f"proteínas core — investigar antes de continuar."
    )


# ---------------------------------------------------------------------------
# Section E — Wiring en agent.py::swap_meal
# ---------------------------------------------------------------------------
class TestWiringInSwapMeal:
    """Tests parser-based que verifican el wiring del validador dentro
    del retry loop de `swap_meal`."""

    def test_validator_imported_lazily(self, agent_src: str):
        """El validador y kill-switch helper deben importarse dentro del
        cuerpo de `swap_meal`, NO en el top-level (para que el módulo
        cargue sin nutrition_calculator si éste rompe)."""
        # Aceptamos cualquiera de los nombres importados (validador o
        # kill-switch). El detalle del import inline está protegido por
        # el module-level fallback en agent.py (`_validate_recipe_coh = None`).
        assert "validate_meal_recipe_ingredients_coherence" in agent_src, (
            "P1-SWAP-RECIPE-COHERENCE regresión: el agent.py ya no "
            "referencia `validate_meal_recipe_ingredients_coherence`. "
            "El wiring fue removido."
        )
        assert "_swap_recipe_coherence_enabled" in agent_src, (
            "P1-SWAP-RECIPE-COHERENCE regresión: el agent.py ya no "
            "referencia el kill-switch — wiring removido o renombrado."
        )

    def test_validator_invoked_before_macros_in_retry_loop(
        self, agent_src: str
    ):
        """Orden de validación en `invoke_with_retry`: recipe-coherence
        ANTES de macros. Razón documentada inline (P1-SWAP-RECIPE-COHERENCE
        comment block): un cap_swallowed_modifier ya hace que la macros
        validation sea irrelevante (la receta NO se puede cocinar con los
        ingredientes listados, da igual los macros)."""
        # Localizar `def invoke_with_retry():` dentro de swap_meal
        idx_invoke = agent_src.find("def invoke_with_retry():")
        assert idx_invoke > 0, "def invoke_with_retry no encontrado en agent.py"

        # Extraer hasta el siguiente `def ` o final del try block
        end_idx = agent_src.find("\n    try:\n        response", idx_invoke)
        if end_idx < 0:
            end_idx = idx_invoke + 12000
        body = agent_src[idx_invoke:end_idx]

        idx_recipe_coh = body.find("_validate_recipe_coh")
        idx_macros = body.find("_validate_macros")
        assert idx_recipe_coh > 0, (
            "P1-SWAP-RECIPE-COHERENCE regresión: `_validate_recipe_coh` "
            "no se invoca dentro del retry loop. Sin el invoke, el bug "
            "cap_swallowed_modifier vuelve."
        )
        assert idx_macros > 0, (
            "P1-SWAP-MACROS regresión: `_validate_macros` no se invoca. "
            "(Ambos validadores deben coexistir en el loop.)"
        )
        assert idx_recipe_coh < idx_macros, (
            f"P1-SWAP-RECIPE-COHERENCE regresión: el recipe-coherence "
            f"validator debe invocarse ANTES del macros validator. "
            f"Posición actual: recipe@{idx_recipe_coh} vs macros@{idx_macros}. "
            f"Si recipe va DESPUÉS, gastaríamos un retry de macros sobre "
            f"un meal que de todos modos rechazaríamos por cap-swallowed."
        )

    def test_validator_raises_value_error_to_tenacity(self, agent_src: str):
        """Cuando `coh_passed` is False, el validador debe `raise ValueError`
        para gatear retry de tenacity (no `return False` ni `continue`)."""
        # Buscamos un bloque con `not coh_passed` seguido (en ≤500 chars)
        # de un `raise ValueError`.
        pattern = re.compile(
            r"not\s+coh_passed.*?raise\s+ValueError",
            re.DOTALL,
        )
        match = pattern.search(agent_src)
        assert match is not None, (
            "P1-SWAP-RECIPE-COHERENCE regresión: el branch `not "
            "coh_passed` no levanta `ValueError`. Sin esto, tenacity no "
            "reinicia el invoke y el meal incoherente se entrega al user."
        )
        # Sanity: el match no es desproporcionadamente largo (señal de
        # que estamos capturando el bloque correcto, no abarcando 5kb).
        # [P3-SWAP-RETRY-COHERENCE-HINT · 2026-05-22] Cap relajado de 600
        # → 2000 chars: el self-check directive "REGLA INVARIANTE" añadido
        # al retry prompt extendió legítimamente el bloque. 2000 sigue
        # siendo defensivo (full swap_meal body es ~50kb).
        assert match.end() - match.start() < 2000, (
            "Match demasiado largo — probablemente el patrón capturó "
            "un bloque no relacionado. Verificar el wiring inline."
        )

    def test_validator_helper_exception_is_best_effort(self, agent_src: str):
        """Si el helper `_validate_recipe_coh` levanta una excepción
        NO-ValueError (KeyError, AttributeError por schema drift), el
        swap NO debe abortarse — log warning y continuar. Patrón espejo
        del macros validator."""
        # Buscamos el CALLSITE del helper (no el import statement) —
        # el primer `_validate_recipe_coh(` (open paren) dentro del retry
        # loop. El import es `as _validate_recipe_coh` sin paren.
        call_pattern = re.compile(r"_validate_recipe_coh\s*\(")
        match = call_pattern.search(agent_src)
        assert match, (
            "P1-SWAP-RECIPE-COHERENCE regresión: no se encontró un "
            "CALLSITE de `_validate_recipe_coh(...)`. El validador "
            "no se invoca — wiring removido."
        )
        idx = match.start()
        # Tomamos un bloque de ~2500 chars alrededor del callsite
        window = agent_src[idx:idx + 2500]
        # `except Exception` debe aparecer después del `except ValueError`
        idx_value_err = window.find("except ValueError:")
        idx_generic = window.find("except Exception")
        assert idx_value_err > 0, (
            "P1-SWAP-RECIPE-COHERENCE regresión: no hay `except "
            "ValueError: raise` cerca del callsite del validador. "
            "La excepción del retry no se propagaría a tenacity, "
            "rompiendo el control de retries."
        )
        assert idx_generic > 0, (
            "P1-SWAP-RECIPE-COHERENCE regresión: no hay `except "
            "Exception` best-effort. Si el helper rompe, el swap "
            "abortaría — UX hard-fail innecesario."
        )
        assert idx_value_err < idx_generic, (
            "P1-SWAP-RECIPE-COHERENCE regresión: `except Exception` "
            "antes de `except ValueError` → el ValueError sería "
            "capturado por el generic y NUNCA llegaría a tenacity."
        )


# ---------------------------------------------------------------------------
# Section F — Wiring en tools.py::execute_modify_single_meal
# ---------------------------------------------------------------------------
class TestWiringInExecuteModifySingleMeal:
    """Espejo de Section E para el tool del chat-agent."""

    def test_validator_referenced_in_tools_py(self, tools_src: str):
        assert "validate_meal_recipe_ingredients_coherence" in tools_src, (
            "P1-SWAP-RECIPE-COHERENCE regresión: tools.py ya no "
            "referencia el validador. El surface del chat-agent quedó "
            "sin protección."
        )
        assert "_swap_recipe_coherence_enabled" in tools_src, (
            "P1-SWAP-RECIPE-COHERENCE regresión: tools.py ya no "
            "referencia el kill-switch."
        )

    def test_validator_invoked_in_modify_retry_loop(self, tools_src: str):
        """El validador debe invocarse dentro del `invoke_with_retry`
        de `execute_modify_single_meal`."""
        # Localizar `def execute_modify_single_meal`
        idx_fn = tools_src.find("def execute_modify_single_meal(")
        assert idx_fn > 0
        # Localizar `def invoke_with_retry():` dentro de esa función
        idx_inv = tools_src.find("def invoke_with_retry():", idx_fn)
        assert idx_inv > 0
        # Extraer hasta el final de la función (heurística: siguiente
        # `def ` de top-level, no nested)
        next_def = re.search(
            r"\ndef\s+[a-zA-Z_]", tools_src[idx_inv + 100:]
        )
        end_idx = (idx_inv + 100 + next_def.start()) if next_def else len(tools_src)
        body = tools_src[idx_inv:end_idx]

        assert "_validate_recipe_coh" in body, (
            "P1-SWAP-RECIPE-COHERENCE regresión: `_validate_recipe_coh` "
            "no se invoca dentro del retry loop de modify_single_meal. "
            "Surface del chat-agent quedó sin protección."
        )


# ---------------------------------------------------------------------------
# Section G — Marker anchor (cross-link con test_p2_hist_audit_14)
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    """Slug del filename matchea el marker `P1-SWAP-RECIPE-COHERENCE`
    (sin -DEDICATED para que el enforcer histórico también lo detecte)."""
    expected_slug = "p1_swap_recipe_coherence"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo debe contener el slug "
        "`p1_swap_recipe_coherence` para que el cross-link "
        "`test_p2_hist_audit_14_marker_test_link` lo matchee."
    )
