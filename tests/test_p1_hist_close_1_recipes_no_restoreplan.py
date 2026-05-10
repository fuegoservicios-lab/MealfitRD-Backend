"""[P1-HIST-CLOSE-1 · 2026-05-10] Cross-link Python del fix frontend.

Existe un test Vitest en `frontend/src/__tests__/Recipes.p1_hist_close_1_no_restorePlan.test.js`
que cubre el caso desde el lado del runner JS. Este test Python existe
para satisfacer el contrato del cross-link
`test_p2_hist_audit_14_marker_test_link.py` (slug del marker debe
matchear `tests/test_<slug>*.py`) y para que el suite de pytest detecte
regresiones del flow Recetas↔Historial sin depender exclusivamente del
runner Vitest.

Bug protegido:
    `Recipes.jsx::handleCookClick` usaba `restorePlan(planData)` tras
    expandir una receta. El path legacy `restorePlan` hace UPDATE
    directo client-side de `plan_data + name + calories + macros` —
    duplicando el write que ya hizo `/api/plans/recipe/expand`
    (plans.py:2860 → `update_meal_plan_data`) Y arrastrando
    `name/calories/macros` posiblemente stale del cliente al server.

    Si un chunk worker recalculaba kcal/macros entre page-load y
    cook-click (e.g., al expandir el plan con un nuevo bloque de días),
    el snapshot del cliente quedaba stale y el write redundante pisaba
    los valores frescos del server con los stale del cliente — drift
    `plan_data` ↔ columnas top-level idéntico al que P0-HIST-2 cerró
    para el path Historial.

    Fix: droppear la llamada. Server-side persist es SSOT.
    localStorage update se preserva para consistencia inmediata UI.

Cobertura:
    - Recipes.jsx NO importa/destructura `restorePlan` del context.
    - Recipes.jsx NO invoca `restorePlan(...)` en ningún call site.
    - Recipes.jsx preserva `localStorage.setItem('mealfit_plan', ...)`.
    - El anchor `[P1-HIST-CLOSE-1 · 2026-05-10]` está presente.

Limitaciones:
    - Parser estático: regex sobre el source. Si Recipes.jsx migra a
      TypeScript con un tipado diferente, los patrones pueden necesitar
      actualización.
    - NO valida el endpoint backend (cobertura existente:
      `test_recipe_expand_*` si aplica + integración manual).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BACKEND_DIR.parent.parent
_RECIPES_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Recipes.jsx"


def _read_recipes_source() -> str:
    """Lee Recipes.jsx desde el filesystem. Falla loud si no existe
    (mejor que silently-skip — un rename del archivo debe ser explícito).
    """
    assert _RECIPES_JSX.exists(), (
        f"Recipes.jsx no encontrado en {_RECIPES_JSX}. ¿Fue renombrado/"
        f"movido? Si es intencional, actualizar este test."
    )
    return _RECIPES_JSX.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    """Remueve comments JS (// y /* */) para que matches dentro de
    comentarios explicativos no fallen el test. La documentación del
    fix MENCIONA `restorePlan` en texto descriptivo — ese mention no
    es un call site real.

    Cuidado: no es un parser JS completo; el regex de `//` excluye URLs
    estilo `https://` revisando el carácter previo (`[^:]`).
    """
    # Block comments: /* ... */ (greedy non-greedy mode)
    src = re.sub(r"/\*[\s\S]*?\*/", "", src)
    # Line comments: // ... hasta fin de línea, evitando `https://`.
    src = re.sub(r"(^|[^:])//[^\n]*", r"\1", src, flags=re.MULTILINE)
    return src


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_anchor_present_in_source():
    """Marker `[P1-HIST-CLOSE-1 · 2026-05-10]` debe estar visible para
    que el grep desde MEMORY.md / CLAUDE.md encuentre el cierre.
    """
    src = _read_recipes_source()
    assert re.search(r"\[P1-HIST-CLOSE-1\s*·\s*2026-05-10\]", src), (
        "Marker `[P1-HIST-CLOSE-1 · 2026-05-10]` ausente en Recipes.jsx. "
        "Si el comment fue refactorizado, restaurar el anchor — sin él "
        "el grep desde memoria/CLAUDE.md no encuentra el cierre."
    )


# ---------------------------------------------------------------------------
# 2. NO destructure de `restorePlan` del context
# ---------------------------------------------------------------------------
def test_no_restoreplan_destructure_from_useassessment():
    """`useAssessment()` puede destructurar planData/formData pero NO
    `restorePlan` — usarlo aquí reintroduce el bug.
    """
    src = _read_recipes_source()
    pattern = re.compile(
        r"const\s*\{[^}]*\brestorePlan\b[^}]*\}\s*=\s*useAssessment\s*\(\s*\)"
    )
    assert not pattern.search(src), (
        "Recipes.jsx destructura `restorePlan` desde useAssessment(). "
        "El path legacy duplica el write server-side y arrastra "
        "name/calories/macros stale al UPDATE — droppear la "
        "destructuración."
    )


# ---------------------------------------------------------------------------
# 3. NO call site `restorePlan(...)` (excluye comments)
# ---------------------------------------------------------------------------
def test_no_restoreplan_call_site():
    """Cualquier `restorePlan(<args>)` fuera de comments es regresión."""
    src = _strip_comments(_read_recipes_source())
    matches = re.findall(r"\brestorePlan\s*\(", src)
    assert not matches, (
        f"Recipes.jsx invoca `restorePlan(...)` en {len(matches)} sitio(s) "
        f"de código activo (excluyendo comments). El server-side persist "
        f"de `/api/plans/recipe/expand` es SSOT — la llamada client-side "
        f"es redundante y dañina (pisa el write fresco del server con "
        f"snapshot stale del cliente)."
    )


# ---------------------------------------------------------------------------
# 4. NO import named `restorePlan` desde context o config/api
# ---------------------------------------------------------------------------
def test_no_restoreplan_named_import():
    """Defensa-en-profundidad: aunque hoy se accede vía useAssessment(),
    asegurar que un named import directo tampoco se introduzca en un
    refactor futuro.
    """
    src = _read_recipes_source()
    patterns = [
        re.compile(
            r"import\s*\{[^}]*\brestorePlan\b[^}]*\}\s*from\s*[\'\"]"
            r"[^\'\"]*context/AssessmentContext"
        ),
        re.compile(
            r"import\s*\{[^}]*\brestorePlan\b[^}]*\}\s*from\s*[\'\"]"
            r"[^\'\"]*config/api"
        ),
    ]
    for pattern in patterns:
        assert not pattern.search(src), (
            f"Recipes.jsx tiene un named import de `restorePlan` "
            f"matching `{pattern.pattern}`. Drop el import — el server "
            f"endpoint persiste el cambio."
        )


# ---------------------------------------------------------------------------
# 5. Preserva el localStorage.setItem('mealfit_plan', ...)
# ---------------------------------------------------------------------------
def test_localstorage_setitem_mealfit_plan_preserved():
    """Si un revert quita el `localStorage.setItem` junto con el
    `restorePlan`, el usuario perdería la consistencia inmediata UI
    al navegar a /plan tras expandir. Alertar específicamente.
    """
    src = _read_recipes_source()
    pattern = re.compile(
        r"localStorage\.setItem\s*\(\s*[\'\"]mealfit_plan[\'\"]\s*,\s*"
        r"JSON\.stringify\s*\(\s*planData\s*\)"
    )
    assert pattern.search(src), (
        "Recipes.jsx ya no llama `localStorage.setItem('mealfit_plan', "
        "JSON.stringify(planData))` tras expandir la receta. Esto rompe "
        "la consistencia UI inmediata: al navegar a /plan post-expand, "
        "el usuario vería la receta original hasta que el siguiente "
        "fetch del server actualice el cache local."
    )


# ---------------------------------------------------------------------------
# 6. Anchor pedagógico: comment menciona el endpoint backend SSOT
# ---------------------------------------------------------------------------
def test_comment_references_backend_endpoint_as_ssot():
    """El comment del fix debe apuntar al endpoint backend que persiste
    el cambio. Sin esto, un futuro contributor podría borrar el comment
    "limpiando" el código y perder la justificación del single-write.
    """
    src = _read_recipes_source()
    assert "/api/plans/recipe/expand" in src, (
        "Comment del fix no menciona el endpoint backend "
        "`/api/plans/recipe/expand` que persiste server-side. Restaurar "
        "el anchor pedagógico para que un futuro reader entienda el "
        "porqué del single-write."
    )


# ---------------------------------------------------------------------------
# 7. Sanity del strip_comments helper
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("input_src,should_contain_restoreplan", [
    ("// restorePlan(foo)", False),                          # line comment
    ("/* restorePlan(foo) */", False),                       # block comment
    ("foo(); restorePlan(bar);", True),                      # actual call
    ("const url = 'https://example.com/restorePlan';", False),  # URL false-positive
    ("/*\n * restorePlan(foo)\n */", False),                 # multiline block
])
def test_strip_comments_handles_edge_cases(input_src, should_contain_restoreplan):
    """El helper `_strip_comments` debe remover correctamente los
    contextos donde `restorePlan` aparece en docs sin ser un call
    site real.
    """
    stripped = _strip_comments(input_src)
    has_call = bool(re.search(r"\brestorePlan\s*\(", stripped))
    assert has_call == should_contain_restoreplan, (
        f"`_strip_comments({input_src!r})` → {stripped!r} "
        f"debió {'preservar' if should_contain_restoreplan else 'remover'} "
        f"la mención de `restorePlan(`."
    )
