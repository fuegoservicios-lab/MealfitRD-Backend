"""[P3-AUDIT-1 · 2026-05-15] Test parser-based: el handler PDF de recetas
en `Recipes.jsx::handleDownloadPDF` emite telemetría success/failure y
genera filename con discriminador único.

Por qué este test:
    P3-SHOPPING-4 (2026-05-14) instrumentó telemetría
    `pdf_download_success` + `pdf_download_failed` en Dashboard.jsx para el
    PDF de lista de compras. Recipes.jsx (PDF de receta individual) quedó
    sin telemetría — operador no puede distinguir "feature no usado" de
    "feature roto", ambos producen 0 events. Adicional, el filename
    `Receta-${meal.name}.pdf` colisionaba para 2 recetas del mismo nombre
    en planes distintos (común: "Pollo guisado" aparece en múltiples
    planes); cada PDF nuevo sobrescribía al anterior en Downloads.

Fix esperado:
    - `trackEvent('recipe_pdf_download_success', {plan_id, meal_name,
       meal_type, recipe_steps, ingredients_count, is_expanded})` en el
       success branch.
    - `trackEvent('recipe_pdf_download_failed', {plan_id, meal_name,
       meal_type, error_name, error_message})` en el catch branch.
    - Filename con discriminador `Receta_<slug>_<plan_id[:8]>_<YYYY-MM-DD>.pdf`.
    - Ambos trackEvent en try/catch best-effort (analytics SDK falla NO debe
      romper el handler PDF).

Cross-link convention (P2-HIST-AUDIT-14): slug `p3_audit_1`.

Tooltip-anchor: P3-AUDIT-1-START | gap audit 2026-05-15
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RECIPES_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Recipes.jsx"


@pytest.fixture(scope="module")
def recipes_src() -> str:
    return _RECIPES_JSX.read_text(encoding="utf-8")


def _extract_handler_body(src: str) -> str:
    """Aísla `const handleDownloadPDF = async (meal) => { ... }` hasta el
    siguiente `const <name>` top-level del componente."""
    anchor = re.search(
        r"const\s+handleDownloadPDF\s*=\s*async\s*\(\s*meal\s*\)\s*=>\s*\{",
        src,
    )
    assert anchor is not None, (
        "P3-AUDIT-1 regresión: `const handleDownloadPDF = async (meal) => {` "
        "no encontrado en Recipes.jsx."
    )
    start = anchor.end()
    rest = src[start:]
    next_block = re.search(r"\n    (?:const\s+\w+\s*=|return\s*\()", rest)
    end = start + (next_block.start() if next_block else len(rest))
    return src[start:end]


@pytest.fixture(scope="module")
def handler_body(recipes_src: str) -> str:
    return _extract_handler_body(recipes_src)


# ---------------------------------------------------------------------------
# 1. Filename con discriminador único
# ---------------------------------------------------------------------------
def test_filename_includes_plan_id_prefix(handler_body: str):
    """El filename debe contener `${_planIdPrefix}` o equivalente derivado
    de `planData.id` truncado a 8 chars."""
    # Buscar pattern: filename: `Receta_..._${...planData?.id...slice(0, 8)...}_...`
    # Aceptamos cualquier forma que extraiga `planData.id` y la trunque.
    assert re.search(
        r"planData\?\.id[^;]*\.slice\(\s*0\s*,\s*8\s*\)",
        handler_body,
    ), (
        "P3-AUDIT-1 regresión: el filename no incluye `planData?.id` "
        "truncado a 8 chars como discriminador. Sin él, descargar 2 recetas "
        "del mismo nombre en planes distintos colisiona en Downloads. "
        "Aplicar `_planIdPrefix = (planData?.id || '').toString().slice(0, 8)`."
    )


def test_filename_includes_iso_date(handler_body: str):
    """El filename debe contener `${...new Date().toISOString().slice(0, 10)}`
    para discriminar re-descargas del mismo plan en días distintos."""
    assert re.search(
        r"new\s+Date\(\)\s*\.\s*toISOString\(\)\s*\.\s*slice\(\s*0\s*,\s*10\s*\)",
        handler_body,
    ), (
        "P3-AUDIT-1 regresión: el filename no incluye la fecha ISO de hoy "
        "como discriminador. Re-descargar la misma receta del mismo plan en "
        "días distintos sobrescribe la copia anterior. Aplicar "
        "`_today = new Date().toISOString().slice(0, 10)`."
    )


def test_filename_pattern_includes_both_discriminators(handler_body: str):
    """El template literal del filename debe interpolar AMBOS discriminadores."""
    # Patrón canónico: `filename: \`Receta_${slug}_${planIdPrefix}_${today}.pdf\``
    # Tolerar variaciones de orden, prefijos, etc. siempre que aparezcan
    # ambas variables en la línea `filename:`.
    filename_match = re.search(
        r"filename\s*:\s*`[^`]+`",
        handler_body,
    )
    assert filename_match, (
        "P3-AUDIT-1 regresión: línea `filename: \\`...\\`` no encontrada en "
        "el opt de html2pdf."
    )
    fname = filename_match.group(0)
    assert "_planIdPrefix" in fname or "planData" in fname, (
        f"P3-AUDIT-1 regresión: filename `{fname}` no interpola plan_id "
        f"discriminator. Patrón canónico: "
        f"`Receta_${{slug}}_${{_planIdPrefix}}_${{_today}}.pdf`."
    )
    assert "_today" in fname or "toISOString" in fname, (
        f"P3-AUDIT-1 regresión: filename `{fname}` no interpola fecha "
        f"discriminator."
    )


# ---------------------------------------------------------------------------
# 2. Telemetría success
# ---------------------------------------------------------------------------
def test_telemetry_success_event_emitted(handler_body: str):
    """`trackEvent('recipe_pdf_download_success', { plan_id, meal_name, ... })`
    debe aparecer en el success branch (después del save() exitoso)."""
    assert "recipe_pdf_download_success" in handler_body, (
        "P3-AUDIT-1 regresión: `trackEvent('recipe_pdf_download_success', ...)` "
        "no encontrado en handleDownloadPDF. Sin él, el operador no puede "
        "medir adopción del feature (success_rate = success / (success+failure))."
    )


def test_telemetry_success_includes_dimensions(handler_body: str):
    """El payload de success debe incluir `plan_id`, `meal_name`,
    `meal_type` como mínimo para correlación cross-canal + filtros básicos."""
    success_idx = handler_body.find("recipe_pdf_download_success")
    assert success_idx > -1
    window = handler_body[success_idx:success_idx + 800]
    for required_key in ("plan_id", "meal_name", "meal_type"):
        assert required_key in window, (
            f"P3-AUDIT-1 regresión: payload de `recipe_pdf_download_success` "
            f"no incluye `{required_key}`. Sin esa dimensión, el operador no "
            f"puede filtrar/agrupar el feature en analytics."
        )


def test_telemetry_success_is_best_effort(handler_body: str):
    """El `trackEvent` debe estar dentro de `try/catch` — analytics SDK
    fallando NO debe romper la generación del PDF."""
    success_idx = handler_body.find("recipe_pdf_download_success")
    assert success_idx > -1
    # Buscar `try {` antes del trackEvent, dentro de ~400 chars.
    window_back = handler_body[max(0, success_idx - 400):success_idx]
    assert re.search(r"try\s*\{", window_back), (
        "P3-AUDIT-1 regresión: `trackEvent('recipe_pdf_download_success')` no "
        "está envuelto en `try/catch`. Un fallo del SDK analytics puede "
        "romper el flow del handler."
    )


# ---------------------------------------------------------------------------
# 3. Telemetría failure
# ---------------------------------------------------------------------------
def test_telemetry_failure_event_emitted(handler_body: str):
    assert "recipe_pdf_download_failed" in handler_body, (
        "P3-AUDIT-1 regresión: `trackEvent('recipe_pdf_download_failed', ...)` "
        "no encontrado en el catch branch."
    )


def test_telemetry_failure_truncates_error_message(handler_body: str):
    """`error_message` debe truncarse a ≤200 chars para evitar payloads
    gigantes en GA/PostHog (algunos backends cortan a 256)."""
    failed_idx = handler_body.find("recipe_pdf_download_failed")
    assert failed_idx > -1
    window_back = handler_body[max(0, failed_idx - 600):failed_idx]
    assert re.search(r"\.slice\(\s*0\s*,\s*200\s*\)", window_back), (
        "P3-AUDIT-1 regresión: `error_message` no se trunca a 200 chars antes "
        "de emitirse en `recipe_pdf_download_failed`. Sin truncate, un stack "
        "trace largo del LLM puede saturar el payload analytics."
    )


def test_telemetry_failure_truncates_error_name(handler_body: str):
    """`error_name` debe truncarse a ≤64 chars."""
    failed_idx = handler_body.find("recipe_pdf_download_failed")
    window_back = handler_body[max(0, failed_idx - 600):failed_idx]
    assert re.search(r"\.slice\(\s*0\s*,\s*64\s*\)", window_back), (
        "P3-AUDIT-1 regresión: `error_name` no se trunca a 64 chars. Mismo "
        "patrón que P3-SHOPPING-4 en Dashboard."
    )


# ---------------------------------------------------------------------------
# 4. Anchor textual P3-AUDIT-1 presente
# ---------------------------------------------------------------------------
def test_anchor_present(recipes_src: str):
    assert "P3-AUDIT-1" in recipes_src, (
        "P3-AUDIT-1 regresión: anchor textual `P3-AUDIT-1` perdido en "
        "Recipes.jsx."
    )


# ---------------------------------------------------------------------------
# 5. Import del helper trackEvent
# ---------------------------------------------------------------------------
def test_trackevent_imported(recipes_src: str):
    assert re.search(
        r"import\s*\{[^}]*\btrackEvent\b[^}]*\}\s*from\s*['\"][^'\"]*analytics['\"]",
        recipes_src,
    ), (
        "P3-AUDIT-1 regresión: `import { trackEvent } from '../utils/analytics'` "
        "no encontrado en Recipes.jsx."
    )
