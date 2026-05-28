"""[P2-PROD-FINAL-2 · 2026-05-23] Regression guards del bundle P2 que
cierra el audit prod-readiness 2026-05-23 (post P1-PROD-FINAL-1).

Bundle scope (5 P2):
  1. **P2-MIGRATION-SSOT-LINT**  — `test_p2_migrations_ssot_no_drift.py`
     (archivo separado, 4 tests).
  2. **P2-PANTRY-TURBO-HOLD-CLEANUP** — useEffect unmount cleanup en
     `Pantry.jsx` que clearea TODOS los `holdIntervalRef`/`holdTimeoutRef`
     entries pendientes al desmontar (cierra edge case donde stopHolding
     solo cubre el id explícito del pointer event).
  3. **P2-FRONTEND-LOCALSTORAGE-LINT** — ESLint `no-restricted-syntax` rule
     a nivel `warn` que matchea `localStorage.{get,set,remove}Item(...)` raw.
     Override en `utils/safeLocalStorage.js` (wrapper SSOT) y en tests.
     Nivel warn (no error) por 60 callsites legacy boy-scout.
  4. **P2-KNOBS-OPERATIONAL-DOC** — `backend/docs/knobs_reference.md` con
     mecanismo de descubrimiento + knobs de alto valor + link desde
     `.env.example`. Cierra el gap "operador nuevo no sabe qué tunear".
  5. **P2-CSS-IMPORTANT-OVERLOAD** — diferido como decisión (memoria
     `decision_p2_css_important_overload_2026_05_23.md`). 162 ocurrencias
     en 11 CSS Modules son bound a scope componente, no specificity wars.

Tests de regresión (parser-based, no ejecutan JSX/Python productivo).

Tooltip-anchor: `P2-PROD-FINAL-2-START` | bundle anchor 2026-05-23.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_FRONTEND_SRC = _REPO_ROOT / "frontend" / "src"


# ---------------------------------------------------------------------------
# Sección 1 — P2-PANTRY-TURBO-HOLD-CLEANUP
# ---------------------------------------------------------------------------
def test_pantry_has_unmount_cleanup_for_hold_refs():
    """`Pantry.jsx` debe tener un useEffect dedicado que cleare TODOS los
    `holdIntervalRef.current` y `holdTimeoutRef.current` entries en unmount.

    Pre-fix `stopHolding(id)` solo limpiaba el id del pointer event, así
    que un timeout(400ms) pending O un interval(80ms) firing al momento
    del unmount sobrevivían y disparaban handleUpdateQuantity sobre estado
    stale.
    """
    pantry = _FRONTEND_SRC / "pages" / "Pantry.jsx"
    text = pantry.read_text(encoding="utf-8")
    assert "P2-PANTRY-TURBO-HOLD-CLEANUP" in text, (
        "Pantry.jsx no tiene el anchor `P2-PANTRY-TURBO-HOLD-CLEANUP` — "
        "un refactor pudo haber removido el cleanup."
    )
    # Debe haber un useEffect que itera holdTimeoutRef y holdIntervalRef.
    # Estrategia: confirmamos que el bloque contiene `Object.values(...)` +
    # clearTimeout + clearInterval en una proximidad razonable.
    assert "Object.values(_timeouts).forEach" in text or "Object.values(holdTimeoutRef" in text, (
        "Pantry.jsx no itera Object.values(holdTimeoutRef.current). "
        "Sin iteración, cleanup parcial solo libera el id reciente."
    )
    # clearTimeout y clearInterval ambos presentes en cleanup.
    cleanup_block_pattern = re.compile(
        r"P2-PANTRY-TURBO-HOLD-CLEANUP[\s\S]{0,1500}clearTimeout[\s\S]{0,500}clearInterval",
        re.MULTILINE,
    )
    assert cleanup_block_pattern.search(text), (
        "Bloque cleanup de Pantry.jsx no contiene tanto clearTimeout como "
        "clearInterval entre el anchor y los 1500 chars siguientes."
    )


# ---------------------------------------------------------------------------
# Sección 2 — P2-FRONTEND-LOCALSTORAGE-LINT
# ---------------------------------------------------------------------------
def test_eslint_config_has_no_restricted_syntax_rule():
    """`frontend/eslint.config.js` debe tener una rule `no-restricted-syntax`
    que matchea `localStorage.{get,set,remove}Item(...)` con el selector
    AST. Sin la rule, las regresiones a raw localStorage pasan silenciosas."""
    eslint = _REPO_ROOT / "frontend" / "eslint.config.js"
    text = eslint.read_text(encoding="utf-8")
    assert "P2-FRONTEND-LOCALSTORAGE-LINT" in text, (
        "eslint.config.js no tiene el anchor `P2-FRONTEND-LOCALSTORAGE-LINT`."
    )
    assert "no-restricted-syntax" in text, (
        "eslint.config.js no declara la rule no-restricted-syntax."
    )
    # Selector debe matchear localStorage.{getItem,setItem,removeItem}
    assert "localStorage" in text and "getItem|setItem|removeItem" in text, (
        "eslint.config.js no contiene el selector AST que matchea las 3 "
        "métodos de localStorage. El patrón esperado: "
        "callee.object.name='localStorage' + callee.property.name=getItem|setItem|removeItem"
    )


def test_eslint_config_allows_localstorage_in_wrapper():
    """El wrapper `safeLocalStorage.js` DEBE usar raw localStorage —
    es el código que la rule existe para evitar duplicar. Override
    debe estar declarado para ese archivo."""
    eslint = _REPO_ROOT / "frontend" / "eslint.config.js"
    text = eslint.read_text(encoding="utf-8")
    # El override debe mencionar safeLocalStorage.js en su `files` glob.
    assert "safeLocalStorage.js" in text, (
        "eslint.config.js no tiene override para `src/utils/safeLocalStorage.js`. "
        "El wrapper SSOT lanza false-positive de la rule sin el override."
    )


# ---------------------------------------------------------------------------
# Sección 3 — P2-KNOBS-OPERATIONAL-DOC
# ---------------------------------------------------------------------------
def test_knobs_reference_doc_exists():
    """El doc operacional debe existir."""
    doc = _BACKEND_ROOT / "docs" / "knobs_reference.md"
    assert doc.exists(), (
        f"Doc `{doc}` no existe. Cierra el gap operacional "
        "P2-KNOBS-OPERATIONAL-DOC del audit 2026-05-23."
    )


def test_knobs_reference_doc_has_required_sections():
    """El doc debe cubrir las 3 secciones que el operador necesita:
    mecanismo de auto-registro, endpoint público, knobs de alto valor."""
    doc = _BACKEND_ROOT / "docs" / "knobs_reference.md"
    text = doc.read_text(encoding="utf-8")
    required = [
        "P2-KNOBS-OPERATIONAL-DOC",          # anchor
        "_KNOBS_REGISTRY",                    # mecanismo
        "/health/version",                    # endpoint
        "get_knobs_registry_snapshot",        # Python interactivo
        "MEALFIT_SHOPPING_COHERENCE_GUARD",   # ejemplo high-value
        "MEALFIT_CB_FAILURE_THRESHOLD",       # ejemplo CB
        "Cómo añadir un knob nuevo",          # workflow
    ]
    missing = [r for r in required if r not in text]
    assert not missing, (
        f"Doc knobs_reference.md no contiene secciones requeridas: "
        f"{missing}. El doc debe ser navegable para un operador nuevo "
        f"sin grep cross-codebase."
    )


def test_env_example_links_to_knobs_doc():
    """`backend/.env.example` debe linkear al doc operacional. Sin el
    link, un operador que lee solo `.env.example` no se entera de los
    150+ knobs ocultos."""
    env_example = _BACKEND_ROOT / ".env.example"
    text = env_example.read_text(encoding="utf-8")
    assert "backend/docs/knobs_reference.md" in text, (
        ".env.example no linkea a `backend/docs/knobs_reference.md`. "
        "Añadir un comentario en la sección 'Knobs operacionales' que "
        "apunte al doc."
    )


# ---------------------------------------------------------------------------
# Sección 4 — Marker + CSS decision anchor
# ---------------------------------------------------------------------------
def test_last_known_pfix_marker_bumped():
    """`_LAST_KNOWN_PFIX` en app.py debe tener fecha >= 2026-05-24 (cierre
    del bundle P2-PROD-FINAL-2). Un bundle posterior puede supersede; lo
    que NO puede ocurrir es que se revierta a una fecha previa sin revertir
    los 4 cambios del bundle.

    [Relajado del exact-match `P2-PROD-FINAL-2 · 2026-05-24` original
    2026-05-24: mismo patrón aplicado en P1-PROD-FINAL-1 (ver
    test_p1_prod_final_1.py). Las otras 8 assertions del archivo enforzan
    que los 4 fixes siguen vivos.]"""
    from datetime import date, datetime

    app_py = _BACKEND_ROOT / "app.py"
    text = app_py.read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py."
    marker = m.group(1)
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_m, f"Marker `{marker}` no contiene fecha ISO."
    marker_date = datetime.strptime(date_m.group(1), "%Y-%m-%d").date()
    floor = date(2026, 5, 24)
    assert marker_date >= floor, (
        f"Marker `{marker}` con fecha {marker_date} < floor {floor} "
        "(P2-PROD-FINAL-2). Si revertiste el marker debes también revertir "
        "los 4 cambios del bundle."
    )


def test_migrations_ssot_test_file_exists():
    """Sanity: el archivo dedicado al gap #1 del bundle existe."""
    assert (_BACKEND_ROOT / "tests" / "test_p2_migrations_ssot_no_drift.py").exists()


# ---------------------------------------------------------------------------
# Sección 5 — CLAUDE.md aún bajo el cap
# ---------------------------------------------------------------------------
def test_claude_md_still_under_cap_after_bundle():
    """El bundle P2-PROD-FINAL-2 NO toca CLAUDE.md (todo el contenido
    nuevo vive en `backend/docs/` o memoria). Sanity check del cap
    para detectar que un edit accidental no haya engordado CLAUDE.md."""
    claude_md = _REPO_ROOT / "CLAUDE.md"
    cap_test = _BACKEND_ROOT / "tests" / "test_p3_claudemd_cap.py"
    cap_text = cap_test.read_text(encoding="utf-8")
    cap_match = re.search(r"_DEFAULT_CAP\s*=\s*(\d+)", cap_text)
    assert cap_match, "No pude parsear _DEFAULT_CAP del test cap."
    cap = int(cap_match.group(1))
    size = claude_md.stat().st_size
    assert size <= cap, (
        f"CLAUDE.md = {size} > cap {cap}. El bundle P2-PROD-FINAL-2 no "
        f"debería haber tocado CLAUDE.md — ¿edit accidental?"
    )
