"""[P1-PDF-XSS-BLANKET · 2026-05-12] Defensa preventiva contra XSS via
`element.innerHTML = ...` en frontend bundle.

Estado actual: el único `innerHTML =` activo es el del PDF builder de
`Dashboard.jsx:1478` (handleDownloadShoppingList). Ese bloque YA está
auditado — todas las interpolaciones de data no-confiable
(`cat`, `display`, `displayQty`, `_inventoryNote`, `durationText`,
`emptyMessageTitle`, `emptyMessageDesc`, `deltaItemsRemoved`) pasan por
`escapeHtml(...)`. Tests P1-1 (`escapeHtml.test.js`) anchorearon esas 4
interpolaciones críticas.

Problema que este test cierra:
    Si un futuro callsite añade un nuevo `element.innerHTML = ...` (por
    ejemplo en otro export PDF, modal con HTML dinámico, etc.), el test
    P1-1 NO lo detecta — solo cubre nombres específicos. Sin red blanket,
    una regresión XSS puede deslizarse silenciosamente.

Diseño:
    Escanea `frontend/src/**/*.{js,jsx}` excluyendo tests/setup. Cuenta
    todas las ocurrencias de `\.innerHTML\s*=`. Falla si:
      - El count excede el baseline de callsites whitelisted con marker
        `// [P1-PDF-XSS-AUDITED: <razón>]` o `// [XSS-SAFE: <razón>]`
        en las 5 líneas previas.

Whitelist por proximidad (pattern simple, mismo que P0-FRONTEND-ANALYTICS):
    Cualquier nuevo `innerHTML =` debe llevar un comment marker en las 5
    líneas previas justificando la auditoría. Sin marker → violation.

Excluye:
    - `__tests__/` (Vitest, no llega al bundle).
    - `*.test.{js,jsx}` (mismo).
    - `setupTests.js` (mismo).
    - `node_modules/` (no aplicable, ya fuera de src).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND_SRC = _REPO_ROOT / "frontend" / "src"

_INNERHTML_RE = re.compile(r"\.innerHTML\s*=")
_WHITELIST_MARKERS = (
    "P1-PDF-XSS-AUDITED",
    "XSS-SAFE",
)
_EXCLUDED_GLOBS = (
    "__tests__/",
    ".test.js",
    ".test.jsx",
    "setupTests.js",
)


def _iter_frontend_src_files():
    for ext in ("*.js", "*.jsx"):
        for fp in _FRONTEND_SRC.rglob(ext):
            rel = fp.relative_to(_FRONTEND_SRC).as_posix()
            if any(excl in rel for excl in _EXCLUDED_GLOBS):
                continue
            yield fp


def _has_whitelist_marker_nearby(lines: list[str], idx: int, window: int = 5) -> bool:
    start = max(0, idx - window)
    context = "\n".join(lines[start:idx])
    return any(marker in context for marker in _WHITELIST_MARKERS)


# ---------------------------------------------------------------------------
# 1. Estado actual: solo Dashboard.jsx:1478 tiene innerHTML, y es el único
#    callsite del PDF builder ya auditado por P1-1.
# ---------------------------------------------------------------------------
def test_known_callsite_dashboard_pdf_builder():
    """El callsite del PDF builder en Dashboard.jsx:1478 es el ÚNICO
    `innerHTML =` activo conocido. Si este test falla porque el archivo
    se renombró o el callsite se movió, actualizar este anchor — pero
    NO eliminar la regla blanket de abajo."""
    dash = _FRONTEND_SRC / "pages" / "Dashboard.jsx"
    assert dash.exists(), (
        "Dashboard.jsx no encontrado — refactor mayor del frontend? "
        "Actualizar este test para reflejar nueva ubicación del PDF builder."
    )
    text = dash.read_text(encoding="utf-8")
    matches = _INNERHTML_RE.findall(text)
    assert matches, (
        "Dashboard.jsx perdió su `innerHTML =` — si el PDF builder se "
        "refactorizó a DocumentFragment/createElement (más seguro), "
        "BIEN — eliminar este anchor. Si se eliminó accidentalmente, "
        "el PDF de lista de compras está roto."
    )


# ---------------------------------------------------------------------------
# 2. Blanket: cualquier callsite NUEVO sin marker es violation.
# ---------------------------------------------------------------------------
def test_no_unwhitelisted_innerhtml_assignment():
    """Cada `\\.innerHTML\\s*=` en frontend/src debe tener marker
    `[P1-PDF-XSS-AUDITED: <razón>]` o `[XSS-SAFE: <razón>]` en las 5
    líneas previas.

    Si tu callsite es legítimamente seguro (data 100% controlada por el
    código mismo, no LLM/usuario/Supabase), añade el marker. Si NO puedes
    justificar la seguridad, refactoriza a `textContent =` o construye con
    `createElement` + `appendChild`.
    """
    violations = []
    for fp in _iter_frontend_src_files():
        text = fp.read_text(encoding="utf-8")
        if ".innerHTML" not in text:
            continue
        lines = text.split("\n")
        for idx, line in enumerate(lines):
            if not _INNERHTML_RE.search(line):
                continue
            # Skip línea-comentario que solo MENCIONA innerHTML (este test
            # lo hace en su propio source).
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            if _has_whitelist_marker_nearby(lines, idx):
                continue
            rel = fp.relative_to(_REPO_ROOT).as_posix()
            violations.append(f"{rel}:{idx + 1}: {line.strip()[:120]}")

    # [P3-PDF-POLISH-4 · 2026-05-14] Pre-fix había whitelist brittle por
    # path:line (Dashboard.jsx:1478) que driftaba cada vez que se añadía
    # contenido arriba del callsite. Solución sostenible: el callsite
    # ahora tiene marker `[P1-PDF-XSS-AUDITED: ...]` inline en las 5
    # líneas previas (auto-detectado por `_has_whitelist_marker_nearby`).
    # Si alguien refactoriza el PDF builder, debe preservar (o añadir)
    # el marker al nuevo callsite — no hay whitelist hardcoded que
    # mantener.
    real_violations = list(violations)

    assert not real_violations, (
        "P1-PDF-XSS-BLANKET violation: nuevos `innerHTML =` sin marker "
        "de auditoría en frontend bundle. Cada callsite es vector "
        "potencial de XSS si interpola data no-confiable.\n\n"
        "Fix: si el callsite es safe, añadir comment "
        "`// [P1-PDF-XSS-AUDITED: <razón>]` o `// [XSS-SAFE: <razón>]` "
        "en las 5 líneas previas. Si NO puedes justificar, refactoriza "
        "a `textContent =` o `createElement`/`appendChild`.\n\n"
        "Violations:\n  " + "\n  ".join(real_violations)
    )


# ---------------------------------------------------------------------------
# 3. Sanity: el escáner respeta los markers
# ---------------------------------------------------------------------------
def test_whitelist_marker_is_respected(tmp_path):
    fake_src = tmp_path / "src"
    fake_src.mkdir()
    fake_file = fake_src / "modal.js"
    fake_file.write_text(
        "function renderModal() {\n"
        "    // Static HTML, no user data interpolated.\n"
        "    // [XSS-SAFE: literal string from constants module]\n"
        "    el.innerHTML = MODAL_TEMPLATE;\n"
        "}\n",
        encoding="utf-8",
    )
    text = fake_file.read_text(encoding="utf-8")
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        if _INNERHTML_RE.search(line):
            assert _has_whitelist_marker_nearby(lines, idx), (
                "Parser está roto: marker `[XSS-SAFE: ...]` no detectado "
                "en window de 5 líneas previas."
            )
            return
    pytest.fail("Test fixture mal construido: no encontró innerHTML.")


def test_unwhitelisted_callsite_is_caught(tmp_path):
    fake_src = tmp_path / "src"
    fake_src.mkdir()
    fake_file = fake_src / "broken.js"
    fake_file.write_text(
        "function renderUserMessage(msg) {\n"
        "    el.innerHTML = msg; // sin marker, sin escape — vector XSS\n"
        "}\n",
        encoding="utf-8",
    )
    text = fake_file.read_text(encoding="utf-8")
    lines = text.split("\n")
    found_unmarked = False
    for idx, line in enumerate(lines):
        if _INNERHTML_RE.search(line):
            if not _has_whitelist_marker_nearby(lines, idx):
                found_unmarked = True
                break
    assert found_unmarked, (
        "Parser está roto: callsite sin marker debe ser flageado."
    )


# ---------------------------------------------------------------------------
# 4. Anchor preservado para bookkeeping interno
# ---------------------------------------------------------------------------
def test_anchor_present_in_test_file():
    this_file = Path(__file__).read_text(encoding="utf-8")
    assert "P1-PDF-XSS-BLANKET" in this_file
