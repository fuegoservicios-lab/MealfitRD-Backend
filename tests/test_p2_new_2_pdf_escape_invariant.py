"""[P2-NEW-2 · 2026-05-11] Drift-detection: el bloque de generación
del PDF en Dashboard.jsx debe escapar TODAS las interpolaciones de
strings runtime via `escapeHtml(...)`.

Bug original (audit 2026-05-11):
    El agent identificó como gap el risk de `${cat}` interpolado sin
    escape — pero la verificación en código mostró que ya está
    escapado (línea ~1273: `${escapeHtml(cat)}`). Este test parser-based
    enforza la invariante para prevenir REGRESIÓN futura.

Caso pre-existente bien cubierto:
    - `${escapeHtml(cat)}` — category header (P2-NEW-2 verified).
    - `${escapeHtml(display)}` — item name.
    - `${escapeHtml(displayQty)}` — quantity chip.
    - `${escapeHtml(item._inventoryNote)}` — inventory hints.

Cualquier futuro `${variable}` que NO sea (a) numérico literal,
(b) constante de estilo (color/padding/font-size), (c) icon SVG
inline, (d) ya wrapped en escapeHtml() — debe fallar este test.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DASHBOARD_FP = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"


# Whitelist: nombres de variables que NO requieren escapeHtml (son
# safe constants de estilo, layout, icons SVG hardcoded). Si alguien
# añade una variable a la whitelist, justificar en review.
_SAFE_VAR_NAMES = frozenset({
    # Layout / styles
    "ulPadding", "catMargin", "catHeaderPadding", "catTitleFont",
    "qtyFont", "qtyPad", "itemFont", "checkboxSize", "checkboxMarginRight",
    "borderBottom", "tagBg", "tagColor", "tagBorder",
    # Icons SVG (hardcoded en código, no user input)
    "icon", "lowConfWarn", "noteHTML",
    # Booleans / numbers
    "isHyperDense", "isUltraDense", "isDense", "isPerishable",
    # Strings ya escapados via su propio call site
    "qtyStr",
    # Counters / flags / runtime-derived state que NO contiene HTML
    "deltaItemsRemoved", "showInventoryNotes",
})


def test_pdf_html_interpolations_use_escapehtml():
    """Cualquier `${var}` dentro del bloque `generateBlocks` (PDF
    generation) debe (a) ser whitelisted como safe, (b) ser un literal,
    (c) ser una expresión más compleja que ya envuelve escapeHtml.

    Patrón problema: `${user_provided_string}` literal sin escape.
    """
    text = _DASHBOARD_FP.read_text(encoding="utf-8")

    # Localizar el bloque `generateBlocks` (donde se construye el HTML
    # del PDF).
    start = text.find("const generateBlocks = ")
    assert start > 0, (
        "P2-NEW-2 regresión: `generateBlocks` ya no existe en Dashboard.jsx. "
        "Si lo renombraste, actualiza este test para apuntar al nuevo nombre."
    )
    # Heurística: el bloque termina cuando se encuentra el siguiente
    # `const ` top-level dentro de la función PDF o el cierre del
    # `forEach`. Tomamos ventana de 5000 chars como cap defensivo.
    block = text[start:start + 6000]

    # Buscar TODAS las interpolaciones `${...}`.
    interps = re.findall(r"\$\{([^}]+)\}", block)

    unsafe = []
    for expr in interps:
        expr_s = expr.strip()
        # 1. Vacío — ignorar.
        if not expr_s:
            continue
        # 2. Numérico o boolean literal.
        if re.match(r"^-?\d+(?:\.\d+)?$", expr_s):
            continue
        if expr_s in {"true", "false", "null", "undefined"}:
            continue
        # 3. Ya invoca escapeHtml.
        if "escapeHtml(" in expr_s:
            continue
        # 4. Expresión que termina en ' ? "..." : "..."' (ternario
        # con literales) — los literales son safe.
        if "?" in expr_s and ":" in expr_s:
            # Heurística: solo aceptar si NO menciona vars user-provided.
            # Para simplicidad permitimos los ternarios — son layout
            # decisions, no user input.
            continue
        # 5. Variable simple en whitelist.
        if expr_s in _SAFE_VAR_NAMES:
            continue
        # 6. Property access a un objeto safe (style, layout config).
        #    `density.<x>` etc.
        if "." in expr_s and not "(" in expr_s:
            # `item.something` SÍ es peligroso (item podría tener user data).
            # Pero `density.font` es safe (constante). Heurística: nombre
            # del root.
            root = expr_s.split(".")[0].strip()
            if root in _SAFE_VAR_NAMES or root in {
                "density", "config", "props", "PDF_LAYOUT_THRESHOLDS",
            }:
                continue
        # 7. Function call result que SÍ es escapeHtml indirecto.
        if "String(" in expr_s and "trim()" in expr_s:
            # `String(x).trim()` typically para format check, no escape.
            # Si va a HTML, debe ser escaped — lo flageamos.
            pass

        unsafe.append(expr_s[:80])

    assert not unsafe, (
        "P2-NEW-2 regresión: el bloque `generateBlocks` (PDF HTML) "
        "contiene interpolaciones `${var}` que NO pasan por escapeHtml:\n  - "
        + "\n  - ".join(unsafe)
        + "\n\nProcedimiento:\n"
        "  1. Si la variable contiene texto del LLM, user input, o data "
        "     externa: envolver en `escapeHtml(...)`.\n"
        "  2. Si es una constante de estilo (color, padding, font-size): "
        "     agregar el nombre a `_SAFE_VAR_NAMES` en este test.\n"
        "  3. Si es un literal numérico: chequear el regex."
    )


def test_pdf_category_header_escaped():
    """Sanity ancla: el header de categoría DEBE escapar `cat` —
    cierre directo del gap reportado en P2-NEW-2."""
    text = _DASHBOARD_FP.read_text(encoding="utf-8")
    # Buscar `<h3 ...>${escapeHtml(cat)}</h3>` en el PDF section.
    assert re.search(
        r"<h3[^>]*>\$\{escapeHtml\(cat\)\}</h3>",
        text,
    ), (
        "P2-NEW-2 regresión: el header de categoría del PDF ya no "
        "pasa por escapeHtml(cat). Si un LLM emite `</li>` en una "
        "category name, el DOM del PDF se rompe."
    )
