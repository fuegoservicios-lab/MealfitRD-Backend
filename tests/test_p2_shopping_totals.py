"""[P2-SHOPPING-TOTALS · 2026-05-16] Mostrar conteos de items en la lista
de compras (header chip "Total: N ítems" + per-section counts).

Antes (plan aeb25e1c): el usuario no sabía a primera vista cuánto iba a
tomar la compra (1 trip vs 2 trips). Tenía que contar mentalmente o
asumir. Beneficio UX:
  - "Total: 25 ítems" en el header → trip planning
  - "PERECEDEROS · 15 ítems" / "ESTABLES · 10 ítems" → balance entre
    secciones visible a primera vista

Fix solo-frontend (PDF rendering en Dashboard.jsx). Cero riesgo backend.
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DASHBOARD_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"


def test_section_counts_computed():
    """`perishableItemCount` y `stableItemCount` derivados de
    `Object.values(perishables/stables).reduce(...)`."""
    src = _DASHBOARD_JSX.read_text(encoding="utf-8")
    assert "const perishableItemCount = Object.values(perishables).reduce(" in src, (
        "perishableItemCount no calculado vía reduce sobre Object.values."
    )
    assert "const stableItemCount = Object.values(stables).reduce(" in src, (
        "stableItemCount no calculado."
    )


def test_total_items_already_exists():
    """`totalItems` ya está calculado más arriba (P1-PDF-3 pre-existente).
    Lo reusamos, no necesitamos duplicar."""
    src = _DASHBOARD_JSX.read_text(encoding="utf-8")
    assert "const totalItems = Object.values(consData).length;" in src, (
        "totalItems removido en algún refactor — sin él, el chip 'Total' "
        "del header no tiene la fuente esperada."
    )


def test_fmt_items_helper_pluralizes():
    """Helper `_fmtItems(n)` debe pluralizar — 1 ítem vs N ítems. Sin esto,
    veríamos textos awkward como '1 ítems' en planes muy chicos."""
    src = _DASHBOARD_JSX.read_text(encoding="utf-8")
    assert "const _fmtItems" in src
    # Pluralización vía ternary
    assert re.search(r"n === 1 \? ['\"]ítem['\"]\s*:\s*['\"]ítems['\"]", src), (
        "Helper _fmtItems no pluraliza correctamente (1 → 'ítem', N → 'ítems')."
    )


def test_header_includes_total_chip():
    """Header del PDF debe incluir chip 'Total: X ítems' además de
    'Ciclo' + 'Generado'."""
    src = _DASHBOARD_JSX.read_text(encoding="utf-8")
    # El chip Total: usa _fmtItems(totalItems)
    assert "Total: ${escapeHtml(_fmtItems(totalItems))}" in src, (
        "Chip 'Total' ausente del header del PDF. Sin él, el usuario no ve "
        "el conteo agregado del shopping list."
    )


def test_section_labels_include_counts():
    """Los labels de PERECEDEROS y ESTABLES deben incluir el count
    de items inline (e.g. 'PERECEDEROS · 15 ítems')."""
    src = _DASHBOARD_JSX.read_text(encoding="utf-8")
    assert "_fmtItems(perishableItemCount)" in src, (
        "Section label perishable no muestra el count — '· N ítems' ausente."
    )
    assert "_fmtItems(stableItemCount)" in src, (
        "Section label stable no muestra el count."
    )


def test_counts_escape_html():
    """Defense in depth: aunque _fmtItems devuelve strings controlados,
    todos los interpolados al PDF van por escapeHtml (consistencia con
    P1-PDF-XSS-AUDITED). Si futuro refactor cambia _fmtItems para
    aceptar input user-controlled, escapeHtml ya estaría aplicado."""
    src = _DASHBOARD_JSX.read_text(encoding="utf-8")
    # Las 3 ocurrencias de _fmtItems en interpolación al HTML deben usar escapeHtml
    interpolations = re.findall(r"\$\{escapeHtml\(_fmtItems\(\w+\)\)\}", src)
    assert len(interpolations) >= 3, (
        f"Esperaba ≥3 usos de `escapeHtml(_fmtItems(...))` en el HTML "
        f"(header + 2 section labels). Encontrados: {len(interpolations)}."
    )
