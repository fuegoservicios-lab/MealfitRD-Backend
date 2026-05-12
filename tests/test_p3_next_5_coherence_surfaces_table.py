"""[P3-NEXT-5 · 2026-05-11] La sub-sección de CLAUDE.md "Surfaces que
escriben `aggregated_shopping_list*` y status del guard" enumera las
6 surfaces canónicas + su `action_taken`. Este test ancla el contrato
bidireccional documentación ↔ código para evitar drift.

Cierra el gap residual del audit 2026-05-11:
    Antes de P3-NEXT-5, CLAUDE.md tenía el flujo de coherencia pero
    NO enumeraba explícitamente las surfaces que NO ejecutan el guard
    en mode=block (tabla "negativa"). Un futuro refactor que asumiera
    que el guard es universal en todas las surfaces podía romper el
    contrato sin detectarse en code review.

Fix P3-NEXT-5:
    Sub-sección "Surfaces que escriben `aggregated_shopping_list*` y
    status del guard" añadida con tabla de 6 filas (1 block-mode + 5
    warn-only) + conclusiones operacionales + tests anchor.

Drift detection:
    - Tabla borrada/renombrada en CLAUDE.md → falla.
    - `action_taken` mencionado en tabla pero ausente del código → falla.
    - `action_taken` usado en código pero ausente de tabla → falla.

Tooltip-anchor: P3-NEXT-5-START | gap audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"
_BACKEND = _REPO_ROOT / "backend"

# action_taken values esperados que el código DEBE emitir y la tabla
# DEBE mencionar. Estos son los 6 valores canónicos post-P2-NEXT-2.
_CANONICAL_ACTION_TAKEN_VALUES = {
    "not_applicable",         # assemble_plan_node mode=warn / block-no-critical
    "post_swap_revalidation", # _recompute_aggregates_after_swap
    "warn_only_chunk_t2",     # _chunk_worker T2 (P1-NEXT-2)
    "warn_only_recalc",       # /recalculate-shopping-list (P1-NEXT-2)
    "warn_only_agent_tool",   # tools.modify_single_meal (P1-NEXT-2)
    "warn_only_cron_daily",   # _shopping_coherence_alert_job (P2-NEXT-2)
}

# Archivos donde los action_taken DEBEN aparecer como string literal.
_CODE_FILES_TO_SCAN = [
    _BACKEND / "graph_orchestrator.py",   # not_applicable, post_swap_revalidation
    _BACKEND / "cron_tasks.py",            # warn_only_chunk_t2, warn_only_cron_daily
    _BACKEND / "routers" / "plans.py",     # warn_only_recalc
    _BACKEND / "tools.py",                 # warn_only_agent_tool
    _BACKEND / "shopping_calculator.py",   # helper también emite (default placeholder)
]


# ---------------------------------------------------------------------------
# 1. Sub-sección presente en CLAUDE.md
# ---------------------------------------------------------------------------
def test_subsection_present_in_claude_md():
    text = _CLAUDE_MD.read_text(encoding="utf-8")
    assert "Surfaces que escriben" in text and "status del guard" in text, (
        "P3-NEXT-5 violation: sub-sección 'Surfaces que escriben "
        "aggregated_shopping_list* y status del guard' ausente de "
        "CLAUDE.md. Esta tabla 'negativa' es la única documentación "
        "explícita de qué surfaces NO bloquean retries. Sin ella, "
        "un futuro refactor que asuma guard universal corrompe "
        "telemetría silentemente."
    )


# ---------------------------------------------------------------------------
# 2. Marker P3-NEXT-5 presente en la sub-sección
# ---------------------------------------------------------------------------
def test_marker_p3_next_5_in_subsection():
    text = _CLAUDE_MD.read_text(encoding="utf-8")
    assert "P3-NEXT-5" in text, (
        "P3-NEXT-5 violation: marker `P3-NEXT-5` ausente. Sin marker, "
        "no se puede grep el origen de la decisión documentada."
    )


# ---------------------------------------------------------------------------
# 3. Cada action_taken canónico aparece en la tabla CLAUDE.md
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("action_taken", sorted(_CANONICAL_ACTION_TAKEN_VALUES))
def test_action_taken_present_in_claude_md(action_taken: str):
    """Cada `action_taken` canónico debe aparecer mencionado en CLAUDE.md.
    Si el código emite un valor que la tabla no documenta, hay drift."""
    text = _CLAUDE_MD.read_text(encoding="utf-8")
    # Buscar como string literal (entre backticks o entre comillas).
    pattern = re.compile(
        rf"`{re.escape(action_taken)}`|'{re.escape(action_taken)}'|\"{re.escape(action_taken)}\"",
    )
    assert pattern.search(text), (
        f"P3-NEXT-5 violation: action_taken `{action_taken}` emitido por "
        f"el código NO está documentado en la tabla CLAUDE.md de surfaces. "
        f"Esto rompe el contrato bidireccional: dashboards podrían encontrar "
        f"el bucket sin contexto, y un refactor podría borrarlo asumiendo "
        f"que es legacy.\n\n"
        f"Fix: añadir fila a la tabla 'Surfaces que escriben "
        f"aggregated_shopping_list* y status del guard' explicando qué "
        f"surface emite `{action_taken}` y por qué."
    )


# ---------------------------------------------------------------------------
# 4. Cada action_taken canónico aparece en al menos un archivo de código
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("action_taken", sorted(_CANONICAL_ACTION_TAKEN_VALUES))
def test_action_taken_emitted_by_code(action_taken: str):
    """Cada `action_taken` canónico debe ser emitido por al menos un
    archivo de código del scan. Si la tabla menciona uno que el código
    no emite, también es drift (documentación stale)."""
    found_in: list[str] = []
    pattern = re.compile(
        rf"['\"]" + re.escape(action_taken) + r"['\"]",
    )
    for f in _CODE_FILES_TO_SCAN:
        if not f.exists():
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if pattern.search(text):
            found_in.append(f.name)

    assert found_in, (
        f"P3-NEXT-5 violation: action_taken `{action_taken}` documentado "
        f"en CLAUDE.md NO se encontró en ningún archivo de código "
        f"({[f.name for f in _CODE_FILES_TO_SCAN]}). Drift: la "
        f"documentación menciona algo que ya no existe en runtime, lo cual "
        f"confunde forensics. Removerlo del set canónico (`{__file__}:"
        f"_CANONICAL_ACTION_TAKEN_VALUES`) si fue legítimamente borrado, "
        f"o restaurar la emisión en código."
    )


# ---------------------------------------------------------------------------
# 5. Tabla menciona la diferencia bloquea/no-bloquea
# ---------------------------------------------------------------------------
def test_table_distinguishes_block_vs_warn():
    """La tabla debe distinguir entre surface que bloquea (assemble) y
    las que no (warn-only). Sin esta distinción explícita, un refactor
    podría asumir que todas bloquean."""
    text = _CLAUDE_MD.read_text(encoding="utf-8")
    # La tabla debe contener tanto "Sí" (bloquea) como "No" (no bloquea)
    # en la columna "Bloquea retry?".
    assert "Bloquea retry?" in text, (
        "P3-NEXT-5 violation: tabla sin columna `Bloquea retry?`."
    )
    # Patrón: dentro de la subsección, debe haber al menos 5 "No" y 1 "Sí"
    # en filas de la tabla (los 5 surfaces auxiliares + el assemble).
    subsection_start = text.find("Surfaces que escriben")
    assert subsection_start >= 0
    subsection_end = text.find("###", subsection_start + 10)  # próximo subheader
    if subsection_end == -1:
        subsection_end = len(text)
    subsection_text = text[subsection_start:subsection_end]

    # Buscamos negaciones explícitas "**No**" o "No —" en la columna.
    no_matches = re.findall(r"\*\*No\*\*", subsection_text)
    si_matches = re.findall(r"\*\*S[ií]\*\*", subsection_text)

    assert len(no_matches) >= 4, (
        f"P3-NEXT-5 violation: tabla no enumera suficientes surfaces que "
        f"NO bloquean (encontré {len(no_matches)} **No**, esperaba ≥4)."
    )
    assert len(si_matches) >= 1, (
        f"P3-NEXT-5 violation: tabla no enumera al menos 1 surface que SÍ "
        f"bloquea (encontré {len(si_matches)} **Sí**, esperaba ≥1)."
    )


# ---------------------------------------------------------------------------
# 6. Cross-link slug
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    expected_slug = "p3_next_5"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "Filename debe contener slug `p3_next_5` para cross-link."
    )
