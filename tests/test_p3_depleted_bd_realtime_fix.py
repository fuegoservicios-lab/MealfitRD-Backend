"""[P3-DEPLETED-BD-REALTIME-FIX · 2026-05-22] + [P3-PANTRY-INVALIDATE-MISMO-TAB · 2026-05-22]
Tests del bundle de fixes al cross-device + mismo-tab sync del P3-DEPLETED-BD.

Bug verificado 2026-05-22 05:52: user dijo "se me acabo la lechosa" al chat.
Backend SÍ persistió (`user_depleted_items` row + `user_inventory` DELETE),
pero el frontend siguió mostrando lechosa en la sección activa de Frutas.

Causas:
  1. **Realtime publication missing**: `user_depleted_items` NO estaba en
     `supabase_realtime` publication. Frontend se suscribía al canal pero
     NUNCA recibía eventos.
  2. **`storage` event no dispara mismo-tab**: si user tiene Pantry montado
     en el mismo tab donde corre el chat (modal/widget/SPA navigation que
     no destruye el componente), `setItem('mealfit_pantry_dirty_at')` no
     triggerea el listener. Solo cross-tab.

Fixes:
  - Migration `p3_realtime_publication_depleted_items_2026_05_22.sql`:
    `ALTER PUBLICATION supabase_realtime ADD TABLE user_depleted_items`.
    Idempotente via DO $$ + NOT EXISTS check.
  - `AgentPage.jsx`: `window.dispatchEvent(new CustomEvent('mealfit:pantry-dirty'))`
    junto al setItem existente.
  - `Pantry.jsx`: `window.addEventListener('mealfit:pantry-dirty', ...)`
    paralelo al listener storage.

Cross-link convention (P2-HIST-AUDIT-14): slug `p3_depleted_bd_realtime_fix`
matchea este archivo.

Tooltip-anchor: P3-DEPLETED-BD-REALTIME-FIX, P3-PANTRY-INVALIDATE-MISMO-TAB.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_AGENT_PAGE_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"
_PANTRY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"
_MIGRATION_ROOT = _REPO_ROOT / "supabase" / "migrations" / "p3_realtime_publication_depleted_items_2026_05_22.sql"
_MIGRATION_BACKEND = _BACKEND_ROOT / "supabase" / "migrations" / "p3_realtime_publication_depleted_items_2026_05_22.sql"


# ===========================================================================
# Sección 1 — Migration de la publication
# ===========================================================================

def test_realtime_migration_exists_in_both_dirs():
    """Migration SSOT presente en ambos dirs (root + backend)."""
    assert _MIGRATION_ROOT.exists(), (
        f"P3-DEPLETED-BD-REALTIME-FIX regresión: migration faltante en "
        f"{_MIGRATION_ROOT}."
    )
    assert _MIGRATION_BACKEND.exists(), (
        f"P3-DEPLETED-BD-REALTIME-FIX regresión: migration faltante en "
        f"{_MIGRATION_BACKEND}. Convención P3-MIGRATIONS-SSOT exige sincronía."
    )


def test_realtime_migrations_are_identical():
    """Ambos archivos idénticos (SSOT sin drift)."""
    if not (_MIGRATION_ROOT.exists() and _MIGRATION_BACKEND.exists()):
        pytest.skip("alguno de los dos archivos no existe")
    assert _MIGRATION_ROOT.read_text(encoding="utf-8") == _MIGRATION_BACKEND.read_text(encoding="utf-8"), (
        "P3-DEPLETED-BD-REALTIME-FIX: SSOT drift root↔backend. Re-sincronizar."
    )


def test_realtime_migration_alters_publication():
    """La migration debe contener `ALTER PUBLICATION supabase_realtime ADD TABLE`."""
    src = _MIGRATION_ROOT.read_text(encoding="utf-8")
    assert "ALTER PUBLICATION supabase_realtime ADD TABLE" in src, (
        "P3-DEPLETED-BD-REALTIME-FIX regresión: migration NO altera la "
        "publication. Sin ALTER PUBLICATION los eventos realtime nunca llegan."
    )
    assert "user_depleted_items" in src, (
        "P3-DEPLETED-BD-REALTIME-FIX regresión: tabla `user_depleted_items` "
        "no mencionada en la migration."
    )


def test_realtime_migration_is_idempotent():
    """`NOT EXISTS` check pre-ALTER + sanity post-apply."""
    src = _MIGRATION_ROOT.read_text(encoding="utf-8")
    assert "NOT EXISTS" in src and "pg_publication_tables" in src, (
        "P3-DEPLETED-BD-REALTIME-FIX regresión: migration sin NOT EXISTS check "
        "pre-ALTER — re-apply en BD donde la tabla ya está → error nativo "
        "de Postgres (ALTER PUBLICATION no es idempotente sin wrap)."
    )
    assert "RAISE EXCEPTION" in src, (
        "P3-DEPLETED-BD-REALTIME-FIX regresión: sanity check post-apply removido."
    )


# ===========================================================================
# Sección 2 — Custom event mismo-tab
# ===========================================================================

def test_agent_page_dispatches_custom_event():
    """AgentPage debe disparar `CustomEvent('mealfit:pantry-dirty')` junto
    al setItem existente — sin esto el listener mismo-tab nunca se entera."""
    src = _AGENT_PAGE_JSX.read_text(encoding="utf-8")
    assert "mealfit:pantry-dirty" in src, (
        "P3-PANTRY-INVALIDATE-MISMO-TAB regresión: AgentPage no dispatcha "
        "el CustomEvent. Sin esto, si user tiene Pantry montado en el "
        "mismo tab durante el chat (modal/widget/SPA), el cache stale se "
        "queda visible."
    )
    assert "dispatchEvent" in src and "CustomEvent" in src, (
        "P3-PANTRY-INVALIDATE-MISMO-TAB regresión: pattern `dispatchEvent + "
        "CustomEvent` removido del AgentPage."
    )


def test_agent_page_dispatch_after_setitem():
    """El dispatch del CustomEvent debe estar CERCA del setItem de
    `mealfit_pantry_dirty_at` (mismo bloque) — ambos son sincrónicos."""
    src = _AGENT_PAGE_JSX.read_text(encoding="utf-8")
    set_idx = src.find("mealfit_pantry_dirty_at")
    custom_idx = src.find("mealfit:pantry-dirty")
    assert set_idx >= 0 and custom_idx >= 0
    distance = abs(custom_idx - set_idx)
    assert distance < 2000, (
        f"P3-PANTRY-INVALIDATE-MISMO-TAB regresión: dispatchEvent y "
        f"setItem están desconectados ({distance} chars). Patrón espera "
        f"ambos en el mismo handler — separación sugiere refactor que "
        f"rompió el pareo."
    )


def test_pantry_listens_custom_event():
    """Pantry.jsx debe agregar listener para `mealfit:pantry-dirty`."""
    src = _PANTRY_JSX.read_text(encoding="utf-8")
    assert "mealfit:pantry-dirty" in src, (
        "P3-PANTRY-INVALIDATE-MISMO-TAB regresión: Pantry.jsx no escucha "
        "el CustomEvent. Sin esto, el dispatch del AgentPage queda huérfano "
        "→ mismo-tab sync no funciona."
    )
    assert re.search(
        r"addEventListener\(\s*['\"]mealfit:pantry-dirty['\"]",
        src,
    ), (
        "P3-PANTRY-INVALIDATE-MISMO-TAB regresión: `addEventListener` con "
        "el CustomEvent ausente en Pantry.jsx."
    )


def test_pantry_cleanup_removes_custom_listener():
    """El listener debe limpiarse en el return del useEffect (memory leak)."""
    src = _PANTRY_JSX.read_text(encoding="utf-8")
    assert re.search(
        r"removeEventListener\(\s*['\"]mealfit:pantry-dirty['\"]",
        src,
    ), (
        "P3-PANTRY-INVALIDATE-MISMO-TAB regresión: Pantry.jsx NO cleanup "
        "el listener `mealfit:pantry-dirty`. Memory leak en mounts/unmounts "
        "del componente."
    )


# ===========================================================================
# Sección 3 — Tooltip anchors presentes
# ===========================================================================

def test_anchor_realtime_fix_present():
    """Anchor `P3-DEPLETED-BD-REALTIME-FIX` presente en migration."""
    src = _MIGRATION_ROOT.read_text(encoding="utf-8")
    assert "P3-DEPLETED-BD-REALTIME-FIX" in src


def test_anchor_mismo_tab_present():
    """Anchor `P3-PANTRY-INVALIDATE-MISMO-TAB` presente en AgentPage + Pantry."""
    agent_src = _AGENT_PAGE_JSX.read_text(encoding="utf-8")
    pantry_src = _PANTRY_JSX.read_text(encoding="utf-8")
    assert "P3-PANTRY-INVALIDATE-MISMO-TAB" in agent_src, (
        "Anchor ausente en AgentPage.jsx."
    )
    assert "P3-PANTRY-INVALIDATE-MISMO-TAB" in pantry_src, (
        "Anchor ausente en Pantry.jsx."
    )
