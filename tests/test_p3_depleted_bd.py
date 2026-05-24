"""[P3-DEPLETED-BD · 2026-05-22] Tests de la migración de `depletedItems` de
localStorage a tabla BD `user_depleted_items` para cross-device sync.

Cierra la limitación verificada 2026-05-22 tras P3-AGENT-DEPLETE:
"cross-device sync limitado: si el user agota desde mobile y mira desde
desktop, no ve el agotado".

Stack:
  - Migration SSOT: `supabase/migrations/p3_user_depleted_items_2026_05_22.sql`
    + `backend/supabase/migrations/p3_user_depleted_items_2026_05_22.sql`.
  - Backend helpers en `db_inventory.py`: `add_depleted_item`,
    `list_depleted_items`, `delete_depleted_item`, `bulk_upsert_depleted_items`.
  - Endpoints REST en `routers/plans.py`: GET/POST `/depleted-items`,
    DELETE `/depleted-items/{item_id}`.
  - Chat agent (`tools.modify_pantry_inventory` con `items_to_deplete`)
    persiste a BD via `add_depleted_item` (en vez de marker JSON localStorage).
  - Frontend Pantry.jsx: fetch desde BD + realtime channel + one-shot
    migration del localStorage existente.

Cross-link convention (P2-HIST-AUDIT-14): slug `p3_depleted_bd` matchea
este archivo.

Tooltip-anchor: P3-DEPLETED-BD.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_DB_INVENTORY_PY = _BACKEND_ROOT / "db_inventory.py"
_TOOLS_PY = _BACKEND_ROOT / "tools.py"
_PLANS_PY = _BACKEND_ROOT / "routers" / "plans.py"
_PANTRY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"
_MIGRATION_ROOT = _REPO_ROOT / "supabase" / "migrations" / "p3_user_depleted_items_2026_05_22.sql"
_MIGRATION_BACKEND = _BACKEND_ROOT / "supabase" / "migrations" / "p3_user_depleted_items_2026_05_22.sql"


# ===========================================================================
# Sección 1 — Migration SSOT existe en ambos dirs y son idénticas
# ===========================================================================

def test_migration_exists_in_root():
    """Migration debe vivir en `supabase/migrations/` (workspace root)."""
    assert _MIGRATION_ROOT.exists(), (
        f"P3-DEPLETED-BD regresión: migration faltante en {_MIGRATION_ROOT}. "
        f"Convención P3-MIGRATIONS-SSOT exige el archivo en este dir."
    )


def test_migration_exists_in_backend():
    """Migration debe vivir TAMBIÉN en `backend/supabase/migrations/`."""
    assert _MIGRATION_BACKEND.exists(), (
        f"P3-DEPLETED-BD regresión: migration faltante en {_MIGRATION_BACKEND}. "
        f"Convención P3-MIGRATIONS-SSOT exige sincronía con root."
    )


def test_migrations_are_identical():
    """Ambos archivos de migration deben tener contenido idéntico (SSOT
    sincronizado). Diferencia indica drift entre repos hermanos."""
    if not (_MIGRATION_ROOT.exists() and _MIGRATION_BACKEND.exists()):
        pytest.skip("alguno de los dos archivos no existe — test de existencia ya falló")
    a = _MIGRATION_ROOT.read_text(encoding="utf-8")
    b = _MIGRATION_BACKEND.read_text(encoding="utf-8")
    assert a == b, (
        "P3-DEPLETED-BD regresión: SSOT drift entre "
        f"`{_MIGRATION_ROOT.name}` (root) y `{_MIGRATION_BACKEND.name}` (backend). "
        "Re-sincronizar con `cp` desde root → backend."
    )


def test_migration_is_idempotent():
    """Migration debe usar `IF NOT EXISTS` + `DROP POLICY IF EXISTS` para
    poder re-aplicarse sin error (convención P3-MIGRATION-IDEMPOTENCE-DOC)."""
    src = _MIGRATION_ROOT.read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS" in src
    assert "CREATE UNIQUE INDEX IF NOT EXISTS" in src or "CREATE INDEX IF NOT EXISTS" in src
    assert "DROP POLICY IF EXISTS" in src, (
        "P3-DEPLETED-BD regresión: migration no es idempotente — falta "
        "`DROP POLICY IF EXISTS` antes de CREATE POLICY (re-apply rompería)."
    )


def test_migration_has_sanity_check():
    """DO $$ RAISE EXCEPTION sanity al final."""
    src = _MIGRATION_ROOT.read_text(encoding="utf-8")
    assert "RAISE EXCEPTION" in src, (
        "P3-DEPLETED-BD regresión: migration sin sanity check post-apply."
    )


# ===========================================================================
# Sección 2 — Backend helpers en db_inventory.py
# ===========================================================================

def test_add_depleted_item_helper_exists():
    src = _DB_INVENTORY_PY.read_text(encoding="utf-8")
    assert re.search(r"^def\s+add_depleted_item\s*\(", src, re.MULTILINE), (
        "P3-DEPLETED-BD regresión: helper `add_depleted_item` removido."
    )


def test_list_depleted_items_helper_exists():
    src = _DB_INVENTORY_PY.read_text(encoding="utf-8")
    assert re.search(r"^def\s+list_depleted_items\s*\(", src, re.MULTILINE), (
        "P3-DEPLETED-BD regresión: helper `list_depleted_items` removido."
    )


def test_delete_depleted_item_helper_exists():
    src = _DB_INVENTORY_PY.read_text(encoding="utf-8")
    assert re.search(r"^def\s+delete_depleted_item\s*\(", src, re.MULTILINE), (
        "P3-DEPLETED-BD regresión: helper `delete_depleted_item` removido."
    )


def test_bulk_upsert_depleted_items_helper_exists():
    """Helper para la migration one-shot desde localStorage en frontend."""
    src = _DB_INVENTORY_PY.read_text(encoding="utf-8")
    assert re.search(r"^def\s+bulk_upsert_depleted_items\s*\(", src, re.MULTILINE), (
        "P3-DEPLETED-BD regresión: helper `bulk_upsert_depleted_items` "
        "removido — Pantry.jsx no puede migrar el localStorage legacy."
    )


def test_helpers_use_user_id_filter():
    """Defense in depth: helpers DEBEN filtrar por user_id explícito (RLS
    es defensa principal pero backend conecta como postgres que la
    bypassea)."""
    src = _DB_INVENTORY_PY.read_text(encoding="utf-8")
    # Extraer las 3 funciones y verificar que filtran.
    for fn_name in ("add_depleted_item", "list_depleted_items", "delete_depleted_item"):
        fn_re = re.compile(rf"def\s+{fn_name}\s*\(.*?(?=\ndef\s|\Z)", re.DOTALL)
        m = fn_re.search(src)
        assert m is not None, f"helper {fn_name} no encontrado"
        body = m.group(0)
        assert "user_id" in body, (
            f"P3-DEPLETED-BD: helper {fn_name} no menciona user_id — "
            f"violación defensa-en-profundidad."
        )


# ===========================================================================
# Sección 3 — tools.modify_pantry_inventory persiste a BD (no marker JSON)
# ===========================================================================

def test_tool_invokes_add_depleted_item():
    """`tools.modify_pantry_inventory` debe invocar `add_depleted_item` para
    persistir en BD (en lugar de solo el marker JSON inline pre-fix)."""
    src = _TOOLS_PY.read_text(encoding="utf-8")
    fn_re = re.compile(
        r"def\s+modify_pantry_inventory\s*\(.*?(?=\n@\w|\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(src)
    assert m is not None
    body = m.group(0)
    assert "add_depleted_item" in body, (
        "P3-DEPLETED-BD regresión: `modify_pantry_inventory` no invoca "
        "`add_depleted_item`. Los items agotados via chat no se persisten "
        "cross-device."
    )


# ===========================================================================
# Sección 4 — Endpoints REST en routers/plans.py
# ===========================================================================

def test_get_depleted_items_endpoint_exists():
    src = _PLANS_PY.read_text(encoding="utf-8")
    assert re.search(r'@router\.get\(\s*"/depleted-items"', src), (
        "P3-DEPLETED-BD regresión: endpoint GET `/depleted-items` removido."
    )


def test_post_depleted_items_endpoint_exists():
    src = _PLANS_PY.read_text(encoding="utf-8")
    assert re.search(r'@router\.post\(\s*"/depleted-items"', src), (
        "P3-DEPLETED-BD regresión: endpoint POST `/depleted-items` removido."
    )


def test_delete_depleted_item_endpoint_exists():
    src = _PLANS_PY.read_text(encoding="utf-8")
    assert re.search(r'@router\.delete\(\s*"/depleted-items/\{item_id\}"', src), (
        "P3-DEPLETED-BD regresión: endpoint DELETE `/depleted-items/{item_id}` "
        "removido — restock pierde el path de cleanup BD."
    )


def test_endpoints_use_get_verified_user_id():
    """Auth: endpoints usan `get_verified_user_id` (no `verify_api_quota` —
    son operaciones cero costo LLM, patrón P1-AUDIT-3)."""
    src = _PLANS_PY.read_text(encoding="utf-8")
    # Buscar la sección de los 3 endpoints depleted.
    idx = src.find('@router.get("/depleted-items")')
    if idx < 0:
        pytest.skip("endpoint GET no encontrado")
    section = src[idx:idx + 4000]
    # Las 3 funciones tienen `Depends(get_verified_user_id)`.
    matches = re.findall(r"Depends\(get_verified_user_id\)", section)
    assert len(matches) >= 3, (
        "P3-DEPLETED-BD regresión: endpoints de `/depleted-items` no usan "
        "`get_verified_user_id` consistentemente. Sin esto, auth queda gappy."
    )


# ===========================================================================
# Sección 5 — Frontend Pantry.jsx fetch desde BD + realtime + migration
# ===========================================================================

def test_pantry_fetches_from_endpoint():
    """`Pantry.jsx` debe fetchear de `/api/plans/depleted-items` (GET)."""
    src = _PANTRY_JSX.read_text(encoding="utf-8")
    assert "/api/plans/depleted-items" in src, (
        "P3-DEPLETED-BD regresión: `Pantry.jsx` no consulta el endpoint "
        "GET `/depleted-items`. State queda dependiente solo de localStorage "
        "(cross-device gap NO se cierra)."
    )


def test_pantry_subscribes_realtime_channel():
    """Realtime channel sobre `user_depleted_items` para sync cross-device."""
    src = _PANTRY_JSX.read_text(encoding="utf-8")
    assert "user_depleted_items" in src, (
        "P3-DEPLETED-BD regresión: `Pantry.jsx` no menciona "
        "`user_depleted_items` (canal realtime). Cross-tab/device sync "
        "regresa al gap pre-fix."
    )


def test_pantry_has_one_shot_migration_flag():
    """Pantry.jsx debe consumir `mealfit_depleted_items_migrated_at` flag
    para que la migration del localStorage corra solo una vez."""
    src = _PANTRY_JSX.read_text(encoding="utf-8")
    assert "mealfit_depleted_items_migrated_at" in src, (
        "P3-DEPLETED-BD regresión: flag `mealfit_depleted_items_migrated_at` "
        "removido. Sin él, la migration POST se ejecutaría en cada mount."
    )


def test_pantry_posts_legacy_to_bd():
    """One-shot migration: POST legacy localStorage al endpoint."""
    src = _PANTRY_JSX.read_text(encoding="utf-8")
    idx = src.find("_runOneShotMigration")
    assert idx >= 0, (
        "P3-DEPLETED-BD regresión: helper `_runOneShotMigration` removido."
    )
    section = src[idx:idx + 3000]
    assert "POST" in section.upper() and "depleted-items" in section, (
        "P3-DEPLETED-BD regresión: migration no hace POST al endpoint."
    )
