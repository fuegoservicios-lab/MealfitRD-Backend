"""P3-RESTOCK-STALE-DEDUP · 2026-05-17 — self-heal del dedup `restocked_items`
cuando `user_inventory` quedó vacío.

Bug reportado por el usuario (plan dfb03329, 2026-05-17): tras borrar todos los
items de la nevera y volver a clicar "Agregar a la Nevera", solo 3 alimentos se
agregaban (en vez de los ~36 esperados) y el PDF de la lista de compras mostraba
"¡Felicidades, Lista Vacía! — 36 ingredientes ya están en tu Nevera" pese a que
la nevera real estaba vacía. Causa raíz:

  1. Pantry.jsx::confirmDeleteAll borra `user_inventory` vía Supabase RLS directo
     (whitelist P3-AUDIT-8) y luego llama `_recalcShoppingListAfterPantryChange(
     {clearRestockedFlag: true})`. El helper hace `delete result.plan_data.is_restocked`
     SOLO sobre el objeto local que viaja a localStorage + setPlanData (Pantry.jsx:372).
     **NO persiste a DB.**

  2. `/recalculate-shopping-list` (routers/plans.py:4982) limpia `is_restocked` +
     `restocked_items` + `restocked_at_iso` SOLO cuando `householdSize` o
     `groceryDuration` cambiaron (`has_changed=True`). confirmDeleteAll NO cambia
     esos params → branch no dispara → DB queda con `is_restocked=true` +
     `restocked_items` con 36-38 entries timestamps recientes.

  3. Siguiente `/restock`: `_existing_restocked` lee 38 entries con `prev_ts < 7d`
     → `_in_cycle(prev_ts) == True` para todas → `skipped_dupes.append; continue`
     filtra ~35 items → solo los 3 items añadidos POST-restock-original (swaps +
     recipe expand) pasan el dedup.

  4. Dashboard.jsx::buildDeltaShoppingList lee `planData?.is_restocked=true` →
     suprime items vía `itemsRemoved++` (Dashboard.jsx:841-843) → muestra "Lista
     Vacía" pese a nevera real con 3 items.

Fix dos capas:

  - Capa 1 (backend self-heal, raíz): /restock cuenta `user_inventory` para
    `user_id`; si 0 + `restocked_items` no vacío, resetea `is_restocked` +
    `restocked_items` + `restocked_at_iso` ANTES de procesar dedup.

  - Capa 2 (frontend defensive): `buildDeltaShoppingList` ignora `is_restocked`
    cuando `inventoryToUse.length < max(3, restocked_items_count * 0.5)`. Cubre
    la ventana entre la mutación stale en DB y el próximo /restock + rutas que
    vacíen pantry skip del helper SSOT.

Verificación DB pre-fix (Supabase MCP, 2026-05-17): plan dfb03329 user
bf6f1383, `plan_data.is_restocked=true`, `restocked_items` 38 entries,
`user_inventory.count=3` → mismatch exacto del bug reportado.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PLANS = (_BACKEND_ROOT / "routers" / "plans.py").read_text(encoding="utf-8")
_DASHBOARD = (
    _BACKEND_ROOT.parent / "frontend" / "src" / "pages" / "Dashboard.jsx"
).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Capa 1 — backend self-heal en /restock
# ---------------------------------------------------------------------------


def test_backend_self_heal_marker_present():
    """Marker tooltip-anchor `P3-RESTOCK-STALE-DEDUP-START` debe vivir en
    routers/plans.py. Sin el anchor, un refactor que renombre el bloque
    rompería la cross-reference de los tests de regresión."""
    assert "P3-RESTOCK-STALE-DEDUP-START" in _PLANS, (
        "Anchor `P3-RESTOCK-STALE-DEDUP-START` ausente en routers/plans.py. "
        "Revierte P3-RESTOCK-STALE-DEDUP."
    )
    assert "P3-RESTOCK-STALE-DEDUP-END" in _PLANS, (
        "Anchor `P3-RESTOCK-STALE-DEDUP-END` ausente. El bloque debe cerrarse "
        "explícitamente para anclar la región de código."
    )


def test_backend_self_heal_lives_inside_api_restock():
    """El bloque self-heal debe vivir DENTRO de `def api_restock` (no flotando
    en un helper desconectado). Un parser-based asserta esta colocación."""
    api_restock_start = _PLANS.find("def api_restock(")
    assert api_restock_start != -1, "`def api_restock(` no encontrado."
    # El siguiente `def ` top-level (línea-start) marca el final del handler.
    rest = _PLANS[api_restock_start + 1 :]
    next_def = re.search(r"\n@router\.", rest)
    end_idx = (api_restock_start + 1 + next_def.start()) if next_def else len(_PLANS)
    body = _PLANS[api_restock_start:end_idx]
    assert "P3-RESTOCK-STALE-DEDUP-START" in body, (
        "El bloque self-heal NO está dentro de `api_restock`. Movido fuera del "
        "handler → no se ejecuta en el path real del bug."
    )


def test_backend_self_heal_resets_three_keys():
    """El self-heal DEBE popear las 3 keys obsoletas: `is_restocked`,
    `restocked_items`, `restocked_at_iso`. Olvidar `restocked_at_iso` reintroduce
    bug por la rama legacy `_build_hybrid_shopping_list` que filtra perecederos
    por timestamp blanket (RIESGO-1)."""
    anchor = _PLANS.find("P3-RESTOCK-STALE-DEDUP-START")
    end = _PLANS.find("P3-RESTOCK-STALE-DEDUP-END", anchor)
    assert anchor != -1 and end != -1, "Anchors del bloque self-heal no encontrados."
    block = _PLANS[anchor:end]
    for key in ("is_restocked", "restocked_items", "restocked_at_iso"):
        assert f'plan_data.pop("{key}"' in block, (
            f"`plan_data.pop({key!r}, ...)` ausente en el self-heal. "
            f"Si solo popea un subset, el dedup del próximo /restock seguirá "
            f"siendo parcialmente stale."
        )


def test_backend_self_heal_gated_on_empty_inventory():
    """El reset SOLO debe disparar cuando `user_inventory` está vacío. Si
    dispara con inventario no-vacío, borraría dedup legítimo de un restock
    parcial reciente (ej: usuario compró fresas el lunes, pollo el jueves).

    [P3-RESTOCK-STALE-DEDUP-V2 · 2026-05-17] Refactor de supabase-py
    `count="exact"` → SQL raw via `execute_sql_query` (helper canónico del
    repo) por confiabilidad: el cliente python a veces retornaba count=None
    bajo rate-limit/proxy, dejando el self-heal silenciado. Raw SQL retorna
    int directo desde la fila result."""
    anchor = _PLANS.find("P3-RESTOCK-STALE-DEDUP-START")
    end = _PLANS.find("P3-RESTOCK-STALE-DEDUP-END", anchor)
    block = _PLANS[anchor:end]
    # Cuenta exacta sobre user_inventory para el user_id autenticado via SQL raw.
    assert "execute_sql_query" in block, (
        "Self-heal no usa `execute_sql_query` (helper canónico). Revierte el "
        "refactor V2 — vuelve a tener supabase-py `count='exact'` que puede "
        "retornar None silenciosamente."
    )
    assert re.search(
        r"FROM\s+user_inventory\s+WHERE\s+user_id\s*=\s*%s",
        block,
        re.IGNORECASE,
    ), (
        "SELECT count del self-heal no filtra `WHERE user_id = %s`. Sin filtro, "
        "user A con inventario vacío y user B sin items aún podrían interferir."
    )
    assert "_inv_count == 0" in block, (
        "Gate `== 0` ausente. El reset debe condicionarse a inventario "
        "completamente vacío, no a `<= N`."
    )


def test_backend_self_heal_idempotent_skip_when_no_dedup():
    """Si `_existing_restocked` está vacío (restock inicial o post-reset), el
    self-heal NO debe disparar la query — es un no-op de paso, sin costo DB."""
    anchor = _PLANS.find("P3-RESTOCK-STALE-DEDUP-START")
    end = _PLANS.find("P3-RESTOCK-STALE-DEDUP-END", anchor)
    block = _PLANS[anchor:end]
    # El gate `if user_id and _existing_restocked:` evita la query.
    # (Pre-V2 incluía `supabase and`; tras refactor a execute_sql_query
    # ya no se necesita la referencia global del cliente.)
    assert re.search(
        r"if\s+user_id\s+and\s+_existing_restocked\s*:",
        block,
    ), (
        "Gate `if user_id and _existing_restocked` ausente. Sin él, cada "
        "/restock haría una query extra a user_inventory aunque no haya dedup "
        "que limpiar."
    )


def test_backend_self_heal_swallows_exception_continues_flow():
    """Si el count check falla (DB transient), el self-heal NO debe abortar el
    /restock. Loggea WARN + continua con el dedup actual (posible undercount,
    pero el restock sigue funcional). Patrón consistente con el `try/except`
    legacy del bloque de marcado post-restock (líneas 4324-4325)."""
    anchor = _PLANS.find("P3-RESTOCK-STALE-DEDUP-START")
    end = _PLANS.find("P3-RESTOCK-STALE-DEDUP-END", anchor)
    block = _PLANS[anchor:end]
    assert "except Exception" in block, (
        "El self-heal no tiene `except Exception`. Una DB hiccup tumbaría el "
        "restock entero — defense-in-depth requiere swallow + WARN."
    )
    assert "logger.warning" in block, (
        "Sin `logger.warning` en el except, una excepción del count check "
        "queda silenciosa en logs — operador no puede correlacionar."
    )


# ---------------------------------------------------------------------------
# Capa 2 — frontend defensive en buildDeltaShoppingList
# ---------------------------------------------------------------------------


def test_frontend_defensive_guard_marker_present():
    """Marker `P3-RESTOCK-STALE-DEDUP` debe vivir en Dashboard.jsx vinculando
    el guard al P-fix. Sin el marker el cross-link slug→test del enforcer
    P2-HIST-AUDIT-14 sigue funcionando solo gracias al backend; el frontend
    debe declararse ancla de forma independiente."""
    assert "P3-RESTOCK-STALE-DEDUP" in _DASHBOARD, (
        "Marker `P3-RESTOCK-STALE-DEDUP` ausente en Dashboard.jsx. "
        "Sin el marker un futuro refactor podría borrar el guard creyendo "
        "que es dead code."
    )


def test_frontend_computes_restocked_items_count():
    """El guard debe leer el tamaño de `planData.restocked_items` para
    comparar contra `inventoryToUse.length`. Sin esa cuenta, no hay forma
    de detectar el mismatch."""
    # Aceptamos cualquier nombre de variable razonable pero exigimos que el
    # acceso a `restocked_items` exista en el contexto del guard.
    assert re.search(
        r"planData\?\.restocked_items",
        _DASHBOARD,
    ), (
        "`planData?.restocked_items` no referenciado en Dashboard.jsx. El "
        "guard no puede detectar drift sin contar las entries del dict."
    )
    # Y debe usar Object.keys(...).length para contar (no `.length` directo,
    # que retornaría undefined sobre un object).
    assert "Object.keys(" in _DASHBOARD, (
        "`Object.keys(restocked_items).length` ausente. `restocked_items` es "
        "un dict, no un array — `.length` directo sería undefined."
    )


def test_frontend_isPostRestockRotation_respects_stale_flag():
    """`isPostRestockRotation` debe consultar `_staleDedup` (o equivalente) y
    forzar `false` cuando el flag es stale. Sin esto, el guard cuenta pero no
    cierra el bug."""
    # Buscamos la asignación de `isPostRestockRotation`. Debe NO ser el `!!planData?.is_restocked`
    # crudo de pre-fix (que era exactamente esa expresión sola).
    m = re.search(
        r"const\s+isPostRestockRotation\s*=\s*([^;\n]+);",
        _DASHBOARD,
    )
    assert m, "`const isPostRestockRotation` no encontrada en Dashboard.jsx."
    rhs = m.group(1).strip()
    assert rhs != "!!planData?.is_restocked", (
        f"`isPostRestockRotation` revertida al RHS crudo `!!planData?.is_restocked`. "
        f"Sin combinar con stale-check el guard no aplica."
    )
    # La RHS final debe incluir negación del stale-dedup (referencia textual).
    assert "_staleDedup" in rhs or "staleDedup" in rhs, (
        f"`isPostRestockRotation` no combina con `_staleDedup` flag. RHS "
        f"actual: {rhs}. Reintroduciría el bug de PDF 'Lista Vacía' cuando "
        f"DB tiene flag stale + nevera real vacía."
    )


def test_frontend_threshold_uses_half_restocked_count_with_floor_3():
    """La heurística del threshold debe ser:
       `inventory_count < max(3, floor(restocked_count * 0.5))`.
    El floor 3 evita un edge case donde restocked_items tiene 1-2 entries
    (plan minimal) y el threshold sería 0-1, nunca disparando. El 0.5
    asume que vaciar >50% de una nevera marcada como restocked es señal
    inequívoca de drift."""
    assert "Math.max(3" in _DASHBOARD and "Math.floor(" in _DASHBOARD and "0.5" in _DASHBOARD, (
        "Threshold del guard no usa `Math.max(3, Math.floor(_restockedCount * 0.5))`. "
        "Cambiarlo a >50% reduce false-positives pero también la cobertura del bug; "
        "<50% (ej: 30%) sería demasiado agresivo y borraría rotation legítima."
    )


# ---------------------------------------------------------------------------
# Cross-link sanity
# ---------------------------------------------------------------------------


def test_frontend_capa3_removed_for_fluidity():
    """Capa 3 (workaround pre-flight con 2 POSTs a /recalculate-shopping-list)
    fue REMOVIDA por fluidez: añadía ~600-1000ms extra antes del /restock real.
    Mientras Capa 1 backend self-heal esté activa, el workaround es peso muerto.

    El marker `P3-RESTOCK-STALE-DEDUP-CAPA3-REMOVED` ancla la decisión y
    previene reintroducción accidental. Si Capa 1 backend regresa/falla,
    restaurar Capa 3 desde el git log (mensaje original con el toggle
    householdSize)."""
    # Anti-regresión: el código Capa 3 original (toggle householdSize +
    # 2 POSTs a recalculate-shopping-list) NO debe estar dentro de handleRestock.
    handle_idx = _DASHBOARD.find("const handleRestock")
    assert handle_idx != -1, "`const handleRestock` no encontrado."
    restock_post_idx = _DASHBOARD.find("/api/plans/restock", handle_idx)
    assert restock_post_idx > handle_idx, "POST /api/plans/restock no encontrado."
    handler_body = _DASHBOARD[handle_idx:restock_post_idx]
    # Marker removed presente
    assert "P3-RESTOCK-STALE-DEDUP-CAPA3-REMOVED" in handler_body, (
        "Marker `P3-RESTOCK-STALE-DEDUP-CAPA3-REMOVED` ausente. Sin él, un "
        "futuro refactor podría re-introducir el workaround creyendo que es "
        "necesario."
    )
    # Anti-regresión: no debe haber un POST a recalculate-shopping-list ANTES
    # del POST a /restock dentro del handler (Capa 3 lo hacía 2 veces).
    assert "/api/plans/recalculate-shopping-list" not in handler_body, (
        "Encontrado POST a `/api/plans/recalculate-shopping-list` DENTRO de "
        "handleRestock antes del POST a /restock. Capa 3 fue removida por "
        "fluidez — si necesitas restaurar, actualiza este test al patrón "
        "original."
    )
    # `_staleDedupDetected` era la variable señal de Capa 3 — no debe vivir
    # en handleRestock tras la remoción.
    assert "_staleDedupDetected" not in handler_body, (
        "Variable `_staleDedupDetected` (Capa 3) sigue en handleRestock. "
        "Remoción incompleta."
    )


def test_marker_slug_matches_test_filename():
    """Cross-link enforcer P2-HIST-AUDIT-14: el slug del marker
    `P3-RESTOCK-STALE-DEDUP` → `p3_restock_stale_dedup` debe matchear el
    nombre de este archivo. Si renombras este test, también re-anchorea el
    marker — el enforcer falla si pierdes este link."""
    expected_slug = "p3_restock_stale_dedup"
    this_file = Path(__file__).stem
    assert this_file.startswith(f"test_{expected_slug}"), (
        f"Filename `{this_file}` no matchea slug esperado "
        f"`test_{expected_slug}*` del marker `P3-RESTOCK-STALE-DEDUP`."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
