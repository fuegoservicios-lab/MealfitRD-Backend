"""[P0-RESTOCK-DEDUP-NAME · 2026-05-20] Fix del bug "27 items en lista de
compras → 24 items en la Nevera tras restock".

ROOT CAUSE

  `restock_inventory` ([backend/db_inventory.py:1487]) llamaba a
  `normalize_name()` en la ruta estructurada para cada item del shopping
  list. `normalize_name()` ([backend/shopping_calculator.py:973]) aplica
  4 niveles de alias matching contra `master_ingredients` + un
  **semantic fallback** vía embeddings Gemini con umbral 0.70.

  Items sin alias match exacto (típicamente regional/brand-specific:
  `Maní`, etc.) caían al semantic fallback que puede mapearlos al
  canonical de OTRO item de la misma lista (e.g. `Maní` → `Almendras
  fileteadas` por similitud "fruto seco"). Resultado:
    - Colisión silenciosa: 2 items distintos → 1 row en user_inventory.
    - O UNIQUE violation: si normalize colapsa name+unit idénticos.

  Producción verificada (user angelobrito500@gmail.com, plan 713ff43a-...
  el 2026-05-20 01:06 UTC): shopping list de 27 items → 24 rows en
  user_inventory + restocked_items marcado con los 27 names aspiracionales
  (ledger drift).

FIX

  Ruta estructurada de `restock_inventory` PRESERVA el name raw del
  shopping list (NO segunda canonicalización via normalize_name). El
  aggregator ya canonicalizó al producir el shopping list — re-canonicalizar
  via semantic embedding es destructivo.

  Retorno cambiado de `bool` a `(success: bool, persisted_names: List[str])`
  para que `api_restock` marque `restocked_items` SOLO con names que
  efectivamente persistieron (no aspiracionales). Cierra el ledger drift
  colateral.

INVARIANTES VALIDADAS POR ESTE TEST

  1. `restock_inventory` retorna tupla (bool, list) — no bool legacy.
  2. La ruta estructurada NO importa `normalize_name` desde
     `shopping_calculator`.
  3. La ruta estructurada NO llama `normalize_name(item["name"])`.
  4. `add_or_update_inventory_item` se invoca con el name exacto del
     shopping list (preserve identity).
  5. `api_restock` usa `persisted_names` para anotar `restocked_items`
     en lugar de iterar `filtered_ingredients` aspiracional.
  6. Tooltip-anchor `P0-RESTOCK-DEDUP-NAME` presente en código.
  7. Los 3 callsites legacy de `restock_inventory` están actualizados
     para manejar la tupla.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DB_INVENTORY_FP = _REPO_ROOT / "backend" / "db_inventory.py"
_ROUTERS_PLANS_FP = _REPO_ROOT / "backend" / "routers" / "plans.py"
_TOOLS_FP = _REPO_ROOT / "backend" / "tools.py"


@pytest.fixture(scope="module")
def db_inv_src() -> str:
    return _DB_INVENTORY_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _ROUTERS_PLANS_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tools_src() -> str:
    return _TOOLS_FP.read_text(encoding="utf-8")


# ===========================================================================
# Fix #1 — restock_inventory preserva name raw + retorna tupla
# ===========================================================================

def _find_restock_inventory_body(src: str) -> str:
    """Aísla el cuerpo de `def restock_inventory(...)` hasta la próxima
    `def` top-level."""
    fn_idx = src.find("def restock_inventory(")
    assert fn_idx >= 0, "restock_inventory no encontrado en db_inventory.py"
    next_def = re.search(r"\ndef\s", src[fn_idx + 10:])
    end = (fn_idx + 10 + next_def.start()) if next_def else len(src)
    return src[fn_idx:end]


def test_restock_does_not_import_normalize_name_top_level(db_inv_src: str) -> None:
    """[P0-RESTOCK-DEDUP-NAME] el import `from shopping_calculator import
    normalize_name` legacy DEBE estar eliminado del body de
    `restock_inventory`. Si se restaura, regresa el bug semántico."""
    body = _find_restock_inventory_body(db_inv_src)
    assert "from shopping_calculator import normalize_name" not in body, (
        "[P0-RESTOCK-DEDUP-NAME] `from shopping_calculator import normalize_name` "
        "NO debe estar dentro del body de `restock_inventory`. El fix preserva "
        "el name raw del shopping list — re-canonicalizar via semantic embedding "
        "colapsa items distintos al mismo canonical (bug 27→24)."
    )


def test_restock_structured_path_uses_raw_name(db_inv_src: str) -> None:
    """[P0-RESTOCK-DEDUP-NAME] la ruta estructurada extrae `raw_name =
    str(item["name"]).strip()` SIN pasar por normalize_name."""
    body = _find_restock_inventory_body(db_inv_src)
    # Patrón canónico: raw_name = str(item["name"]).strip()
    assert re.search(
        r'raw_name\s*=\s*str\(item\[["\']name["\']\]\)\.strip\(\)',
        body,
    ), (
        "[P0-RESTOCK-DEDUP-NAME] la ruta estructurada debe definir "
        "`raw_name = str(item[\"name\"]).strip()` y usarlo como `name`. "
        "NO debe llamar `normalize_name(item[\"name\"])`."
    )
    # NO debe haber `normalize_name(item["name"])` en el body.
    assert not re.search(r'normalize_name\(item\[["\']name["\']\]\)', body), (
        "[P0-RESTOCK-DEDUP-NAME] `normalize_name(item[\"name\"])` debe estar "
        "REMOVIDO de la ruta estructurada — preserva el name raw."
    )


def test_restock_returns_tuple_signature(db_inv_src: str) -> None:
    """[P0-RESTOCK-DEDUP-NAME] retorno cambiado de `bool` a
    `(bool, List[str])`. Permite que el caller anote `restocked_items`
    SOLO con names que persistieron exitosamente."""
    body = _find_restock_inventory_body(db_inv_src)
    # Early-out empty list debe retornar tupla.
    assert re.search(r"return\s+False,\s*\[\]", body), (
        "[P0-RESTOCK-DEDUP-NAME] early-out cuando `ingredients_list` es vacío "
        "debe retornar `False, []` (tupla), no `False` solo."
    )
    # `persisted_names` tracking list debe existir.
    assert "persisted_names" in body, (
        "[P0-RESTOCK-DEDUP-NAME] la función debe trackear `persisted_names` "
        "(lista de items que efectivamente llegaron a DB)."
    )
    # Return final: `return success, persisted_names` (sin solo `success`).
    assert re.search(r"return\s+success,\s*persisted_names", body), (
        "[P0-RESTOCK-DEDUP-NAME] return final debe ser "
        "`return success, persisted_names` (tupla)."
    )


def test_persisted_names_appended_only_on_success(db_inv_src: str) -> None:
    """[P0-RESTOCK-DEDUP-NAME] `persisted_names.append(name)` debe vivir
    SOLO dentro del branch `if res:` (i.e., cuando
    add_or_update_inventory_item retornó truthy). NO debe append a la lista
    en path de error."""
    body = _find_restock_inventory_body(db_inv_src)
    # Debe haber EXACTAMENTE 2 appends (uno por ruta estructurada + uno por legacy).
    append_count = len(re.findall(r"persisted_names\.append\(name\)", body))
    assert append_count == 2, (
        f"[P0-RESTOCK-DEDUP-NAME] esperaba 2 callsites de "
        f"`persisted_names.append(name)` (ruta estructurada + ruta legacy). "
        f"Encontradas: {append_count}."
    )


# ===========================================================================
# Fix #2 — api_restock usa persisted_names para anotar restocked_items
# ===========================================================================

def test_api_restock_unpacks_tuple_from_restock_inventory(plans_src: str) -> None:
    """[P0-RESTOCK-DEDUP-NAME] el endpoint debe unpack la tupla retornada
    por `restock_inventory`."""
    fn_idx = plans_src.find("def api_restock(")
    assert fn_idx >= 0
    next_def = re.search(r"\n(?:@router|def\s)", plans_src[fn_idx + 10:])
    end = (fn_idx + 10 + next_def.start()) if next_def else len(plans_src)
    body = plans_src[fn_idx:end]

    # Debe asignar el resultado a `_restock_res` o similar y unpackear.
    assert re.search(
        r"_restock_res\s*=\s*restock_inventory\(",
        body,
    ), (
        "[P0-RESTOCK-DEDUP-NAME] `api_restock` debe capturar el retorno como "
        "`_restock_res = restock_inventory(...)` para luego unpackear (tupla "
        "o bool legacy)."
    )
    # `persisted_names` debe estar disponible en scope.
    assert "persisted_names" in body, (
        "[P0-RESTOCK-DEDUP-NAME] `api_restock` debe extraer `persisted_names` "
        "del retorno de `restock_inventory` para anotar `restocked_items` "
        "selectivamente."
    )


def test_api_restock_marks_restocked_items_with_persisted_names(plans_src: str) -> None:
    """[P0-RESTOCK-DEDUP-NAME] `restocked_items` debe anotarse con
    `persisted_names` (preferido) y caer a `filtered_ingredients` solo
    como fallback para callers legacy."""
    fn_idx = plans_src.find("def api_restock(")
    next_def = re.search(r"\n(?:@router|def\s)", plans_src[fn_idx + 10:])
    end = (fn_idx + 10 + next_def.start()) if next_def else len(plans_src)
    body = plans_src[fn_idx:end]
    # Patrón canónico: _names_to_mark = persisted_names if persisted_names else [...]
    assert "_names_to_mark" in body, (
        "[P0-RESTOCK-DEDUP-NAME] usar una var local `_names_to_mark` que "
        "expresa la decisión: persisted_names cuando hay, fallback a "
        "filtered_ingredients si vacío."
    )
    assert re.search(
        r"_names_to_mark\s*=\s*\(\s*\n?\s*persisted_names",
        body,
    ), (
        "[P0-RESTOCK-DEDUP-NAME] `_names_to_mark = (persisted_names if "
        "persisted_names else [...])` debe estar presente."
    )


def test_api_restock_response_includes_counts(plans_src: str) -> None:
    """[P0-RESTOCK-DEDUP-NAME] la response retorna `persisted_count` +
    `requested_count` para que el frontend pueda detectar el delta y mostrar
    al user si hubo merge silencioso."""
    fn_idx = plans_src.find("def api_restock(")
    next_def = re.search(r"\n(?:@router|def\s)", plans_src[fn_idx + 10:])
    end = (fn_idx + 10 + next_def.start()) if next_def else len(plans_src)
    body = plans_src[fn_idx:end]
    assert '"persisted_count"' in body, (
        "[P0-RESTOCK-DEDUP-NAME] response debe incluir `persisted_count` "
        "(items que llegaron a DB). Sin esto, el frontend no detecta "
        "merges silenciosos."
    )
    assert '"requested_count"' in body


# ===========================================================================
# Callsites legacy actualizados
# ===========================================================================

def test_replace_shopping_list_handles_tuple_return(db_inv_src: str) -> None:
    """[P0-RESTOCK-DEDUP-NAME] `replace_shopping_list_only_items` consume
    `restock_inventory` y maneja el tuple return (success only — no
    necesita persisted_names para su flujo de DELETE+INSERT rollback)."""
    fn_idx = db_inv_src.find("def replace_shopping_list_only_items(")
    assert fn_idx >= 0
    next_def = re.search(r"\ndef\s", db_inv_src[fn_idx + 10:])
    end = (fn_idx + 10 + next_def.start()) if next_def else len(db_inv_src)
    body = db_inv_src[fn_idx:end]
    # Debe extraer success del tuple, no asignar tupla cruda a `res`.
    assert re.search(
        r"res_tuple\s*=\s*restock_inventory\(",
        body,
    ), (
        "[P0-RESTOCK-DEDUP-NAME] `replace_shopping_list_only_items` debe "
        "capturar el retorno como `res_tuple = restock_inventory(...)` y "
        "extraer `success` con `isinstance(res_tuple, tuple)` check."
    )


def test_agent_tool_handles_tuple_return(tools_src: str) -> None:
    """[P0-RESTOCK-DEDUP-NAME] El agent tool `mark_shopping_list_purchased`
    también maneja el tuple return."""
    # Buscar la invocación dentro del tool.
    invocation = re.search(
        r"_restock_res\s*=\s*restock_inventory\(",
        tools_src,
    )
    assert invocation, (
        "[P0-RESTOCK-DEDUP-NAME] el tool `mark_shopping_list_purchased` debe "
        "capturar el retorno como `_restock_res = restock_inventory(...)` y "
        "extraer `success` con isinstance tuple check (igual que api_restock "
        "y replace_shopping_list_only_items)."
    )


# ===========================================================================
# Tooltip-anchors preservados
# ===========================================================================

def test_tooltip_anchor_present(db_inv_src: str, plans_src: str, tools_src: str) -> None:
    """[P0-RESTOCK-DEDUP-NAME] marker textual en los 3 archivos modificados."""
    assert db_inv_src.count("P0-RESTOCK-DEDUP-NAME") >= 4, (
        "marker en db_inventory.py (docstring restock_inventory + 2-3 "
        "inline comments del fix)."
    )
    assert plans_src.count("P0-RESTOCK-DEDUP-NAME") >= 2, (
        "marker en routers/plans.py (intro + comment en bloque restocked_items)."
    )
    assert tools_src.count("P0-RESTOCK-DEDUP-NAME") >= 1, (
        "marker en tools.py (comment en mark_shopping_list_purchased)."
    )


# ===========================================================================
# Tests funcionales (skipped si supabase no instalado)
# ===========================================================================

@pytest.fixture
def db_inv_module():
    """Importa db_inventory si las deps están disponibles."""
    pytest.importorskip("supabase")
    import sys
    backend_dir = str(_REPO_ROOT / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    import db_inventory as mod  # type: ignore
    return mod


def test_restock_empty_input_returns_tuple_false_empty(db_inv_module) -> None:
    """[P0-RESTOCK-DEDUP-NAME] `restock_inventory(user, [])` retorna
    `(False, [])` — tupla, no `False` solo. Backward compat: callers
    legacy que hagan `if not result:` siguen viendo falsy (tupla con
    primer elemento False)."""
    result = db_inv_module.restock_inventory("u1", [])
    assert isinstance(result, tuple)
    assert result == (False, [])
    # Backward compat truthy check.
    assert not result[0]


def test_persisted_names_preserves_raw_input(db_inv_module, monkeypatch) -> None:
    """[P0-RESTOCK-DEDUP-NAME] cuando hay éxito, `persisted_names`
    contiene los names EXACTOS del input (sin normalize_name canonicalización).
    Stub `add_or_update_inventory_item` para aislar el test del DB real."""
    captured_names = []

    def _stub_add(user_id, name, qty, unit, **kwargs):
        captured_names.append(name)
        return True

    monkeypatch.setattr(db_inv_module, "add_or_update_inventory_item", _stub_add)
    # [P1-NEON-DB-MIGRATION · 2026-06-12] El guard de disponibilidad pasó de
    # `if not supabase:` a `_db_available()` que lee `db_core.connection_pool`
    # lazy — patchear el pool a truthy reemplaza el viejo patch de `supabase`.
    import db_core
    monkeypatch.setattr(db_core, "connection_pool", object())  # truthy non-None

    success, persisted = db_inv_module.restock_inventory("u1", [
        {"name": "Maní", "quantity": 0.25, "unit": "lb"},  # sin master match → caería al semantic fallback PRE-FIX
        {"name": "Almendras fileteadas", "quantity": 1, "unit": "paquete"},  # canonical match
    ])

    assert success is True
    assert persisted == ["Maní", "Almendras fileteadas"], (
        f"[P0-RESTOCK-DEDUP-NAME] esperaba que `persisted_names` preservara "
        f"los names RAW del input. Recibido: {persisted}. Si esta lista "
        f"contiene 'Almendras fileteadas' para AMBOS items, el bug del "
        f"semantic fallback regresó."
    )
    # Los add_or_update_inventory_item calls reciben names raw.
    assert captured_names == ["Maní", "Almendras fileteadas"]
