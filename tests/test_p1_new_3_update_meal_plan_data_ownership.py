"""[P1-NEW-3 · 2026-05-10] `update_meal_plan_data` debe aceptar `user_id`
opcional y filtrar el UPDATE por `(id, user_id)` cuando se provee.

Bug original (audit 2026-05-10):
    El helper hacía `UPDATE meal_plans SET plan_data = %s WHERE id = %s`
    sin ownership check. Delegaba 100% al caller la validación de
    ownership. Un callsite futuro que olvidara el check (como ocurrió
    originalmente en `/restock`, cerrado en P0-NEW-1) re-introducía
    IDOR sobre `meal_plans`.

Fix:
    1. Signature `update_meal_plan_data(plan_id, new_plan_data, user_id=None)`.
    2. Si `user_id` se pasa: WHERE filtra `id AND user_id` (defense-in-depth
       a DB-level — mismo patrón que P0-HIST-IDOR-1/2 y P0-NEW-1).
    3. Si `user_id` es None: comportamiento legacy + warning de log
       (DEPRECATED, fuerza migración).
    4. Migrados 4 callsites de producción:
       - `proactive_agent.py:464` (JIT week2 background).
       - `routers/plans.py:2985` (recipe/expand).
       - `routers/plans.py:3451` (recalculate-shopping-list).
       - `tools.py:523` (modify_single_meal).

Estrategia del test (parser estático sobre db_plans.py + los 4 callsites):
    1. Verificar signature con `user_id` opcional.
    2. Verificar branch `if user_id is None: legacy_warning + legacy_path`.
    3. Verificar branch ownership con `WHERE id = %s AND user_id = %s`.
    4. [P1-NEON-DB-MIGRATION · 2026-06-12] Verificar que NO existe un path
       PostgREST (`.table(`/`.eq(`) en el helper — el fallback supabase-py
       fue eliminado fail-loud (no soporta advisory locks I7 y post-Neon
       apuntaría a la DB equivocada).
    5. Verificar que LOS 4 callsites de producción pasan `user_id=`.

Drift detection:
    - Si un callsite revierte a `update_meal_plan_data(plan_id, plan_data)`
      sin user_id → falla `test_all_production_callsites_pass_user_id`.
    - Si la signature pierde el param user_id → falla
      `test_signature_accepts_user_id`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DB_PLANS_PY = _REPO_ROOT / "backend" / "db_plans.py"
_PROACTIVE_PY = _REPO_ROOT / "backend" / "proactive_agent.py"
_PLANS_ROUTER_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"
_TOOLS_PY = _REPO_ROOT / "backend" / "tools.py"


@pytest.fixture(scope="module")
def db_plans_src() -> str:
    return _DB_PLANS_PY.read_text(encoding="utf-8")


def _extract_function_body(src: str, fn_name: str) -> str:
    """Extrae cuerpo de `def <fn_name>(` hasta el siguiente top-level def."""
    m = re.search(rf"def\s+{re.escape(fn_name)}\s*\(", src)
    assert m, f"No se encontró `def {fn_name}(` en el archivo."
    start = m.start()
    next_def = re.search(r"\n(?:def\s|@app\.|@router\.)", src[start + 1:])
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


def test_signature_accepts_user_id(db_plans_src: str):
    """`update_meal_plan_data(plan_id, new_plan_data, user_id=None)` debe
    aceptar `user_id` como kwarg opcional con default `None`."""
    # [stale-parser fix] La signature usa `user_id: Optional[str] = None`
    # (tipado correcto para un kwarg opcional); el regex acepta tanto el
    # literal `str` como `Optional[str]`. El contrato (param opcional con
    # default None) se mantiene intacto.
    pattern = re.compile(
        r"def\s+update_meal_plan_data\s*\(\s*plan_id\s*:\s*str\s*,\s*"
        r"new_plan_data\s*:\s*dict\s*,\s*user_id\s*:\s*"
        r"(?:Optional\[\s*str\s*\]|str)\s*=\s*None\s*\)\s*:",
    )
    assert pattern.search(db_plans_src), (
        "P1-NEW-3 regresión: la signature de `update_meal_plan_data` "
        "perdió el param `user_id: str = None`. Sin él, los callsites "
        "no pueden pasar user_id y se pierde el defense-in-depth a DB-level."
    )


def test_legacy_path_warns_on_none_user_id(db_plans_src: str):
    """La rama `if user_id is None:` debe emitir un warning de log
    señalando que es DEPRECATED — sin esto, los callsites nunca migran.
    """
    body = _extract_function_body(db_plans_src, "update_meal_plan_data")
    pattern = re.compile(
        r"if\s+user_id\s+is\s+None\s*:\s*[\r\n]+\s*logger\.warning",
    )
    assert pattern.search(body), (
        "P1-NEW-3 regresión: la rama `if user_id is None:` ya no emite "
        "warning. Sin él, los callsites legacy persisten sin presión a "
        "migrar y el defense-in-depth queda parcial."
    )


def test_ownership_branch_uses_user_id_in_where(db_plans_src: str):
    """La rama nueva (user_id provisto) debe ejecutar UPDATE con
    `WHERE id = %s AND user_id = %s` en el path psycopg (único path
    tras P1-NEON-DB-MIGRATION — el fallback supabase-py fue eliminado).
    """
    body = _extract_function_body(db_plans_src, "update_meal_plan_data")

    psycopg_pattern = re.compile(
        r"WHERE\s+id\s*=\s*%s\s+AND\s+user_id\s*=\s*%s",
    )
    assert psycopg_pattern.search(body), (
        "P1-NEW-3 regresión: el path psycopg no filtra por "
        "`AND user_id = %s` en el WHERE. IDOR re-abierto a DB-level."
    )

    # El UPDATE con ownership debe bindear (plan_id, user_id) — el WHERE
    # de arriba sin estos params sería un literal muerto.
    params_pattern = re.compile(
        r"\(\s*Jsonb\(new_plan_data\)\s*,\s*plan_id\s*,\s*user_id\s*,?\s*\)",
    )
    assert params_pattern.search(body), (
        "P1-NEW-3 regresión: el execute del path ownership no bindea "
        "`(Jsonb(new_plan_data), plan_id, user_id)`. El filtro user_id "
        "quedó sin parámetro — IDOR re-abierto a DB-level."
    )


def test_no_postgrest_fallback_path(db_plans_src: str):
    """[P1-NEON-DB-MIGRATION · 2026-06-12] El fallback PostgREST
    (`supabase.table("meal_plans").update(...).eq(...)`) fue eliminado
    fail-loud del helper: PostgREST no soporta advisory locks (violaba
    I7) y post-Neon apuntaría a la DB equivocada. Si alguien lo
    re-introduce, este test falla antes de que el lost-update vuelva.
    """
    body = _extract_function_body(db_plans_src, "update_meal_plan_data")
    builder_pattern = re.compile(r"\.table\(|\.eq\(")
    assert not builder_pattern.search(body), (
        "P1-NEON-DB-MIGRATION regresión: `update_meal_plan_data` volvió a "
        "tener un path builder PostgREST (`.table(`/`.eq(`). El único "
        "transporte permitido es psycopg via connection_pool (advisory "
        "lock + WHERE id/user_id en la misma transacción)."
    )
    assert "if not connection_pool" in body, (
        "P1-NEON-DB-MIGRATION regresión: desapareció el guard fail-loud "
        "`if not connection_pool` del helper. Sin pool el helper debe "
        "fallar ruidoso (RuntimeError → except → None), no degradar a "
        "otro transporte."
    )


def test_all_production_callsites_pass_user_id():
    """Los 4 callsites de producción identificados en el audit deben
    pasar `user_id=` al helper. Drift detection: si un nuevo callsite
    olvida user_id, este test lo captura."""
    files_to_check = [
        _PROACTIVE_PY,
        _PLANS_ROUTER_PY,
        _TOOLS_PY,
    ]
    # Patrón para detectar callsites que NO pasen user_id:
    # `update_meal_plan_data(X, Y)` sin un tercer arg ni kwarg user_id.
    callsite_pattern = re.compile(
        r"update_meal_plan_data\s*\(\s*([^()]*(?:\([^)]*\)[^()]*)*)\s*\)",
        re.DOTALL,
    )

    offenders = []
    for f in files_to_check:
        if not f.exists():
            continue
        src = f.read_text(encoding="utf-8")
        for m in callsite_pattern.finditer(src):
            args = m.group(1)
            # Excluimos el blueprint del helper (en `def update_meal_plan_data`
            # con `user_id: str = None` — no es un callsite).
            preceding_chars = src[max(0, m.start() - 30):m.start()]
            if "def " in preceding_chars:
                continue
            # Si el call contiene `user_id=` o pasa 3 args posicionales, OK.
            if "user_id=" in args:
                continue
            # Conteo de comas a top-level (ignorando paréntesis anidados) — 2+
            # implica 3 args posicionales.
            top_level_commas = 0
            depth = 0
            for ch in args:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                elif ch == "," and depth == 0:
                    top_level_commas += 1
            if top_level_commas >= 2:
                continue
            # Localizar línea para reporte.
            line_no = src.count("\n", 0, m.start()) + 1
            offenders.append(f"  {f.name}:{line_no} → update_meal_plan_data({args.strip()[:80]}...)")

    assert not offenders, (
        "P1-NEW-3 regresión: callsites de `update_meal_plan_data` que "
        "NO pasan `user_id=` (re-introducen IDOR a DB-level):\n"
        + "\n".join(offenders)
        + "\n\nMigrar cada uno para pasar `user_id=<scope>` como tercer arg."
    )


def test_db_plans_anchor_present(db_plans_src: str):
    """Anchor textual `P1-NEW-3` en docstring del helper para `grep`
    rápido del fix."""
    assert "P1-NEW-3" in db_plans_src, (
        "P1-NEW-3 regresión: el anchor `P1-NEW-3` desapareció de "
        "`update_meal_plan_data`. Restaurar para `grep -r P1-NEW-3`."
    )
