"""[P1-VEG-BACKFILL-HONESTY · 2026-08-02] Backfill one-shot para planes YA PERSISTIDOS cuya
receta (texto) contradice su propia lista de compras — el residuo de DATOS que Task 7
(P1-RECONCILE-CDA-DENSITY) no toca (esa arregla el lado GENERADOR, sólo aplica a planes NUEVOS).

## Evidencia medida en producción (SELECT, sin escritura — ver task-8-brief.md)

  - plan `5f4bb17e`: receta dice "600 g de espárragos" en la cena; la lista SEMANAL persistida
    compra sólo 583.33 g — una sola cena agota el 103% de la compra de la semana. `capped_by=null`,
    sin aviso: espárragos no vive en `_VEG_PER_WEEK_PER_PERSON` (P5-VEG-CAP) ni en ningún otro cap
    por categoría, así que el déficit nunca se explicó.
  - plan `8d3f246a`: "470 g de tayota" vs lista 891 g (ratio 0.635 lado guard).
  - plan `cf3a81fb`: vainitas 933 g (=400×7/3, la inflación del solver entregada COMPLETA) +
    calabacín 1372 g para una persona.

## Qué hace

  1. SELECT (read-only) `id, user_id, plan_data` de `meal_plans` "activos": `generation_status
     = 'complete'` (I8 garantiza `days` no vacío en ese estado).
  2. Por cada plan: sobre una COPIA profunda de `plan_data["days"]`, corre
     `graph_orchestrator._cap_unrealistic_portions(days, db)` — el MISMO mecanismo que Task 7
     (P1-RECONCILE-CDA-DENSITY) ya corrigió para plan nuevos (masa implícita taza/cda/cdta/conteo,
     techo `REALISM_VEG_VOLUME_CAP_G`, reescala en lockstep `ingredients`+`ingredients_raw`+pasos
     vía `rescale_ingredient_string`/`quantize_ingredient_string`). NO se reimplementa el cap: se
     REUSA el SSOT ya fixeado, aplicado retroactivamente a `days` que nacieron ANTES del fix.
  3. Si el cap recortó algo (retorno > 0): recomputa las 4 `aggregated_shopping_list*` sobre la
     COPIA vía `routers.plans._rebuild_plan_shopping_lists_inline` (misma matemática que el recalc
     inline — `is_new_plan=True`, sin netear contra inventario, igual que la generación original)
     y compara ítem-por-ítem contra la lista YA persistida para mostrar el delta.
  4. `--dry-run` (comportamiento DEFAULT — sin flags): sólo SELECT + prints, CERO escritura. Por
     plan: qué línea(s) de receta se recortarían (de → a) y qué ítem(s) de
     `aggregated_shopping_list_weekly` cambiarían (antes → después).
  5. `--apply`: persiste vía `db_plans.update_plan_data_atomic(plan_id, mutator, user_id=user_id)`
     — `SELECT … FOR UPDATE` + callback fresh, invariante I7 (full-overwrite de `plan_data`
     requiere este patrón o un advisory lock explícito). El mutator SOLO transforma el dict que
     recibe (row-lock activo, P2-MUTATOR-PURITY): el catálogo (`get_master_ingredients()`,
     `_watery_veg_tokens()`, `IngredientNutritionDB`) se calienta ANTES de entrar al mutator —
     mismo patrón que `routers/plans.py::api_swap_meal_persist` (líneas ~7010-7023, comentario
     "warm-up fuera del FOR UPDATE") — para que dentro de la transacción sólo haya cache-hits.

## ⚠️ Esta sesión NO ejecuta `--apply`

`--apply` requiere ADEMÁS pasar `--yes-i-mean-it` (doble flag, defensa contra un `--apply` suelto
en el historial de shell) y modifica datos de PRODUCCIÓN. Este script fue escrito y revisado pero
**no ejecutado ni en `--dry-run`** en la sesión que lo escribió — el worktree no tenía `.env`
(sin `NEON_DATABASE_URL`), así que no había forma de conectar. Antes de `--apply` en producción:
correr `--dry-run` primero, leer el diff completo, y obtener OK explícito del dueño (toca datos de
usuarios reales).

## Uso

    python scripts/backfill_veg_lines_v7.py                          # dry-run (SELECT + print)
    python scripts/backfill_veg_lines_v7.py --apply --yes-i-mean-it   # aplica el UPDATE (NUNCA
                                                                       # invocado en esta sesión)
    python scripts/backfill_veg_lines_v7.py --limit 5                # dry-run acotado (debug)
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import db_core  # noqa: E402
from db_core import execute_sql_query  # noqa: E402


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--apply", action="store_true",
                    help="Persiste los cambios (requiere --yes-i-mean-it). Default: dry-run.")
    p.add_argument("--yes-i-mean-it", action="store_true",
                    help="Confirmación explícita para --apply — sin esto, --apply es no-op.")
    p.add_argument("--limit", type=int, default=None,
                    help="Tope de planes a procesar (debug/muestreo). Default: todos.")
    return p.parse_args(argv)


def _select_active_plans(limit: int | None):
    """READ-ONLY. Planes 'activos': generation_status='complete' (I8 garantiza days no vacío)."""
    query = (
        "SELECT id::text AS plan_id, user_id::text AS user_id, plan_data "
        "FROM meal_plans "
        "WHERE plan_data->>'generation_status' = 'complete' "
        "ORDER BY created_at DESC"
    )
    if limit:
        query += " LIMIT %s"
        rows = execute_sql_query(query, (int(limit),), fetch_all=True)
    else:
        rows = execute_sql_query(query, fetch_all=True)
    return rows or []


def _diff_ingredient_lines(before_days: list, after_days: list) -> list[dict]:
    """Compara `ingredients`/`ingredients_raw` línea a línea entre el plan_data ORIGINAL y la
    copia ya pasada por `_cap_unrealistic_portions`. Devuelve sólo las líneas que cambiaron."""
    diffs = []
    for d_idx, (before_day, after_day) in enumerate(zip(before_days or [], after_days or [])):
        if not isinstance(before_day, dict) or not isinstance(after_day, dict):
            continue
        before_meals = before_day.get("meals") or []
        after_meals = after_day.get("meals") or []
        for m_idx, (before_meal, after_meal) in enumerate(zip(before_meals, after_meals)):
            if not isinstance(before_meal, dict) or not isinstance(after_meal, dict):
                continue
            meal_name = str(after_meal.get("name") or before_meal.get("name") or "?")[:60]
            for field in ("ingredients", "ingredients_raw"):
                before_list = before_meal.get(field) or []
                after_list = after_meal.get(field) or []
                for i_idx, (b, a) in enumerate(zip(before_list, after_list)):
                    if str(b) != str(a):
                        diffs.append({
                            "day": d_idx + 1, "meal": meal_name, "field": field,
                            "index": i_idx, "before": str(b), "after": str(a),
                        })
    return diffs


def _diff_shopping_list_items(before_list: list, after_list: list) -> list[dict]:
    """Compara `aggregated_shopping_list_weekly` (u otro surface) por nombre. Sólo los ítems cuyo
    `display_qty` cambió (recorte que se propagó a la lista) aparecen en el resultado."""
    def _by_name(lst):
        out = {}
        for it in lst or []:
            if isinstance(it, dict) and it.get("name"):
                out[it["name"]] = it
        return out

    before_map = _by_name(before_list)
    after_map = _by_name(after_list)
    diffs = []
    for name in sorted(set(before_map) | set(after_map)):
        b = before_map.get(name)
        a = after_map.get(name)
        b_qty = b.get("display_qty") if b else None
        a_qty = a.get("display_qty") if a else None
        if b_qty != a_qty:
            diffs.append({"name": name, "before": b_qty, "after": a_qty})
    return diffs


def _process_one_plan(row: dict, *, db, warm_master_map_note: bool = False) -> dict | None:
    """Corre el recorte + rebuild sobre una COPIA de `plan_data`. NO escribe nada — el caller
    decide si persiste (`--apply`) o sólo imprime (`--dry-run`, default). Devuelve None si el
    plan no tuvo cambios (no-op, nada que reportar)."""
    from graph_orchestrator import _cap_unrealistic_portions

    plan_id = row["plan_id"]
    user_id = row["user_id"]
    original = row.get("plan_data") or {}
    if not isinstance(original, dict) or not original.get("days"):
        return None

    working = copy.deepcopy(original)
    days = working.get("days") or []
    n_capped = _cap_unrealistic_portions(days, db=db)
    if not n_capped:
        return None

    line_diffs = _diff_ingredient_lines(original.get("days") or [], days)

    # Recompute de las 4 listas sobre la COPIA (misma matemática que el recalc inline —
    # is_new_plan=True, sin netear inventario). Fail-open: si el rebuild falla, seguimos
    # reportando el recorte de líneas (la parte más importante del backfill) sin el diff de lista.
    list_diffs = []
    try:
        from routers.plans import _rebuild_plan_shopping_lists_inline
        _rebuild_plan_shopping_lists_inline(
            working, user_id, surface="backfill_veg_lines_v7_dry_run", plan_id_hint=plan_id,
        )
        list_diffs = _diff_shopping_list_items(
            original.get("aggregated_shopping_list_weekly") or [],
            working.get("aggregated_shopping_list_weekly") or [],
        )
    except Exception as _rebuild_exc:
        print(f"  ⚠️ rebuild de lista falló (fail-open, sólo se reporta el recorte de líneas): "
              f"{type(_rebuild_exc).__name__}: {_rebuild_exc}")

    return {
        "plan_id": plan_id, "user_id": user_id, "n_capped": n_capped,
        "line_diffs": line_diffs, "list_diffs": list_diffs, "working": working,
    }


def _print_report(result: dict) -> None:
    print(f"\n=== plan {result['plan_id']} (user {result['user_id']}) — "
          f"{result['n_capped']} recorte(s) ===")
    for d in result["line_diffs"]:
        print(f"  día {d['day']} · {d['meal']} · {d['field']}[{d['index']}]:")
        print(f"    ANTES:   {d['before']}")
        print(f"    DESPUÉS: {d['after']}")
    if result["list_diffs"]:
        print("  -- lista de compras (semanal) --")
        for it in result["list_diffs"]:
            print(f"    {it['name']}: {it['before']!r} -> {it['after']!r}")
    else:
        print("  (sin cambio detectable en aggregated_shopping_list_weekly — sólo el texto de "
              "la receta se corrigió, o el rebuild falló arriba)")


def _apply_one_plan(plan_id: str, user_id: str, *, db) -> bool:
    """[NUNCA invocado en esta sesión] Persiste el recorte vía `update_plan_data_atomic`
    (invariante I7 — full-overwrite de `plan_data` bajo row lock). El mutator corre DENTRO del
    `SELECT … FOR UPDATE`: por P2-MUTATOR-PURITY debe ser CPU-only sobre el dict que recibe — el
    catálogo (`db`, ya resuelto por el caller ANTES de esta función) entra por closure, ninguna
    llamada nueva a la DB ocurre dentro del mutator."""
    from db_plans import update_plan_data_atomic
    from graph_orchestrator import _cap_unrealistic_portions
    from routers.plans import _rebuild_plan_shopping_lists_inline

    def _mutator(plan_data: dict):
        days = plan_data.get("days") or []
        n = _cap_unrealistic_portions(days, db=db)
        if not n:
            return False  # nada que cambiar — aborta el UPDATE (ver contrato de la función)
        try:
            _rebuild_plan_shopping_lists_inline(
                plan_data, user_id, surface="backfill_veg_lines_v7_apply", plan_id_hint=plan_id,
            )
        except Exception:
            pass  # fail-open: el recorte de líneas ya vale la pena persistir sin el rebuild
        return plan_data

    result = update_plan_data_atomic(plan_id, _mutator, user_id=user_id)
    return bool(result)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    do_apply = args.apply and args.yes_i_mean_it
    if args.apply and not args.yes_i_mean_it:
        print("[backfill-veg-lines-v7] --apply sin --yes-i-mean-it -> tratado como dry-run "
              "(doble confirmación requerida para tocar producción).")

    if db_core.connection_pool is None:
        print("[backfill-veg-lines-v7] db_core.connection_pool no está configurado "
              "(faltan NEON_DATABASE_URL/NEON_DATABASE_URL_POOLED en el entorno). Abortando.",
              file=sys.stderr)
        return 1
    # [P2-MUTATOR-PURITY] El pool nace cerrado (open=False) fuera de FastAPI — hay que abrirlo
    # explícitamente (CLAUDE.md: "abrir el pool fuera de FastAPI o mides el vacío, no el sistema").
    db_core.connection_pool.open()

    try:
        # Warm-up de catálogo ANTES de tocar ningún row lock (mismo patrón que
        # routers/plans.py::api_swap_meal_persist, comentario "warm-up fuera del FOR UPDATE").
        from shopping_calculator import get_master_ingredients
        from graph_orchestrator import _watery_veg_tokens
        from nutrition_db import IngredientNutritionDB
        get_master_ingredients()
        _watery_veg_tokens()
        db = IngredientNutritionDB()

        rows = _select_active_plans(args.limit)
        print(f"[backfill-veg-lines-v7] {len(rows)} plan(es) activo(s) encontrados "
              f"(generation_status=complete). Modo: {'APPLY' if do_apply else 'DRY-RUN'}.")

        n_changed = 0
        n_applied = 0
        for row in rows:
            result = _process_one_plan(row, db=db)
            if result is None:
                continue
            n_changed += 1
            _print_report(result)
            if do_apply:
                ok = _apply_one_plan(result["plan_id"], result["user_id"], db=db)
                if ok:
                    n_applied += 1
                    print(f"  ✅ APLICADO (plan {result['plan_id']}).")
                else:
                    print(f"  ⚠️ no se aplicó (fila desapareció o no pertenece al user_id, "
                          f"plan {result['plan_id']}).")

        print(f"\n[backfill-veg-lines-v7] {n_changed}/{len(rows)} plan(es) con líneas para "
              f"recortar." + (f" {n_applied} aplicado(s)." if do_apply else " Dry-run — nada "
              f"escrito (re-correr con --apply --yes-i-mean-it tras revisar el diff arriba)."))
        return 0
    finally:
        try:
            db_core.connection_pool.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
