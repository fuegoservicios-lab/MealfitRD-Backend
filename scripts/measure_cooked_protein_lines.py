"""[P2-PROTEIN-YIELD-CANONICAL · 2026-08-03] Medición READ-ONLY (SELECT, cero escritura) de
cuántas líneas de `ingredients_raw` en los planes ACTIVOS matchean la regla #2 de
`_calculate_yield_multiplier` (proteínas cocidas → 1.35× crudo, `shopping_calculator.py`):
adjetivo de cocción (cocido/hervido/asado/horneado/desmenuzado/frito) + palabra de proteína
(pollo/carne/res/pescado/cerdo/camarón/pavo/salmón/filete).

## Por qué esta medición decide la tarea

`_calculate_yield_multiplier` regla #2 EXISTE desde P1-2, pero el aggregator la apaga siempre
(`apply_yield_multiplier=False, apply_legumbres_yield_only=True`) por la asimetría plan↔inventario:
el `physical_inventory` que el usuario tipea en su Nevera está en peso literal sin "cocido", así
que aplicar yield solo al lado del plan sesgaba el delta hacia over-buy. Esa razón NO aplica en la
lista CANÓNICA (`get_shopping_list_delta(is_new_plan=True)` fuerza `physical_inventory=[]` —
P3-CANONICAL-AGG-WEEKLY), pero antes de tocar código hay que medir si el patrón aparece lo
suficiente como para justificar el cambio (no un no-op).

## Resultado de la corrida (2026-08-03, contra los 23 planes vivos)

    12/5.899 líneas de `ingredients_raw` (0,203%) matchean la regla #2, pero
    **5/23 planes (~22%) tienen al menos una línea**. Ejemplos reales:
      - «205 g de pollo cocido y desmenuzado»
      - «160 g de pescado cocido»
      - «45 g de costilla de cerdo cocida y desmenuzada»
      - «100 g de cerdo magro cocido y desmenuzado»

    Cada match es ~26% de under-buy de proteína en ese alimento (1 lb cocida declarada ⇒ solo
    0,74 lb cruda comprada tras la cocción). Conclusión: NO es un no-op — procede la
    implementación con el knob `MEALFIT_PROTEIN_YIELD_ON_CANONICAL` default OFF (A/B antes de
    encender), documentada en `_protein_yield_on_canonical_enabled()` (shopping_calculator.py).

    Caso borde detectado en los datos reales: una línea trae un marcador de REUSO —
    «205 g de pollo cocido y desmenuzado (del almuerzo o preparado extra)» — la proteína ya se
    compró para otra comida; aplicarle yield la sobre-compraría. La implementación excluye
    estas líneas vía `_PROTEIN_REUSE_PAREN_RE` (paréntesis con desayuno/almuerzo/cena/
    merienda/sobrante/preparado extra).

## Qué hace este script

  1. SELECT (read-only) `id, user_id, plan_data` de `meal_plans` "activos": `generation_status
     = 'complete'` (I8 garantiza `days` no vacío en ese estado) — mismo query que
     `scripts/backfill_veg_lines_v7.py::_select_active_plans`.
  2. Por cada plan: recorre `days[*].meals[*].ingredients_raw` (fallback `ingredients`) y
     clasifica cada línea de texto (≥3 chars) contra los MISMOS regex que
     `shopping_calculator._PROTEIN_COOKED_ADJ_RE`/`_PROTEIN_FOOD_WORDS_RE` (importados del
     módulo real — SSOT, cero riesgo de que el script derive de un patrón distinto al que
     corre en producción) y contra `_PROTEIN_REUSE_PAREN_RE` (líneas de reuso, reportadas aparte).
  3. Imprime: total de líneas / líneas que matchean la regla #2 (numerador/denominador global),
     planes con ≥1 match / total de planes, y hasta 15 líneas de ejemplo (con su plan_id truncado)
     para verificación manual.
  4. CERO escritura. No hay flag `--apply` porque este script no modifica nada — es
     estrictamente de medición, análogo al Step 1 del brief de la tarea.

## ⚠️ Esta sesión NO ejecutó este script

El worktree de implementación no tiene `.env` (sin `NEON_DATABASE_URL`), así que no hay forma de
conectar a Neon desde aquí — el `.env` real apunta a PRODUCCIÓN y correr contra él desde un
sandbox de desarrollo violaría el aislamiento de tests. Los números de la sección "Resultado de
la corrida" arriba vienen de la medición YA ejecutada por el operador antes de esta tarea (SOP
"SQL forense antes de tocar código", CLAUDE.md). Este script formaliza esa medición para que sea
reproducible/re-corrible cuando cambien los datos (ej. antes de decidir si subir el knob a
default `True`).

## Uso

    python scripts/measure_cooked_protein_lines.py                # todos los planes activos
    python scripts/measure_cooked_protein_lines.py --limit 5       # acotado (debug/muestreo)
    python scripts/measure_cooked_protein_lines.py --show 30       # más líneas de ejemplo
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import db_core  # noqa: E402
from db_core import execute_sql_query  # noqa: E402


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--limit", type=int, default=None,
                   help="Tope de planes a procesar (debug/muestreo). Default: todos.")
    p.add_argument("--show", type=int, default=15,
                   help="Cuántas líneas de ejemplo imprimir (default 15).")
    return p.parse_args(argv)


def _select_active_plans(limit: int | None):
    """READ-ONLY. Planes 'activos': generation_status='complete' (I8 garantiza days no vacío).
    Mismo query que scripts/backfill_veg_lines_v7.py::_select_active_plans (SSOT del filtro)."""
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


def _iter_ingredient_lines(plan_data: dict):
    """Recorre days[*].meals[*].ingredients_raw (fallback ingredients) — mismo fallback que
    `get_shopping_list_delta`/`expected_sum_from_recipes` en shopping_calculator.py."""
    if not isinstance(plan_data, dict):
        return
    for day in plan_data.get("days") or []:
        if not isinstance(day, dict):
            continue
        for meal in day.get("meals") or []:
            if not isinstance(meal, dict):
                continue
            ingredients = meal.get("ingredients_raw") or meal.get("ingredients") or []
            for ing in ingredients:
                if isinstance(ing, str) and len(ing.strip()) >= 3:
                    yield ing


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    if db_core.connection_pool is None:
        print("[measure-cooked-protein-lines] db_core.connection_pool no está configurado "
              "(faltan NEON_DATABASE_URL/NEON_DATABASE_URL_POOLED en el entorno). Abortando.",
              file=sys.stderr)
        return 1
    # [P2-MUTATOR-PURITY] El pool nace cerrado (open=False) fuera de FastAPI — hay que abrirlo
    # explícitamente (CLAUDE.md: "abrir el pool fuera de FastAPI o mides el vacío, no el sistema").
    db_core.connection_pool.open()

    try:
        # SSOT: mismos regex que producción (shopping_calculator._calculate_yield_multiplier
        # regla #2). Importar en vez de reimplementar evita que este script mida un patrón
        # ligeramente distinto del que realmente corre.
        from shopping_calculator import (
            _PROTEIN_COOKED_ADJ_RE, _PROTEIN_FOOD_WORDS_RE, _PROTEIN_REUSE_PAREN_RE,
        )

        rows = _select_active_plans(args.limit)
        print(f"[measure-cooked-protein-lines] {len(rows)} plan(es) activo(s) "
              f"(generation_status=complete).")

        total_lines = 0
        matched_lines: list[tuple[str, str]] = []  # (plan_id, line)
        reuse_lines: list[tuple[str, str]] = []
        plans_with_match = 0

        for row in rows:
            plan_id = str(row.get("plan_id") or "?")[:8]
            plan_data = row.get("plan_data") or {}
            plan_has_match = False
            for line in _iter_ingredient_lines(plan_data):
                total_lines += 1
                n = line.lower()
                if _PROTEIN_COOKED_ADJ_RE.search(n) and _PROTEIN_FOOD_WORDS_RE.search(n):
                    plan_has_match = True
                    if _PROTEIN_REUSE_PAREN_RE.search(n):
                        reuse_lines.append((plan_id, line))
                    else:
                        matched_lines.append((plan_id, line))
            if plan_has_match:
                plans_with_match += 1

        n_matched = len(matched_lines) + len(reuse_lines)
        pct_lines = (100.0 * n_matched / total_lines) if total_lines else 0.0
        pct_plans = (100.0 * plans_with_match / len(rows)) if rows else 0.0

        print(f"\n[measure-cooked-protein-lines] {n_matched}/{total_lines} línea(s) de "
              f"ingredients_raw matchean la regla #2 ({pct_lines:.3f}%).")
        print(f"[measure-cooked-protein-lines] {plans_with_match}/{len(rows)} plan(es) "
              f"tienen AL MENOS una línea ({pct_plans:.1f}%).")
        if reuse_lines:
            print(f"[measure-cooked-protein-lines] {len(reuse_lines)} de esas líneas traen "
                  f"marcador de REUSO (excluidas del yield por _PROTEIN_REUSE_PAREN_RE):")
            for pid, line in reuse_lines[: args.show]:
                print(f"  - [{pid}] {line}")

        if matched_lines:
            print(f"\n[measure-cooked-protein-lines] muestra de hasta {args.show} línea(s) "
                  f"que SÍ recibirían yield (knob ON):")
            for pid, line in matched_lines[: args.show]:
                print(f"  - [{pid}] {line}")

        return 0
    finally:
        try:
            db_core.connection_pool.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
