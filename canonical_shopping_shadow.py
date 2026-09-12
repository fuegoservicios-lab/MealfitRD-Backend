# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-19 · 2026-09-12 · E5-A] Sombra de la lista de compras CANÓNICA: mide, no escribe.

`ARQ30-P1-01` propone que la lista de compras se DERIVE de la representación canónica
(`canonical_recipe.IngredientLine`). El encargo fija el camino —expand → sombra → canario por cohorte → promoción →
retirada— y dice que nunca se borre la vía anterior en el mismo paso que transfiere su autoridad. `canonical_recipe.py`
fue el expand (solo lectura, 2026-09-06). Esto es la SOMBRA:

  · corre donde corre el guard de coherencia (`shopping_calculator.run_shopping_coherence_guard`: las 6 superficies de
    `docs/coherence_surfaces_table.md`), con el MISMO multiplicador efectivo que el guard aplica al lado esperado
    (hogar × base de días), así que compara like con like;
  · construye la lista canónica por alimento —gramos de cada `IngredientLine`, sumados sobre `shopping_source_days`, el
    SSOT del ciclo (P0-SHOPPING-CYCLE-DAYS)— y la compara con la lista que HOY se entrega
    (`aggregated_shopping_list_weekly`, o la activa si no hay semanal);
  · persiste SÓLO la comparación en `pipeline_metrics` (node `canonical_shopping_shadow`). **No toca `plan_result`.**

Sale a la fase B (canario por cohorte, decisión del dueño) cuando ≥ 30 planes distintos den `parse_fail` < 1 % y
divergencia > 10 % en < 5 % de los alimentos comparables — lo mide `scripts/measure_canonical_shadow.py`.

Tres trampas ya pagadas que este módulo respeta (`docs/arq30_e5_e7_diseno_canario.md`): (1) la identidad de un alimento
es el nombre del catálogo canonicalizado como lo hace el guard (`_canonicalize_for_coherence`), nunca el texto
normalizado; (2) la lista servida y la referencia del guard no cambian de fuente: aquí no se sirve nada; (3) el shift no
reconstruye la lista, así que la sombra corre también sobre planes ya desplazados (el cron diario re-evalúa) y anota
`dias_archivados`.

Knob: `MEALFIT_CANONICAL_SHOPPING_SHADOW` (default True; apagarlo quita la sombra sin redeploy).
tooltip-anchor: P1-PLAN-LOTE-19-CANONICAL-SHADOW
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from typing import Any, Optional

from canonical_recipe import parse_line

logger = logging.getLogger(__name__)

NODE = "canonical_shopping_shadow"
#: La misma tolerancia con la que el guard juzga magnitudes (`MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT` default 0,10).
TOLERANCIA = 0.10
#: Criterio de salida a la fase B (`arq30_e5_e7_diseno_canario.md`, tabla E5).
GATE_MIN_PLANES = 30
GATE_PARSE_FAIL_PCT_MAX = 1.0
GATE_DIVERGENTES_PCT_MAX = 5.0

_HAS_QTY = re.compile(r"\d|[½¼¾⅓⅔⅛]")
_NDB: dict[str, Any] = {"inst": None, "ts": 0.0}
_NDB_TTL_S = 300.0


def shadow_enabled() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_CANONICAL_SHOPPING_SHADOW", True)
    except Exception:
        return True


def _ndb():
    """UNA instancia del catálogo nutricional por ventana de 5 min: construir el índice compila ~700 regex."""
    now = time.time()
    if _NDB["inst"] is None or now - _NDB["ts"] > _NDB_TTL_S:
        from nutrition_db import IngredientNutritionDB
        _NDB["inst"], _NDB["ts"] = IngredientNutritionDB(), now
    return _NDB["inst"]


def _canon(name: str) -> str:
    """La identidad del alimento como la decide el GUARD, para que las dos listas hablen del mismo alimento."""
    try:
        from shopping_calculator import _canonicalize_for_coherence
        out = _canonicalize_for_coherence({str(name).strip()})
        if out:
            return str(next(iter(out)))
    except Exception:
        pass
    return str(name or "").strip().lower()


def _lineas_de(meal: dict) -> list[str]:
    """Las líneas que el agregador y el guard leen, con su mismo fallback (`expected_sum_from_recipes`)."""
    ings = meal.get("ingredients_raw") or meal.get("ingredients") or []
    if not ings:
        rec = meal.get("recipe")
        if isinstance(rec, dict):
            ings = rec.get("ingredients") or []
    out = []
    for ing in ings:
        if isinstance(ing, str):
            out.append(ing)
        elif isinstance(ing, dict):
            q, u = ing.get("quantity", 0), ing.get("unit", "unidad")
            n = ing.get("name") or ing.get("item_name") or ing.get("display_name") or "Desconocido"
            try:
                con_q = float(q) > 0
            except (TypeError, ValueError):
                con_q = False
            out.append(f"{q} {u} de {n}" if con_q or u in ("pizca", "al gusto", "cantidad necesaria") else str(n))
    return [s for s in out if s and len(str(s)) >= 3]


def _gramos(qty: Any, unit: Any, name: str, ndb) -> Optional[float]:
    """Gramos de (cantidad, unidad) preguntando a las MISMAS autoridades que `parse_line`; `None` = no convertible."""
    try:
        q = float(qty)
    except (TypeError, ValueError):
        return None
    if math.isnan(q) or math.isinf(q) or q <= 0:
        return None
    u = str(unit or "").strip()
    try:
        from canonical_units import canonicalize_unit, to_base_amount
        uc = canonicalize_unit(u) if u else None
        if uc:
            base, base_unit = to_base_amount(q, uc)
            if base is not None and str(base_unit or "").lower() in ("g", "gramo", "gramos"):
                return float(base)
    except Exception:
        uc = None
    try:
        if ndb is not None:
            info = ndb.lookup(name)
            if info is not None:
                g = ndb.to_grams(q, uc or u or "", info)
                if g is not None and g > 0:
                    return float(g)
    except Exception:
        pass
    return None


def _item_gramos(item: dict, ndb) -> Optional[float]:
    """Gramos de un item de la lista PERSISTIDA. La lista lleva `base_qty`/`base_unit` (la magnitud en gramos que el
    agregador calculó, medido: 46 de 47 items en `g`; el resto es una unidad de envase, «cartón (20 uds.)», que el
    guard ya trata como `unit_mismatch`); `quantity`/`unit` es la forma de los fixtures y de listas viejas."""
    nombre = str(item.get("name") or item.get("display_name") or "")
    if item.get("base_qty") is not None:
        g = _gramos(item.get("base_qty"), item.get("base_unit"), nombre, ndb)
        if g is not None:
            return g
    for k in ("grams", "quantity_g", "grams_total"):
        try:
            v = float(item.get(k))
            if v > 0:
                return v
        except (TypeError, ValueError):
            continue
    if item.get("quantity") is not None:
        return _gramos(item.get("quantity"), item.get("unit"), nombre, ndb)
    return None


def _factor_politica(raw: str, protein_seal: bool) -> float:
    """La POLÍTICA de compra que la lista ya lleva sellada, preguntada al parser legacy y no reinventada: rendimiento
    de legumbres (seco→cocido) y, si la lista lo sella, el rendimiento de proteína. Es lo que `expected_sum_from_recipes`
    espeja para el guard; la canónica registra lo que el TEXTO dice (`apply_yield_multiplier=False`), así que para
    comparar like con like se le aplica el mismo factor. La fase B decidirá si la canónica adopta esa política."""
    try:
        from shopping_calculator import _parse_quantity
        q0, _, _ = _parse_quantity(raw, apply_yield_multiplier=False)
        q1, _, _ = _parse_quantity(raw, apply_yield_multiplier=False, apply_legumbres_yield_only=True,
                                   apply_protein_yield=bool(protein_seal))
        if q0 and q1 and float(q0) > 0 and float(q1) > 0:
            return float(q1) / float(q0)
    except Exception:
        pass
    return 1.0


def effective_multiplier_like_guard(plan_result: dict, multiplier: Optional[float] = None) -> float:
    """El multiplicador que el guard aplica al lado esperado: hogar × (7 / días fuente) cuando compara contra la lista
    SEMANAL (P1-COHERENCE-DAY-BASIS). Espejo para las mediciones fuera del guard; dentro, el hook pasa el suyo."""
    try:
        mult = float(multiplier if multiplier is not None else (plan_result.get("calc_household_multiplier") or 1.0))
    except (TypeError, ValueError):
        mult = 1.0
    if math.isnan(mult) or math.isinf(mult) or mult <= 0:
        mult = 1.0
    try:
        from shopping_calculator import _get_coherence_day_basis_norm_knob, shopping_source_days
        if _get_coherence_day_basis_norm_knob() and plan_result.get("aggregated_shopping_list_weekly"):
            n = len(shopping_source_days(plan_result))
            if n > 0:
                mult *= 7.0 / float(n)
    except Exception:
        pass
    return mult


def plan_fingerprint(plan_result: dict) -> Optional[str]:
    """Identidad de CONTENIDO del plan para contar planes distintos aunque el llamante no sepa el `plan_id`."""
    try:
        filas = []
        for d in (plan_result or {}).get("days") or []:
            for m in (d or {}).get("meals") or []:
                if isinstance(m, dict):
                    filas.append([str(m.get("name") or ""), [str(x) for x in (m.get("ingredients_raw") or m.get("ingredients") or [])]])
        if not filas:
            return None
        return hashlib.sha256(json.dumps(filas, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    except Exception:
        return None


def compute_shadow(plan_result: dict, *, multiplier: float = 1.0, nutrition_db=None,
                   tolerance: float = TOLERANCIA) -> dict:
    """La comparación, pura: lista canónica (IngredientLine sobre `shopping_source_days`) frente a la lista entregada.

    No escribe en `plan_result`. `None` en gramos significa «no convertible», nunca cero: esos alimentos van a
    `no_comparables`, no a divergentes ni a coincidentes.
    """
    from shopping_calculator import _should_skip_meal_for_aggregation, shopping_source_days

    ndb = nutrition_db if nutrition_db is not None else _ndb()
    try:
        mult = float(multiplier)
        if math.isnan(mult) or math.isinf(mult) or mult <= 0:
            mult = 1.0
    except (TypeError, ValueError):
        mult = 1.0

    # La lista con la que se compara, y sus SELLOS: la política de compra que ya lleva puesta (rendimiento de proteína,
    # deducción de nevera). Se leen ANTES de recorrer las recetas porque el lado canónico los espeja.
    lista_usada = None
    lista = plan_result.get("aggregated_shopping_list_weekly")
    if lista:
        lista_usada = "weekly"
    else:
        lista = plan_result.get("aggregated_shopping_list") or []
        lista_usada = "active" if lista else None
    protein_seal, pantry_seal = False, None
    try:
        from shopping_calculator import _pantry_deduction_seal, _protein_yield_seal_applied
        protein_seal = bool(_protein_yield_seal_applied(lista or []))
        pantry_seal = _pantry_deduction_seal(lista or [])
    except Exception:
        pass

    dias = shopping_source_days(plan_result) or []
    lines = parse_fail = sin_cantidad = sin_gramos = 0
    fuentes, estados, sin_gramos_por_alimento = Counter(), Counter(), Counter()
    canon_g: dict[str, float] = defaultdict(float)
    canon_sin_g: set[str] = set()
    for d in dias:
        if not isinstance(d, dict):
            continue
        for meal in d.get("meals") or []:
            if not isinstance(meal, dict) or _should_skip_meal_for_aggregation(meal):
                continue
            for raw in _lineas_de(meal):
                lines += 1
                ln = parse_line(raw, ndb)
                if ln is None:
                    if _HAS_QTY.search(str(raw)):
                        parse_fail += 1
                    else:
                        sin_cantidad += 1
                    continue
                if ln.qty is None or ln.qty <= 0:
                    sin_cantidad += 1      # «Sal al gusto»: sin número no hay nada que comprar por gramos
                    continue
                estados[ln.state] += 1
                key = _canon(ln.name)
                if ln.grams is None:
                    sin_gramos += 1
                    sin_gramos_por_alimento[ln.name] += 1
                    canon_sin_g.add(key)
                    continue
                fuentes[str(ln.grams_source)] += 1
                canon_g[key] += float(ln.grams) * mult * _factor_politica(raw, protein_seal)

    agg_g: dict[str, float] = defaultdict(float)
    agg_sin_g: set[str] = set()
    for item in lista or []:
        if not isinstance(item, dict):
            continue
        cat = str(item.get("category") or item.get("display_category") or "").lower()
        if "urgente" in cat or item.get("is_staple") is True:
            continue
        nm = item.get("name") or item.get("display_name")
        if not nm:
            continue
        key = _canon(str(nm))
        g = _item_gramos(item, ndb)
        if g is None:
            agg_sin_g.add(key)
        else:
            agg_g[key] += g

    comparables = descontado_nevera = 0
    divergentes, canonical_only, aggregated_only = [], [], []
    no_comparables = 0
    for food in sorted(set(canon_g) | set(agg_g) | canon_sin_g | agg_sin_g):
        c, a = canon_g.get(food), agg_g.get(food)
        if c is not None and a is not None:
            comparables += 1
            delta = (abs(a - c) / c) if c > 0 else float("inf")
            if delta > tolerance and pantry_seal is True and a < c:
                # la lista es NETA de nevera: comprar menos de lo que la receta pide es la deducción, no una divergencia
                descontado_nevera += 1
                continue
            if delta > tolerance:
                divergentes.append({"food": food, "canonical_g": round(c, 1), "aggregated_g": round(a, 1),
                                    "delta_pct": (round(delta * 100.0, 1) if delta != float("inf") else None)})
        elif c is not None:
            if food in agg_sin_g:
                no_comparables += 1
            else:
                canonical_only.append(food)
        elif a is not None:
            if food in canon_sin_g:
                no_comparables += 1
            else:
                aggregated_only.append(food)
        else:
            no_comparables += 1
    divergentes.sort(key=lambda x: (x["delta_pct"] is not None, -(x["delta_pct"] or 0)))

    def pct(n: int, den: int) -> Optional[float]:
        return round(100.0 * n / den, 2) if den else None

    return {
        "lines": lines, "parse_fail": parse_fail, "parse_fail_pct": pct(parse_fail, lines),
        "sin_cantidad": sin_cantidad, "sin_gramos": sin_gramos,
        "sin_gramos_ej": [f"{n} ×{c}" for n, c in sin_gramos_por_alimento.most_common(8)],
        "grams_source": dict(fuentes), "estados": dict(estados),
        "foods_canonical": len(canon_g), "foods_aggregated": len(agg_g),
        "comparables": comparables, "divergentes": len(divergentes),
        "divergentes_pct": pct(len(divergentes), comparables),
        "canonical_only": len(canonical_only), "canonical_only_ej": canonical_only[:8],
        "aggregated_only": len(aggregated_only), "aggregated_only_ej": aggregated_only[:8],
        "no_comparables": no_comparables, "descontado_nevera": descontado_nevera,
        "sellos": {"protein_yield_applied": protein_seal, "pantry_deduction_applied": pantry_seal},
        "ejemplos": divergentes[:5],
        "multiplier": round(mult, 4), "tolerance": tolerance,
        "source_days": len(dias),
        "dias_archivados": len(plan_result.get("_archived_days") or []),
        "lista_usada": lista_usada,
    }


def emit_canonical_shopping_shadow(plan_result: dict, *, multiplier: Optional[float] = None, surface: Optional[str] = None,
                                   plan_id: Optional[str] = None, user_id: Optional[str] = None) -> Optional[dict]:
    """Calcula la sombra y la persiste en `pipeline_metrics` (best-effort). Devuelve la comparación o `None`.

    Fail-safe total y de SOLO LECTURA sobre `plan_result`: un fallo aquí no puede tocar la lista que se entrega ni
    el veredicto del guard.
    """
    try:
        if not shadow_enabled() or not isinstance(plan_result, dict) or not plan_result.get("days"):
            return None
        t0 = time.time()
        mult = multiplier if multiplier is not None else effective_multiplier_like_guard(plan_result)
        r = compute_shadow(plan_result, multiplier=mult)
        r["surface"] = surface
        r["plan_id"] = str(plan_id) if plan_id else None
        r["plan_fp"] = plan_fingerprint(plan_result)
        dur = int((time.time() - t0) * 1000)
        try:
            from db_core import execute_sql_write
            execute_sql_write(
                """
                INSERT INTO pipeline_metrics
                    (user_id, session_id, node, duration_ms, retries, tokens_estimated, confidence, metadata)
                VALUES (%s, NULL, %s, %s, 0, 0, 0, %s::jsonb)
                """,
                (user_id, NODE, dur, json.dumps(r, ensure_ascii=False, default=str)),
            )
        except Exception as e:  # la métrica es telemetría: jamás rompe al llamante
            logger.debug(f"[P1-PLAN-LOTE-19] sombra canónica sin persistir: {type(e).__name__}: {e}")
        return r
    except Exception as e:
        logger.debug(f"[P1-PLAN-LOTE-19] sombra canónica no calculada: {type(e).__name__}: {e}")
        return None


def gate_verdict(planes: int, parse_fail_pct: Optional[float], divergentes_pct: Optional[float]) -> str:
    """Tres salidas, ninguna ambigua: sin ≥ 30 planes distintos NO hay veredicto (una medición sin datos no colapsa a
    ningún lado); con ellos, PASA o NO PASA contra los dos umbrales del diseño."""
    if planes < GATE_MIN_PLANES or parse_fail_pct is None or divergentes_pct is None:
        return f"NO CONCLUYENTE: {planes} planes distintos con sombra (hacen falta ≥ {GATE_MIN_PLANES})"
    if parse_fail_pct < GATE_PARSE_FAIL_PCT_MAX and divergentes_pct < GATE_DIVERGENTES_PCT_MAX:
        return (f"PASA la fase A: parse_fail {parse_fail_pct} % < {GATE_PARSE_FAIL_PCT_MAX} % y divergencia "
                f"{divergentes_pct} % < {GATE_DIVERGENTES_PCT_MAX} % ⇒ fase B posible (cohorte: dueño)")
    return (f"NO PASA: parse_fail {parse_fail_pct} % (tope {GATE_PARSE_FAIL_PCT_MAX}) · divergencia {divergentes_pct} % "
            f"(tope {GATE_DIVERGENTES_PCT_MAX}) ⇒ seguir en sombra y leer los ejemplos")


__all__ = ["NODE", "TOLERANCIA", "compute_shadow", "emit_canonical_shopping_shadow", "effective_multiplier_like_guard",
           "plan_fingerprint", "shadow_enabled", "gate_verdict", "GATE_MIN_PLANES"]
