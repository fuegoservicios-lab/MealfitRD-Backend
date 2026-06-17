"""[P2-PRICES-ENGINE-1 · 2026-06-16] Motor de precios + inflación para la lista de compras.

CONTEXTO: el costeo ya existe — `shopping_calculator.aggregate_and_deduct_shopping_list`
lee `master_ingredients.price_per_lb`/`price_per_unit` y emite `estimated_cost_rd` por
ítem. Este módulo es la capa que **mantiene esos precios vivos** sin re-encuestar el
supermercado constantemente:

  - precio_vivo (`price_per_lb`/`price_per_unit`, lo que lee el calculador)
        = precio_base (`price_per_lb_base`/`price_per_unit_base`)
          × (food_cpi_actual / food_cpi_del_período_base)

La separación base↔vivo evita compoundear el ajuste (la base nunca cambia; el cron
sólo reescala el vivo). Un precio proyectado por índice es ESTIMADO — la columna
`price_confidence` permite a la UI etiquetarlo honestamente. El re-anclaje periódico
(re-captura de la base vía `import_base_prices`) corrige el drift.

Diseño:
  - Cero dependencia de graph_orchestrator (sólo `db_core` + `knobs`) → importable sin ciclos.
  - Todas las funciones fail-soft: sin índice ingerido o sin precios base, son no-op.
  - OFF por default (`MEALFIT_PRICES_ENABLED=false`): el cron registra pero no escribe.

Surfaces:
  - `ingest_inflation_index(period, food_cpi)`  → puebla la serie BCRD (script/admin).
  - `import_base_prices(rows)`                  → puebla precios base online (script CSV).
  - `recompute_adjusted_prices()`               → reescala vivo = base × factor (cron diario).
  - `price_staleness_report()`                  → cobertura + staleness (observabilidad).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from db_core import execute_sql_query, execute_sql_write
from knobs import _env_bool, _env_int, _env_float, _env_str

logger = logging.getLogger(__name__)

_PERIOD_RE = re.compile(r"^[0-9]{4}-[0-9]{2}$")

# Fuentes válidas de precio base (proveniencia). Free-text en DB; este set es
# para validación blanda del importer — el resto cae a 'manual'.
_KNOWN_PRICE_SOURCES = {
    "nacional_online", "sirena_online", "jumbo_online", "pricesmart_online",
    "plaza_lama_online", "manual", "crowdsource", "encuesta",
}
_VALID_CONFIDENCE = {"high", "medium", "low"}


# ── Knobs ───────────────────────────────────────────────────────────────────
def prices_enabled() -> bool:
    """Gate maestro del feature. OFF por default → rollout/rollback sin redeploy."""
    return _env_bool("MEALFIT_PRICES_ENABLED", False)


def _staleness_days() -> int:
    return _env_int("MEALFIT_PRICE_STALENESS_DAYS", 180, validator=lambda v: 1 <= v <= 1095)


def adjust_interval_h() -> int:
    """Frecuencia del cron de reescala. Default 24h (el índice es mensual; correr
    diario es idempotente y auto-sanador). Clamp [1, 168]."""
    return _env_int("MEALFIT_PRICE_ADJUST_INTERVAL_H", 24, validator=lambda v: 1 <= v <= 168)


def _max_inflation_factor() -> float:
    """Clamp de sanidad del factor de ajuste. Protege contra datos malos del índice
    (ej. food_cpi ingerido en escala equivocada → factor 100×). Default 3.0 →
    el vivo nunca se aleja >3× ni <1/3× de la base. Clamp del knob [1.1, 50]."""
    return _env_float("MEALFIT_PRICE_INFLATION_MAX_FACTOR", 3.0, validator=lambda v: 1.1 <= v <= 50.0)


def _default_price_source() -> str:
    return _env_str("MEALFIT_PRICE_REGION_DEFAULT", "nacional_online")


# ── Helpers puros (testables sin DB) ─────────────────────────────────────────
def clamp_factor(factor: float, max_factor: float) -> float:
    """Acota el factor de inflación a [1/max_factor, max_factor]. Defensivo contra
    índices corruptos. `max_factor` se asume ≥ 1 (lo garantiza el knob)."""
    if max_factor < 1.0:
        max_factor = 1.0
    lo = 1.0 / max_factor
    if factor < lo:
        return lo
    if factor > max_factor:
        return max_factor
    return factor


def _as_float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def adjusted_price(base: Any, factor: float) -> Optional[float]:
    """precio_vivo = round(base × factor, 2). None-safe."""
    b = _as_float_or_none(base)
    if b is None:
        return None
    return round(b * factor, 2)


# ── Índice de inflación (serie BCRD) ─────────────────────────────────────────
def ingest_inflation_index(
    period: str, food_cpi: float, source: Optional[str] = None, note: Optional[str] = None
) -> dict:
    """Upsert de un punto mensual del subíndice IPC de alimentos.

    `period` formato YYYY-MM. `food_cpi` > 0 (el valor del índice del BCRD, no un %).
    Idempotente por período (ON CONFLICT). Lanza ValueError en input inválido.
    """
    period = str(period or "").strip()
    if not _PERIOD_RE.match(period):
        raise ValueError(f"period inválido {period!r}: se espera YYYY-MM")
    cpi = _as_float_or_none(food_cpi)
    if cpi is None or cpi <= 0:
        raise ValueError(f"food_cpi inválido {food_cpi!r}: debe ser numérico > 0")
    execute_sql_write(
        """
        INSERT INTO price_inflation_index (period, food_cpi, source, note)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (period) DO UPDATE
          SET food_cpi = EXCLUDED.food_cpi,
              source   = EXCLUDED.source,
              note     = EXCLUDED.note,
              ingested_at = now()
        """,
        (period, cpi, (source or "bcrd"), note),
    )
    logger.info(f"[P2-PRICES-ENGINE-1] índice ingerido: {period} food_cpi={cpi} source={source or 'bcrd'}")
    return {"period": period, "food_cpi": cpi}


def latest_index() -> Optional[dict]:
    """Último período ingerido (mayor YYYY-MM). None si la serie está vacía."""
    row = execute_sql_query(
        "SELECT period, food_cpi FROM price_inflation_index ORDER BY period DESC LIMIT 1",
        fetch_one=True,
    )
    if not row:
        return None
    return {"period": row["period"], "food_cpi": _as_float_or_none(row["food_cpi"])}


def _index_map() -> dict[str, float]:
    rows = execute_sql_query("SELECT period, food_cpi FROM price_inflation_index", fetch_all=True) or []
    out: dict[str, float] = {}
    for r in rows:
        cpi = _as_float_or_none(r.get("food_cpi"))
        if r.get("period") and cpi:
            out[r["period"]] = cpi
    return out


# ── Reescala vivo = base × factor (núcleo del cron) ──────────────────────────
def recompute_adjusted_prices(*, force: bool = False) -> dict:
    """Reescala `price_per_lb`/`price_per_unit` (vivo) desde la base × factor de
    inflación. Idempotente: sólo escribe filas cuyo valor efectivo cambia (evita
    bloat en master_ingredients, tabla read-heavy). No-op si el feature está OFF
    (salvo `force=True`) o no hay índice ingerido.
    """
    if not force and not prices_enabled():
        return {"status": "disabled", "updated": 0, "skipped": 0}

    latest = latest_index()
    if not latest or not latest.get("food_cpi"):
        logger.warning("[P2-PRICES-ENGINE-1] recompute: sin índice ingerido — nada que reescalar.")
        return {"status": "no_index", "updated": 0, "skipped": 0}

    idx_map = _index_map()
    latest_cpi = float(latest["food_cpi"])
    max_factor = _max_inflation_factor()

    rows = execute_sql_query(
        """
        SELECT id, name, price_per_lb, price_per_unit,
               price_per_lb_base, price_per_unit_base, price_base_period
        FROM master_ingredients
        WHERE price_per_lb_base IS NOT NULL OR price_per_unit_base IS NOT NULL
        """,
        fetch_all=True,
    ) or []

    updated = skipped = 0
    for r in rows:
        base_period = r.get("price_base_period")
        base_cpi = idx_map.get(base_period) if base_period else None
        if not base_cpi:
            # Sin período base (o su índice no fue ingerido) no podemos proyectar.
            skipped += 1
            continue
        factor = clamp_factor(latest_cpi / float(base_cpi), max_factor)
        new_lb = adjusted_price(r.get("price_per_lb_base"), factor)
        new_unit = adjusted_price(r.get("price_per_unit_base"), factor)
        cur_lb = _as_float_or_none(r.get("price_per_lb"))
        cur_unit = _as_float_or_none(r.get("price_per_unit"))
        # Valor efectivo tras COALESCE (no pisamos columnas sin base).
        eff_lb = new_lb if new_lb is not None else cur_lb
        eff_unit = new_unit if new_unit is not None else cur_unit
        if eff_lb == cur_lb and eff_unit == cur_unit:
            continue  # idempotente — sin cambio efectivo, no escribir
        execute_sql_write(
            """
            UPDATE master_ingredients
               SET price_per_lb   = COALESCE(%s, price_per_lb),
                   price_per_unit = COALESCE(%s, price_per_unit),
                   price_adjusted_at = now()
             WHERE id = %s
            """,
            (new_lb, new_unit, r["id"]),
        )
        updated += 1

    result = {
        "status": "ok",
        "updated": updated,
        "skipped": skipped,
        "n_priced": len(rows),
        "latest_period": latest["period"],
    }
    logger.info(f"[P2-PRICES-ENGINE-1] recompute: {result}")
    return result


# ── Ingesta de precios base online (importer) ────────────────────────────────
def import_base_prices(
    rows: list[dict], *, default_period: Optional[str] = None, default_source: Optional[str] = None
) -> dict:
    """Puebla/actualiza precios BASE en master_ingredients desde datos online.

    Cada `row` puede traer: `slug` o `name` (al menos uno, para hacer match),
    `price_per_lb_base` (o `price_per_lb`), `price_per_unit_base` (o `price_per_unit`),
    `price_base_period` (YYYY-MM), `price_source`, `price_confidence`, `price_captured_at`.

    Match: por `slug` exacto si viene; si no, por `name` case-insensitive. COALESCE
    → sólo pisa lo que el row provee (no borra datos previos con NULLs). Devuelve
    conteos + lista de no-matcheados para que el caller los revise.
    """
    matched = unmatched = errors = 0
    unmatched_keys: list[str] = []
    for row in rows:
        slug = (row.get("slug") or "").strip()
        name = (row.get("name") or "").strip()
        if not slug and not name:
            errors += 1
            continue
        lb_base = row.get("price_per_lb_base", row.get("price_per_lb"))
        unit_base = row.get("price_per_unit_base", row.get("price_per_unit"))
        period = (row.get("price_base_period") or default_period or "")
        period = str(period).strip() or None
        if period and not _PERIOD_RE.match(period):
            logger.warning(f"[P2-PRICES-ENGINE-1] import: período inválido {period!r} para {slug or name}, ignorado")
            period = None
        source = (row.get("price_source") or default_source or _default_price_source())
        confidence = row.get("price_confidence")
        if confidence is not None and str(confidence).lower() not in _VALID_CONFIDENCE:
            confidence = None
        captured = row.get("price_captured_at")

        params = (
            _as_float_or_none(lb_base), _as_float_or_none(unit_base),
            period, source,
            (str(confidence).lower() if confidence else None),
            captured,
        )
        set_clause = """
               SET price_per_lb_base   = COALESCE(%s, price_per_lb_base),
                   price_per_unit_base = COALESCE(%s, price_per_unit_base),
                   price_base_period   = COALESCE(%s, price_base_period),
                   price_source        = COALESCE(%s, price_source),
                   price_confidence    = COALESCE(%s, price_confidence),
                   price_captured_at   = COALESCE(%s, price_captured_at)
        """
        try:
            if slug:
                res = execute_sql_write(
                    f"UPDATE master_ingredients {set_clause} WHERE slug = %s RETURNING id",
                    (*params, slug), returning=True,
                )
            else:
                res = execute_sql_write(
                    f"UPDATE master_ingredients {set_clause} WHERE lower(name) = lower(%s) RETURNING id",
                    (*params, name), returning=True,
                )
        except Exception as e:
            logger.warning(f"[P2-PRICES-ENGINE-1] import error {slug or name}: {type(e).__name__}: {e}")
            errors += 1
            continue
        if res:
            matched += len(res)
        else:
            unmatched += 1
            unmatched_keys.append(slug or name)

    result = {"matched": matched, "unmatched": unmatched, "errors": errors, "unmatched_keys": unmatched_keys}
    logger.info(f"[P2-PRICES-ENGINE-1] import_base_prices: matched={matched} unmatched={unmatched} errors={errors}")
    return result


# ── Observabilidad ───────────────────────────────────────────────────────────
def price_staleness_report() -> dict:
    """Cobertura + staleness del catálogo de precios. Para el cron / endpoint admin."""
    staleness_days = _staleness_days()
    row = execute_sql_query(
        """
        SELECT
          COUNT(*) AS total,
          COUNT(*) FILTER (WHERE price_per_lb_base IS NOT NULL OR price_per_unit_base IS NOT NULL) AS priced,
          COUNT(*) FILTER (WHERE price_captured_at IS NOT NULL
                             AND price_captured_at < (now()::date - %s)) AS stale
        FROM master_ingredients
        """,
        (staleness_days,),
        fetch_one=True,
    ) or {}
    latest = latest_index()
    total = int(row.get("total") or 0)
    priced = int(row.get("priced") or 0)
    return {
        "total": total,
        "priced": priced,
        "missing": total - priced,
        "stale": int(row.get("stale") or 0),
        "coverage_pct": round(priced / total, 3) if total else 0.0,
        "staleness_days": staleness_days,
        "latest_index_period": latest["period"] if latest else None,
        "enabled": prices_enabled(),
    }


# Warm-register de knobs en el _KNOBS_REGISTRY al import (para que el inventario
# de startup los liste aunque el cron no haya corrido aún).
def _register_price_knobs() -> None:
    try:
        prices_enabled(); _staleness_days(); adjust_interval_h()
        _max_inflation_factor(); _default_price_source()
    except Exception:
        pass


_register_price_knobs()
