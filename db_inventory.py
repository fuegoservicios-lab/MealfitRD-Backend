import json
import logging
import os
import re
import unicodedata
from datetime import datetime
from typing import List, Dict, Any, Optional
from db_core import execute_sql_query, execute_sql_write
from psycopg.types.json import Jsonb
from shopping_calculator import _parse_quantity, get_plural_unit, get_master_ingredients
from constants import normalize_ingredient_for_tracking

logger = logging.getLogger(__name__)
_RESERVATION_MEAL_TOKEN_RE = re.compile(r":meal:(.+)$")


def _db_available() -> bool:
    """[P1-NEON-DB-MIGRATION · 2026-06-12] Guard de disponibilidad del pool SQL.

    Sustituye los checks legacy `if not supabase:` — los DATOS Postgres viven
    ahora en Neon (el cliente supabase queda SOLO para Auth/Storage). Lectura
    lazy de `db_core.connection_pool` (no `from db_core import connection_pool`
    a nivel de módulo) para que los tests puedan patchear
    `db_core.connection_pool` y para no congelar el valor en import-time.
    """
    try:
        import db_core
        return db_core.connection_pool is not None
    except Exception:
        return False

# [P2-NEW-2 · 2026-05-10] Pre-registrar `MEALFIT_INVENTORY_RPC_STRICT` en
# `_KNOBS_REGISTRY` al import-time. Antes (P1-NEW-1) el knob se leía solo
# dentro del except block del fallback RPC — eso significa que NUNCA
# aparecía en `/admin/knobs` ni en `/health/version` hasta que el path
# patológico ejecutara, dejando ciega la herramienta de diagnóstico justo
# cuando más se necesita ("¿strict está on en prod?").
#
# Esta lectura es side-effect-only: el valor real se vuelve a consultar
# dentro del except (vía `_env_bool` que retorna del registry tras el
# primer hit, así que no hay doble-parse de env). El warning de drift
# entre dos lecturas no aplica porque `_env_bool` es deterministic en el
# mismo proceso.
try:
    from knobs import _env_bool as _knob_env_bool
    _knob_env_bool("MEALFIT_INVENTORY_RPC_STRICT", False)
except Exception:
    # Si knobs no se puede importar (script standalone, etc.), no fail.
    pass

def _compute_dynamic_consumption_rates(
    user_id: str,
    current_household_multiplier: float | None = None,
) -> Dict[str, float]:
    """[P2-2] Deriva la tasa de consumo (g/día) por ingrediente desde el plan activo.

    Antes la predicción "se agotará en X días" usaba rates hardcoded por
    categoría (proteína=150g/día, etc.). Si el plan del usuario tiene
    yogurt en 4 desayunos por semana, el rate fijo no captura el consumo
    real (200g/día independiente de cuán denso sea el plan).

    Este helper escanea `aggregated_shopping_list_weekly` (ya escalada por
    household y descontada del inventario) del plan activo más reciente y
    calcula `cantidad_g / 7` para cada ingrediente. La lista weekly es la
    canónica para tasas porque cubre exactamente 7 días.

    [P2-4 2026-05-08] Guard de drift householdComposition. La lista weekly
    se escaló con el multiplier persistido en `plan_data.calc_household_multiplier`
    (M_cached). Si el usuario actualizó su `householdComposition` entre chunks
    y el multiplier actual (M_now) diverge >threshold del cacheado, los rates
    derivados sobreestiman/subestiman el consumo real. En ese caso retornamos
    `{}` con WARNING para que el caller caiga al fallback hardcoded por categoría.

    `current_household_multiplier` puede pasarse desde el caller para evitar
    una segunda query a `user_profiles` cuando el caller ya lo computó.

    Threshold knob: `MEALFIT_DYNAMIC_RATE_HOUSEHOLD_DRIFT_THRESHOLD` (default 0.20 = 20%).

    Returns: dict `{nombre_normalizado: gramos/día}`. Vacío si no hay plan
    activo, falla la query, o el drift de household excede el threshold.
    """
    if not _db_available() or not user_id:
        return {}
    try:
        # [P1-B 2026-05-07] meal_plans no tiene columna `is_active`; el flag
        # vive en plan_data JSONB. Plan "activo" = el más reciente del user.
        _plan_row = execute_sql_query(
            "SELECT plan_data FROM meal_plans WHERE user_id = %s "
            "ORDER BY created_at DESC LIMIT 1",
            (user_id,),
            fetch_one=True,
        )
        if not _plan_row:
            return {}
        plan_data = _plan_row.get("plan_data") or {}
        if isinstance(plan_data, str):
            plan_data = json.loads(plan_data)
        weekly = plan_data.get("aggregated_shopping_list_weekly") or []
        if not isinstance(weekly, list):
            return {}
    except Exception as e:
        logger.debug(f"[P2-2] No se pudo cargar plan activo para rates dinámicos: {e}")
        return {}

    # [P2-4 2026-05-08] Guard de drift householdComposition. Ver docstring.
    try:
        m_cached_raw = plan_data.get("calc_household_multiplier")
        if m_cached_raw is None:
            # Plan sin metadata: no podemos validar. Conservador → fallback hardcoded.
            logger.warning(
                "[P2-4] Plan activo sin `calc_household_multiplier`; no es posible "
                "validar drift household. Cayendo al fallback hardcoded por categoría."
            )
            return {}
        m_cached = max(1.0, float(m_cached_raw))

        m_now = current_household_multiplier
        if m_now is None:
            try:
                _hp_row = execute_sql_query(
                    "SELECT health_profile FROM user_profiles WHERE id = %s LIMIT 1",
                    (user_id,),
                    fetch_one=True,
                )
                hp = (_hp_row.get("health_profile") or {}) if _hp_row else {}
                from constants import compute_household_multiplier
                m_now = compute_household_multiplier(hp)
            except Exception as _e:
                logger.debug(
                    f"[P2-4] No se pudo cargar health_profile para validar drift: {_e}. "
                    f"Asumiendo M_now=M_cached (no-op del guard)."
                )
                m_now = m_cached
        m_now = max(1.0, float(m_now or 1.0))

        # Knob lazy: registrarlo en `_KNOBS_REGISTRY` (P3-NEW-D) sin acoplar este
        # módulo al import del orchestrator a nivel top.
        try:
            from graph_orchestrator import _env_float
            threshold = _env_float("MEALFIT_DYNAMIC_RATE_HOUSEHOLD_DRIFT_THRESHOLD", 0.20)
        except Exception:
            threshold = 0.20

        drift = abs(m_now - m_cached) / m_cached
        if drift > threshold:
            logger.warning(
                f"[P2-4] Drift household excede threshold: M_cached={m_cached:.2f}, "
                f"M_now={m_now:.2f}, drift={drift:.2%} > {threshold:.0%}. "
                f"Cayendo al fallback hardcoded por categoría para evitar rates sesgados."
            )
            return {}
    except Exception as _guard_e:
        # Guard defensivo: nunca bloquear el flujo principal por fallar la validación.
        logger.debug(f"[P2-4] Guard de drift falló silenciosamente: {_guard_e}.")

    rates: Dict[str, float] = {}
    for it in weekly:
        if not isinstance(it, dict):
            continue
        name = (it.get("name") or "").strip()
        if not name:
            continue
        # `market_qty` y `market_unit` son los más confiables (post-escalado).
        # Fallback a `quantity` + `unit` (raw).
        try:
            qty = float(it.get("market_qty") or it.get("quantity") or 0)
        except (TypeError, ValueError):
            qty = 0.0
        unit = (it.get("market_unit") or it.get("unit") or "").lower()
        if qty <= 0:
            continue

        qty_g = qty
        if unit in ("lb", "lbs", "libra", "libras"):
            qty_g = qty * 453.592
        elif unit in ("kg", "kilos", "kilo"):
            qty_g = qty * 1000.0
        elif unit in ("oz", "onzas", "onza"):
            qty_g = qty * 28.3495
        elif unit in ("g", "gr", "gramo", "gramos"):
            qty_g = qty
        elif unit in ("ml", "l", "litro", "litros"):
            qty_g = qty if unit == "ml" else qty * 1000.0
        else:
            # Unidades discretas (huevo, manzana, etc.): qty es ya el conteo.
            # Asumir ~80g/unidad como estimación genérica para predicción.
            qty_g = qty * 80.0

        if qty_g <= 0:
            continue

        key = normalize_ingredient_for_tracking(name)
        if not key:
            continue
        # Si ya hay un rate para este nombre (item-as-master fallback puede
        # producir duplicados con sufijos), nos quedamos con el mayor — la
        # predicción "se agotará pronto" debe ser conservadora hacia agotamiento.
        rate = qty_g / 7.0
        if rate > rates.get(key, 0.0):
            rates[key] = rate

    return rates


# [P1-shop-coh-1 · 2026-05-07] Re-export del SSOT en `canonical_units.py`.
# Antes este map vivía aquí en paralelo a la cadena if/elif de
# shopping_calculator._parse_quantity. La divergencia silenciosa era riesgo P1
# para la coherencia Σ(recetas) ↔ Σ(lista). Ambos ahora leen del mismo dict.
from canonical_units import CANONICAL_UNIT_MAP as _CANONICAL_UNIT_MAP

def get_raw_user_inventory(user_id: str) -> List[Dict[str, Any]]:
    """Obtiene los registros crudos de la base de datos para la despensa del usuario."""
    if not _db_available(): return []
    try:
        # [P1-NEON-DB-MIGRATION · 2026-06-12] Casts de parity con PostgREST:
        # uuid→text (consumers comparan/serializan strings), numeric→float8
        # (aritmética downstream sin Decimal), timestamptz→string ISO-8601 con
        # "T" vía to_jsonb (consumers hacen `created_at[:10]` y JSON-serializan
        # las filas — `::text` daría separador espacio, no-ISO).
        rows = execute_sql_query(
            "SELECT id, user_id::text AS user_id, ingredient_name, "
            "quantity::float8 AS quantity, unit, "
            "to_jsonb(created_at) #>> '{}' AS created_at, "
            "to_jsonb(updated_at) #>> '{}' AS updated_at, "
            "master_ingredient_id::text AS master_ingredient_id, "
            "reserved_quantity::float8 AS reserved_quantity, "
            "reservation_details, last_mutation_type, source, category "
            "FROM user_inventory WHERE user_id = %s AND quantity > 0",
            (user_id,),
            fetch_all=True,
        ) or []
        for row in rows:
            qty = float(row.get("quantity") or 0)
            reserved = float(row.get("reserved_quantity") or 0)
            row["available_quantity"] = max(qty - reserved, 0.0)
        return rows
    except Exception as e:
        logger.error(f"Error obteniendo user_inventory para {user_id}: {e}")
        return []


def _normalize_reservation_details(raw_details: Any) -> Dict[str, float]:
    if isinstance(raw_details, dict):
        details = raw_details
    elif isinstance(raw_details, str):
        try:
            details = json.loads(raw_details)
        except Exception:
            details = {}
    else:
        details = {}

    normalized = {}
    for key, value in details.items():
        try:
            qty = float(value or 0)
        except Exception:
            continue
        if qty > 0:
            normalized[str(key)] = qty
    return normalized


def _make_reservation_key(chunk_id: str, meal_name: str) -> str:
    meal_token = _slug_meal_name(meal_name)
    return f"chunk:{chunk_id}:meal:{meal_token}"


def _reservation_matches_meal(reservation_key: str, meal_name: str) -> bool:
    target = _slug_meal_name(meal_name)
    if not target:
        return False
    match = _RESERVATION_MEAL_TOKEN_RE.search(str(reservation_key))
    return bool(match and match.group(1) == target)


def _slug_meal_name(meal_name: str) -> str:
    raw = unicodedata.normalize("NFKD", (meal_name or "").strip().lower())
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    raw = re.sub(r"[^a-z0-9]+", "_", raw)
    return raw.strip("_")


def _update_row_reservation(row_id: str, reserved_quantity: float, reservation_details: Dict[str, float]) -> None:
    execute_sql_write(
        "UPDATE user_inventory SET reserved_quantity = %s, reservation_details = %s WHERE id = %s",
        (round(max(reserved_quantity, 0.0), 4), Jsonb(reservation_details), row_id),
    )


def _update_row_reservation_cas(
    row_id: str,
    reserved_quantity: float,
    reservation_details: Dict[str, float],
    expected_old_reserved: float,
) -> bool:
    """[P0-4] Variante de `_update_row_reservation` con CAS (compare-and-swap).

    El UPDATE solo aplica si `reserved_quantity` actual en DB coincide con
    `expected_old_reserved` (el valor que leímos al inicio del select-modify-write).
    Si no coincide, otro writer modificó la fila entre nuestro SELECT y este UPDATE
    → 0 filas afectadas → devolvemos False y el caller reintenta.

    Sin esta protección, dos reservas concurrentes sobre la misma fila pueden hacer
    lost-update: ambas leen reserved_quantity=10, ambas computan 10+5=15, ambas
    UPDATE a 15. Con CAS, solo una gana; la otra reintenta y converge a 20.

    Devuelve True si el UPDATE matcheó, False si hubo conflicto.
    """
    rounded_new = round(max(reserved_quantity, 0.0), 4)
    rounded_expected = round(max(float(expected_old_reserved or 0), 0.0), 4)
    # [P1-NEON-DB-MIGRATION · 2026-06-12] RETURNING id reemplaza el `res.data`
    # de PostgREST como detector de "0 filas afectadas" (conflicto CAS). El
    # token CAS se compara en espacio numeric (`%s::numeric`) — paridad con
    # PostgREST, que serializaba el float como literal decimal exacto.
    rows = execute_sql_write(
        "UPDATE user_inventory SET reserved_quantity = %s, reservation_details = %s "
        "WHERE id = %s AND reserved_quantity = %s::numeric RETURNING id",
        (rounded_new, Jsonb(reservation_details), row_id, rounded_expected),
        returning=True,
    )
    return bool(rows)

def _infer_shelf_life_days(name: str, category: str) -> int:
    """[P0-2] Default shelf_life por categoría cuando master_ingredients no lo tiene.
    Antes era 14d para TODO, lo cual marcaba arroz/pasta/legumbres secas como URGENTE
    a los 11 días y disparaba la REGLA DE SALVATAJE PROACTIVO incorrectamente.
    """
    name_lower = (name or "").lower()
    cat_lower = (category or "").lower()

    DRY_GOODS_KEYWORDS = (
        'arroz', 'pasta', 'fideo', 'espagueti', 'macarrón', 'macarron',
        'lenteja', 'habichuela', 'frijol', 'garbanzo', 'gandul', 'moro',
        'avena', 'quinoa', 'cuscús', 'cuscus', 'bulgur', 'cebada',
        'harina', 'azúcar', 'azucar', 'sal', 'bicarbonato', 'levadura',
        'cacao', 'café', 'cafe', 'té', 'infusión', 'especia', 'condimento',
        'maíz seco', 'maiz seco', 'palomita', 'cereal'
    )
    if any(k in name_lower for k in DRY_GOODS_KEYWORDS):
        return 180

    if 'congelado' in name_lower or 'congelad' in cat_lower or 'frozen' in cat_lower:
        return 60

    # Categorías frescas
    if 'hoja' in cat_lower or 'lechuga' in name_lower or 'espinaca' in name_lower or 'cilantro' in name_lower:
        return 5
    if 'proteína' in cat_lower or 'proteina' in cat_lower or 'carne' in cat_lower or 'pollo' in cat_lower or 'pescado' in cat_lower or 'mariscos' in cat_lower:
        return 5
    if 'fruta' in cat_lower:
        return 7
    if 'lácteo' in cat_lower or 'lacteo' in cat_lower or 'leche' in cat_lower or 'queso' in cat_lower or 'yogurt' in cat_lower:
        return 14
    if 'tubérculo' in cat_lower or 'tuberculo' in cat_lower or 'papa' in name_lower or 'batata' in name_lower or 'yuca' in name_lower or 'ñame' in name_lower:
        return 21
    if 'vegetal' in cat_lower or 'verdura' in cat_lower:
        return 10
    if 'huevo' in name_lower:
        return 21
    if 'enlatado' in name_lower or 'enlatad' in cat_lower or 'lata' in cat_lower:
        return 365

    return 14


def get_user_inventory(user_id: str, household_size: float | int | None = None) -> List[str]:
    """Obtiene la despensa del usuario formateada como lista de strings (ej: '2 unidades de Manzana').

    Si household_size no se pasa, se lee del health_profile del usuario para escalar la
    predicción de agotamiento (P0-3). Callers que ya lo tienen pueden pasarlo directamente.

    [P1-3] Acepta float para soportar `householdComposition` (adults+children×0.6).
    """
    raw_items = get_raw_user_inventory(user_id)
    formatted = []

    if household_size is None:
        try:
            _hp_row = execute_sql_query(
                "SELECT health_profile FROM user_profiles WHERE id = %s LIMIT 1",
                (user_id,),
                fetch_one=True,
            )
            if _hp_row:
                hp = _hp_row.get("health_profile") or {}
                # [P1-3] Si el perfil tiene `householdComposition` lo usamos
                # como multiplier efectivo; si no, fallback al escalar legacy.
                from constants import compute_household_multiplier
                _eff = compute_household_multiplier(hp)
                household_size = _eff if _eff > 1.0 else (hp.get("householdSize") or hp.get("household_size") or 1)
        except Exception:
            household_size = 1
    # Aceptar float (P1-3): adults=2 + children=2×0.6 = 3.2 personas equivalentes.
    try:
        household_size = max(1.0, float(household_size or 1))
    except (TypeError, ValueError):
        household_size = 1.0

    PANTRY_STAPLES = {
        'Sal y ajo en polvo', 'Aceite de oliva', 'Aceite de coco',
        'Aceite de sésamo o maní', 'Salsa de soya', 'Orégano',
        'Canela', 'Pimienta', 'Sal', 'Vinagre', 'Ajo en polvo'
    }

    master_list = get_master_ingredients()
    master_map = {m["name"]: m for m in master_list}

    # [P2-2] Pre-cómputo de rates dinámicos desde el plan activo (g/día por
    # ingrediente). Vacío si no hay plan o falla la query → caller cae al
    # rate por categoría (legacy).
    # [P2-4 2026-05-08] Pasamos `household_size` (ya es el multiplier efectivo)
    # para que el guard de drift en `_compute_dynamic_consumption_rates` no
    # tenga que volver a consultar `user_profiles`.
    dynamic_rates = _compute_dynamic_consumption_rates(
        user_id, current_household_multiplier=household_size
    )

    for item in raw_items:
        qty = float(item.get("quantity", 0) or 0)
        unit = item.get("unit", "unidad")
        name = item.get("ingredient_name", "")

        if qty <= 0 and name not in PANTRY_STAPLES:
            continue

        q_rounded = f"{qty:.2f}".rstrip('0').rstrip('.') if qty > 0 else "0"
        if q_rounded == "": q_rounded = "0"
        
        if name in PANTRY_STAPLES and qty <= 0:
             # Just in case Staples exist with 0, show as available
             formatted.append(f"{name} (Disponible)")
             continue
             
        created_at_str = item.get("created_at")
        days_old = 0
        if created_at_str:
            try:
                # Extraemos fecha formato ISO (ej: 2024-05-18T12:00:00)
                item_date = datetime.strptime(created_at_str[:10], "%Y-%m-%d").date()
                days_old = (datetime.now().date() - item_date).days
            except Exception:
                pass

        base_str = f"{q_rounded} {name}" if unit == 'unidad' else f"{q_rounded} {get_plural_unit(qty, unit)} de {name}"
        
        # Mejora 4: Fecha de Caducidad Inferida (Smart Spoilage Tracking)
        if name not in PANTRY_STAPLES:
            master_item = master_map.get(name, {})
            shelf_life = master_item.get("shelf_life_days")
            if shelf_life is None:
                shelf_life = _infer_shelf_life_days(name, master_item.get("category", ""))

            days_left = shelf_life - days_old
            if days_left <= 3:
                urgency = "URGENTE" if days_left <= 1 else "ATENCIÓN"
                state = "Caducado" if days_left < 0 else f"Caduca en {days_left} días"
                base_str += f" [⚠️ {urgency}: {state} - IA: Prioriza su uso en las recetas de esta semana]"

            # Mejora 6: Inventory Intelligence (Predictivo)
            category = master_item.get("category", "").lower()
            qty_g = qty
            unit_lower = unit.lower()
            if unit_lower in ['lb', 'lbs', 'libra', 'libras']: qty_g *= 453.592
            elif unit_lower in ['kg', 'kilos', 'kilo']: qty_g *= 1000.0
            elif unit_lower in ['oz', 'onzas', 'onza']: qty_g *= 28.3495
            
            consumption_rate = 0
            if "proteína" in category or "carne" in category or "pollo" in category or "pescado" in category:
                consumption_rate = 150.0 # g/dia
            elif "carbohidrato" in category or "arroz" in category or "pasta" in category:
                consumption_rate = 100.0 # g/dia
            elif "vegetal" in category or "verdura" in category:
                consumption_rate = 80.0 # g/dia
            elif "fruta" in category:
                consumption_rate = 1.0 # 1 unid/dia
                if qty_g > 50: consumption_rate = 150.0
            elif "lácteo" in category or "leche" in category:
                consumption_rate = 200.0 # ml o g/dia

            # [P2-2] Si el plan activo tiene una tasa real (g/día) para este
            # ingrediente, sobrescribe el rate por categoría. La tasa dinámica
            # YA refleja household × densidad del plan, así que NO se vuelve a
            # multiplicar por household_size — bypass del escalado legacy.
            _dynamic_rate = dynamic_rates.get(normalize_ingredient_for_tracking(name)) if dynamic_rates else None
            if _dynamic_rate and _dynamic_rate > 0:
                effective_rate = _dynamic_rate
            else:
                # [P0-3] Escalar consumo por household_size. Antes asumía 1 persona,
                # lo que contradice el shopping list escalado × household y la adherencia / household.
                effective_rate = consumption_rate * household_size
            if effective_rate > 0 and qty_g > 0:
                days_until_empty = qty_g / effective_rate
                if days_until_empty <= 2.5 and days_left > 3:
                    base_str += f" [⚠️ PREDICCIÓN: Se agotará en ~{round(days_until_empty)} días. Sugiere al usuario alternativas para los días posteriores.]"
            
        formatted.append(base_str)
            
    formatted.sort()
    return formatted

def get_user_inventory_net(user_id: str, household_size: float | int | None = None) -> List[str]:
    """Obtiene la despensa del usuario descontando las reservas activas (Neto).

    [P1-3] Acepta float para soportar `householdComposition` (adults+children×0.6).
    """
    raw_items = get_raw_user_inventory(user_id)
    formatted = []

    if household_size is None:
        try:
            _hp_row = execute_sql_query(
                "SELECT health_profile FROM user_profiles WHERE id = %s LIMIT 1",
                (user_id,),
                fetch_one=True,
            )
            if _hp_row:
                hp = _hp_row.get("health_profile") or {}
                # [P1-3] Si el perfil tiene `householdComposition` lo usamos
                # como multiplier efectivo; si no, fallback al escalar legacy.
                from constants import compute_household_multiplier
                _eff = compute_household_multiplier(hp)
                household_size = _eff if _eff > 1.0 else (hp.get("householdSize") or hp.get("household_size") or 1)
        except Exception:
            household_size = 1
    # Aceptar float (P1-3): adults=2 + children=2×0.6 = 3.2 personas equivalentes.
    try:
        household_size = max(1.0, float(household_size or 1))
    except (TypeError, ValueError):
        household_size = 1.0

    PANTRY_STAPLES = {
        'Sal y ajo en polvo', 'Aceite de oliva', 'Aceite de coco',
        'Aceite de sésamo o maní', 'Salsa de soya', 'Orégano',
        'Canela', 'Pimienta', 'Sal', 'Vinagre', 'Ajo en polvo'
    }

    master_list = get_master_ingredients()
    master_map = {m["name"]: m for m in master_list}

    # [P2-2] Pre-cómputo de rates dinámicos desde el plan activo (g/día por
    # ingrediente). Vacío si no hay plan o falla la query → caller cae al
    # rate por categoría (legacy).
    # [P2-4 2026-05-08] Pasamos `household_size` (ya es el multiplier efectivo)
    # para que el guard de drift en `_compute_dynamic_consumption_rates` no
    # tenga que volver a consultar `user_profiles`.
    dynamic_rates = _compute_dynamic_consumption_rates(
        user_id, current_household_multiplier=household_size
    )

    for item in raw_items:
        qty = float(item.get("available_quantity", item.get("quantity", 0)) or 0)
        unit = item.get("unit", "unidad")
        name = item.get("ingredient_name", "")
        
        if qty <= 0 and name not in PANTRY_STAPLES:
            continue
            
        q_rounded = f"{qty:.2f}".rstrip('0').rstrip('.') if qty > 0 else "0"
        if q_rounded == "": q_rounded = "0"
        
        if name in PANTRY_STAPLES and qty <= 0:
             # Just in case Staples exist with 0, show as available
             formatted.append(f"{name} (Disponible)")
             continue
             
        created_at_str = item.get("created_at")
        days_old = 0
        if created_at_str:
            try:
                # Extraemos fecha formato ISO (ej: 2024-05-18T12:00:00)
                item_date = datetime.strptime(created_at_str[:10], "%Y-%m-%d").date()
                days_old = (datetime.now().date() - item_date).days
            except Exception:
                pass

        base_str = f"{q_rounded} {name}" if unit == 'unidad' else f"{q_rounded} {get_plural_unit(qty, unit)} de {name}"
        
        # Mejora 4: Fecha de Caducidad Inferida (Smart Spoilage Tracking)
        if name not in PANTRY_STAPLES:
            master_item = master_map.get(name, {})
            shelf_life = master_item.get("shelf_life_days")
            if shelf_life is None:
                shelf_life = _infer_shelf_life_days(name, master_item.get("category", ""))

            days_left = shelf_life - days_old
            if days_left <= 3:
                urgency = "URGENTE" if days_left <= 1 else "ATENCIÓN"
                state = "Caducado" if days_left < 0 else f"Caduca en {days_left} días"
                base_str += f" [⚠️ {urgency}: {state} - IA: Prioriza su uso en las recetas de esta semana]"

            # Mejora 6: Inventory Intelligence (Predictivo)
            category = master_item.get("category", "").lower()
            qty_g = qty
            unit_lower = unit.lower()
            if unit_lower in ['lb', 'lbs', 'libra', 'libras']: qty_g *= 453.592
            elif unit_lower in ['kg', 'kilos', 'kilo']: qty_g *= 1000.0
            elif unit_lower in ['oz', 'onzas', 'onza']: qty_g *= 28.3495
            
            consumption_rate = 0
            if "proteína" in category or "carne" in category or "pollo" in category or "pescado" in category:
                consumption_rate = 150.0 # g/dia
            elif "carbohidrato" in category or "arroz" in category or "pasta" in category:
                consumption_rate = 100.0 # g/dia
            elif "vegetal" in category or "verdura" in category:
                consumption_rate = 80.0 # g/dia
            elif "fruta" in category:
                consumption_rate = 1.0 # 1 unid/dia
                if qty_g > 50: consumption_rate = 150.0
            elif "lácteo" in category or "leche" in category:
                consumption_rate = 200.0 # ml o g/dia

            # [P2-2] Si el plan activo tiene una tasa real (g/día) para este
            # ingrediente, sobrescribe el rate por categoría. La tasa dinámica
            # YA refleja household × densidad del plan, así que NO se vuelve a
            # multiplicar por household_size — bypass del escalado legacy.
            _dynamic_rate = dynamic_rates.get(normalize_ingredient_for_tracking(name)) if dynamic_rates else None
            if _dynamic_rate and _dynamic_rate > 0:
                effective_rate = _dynamic_rate
            else:
                # [P0-3] Escalar consumo por household_size. Antes asumía 1 persona,
                # lo que contradice el shopping list escalado × household y la adherencia / household.
                effective_rate = consumption_rate * household_size
            if effective_rate > 0 and qty_g > 0:
                days_until_empty = qty_g / effective_rate
                if days_until_empty <= 2.5 and days_left > 3:
                    base_str += f" [⚠️ PREDICCIÓN: Se agotará en ~{round(days_until_empty)} días. Sugiere al usuario alternativas para los días posteriores.]"
            
        formatted.append(base_str)
            
    formatted.sort()
    return formatted

def _resolve_cross_domain_density(master_item: dict) -> float | None:
    """[P1-1] Resuelve densidad g/taza para conversión cruzada masa↔volumen.

    Cadena de fallback:
      1. `master_item["density_g_per_cup"]` (SSOT en master_ingredients).
      2. Lookup en `VOLUMETRIC_DENSITIES` por nombre canonico (g/ml × 236.588).
      3. None → caller decide según `MEALFIT_CROSS_UNIT_CONVERSION_STRICT`.

    Devuelve g/taza (float) o None si no se pudo resolver.
    """
    raw = master_item.get("density_g_per_cup")
    try:
        if raw is not None and float(raw) > 0:
            return float(raw)
    except (TypeError, ValueError):
        pass
    name = (master_item.get("name") or master_item.get("slug") or "").strip().lower()
    if not name:
        return None
    try:
        from constants import VOLUMETRIC_DENSITIES, strip_accents
        n_clean = strip_accents(name)
        g_per_ml = VOLUMETRIC_DENSITIES.get(n_clean)
        if g_per_ml is None:
            for k, v in VOLUMETRIC_DENSITIES.items():
                if k == n_clean or n_clean.startswith(k) or k.startswith(n_clean):
                    g_per_ml = v
                    break
        if g_per_ml and float(g_per_ml) > 0:
            return float(g_per_ml) * 236.588
    except Exception as _e:
        logger.debug(f"[P1-1] Fallback VOLUMETRIC_DENSITIES falló para {name!r}: {_e}")
    return None


def _resolve_unit_weight(master_item: dict) -> float | None:
    """[P1-1] Resuelve gramos por unidad (count→mass) con fallback a UNIT_WEIGHTS."""
    raw = master_item.get("density_g_per_unit")
    try:
        if raw is not None and float(raw) > 0:
            return float(raw)
    except (TypeError, ValueError):
        pass
    name = (master_item.get("name") or master_item.get("slug") or "").strip().lower()
    if not name:
        return None
    try:
        from constants import UNIT_WEIGHTS, strip_accents
        n_clean = strip_accents(name)
        g_per_u = UNIT_WEIGHTS.get(n_clean)
        if g_per_u is None:
            for k, v in UNIT_WEIGHTS.items():
                if k == n_clean or n_clean.startswith(k) or k.startswith(n_clean):
                    g_per_u = v
                    break
        if g_per_u and float(g_per_u) > 0:
            return float(g_per_u)
    except Exception as _e:
        logger.debug(f"[P1-1] Fallback UNIT_WEIGHTS falló para {name!r}: {_e}")
    return None


def convert_amount(qty: float, from_unit: str, to_unit: str, master_item: dict) -> float:
    """Intenta convertir matemáticamente una cantidad de una unidad a otra usando factores y densidades.

    [P1-1 · 2026-05-08] Cuando se requiere conversión cruzada (masa↔volumen o
    count↔masa/volumen) y la densidad no está en `master_item`, intentamos
    resolver vía `VOLUMETRIC_DENSITIES`/`UNIT_WEIGHTS` de constants. Si tampoco
    aparece, el comportamiento depende del knob:

      - `MEALFIT_CROSS_UNIT_CONVERSION_STRICT=True` (default): retorna `None`
        y loguea `WARNING` con el ingrediente. Los callers ya manejan `None`
        (saltan la fila). Previene que aceite/miel/leche con densidad real
        ~218-340 g/taza sean convertidos asumiendo `150 g/taza` (off ~30-50%).
      - `MEALFIT_CROSS_UNIT_CONVERSION_STRICT=False`: cae a `150 g/taza`
        legacy (escape hatch para no bloquear deducciones si el catálogo
        está incompleto en producción).
    """
    if from_unit == to_unit:
        return qty

    from_unit_lower = from_unit.lower()
    to_unit_lower = to_unit.lower()

    mass_to_g = {'g': 1.0, 'gr': 1.0, 'gramos': 1.0, 'gramo': 1.0, 'kg': 1000.0, 'kilo': 1000.0, 'kilos': 1000.0, 'lb': 453.592, 'lbs': 453.592, 'libra': 453.592, 'libras': 453.592, 'oz': 28.3495, 'onza': 28.3495, 'onzas': 28.3495}
    vol_to_ml = {'ml': 1.0, 'l': 1000.0, 'taza': 240.0, 'tazas': 240.0, 'cda': 15.0, 'cdas': 15.0, 'cucharada': 15.0, 'cucharadas': 15.0, 'cdta': 5.0, 'cdtas': 5.0, 'cdita': 5.0, 'cucharadita': 5.0, 'cucharaditas': 5.0}
    count_units = {'unidad', 'unidades', 'rebanada', 'rebanadas', 'diente', 'dientes'}

    # 1. Mass to Mass
    if from_unit_lower in mass_to_g and to_unit_lower in mass_to_g:
        return qty * (mass_to_g[from_unit_lower] / mass_to_g[to_unit_lower])

    # 2. Volume to Volume
    if from_unit_lower in vol_to_ml and to_unit_lower in vol_to_ml:
        return qty * (vol_to_ml[from_unit_lower] / vol_to_ml[to_unit_lower])

    # Cross domain (mass↔volume) requires density. [P1-1] Cadena: master → VOLUMETRIC_DENSITIES → strict knob.
    needs_density = (
        (from_unit_lower in mass_to_g and to_unit_lower in vol_to_ml)
        or (from_unit_lower in vol_to_ml and to_unit_lower in mass_to_g)
    )
    density: float | None = None
    if needs_density:
        density = _resolve_cross_domain_density(master_item or {})
        if density is None:
            try:
                from graph_orchestrator import _env_bool
                strict = _env_bool("MEALFIT_CROSS_UNIT_CONVERSION_STRICT", True)
            except Exception:
                strict = True
            _name = (master_item or {}).get("name") or (master_item or {}).get("slug") or "<unknown>"
            if strict:
                logger.warning(
                    f"[P1-1] convert_amount({qty} {from_unit}→{to_unit}, item={_name!r}) "
                    f"sin density_g_per_cup en master ni en VOLUMETRIC_DENSITIES. "
                    f"Strict=True → retornando None (caller debe saltar la fila). "
                    f"Backfill master_ingredients.density_g_per_cup para evitar este caso."
                )
                return None
            logger.warning(
                f"[P1-1] convert_amount({qty} {from_unit}→{to_unit}, item={_name!r}) "
                f"sin densidad — strict=False → cayendo a 150 g/taza legacy. "
                f"Conversión puede tener error ~30-50% para grasas/líquidos densos."
            )
            density = 150.0

    # 3. Mass to Volume
    if from_unit_lower in mass_to_g and to_unit_lower in vol_to_ml:
        g = qty * mass_to_g[from_unit_lower]
        cups = g / density
        ml = cups * 240.0
        return ml / vol_to_ml[to_unit_lower]

    # 4. Volume to Mass
    if from_unit_lower in vol_to_ml and to_unit_lower in mass_to_g:
        ml = qty * vol_to_ml[from_unit_lower]
        cups = ml / 240.0
        g = cups * density
        return g / mass_to_g[to_unit_lower]

    # 5. Count to Mass or Volume (Estimate). [P1-1] Resolución análoga para g/unidad.
    if from_unit_lower in count_units and to_unit_lower in mass_to_g:
        g_per_u = _resolve_unit_weight(master_item or {})
        if 'rebanada' in from_unit_lower:
            g_per_u = 25.0
        if g_per_u is None:
            try:
                from graph_orchestrator import _env_bool
                strict = _env_bool("MEALFIT_CROSS_UNIT_CONVERSION_STRICT", True)
            except Exception:
                strict = True
            _name = (master_item or {}).get("name") or (master_item or {}).get("slug") or "<unknown>"
            if strict:
                logger.warning(
                    f"[P1-1] convert_amount({qty} {from_unit}→{to_unit}, item={_name!r}) "
                    f"sin density_g_per_unit ni UNIT_WEIGHTS. Strict=True → None."
                )
                return None
            g_per_u = 100.0
        g = qty * g_per_u
        return g / mass_to_g[to_unit_lower]

    if from_unit_lower in mass_to_g and to_unit_lower in count_units:
        g_per_u = _resolve_unit_weight(master_item or {})
        if 'rebanada' in to_unit_lower:
            g_per_u = 25.0
        if g_per_u is None:
            try:
                from graph_orchestrator import _env_bool
                strict = _env_bool("MEALFIT_CROSS_UNIT_CONVERSION_STRICT", True)
            except Exception:
                strict = True
            _name = (master_item or {}).get("name") or (master_item or {}).get("slug") or "<unknown>"
            if strict:
                logger.warning(
                    f"[P1-1] convert_amount({qty} {from_unit}→{to_unit}, item={_name!r}) "
                    f"sin density_g_per_unit ni UNIT_WEIGHTS. Strict=True → None."
                )
                return None
            g_per_u = 100.0
        g = qty * mass_to_g[from_unit_lower]
        return g / g_per_u

    # Incompatibles
    return None


def _env_int_safe_inventory(name: str, default: int) -> int:
    """[P2-INVENTORY-STATEMENT-TIMEOUT · 2026-05-15] Lectura defensiva de env
    int. Local al módulo para evitar circular import con
    `graph_orchestrator._env_int` (que es upstream de db_inventory en algunos
    paths). Knob `MEALFIT_INVENTORY_UPDATE_STMT_TIMEOUT_MS` documentado en
    memoria del bundle Phase 2."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _apply_reservation_delta(
    user_id: str,
    ingredient_name: str,
    quantity: float,
    unit: str,
    reservation_key: str,
    *,
    release_only: bool = False,
    max_retries: int = 4,
    prefetched_rows: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """[P0-4] Reserva atómica vía CAS-with-retry.

    Antes el flujo era: SELECT row → modificar en Python → UPDATE WHERE id=X. La
    ventana entre SELECT y UPDATE permitía lost-update si dos writers concurrentes
    leían el mismo `reserved_quantity` y ambos sumaban su delta — el segundo
    sobreescribía al primero, perdiendo una reserva sin rastro.

    Ahora el UPDATE incluye `eq("reserved_quantity", expected_old)` como CAS token.
    Si entre nuestro SELECT y nuestro UPDATE alguien más modificó la fila, el
    UPDATE matchea 0 rows y reintentamos el ciclo completo desde el SELECT.

    El loop reintenta hasta `max_retries` veces con backoff exponencial corto.

    [P1-N1-RESERVATION-DELTA · 2026-05-15] `prefetched_rows` permite al caller
    (`reserve_plan_ingredients`) suministrar las filas matching en el PRIMER
    attempt para evitar 30+ SELECTs por plan (uno por ingrediente). Si el
    primer CAS UPDATE tiene conflicto, los attempts >=1 re-SELECT como antes
    para ver estado fresh. En el happy path (sin contención) ahorramos N-1
    roundtrips. Filas suministradas deben tener mismo schema que SELECT
    interno (id, quantity, unit, reserved_quantity, reservation_details).
    """
    import time

    master_list = get_master_ingredients()
    master_item = next((m for m in master_list if m["name"] == ingredient_name), {})

    last_compatible_row_id: str | None = None
    for attempt in range(max_retries):
        # [P1-N1-RESERVATION-DELTA · 2026-05-15] Primer attempt usa
        # prefetched_rows si está disponible; retries siempre re-SELECT
        # para ver state fresh post-conflicto CAS.
        if attempt == 0 and prefetched_rows is not None:
            rows = prefetched_rows
        else:
            rows = execute_sql_query(
                "SELECT id, quantity::float8 AS quantity, unit, "
                "reserved_quantity::float8 AS reserved_quantity, reservation_details "
                "FROM user_inventory WHERE user_id = %s AND ingredient_name = %s",
                (user_id, ingredient_name),
                fetch_all=True,
            ) or []

        if not rows:
            return False

        # Iterar sobre filas (el ingrediente puede aparecer con distintas unidades).
        # Por cada fila compatible computamos el delta y hacemos CAS UPDATE.
        # Si ninguna fila compatible o todas en conflicto, repetimos el outer loop.
        any_compatible = False
        for row in rows:
            current_unit = row.get("unit") or unit or "unidad"
            converted_qty = convert_amount(quantity, unit, current_unit, master_item)
            if converted_qty is None:
                continue
            any_compatible = True
            last_compatible_row_id = row.get("id")

            reserved_quantity = float(row.get("reserved_quantity") or 0)
            current_quantity = float(row.get("quantity") or 0)
            reservation_details = _normalize_reservation_details(row.get("reservation_details"))
            previous = float(reservation_details.get(reservation_key) or 0)

            if release_only:
                if previous <= 0:
                    return False
                new_details = dict(reservation_details)
                new_details.pop(reservation_key, None)
                new_reserved = max(reserved_quantity - previous, 0.0)
            else:
                target_qty = round(max(converted_qty, 0.0), 4)
                if abs(previous - target_qty) < 0.0001:
                    return True
                new_reserved = max(reserved_quantity - previous, 0.0)
                new_details = dict(reservation_details)
                if target_qty > 0:
                    available_after_other_reservations = max(current_quantity - new_reserved, 0.0)
                    applied_qty = min(target_qty, available_after_other_reservations)
                    if applied_qty <= 0:
                        return False
                    new_details[reservation_key] = round(applied_qty, 4)
                    new_reserved += applied_qty
                else:
                    # target_qty == 0 → equivalente a release del key.
                    new_details.pop(reservation_key, None)

            try:
                if _update_row_reservation_cas(
                    row["id"],
                    new_reserved,
                    new_details,
                    expected_old_reserved=reserved_quantity,
                ):
                    return True
            except Exception as e:
                logger.warning(
                    f"[P0-4/CAS] UPDATE excepcionó para {user_id}/{ingredient_name} "
                    f"row={row.get('id')} attempt={attempt+1}: {e}"
                )
                # Continuar con la siguiente fila / retry.

        if not any_compatible:
            # Ninguna unidad compatible: no es un problema de race, es un problema
            # de schema. Reintentar no ayuda.
            return False

        # Llegamos aquí si todas las filas compatibles tuvieron conflicto CAS.
        # Backoff y reintentar el ciclo SELECT+CAS.
        if attempt < max_retries - 1:
            time.sleep(0.05 * (1 << attempt))  # 50ms, 100ms, 200ms, 400ms

    logger.error(
        f"[P0-4/CAS] Reserva agotó {max_retries} reintentos por conflicto persistente: "
        f"user={user_id} ingredient={ingredient_name} key={reservation_key} "
        f"row={last_compatible_row_id}. Posible alta contención o bug de mismatch."
    )
    return False


def reserve_plan_ingredients(user_id: str, chunk_id: str, days: List[Dict[str, Any]]) -> int:
    """Reserva ingredientes de un chunk confirmado para que el siguiente vea stock disponible real.

    [P1-N1-RESERVATION-DELTA · 2026-05-15] Batch-fetch del inventory del
    usuario UNA VEZ al inicio, indexado por `ingredient_name`. Cada call a
    `_apply_reservation_delta` recibe `prefetched_rows=` para evitar el
    SELECT-per-ingredient. Plan de 30 ingredientes pre-fix = 30 roundtrips a
    Supabase; post-fix = 1 roundtrip en el happy path (sin contención CAS).
    Cuando CAS conflicta, retry attempts re-SELECT como antes (fresh state
    obligatorio post-conflict).
    """
    if not _db_available() or not days:
        return 0

    # [P1-N1-RESERVATION-DELTA · 2026-05-15] Batch fetch del inventory completo
    # del usuario. Best-effort: si falla, fallback al patrón legacy (None →
    # cada `_apply_reservation_delta` hace su propio SELECT).
    rows_by_name: Optional[Dict[str, List[Dict[str, Any]]]] = None
    try:
        _batch_rows = execute_sql_query(
            "SELECT id, ingredient_name, quantity::float8 AS quantity, unit, "
            "reserved_quantity::float8 AS reserved_quantity, reservation_details "
            "FROM user_inventory WHERE user_id = %s",
            (user_id,),
            fetch_all=True,
        ) or []
        rows_by_name = {}
        for _r in _batch_rows:
            _nm = _r.get("ingredient_name")
            if not _nm:
                continue
            rows_by_name.setdefault(_nm, []).append(_r)
    except Exception as _batch_err:
        logger.debug(
            f"[P1-N1-RESERVATION-DELTA] batch-fetch falló (best-effort, "
            f"fallback a SELECT per-ingredient): {_batch_err}"
        )
        rows_by_name = None

    reserved_items = 0
    for day in days:
        for meal in (day or {}).get("meals", []):
            reservation_key = _make_reservation_key(chunk_id, meal.get("name", ""))
            for item in meal.get("ingredients", []) or []:
                if not item or len(str(item).strip()) < 3:
                    continue
                try:
                    qty, unit, name = _parse_quantity(str(item))
                    if name and qty > 0:
                        _prefetched = rows_by_name.get(name) if rows_by_name is not None else None
                        if _apply_reservation_delta(
                            user_id, name, qty, unit, reservation_key,
                            prefetched_rows=_prefetched,
                        ):
                            reserved_items += 1
                except Exception as e:
                    logger.error(f"Error reservando '{item}' para {user_id}: {e}")
    return reserved_items


def release_meal_reservation(user_id: str, meal_name: str) -> int:
    """Libera reservas asociadas a una comida rechazada."""
    if not _db_available() or not user_id or not meal_name:
        return 0

    released = 0
    try:
        rows = execute_sql_query(
            "SELECT id, reserved_quantity::float8 AS reserved_quantity, reservation_details "
            "FROM user_inventory WHERE user_id = %s AND reserved_quantity > 0",
            (user_id,),
            fetch_all=True,
        )
        for row in rows or []:
            reservation_details = _normalize_reservation_details(row.get("reservation_details"))
            keys_to_remove = [k for k in reservation_details.keys() if _reservation_matches_meal(k, meal_name)]
            if not keys_to_remove:
                continue
            reserved_quantity = float(row.get("reserved_quantity") or 0)
            for key in keys_to_remove:
                reserved_quantity = max(reserved_quantity - float(reservation_details.get(key) or 0), 0.0)
                reservation_details.pop(key, None)
                released += 1
            _update_row_reservation(row["id"], reserved_quantity, reservation_details)
    except Exception as e:
        logger.error(f"Error liberando reserva de '{meal_name}' para {user_id}: {e}")
    return released


def release_chunk_reservations(user_id: str, chunk_id: str) -> int:
    """[P0-4][P1-A] Libera TODAS las reservas de inventario asociadas a un chunk
    cancelado/fallido. Operación all-or-nothing: si el batch de UPDATEs falla a mitad,
    NINGUNA fila queda modificada (rollback) — el cron `_recover_orphan_chunk_reservations`
    recogerá el chunk en el siguiente ciclo.

    Antes (P0-4 original): el loop ejecutaba un `_update_row_reservation` por fila vía
    Supabase REST. Si Supabase devolvía 503 a la mitad (5 filas, 3 actualizadas, 2 no),
    el chunk quedaba con reservas fantasma. La próxima llamada NO veía las 3 ya liberadas
    (sus keys ya no existían en `reservation_details`) pero las 2 pendientes seguían
    inflando `reserved_quantity` indefinidamente.

    Ahora (P1-A): batch atómico vía `execute_sql_transaction`. El SELECT inicial sigue
    yendo por Supabase REST porque es read-only y no participa en la transacción de
    escritura. Los UPDATEs se acumulan en memoria y se ejecutan dentro de un único
    BEGIN/COMMIT — psycopg revierte automáticamente si cualquiera de ellos falla.

    Devuelve la cantidad de keys de reserva eliminadas (NO filas afectadas), preservando
    el contrato histórico para callers que loguean el conteo.
    """
    if not user_id or not chunk_id:
        return 0

    prefix = f"chunk:{chunk_id}:"

    # 1. Read snapshot de filas con reservas. [P1-NEON-DB-MIGRATION · 2026-06-12]
    # Antes había rama dual supabase-REST/SQL; ahora el SQL directo es el único path.
    rows: List[Dict[str, Any]] = []
    try:
        rows = execute_sql_query(
            "SELECT id, reserved_quantity, reservation_details FROM user_inventory "
            "WHERE user_id = %s AND reserved_quantity > 0",
            (user_id,),
            fetch_all=True,
        ) or []
    except Exception as e:
        logger.error(f"[P1-A] SELECT inicial falló para chunk {chunk_id} user {user_id}: {e}")
        return 0

    # 2. Compute updates en memoria. Skip filas sin keys del chunk objetivo.
    update_specs: List[Dict[str, Any]] = []
    released = 0
    for row in rows:
        reservation_details = _normalize_reservation_details(row.get("reservation_details"))
        keys_to_remove = [k for k in reservation_details if k.startswith(prefix)]
        if not keys_to_remove:
            continue
        reserved_quantity = float(row.get("reserved_quantity") or 0)
        for key in keys_to_remove:
            reserved_quantity = max(reserved_quantity - float(reservation_details.get(key) or 0), 0.0)
            reservation_details.pop(key, None)
            released += 1
        update_specs.append({
            "row_id": row["id"],
            "new_reserved": round(max(reserved_quantity, 0.0), 4),
            "new_details": reservation_details,
        })

    if not update_specs:
        return 0

    # 3. Build queries para transacción atómica. Jsonb wrap garantiza serialización
    # nativa a jsonb (psycopg lo trataría como texto si pasáramos dict crudo).
    try:
        from db_core import connection_pool, execute_sql_transaction
    except Exception:
        connection_pool = None
        execute_sql_transaction = None

    if connection_pool is not None and execute_sql_transaction is not None:
        from psycopg.types.json import Jsonb
        # [P2-INVENTORY-STATEMENT-TIMEOUT · 2026-05-15] `SET LOCAL statement_timeout`
        # como primer statement de la transacción. ANTES, los UPDATE de
        # liberación de reservas dependían del timeout global del pool
        # Supavisor (~60s) — bajo contención alta (lock fight con
        # `_chunk_worker` activo), un UPDATE bloqueado 60s monopolizaba un
        # slot del pool con repercusión a otras requests.
        #
        # Knob: `MEALFIT_INVENTORY_UPDATE_STMT_TIMEOUT_MS` (default 5000).
        # `execute_sql_transaction` ejecuta queries en orden DENTRO de una
        # misma tx (BEGIN ... COMMIT/ROLLBACK), por lo que `SET LOCAL`
        # surte efecto sobre los UPDATE subsecuentes y se descarta al
        # COMMIT/ROLLBACK (sin contaminar otras conexiones del pool).
        _stmt_timeout_ms = _env_int_safe_inventory(
            "MEALFIT_INVENTORY_UPDATE_STMT_TIMEOUT_MS", 5000
        )
        if _stmt_timeout_ms < 100:
            _stmt_timeout_ms = 5000  # defensa contra valores absurdos
        queries = [
            (f"SET LOCAL statement_timeout = {int(_stmt_timeout_ms)}", ()),
        ]
        queries.extend(
            (
                "UPDATE user_inventory SET reserved_quantity = %s, reservation_details = %s WHERE id = %s",
                (spec["new_reserved"], Jsonb(spec["new_details"]), spec["row_id"]),
            )
            for spec in update_specs
        )
        try:
            execute_sql_transaction(queries)
        except Exception as e:
            logger.error(
                f"[P1-A] Liberación atómica de {len(queries)} reservas falló para chunk "
                f"{chunk_id} user {user_id}: {e}. Sin cambios aplicados (rollback automático). "
                f"`_recover_orphan_chunk_reservations` recogerá las huérfanas."
            )
            return 0
    else:
        # Fallback no-transaccional (legacy, p. ej. tests sin pool montado): el riesgo
        # de liberación parcial vuelve, pero al menos preservamos compatibilidad.
        for spec in update_specs:
            try:
                _update_row_reservation(spec["row_id"], spec["new_reserved"], spec["new_details"])
            except Exception as e:
                logger.error(
                    f"[P1-A/FALLBACK] UPDATE no-transaccional falló para row {spec['row_id']} "
                    f"chunk {chunk_id} user {user_id}: {e}. Riesgo de liberación parcial; "
                    f"el cron de cleanup recogerá las huérfanas si quedan."
                )

    if released:
        logger.info(f"[P0-4/P1-A] Liberadas {released} reservas del chunk {chunk_id} para {user_id} (atómico).")
    return released


def _consume_reserved_inventory(user_id: str, ingredient_name: str, quantity: float, unit: str) -> bool:
    """Convierte reserva planificada en consumo real reduciendo reserved_quantity antes del descuento físico."""
    if not _db_available() or quantity <= 0:
        return False

    existing_rows = execute_sql_query(
        "SELECT id, unit, reserved_quantity::float8 AS reserved_quantity, reservation_details "
        "FROM user_inventory WHERE user_id = %s AND ingredient_name = %s AND reserved_quantity > 0",
        (user_id, ingredient_name),
        fetch_all=True,
    )

    master_list = get_master_ingredients()
    master_item = next((m for m in master_list if m["name"] == ingredient_name), {})

    for row in existing_rows or []:
        current_unit = row.get("unit") or unit or "unidad"
        converted_qty = convert_amount(quantity, unit, current_unit, master_item)
        if converted_qty is None:
            continue

        remaining = round(max(converted_qty, 0.0), 4)
        if remaining <= 0:
            return False

        reservation_details = _normalize_reservation_details(row.get("reservation_details"))
        for key in list(reservation_details.keys()):
            if remaining <= 0:
                break
            current_reserved = float(reservation_details.get(key) or 0)
            consumed = min(current_reserved, remaining)
            leftover = round(current_reserved - consumed, 4)
            remaining = round(remaining - consumed, 4)
            if leftover > 0:
                reservation_details[key] = leftover
            else:
                reservation_details.pop(key, None)

        reserved_quantity = sum(float(v) for v in reservation_details.values())
        _update_row_reservation(row["id"], reserved_quantity, reservation_details)
        return True

    return False

def add_or_update_inventory_item(user_id: str, ingredient_name: str, quantity: float, unit: str, mutation_type: str = "manual", source: str = "manual"):
    """
    Agrega o actualiza un ingrediente en la despensa del usuario.
    Resuelve empates de unidades (ej. si hay kg y pides restar g, restará del kg correctamente).

    [P0.2] El parámetro `source` registra la procedencia de la fila (manual /
    shopping_list / unknown) y sólo se persiste en INSERT. En UPDATE NO se
    sobrescribe — first-writer-wins — para preservar la identidad de items
    manuales aun cuando un restock posterior sume cantidad sobre ellos.

    [P0-4 · 2026-05-10] El path UPDATE/DELETE sobre filas existentes ahora
    delega a la RPC `apply_inventory_delta` (SECURITY DEFINER, lock
    FOR UPDATE + UPDATE/DELETE en misma TX). Antes el SELECT-MODIFY-WRITE
    en app-layer podía perder updates bajo concurrencia (dos chunks
    deduciendo en paralelo el mismo ingrediente, o usuario loggeando 2×
    la misma comida). El INSERT de fila nueva sigue en app-layer porque
    no hay race posible: la fila no existe hasta este INSERT.
    """
    if not _db_available(): return False
    try:
        # Extraemos sin filtrar por 'unit' para buscar compatibles
        existing_rows = execute_sql_query(
            "SELECT id, quantity::float8 AS quantity, unit FROM user_inventory "
            "WHERE user_id = %s AND ingredient_name = %s",
            (user_id, ingredient_name),
            fetch_all=True,
        ) or []

        master_list = get_master_ingredients()
        master_item = next((m for m in master_list if m["name"] == ingredient_name), {})
        master_id_raw = master_item.get("id", None) if master_item else None
        master_id = str(master_id_raw) if master_id_raw is not None else None

        updated = False

        if existing_rows:
            # Buscar el primer registro temporalmente compatible
            for row in existing_rows:
                row_id = row["id"]
                current_qty = float(row["quantity"])
                current_unit = row["unit"]

                converted_qty = convert_amount(quantity, unit, current_unit, master_item)

                if converted_qty is not None:
                    # [P0-4 · 2026-05-10] Atomic delta vía RPC. La RPC hace
                    # `SELECT … FOR UPDATE` + UPDATE/DELETE en la misma
                    # transacción, lo que serializa concurrent calls sobre
                    # la MISMA fila y elimina la lost-update race del
                    # path SELECT-MODIFY-WRITE legacy.
                    #
                    # Si la RPC retorna `status=not_found` (ownership
                    # mismatch o fila eliminada entre el SELECT de
                    # `existing` y este UPDATE), caemos al INSERT como
                    # path de recuperación.
                    try:
                        # [P1-NEON-DB-MIGRATION · 2026-06-12] RPC vía SELECT directo.
                        # La función sigue existiendo en la DB (portada a Neon);
                        # con conexión directa como postgres la atomicidad
                        # FOR UPDATE se preserva. `%s::numeric` explícito porque
                        # float8→numeric NO es cast implícito en resolución de
                        # funciones.
                        _rpc_row = execute_sql_query(
                            "SELECT public.apply_inventory_delta("
                            "%s::uuid, %s::bigint, %s::numeric, %s, %s::uuid"
                            ") AS result",
                            (
                                user_id,
                                row_id,
                                round(converted_qty, 4),
                                mutation_type,
                                master_id,
                            ),
                            fetch_one=True,
                        )
                        rpc_data = _rpc_row.get("result") if _rpc_row else None
                        if isinstance(rpc_data, dict) and rpc_data.get("status") == "not_found":
                            # Race: fila desapareció entre nuestro SELECT y
                            # el FOR UPDATE. Loguear sin retry — el INSERT
                            # downstream re-creará la fila si la cantidad
                            # neta es positiva.
                            logger.warning(
                                f"[P0-4] apply_inventory_delta returned not_found "
                                f"for user={user_id} row_id={row_id} "
                                f"ingredient={ingredient_name!r} — race-recovery "
                                f"vía INSERT."
                            )
                            updated = False
                        else:
                            updated = True
                    except Exception as rpc_err:
                        # [P1-NEW-1 · 2026-05-10] Fallback al path legacy si la
                        # RPC no está disponible (deploy lag, permisos). El
                        # log original era observabilidad floja: sin entry en
                        # `system_alerts`, producción podía operar bajo
                        # lost-update race silenciosamente — el fix P0-4
                        # quedaría desactivado sin que nadie lo notara.
                        #
                        # Cambios:
                        #   1. `system_alerts.inventory_rpc_fallback` con
                        #      `severity=critical` (la RPC NO debería fallar
                        #      en producción; fallar implica regression).
                        #   2. Knob `MEALFIT_INVENTORY_RPC_STRICT` (default
                        #      False): si True, re-raise `rpc_err` para que
                        #      el caller falle loud en vez de continuar bajo
                        #      el path no-atómico. Recomendado para prod tras
                        #      verificar que la RPC está estable.
                        #   3. Solo entonces ejecutamos el fallback UPDATE
                        #      legacy (sin race-control). El alert ya está
                        #      emitido para que ops sepa que ocurrió.
                        from knobs import _env_bool as _knob_env_bool
                        _strict = _knob_env_bool("MEALFIT_INVENTORY_RPC_STRICT", False)
                        try:
                            # [P1-NEON-DB-MIGRATION · 2026-06-12] Upsert por
                            # alert_key (mismo patrón SQL que el alert P3-1 de
                            # replace_shopping_list_only_items). triggered_at/
                            # resolved_at usan defaults DB en INSERT y se
                            # re-arman en el DO UPDATE.
                            _alert_message = (
                                f"RPC apply_inventory_delta lanzó "
                                f"{type(rpc_err).__name__}: {str(rpc_err)[:200]}. "
                                f"Path de deducción cayó al UPDATE legacy "
                                f"(SELECT-MODIFY-WRITE en app-layer) que NO "
                                f"serializa concurrent calls — lost-update "
                                f"race re-introducido temporalmente. Revisar "
                                f"deploy lag (¿migración p0_4_apply_inventory_delta_rpc.sql "
                                f"aplicada?) y permisos service_role."
                            )
                            _alert_metadata = {
                                "user_id": user_id,
                                "row_id": row_id,
                                "ingredient": ingredient_name,
                                "exc_type": type(rpc_err).__name__,
                                "strict_mode": _strict,
                            }
                            execute_sql_write(
                                """
                                INSERT INTO system_alerts
                                    (alert_key, alert_type, severity, title, message, metadata)
                                VALUES (%s, 'inventory', 'critical', %s, %s, %s::jsonb)
                                ON CONFLICT (alert_key) DO UPDATE
                                    SET triggered_at = NOW(),
                                        message = EXCLUDED.message,
                                        metadata = EXCLUDED.metadata,
                                        resolved_at = NULL
                                """,
                                (
                                    "inventory_rpc_fallback",
                                    "apply_inventory_delta RPC falló — fallback no-atómico",
                                    _alert_message,
                                    json.dumps(_alert_metadata, ensure_ascii=False),
                                ),
                            )
                        except Exception as _alert_err:
                            logger.warning(
                                f"[P1-NEW-1] No se pudo emitir system_alert "
                                f"para inventory_rpc_fallback: {_alert_err}"
                            )
                        logger.error(
                            f"[P0-4] apply_inventory_delta RPC falló para "
                            f"user={user_id} row_id={row_id}: {rpc_err}. "
                            f"Fallback al UPDATE legacy (no-atomic). "
                            f"strict_mode={_strict}."
                        )
                        if _strict:
                            # Fail-loud: propagar la excepción para que el
                            # caller (chunk worker, log_consumed_meal, etc.)
                            # haga retry o registre el fallo en
                            # `failed_inventory_deductions` (P0-5).
                            raise
                        new_qty = round(current_qty + converted_qty, 4)
                        if new_qty < 0.01:
                            execute_sql_write(
                                "DELETE FROM user_inventory WHERE id = %s",
                                (row_id,),
                            )
                        else:
                            execute_sql_write(
                                "UPDATE user_inventory SET quantity = %s, "
                                "master_ingredient_id = %s::uuid, last_mutation_type = %s "
                                "WHERE id = %s",
                                (new_qty, master_id, mutation_type, row_id),
                            )
                        updated = True
                    break

        # Si no se encontró un registro compatible o no existía, insertamos uno nuevo.
        # [P3-PROD-AUDIT-2 · 2026-05-30 · cerrado P1-NEON-DB-MIGRATION 2026-06-12]
        # La race cross-call aceptada (dos escritores concurrentes del MISMO item
        # nuevo + unit + user veían `existing` vacío → doble INSERT → el 2º violaba
        # el unique `(user_id, ingredient_name, unit)` → add perdido) queda cerrada
        # "de gratis" con la migración a raw SQL: `INSERT ... ON CONFLICT DO UPDATE
        # SET quantity = quantity + EXCLUDED.quantity` (el increment que PostgREST
        # upsert no soportaba). `source` NO se toca en el DO UPDATE — first-writer-
        # wins (P0.2).
        if not updated:
            if quantity >= 0.01:
                execute_sql_write(
                    """
                    INSERT INTO user_inventory
                        (user_id, ingredient_name, quantity, unit,
                         master_ingredient_id, last_mutation_type, source)
                    VALUES (%s, %s, %s, %s, %s::uuid, %s, %s)
                    ON CONFLICT (user_id, ingredient_name, unit) DO UPDATE
                        SET quantity = user_inventory.quantity + EXCLUDED.quantity,
                            master_ingredient_id = COALESCE(EXCLUDED.master_ingredient_id, user_inventory.master_ingredient_id),
                            last_mutation_type = EXCLUDED.last_mutation_type
                    """,
                    (
                        user_id,
                        ingredient_name,
                        round(quantity, 4),  # Evitar floating point overflow
                        unit,
                        master_id,
                        mutation_type,
                        source,
                    ),
                )
            elif quantity < 0 and existing_rows:
                # [P2-DEDUCT-NOOP-INCOMPAT-UNIT · 2026-05-30] Deducción no-op
                # silenciosa: HABÍA fila(s) del mismo ingrediente pero NINGUNA en
                # unidad convertible (convert_amount → None para todas — típico
                # cuando master_ingredients no tiene density_g_per_cup/unit y
                # MEALFIT_CROSS_UNIT_CONVERSION_STRICT=True). Pre-fix retornábamos
                # `True` aquí → deduct_consumed_meal_from_inventory lo clasificaba
                # como succeeded → el item NUNCA llegaba a failed_inventory_deductions
                # → ni se descontaba la nevera ni se reintentaba ni alertaba el cron
                # de backlog. Retornar False lo enruta al replay queue
                # (reason=deduction_returned_false) y señala el master incompleto.
                # OJO: el caso `existing_rows` VACÍO (item ausente en la nevera) NO
                # entra aquí — sigue retornando True (ausencia ≠ failure, correcto).
                logger.warning(
                    f"[P2-DEDUCT-NOOP] deducción no-op: {len(existing_rows)} fila(s) "
                    f"existente(s) del ingrediente pero unidad incompatible — "
                    f"user={user_id} ingredient={ingredient_name!r} unit={unit!r}. "
                    f"Backfill density_g_per_cup/density_g_per_unit en master_ingredients."
                )
                return False
        return True
    except Exception as e:
        logger.error(f"Error actualizando inventario para {user_id}: {e}")
        return False

def get_inventory_activity_since(user_id: str, since_iso: str) -> Dict[str, Any]:
    """[P0-D] Señal implícita de adherencia: ¿el usuario tocó su nevera desde `since_iso`?

    Usado por _check_chunk_learning_ready cuando NO hay logs explícitos de comidas
    (zero_log_proxy=True). Si el inventario muestra mutaciones desde la fecha de inicio
    del chunk previo, asumimos que el usuario está consumiendo el plan aunque no loguee.

    Retorna {mutations_count, last_mutation_at, low_stock_items, consumption_mutations_count, manual_mutations_count}
    """
    if not _db_available() or not user_id or not since_iso:
        return {"mutations_count": 0, "last_mutation_at": None, "low_stock_items": 0, "consumption_mutations_count": 0, "manual_mutations_count": 0}
    try:
        # [P1-NEON-DB-MIGRATION · 2026-06-12] updated_at → string ISO (parity
        # PostgREST): el max() de abajo compara strings y `last_mutation_at`
        # se serializa a JSON aguas arriba.
        rows = execute_sql_query(
            "SELECT id, ingredient_name, quantity::float8 AS quantity, "
            "to_jsonb(updated_at) #>> '{}' AS updated_at, last_mutation_type "
            "FROM user_inventory WHERE user_id = %s AND updated_at >= %s",
            (user_id, since_iso),
            fetch_all=True,
        ) or []
        if not rows:
            return {"mutations_count": 0, "last_mutation_at": None, "low_stock_items": 0, "consumption_mutations_count": 0, "manual_mutations_count": 0}
        last_mutation = max((r.get("updated_at") for r in rows if r.get("updated_at")), default=None)
        # Items con stock muy bajo (≤ 0.5 en unidad arbitraria) sugieren consumo reciente
        low_stock = sum(1 for r in rows if float(r.get("quantity") or 0) <= 0.5)
        
        consumption_mutations = sum(1 for r in rows if r.get("last_mutation_type") == "consumption")
        manual_mutations = sum(1 for r in rows if r.get("last_mutation_type") != "consumption")

        return {
            "mutations_count": len(rows),
            "last_mutation_at": last_mutation,
            "low_stock_items": low_stock,
            "consumption_mutations_count": consumption_mutations,
            "manual_mutations_count": manual_mutations
        }
    except Exception as e:
        logger.warning(f"[P0-D] Error consultando actividad de inventario para {user_id}: {e}")
        return {"mutations_count": 0, "last_mutation_at": None, "low_stock_items": 0, "consumption_mutations_count": 0, "manual_mutations_count": 0}


def _persist_failed_inventory_deductions(user_id: str, failed_items: list) -> None:
    """[P0-5 · 2026-05-10] Persiste items que fallaron al deducirse del inventario
    en `failed_inventory_deductions` para observabilidad + retry posterior.

    Antes de P0-5 la tabla existía con RLS forzado e índice KEEP (P2-PERF-1),
    pero NADIE escribía a ella — los fallos quedaban solo en logs locales,
    invisibles a alertas. Cron `_alert_failed_inventory_deductions_backlog`
    (cron_tasks.py) lee esta tabla y emite `system_alerts` cuando el backlog
    crece, cerrando el gap "write-only orphan".

    El INSERT es best-effort: si falla, loguea warning y continúa — no
    debemos bloquear la deducción principal por una falla de observabilidad
    secundaria. La idempotencia no es necesaria (cada deducción que falla
    es un evento único; duplicados sub-segundo son posibles pero no rompen
    nada — el alert cron solo cuenta filas).

    `failed_items` shape: lista de dicts con `{item, name?, qty?, unit?, reason}`.
    Persistido como jsonb en columna `ingredients`. Schema: `attempts=0` inicial,
    `created_at`/`updated_at` defaults a `now()`.
    """
    if not _db_available() or not user_id or not failed_items:
        return
    try:
        execute_sql_write(
            "INSERT INTO failed_inventory_deductions (user_id, ingredients, attempts) "
            "VALUES (%s, %s, 0)",
            (user_id, Jsonb(failed_items)),
        )
    except Exception as insert_err:
        # Warning, no error: la deducción principal ya completó (o no);
        # esto es solo telemetría.
        logger.warning(
            f"[P0-5] No se pudo persistir failed_inventory_deductions para "
            f"user={user_id}: {insert_err}"
        )


# [P1-PANTRY-INFER · 2026-05-22] Tabla canónica de porciones típicas es-DO
# por ingrediente. Cierra el gap "LLM extrae nombres del usuario sin cantidades
# parseables" — cuando el chat agent registra una comida desde texto natural
# ("me comí una taza de avena con chía y mantequilla de maní"), `_parse_quantity`
# retorna `(0.0, 'cantidad necesaria', name)` para cada ingrediente porque no
# hay dígitos al inicio del string. Pre-fix esto contaminaba
# `failed_inventory_deductions` con `reason=parse_failed_or_invalid_qty` y la
# nevera nunca se descontaba (3 incidentes verificados en BD prod a 2026-05-22:
# avena/chía/maní, bandeja paisa 9 items, mayonesa/kétchup).
#
# Las cantidades aquí son porciones individuales adultas típicas en es-DO.
# Items vendidos al peso se expresan en gramos crudos — `convert_amount` los
# transforma a la unidad del row pantry (lb/paquete/lata) vía `master_item`.
# Items unitarios (huevo, pan, manzana, plátano) usan `unidad`.
#
# Tooltip-anchor: P1-PANTRY-INFER-TYPICAL-PORTION-TABLE
_TYPICAL_PORTION_BY_NAME: Dict[str, tuple] = {
    # --- Granos / cereales (gramos crudos) ---
    "avena": (40.0, "g"),
    "arroz": (50.0, "g"),
    "arroz blanco": (50.0, "g"),
    "arroz integral": (50.0, "g"),
    "habichuelas": (50.0, "g"),
    "habichuelas negras": (50.0, "g"),
    "habichuelas rojas": (50.0, "g"),
    "habichuelas blancas": (50.0, "g"),
    "frijoles": (50.0, "g"),
    "frijoles rojos": (50.0, "g"),
    "frijoles negros": (50.0, "g"),
    "gandules": (50.0, "g"),
    "lentejas": (50.0, "g"),
    "garbanzos": (50.0, "g"),
    "quinoa": (50.0, "g"),
    "pasta": (75.0, "g"),
    "espagueti": (75.0, "g"),
    "macarrones": (75.0, "g"),
    "crema de trigo": (30.0, "g"),
    "granola": (30.0, "g"),
    "cereal": (30.0, "g"),
    "harina": (30.0, "g"),
    "harina de avena": (30.0, "g"),
    "harina de trigo": (30.0, "g"),
    # --- Proteínas (gramos crudos) ---
    "pollo": (120.0, "g"),
    "pechuga de pollo": (120.0, "g"),
    "muslo de pollo": (120.0, "g"),
    "carne": (120.0, "g"),
    "carne molida": (100.0, "g"),
    "carne de res": (120.0, "g"),
    "res": (120.0, "g"),
    "pescado": (120.0, "g"),
    "filete de pescado": (120.0, "g"),
    "filete de pescado blanco": (120.0, "g"),
    "chillo": (120.0, "g"),
    "salmon": (120.0, "g"),
    "tilapia": (120.0, "g"),
    "atun": (60.0, "g"),
    "atun en agua": (60.0, "g"),
    "pavo": (100.0, "g"),
    "cerdo": (120.0, "g"),
    "chorizo": (50.0, "g"),
    "chicharron": (40.0, "g"),
    "salami": (40.0, "g"),
    "jamon": (30.0, "g"),
    "tocineta": (20.0, "g"),
    "huevo": (2.0, "unidad"),
    "huevos": (2.0, "unidad"),
    "huevo frito": (2.0, "unidad"),
    "claras de huevo": (3.0, "unidad"),
    "tofu": (100.0, "g"),
    "tempeh": (100.0, "g"),
    # --- Semillas / nuts / mantequillas ---
    "mani": (30.0, "g"),
    "almendras": (30.0, "g"),
    "almendras fileteadas": (15.0, "g"),
    "nueces": (30.0, "g"),
    "merey": (30.0, "g"),
    "pasas": (20.0, "g"),
    "chia": (15.0, "g"),
    "semillas de chia": (15.0, "g"),
    "linaza": (15.0, "g"),
    "mantequilla de mani": (16.0, "g"),
    "mantequilla": (10.0, "g"),
    # --- Lácteos (gramos / unidad) ---
    "queso": (40.0, "g"),
    "queso blanco": (40.0, "g"),
    "queso fresco": (40.0, "g"),
    "queso cottage": (100.0, "g"),
    "queso ricotta": (50.0, "g"),
    "queso mozzarella": (40.0, "g"),
    "yogurt": (170.0, "g"),
    "yogurt griego": (170.0, "g"),
    "yogurt griego sin azucar": (170.0, "g"),
    "leche": (240.0, "g"),
    "leche descremada": (240.0, "g"),
    "leche entera": (240.0, "g"),
    # --- Aceites / condimentos / dulces ---
    "aceite": (14.0, "g"),
    "aceite de oliva": (14.0, "g"),
    "aceite de coco": (14.0, "g"),
    "mayonesa": (15.0, "g"),
    "ketchup": (15.0, "g"),
    "mostaza": (15.0, "g"),
    "salsa de tomate": (30.0, "g"),
    "miel": (20.0, "g"),
    "azucar": (12.0, "g"),
    "sal": (3.0, "g"),
    "extracto de vainilla": (2.0, "g"),
    "hummus": (30.0, "g"),
    # --- Verduras (mix unidad/gramos) ---
    "tomate": (1.0, "unidad"),
    "cebolla": (0.5, "unidad"),
    "ajo": (2.0, "diente"),
    "pimiento": (0.5, "unidad"),
    "pimenton": (0.5, "unidad"),
    "aguacate": (0.5, "unidad"),
    "lechuga": (50.0, "g"),
    "espinaca": (50.0, "g"),
    "brocoli": (100.0, "g"),
    "coliflor": (100.0, "g"),
    "zanahoria": (1.0, "unidad"),
    "auyama": (150.0, "g"),
    "vainitas": (100.0, "g"),
    "berenjena": (100.0, "g"),
    "calabacin": (100.0, "g"),
    "remolacha": (100.0, "g"),
    "pepino": (0.5, "unidad"),
    "cilantro": (10.0, "g"),
    "perejil": (10.0, "g"),
    "limon": (1.0, "unidad"),
    "apio": (1.0, "unidad"),
    # --- Frutas ---
    "manzana": (1.0, "unidad"),
    "guineo": (1.0, "unidad"),
    "platano": (1.0, "unidad"),
    "platano maduro": (1.0, "unidad"),
    "platano verde": (1.0, "unidad"),
    "mango": (1.0, "unidad"),
    "pina": (150.0, "g"),
    "papaya": (150.0, "g"),
    "lechosa": (150.0, "g"),
    "fresas": (100.0, "g"),
    "fresas frescas": (100.0, "g"),
    "uvas": (100.0, "g"),
    "pera": (1.0, "unidad"),
    "naranja": (1.0, "unidad"),
    "mandarina": (1.0, "unidad"),
    "melon": (150.0, "g"),
    "sandia": (200.0, "g"),
    "kiwi": (1.0, "unidad"),
    # --- Tubérculos ---
    "papa": (150.0, "g"),
    "papas": (150.0, "g"),
    "batata": (150.0, "g"),
    "yuca": (150.0, "g"),
    "yautia": (150.0, "g"),
    # --- Panes / derivados ---
    "pan": (2.0, "rebanada"),
    "pan integral": (2.0, "rebanada"),
    "pan de molde": (2.0, "rebanada"),
    "casabe": (1.0, "unidad"),
    "tortilla": (2.0, "unidad"),
    "tortilla de maiz": (2.0, "unidad"),
    "arepa": (1.0, "unidad"),
    "galleta": (4.0, "unidad"),
    "galletas": (4.0, "unidad"),
    "crackers": (4.0, "unidad"),
}

# Sustantivos que un user típicamente cuenta "1 X" (no en gramos). El fallback
# genérico cuando NO hay match en el dict canónico usa esta lista para decidir
# entre `(1.0, 'unidad')` vs `(50.0, 'g')`.
_UNITARY_NAME_HINTS = (
    "manzana", "guineo", "platano", "mango", "pera", "naranja", "kiwi",
    "aguacate", "limon", "mandarina", "tomate", "cebolla", "zanahoria",
    "pepino", "pimiento", "pimenton", "apio", "huevo", "pan", "casabe",
    "arepa", "galleta", "tortilla",
)


def _strip_accents_lower(s: str) -> str:
    """Lowercase + diacritics stripped, para lookup contra _TYPICAL_PORTION_BY_NAME."""
    nfd = unicodedata.normalize("NFD", str(s))
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").lower().strip()


def _infer_typical_portion(name: str) -> Optional[tuple]:
    """[P1-PANTRY-INFER · 2026-05-22] Devuelve `(qty, unit)` inferida cuando
    el `_parse_quantity` no extrajo cantidad pero hay un name limpio.

    Pipeline:
      1. Normalizar (lower + strip accents).
      2. Match exacto contra `_TYPICAL_PORTION_BY_NAME`.
      3. Match parcial: si el key del dict aparece como substring del name
         normalizado (e.g. "mantequilla de mani natural sin azucar" → "mantequilla de mani").
      4. Fallback heurístico: si contiene un `_UNITARY_NAME_HINTS` → (1.0, "unidad");
         else (50.0, "g").

    Tooltip-anchor: P1-PANTRY-INFER-INFER-FN
    """
    if not name or not str(name).strip():
        return None
    n = _strip_accents_lower(name)
    if not n:
        return None
    # 1. Exact match
    if n in _TYPICAL_PORTION_BY_NAME:
        return _TYPICAL_PORTION_BY_NAME[n]
    # 2. Longest substring match — preferir keys más específicos primero
    candidates = [k for k in _TYPICAL_PORTION_BY_NAME if k in n]
    if candidates:
        best = max(candidates, key=len)
        return _TYPICAL_PORTION_BY_NAME[best]
    # 3. Fallback heurístico
    for hint in _UNITARY_NAME_HINTS:
        if hint in n:
            return (1.0, "unidad")
    return (50.0, "g")


def deduct_consumed_meal_from_inventory(user_id: str, ingredients_list: List[str]):
    """
    Resta matemáticamente una lista de ingredientes crudos (los de una comida consumida)
    de la tabla de inventario físico.

    [P0-5 · 2026-05-10] Items que fallan (parse error, exception, deducción
    devuelve False) se acumulan en `failed_items` y se persisten al final
    en `failed_inventory_deductions` para que el cron de alerta los detecte.
    Item ausente en pantry NO es failure — el usuario puede haber consumido
    algo que no tenía registrado (el deduct devuelve True silencioso si no
    hay row compatible).

    [P1-PANTRY-INFER · 2026-05-22] Cuando el chat agent registra una comida
    desde texto natural ("me comí una taza de avena"), `_parse_quantity`
    retorna `qty=0` con `name` poblado. Pre-fix esos items iban directo a
    `failed_inventory_deductions` con `reason=parse_failed_or_invalid_qty`
    y la nevera nunca se descontaba. Ahora se invoca `_infer_typical_portion`
    para asignar una porción adulta típica antes de fallar. Knob
    `MEALFIT_PANTRY_INFER_TYPICAL_PORTION` (default True) permite rollback
    sin redeploy. Cuando una inferencia se aplica, se emite WARN
    `[P1-PANTRY-INFER]` con el item original y la porción inferida — útil
    para auditar qué items necesitan entry explícita en
    `_TYPICAL_PORTION_BY_NAME`.
    """
    if not _db_available() or not ingredients_list: return

    try:
        from knobs import _env_bool as _knob_env_bool
        _infer_enabled = _knob_env_bool("MEALFIT_PANTRY_INFER_TYPICAL_PORTION", True)
    except Exception:
        _infer_enabled = True

    failed_items = []
    # [P1-AGENT-HINT · 2026-05-22] Trackers de los strings originales para que
    # `log_consumed_meal` pueda construir el hint a la LLM cuando algún item
    # no se pudo procesar (red de seguridad sobre #1: si la inferencia retornó
    # None — caso raro: name vacío post-parse — la LLM puede reintentar pidiéndole
    # al usuario que confirme la cantidad).
    succeeded_strs: List[str] = []
    inferred_strs: List[str] = []
    failed_strs: List[str] = []
    for item in ingredients_list:
        if not item or len(item) < 3: continue
        try:
            qty, unit, name = _parse_quantity(item)
            used_inference = False
            if not (name and qty > 0):
                # [P1-PANTRY-INFER] El parse no extrajo qty parseable pero
                # puede haber name limpio (ej "Avena", "Mantequilla de maní"
                # emitidos por la LLM sin cantidad). Intentar inferencia.
                inferred = _infer_typical_portion(name) if (_infer_enabled and name) else None
                if inferred is not None:
                    qty, unit = inferred
                    used_inference = True
                    logger.warning(
                        f"[P1-PANTRY-INFER] Inferida porción típica para "
                        f"item={item!r} → name={name!r} qty={qty} unit={unit!r} "
                        f"(user={user_id})"
                    )
                else:
                    # Parse falló y no se pudo inferir — el item no es procesable.
                    failed_items.append({
                        "item": str(item)[:200],
                        "reason": "parse_failed_or_invalid_qty",
                    })
                    failed_strs.append(str(item))
                    continue
            _consume_reserved_inventory(user_id, name, qty, unit)
            # Actualizar restando
            ok = add_or_update_inventory_item(user_id, name, -qty, unit, mutation_type="consumption")
            if ok is False:
                # `add_or_update_inventory_item` devolvió False explícitamente
                # (excepción interna capturada + logueada). Persistir como
                # failure para que la alerta lo detecte.
                failed_items.append({
                    "item": str(item)[:200],
                    "name": name,
                    "qty": qty,
                    "unit": unit,
                    "reason": "deduction_returned_false",
                })
                failed_strs.append(str(item))
            else:
                if used_inference:
                    inferred_strs.append(str(item))
                else:
                    succeeded_strs.append(str(item))
        except Exception as e:
            logger.error(f"Error parseando y restando '{item}' de despensa física: {e}")
            failed_items.append({
                "item": str(item)[:200],
                "reason": "exception",
                "error": str(e)[:200],
            })
            failed_strs.append(str(item))

    # [P0-5 · 2026-05-10] Persistir TODOS los failures como un solo INSERT
    # (la columna `ingredients` es jsonb array, mantenemos correlación entre
    # los items del mismo log_consumed_meal). Si la lista está vacía, no
    # hacemos round-trip a DB.
    _persist_failed_inventory_deductions(user_id, failed_items)

    # [P1-AGENT-HINT · 2026-05-22] Retornar resumen para que el caller (típicamente
    # `tools.log_consumed_meal`) pueda enriquecer el ToolMessage con un hint a la
    # LLM cuando algún item quedó sin procesar. Callers legacy que ignoran el
    # return value siguen funcionando — Python no enforza el tipo.
    return {
        "succeeded": succeeded_strs,
        "inferred": inferred_strs,
        "failed_to_deduct": failed_strs,
    }

def restock_inventory(user_id: str, ingredients_list: list):
    """
    Agrega matemáticamente una lista de ingredientes (como una lista de compras)
    al inventario físico actual, sumando cantidades según sus unidades base.

    Acepta tanto strings legados como objetos estructurados {name, quantity, unit}.

    Retorna:
        (success: bool, persisted_names: List[str]) en lugar del bool legacy.
        `persisted_names` contiene los nombres EFECTIVAMENTE persistidos
        (post-dedup intra-call) en el orden de aparición — el caller usa
        esta lista para anotar `restocked_items` SOLO con lo que llegó a DB
        (no los 27 aspiracionales). Callers legacy que esperaban bool
        siguen funcionando vía `bool(...)` truthy check.

    [P0-RESTOCK-DEDUP-NAME · 2026-05-20] Fix del bug "lista 27 → inventario
    24". Pre-fix la ruta estructurada llamaba `normalize_name()` que
    aplica 4 niveles de alias + semantic fallback vía embeddings Gemini
    sobre `master_ingredients`. Items sin alias match exacto (típicamente
    `Maní`, ingredientes regionales, brands específicos) caían al
    semantic fallback que puede mapearlos a OTRO master con similitud
    embedding ≥0.70 — colapsando items distintos al mismo canonical.

    Vector cerrado con la lista del user angelo (27 items reales):
      - 24 items: alias match exacto/regex → INSERT distinct rows.
      - 3 items sin match: cayeron al semantic fallback. Si el embedding
        mapeaba al canonical de otro item de la lista (ej. `Maní` →
        `Almendras fileteadas` por similitud "fruto seco"), la fila se
        merge'eaba con la previa o disparaba UNIQUE violation silenciosa.
      - Resultado en producción (plan 713ff43a-...): `restocked_items`
        marcaba los 27 nombres aspiracionales pero `user_inventory` solo
        tenía 24 rows, dejando el ledger inconsistente para el próximo
        ciclo.

    Fix:
      - Ruta estructurada (input desde frontend con `name` ya canonical
        del aggregator) PRESERVA el name raw — NO segundo round de
        `normalize_name`. El aggregator ya canonicalizó al producir el
        shopping list; re-canonicalizar via semantic embedding es
        destructivo.
      - Ruta legacy (input de strings crudos del chat agent o callers
        externos) sigue usando `_parse_quantity` para extraer
        qty/unit/name del string, sin pasar por `normalize_name`. El
        caller que envía strings debe garantizar el name correcto.
      - Defense in-depth: `seen_canonicals` detecta colisiones dentro
        del mismo restock call (caso edge: 2 items con MISMO name post-
        strip pasan como inputs distintos). Si pasa, conserva ambas
        entries para que el merge dentro de `add_or_update_inventory_item`
        las consolide correctamente sumando qty.

    Tooltip-anchor: P0-RESTOCK-DEDUP-NAME.
    """
    if not _db_available() or not ingredients_list:
        return False, []

    # 🔄 [MEJORA] Ya NO borramos el inventario completo. El frontend aplica Delta Shopping
    # para enviar solo ingredientes que el usuario no tiene, preservando items manuales.
    # add_or_update_inventory_item() maneja el upsert individual (suma si existe, inserta si no).

    success = True
    items_saved = 0
    persisted_names: List[str] = []  # P0-RESTOCK-DEDUP-NAME: tracking para el caller
    for item in ingredients_list:
        try:
            # Ruta Estructurada: el frontend envía {name, quantity, unit} directamente
            if isinstance(item, dict) and "name" in item:
                # [P0-RESTOCK-DEDUP-NAME · 2026-05-20] PRESERVAR el name
                # raw del shopping list. NO llamar normalize_name() —
                # el aggregator ya canonicalizó, re-canonicalizar via
                # semantic embedding puede colapsar items distintos.
                raw_name = str(item["name"]).strip()
                if not raw_name:
                    continue
                name = raw_name
                qty = float(item.get("quantity", 0))
                unit = item.get("unit", "unidad")
                # Normalizar unidades de display a canónicas de inventario
                unit_lower = unit.lower().rstrip('.')
                UNIT_NORMALIZE = _CANONICAL_UNIT_MAP
                unit = UNIT_NORMALIZE.get(unit_lower, unit_lower if unit_lower else 'unidad')
                if qty > 0:
                    # [P0.2] source='shopping_list' marca la fila para que la lógica
                    # de "MERGE inteligente" pueda distinguirla de items manuales.
                    res = add_or_update_inventory_item(user_id, name, qty, unit, source='shopping_list')
                    if res:
                        items_saved += 1
                        persisted_names.append(name)
                    else:
                        success = False
                continue

            # Ruta Legacy: re-parsear strings crudos
            if not item or (isinstance(item, str) and len(item) < 3):
                continue
            qty, unit, name = _parse_quantity(str(item))
            if name and qty > 0:
                res = add_or_update_inventory_item(user_id, name, qty, unit, source='shopping_list')
                if res:
                    items_saved += 1
                    persisted_names.append(name)
                else:
                    success = False
        except Exception as e:
            logger.error(f"Error parseando y sumando '{item}' a la despensa física: {e}")
            success = False

    logger.info(
        f"📦 [RESTOCK] {items_saved}/{len(ingredients_list)} ingredientes guardados para {user_id} "
        f"(persisted={len(persisted_names)}, success={success})"
    )

    # Si ningún ingrediente fue guardado, es un fallo (no marcar plan como restocked)
    if items_saved == 0:
        return False, []

    return success, persisted_names


def replace_shopping_list_only_items(user_id: str, ingredients_list: list) -> Dict[str, int]:
    """
    [P0.2] MERGE inteligente: limpia los items con source='shopping_list' del
    usuario y los reemplaza con la nueva lista de compras. Items con
    source='manual' (o 'unknown') quedan intactos.

    Útil cuando se quiere "resetear" la lista de compras al registrar una nueva
    sin destruir lo que el usuario haya añadido a mano. Diferente de
    `restock_inventory` (que sólo suma): aquí los shopping_list previos se
    descartan completamente porque la nueva lista los reemplaza por diseño.

    [P3-D · 2026-05-07] Rollback de seguridad. ANTES, si DELETE tenía éxito
    pero `restock_inventory` fallaba (network blip, DB error, partial insert),
    el usuario quedaba sin lista de compras y sin recuperación posible. AHORA:
      1. Antes del DELETE, snapshot completo de las filas `source='shopping_list'`.
      2. DELETE.
      3. INSERT vía `restock_inventory`.
      4. Si el insert falla (False/excepción), se restauran las filas snapshotted.
    No es atomicidad real (un crash entre DELETE e INSERT pierde datos), pero
    elimina el principal modo de fallo: errores capturables en `restock_inventory`.
    Trade-off aceptado: lograr atomicidad estricta requeriría envolver
    snapshot→DELETE→INSERT en una transacción única — posible ahora que el
    módulo va por SQL directo (P1-NEON-DB-MIGRATION), pero la paridad de
    comportamiento se preserva tal cual (P3-D docstring).

    Knob `MEALFIT_SHOPPING_LIST_REPLACE_ROLLBACK` (default `on`):
      - `on`   : restore-on-failure activado (comportamiento P3-D).
      - `off`  : modo legacy (no rollback, comportamiento previo a P3-D).
                 Kill switch operacional sin redeploy.

    Returns:
        {"deleted_shopping_rows": N, "inserted_rows": M, "preserved_manual_rows": K,
         "rolled_back": bool,                # P3-D: true si se restauró ≥1 row.
         "rolled_back_count": int,           # P3-1: filas efectivamente restauradas.
         "rolled_back_total": int,           # P3-1: filas en snapshot pre-DELETE.
         "rolled_back_partial": bool}        # P3-1: true si 0 < count < total.

    [P3-1 · 2026-05-08] Distinción `partial` vs `full` rollback. ANTES (P3-D):
    si `restored > 0`, `rolled_back=True` independiente de si se restauraron
    todas o solo algunas. Imposible distinguir "rollback exitoso completo" de
    "queda inventario inconsistente" sin grep manual al log. AHORA: cuando el
    rollback es parcial (restored < total), se emite `system_alerts` con
    `severity='critical'` y metadata para SOP de recovery manual (ver alert).
    """
    stats: Dict[str, Any] = {
        "deleted_shopping_rows": 0,
        "inserted_rows": 0,
        "preserved_manual_rows": 0,
        "rolled_back": False,
        "rolled_back_count": 0,
        "rolled_back_total": 0,
        "rolled_back_partial": False,
    }
    if not _db_available() or not user_id:
        return stats

    # [P2-1 · 2026-05-08] `_env_str` registra en `_KNOBS_REGISTRY`.
    # Semántica preservada: `off` desactiva el rollback; cualquier otro valor lo habilita.
    from knobs import _env_str
    rollback_enabled = _env_str("MEALFIT_SHOPPING_LIST_REPLACE_ROLLBACK", "on") != "off"

    snapshot_rows: List[Dict[str, Any]] = []

    try:
        _preserved_row = execute_sql_query(
            "SELECT count(*) AS count FROM user_inventory "
            "WHERE user_id = %s AND source <> 'shopping_list'",
            (user_id,),
            fetch_one=True,
        )
        stats["preserved_manual_rows"] = int((_preserved_row or {}).get("count") or 0)

        # [P3-D] Snapshot full row data antes del DELETE para poder restaurar
        # si el INSERT falla downstream. [P1-NEON-DB-MIGRATION · 2026-06-12]
        # Set de columnas fijado explícitamente (antes `select("*")`): `id`
        # solo para el log de recovery (managed_cols lo excluye del re-INSERT);
        # uuid→text y numeric→float8 para que el re-INSERT y el log JSON sean
        # type-safe.
        if rollback_enabled:
            try:
                snapshot_rows = execute_sql_query(
                    "SELECT id, user_id::text AS user_id, ingredient_name, "
                    "quantity::float8 AS quantity, unit, "
                    "master_ingredient_id::text AS master_ingredient_id, "
                    "reserved_quantity::float8 AS reserved_quantity, "
                    "reservation_details, last_mutation_type, source, category "
                    "FROM user_inventory WHERE user_id = %s AND source = 'shopping_list'",
                    (user_id,),
                    fetch_all=True,
                ) or []
                snapshot_rows = list(snapshot_rows)
            except Exception as snap_err:
                # Snapshot falló — degradar a modo legacy (sin rollback)
                # en lugar de abortar. Loggear para que SRE detecte el patrón.
                logger.warning(
                    f"[P3-D] Snapshot pre-DELETE falló para {user_id}: {snap_err}. "
                    f"Continuando en modo legacy (sin rollback) para no bloquear al usuario."
                )
                snapshot_rows = []

        # RETURNING id reemplaza el count='exact' de PostgREST.
        _deleted_rows = execute_sql_write(
            "DELETE FROM user_inventory WHERE user_id = %s AND source = 'shopping_list' "
            "RETURNING id",
            (user_id,),
            returning=True,
        )
        stats["deleted_shopping_rows"] = len(_deleted_rows or [])
    except Exception as e:
        logger.error(
            f"[P0.2] Error eliminando items shopping_list para {user_id}: {e}. "
            f"Abortando reemplazo para no dejar inventario inconsistente."
        )
        return stats

    if not ingredients_list:
        return stats

    inserted_before = stats["inserted_rows"]
    insert_failed = False
    try:
        # [P0-RESTOCK-DEDUP-NAME · 2026-05-20] restock_inventory ahora retorna
        # tupla (success, persisted_names). Aquí solo usamos `success` —
        # `persisted_names` se ignora porque este caller (replace_shopping_list_only_items)
        # no necesita anotar restocked_items.
        res_tuple = restock_inventory(user_id, ingredients_list)
        res = bool(res_tuple[0]) if isinstance(res_tuple, tuple) else bool(res_tuple)
        if not res:
            insert_failed = True
            logger.warning(
                f"[P3-D] restock_inventory retornó False para {user_id} "
                f"({len(ingredients_list)} ingredientes intentados)."
            )
    except Exception as restock_err:
        insert_failed = True
        logger.error(
            f"[P3-D] restock_inventory lanzó excepción para {user_id}: {restock_err}"
        )

    # [P3-D] ROLLBACK: si INSERT falló y tenemos snapshot, restaurar.
    # Sin snapshot (rollback_enabled=False o snapshot falló), degradamos al
    # comportamiento legacy: stats reflejan delete sin insert (UI verá lista
    # vacía y disparará re-fetch).
    if insert_failed and snapshot_rows:
        restored = 0
        failed_row_ids: List[str] = []  # [P3-1] para SOP recovery
        # Limpiar columnas managed por DB del snapshot. `id`/`created_at` son
        # inmutables generadas en INSERT; pasarlas reventaría con conflict.
        # `updated_at` típicamente trigger-generated también.
        managed_cols = {"id", "created_at", "updated_at"}
        for row in snapshot_rows:
            try:
                clean_row = {k: v for k, v in row.items() if k not in managed_cols}
                if not clean_row.get("user_id"):
                    clean_row["user_id"] = user_id
                if not clean_row.get("source"):
                    clean_row["source"] = "shopping_list"
                # [P1-NEON-DB-MIGRATION · 2026-06-12] INSERT con columnas
                # dinámicas del snapshot (keys vienen de NUESTRO SELECT, no de
                # input externo). Jsonb() para columnas jsonb
                # (reservation_details).
                _cols = list(clean_row.keys())
                _vals = tuple(
                    Jsonb(v) if isinstance(v, (dict, list)) else v
                    for v in clean_row.values()
                )
                execute_sql_write(
                    "INSERT INTO user_inventory ({}) VALUES ({})".format(
                        ", ".join(_cols), ", ".join(["%s"] * len(_cols))
                    ),
                    _vals,
                )
                restored += 1
            except Exception as restore_err:
                # Continuar con las demás filas — best-effort. Si una falla,
                # el resto puede aún restaurarse parcialmente.
                logger.error(
                    f"[P3-D/ROLLBACK] Error restaurando row para {user_id}: {restore_err}"
                )
                # [P3-1] Capturar id + name para SOP de recovery manual.
                failed_row_ids.append(str(row.get("id") or row.get("name") or "<unknown>"))
        total = len(snapshot_rows)
        is_partial = 0 < restored < total
        logger.warning(
            f"[P3-D/ROLLBACK] Insert falló para {user_id} → restauradas "
            f"{restored}/{total} filas snapshotted. "
            f"Stats refleja estado post-rollback. "
            f"{'⚠️ PARCIAL — SOP requerido.' if is_partial else 'Completo.'}"
        )
        stats["rolled_back"] = restored > 0
        stats["rolled_back_count"] = restored
        stats["rolled_back_total"] = total
        stats["rolled_back_partial"] = is_partial
        # Las filas restauradas no son "deleted" en el resultado neto.
        stats["deleted_shopping_rows"] = max(0, stats["deleted_shopping_rows"] - restored)

        # [P3-1 · 2026-05-08] Escalar rollback parcial a system_alerts. La fila
        # parcial deja inventario inconsistente: `restored` items recuperados
        # + `total - restored` perdidos en el limbo (snapshot existe en logs
        # pero no en DB). El SRE necesita señal proactiva para recovery manual.
        if is_partial:
            try:
                from datetime import datetime as _p31_dt, timezone as _p31_tz
                alert_key = f"shopping_list_replace_partial_rollback:{user_id}"
                alert_metadata = {
                    "user_id": str(user_id),
                    "rows_restored": restored,
                    "rows_in_snapshot": total,
                    "rows_lost": total - restored,
                    "failed_row_ids": failed_row_ids[:50],  # cap para evitar bloat
                    "timestamp": _p31_dt.now(_p31_tz.utc).isoformat(),
                    "ingredients_attempted": len(ingredients_list),
                }
                alert_message = (
                    f"Rollback parcial al reemplazar shopping_list de user={user_id}. "
                    f"Restauradas {restored}/{total} filas; {total - restored} "
                    f"perdidas en el limbo (snapshot en logs, NO en DB). "
                    f"\n\nSOP recovery manual:\n"
                    f"  1. Buscar en logs líneas `[P3-D/ROLLBACK] Error restaurando "
                    f"row para {user_id}` para ver qué filas fallaron.\n"
                    f"  2. Buscar log previo `[P3-D/ROLLBACK] Insert falló` con el "
                    f"snapshot completo (filas pre-DELETE).\n"
                    f"  3. Re-INSERT manual a `user_inventory` con `source='shopping_list'` "
                    f"para las filas faltantes (sin id/created_at/updated_at).\n"
                    f"  4. Verificar consistencia: `SELECT COUNT(*) FROM user_inventory "
                    f"WHERE user_id='{user_id}' AND source='shopping_list'`."
                )
                execute_sql_write(
                    """
                    INSERT INTO system_alerts
                        (alert_key, alert_type, severity, title, message, metadata, affected_user_ids)
                    VALUES (%s, 'shopping_list_partial_rollback', 'critical', %s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (alert_key) DO UPDATE
                        SET triggered_at = NOW(),
                            message = EXCLUDED.message,
                            metadata = EXCLUDED.metadata,
                            affected_user_ids = EXCLUDED.affected_user_ids,
                            resolved_at = NULL
                    """,
                    (
                        alert_key,
                        f"Shopping list rollback parcial: user {user_id}",
                        alert_message,
                        json.dumps(alert_metadata, ensure_ascii=False),
                        json.dumps([str(user_id)]),
                    ),
                )
                logger.error(
                    f"[P3-1/PARTIAL-ROLLBACK-ALERT] Alert persistido: user={user_id} "
                    f"restored={restored}/{total} failed_ids={failed_row_ids[:5]}..."
                )
            except Exception as alert_err:
                # Best-effort: si el alert mismo falla, NO queremos abortar
                # el flujo del usuario. El log warning de arriba ya capturó
                # el incident; el alert solo era para escalación SRE.
                logger.error(
                    f"[P3-1/PARTIAL-ROLLBACK-ALERT] Falló persistir alert "
                    f"para user={user_id}: {alert_err!r}"
                )
        return stats

    if not insert_failed:
        try:
            _count_row = execute_sql_query(
                "SELECT count(*) AS count FROM user_inventory "
                "WHERE user_id = %s AND source = 'shopping_list'",
                (user_id,),
                fetch_one=True,
            )
            stats["inserted_rows"] = int((_count_row or {}).get("count") or 0)
        except Exception:
            stats["inserted_rows"] = inserted_before  # best-effort
    return stats


def consume_inventory_items_completely(user_id: str, ingredient_names: List[str]):
    """
    Vacia el inventario físico (quantity = 0) para los ingredientes especificados.
    """
    if not _db_available() or not ingredient_names: return False
    try:
        names_lower = [n.lower().strip() for n in ingredient_names]

        if names_lower:
            execute_sql_write(
                "UPDATE user_inventory SET quantity = 0 WHERE user_id = %s AND LOWER(TRIM(ingredient_name)) = ANY(%s)",
                (user_id, names_lower)
            )
        return True
    except Exception as e:
        logger.error(f"Error vaciando ingredientes (consumo completo) para {user_id}: {e}")
        return False


def sync_inventory_after_chunk_completion(
    user_id: str,
    chunk_window_start_iso: str,
    chunk_window_end_iso: str,
) -> Dict[str, int]:
    """
    [P0.1] Reconcilia user_inventory contra consumed_meals al cerrar un chunk.

    Para cada fila de consumed_meals del usuario en la ventana
    [chunk_window_start, chunk_window_end) cuyas ingredients aún no se han
    descontado (`inventory_synced_at IS NULL`), deduce los ingredientes y marca
    la fila como sincronizada. Idempotente: si el chunk se reprocesa o el cron
    pasa dos veces, las filas ya marcadas se omiten gracias al filtro WHERE.

    Args:
        user_id: usuario dueño de las consumed_meals.
        chunk_window_start_iso: inicio de la ventana del chunk (UTC ISO 8601).
        chunk_window_end_iso: fin exclusivo de la ventana del chunk.

    Returns:
        {"reconciled_count": N, "items_deducted": M} — N filas procesadas, M
        items individuales (líneas de ingrediente) restados del inventario.
    """
    stats = {"reconciled_count": 0, "items_deducted": 0}
    if not user_id or not chunk_window_start_iso or not chunk_window_end_iso:
        return stats

    try:
        # Import local: el `connection_pool` real vive en db_core; importarlo
        # aquí (en lugar de a nivel de módulo) facilita patcheo en tests vía
        # `patch("db_core.connection_pool", ...)`.
        from db_core import connection_pool as _pool, execute_sql_query as _q
        if _pool:
            rows = _q(
                """
                SELECT id, ingredients, consumed_at
                FROM consumed_meals
                WHERE user_id = %s
                  AND consumed_at >= %s
                  AND consumed_at <  %s
                  AND inventory_synced_at IS NULL
                """,
                (user_id, chunk_window_start_iso, chunk_window_end_iso),
                fetch_all=True,
            ) or []
        else:
            # [P1-NEON-DB-MIGRATION · 2026-06-12] Eliminada la rama legacy
            # `elif supabase:` (PostgREST). Sin pool no hay DB de datos.
            return stats
    except Exception as e:
        logger.warning(
            f"[P0.1/PANTRY-SYNC] Error consultando consumed_meals pendientes "
            f"para user {user_id}: {e}"
        )
        return stats

    if not rows:
        return stats

    for row in rows:
        row_id = row.get("id")
        raw_ingredients = row.get("ingredients") or []

        # ingredients puede venir como list[str] (path actual) o como JSON string
        # legacy. Normalizamos a list[str] sin romper en payloads inesperados.
        if isinstance(raw_ingredients, str):
            try:
                raw_ingredients = json.loads(raw_ingredients)
            except Exception:
                raw_ingredients = []
        if not isinstance(raw_ingredients, list):
            raw_ingredients = []

        ingredients_list = [str(i) for i in raw_ingredients if i and len(str(i).strip()) >= 3]

        # Si la fila no tiene ingredientes parseables, igual la marcamos sincronizada
        # para que no la reintentemos en cada cierre de chunk (ej. logs manuales del
        # frontend que no envían lista de ingredientes).
        try:
            if ingredients_list:
                # [P3-CHUNK-RECONCILE-TELEMETRY · 2026-05-30] Capturar el summary
                # de deduct para contar lo REALMENTE deducido (succeeded+inferred),
                # no el len aspiracional de ingredients_list (pre-fix la telemetría
                # del cron [P0.1/PANTRY-SYNC] mentía sobre cuánto se dedujo). Los
                # fallos ya se persisten en failed_inventory_deductions dentro de
                # la propia deduct (replay SSOT). El marcado de inventory_synced_at
                # sigue siendo INCONDICIONAL e intencional (filas sin ingredientes
                # parseables NO deben reintentarse cada cierre de chunk) — solo
                # corregimos la exactitud del contador.
                _summary = deduct_consumed_meal_from_inventory(user_id, ingredients_list)
                if isinstance(_summary, dict):
                    stats["items_deducted"] += (
                        len(_summary.get("succeeded", [])) + len(_summary.get("inferred", []))
                    )
                    _failed = _summary.get("failed_to_deduct", [])
                    if _failed:
                        stats["items_failed"] = stats.get("items_failed", 0) + len(_failed)
                else:
                    # Caller legacy / early-return None: conservar el conteo previo.
                    stats["items_deducted"] += len(ingredients_list)

            execute_sql_write(
                "UPDATE consumed_meals SET inventory_synced_at = NOW() WHERE id = %s",
                (row_id,),
            )
            stats["reconciled_count"] += 1
        except Exception as row_err:
            logger.warning(
                f"[P0.1/PANTRY-SYNC] Error reconciliando consumed_meals.id={row_id} "
                f"para user {user_id}: {row_err}"
            )
            continue

    return stats


# ============================================================
# [P3-DEPLETED-BD · 2026-05-22] Helpers de `user_depleted_items`.
# ============================================================
# Tabla nueva (migration `p3_user_depleted_items_2026_05_22.sql`) que persiste
# el estado "agotado" del Pantry cross-device. Sustituye el localStorage
# pre-existente. Tooltip-anchor: P3-DEPLETED-BD-HELPERS.

def add_depleted_item(
    user_id: str,
    *,
    ingredient_name: str,
    quantity: float = 1.0,
    unit: str = "unidad",
    master_ingredient_id: Optional[str] = None,
    category: Optional[str] = None,
    shelf_life_days: Optional[int] = None,
    depleted_at: Optional[str] = None,
) -> bool:
    """[P3-DEPLETED-BD · 2026-05-22] Upsert un item agotado en `user_depleted_items`.

    Idempotente: usa los unique indexes partial sobre `(user_id,
    master_ingredient_id)` cuando master no es NULL y `(user_id,
    lower(trim(ingredient_name)))` cuando es NULL — re-agotar el mismo item
    actualiza `depleted_at` y `quantity` snapshot.

    Filtro `WHERE user_id = %s` defensa-en-profundidad (RLS owner-only ya
    aplica, pero el backend conecta como `postgres` que bypassea RLS).

    Returns:
        True si la operación tuvo éxito, False si la DB no está disponible o
        excepción.
    """
    if not _db_available() or not user_id or not ingredient_name:
        return False
    try:
        # ON CONFLICT con dos unique indexes partial — Postgres elige el
        # que aplique según si master_ingredient_id es NULL o NOT NULL.
        # Mantenemos dos branches para que el SQL sea explícito.
        if master_ingredient_id:
            # [P3-DEPLETED-DUAL-KEY · 2026-05-30] Borrar la fila name-keyed
            # huérfana (master_ingredient_id NULL) del MISMO ingrediente antes
            # del upsert master-keyed. Los dos unique indexes parciales
            # (user,master) y (user,lower(name)) WHERE master IS NULL son
            # disjuntos: sin esto, el mismo ingrediente lógico puede vivir como
            # DOS filas — una creada por la migración one-shot del localStorage
            # (legacy sin master_id) y otra por el chat agent / Pantry que SÍ
            # resuelve master_id → "Maní" duplicado en Agotados, y un solo
            # delete_depleted_item borra solo una. Best-effort: si falla, el
            # upsert master procede igual (no bloquea el flujo principal).
            try:
                execute_sql_write(
                    "DELETE FROM user_depleted_items "
                    "WHERE user_id = %s::uuid AND master_ingredient_id IS NULL "
                    "AND lower(trim(ingredient_name)) = lower(trim(%s))",
                    (user_id, ingredient_name),
                )
            except Exception as _dual_err:
                logger.debug(
                    f"[P3-DEPLETED-DUAL-KEY] cleanup de fila name-keyed huérfana "
                    f"falló (best-effort) user={user_id} name={ingredient_name!r}: {_dual_err}"
                )
            query = """
                INSERT INTO user_depleted_items (
                    user_id, master_ingredient_id, ingredient_name,
                    quantity, unit, category, shelf_life_days, depleted_at
                )
                VALUES (%s::uuid, %s::uuid, %s, %s, %s, %s, %s, COALESCE(%s::timestamptz, NOW()))
                ON CONFLICT (user_id, master_ingredient_id)
                WHERE master_ingredient_id IS NOT NULL
                DO UPDATE SET
                    ingredient_name = EXCLUDED.ingredient_name,
                    quantity = EXCLUDED.quantity,
                    unit = EXCLUDED.unit,
                    category = COALESCE(EXCLUDED.category, user_depleted_items.category),
                    shelf_life_days = COALESCE(EXCLUDED.shelf_life_days, user_depleted_items.shelf_life_days),
                    depleted_at = EXCLUDED.depleted_at
            """
            execute_sql_write(query, (
                user_id, master_ingredient_id, ingredient_name,
                float(quantity), unit, category, shelf_life_days, depleted_at,
            ))
        else:
            query = """
                INSERT INTO user_depleted_items (
                    user_id, ingredient_name,
                    quantity, unit, category, shelf_life_days, depleted_at
                )
                VALUES (%s::uuid, %s, %s, %s, %s, %s, COALESCE(%s::timestamptz, NOW()))
                ON CONFLICT (user_id, (lower(trim(ingredient_name))))
                WHERE master_ingredient_id IS NULL
                DO UPDATE SET
                    quantity = EXCLUDED.quantity,
                    unit = EXCLUDED.unit,
                    category = COALESCE(EXCLUDED.category, user_depleted_items.category),
                    shelf_life_days = COALESCE(EXCLUDED.shelf_life_days, user_depleted_items.shelf_life_days),
                    depleted_at = EXCLUDED.depleted_at
            """
            execute_sql_write(query, (
                user_id, ingredient_name,
                float(quantity), unit, category, shelf_life_days, depleted_at,
            ))
        return True
    except Exception as e:
        logger.warning(
            f"[P3-DEPLETED-BD] add_depleted_item falló user={user_id} "
            f"name={ingredient_name!r}: {e}"
        )
        return False


def list_depleted_items(user_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    """[P3-DEPLETED-BD · 2026-05-22] Lista los items agotados del usuario,
    ordenados por `depleted_at DESC` (más reciente primero).

    Returns:
        Lista de dicts con `{id, master_ingredient_id, ingredient_name,
        quantity, unit, category, shelf_life_days, depleted_at}`. Lista
        vacía si la DB no está disponible o el user no tiene agotados.
    """
    if not _db_available() or not user_id:
        return []
    limit = max(1, min(int(limit), 500))
    try:
        # [P1-NEON-DB-MIGRATION · 2026-06-12] uuid→text, numeric→float8 y
        # depleted_at→string ISO-8601 con "T" (to_jsonb): las filas van al
        # frontend vía /api/plans/depleted-items y `new Date(...)` de Safari
        # no parsea el formato espacio-separado de `::text`.
        rows = execute_sql_query(
            "SELECT id, master_ingredient_id::text AS master_ingredient_id, "
            "ingredient_name, quantity::float8 AS quantity, unit, category, "
            "shelf_life_days, to_jsonb(depleted_at) #>> '{}' AS depleted_at "
            "FROM user_depleted_items WHERE user_id = %s "
            "ORDER BY depleted_at DESC LIMIT %s",
            (user_id, limit),
            fetch_all=True,
        )
        return rows or []
    except Exception as e:
        logger.warning(f"[P3-DEPLETED-BD] list_depleted_items falló user={user_id}: {e}")
        return []


def delete_depleted_item(
    user_id: str,
    *,
    item_id: Optional[int] = None,
    master_ingredient_id: Optional[str] = None,
    ingredient_name: Optional[str] = None,
) -> bool:
    """[P3-DEPLETED-BD · 2026-05-22] DELETE un item de `user_depleted_items`.

    Una de las 3 keys debe ser provided (mutex). Filtro `user_id` defensa-en-
    profundidad. Returns True si una row fue eliminada, False si nada
    matched o excepción.
    """
    if not _db_available() or not user_id:
        return False
    if not (item_id or master_ingredient_id or ingredient_name):
        logger.warning("[P3-DEPLETED-BD] delete_depleted_item: ningún criterio dado")
        return False
    try:
        # [P1-NEON-DB-MIGRATION · 2026-06-12] RETURNING id reemplaza el
        # `res.data` de PostgREST como detector de "≥1 fila borrada". El match
        # por nombre usa lower(trim()) = lower(trim(%s)) — semántica
        # equivalente al `ilike` legacy sin el riesgo de que `%`/`_` en el
        # nombre actúen como wildcards (mismo patrón que bulk_delete).
        if item_id is not None:
            _q = ("DELETE FROM user_depleted_items WHERE user_id = %s AND id = %s "
                  "RETURNING id")
            _params = (user_id, int(item_id))
        elif master_ingredient_id is not None:
            _q = ("DELETE FROM user_depleted_items WHERE user_id = %s "
                  "AND master_ingredient_id = %s::uuid RETURNING id")
            _params = (user_id, master_ingredient_id)
        else:
            _q = ("DELETE FROM user_depleted_items WHERE user_id = %s "
                  "AND lower(trim(ingredient_name)) = lower(trim(%s)) RETURNING id")
            _params = (user_id, str(ingredient_name).strip())
        rows = execute_sql_write(_q, _params, returning=True)
        return bool(rows)
    except Exception as e:
        logger.warning(f"[P3-DEPLETED-BD] delete_depleted_item falló user={user_id}: {e}")
        return False


def delete_all_depleted_items(user_id: str) -> int:
    """[P3-DELETEALL-DEPLETED · 2026-05-30] Borra TODAS las filas de
    `user_depleted_items` del usuario. Lo usa 'Vaciar Nevera' (Pantry
    confirmDeleteAll): vaciar la nevera implica que los recordatorios de
    "agotado" también son obsoletos. Sin esto, el clear de `_persistDepleted([])`
    del frontend era puramente cosmético — la fuente de verdad es esta tabla
    (cross-device) y `_fetchAndApply` los repoblaba al próximo mount.

    Filtro `WHERE user_id = %s` defensa-en-profundidad. Returns: count de filas
    borradas (best-effort 0 si la DB no está disponible o excepción).
    """
    if not _db_available() or not user_id:
        return 0
    try:
        rows = execute_sql_write(
            "DELETE FROM user_depleted_items WHERE user_id = %s RETURNING id",
            (user_id,),
            returning=True,
        )
        return len(rows or [])
    except Exception as e:
        logger.warning(f"[P3-DELETEALL-DEPLETED] delete_all_depleted_items falló user={user_id}: {e}")
        return 0


def bulk_delete_depleted_items(user_id: str, names) -> int:
    """[P3-BACKEND-AUDIT · 2026-06-01] Borra en UN solo round-trip las filas de
    `user_depleted_items` del usuario cuyo `ingredient_name` matchee (case-insensitive,
    trim) cualquiera de `names`. Reemplaza el loop N+1 de `delete_depleted_item`
    por nombre en `/restock` (hasta ~N DELETEs secuenciales colapsados a 1).

    Preserva la semántica case-insensitive del helper singular vía
    `lower(trim(ingredient_name)) = ANY(%s)` con `names` normalizados — más robusto
    que `ilike` (que interpretaría `_`/`%` como wildcards). `WHERE user_id = %s`
    defensa-en-profundidad (invariante I2). Best-effort: returns count de nombres
    únicos solicitados (0 si supabase/pool no disponible o excepción). Anchor:
    P3-RESTOCK-BULK-DELETE.
    """
    if not user_id or not names:
        return 0
    norm = sorted({str(n).strip().lower() for n in names if n and str(n).strip()})
    if not norm:
        return 0
    try:
        execute_sql_write(
            "DELETE FROM user_depleted_items "
            "WHERE user_id = %s AND lower(trim(ingredient_name)) = ANY(%s)",
            (user_id, norm),
        )
        return len(norm)
    except Exception as e:
        logger.warning(f"[P3-RESTOCK-BULK-DELETE] bulk_delete_depleted_items falló user={user_id}: {e}")
        return 0


def bulk_upsert_depleted_items(user_id: str, items: List[Dict[str, Any]]) -> int:
    """[P3-DEPLETED-BD · 2026-05-22] Upsert batch de items agotados — usado
    por la migration one-shot del localStorage existente al BD (frontend
    POST-ea su localStorage en el primer mount post-deploy).

    Returns: count de items upserted exitosamente.
    """
    if not user_id or not items:
        return 0
    ok_count = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("ingredient_name") or item.get("name")
        if not name:
            continue
        # [P2-DEPLETED-QTY-VALIDATE · 2026-05-30] Coerción defensiva de `quantity`
        # ANTES de llamar add_depleted_item. El `float(...)` se evaluaba en el
        # frame del CALLER (argumento), FUERA del try/except interno de
        # add_depleted_item → un item legacy con quantity='½ lb'/'abc' (clase de
        # data corrupta del audit 2026-05-23, manipulable vía DevTools) lanzaba
        # ValueError que abortaba TODO el batch → HTTP 500 → la migración one-shot
        # (Pantry.jsx solo setea el flag migrated_at en resp.ok) re-POST-eaba en
        # CADA mount para siempre + perdía los items válidos del batch. Best-effort
        # per-item: quantity corrupta cae a 1.0 (consistente con el patrón del módulo).
        _raw_qty = item.get("quantity")
        try:
            qty = float(_raw_qty) if _raw_qty is not None else 1.0
            # NaN (qty != qty), ±Inf y negativos → fallback 1.0 (sin import math).
            if qty != qty or qty in (float("inf"), float("-inf")) or qty < 0:
                qty = 1.0
            qty = min(qty, 9999.0)  # cap defensivo contra valores absurdos
        except (TypeError, ValueError):
            qty = 1.0
        ok = add_depleted_item(
            user_id,
            ingredient_name=str(name),
            quantity=qty,
            unit=str(item.get("unit") or "unidad"),
            master_ingredient_id=item.get("master_ingredient_id"),
            category=item.get("category"),
            shelf_life_days=item.get("shelf_life_days"),
            depleted_at=item.get("depleted_at"),
        )
        if ok:
            ok_count += 1
    return ok_count

