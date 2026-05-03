import json
import logging
import re
import unicodedata
from datetime import datetime
from typing import List, Dict, Any
from db_core import supabase, execute_sql_write
from shopping_calculator import _parse_quantity, get_plural_unit, get_master_ingredients
from constants import normalize_ingredient_for_tracking

logger = logging.getLogger(__name__)
_RESERVATION_MEAL_TOKEN_RE = re.compile(r":meal:(.+)$")

_CANONICAL_UNIT_MAP = {
    'ud': 'unidad', 'uds': 'unidad', 'unidades': 'unidad',
    'lbs': 'lb', 'libra': 'lb', 'libras': 'lb',
    'paquetes': 'paquete', 'potes': 'pote', 'mazos': 'mazo',
    'cabezas': 'cabeza', 'sobres': 'sobre', 'latas': 'lata',
    'botellas': 'botella', 'hojas': 'hoja', 'rebanadas': 'rebanada',
    'dientes': 'diente', 'gramos': 'g', 'gr': 'g',
    'kilos': 'kg', 'kilogramo': 'kg', 'kilogramos': 'kg',
    'onzas': 'oz', 'onza': 'oz',
    'tazas': 'taza', 'cucharada': 'cda', 'cucharadas': 'cda',
    'cucharadita': 'cdta', 'cucharaditas': 'cdta',
    'al gusto': 'pizca',
    'cartón': 'paquete', 'carton': 'paquete', 'cartones': 'paquete',
    'fundita': 'paquete', 'funditas': 'paquete', 'funda': 'paquete', 'fundas': 'paquete',
    'envase': 'pote', 'envases': 'pote',
}

def get_raw_user_inventory(user_id: str) -> List[Dict[str, Any]]:
    """Obtiene los registros crudos de la base de datos para la despensa del usuario."""
    if not supabase: return []
    try:
        res = supabase.table("user_inventory").select("*").eq("user_id", user_id).gt("quantity", 0).execute()
        rows = res.data or []
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
    supabase.table("user_inventory").update({
        "reserved_quantity": round(max(reserved_quantity, 0.0), 4),
        "reservation_details": reservation_details,
    }).eq("id", row_id).execute()


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
    res = (
        supabase.table("user_inventory")
        .update({
            "reserved_quantity": rounded_new,
            "reservation_details": reservation_details,
        })
        .eq("id", row_id)
        .eq("reserved_quantity", rounded_expected)
        .execute()
    )
    return bool(getattr(res, "data", None))

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


def get_user_inventory(user_id: str, household_size: int = None) -> List[str]:
    """Obtiene la despensa del usuario formateada como lista de strings (ej: '2 unidades de Manzana').

    Si household_size no se pasa, se lee del health_profile del usuario para escalar la
    predicción de agotamiento (P0-3). Callers que ya lo tienen pueden pasarlo directamente.
    """
    raw_items = get_raw_user_inventory(user_id)
    formatted = []

    if household_size is None:
        try:
            res = supabase.table("user_profiles").select("health_profile").eq("id", user_id).limit(1).execute()
            if res.data:
                hp = res.data[0].get("health_profile") or {}
                household_size = hp.get("householdSize") or hp.get("household_size") or 1
        except Exception:
            household_size = 1
    household_size = max(1, int(household_size or 1))

    PANTRY_STAPLES = {
        'Sal y ajo en polvo', 'Aceite de oliva', 'Aceite de coco',
        'Aceite de sésamo o maní', 'Salsa de soya', 'Orégano',
        'Canela', 'Pimienta', 'Sal', 'Vinagre', 'Ajo en polvo'
    }

    master_list = get_master_ingredients()
    master_map = {m["name"]: m for m in master_list}
    
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

def get_user_inventory_net(user_id: str, household_size: int = None) -> List[str]:
    """Obtiene la despensa del usuario descontando las reservas activas (Neto)."""
    raw_items = get_raw_user_inventory(user_id)
    formatted = []

    if household_size is None:
        try:
            res = supabase.table("user_profiles").select("health_profile").eq("id", user_id).limit(1).execute()
            if res.data:
                hp = res.data[0].get("health_profile") or {}
                household_size = hp.get("householdSize") or hp.get("household_size") or 1
        except Exception:
            household_size = 1
    household_size = max(1, int(household_size or 1))

    PANTRY_STAPLES = {
        'Sal y ajo en polvo', 'Aceite de oliva', 'Aceite de coco',
        'Aceite de sésamo o maní', 'Salsa de soya', 'Orégano',
        'Canela', 'Pimienta', 'Sal', 'Vinagre', 'Ajo en polvo'
    }

    master_list = get_master_ingredients()
    master_map = {m["name"]: m for m in master_list}
    
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

def convert_amount(qty: float, from_unit: str, to_unit: str, master_item: dict) -> float:
    """Intenta convertir matemáticamente una cantidad de una unidad a otra usando factores y densidades."""
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
        
    # Cross domain conversions require density
    density = float(master_item.get("density_g_per_cup") or 150.0)
    
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
        
    # 5. Count to Mass or Volume (Estimate)
    if from_unit_lower in count_units and to_unit_lower in mass_to_g:
        g_per_u = float(master_item.get("density_g_per_unit") or 100.0)
        if 'rebanada' in from_unit_lower: g_per_u = 25.0
        g = qty * g_per_u
        return g / mass_to_g[to_unit_lower]
        
    if from_unit_lower in mass_to_g and to_unit_lower in count_units:
        g_per_u = float(master_item.get("density_g_per_unit") or 100.0)
        if 'rebanada' in to_unit_lower: g_per_u = 25.0
        g = qty * mass_to_g[from_unit_lower]
        return g / g_per_u
        
    # Incompatibles
    return None


def _apply_reservation_delta(
    user_id: str,
    ingredient_name: str,
    quantity: float,
    unit: str,
    reservation_key: str,
    *,
    release_only: bool = False,
    max_retries: int = 4,
) -> bool:
    """[P0-4] Reserva atómica vía CAS-with-retry.

    Antes el flujo era: SELECT row → modificar en Python → UPDATE WHERE id=X. La
    ventana entre SELECT y UPDATE permitía lost-update si dos writers concurrentes
    leían el mismo `reserved_quantity` y ambos sumaban su delta — el segundo
    sobreescribía al primero, perdiendo una reserva sin rastro. En la práctica
    esto generaba `reserved_quantity` que no sumaba con sus `reservation_details`,
    e ingredientes "consumidos" que el sistema seguía considerando disponibles.

    Ahora el UPDATE incluye `eq("reserved_quantity", expected_old)` como CAS token.
    Si entre nuestro SELECT y nuestro UPDATE alguien más modificó la fila, el
    UPDATE matchea 0 rows y reintentamos el ciclo completo desde el SELECT.

    El loop reintenta hasta `max_retries` veces con backoff exponencial corto.
    Si tras todos los intentos sigue habiendo conflicto, devolvemos False y
    logueamos para que el caller reaccione (típicamente reintentar el chunk).
    """
    import time

    master_list = get_master_ingredients()
    master_item = next((m for m in master_list if m["name"] == ingredient_name), {})

    last_compatible_row_id: str | None = None
    for attempt in range(max_retries):
        existing = supabase.table("user_inventory").select(
            "id, quantity, unit, reserved_quantity, reservation_details"
        ).eq("user_id", user_id).eq("ingredient_name", ingredient_name).execute()

        rows = getattr(existing, "data", None) or []
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
    """Reserva ingredientes de un chunk confirmado para que el siguiente vea stock disponible real."""
    if not supabase or not days:
        return 0

    reserved_items = 0
    for day in days:
        for meal in (day or {}).get("meals", []):
            reservation_key = _make_reservation_key(chunk_id, meal.get("name", ""))
            for item in meal.get("ingredients", []) or []:
                if not item or len(str(item).strip()) < 3:
                    continue
                try:
                    qty, unit, name = _parse_quantity(str(item))
                    if name and qty > 0 and _apply_reservation_delta(user_id, name, qty, unit, reservation_key):
                        reserved_items += 1
                except Exception as e:
                    logger.error(f"Error reservando '{item}' para {user_id}: {e}")
    return reserved_items


def release_meal_reservation(user_id: str, meal_name: str) -> int:
    """Libera reservas asociadas a una comida rechazada."""
    if not supabase or not user_id or not meal_name:
        return 0

    released = 0
    try:
        res = supabase.table("user_inventory").select(
            "id, reserved_quantity, reservation_details"
        ).eq("user_id", user_id).gt("reserved_quantity", 0).execute()
        for row in res.data or []:
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

    # 1. Read snapshot de filas con reservas. Path supabase + fallback a SQL crudo si no
    # hay client (p. ej. tests). Si ambos faltan, no hay nada que liberar.
    rows: List[Dict[str, Any]] = []
    if supabase:
        try:
            res = supabase.table("user_inventory").select(
                "id, reserved_quantity, reservation_details"
            ).eq("user_id", user_id).gt("reserved_quantity", 0).execute()
            rows = res.data or []
        except Exception as e:
            logger.error(f"[P1-A] SELECT inicial falló para chunk {chunk_id} user {user_id}: {e}")
            return 0
    else:
        try:
            from db_core import execute_sql_query
            rows = execute_sql_query(
                "SELECT id, reserved_quantity, reservation_details FROM user_inventory "
                "WHERE user_id = %s AND reserved_quantity > 0",
                (user_id,),
                fetch_all=True,
            ) or []
        except Exception as e:
            logger.error(f"[P1-A] SELECT crudo falló para chunk {chunk_id} user {user_id}: {e}")
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
        queries = [
            (
                "UPDATE user_inventory SET reserved_quantity = %s, reservation_details = %s WHERE id = %s",
                (spec["new_reserved"], Jsonb(spec["new_details"]), spec["row_id"]),
            )
            for spec in update_specs
        ]
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
    if not supabase or quantity <= 0:
        return False

    existing = supabase.table("user_inventory").select(
        "id, unit, reserved_quantity, reservation_details"
    ).eq("user_id", user_id).eq("ingredient_name", ingredient_name).gt("reserved_quantity", 0).execute()

    master_list = get_master_ingredients()
    master_item = next((m for m in master_list if m["name"] == ingredient_name), {})

    for row in existing.data or []:
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
    """
    if not supabase: return False
    try:
        # Extraemos sin filtrar por 'unit' para buscar compatibles
        existing = supabase.table("user_inventory").select("id, quantity, unit").eq("user_id", user_id).eq("ingredient_name", ingredient_name).execute()

        master_list = get_master_ingredients()
        master_item = next((m for m in master_list if m["name"] == ingredient_name), {})
        master_id_raw = master_item.get("id", None) if master_item else None
        master_id = str(master_id_raw) if master_id_raw is not None else None

        updated = False

        if existing.data:
            # Buscar el primer registro temporalmente compatible
            for row in existing.data:
                row_id = row["id"]
                current_qty = float(row["quantity"])
                current_unit = row["unit"]

                converted_qty = convert_amount(quantity, unit, current_unit, master_item)

                if converted_qty is not None:
                    # Logramos convertir y emparejar. Aplicar suma/resta.
                    new_qty = round(current_qty + converted_qty, 4)  # Evitar floating point overflow

                    if new_qty < 0.01:
                        supabase.table("user_inventory").delete().eq("id", row_id).execute()
                    else:
                        # [P0.2] UPDATE NO toca `source` para preservar provenance
                        # (first-writer-wins). Si la fila se creó como 'manual',
                        # un restock posterior que sume aquí mantiene 'manual'.
                        supabase.table("user_inventory").update({"quantity": new_qty, "master_ingredient_id": master_id, "last_mutation_type": mutation_type}).eq("id", row_id).execute()

                    updated = True
                    break

        # Si no se encontró un registro compatible o no existía, insertamos uno nuevo
        if not updated:
            if quantity >= 0.01:
                supabase.table("user_inventory").insert({
                    "user_id": user_id,
                    "ingredient_name": ingredient_name,
                    "quantity": round(quantity, 4),  # Evitar floating point overflow
                    "unit": unit,
                    "master_ingredient_id": master_id,
                    "last_mutation_type": mutation_type,
                    "source": source,
                }).execute()
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
    if not supabase or not user_id or not since_iso:
        return {"mutations_count": 0, "last_mutation_at": None, "low_stock_items": 0, "consumption_mutations_count": 0, "manual_mutations_count": 0}
    try:
        res = (
            supabase.table("user_inventory")
            .select("id, ingredient_name, quantity, updated_at, last_mutation_type")
            .eq("user_id", user_id)
            .gte("updated_at", since_iso)
            .execute()
        )
        rows = res.data or []
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


def deduct_consumed_meal_from_inventory(user_id: str, ingredients_list: List[str]):
    """
    Resta matemáticamente una lista de ingredientes crudos (los de una comida consumida)
    de la tabla de inventario físico.
    """
    if not supabase or not ingredients_list: return
    
    for item in ingredients_list:
        if not item or len(item) < 3: continue
        try:
            qty, unit, name = _parse_quantity(item)
            if name and qty > 0:
                _consume_reserved_inventory(user_id, name, qty, unit)
                # Actualizar restando
                add_or_update_inventory_item(user_id, name, -qty, unit, mutation_type="consumption")
        except Exception as e:
            logger.error(f"Error parseando y restando '{item}' de despensa física: {e}")

def restock_inventory(user_id: str, ingredients_list: list):
    """
    Agrega matemáticamente una lista de ingredientes (como una lista de compras)
    al inventario físico actual, sumando cantidades según sus unidades base.
    
    Acepta tanto strings legados como objetos estructurados {name, quantity, unit}.
    """
    if not supabase or not ingredients_list: return False
    
    from shopping_calculator import normalize_name
    
    # 🔄 [MEJORA] Ya NO borramos el inventario completo. El frontend aplica Delta Shopping
    # para enviar solo ingredientes que el usuario no tiene, preservando items manuales.
    # add_or_update_inventory_item() maneja el upsert individual (suma si existe, inserta si no).
    
    success = True
    items_saved = 0
    for item in ingredients_list:
        try:
            # Ruta Estructurada: el frontend envía {name, quantity, unit} directamente
            if isinstance(item, dict) and "name" in item:
                name = normalize_name(item["name"])
                qty = float(item.get("quantity", 0))
                unit = item.get("unit", "unidad")
                # Normalizar unidades de display a canónicas de inventario
                unit_lower = unit.lower().rstrip('.')
                UNIT_NORMALIZE = _CANONICAL_UNIT_MAP
                unit = UNIT_NORMALIZE.get(unit_lower, unit_lower if unit_lower else 'unidad')
                if not name:
                    continue
                if qty > 0:
                    # [P0.2] source='shopping_list' marca la fila para que la lógica
                    # de "MERGE inteligente" pueda distinguirla de items manuales.
                    res = add_or_update_inventory_item(user_id, name, qty, unit, source='shopping_list')
                    if res:
                        items_saved += 1
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
                else:
                    success = False
        except Exception as e:
            logger.error(f"Error parseando y sumando '{item}' a la despensa física: {e}")
            success = False
    
    logger.info(f"📦 [RESTOCK] {items_saved}/{len(ingredients_list)} ingredientes guardados para {user_id}")
    
    # Si ningún ingrediente fue guardado, es un fallo (no marcar plan como restocked)
    if items_saved == 0:
        return False

    return success


def replace_shopping_list_only_items(user_id: str, ingredients_list: list) -> Dict[str, int]:
    """
    [P0.2] MERGE inteligente: limpia los items con source='shopping_list' del
    usuario y los reemplaza con la nueva lista de compras. Items con
    source='manual' (o 'unknown') quedan intactos.

    Útil cuando se quiere "resetear" la lista de compras al registrar una nueva
    sin destruir lo que el usuario haya añadido a mano. Diferente de
    `restock_inventory` (que sólo suma): aquí los shopping_list previos se
    descartan completamente porque la nueva lista los reemplaza por diseño.

    Returns:
        {"deleted_shopping_rows": N, "inserted_rows": M, "preserved_manual_rows": K}
    """
    stats = {"deleted_shopping_rows": 0, "inserted_rows": 0, "preserved_manual_rows": 0}
    if not supabase or not user_id:
        return stats

    try:
        preserved = (
            supabase.table("user_inventory")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .neq("source", "shopping_list")
            .execute()
        )
        stats["preserved_manual_rows"] = int(getattr(preserved, "count", 0) or len(preserved.data or []))

        deleted = (
            supabase.table("user_inventory")
            .delete(count="exact")
            .eq("user_id", user_id)
            .eq("source", "shopping_list")
            .execute()
        )
        stats["deleted_shopping_rows"] = int(getattr(deleted, "count", 0) or 0)
    except Exception as e:
        logger.error(
            f"[P0.2] Error eliminando items shopping_list para {user_id}: {e}. "
            f"Abortando reemplazo para no dejar inventario inconsistente."
        )
        return stats

    if not ingredients_list:
        return stats

    inserted_before = stats["inserted_rows"]
    res = restock_inventory(user_id, ingredients_list)
    if res:
        try:
            count_res = (
                supabase.table("user_inventory")
                .select("id", count="exact")
                .eq("user_id", user_id)
                .eq("source", "shopping_list")
                .execute()
            )
            stats["inserted_rows"] = int(getattr(count_res, "count", 0) or 0)
        except Exception:
            stats["inserted_rows"] = inserted_before  # best-effort
    return stats


def consume_inventory_items_completely(user_id: str, ingredient_names: List[str]):
    """
    Vacia el inventario físico (quantity = 0) para los ingredientes especificados.
    """
    if not supabase or not ingredient_names: return False
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
        elif supabase:
            res = (
                supabase.table("consumed_meals")
                .select("id, ingredients, consumed_at")
                .eq("user_id", user_id)
                .gte("consumed_at", chunk_window_start_iso)
                .lt("consumed_at", chunk_window_end_iso)
                .is_("inventory_synced_at", "null")
                .execute()
            )
            rows = res.data or []
        else:
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
                deduct_consumed_meal_from_inventory(user_id, ingredients_list)
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

