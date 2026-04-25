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
) -> bool:
    existing = supabase.table("user_inventory").select(
        "id, quantity, unit, reserved_quantity, reservation_details"
    ).eq("user_id", user_id).eq("ingredient_name", ingredient_name).execute()

    master_list = get_master_ingredients()
    master_item = next((m for m in master_list if m["name"] == ingredient_name), {})

    for row in existing.data or []:
        current_unit = row.get("unit") or unit or "unidad"
        converted_qty = convert_amount(quantity, unit, current_unit, master_item)
        if converted_qty is None:
            continue

        reserved_quantity = float(row.get("reserved_quantity") or 0)
        current_quantity = float(row.get("quantity") or 0)
        reservation_details = _normalize_reservation_details(row.get("reservation_details"))
        previous = float(reservation_details.get(reservation_key) or 0)

        if release_only:
            if previous <= 0:
                return False
            reservation_details.pop(reservation_key, None)
            reserved_quantity = max(reserved_quantity - previous, 0.0)
        else:
            target_qty = round(max(converted_qty, 0.0), 4)
            if abs(previous - target_qty) < 0.0001:
                return True
            reserved_quantity = max(reserved_quantity - previous, 0.0)
            if target_qty > 0:
                available_after_other_reservations = max(current_quantity - reserved_quantity, 0.0)
                applied_qty = min(target_qty, available_after_other_reservations)
                if applied_qty <= 0:
                    return False
                reservation_details[reservation_key] = round(applied_qty, 4)
                reserved_quantity += applied_qty

        _update_row_reservation(row["id"], reserved_quantity, reservation_details)
        return True

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

def add_or_update_inventory_item(user_id: str, ingredient_name: str, quantity: float, unit: str):
    """
    Agrega o actualiza un ingrediente en la despensa del usuario.
    Resuelve empates de unidades (ej. si hay kg y pides restar g, restará del kg correctamente).
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
                        supabase.table("user_inventory").update({"quantity": new_qty, "master_ingredient_id": master_id}).eq("id", row_id).execute()
                    
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
                    "master_ingredient_id": master_id
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

    Retorna {mutations_count, last_mutation_at, low_stock_items} — el caller decide el umbral.
    """
    if not supabase or not user_id or not since_iso:
        return {"mutations_count": 0, "last_mutation_at": None, "low_stock_items": 0}
    try:
        res = (
            supabase.table("user_inventory")
            .select("id, ingredient_name, quantity, updated_at")
            .eq("user_id", user_id)
            .gte("updated_at", since_iso)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return {"mutations_count": 0, "last_mutation_at": None, "low_stock_items": 0}
        last_mutation = max((r.get("updated_at") for r in rows if r.get("updated_at")), default=None)
        # Items con stock muy bajo (≤ 0.5 en unidad arbitraria) sugieren consumo reciente
        low_stock = sum(1 for r in rows if float(r.get("quantity") or 0) <= 0.5)
        return {
            "mutations_count": len(rows),
            "last_mutation_at": last_mutation,
            "low_stock_items": low_stock,
        }
    except Exception as e:
        logger.warning(f"[P0-D] Error consultando actividad de inventario para {user_id}: {e}")
        return {"mutations_count": 0, "last_mutation_at": None, "low_stock_items": 0}


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
                add_or_update_inventory_item(user_id, name, -qty, unit)
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
                    res = add_or_update_inventory_item(user_id, name, qty, unit)
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
                res = add_or_update_inventory_item(user_id, name, qty, unit)
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

