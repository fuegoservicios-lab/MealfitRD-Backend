import logging
from datetime import datetime
from typing import List, Dict, Any
from db_core import supabase, execute_sql_write
from shopping_calculator import _parse_quantity, get_plural_unit, get_master_ingredients

logger = logging.getLogger(__name__)

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
        return res.data or []
    except Exception as e:
        logger.error(f"Error obteniendo user_inventory para {user_id}: {e}")
        return []

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
        qty = float(item.get("quantity", 0))
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

def merge_inventory_after_rotation(user_id: str, plan_data: dict) -> int:
    """
    Merge inteligente post-rotación: sincroniza la Nevera con el nuevo plan
    SIN borrar el inventario existente.
    
    - Ingredientes que ya están en la Nevera → no se tocan (respeta cantidades del usuario)
    - Ingredientes nuevos del plan → se insertan con la cantidad del plan
    - Ingredientes manuales del usuario → se preservan intactos
    
    Returns: número de ingredientes nuevos añadidos.
    """
    if not supabase or not plan_data:
        return 0
    
    from shopping_calculator import normalize_name
    
    # 1. Leer la lista de compras del nuevo plan — preferir la lista del ciclo actual
    # Cascada: weekly (7d) > biweekly (15d) > monthly (30d) > genérica (puede estar inflada)
    shopping_list = None
    source_key = "aggregated_shopping_list"
    
    for key in ("aggregated_shopping_list_weekly", "aggregated_shopping_list_biweekly", "aggregated_shopping_list_monthly", "aggregated_shopping_list"):
        candidate = plan_data.get(key, [])
        if candidate and isinstance(candidate, list) and len(candidate) > 0:
            shopping_list = candidate
            source_key = key
            break
    
    if not shopping_list:
        logger.info(f"🔄 [MERGE] Sin lista de compras en el plan, omitiendo merge para {user_id}")
        return 0
    
    logger.info(f"🔄 [MERGE] Usando '{source_key}' ({len(shopping_list)} items) para merge de {user_id}")
    
    # 2. Leer inventario actual (incluyendo items con qty > 0)
    try:
        existing_res = supabase.table("user_inventory") \
            .select("ingredient_name") \
            .eq("user_id", user_id) \
            .gt("quantity", 0) \
            .execute()
        existing_names = set()
        if existing_res.data:
            existing_names = {row["ingredient_name"].lower().strip() for row in existing_res.data}
    except Exception as e:
        logger.error(f"❌ [MERGE] Error leyendo inventario actual para {user_id}: {e}")
        return 0
    
    # 3. Filtrar: solo procesar ingredientes que NO están ya en la Nevera
    items_added = 0
    UNIT_NORMALIZE = _CANONICAL_UNIT_MAP
    
    for item in shopping_list:
        try:
            # Solo procesamos objetos estructurados (formato moderno)
            if not isinstance(item, dict) or "name" not in item:
                continue
            
            name = normalize_name(item["name"])
            if not name:
                continue
            
            # Si ya existe en la Nevera, no lo tocamos
            if name.lower().strip() in existing_names:
                continue
            
            qty = float(item.get("quantity", item.get("market_qty", 0)) or 0)
            unit = item.get("unit", item.get("market_unit", "unidad")) or "unidad"
            unit_lower = unit.lower().rstrip('.')
            unit = UNIT_NORMALIZE.get(unit_lower, unit_lower if unit_lower else 'unidad')
            
            if qty <= 0:
                continue
            
            # Insertar el nuevo ingrediente
            result = add_or_update_inventory_item(user_id, name, qty, unit)
            if result:
                items_added += 1
                # Marcar como existente para evitar duplicados dentro del mismo batch
                existing_names.add(name.lower().strip())
                
        except Exception as e:
            logger.error(f"⚠️ [MERGE] Error procesando item '{item}': {e}")
            continue
    
    logger.info(f"🔄 [MERGE] Post-rotación: {items_added} ingredientes nuevos añadidos a la Nevera de {user_id} (de {len(shopping_list)} en el plan)")
    return items_added


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

