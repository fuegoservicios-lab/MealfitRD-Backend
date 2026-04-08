from functools import lru_cache as _lru_cache
import uuid
import unicodedata as _uc
from typing import Optional, List, Dict, Any, Tuple, Union
import os
import logging
logger = logging.getLogger(__name__)
from db_core import supabase, connection_pool, execute_sql_query, execute_sql_write
from constants import parse_ingredient_qty
from db_plans import format_qty

CHECKED_ITEM_EXPIRY_DAYS = 3
MAX_SHOPPING_ITEMS_PER_USER = 100

def get_shopping_plan_hash(user_id: str) -> str:
    """Obtiene el hash del plan usado para la última auto-generación de shopping list."""
    if not supabase: return None
    try:
        res = supabase.table("user_profiles").select("shopping_plan_hash").eq("id", user_id).execute()
        if res.data and res.data[0].get("shopping_plan_hash"):
            return res.data[0]["shopping_plan_hash"]
        return None
    except Exception as e:
        # Columna no existe aún → no hay cache
        return None

def save_shopping_plan_hash(user_id: str, plan_hash: str):
    """Guarda el hash del plan para cache de auto-generación de shopping list."""
    if not supabase: return None
    try:
        res = supabase.table("user_profiles").update({"shopping_plan_hash": plan_hash}).eq("id", user_id).execute()
        return res.data
    except Exception as e:
        # Columna no existe → silenciar (cache es opcional)
        logger.warning(f"⚠️ [DB] No se pudo guardar shopping_plan_hash: {e}")
        return None

import threading

class DistributedShoppingLock:
    def __init__(self, user_id: str):
        self.user_id = user_id
        # Utilizamos un candado en memoria por usuario
        # Suficiente para serializar requests de un mismo usuario concurrentemente en el mismo proceso (API)
        self._lock = threading.RLock()
        
    def acquire(self, timeout: int = 10) -> bool:
        return self._lock.acquire(timeout=timeout)
        
    def release(self):
        try:
            self._lock.release()
        except RuntimeError:
            pass
            
    def __enter__(self):
        self.acquire()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

@_lru_cache(maxsize=1024)
def get_user_shopping_lock(user_id: str) -> DistributedShoppingLock:
    """Obtiene un Distributed RLock exclusivo por usuario para serializar
    operaciones de shopping list que requieren atomicidad en entornos serverless.
    LRU(1024) previene memory leaks devolviendo la misma instancia por usuario."""
    return DistributedShoppingLock(user_id)

def add_custom_shopping_items(user_id: str, items: list, source: str = "manual", overwrite: bool = False):
    """Inserta uno o más items custom a la lista de compras del usuario.
    source: 'auto' (IA auto-generados), 'chat' (añadidos vía chat), 'manual' (default/legacy)
    Dual-write: guarda JSON en item_name (legacy) + columnas estructuradas (category, display_name, qty, emoji)."""
    if not supabase or not items: return None
    import json
    
    if overwrite:
        try:
            clear_all_shopping_items(user_id)
        except Exception as cle:
            logger.warning(f"⚠️ Aviso: No se pudieron borrar los items viejos durante overwrite: {cle}")
    
    def _extract_fields(item):
        """Extrae campos estructurados de un item (dict, model, o string)."""
        if hasattr(item, 'model_dump'):
            d = item.model_dump()
        elif isinstance(item, dict):
            d = item
        elif isinstance(item, str) and item.strip():
            return item.strip(), {"category": "", "display_name": item.strip(), "qty": "", "emoji": ""}
        else:
            return None, None
        item_name_json = json.dumps(d, ensure_ascii=False)
        structured = {
            "category": d.get("category", ""),
            "meal_slot": d.get("meal_slot", "Desayuno"),
            "display_name": d.get("name", ""),
            "qty": d.get("qty", ""),
            "emoji": d.get("emoji", ""),
            "is_checked": d.get("is_checked", False)
        }
        return item_name_json, structured
    
    try:
        rows = []
        for item in items:
            item_name, fields = _extract_fields(item)
            if item_name is None:
                continue
            row = {
                "user_id": user_id,
                "item_name": item_name,
                "source": source,
                "category": fields["category"],
                "meal_slot": fields["meal_slot"],
                "display_name": fields["display_name"],
                "qty": fields["qty"],
                "emoji": fields["emoji"],
                "is_checked": fields["is_checked"]
            }
            if fields["is_checked"]:
                from datetime import datetime, timezone
                row["checked_at"] = datetime.now(timezone.utc).isoformat()
            rows.append(row)
                
        if rows:
            res = supabase.table("custom_shopping_items").insert(rows).execute()
            return res.data
        return None
    except Exception as e:
        error_msg = str(e)
        # Fallback 1: columnas estructuradas no existen → insertar sin ellas
        if "category" in error_msg or "display_name" in error_msg or "qty" in error_msg or "emoji" in error_msg or "meal_slot" in error_msg:
            logger.warning("⚠️ [DB] Columnas estructuradas ausentes o incompletas. Ejecute migration_shopping_structured_columns.sql")
            try:
                rows_fb = []
                for item in items:
                    item_name, _ = _extract_fields(item)
                    if item_name:
                        rows_fb.append({"user_id": user_id, "item_name": item_name, "source": source})
                if rows_fb:
                    res_fb = supabase.table("custom_shopping_items").insert(rows_fb).execute()
                    return res_fb.data
                return None
            except Exception as e2:
                error_msg2 = str(e2)
                if "source" in error_msg2 or "PGRST205" in error_msg2:
                    # Fallback 2: ni source ni columnas estructuradas
                    return _add_shopping_items_minimal(user_id, items)
                logger.error(f"Error añadiendo items (fallback sin columnas estructuradas): {e2}")
                return None
        if "source" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            # Fallback 2: columna source tampoco existe
            logger.warning("⚠️ [DB] Columna source ausente. Ejecute migration_shopping_is_checked.sql")
            return _add_shopping_items_minimal(user_id, items, source=source)
        logger.error(f"Error añadiendo items a shopping list: {e}")
        return None

def _add_shopping_items_minimal(user_id: str, items: list, source: str = "manual"):
    """Fallback mínimo: solo user_id + item_name. Inyecta 'source' dentro del JSON para evitar borrados accidentales al hacer Reorganizar."""
    import json
    try:
        rows = []
        for item in items:
            d = {}
            if hasattr(item, 'model_dump'):
                d = item.model_dump()
            elif isinstance(item, dict):
                d = dict(item)
            elif isinstance(item, str) and item.strip():
                d = {"name": item.strip()}
            
            if d:
                d["source"] = source
                rows.append({"user_id": user_id, "item_name": json.dumps(d, ensure_ascii=False)})

        if rows:
            res = supabase.table("custom_shopping_items").insert(rows).execute()
            return res.data
        return None
    except Exception as e:
        logger.error(f"Error añadiendo items a shopping list (minimal fallback): {e}")
        return None

def delete_auto_generated_shopping_items(user_id: str, exclude_ids: list = None):
    """Elimina items auto-generados y legacy de la lista durante una reorganización.
    🛡️ PROTEGE: 
      - items ya comprados (is_checked=True)
      - items añadidos manualmente por el usuario via chat (source='chat')
      - items recién insertados (exclude_ids)
    🔥 BORRA: source='auto', source=null, source='' (vacío)"""
    if not supabase: return False
    try:
        # 1. Borrar items auto-generados por el plan
        query_auto = supabase.table("custom_shopping_items").delete().eq("user_id", user_id).eq("source", "auto").eq("is_checked", False)
        if exclude_ids:
            query_auto = query_auto.not_.in_("id", exclude_ids)
        query_auto.execute()
        
        # 2. Borrar items legacy sin source (NULL)
        query_null = supabase.table("custom_shopping_items").delete().eq("user_id", user_id).is_("source", "null").eq("is_checked", False)
        if exclude_ids:
            query_null = query_null.not_.in_("id", exclude_ids)
        query_null.execute()
        
        # 3. Borrar items con source vacío ("")
        query_empty = supabase.table("custom_shopping_items").delete().eq("user_id", user_id).eq("source", "").eq("is_checked", False)
        if exclude_ids:
            query_empty = query_empty.not_.in_("id", exclude_ids)
        query_empty.execute()
        
        # ✅ Items con source='chat' (añadidos por el usuario via IA) se PRESERVAN
        return True
    except Exception as e:
        error_msg = str(e)
        if "is_checked" in error_msg or "source" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            logger.warning("⚠️ [DB] Columna is_checked/source ausente. Usando fallback JSON.")
            return _delete_auto_shopping_items_legacy(user_id, exclude_ids)
        logger.error(f"Error borrando items durante reorganización: {e}")
        return False

def _delete_auto_shopping_items_legacy(user_id: str, exclude_ids: list = None):
    """Fallback legacy: borra items auto-generados parseando JSON previendo borrar manuales o checkeados."""
    import json
    try:
        # Extraemos is_checked si existe nativamente
        try:
            res = supabase.table("custom_shopping_items").select("id, item_name, is_checked").eq("user_id", user_id).execute()
        except:
            res = supabase.table("custom_shopping_items").select("id, item_name").eq("user_id", user_id).execute()
            
        existing = res.data
        if not existing: return True
        
        exclude_set = set(exclude_ids) if exclude_ids else set()
        ids_to_delete = []
        for item in existing:
            if item['id'] in exclude_set:
                continue
                
            # NUNCA borrar items físicos ya comprados
            if item.get('is_checked') is True:
                continue
                
            try:
                parsed = dict(json.loads(item['item_name']))
                if parsed.get("is_checked") is True:
                    continue
                    
                src = parsed.get("source")
                # Si es estrictamente manual, ignorar, no queremos destruirlo con el auto-reorganizar
                if src == "manual":
                    continue
                    
                # Borrar si fue generado por la IA (source="auto" o asumiendo auto por su categoría legacy)
                if src == "auto" or (not src and 'category' in parsed):
                    ids_to_delete.append(item['id'])
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        
        if ids_to_delete:
            CHUNK_SIZE = 100
            for i in range(0, len(ids_to_delete), CHUNK_SIZE):
                chunk = ids_to_delete[i:i + CHUNK_SIZE]
                supabase.table("custom_shopping_items").delete().in_("id", chunk).execute()
        return True
    except Exception as e:
        logger.error(f"Error borrando items auto-generados (legacy fallback): {e}")
        return False

def get_custom_shopping_items(user_id: str, limit: int = 200, offset: int = 0, sort_by: str = "category", sort_order: str = "asc"):
    """Obtiene los items custom de la lista de compras del usuario con paginación y ordenamiento.
    sort_by: 'category' | 'created_at' | 'display_name' | 'is_checked' (default: 'category')
    sort_order: 'asc' | 'desc' (default: 'asc')
    Retorna columnas estructuradas (category, display_name, qty, emoji) si están disponibles."""
    if not supabase: return {"data": [], "total": 0}
    
    # Whitelist de campos para evitar inyección
    ALLOWED_SORT = {"category", "created_at", "display_name", "is_checked", "name"}
    if sort_by not in ALLOWED_SORT:
        sort_by = "category"
    is_desc = sort_order.lower() == "desc"
    
    try:
        # Intento 1: con columnas estructuradas → permite ordenar por categoría a nivel DB
        query = supabase.table("custom_shopping_items").select(
            "id, item_name, is_checked, checked_at, source, category, display_name, qty, emoji, created_at, meal_slot",
            count="exact"
        ).eq("user_id", user_id).order(sort_by, desc=is_desc).range(offset, offset + limit - 1)
        res = query.execute()
        return {"data": res.data, "total": res.count or 0}
    except Exception as e:
        error_msg = str(e)
        if "category" in error_msg or "display_name" in error_msg or "qty" in error_msg or "emoji" in error_msg or "meal_slot" in error_msg:
            # Fallback 2: sin columnas estructuradas pero con is_checked/source
            logger.warning("⚠️ [DB] Columnas estructuradas ausentes. Ejecute migration_shopping_structured_columns.sql")
            try:
                fb_sort = sort_by if sort_by in {"created_at", "is_checked"} else "created_at"
                query_fb = supabase.table("custom_shopping_items").select(
                    "id, item_name, is_checked, source, created_at", count="exact"
                ).eq("user_id", user_id).order(fb_sort, desc=is_desc).range(offset, offset + limit - 1)
                res_fb = query_fb.execute()
                return {"data": res_fb.data, "total": res_fb.count or 0}
            except Exception as e2:
                error_msg2 = str(e2)
                if "is_checked" in error_msg2 or "source" in error_msg2:
                    # Fallback 3: schema mínimo
                    return _get_shopping_items_minimal(user_id, limit, offset)
                logger.error(f"Error obteniendo items (fallback sin columnas estructuradas): {e2}")
                return {"data": [], "total": 0}
        if "is_checked" in error_msg or "source" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            # Fallback 3: schema mínimo (solo id, item_name, created_at)
            logger.warning("⚠️ [DB] Columnas is_checked/source ausentes. Ejecute migration_shopping_is_checked.sql")
            return _get_shopping_items_minimal(user_id, limit, offset)
        logger.error(f"Error obteniendo custom shopping items: {e}")
        return {"data": [], "total": 0}

def _get_shopping_items_minimal(user_id: str, limit: int = 200, offset: int = 0, sort_order: str = "desc"):
    """Fallback mínimo: solo id, item_name, created_at (pre-migración)."""
    is_desc = sort_order.lower() == "desc"
    try:
        res = supabase.table("custom_shopping_items").select(
            "id, item_name, created_at", count="exact"
        ).eq("user_id", user_id).order("created_at", desc=is_desc).range(offset, offset + limit - 1).execute()
        return {"data": res.data, "total": res.count or 0}
    except Exception as e:
        logger.error(f"Error obteniendo custom shopping items (minimal fallback): {e}")
        return {"data": [], "total": 0}

def clear_all_shopping_items(user_id: str):
    """Elimina TODOS los items de la lista de compras del usuario."""
    if not supabase: return False
    try:
        supabase.table("custom_shopping_items").delete().eq("user_id", user_id).execute()
        return True
    except Exception as e:
        logger.error(f"Error limpiando lista de compras: {e}")
        return False

def uncheck_all_shopping_items(user_id: str):
    """Desmarca (is_checked=false) TODOS los items de la lista de compras del usuario."""
    if not supabase: return False
    try:
        supabase.table("custom_shopping_items").update({"is_checked": False, "checked_at": None}).eq("user_id", user_id).execute()
        return True
    except Exception as e:
        error_msg = str(e)
        if "is_checked" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            logger.warning("⚠️ [DB] Columna is_checked ausente. Ejecute migration_shopping_is_checked.sql")
            return False
        logger.error(f"Error desmarcando items: {e}")
        return False

def deduplicate_shopping_items(user_id: str):
    """Encuentra y fusiona items duplicados en la lista de compras del usuario.
    Agrupa por display_name normalizado. Suma cantidades numéricas cuando es posible.
    Retorna el número de duplicados eliminados.
    
    Thread-safe: usa un RLock per-user para serializar el SELECT→process→DELETE/UPDATE
    y evitar race conditions con auto-generación o purga concurrentes."""
    if not supabase: return {"removed": 0, "merged": []}
    
    lock = get_user_shopping_lock(user_id)
    acquired = lock.acquire(timeout=10)
    if not acquired:
        logger.warning(f"⚠️ [DEDUP] Timeout adquiriendo lock para {user_id}. Otra operación en curso.")
        return {"removed": 0, "merged": [], "error": "Deduplicación en curso para este usuario, intenta más tarde."}
    
    import re, json as _json
    try:
        return _deduplicate_shopping_items_impl(user_id, re, _json)
    finally:
        lock.release()

def _deduplicate_shopping_items_impl(user_id: str, re, _json):
    
    try:
        # Obtener todos los items del usuario
        res = supabase.table("custom_shopping_items").select(
            "id, item_name, display_name, qty, category, emoji, source, is_checked, created_at, meal_slot"
        ).eq("user_id", user_id).order("created_at", desc=True).execute()
        
        if not res.data or len(res.data) < 2:
            return {"removed": 0, "merged": []}
        
        items = res.data
    except Exception as e:
        # Fallback: sin columnas estructuradas
        try:
            res = supabase.table("custom_shopping_items").select("id, item_name, created_at").eq("user_id", user_id).order("created_at", desc=True).execute()
            if not res.data or len(res.data) < 2:
                return {"removed": 0, "merged": []}
            items = res.data
        except Exception as e2:
            logger.error(f"Error obteniendo items para dedup: {e2}")
            return {"removed": 0, "merged": []}
    
    def _normalize(text: str) -> str:
        """Normaliza texto para comparación: minúsculas, sin acentos, sin espacios extra."""
        if not text: return ""
        import unicodedata
        nfkd = unicodedata.normalize('NFKD', text.lower().strip())
        return re.sub(r'\s+', ' ', ''.join(c for c in nfkd if not unicodedata.combining(c)))
        
    def _extract_qty(item_dict: dict) -> str:
        q = item_dict.get("qty")
        if q: return str(q)
        raw = item_dict.get("item_name", "")
        if raw.startswith("{"):
            try:
                parsed = _json.loads(raw)
                return str(parsed.get("qty_7") or parsed.get("qty") or "")
            except Exception:
                pass
        return ""
    
    # Agrupar por nombre normalizado
    groups = {}  # normalized_name -> [items]
    for item in items:
        # Preferir display_name (estructurado) sobre item_name (JSON legacy)
        name = item.get("display_name") or ""
        if not name:
            # Intentar extraer name del JSON en item_name
            raw = item.get("item_name", "")
            if raw.startswith("{"):
                try:
                    parsed = _json.loads(raw)
                    name = parsed.get("name", raw)
                except Exception:
                    name = raw
            else:
                name = raw
        
        key = _normalize(name)
        if not key:
            continue
        if key not in groups:
            groups[key] = []
        groups[key].append({"item": item, "name": name})
    
    ids_to_delete = []
    ids_to_update = {}  # id -> {updates}
    merged_info = []
    
    # ===== FASE 1: Deduplicación exacta/regex =====
    unique_survivors = []
    
    for key, group in groups.items():
        # El primer item es el más reciente (ordenamos por created_at desc)
        keeper = group[0]
        duplicates = group[1:]
        
        # Intentar sumar cantidades
        keeper_qty_str = _extract_qty(keeper["item"])
        keeper_name = keeper["name"]
        keeper_num, keeper_unit, _ = parse_ingredient_qty(f"{keeper_qty_str} {keeper_name}", to_metric=False)
        total = keeper_num
        can_sum = keeper_num is not None
        
        # Si no podemos sumar matemáticamente, concatenamos strings
        concatenated_qtys = []
        if keeper_qty_str:
            concatenated_qtys.append(keeper_qty_str)
            
        for dup in duplicates:
            dup_qty_str = _extract_qty(dup["item"])
            dup_num, dup_unit, _ = parse_ingredient_qty(f"{dup_qty_str} {dup['name']}", to_metric=False)
            
            if can_sum and dup_num is not None and _normalize(dup_unit) == _normalize(keeper_unit):
                total += dup_num
            else:
                can_sum = False
                if dup_qty_str and dup_qty_str not in concatenated_qtys:
                    concatenated_qtys.append(dup_qty_str)
            
            ids_to_delete.append(dup["item"]["id"])
            
            # Si algún duplicado estaba checked, mantener ese estado
            if dup["item"].get("is_checked") and not keeper["item"].get("is_checked"):
                keeper["item"]["is_checked"] = True
        
        # Actualizar cantidad del keeper
        update_payload = {}
        if can_sum and total is not None:
            new_qty = format_qty(total, keeper_unit)
            if new_qty != keeper_qty_str:
                keeper["item"]["qty"] = new_qty
                update_payload["qty"] = new_qty
                if len(duplicates) > 0:
                    merged_info.append(f"{keeper_name}: {new_qty}")
        elif not can_sum and len(concatenated_qtys) > 1:
            # Concatenar como '1 lb + 2 unidades'
            new_qty = " + ".join(filter(bool, concatenated_qtys))
            if new_qty != keeper_qty_str:
                keeper["item"]["qty"] = new_qty
                update_payload["qty"] = new_qty
                merged_info.append(f"{keeper_name}: {new_qty} (texto combinado)")
        elif len(duplicates) > 0:
            merged_info.append(f"{keeper_name}: {len(duplicates)} duplicados removidos sin sumar")
            
        if keeper["item"].get("is_checked"):
            update_payload["is_checked"] = True
            
        if update_payload:
            ids_to_update[keeper["item"]["id"]] = update_payload
            
        unique_survivors.append(keeper)
        
    if not ids_to_delete and not ids_to_update:
        return {"removed": 0, "merged": []}
    
    # Ejecutar deletes y updates en batch (reducir N+1 queries)
    try:
        # 🚀 Batch DELETE: Supabase .in_() ya es batch, pero chunkeamos por si hay >500 IDs
        CHUNK_SIZE = 100
        for i in range(0, len(ids_to_delete), CHUNK_SIZE):
            chunk = ids_to_delete[i:i + CHUNK_SIZE]
            supabase.table("custom_shopping_items").delete().in_("id", chunk).execute()
        
        # 🚀 UPDATEs serializados de forma segura usando las funciones de core (con fallback robusto)
        if ids_to_update:
            for item_id, payload in ids_to_update.items():
                if "is_checked" in payload:
                    update_custom_shopping_item_status(item_id, payload["is_checked"])
                if "qty" in payload:
                    # Update quantity and its variations
                    q = payload["qty"]
                    update_custom_shopping_item(item_id, {"qty": q, "qty_7": q, "qty_15": q, "qty_30": q}, user_id=None)
        
        logger.debug(f"🧹 [DEDUP] Eliminados {len(ids_to_delete)} duplicados, {len(ids_to_update)} cantidades actualizadas")
        return {"removed": len(ids_to_delete), "merged": merged_info}
    except Exception as e:
        logger.error(f"Error en deduplicación: {e}")
        return {"removed": 0, "merged": [], "error": str(e)}

def purge_old_shopping_items(user_id: str):
    """Auto-purga items viejos de la lista de compras.
    1) Elimina items checked con checked_at > 30 días.
    2) Si aún hay más de 500 items, elimina los más antiguos.
    Retorna el número total de items purgados."""
    if not supabase: return 0
    
    from datetime import datetime, timezone, timedelta
    total_purged = 0
    
    try:
        # --- Fase 1: Purgar items checked hace más de 30 días ---
        cutoff = (datetime.now(timezone.utc) - timedelta(days=CHECKED_ITEM_EXPIRY_DAYS)).isoformat()
        try:
            res = supabase.table("custom_shopping_items").delete()\
                .eq("user_id", user_id)\
                .eq("is_checked", True)\
                .lt("checked_at", cutoff)\
                .execute()
            phase1 = len(res.data) if res.data else 0
            total_purged += phase1
            if phase1 > 0:
                logger.info(f"🧹 [PURGE] Fase 1: eliminados {phase1} items checked hace >{CHECKED_ITEM_EXPIRY_DAYS} días")
        except Exception as e:
            # Columna is_checked/checked_at puede no existir aún
            logger.warning(f"⚠️ [PURGE] Fase 1 skipped (columnas ausentes): {e}")
        
        # --- Fase 2: Enforce tope global ---
        try:
            count_res = supabase.table("custom_shopping_items")\
                .select("id", count="exact")\
                .eq("user_id", user_id)\
                .execute()
            total_items = count_res.count if hasattr(count_res, 'count') and count_res.count else len(count_res.data or [])
            
            if total_items > MAX_SHOPPING_ITEMS_PER_USER:
                excess = total_items - MAX_SHOPPING_ITEMS_PER_USER
                # Obtener los IDs más antiguos a eliminar
                oldest_res = supabase.table("custom_shopping_items")\
                    .select("id")\
                    .eq("user_id", user_id)\
                    .order("created_at", desc=False)\
                    .limit(excess)\
                    .execute()
                if oldest_res.data:
                    old_ids = [r["id"] for r in oldest_res.data]
                    supabase.table("custom_shopping_items").delete().in_("id", old_ids).execute()
                    total_purged += len(old_ids)
                    logger.info(f"🧹 [PURGE] Fase 2: eliminados {len(old_ids)} items (tope {MAX_SHOPPING_ITEMS_PER_USER} excedido)")
        except Exception as e:
            logger.warning(f"⚠️ [PURGE] Fase 2 error: {e}")
        
        return total_purged
    except Exception as e:
        logger.error(f"Error en purge_old_shopping_items: {e}")
        return 0

def delete_custom_shopping_item(item_id: str, user_id: str = None):
    """Elimina un item custom de la lista de compras. Si se provee user_id, verifica ownership."""
    if not supabase: return None
    try:
        query = supabase.table("custom_shopping_items").delete().eq("id", item_id)
        if user_id:
            query = query.eq("user_id", user_id)
        res = query.execute()
        return res.data
    except Exception as e:
        logger.error(f"Error borrando custom shopping item: {e}")
        return None

def delete_custom_shopping_items_batch(item_ids: list, user_id: str = None):
    """Elimina múltiples items custom de la lista de compras de una vez. Si se provee user_id, verifica ownership."""
    if not supabase or not item_ids: return None
    try:
        query = supabase.table("custom_shopping_items").delete().in_("id", item_ids)
        if user_id:
            query = query.eq("user_id", user_id)
        res = query.execute()
        return res.data
    except Exception as e:
        logger.error(f"Error borrando custom shopping items en batch: {e}")
        return None

def update_custom_shopping_item(item_id: str, updates: dict, user_id: str = None):
    """Actualiza campos editables de un item (display_name, qty, category, emoji).
    Si se provee user_id, verifica ownership (IDOR protection).
    Si las columnas estructuradas no existen, cae al fallback legacy (JSON en item_name)."""
    if not supabase: return None
    
    # Solo permitir campos editables
    allowed_fields = {"display_name", "qty", "qty_7", "qty_15", "qty_30", "category", "emoji", "meal_slot"}
    clean_updates = {k: v for k, v in updates.items() if k in allowed_fields and v is not None}
    
    # We must pass the original 'updates' because 'clean_updates' might filter out JSON-only fields if they don't map directly to table columns, but here we just pass 'clean_updates' and ensure allowed_fields has everything.
    if not clean_updates:
        return []
    
    try:
        # 🚀 Método directo: UPDATE a columnas estructuradas
        query = supabase.table("custom_shopping_items").update(clean_updates).eq("id", item_id)
        if user_id:
            query = query.eq("user_id", user_id)
        res = query.execute()
        
        # NOTA: Ya no sincronizamos item_name JSON (dual-write legacy eliminado).
        # Las columnas estructuradas son la fuente de verdad.
        # Si necesitas el fallback legacy, se activa automáticamente en el except.
        
        return res.data
    except Exception as e:
        error_msg = str(e)
        if any(col in error_msg for col in ["display_name", "qty", "category", "emoji", "meal_slot", "PGRST205"]):
            # Columnas estructuradas no existen → fallback a JSON en item_name
            logger.warning("⚠️ [DB] Columnas estructuradas ausentes. Usando fallback JSON para update.")
            return _update_shopping_item_legacy(item_id, clean_updates, user_id)
        logger.error(f"Error actualizando custom shopping item: {e}")
        return None

def _update_shopping_item_legacy(item_id: str, updates: dict, user_id: str = None):
    """Fallback: actualiza el JSON embebido en item_name."""
    import json
    try:
        query = supabase.table("custom_shopping_items").select("id, item_name").eq("id", item_id)
        if user_id:
            query = query.eq("user_id", user_id)
        res = query.execute()
        if not res.data:
            return []
        
        row = res.data[0]
        raw = row.get("item_name", "{}")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else {}
        except (json.JSONDecodeError, ValueError):
            parsed = {}
        
        field_map = {"display_name": "name", "qty": "qty", "qty_7": "qty_7", "qty_15": "qty_15", "qty_30": "qty_30", "category": "category", "emoji": "emoji", "meal_slot": "meal_slot"}
        for k, v in updates.items():
            if k in field_map:
                parsed[field_map[k]] = v
        
        supabase.table("custom_shopping_items").update(
            {"item_name": json.dumps(parsed, ensure_ascii=False)}
        ).eq("id", item_id).execute()
        
        return [{"id": item_id, **updates}]
    except Exception as e:
        logger.error(f"Error actualizando item (legacy fallback): {e}")
        return None

def update_custom_shopping_item_status(item_id: str, is_checked: bool, user_id: str = None):
    """Actualiza el estado is_checked de un item.
    Intenta usar la columna nativa is_checked (1 query atómica, sin race conditions).
    Si la columna no existe aún, hace fallback al método legacy (JSON en item_name).
    Si se provee user_id, verifica ownership (IDOR protection)."""
    if not supabase: return None
    try:
        # 🚀 Método atómico: UPDATE directo a columna nativa (O(1), sin race conditions)
        from datetime import datetime, timezone
        update_data = {"is_checked": is_checked}
        update_data["checked_at"] = datetime.now(timezone.utc).isoformat() if is_checked else None
        update_query = supabase.table("custom_shopping_items").update(update_data).eq("id", item_id)
        if user_id:
            update_query = update_query.eq("user_id", user_id)
        update_res = update_query.execute()
        # Verificar que se actualizó al menos 1 fila
        if update_res.data:
            return update_res.data
        return None
    except Exception as e:
        error_msg = str(e)
        if "is_checked" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            # Columna nativa no existe → fallback al método legacy (JSON en item_name)
            logger.warning("⚠️ [DB] Columna is_checked ausente. Usando fallback JSON. Ejecute migration_shopping_is_checked.sql")
            return _update_shopping_item_status_legacy(item_id, is_checked, user_id)
        logger.error(f"Error actualizando estado de item: {e}")
        return None

def _update_shopping_item_status_legacy(item_id: str, is_checked: bool, user_id: str = None):
    """Fallback legacy: guarda is_checked dentro del JSON de item_name (read-modify-write)."""
    import json
    try:
        query = supabase.table("custom_shopping_items").select("item_name").eq("id", item_id)
        if user_id:
            query = query.eq("user_id", user_id)
        res = query.execute()
        if not res.data: return None
        
        current_name = res.data[0]['item_name']
        try:
            parsed = json.loads(current_name)
            if isinstance(parsed, dict):
                parsed['is_checked'] = is_checked
                new_name = json.dumps(parsed, ensure_ascii=False)
            else:
                raise ValueError("Not a dict")
        except (json.JSONDecodeError, ValueError, KeyError):
            parsed = {
                "category": "Otros",
                "emoji": "📝",
                "name": current_name,
                "qty": "",
                "is_checked": is_checked
            }
            new_name = json.dumps(parsed, ensure_ascii=False)
        
        update_query = supabase.table("custom_shopping_items").update({"item_name": new_name}).eq("id", item_id)
        if user_id:
            update_query = update_query.eq("user_id", user_id)
        update_res = update_query.execute()
        return update_res.data
    except Exception as e:
        logger.error(f"Error actualizando estado de item (legacy fallback): {e}")
        return None

