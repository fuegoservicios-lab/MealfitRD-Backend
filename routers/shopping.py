from fastapi import APIRouter, Body, Depends, HTTPException, BackgroundTasks
from typing import Optional
import logging

from auth import get_verified_user_id
from rate_limiter import _shopping_write_limiter, _shopping_autogen_limiter
from db_shopping import (
    get_custom_shopping_items, update_custom_shopping_item,
    update_custom_shopping_item_status, delete_custom_shopping_item, clear_all_shopping_items,
    add_custom_shopping_items, uncheck_all_shopping_items, purge_old_shopping_items,
    deduplicate_shopping_items, get_shopping_plan_hash
)
from db_profiles import log_api_usage
from db_plans import get_latest_meal_plan
import re as _re
def sanitize_shopping_text(text: str, max_length: int = 100) -> str:
    """Escapa y sanitiza inputs en la lista de compras previniendo XSS ciego y control chars destructivos."""
    clean = _re.sub(r'<[^>]+>', '', text).strip()
    clean = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', clean)
    return clean[:max_length]
from constants import categorize_shopping_item
from agent import generate_auto_shopping_list
from services import regenerate_shopping_list_safe, _preserve_shopping_checkmarks, compute_plan_hash

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/shopping",
    tags=["shopping"],
)

@router.post("/auto-generate")
def api_shopping_auto_generate(data: dict = Body(...), verified_user_id: str = Depends(_shopping_autogen_limiter)):
    """Genera y guarda la lista de compras consolidada a partir del plan activo de 3 días.
    Usa cache por hash del plan: si el plan no cambió, retorna la lista existente sin llamar al LLM."""
    try:
        user_id = data.get("user_id")
        force = data.get("force", False)  # Forzar regeneración incluso si el plan no cambió
        days = data.get("days", 7) # Multiplicador de días
        
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido. Token inválido o no coincide.")
                
        if not user_id or user_id == "guest":
            raise HTTPException(status_code=400, detail="Usuario no válido para auto-generar lista.")
            
        current_plan = get_latest_meal_plan(user_id)
        
        if not current_plan:
            raise HTTPException(status_code=404, detail="No se encontró un plan activo para extraer ingredientes.")
        
        # Calcular hash del plan para detectar cambios (SSoT: compute_plan_hash en services.py)
        plan_hash = compute_plan_hash(current_plan)
        
        # Verificar si el plan ya fue procesado y si la lista sigue vigente
        if not force:
            stored_hash = get_shopping_plan_hash(user_id)
            existing = get_custom_shopping_items(user_id)
            existing_items = existing.get("data", []) if isinstance(existing, dict) else existing
            
            # Bloqueo Automático: Si ya hay items, calculamos si todavía no expira según los días seleccionados
            if existing_items:
                try:
                    from datetime import datetime, timezone
                    created_at_strs = [i.get("created_at") for i in existing_items if i.get("created_at")]
                    if created_at_strs:
                        oldest_str = min(created_at_strs)
                        if oldest_str.endswith("Z"):
                            oldest_str = oldest_str[:-1] + "+00:00"
                        created_dt = datetime.fromisoformat(oldest_str)
                        if created_dt.tzinfo is None:
                            created_dt = created_dt.replace(tzinfo=timezone.utc)
                            
                        days_elapsed = (datetime.now(timezone.utc) - created_dt).days
                        
                        # Si los días transcurridos son menores a los días seleccionados, BLOQUEAR regeneración (así cambie el plan)
                        if days_elapsed < days:
                            logger.info(f"🔒 [LISTA BLOQUEADA] Vigente: {days_elapsed}/{days} días (Ignorando cambios en plan).")
                            return {"success": True, "items": existing_items, "cached": True, "locked": True,
                                    "message": f"Lista bloqueada para preservar tus compras de {days} días."}
                except Exception as e:
                    logger.error(f"Error calculando expiración: {e}")
            
            # Si expiró o falló la verificación de tiempo, validamos por hash clásico
            if stored_hash == plan_hash:
                logger.info(f"✅ [CACHE HIT] Plan sin cambios (hash={plan_hash}). Retornando lista existente.")
                return {"success": True, "items": existing_items, "cached": True, "locked": False,
                        "message": "La lista ya está actualizada con tu plan actual."}
            
        items = generate_auto_shopping_list(current_plan)
        existing = get_custom_shopping_items(user_id)
        existing_items = existing.get("data", []) if isinstance(existing, dict) else existing
        
        result = regenerate_shopping_list_safe(user_id, items, existing_items, plan_hash)
        
        if result is not None:
            # Re-fetch from DB so the frontend gets exactly what's actually stored (JSON wrapped with id & source)
            final_items = get_custom_shopping_items(user_id, limit=500)
            final_items_list = final_items.get("data", []) if isinstance(final_items, dict) else final_items
            return {"success": True, "items": final_items_list, "cached": False, "message": f"Se auto-generaron y guardaron {len(items)} ingredientes estructurados en tu lista con éxito."}
        else:
            raise HTTPException(status_code=500, detail="Error al intentar guardar los ingredientes en la base de datos.")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"❌ [ERROR] Error en /api/shopping/auto-generate POST: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/custom")
def api_add_custom_shopping_item(data: dict = Body(...), verified_user_id: str = Depends(_shopping_write_limiter)):
    """Añade uno o más items a la lista de compras manualmente desde el frontend."""
    try:
        user_id = data.get("user_id")
        items = data.get("items", [])  # Lista de strings: ["Leche", "Pan", "Huevos"]
        
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido. Token inválido o no coincide.")
                
        if not user_id or user_id == "guest":
            raise HTTPException(status_code=400, detail="Usuario no válido para añadir items.")
            
        if not items:
            raise HTTPException(status_code=400, detail="No se proporcionaron items para añadir.")
        
        MAX_ITEMS = 150
        alert_msg = ""
        if len(items) > MAX_ITEMS:
            items = items[:MAX_ITEMS]
            alert_msg = f" (Alerta: se excedió el límite máximo y solo se añadieron los primeros {MAX_ITEMS} items)"
        
        # Normalizar a JSON struct consistente con ShoppingItemModel
        structured_items = []
        for item_data in items:
            raw = ""
            meal_slot = "Desayuno"
            qty = ""
            qty_7 = ""
            qty_15 = ""
            qty_30 = ""
            
            if isinstance(item_data, dict):
                raw = item_data.get("name", "").strip()
                meal_slot = item_data.get("meal_slot", "Desayuno")
                qty = item_data.get("qty", "")
                qty_7 = item_data.get("qty_7", qty)
                qty_15 = item_data.get("qty_15", qty)
                qty_30 = item_data.get("qty_30", qty)
            elif isinstance(item_data, str):
                raw = item_data.strip()
                
            name = sanitize_shopping_text(raw) if raw else ""
            if name:
                cat, emoji = categorize_shopping_item(name)
                structured_items.append({
                    "category": cat,
                    "meal_slot": meal_slot,
                    "emoji": emoji,
                    "name": name.capitalize(),
                    "qty": qty,
                    "qty_7": qty_7,
                    "qty_15": qty_15,
                    "qty_30": qty_30
                })
        
        if not structured_items:
            raise HTTPException(status_code=400, detail="Ningún item válido proporcionado.")
            
        result = add_custom_shopping_items(user_id, structured_items, source="manual")
        
        if result is not None:
            # Auto-deduplicar: si el usuario ya tenía "Leche" y añade "leche", se fusionan
            try:
                deduplicate_shopping_items(user_id)
            except Exception:
                pass  # No bloquear la respuesta si falla la dedup
            return {"success": True, "items": result, "message": f"Se añadieron {len(structured_items)} item(s) a tu lista de compras.{alert_msg}"}
        else:
            raise HTTPException(status_code=500, detail="Error al guardar los items en la base de datos.")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/shopping/custom POST: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/custom/{user_id}")
def api_get_custom_shopping_items(user_id: str, limit: int = 200, offset: int = 0, sort_by: str = "category", sort_order: str = "asc", verified_user_id: str = Depends(get_verified_user_id)):
    """Obtiene los items custom de la lista de compras con paginación y ordenamiento.
    sort_by: category | created_at | display_name | is_checked (default: category)
    sort_order: asc | desc (default: asc)"""
    try:
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
                
        if not user_id or user_id == "guest":
            return {"items": [], "total": 0}
        result = get_custom_shopping_items(user_id, limit=limit, offset=offset, sort_by=sort_by, sort_order=sort_order)
        return {"items": result.get("data", []), "total": result.get("total", 0)}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/shopping/custom GET: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/custom/{item_id}")
def api_delete_custom_shopping_item(item_id: str, verified_user_id: str = Depends(get_verified_user_id)):
    """Elimina un item custom de la lista de compras (con validación IDOR)."""
    try:
        if not verified_user_id:
            raise HTTPException(status_code=401, detail="No autorizado. Token requerido para eliminar items.")
        result = delete_custom_shopping_item(item_id, user_id=verified_user_id)
        if result is None or (isinstance(result, list) and len(result) == 0):
            raise HTTPException(status_code=404, detail="Item no encontrado o no pertenece al usuario.")
        return {"success": True, "message": "Item eliminado de la lista."}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/shopping/custom DELETE: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/custom/{item_id}")
def api_update_custom_shopping_item(item_id: str, data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
    """Edita campos de un item existente (display_name, qty, category, emoji) con validación IDOR."""
    try:
        if not verified_user_id:
            raise HTTPException(status_code=401, detail="No autorizado. Token requerido para editar items.")
        
        updates = {}
        for field in ["display_name", "qty_7", "qty_15", "qty_30", "category", "emoji", "meal_slot"]:
            if field in data:
                val = data[field]
                if isinstance(val, str):
                    updates[field] = sanitize_shopping_text(val)
                else:
                    updates[field] = val
        
        # Backward compatibility for qty
        if "qty" in data and "qty_7" not in data:
             updates["qty_7"] = sanitize_shopping_text(data["qty"])
        
        # Si se renombra el item, re-categorizar automáticamente
        if "display_name" in updates and "category" not in data:
            cat, emoji = categorize_shopping_item(updates["display_name"])
            updates["category"] = cat
            updates["emoji"] = emoji
        
        if not updates:
            raise HTTPException(status_code=400, detail="No se proporcionaron campos para actualizar. Campos permitidos: display_name, qty_7, qty_15, qty_30, category, emoji, meal_slot.")
        
        result = update_custom_shopping_item(item_id, updates, user_id=verified_user_id)
        
        if result is None:
            raise HTTPException(status_code=500, detail="Error interno al actualizar el item.")
        if isinstance(result, list) and len(result) == 0:
            raise HTTPException(status_code=404, detail="Item no encontrado o no pertenece al usuario.")
        
        return {"success": True, "message": "Item actualizado.", "updated": updates}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/shopping/custom PUT: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/custom/{item_id}/check")
def api_update_custom_shopping_item_check(item_id: str, data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
    """Actualiza el estado de is_checked de un item en la lista de compras (con validación IDOR)."""
    try:
        if not verified_user_id:
            raise HTTPException(status_code=401, detail="No autorizado. Token requerido.")
        is_checked = data.get("is_checked", False)
        result = update_custom_shopping_item_status(item_id, is_checked, user_id=verified_user_id)
        if result is not None:
            return {"success": True, "message": "Estado actualizado"}
        else:
            raise HTTPException(status_code=404, detail="Item no encontrado o no pertenece al usuario")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/shopping/custom PUT: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/custom/clear/{user_id}")
def api_clear_all_shopping_items(user_id: str, verified_user_id: str = Depends(get_verified_user_id)):
    """Elimina TODOS los items de la lista de compras del usuario."""
    try:
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=403, detail="Prohibido.")
        result = clear_all_shopping_items(user_id)
        if result:
            return {"success": True, "message": "Lista de compras vaciada."}
        raise HTTPException(status_code=500, detail="Error al vaciar la lista.")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/shopping/custom/clear DELETE: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/custom/uncheck-all/{user_id}")
def api_uncheck_all_shopping_items(user_id: str, verified_user_id: str = Depends(get_verified_user_id)):
    """Desmarca todos los items de la lista de compras del usuario."""
    try:
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=403, detail="Prohibido.")
        result = uncheck_all_shopping_items(user_id)
        if result:
            return {"success": True, "message": "Todos los items desmarcados."}
        raise HTTPException(status_code=500, detail="Error al desmarcar items.")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/shopping/custom/uncheck-all PUT: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/custom/deduplicate/{user_id}")
def api_deduplicate_shopping_items(user_id: str, verified_user_id: str = Depends(get_verified_user_id)):
    """Detecta y fusiona items duplicados en la lista de compras. Suma cantidades cuando es posible."""
    try:
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=403, detail="Prohibido.")
        result = deduplicate_shopping_items(user_id)
        return {"success": True, **result}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/shopping/custom/deduplicate POST: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/custom/purge/{user_id}")
def api_purge_shopping_items(user_id: str, verified_user_id: str = Depends(get_verified_user_id)):
    """Purga items checked hace más de 30 días y aplica el tope global de 500 items."""
    try:
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=403, detail="Prohibido.")
        result = purge_old_shopping_items(user_id)
        return {"success": True, "message": "Purga completada.", "details": result}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/shopping/custom/purge POST: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

