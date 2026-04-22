from fastapi import APIRouter, Body, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import Optional
import logging
import uuid
import asyncio
from db_core import supabase
from db_profiles import get_user_profile, log_api_usage
from auth import get_verified_user_id, verify_api_quota
from db import log_consumed_meal, get_consumed_meals_today, save_visual_entry
from vision_agent import process_image_with_vision, get_multimodal_embedding

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/diary",
    tags=["diary"],
)

@router.post("/upload")
async def api_diary_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form("guest"),
    session_id: str = Form(None),
    tz_offset_mins: int = Form(0),
    verified_user_id: str = Depends(verify_api_quota)
):
    try:
        # Validación de seguridad IDOR
        if user_id and user_id != "guest" and user_id != session_id:
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado.")
                
        actual_user_id = user_id if user_id != "guest" else session_id

        
        MAX_FILE_SIZE = 20 * 1024 * 1024 # 20 MB
        file_bytes = b""
        while chunk := await file.read(1024 * 1024):
            file_bytes += chunk
            if len(file_bytes) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="La imagen es demasiado grande. Máximo 20MB permitidos.")
        
        file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        unique_filename = f"{actual_user_id}/{uuid.uuid4().hex}.{file_ext}"
        
        image_url = ""
        upload_success = False

        # 1. Intentar subir a Supabase Storage
        if supabase:
            try:
                res = await asyncio.to_thread(
                    supabase.storage.from_("visual_diary_images").upload,
                    path=unique_filename,
                    file=file_bytes,
                    file_options={"content-type": file.content_type}
                )
                image_url = supabase.storage.from_("visual_diary_images").get_public_url(unique_filename)
                upload_success = True
                logger.info(f"☁️ Imagen guardada en Supabase: {image_url}")
            except Exception as sb_err:
                logger.error(f"⚠️ Error subiendo a Supabase (¿Existe el bucket 'visual_diary_images'?): {sb_err}")
                upload_success = False

        # 2. Si no se pudo subir a Supabase, fallar (evitar guardar localmente en la nube)
        if not upload_success:
            logger.error("❌ No se pudo subir la imagen a Supabase. Abortando.")
            raise HTTPException(status_code=500, detail="Error uploading image to cloud storage.")
            
            
        # 3. Procesar imagen con Visión SINCRÓNICAMENTE
        logger.info("\n-------------------------------------------------------------")
        logger.info("📸 [VISION AGENT] Procesando nueva imagen subida...")
        vision_result = await process_image_with_vision(file_bytes)
        
        description = vision_result.get("description", "No se pudo analizar la imagen.")
        is_food = vision_result.get("is_food", False)
        calories = vision_result.get("calories", 0)
        
        if is_food:
            logger.info(f"✅ Descripción generada: '{description}' (Calorías: {calories})")
            
            # --- EVALUACIÓN DETERMINISTA DE CRONONUTRICIÓN (PYTHON) ---
            if calories > 500:
                from datetime import datetime, timedelta
                # Calcular la hora exacta del usuario usando el offset proporcionado por el frontend
                local_time = datetime.utcnow() - timedelta(minutes=tz_offset_mins)
                local_hour = local_time.hour
                
                # Obtener el perfil para ver el tipo de turno
                schedule_type = "standard"
                if user_id != "guest":
                    profile = get_user_profile(user_id)
                    if profile and profile.get("health_profile"):
                        schedule_type = profile["health_profile"].get("scheduleType", "standard")
                
                # Reglas matemáticas duras e inquebrantables
                is_red_alert = False
                if schedule_type == "standard" and (0 <= local_hour < 6):
                    is_red_alert = True
                elif schedule_type == "night_shift" and (14 <= local_hour < 20):  # Ej: 2 PM a 8 PM
                    is_red_alert = True
                
                if is_red_alert:
                    poison_pill = " [⚠️ INSTRUCCIÓN CLÍNICA PRIORITARIA DE SISTEMA: El análisis de esta imagen indica que el usuario está a punto de consumir más de 500 kcal en una hora sumamente crítica para su reloj biológico (madrugada o contra-turno). TIENES LA ORDEN DIRECTA Y OBLIGATORIA de cambiar tu tono y emitir inmediatamente una advertencia y reprimenda constructiva sobre el grave impacto digestivo de comer así a esta hora.]"
                    description += poison_pill
                    logger.warning(f"🚨 [RULE POISONING] Inyectando alerta roja determinista. Hora local: {local_hour}, Calorías: {calories}, Turno: {schedule_type}")
            # ------------------------------------------------------------
            
            if actual_user_id and actual_user_id != session_id:
                log_api_usage(actual_user_id, "gemini_vision")
                
            # 4. Guardar en DB en segundo plano (embedding + insert)
            background_tasks.add_task(
                _save_visual_entry_background,
                actual_user_id, image_url, description
            )
        else:
            logger.info("➡️ La imagen fue ignorada porque no se detectaron alimentos.")
        
        return {
            "success": True, 
            "is_food": is_food,
            "description": description,
            "image_url": image_url
        }
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/diary/upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def _save_visual_entry_background(user_id: str, image_url: str, description: str):
    """Background task: genera embedding y guarda en la tabla visual_diary."""
    
    embedding = get_multimodal_embedding(description)
    if embedding:
        logger.info(f"📦 Guardando entrada visual en la DB (Vector 768d)...")
        save_visual_entry(user_id=user_id, image_url=image_url, description=description, embedding=embedding)
        logger.info("✅ ¡Imagen registrada en el Diario Visual con éxito!")
    else:
        logger.warning("⚠️ No se pudo vectorizar la imagen. Abortando guardado.")


@router.post("/consumed")
def api_log_consumed_meal(
    data: dict = Body(...), 
    background_tasks: BackgroundTasks = BackgroundTasks(),
    verified_user_id: str = Depends(get_verified_user_id)
):
    """Registra una comida consumida manualmente desde el frontend."""
    try:
        user_id = data.get("user_id")
        meal_name = data.get("meal_name")
        meal_type = data.get("meal_type", "snack")
        calories = data.get("calories", 0)
        protein = data.get("protein", 0)
        carbs = data.get("carbs", 0)
        healthy_fats = data.get("healthy_fats", 0)
        
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
                
        if not user_id or user_id == "guest":
            return {"success": False, "message": "Inicia sesión para registrar comidas."}
            
        log_consumed_meal(user_id, meal_name, int(calories), int(protein), int(carbs), int(healthy_fats), meal_type=meal_type)
        
        # [GAP 4] Latencia de 18+ horas: Recalcular adherencia intradía en background
        from cron_tasks import trigger_incremental_learning
        background_tasks.add_task(trigger_incremental_learning, user_id)
        
        return {"success": True, "message": "Comida registrada exitosamente."}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/diary/consumed POST: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/consumed/{user_id}")
def api_get_consumed_today(user_id: str, date: Optional[str] = None, tzOffset: Optional[int] = None, verified_user_id: str = Depends(get_verified_user_id)):
    """Obtiene las métricas agregadas de las comidas registradas en el día por la IA."""
    try:
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
                
        if not user_id or user_id == "guest":
            return {"meals": [], "totals": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}}
        
        meals = get_consumed_meals_today(user_id, date_str=date, tz_offset_mins=tzOffset)
        
        total_cal = sum(m.get("calories", 0) for m in meals)
        total_pro = sum(m.get("protein", 0) for m in meals)
        total_car = sum(m.get("carbs", 0) for m in meals)
        total_fat = sum(m.get("healthy_fats", 0) for m in meals)
        
        return {
            "meals": meals,
            "totals": {
                "calories": total_cal,
                "protein": total_pro,
                "carbs": total_car,
                "healthy_fats": total_fat
            }
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/diary/consumed GET: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/progress")
def api_log_progress(data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
    """Registra el peso actual en el historial de progreso (Progress Tracker)."""
    try:
        user_id = data.get("user_id")
        weight = float(data.get("weight"))
        unit = data.get("unit", "lb")
        
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
                
        if not user_id or user_id == "guest":
            return {"success": False, "message": "Inicia sesión para registrar progreso."}
            
        profile = get_user_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Perfil no encontrado.")
            
        health_profile = profile.get("health_profile") or {}
        weight_history = health_profile.get("weight_history", [])
        
        from datetime import datetime
        now_date = datetime.now().strftime("%Y-%m-%d")
        
        # Evitar múltiples registros en el mismo día (actualizar si ya existe)
        existing_entry = next((e for e in weight_history if e.get("date") == now_date), None)
        if existing_entry:
            existing_entry["weight"] = weight
            existing_entry["unit"] = unit
        else:
            weight_history.append({"date": now_date, "weight": weight, "unit": unit})
            
        # Ordenar cronológicamente y mantener solo los últimos 30 registros
        weight_history = sorted(weight_history, key=lambda x: x["date"])[-30:]
        
        health_profile["weight_history"] = weight_history
        # También actualizamos el peso estático actual para el UI
        health_profile["weight"] = weight
        health_profile["weightUnit"] = unit
        
        from db_profiles import update_user_health_profile
        res = update_user_health_profile(user_id, health_profile)
        if res is None:
            raise Exception("Error guardando progreso en DB.")
            
        return {"success": True, "message": "Progreso guardado exitosamente.", "weight_history": weight_history}
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/diary/progress POST: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

