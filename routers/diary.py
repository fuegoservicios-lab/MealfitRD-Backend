from fastapi import APIRouter, Body, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from error_utils import safe_error_detail
from typing import Optional
import logging
import math
import uuid
import asyncio
from pydantic import BaseModel, Field, field_validator
from db_core import supabase
from db_profiles import get_user_profile, log_api_usage
from auth import get_verified_user_id, verify_api_quota
from path_validators import assert_valid_uuid
from rate_limiter import RateLimiter
from db import log_consumed_meal, get_consumed_meals_today, save_visual_entry
from vision_agent import process_image_with_vision, get_multimodal_embedding
# [P3-DIARY-LATE-IMPORT · 2026-05-15] Import movido al top — verificado que
# `cron_tasks` NO importa `routers.diary` (no circular). El late import dentro
# del handler era frágil: un refactor que renombrara `trigger_incremental_learning`
# crashearía 500 en runtime en cada POST a `/api/diary/consumed`, no en import-time.
from cron_tasks import trigger_incremental_learning

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/diary",
    tags=["diary"],
)


# ---------------------------------------------------------------------------
# [P2-DIARY-NO-PYDANTIC · 2026-05-15] Pydantic models para validación de input.
#
# ANTES: los endpoints aceptaban `data: dict = Body(...)` sin schema. Inputs
# adversariales (NaN / Infinity / strings en campos numéricos / arrays
# enormes) caían a `int()` / `float()` con fallos silenciosos o 500s sin
# discriminación. Sin schema, OpenAPI también quedaba pobre — clientes no
# sabían qué shape esperar.
#
# DESPUÉS: cada endpoint define su Pydantic model. FastAPI valida antes
# de invocar el handler y devuelve 422 con detalle estructurado en error.
# `model_config = {"extra": "ignore"}` mantiene compat con clientes legacy
# que envíen campos extra (no romper rollouts).
# ---------------------------------------------------------------------------

class ConsumedMealRequest(BaseModel):
    """Payload para `POST /api/diary/consumed`. Campos numéricos clamp a
    rangos sanos para evitar NaN/Infinity contamination en agregados."""
    user_id: Optional[str] = None
    meal_name: str = Field(..., min_length=1, max_length=200)
    meal_type: str = Field(default="snack", max_length=32)
    calories: float = Field(default=0.0, ge=0.0, le=10000.0)
    protein: float = Field(default=0.0, ge=0.0, le=1000.0)
    carbs: float = Field(default=0.0, ge=0.0, le=2000.0)
    healthy_fats: float = Field(default=0.0, ge=0.0, le=1000.0)

    model_config = {"extra": "ignore"}

    @field_validator("calories", "protein", "carbs", "healthy_fats")
    @classmethod
    def _reject_non_finite(cls, v: float) -> float:
        # [P2-DIARY-NO-PYDANTIC] NaN/Infinity bypassean `ge=0` de Pydantic
        # en algunas versiones (Pydantic v2 los acepta como float válidos).
        # Validator explícito.
        if not math.isfinite(v):
            raise ValueError("Macros deben ser números finitos.")
        return v


class ProgressRequest(BaseModel):
    """Payload para `POST /api/diary/progress` (weight tracking)."""
    user_id: Optional[str] = None
    weight: float = Field(..., gt=0.0, le=2000.0)
    unit: str = Field(default="lb", max_length=8)

    model_config = {"extra": "ignore"}

    @field_validator("weight")
    @classmethod
    def _reject_non_finite_weight(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("weight debe ser número finito.")
        return v


# [P2-DIARY-PROGRESS-RATELIMIT · 2026-05-15] Rate limiter para `/progress`.
# ANTES: POST autenticado sin throttling — un user (o bug en mobile que
# tap-spammee el guardar peso) podía inflar `weight_history` JSONB.
# La lista se cappea a últimos 30 (línea ~270) pero la mutación bajo
# advisory lock + roundtrip a Supabase tiene costo. 10/60s es generoso
# (UI espera click manual) y restringe spam claro. Mismo patrón que
# `_RECALC_LIMITER` y `_PDF_TELEMETRY_LIMITER` en routers/plans.py.
_PROGRESS_LIMITER = RateLimiter(max_calls=10, period_seconds=60)


# [P3-VISION-UPLOAD-VALIDATION · 2026-05-20] Whitelist de content_types
# permitidos para `/upload`. Cierra el gap F3 del audit
# `docs/gaps-audit-2026-05.md`: pre-fix el endpoint aceptaba el
# `file.content_type` declarado por el cliente Y lo pasaba directo a
# Supabase Storage + a `process_image_with_vision`. Vectores cerrados:
#
#   1. Cliente declara `content_type=application/octet-stream` (default
#      browser para tipos desconocidos) → bypass del MIME-sniffing del
#      bucket Storage. Si en el futuro un proxy/CDN sirve el content
#      con un Content-Type confiando en el del bucket, podría servir
#      ejecutables/HTML embebidos como "imagen" → XSS / drive-by.
#   2. Cliente declara `image/svg+xml` → el bucket lo sirve y otros
#      clientes que lo renderizan ejecutan JS embebido en el SVG (vector
#      XSS clásico). Gemini Vision NO renderiza SVG y devuelve "imagen
#      sin descripción" → bypass silencioso para almacenamiento.
#   3. Magic-bytes check defense-in-depth: aunque content_type pase el
#      check, los primeros bytes deben ser un signature legítimo de
#      JPEG/PNG/WebP/HEIC. Catch atacantes que envían `Content-Type:
#      image/jpeg` con un body de HTML/exe.
#
# Estrategia conservadora: rechazar lo que NO matchee con HTTP 415
# (Unsupported Media Type). Lista cubre los formatos de cámaras móviles
# modernas (iOS HEIC, Android JPEG/PNG/WebP).
#
# Tooltip-anchor: P3-VISION-UPLOAD-VALIDATION.
_ALLOWED_VISION_CONTENT_TYPES = frozenset({
    "image/jpeg",
    "image/jpg",  # variante histórica, algunos clientes envían así
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
})

# Magic bytes (signatures de los primeros bytes del archivo). Defensa
# contra content-type spoofing: aunque el cliente declare `image/jpeg`,
# si los primeros bytes no son `FF D8 FF`, rechazar.
# - JPEG: FF D8 FF
# - PNG:  89 50 4E 47 0D 0A 1A 0A
# - WebP: RIFF....WEBP (4 bytes RIFF + 4 bytes size + WEBP)
# - HEIC/HEIF: bytes 4-12 contienen `ftypheic`/`ftypheix`/`ftyphevc`/`ftypmif1`
def _detect_image_mime_from_bytes(b: bytes) -> Optional[str]:
    """Sniff magic bytes. Retorna mime detectado o None si no matchea
    ninguno de los formatos permitidos. Defensa contra content-type
    spoofing."""
    if len(b) < 12:
        return None
    if b[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    # HEIC/HEIF: el "ftyp" box está en bytes 4-7, brand en 8-11
    if b[4:8] == b"ftyp":
        brand = b[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"heim", b"heis", b"hevx",
                     b"mif1", b"msf1"):
            return "image/heic"
    return None

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

        # [P3-VISION-UPLOAD-VALIDATION · 2026-05-20] Validación de
        # content_type ANTES de leer bytes. Rechaza con HTTP 415 si
        # el cliente declara MIME no permitido. Defensa contra SVG/XML
        # con JS embebido + sub-types raros. Si content_type es None
        # o vacío (clientes raros), también rechaza — clientes legítimos
        # siempre envían content_type.
        declared_ct = (file.content_type or "").lower().strip()
        if declared_ct not in _ALLOWED_VISION_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Tipo de archivo no soportado: {declared_ct or 'desconocido'}. "
                    f"Solo se permiten imágenes JPEG/PNG/WebP/HEIC."
                ),
            )

        MAX_FILE_SIZE = 20 * 1024 * 1024 # 20 MB
        file_bytes = b""
        while chunk := await file.read(1024 * 1024):
            file_bytes += chunk
            if len(file_bytes) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="La imagen es demasiado grande. Máximo 20MB permitidos.")

        # [P3-VISION-UPLOAD-VALIDATION · 2026-05-20] Magic-bytes check
        # post-read. Defensa-en-profundidad contra content-type spoofing:
        # aunque `declared_ct` pase el whitelist arriba, si los primeros
        # bytes no son una signature válida de imagen, rechazar.
        # Catches atacantes que declaran `image/jpeg` con un body de
        # HTML/exe esperando que pase a Storage + downstream.
        sniffed_ct = _detect_image_mime_from_bytes(file_bytes[:32])
        if sniffed_ct is None:
            raise HTTPException(
                status_code=415,
                detail=(
                    "El archivo no parece una imagen válida (firma de bytes inválida). "
                    "Verifica que sea un JPEG/PNG/WebP/HEIC real."
                ),
            )

        # `filename` defensivo: si el cliente no envió filename (raro pero
        # posible), default a la extensión derivada del sniffed_ct.
        if file.filename and '.' in file.filename:
            file_ext = file.filename.rsplit('.', 1)[-1].lower()
        else:
            file_ext = sniffed_ct.split('/')[-1]  # 'jpeg', 'png', etc.
        # Sanitización defensiva del ext: solo a-z0-9 (3-5 chars) para
        # construir el path en Storage. Evita inyectar `..` o caracteres
        # raros en el filename del bucket.
        import re as _re_ext
        if not _re_ext.match(r'^[a-z0-9]{2,5}$', file_ext):
            file_ext = sniffed_ct.split('/')[-1]
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
        
        # [P1-DIARY-PROMPT-INJECTION · 2026-05-15] Antes el endpoint concatenaba
        # un string instructivo ("poison_pill") al `description` cuando la regla
        # determinista detectaba calorías altas en horas críticas. Eso resolvía
        # un caso de uso (forzar al chat agent a emitir reprimenda) pero abría
        # un vector de prompt-injection: `calories` proviene de `vision_agent`
        # → LLM derivado de la imagen subida por el usuario → controlable.
        # `tz_offset_mins` viene del body del cliente → controlable.
        # Un atacante podía construir un payload que disparase la regla y
        # quedaba un string instructivo persistido en `visual_diary.description`
        # + embebido en futuros contextos del chat agent.
        #
        # Fix: la decisión chrono-nutrición sigue calculándose (regla
        # determinista, no LLM-dependiente), pero NO se concatena texto a
        # `description`. Se persiste a `pipeline_metrics` como signal
        # estructurada (`node='chrono_nutrition_red_alert'`) para que SRE/
        # dashboards la observen, y se devuelve `red_alert` boolean al
        # cliente para que la UX muestre la advertencia sin que la LLM
        # sea instruida por texto poisoned.
        chrono_red_alert = False
        chrono_local_hour: Optional[int] = None
        chrono_schedule_type: Optional[str] = None

        if is_food:
            logger.info(f"✅ Descripción generada: '{description}' (Calorías: {calories})")

            # --- EVALUACIÓN DETERMINISTA DE CRONONUTRICIÓN (PYTHON) ---
            if calories > 500:
                from datetime import datetime, timedelta, timezone
                # [P3-DEPRECATED-UTCNOW · 2026-05-12] tz-aware: ver memoria.
                local_time = datetime.now(timezone.utc) - timedelta(minutes=tz_offset_mins)
                chrono_local_hour = local_time.hour

                chrono_schedule_type = "standard"
                if user_id != "guest":
                    profile = get_user_profile(user_id)
                    if profile and profile.get("health_profile"):
                        chrono_schedule_type = profile["health_profile"].get("scheduleType", "standard")

                if chrono_schedule_type == "standard" and (0 <= chrono_local_hour < 6):
                    chrono_red_alert = True
                elif chrono_schedule_type == "night_shift" and (14 <= chrono_local_hour < 20):
                    chrono_red_alert = True

                if chrono_red_alert:
                    # [P1-DIARY-PROMPT-INJECTION · 2026-05-15] Signal estructurada
                    # a pipeline_metrics (best-effort) en lugar de poison_pill
                    # concatenado a description. Frontend lee `red_alert` y
                    # decide UX; chat agent NO recibe texto instruccional.
                    try:
                        from db_core import execute_sql_write
                        import json as _json_chrono
                        execute_sql_write(
                            """
                            INSERT INTO pipeline_metrics
                                (user_id, session_id, node, duration_ms, retries,
                                 tokens_estimated, confidence, metadata)
                            VALUES (%s, %s, %s, 0, 0, 0, 0, %s::jsonb)
                            """,
                            (
                                actual_user_id if actual_user_id and actual_user_id != session_id else None,
                                session_id,
                                "chrono_nutrition_red_alert",
                                _json_chrono.dumps({
                                    "local_hour": chrono_local_hour,
                                    "calories": int(calories) if calories is not None else None,
                                    "schedule_type": chrono_schedule_type,
                                }, ensure_ascii=False),
                            ),
                        )
                    except Exception as _chrono_err:
                        logger.debug(f"[P1-DIARY-PROMPT-INJECTION] tick chrono falló (best-effort): {_chrono_err}")
                    logger.warning(
                        f"🚨 [CHRONO-NUTRITION] Red alert determinista (sin text poisoning). "
                        f"local_hour={chrono_local_hour}, calorías={calories}, turno={chrono_schedule_type}"
                    )
            # ------------------------------------------------------------

            if actual_user_id and actual_user_id != session_id:
                log_api_usage(actual_user_id, "gemini_vision")

            # 4. Guardar en DB en segundo plano (embedding + insert).
            # `description` se persiste limpio (sin poison_pill) — la
            # signal chrono se canaliza por `pipeline_metrics`.
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
            "image_url": image_url,
            # [P1-DIARY-PROMPT-INJECTION · 2026-05-15] flag estructurado para
            # frontend; reemplaza la inyección de texto en `description`.
            "red_alert": chrono_red_alert,
            "red_alert_meta": {
                "local_hour": chrono_local_hour,
                "schedule_type": chrono_schedule_type,
            } if chrono_red_alert else None,
        }
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/diary/upload: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

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
    payload: ConsumedMealRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    verified_user_id: str = Depends(get_verified_user_id)
):
    """Registra una comida consumida manualmente desde el frontend.

    [P2-DIARY-NO-PYDANTIC · 2026-05-15] Migrado de `data: dict = Body(...)`
    a `ConsumedMealRequest`. Pydantic valida tipos + rangos + NaN/Infinity
    antes del handler (422 con detalle estructurado en lugar de 500
    silencioso).
    """
    try:
        user_id = payload.user_id
        meal_name = payload.meal_name
        meal_type = payload.meal_type
        calories = payload.calories
        protein = payload.protein
        carbs = payload.carbs
        healthy_fats = payload.healthy_fats

        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")

        if not user_id or user_id == "guest":
            return {"success": False, "message": "Inicia sesión para registrar comidas."}

        log_consumed_meal(user_id, meal_name, int(calories), int(protein), int(carbs), int(healthy_fats), meal_type=meal_type)

        # [GAP 4] Latencia de 18+ horas: Recalcular adherencia intradía en background.
        # [P3-DIARY-LATE-IMPORT · 2026-05-15] Import movido al top del archivo.
        background_tasks.add_task(trigger_incremental_learning, user_id)

        return {"success": True, "message": "Comida registrada exitosamente."}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/diary/consumed POST: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.get("/consumed/{user_id}")
def api_get_consumed_today(user_id: str, date: Optional[str] = None, tzOffset: Optional[int] = None, verified_user_id: str = Depends(get_verified_user_id)):
    """Obtiene las métricas agregadas de las comidas registradas en el día por la IA."""
    try:
        # [P1-AUDIT-3 · 2026-05-12] Rechaza UUIDs malformados con 400 antes de SQL.
        assert_valid_uuid(user_id, allow_guest=True)
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
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/progress")
def api_log_progress(
    payload: ProgressRequest,
    verified_user_id: Optional[str] = Depends(_PROGRESS_LIMITER),
):
    """Registra el peso actual en el historial de progreso (Progress Tracker).

    [P2-DIARY-NO-PYDANTIC + P2-DIARY-PROGRESS-RATELIMIT · 2026-05-15]
    - Pydantic `ProgressRequest` valida weight finito + rango (0, 2000].
    - `_PROGRESS_LIMITER` (10/60s per user/IP) bloquea spam de POSTs que
      inflarían `weight_history` JSONB. El limiter inyecta
      `verified_user_id` (mismo patrón que `_CHAT_TTS_LIMITER`).
    """
    try:
        user_id = payload.user_id
        weight = payload.weight
        unit = payload.unit
        
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
                
        if not user_id or user_id == "guest":
            return {"success": False, "message": "Inicia sesión para registrar progreso."}
            
        # [P1-2] Atomic write con FOR UPDATE. Antes, dos POSTs concurrentes a
        # /api/diary/progress (raro pero posible: usuario tap rápido en app
        # móvil + sync del wear device) leían el mismo `weight_history`,
        # cada uno appendeaba un nuevo entry localmente (potentially en el
        # mismo `now_date`), y el último UPDATE pisaba al primero.
        # Resultado: 1 entry de peso perdido silenciosamente. Con el mutator
        # bajo lock, ambos POSTs se serializan: el segundo VE la mutación
        # del primero y aplica la suya encima (existing_entry update vs new
        # append). El existence check del entry de hoy se hace DENTRO del
        # mutator (bajo lock), garantizando idempotencia per-día aún con
        # dobles taps.
        from datetime import datetime
        now_date = datetime.now().strftime("%Y-%m-%d")

        result_box = {"weight_history": None}

        def _weight_mutator(_hp):
            _wh = list(_hp.get("weight_history", []) or [])
            if not isinstance(_wh, list):
                _wh = []
            _existing = next((_e for _e in _wh if isinstance(_e, dict) and _e.get("date") == now_date), None)
            if _existing:
                _existing["weight"] = weight
                _existing["unit"] = unit
            else:
                _wh.append({"date": now_date, "weight": weight, "unit": unit})
            _wh = sorted(
                [_e for _e in _wh if isinstance(_e, dict) and _e.get("date")],
                key=lambda x: x["date"],
            )[-30:]
            _hp["weight_history"] = _wh
            _hp["weight"] = weight
            _hp["weightUnit"] = unit
            result_box["weight_history"] = _wh
            return None

        from db_profiles import update_user_health_profile_atomic
        new_hp = update_user_health_profile_atomic(user_id, _weight_mutator)
        if new_hp is None:
            raise HTTPException(status_code=404, detail="Perfil no encontrado.")

        return {
            "success": True,
            "message": "Progreso guardado exitosamente.",
            "weight_history": result_box["weight_history"] or [],
        }
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/diary/progress POST: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


# [P1-4] Endpoints de preferencia de logging.
# 'manual' (default): el sistema pausa chunks si el usuario deja de loguear comidas.
# 'auto_proxy': el usuario opta por confiar en el plan; el sistema NO pausa por falta de logs.
_P14_VALID_PREFERENCES = ("manual", "auto_proxy")


@router.get("/preferences/logging")
def api_get_logging_preference(verified_user_id: str = Depends(get_verified_user_id)):
    from db_core import execute_sql_query
    row = execute_sql_query(
        "SELECT logging_preference FROM user_profiles WHERE id = %s",
        (verified_user_id,),
        fetch_one=True,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Perfil no encontrado.")
    return {"logging_preference": row.get("logging_preference") or "manual"}


@router.put("/preferences/logging")
def api_set_logging_preference(
    data: dict = Body(...),
    verified_user_id: str = Depends(get_verified_user_id),
):
    pref = (data or {}).get("logging_preference")
    if pref not in _P14_VALID_PREFERENCES:
        raise HTTPException(
            status_code=400,
            detail=f"logging_preference debe ser uno de: {list(_P14_VALID_PREFERENCES)}.",
        )
    from db_core import execute_sql_write
    execute_sql_write(
        "UPDATE user_profiles SET logging_preference = %s WHERE id = %s",
        (pref, verified_user_id),
    )
    return {"success": True, "logging_preference": pref}

