from fastapi import APIRouter, Body, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from error_utils import safe_error_detail
from typing import List, Optional
import json
import logging
import math
import uuid
import asyncio
from pydantic import BaseModel, Field, field_validator
from db_core import _storage_client
# [P1-MEAL-SCAN-GEMMA · 2026-07-12 → P1-VISION-NO-LOCAL · 2026-07-28]
# verify_api_quota removido del import: el único consumidor era /upload, y
# ningún provider de visión (hoy solo cloud, `openai_compatible`) escribe en
# el libro de cuota de planes — ver el gate `log_llm_usage_event` más abajo.
from auth import get_verified_user_id
from path_validators import assert_valid_uuid
from rate_limiter import RateLimiter
# [P3-DB-IMPORTS-FACADE · boy scout] `get_user_profile`/`log_llm_usage_event`
# migrados a la fachada `db` en el mismo edit que tocó este bloque
# (P1-VISION-LUNA · 2026-07-28) — `log_api_usage` salió del import: su único
# call site en este archivo (visión) se reemplazó por `log_llm_usage_event`.
from db import (
    log_consumed_meal, get_consumed_meals_today, save_visual_entry,
    delete_consumed_meal, get_user_profile, log_llm_usage_event,
    # [P1-DIARY-HISTORY · 2026-07-31] Agregación por día de `consumed_meals`
    # (ver `api_get_consumed_range`). Vía la fachada `db`, no `db_core`
    # directo — convención P3-DB-IMPORTS-FACADE.
    execute_sql_query,
    get_or_create_session, create_chat_attachment, build_chat_attachment_url,
)
from vision_agent import (
    process_image_with_vision, get_multimodal_embedding,
    _vision_model_name,
)
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

# [P1-MANUAL-FOOD-LOG · 2026-08-11] El componedor manda REFERENCIAS, nunca macros
# (doctrina de `consumed-from-plan`): la aritmética corre server-side en
# `food_search.resolve_line`, una vez, al enviar. La única excepción es
# `ref="custom"` — un alimento que el catálogo no conoce, donde el usuario declara
# con la misma autoridad que ya tiene el escáner de fotos (y con los mismos topes).
class ManualMealLine(BaseModel):
    ref: str = Field(..., min_length=1, max_length=200)
    qty: float = Field(default=1.0, gt=0.0, le=5000.0)
    unit: str = Field(default="g", max_length=24)
    name: Optional[str] = Field(default=None, max_length=120)
    macros: Optional[dict] = None


class ManualMealRequest(BaseModel):
    lines: List[ManualMealLine] = Field(..., min_length=1, max_length=12)
    meal_name: Optional[str] = Field(default=None, max_length=200)
    # `extra` NO está en `_SLOT_CANON_MAP`: no atenúa ningún plato del plan ni
    # bloquea swap/PDF (P1-EATEN-RECIPE-LOCK). Es el default correcto para el
    # componedor: lo que SÍ es del plan ya tiene «me lo comí».
    meal_type: str = Field(default="extra", max_length=32)
    days_ago: int = Field(default=0, ge=0, le=7)
    # Arranca APAGADO en el cliente: descontar sin pedirlo convierte cada antojo
    # anotado en una mutación de inventario que el usuario no pidió.
    deduct_pantry: bool = False


class RepeatMealRequest(BaseModel):
    """Repetir toma COORDENADAS (el id de una fila propia), no contenido — el
    cliente no puede inventar macros por esta vía tampoco."""
    source_meal_id: str = Field(..., min_length=8, max_length=64)
    meal_type: Optional[str] = Field(default=None, max_length=32)
    days_ago: int = Field(default=0, ge=0, le=7)


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
    # [P1-DIARY-EDITABLE · 2026-07-28] Antes el POST siempre escribía "ahora":
    # un usuario que olvidó loguear el fin de semana no podía corregirlo, y
    # el chat coach (que SÍ lee el diario) veía el hueco como "SIN REGISTRO"
    # para siempre. 0=hoy (default, comportamiento previo intacto) .. 7=hace
    # una semana — mismo tope que `tools._CONSUMED_MAX_DAYS_AGO` tras este fix
    # (antes 3, divergente del endpoint que no aceptaba backdate alguno).
    # ge/le son el contrato de FRONTERA HTTP (422 si el cliente manda basura
    # estructural); `_clamp_diary_days_ago` abajo es una segunda capa
    # defensiva (mismo patrón que `_reject_non_finite` sobre los macros).
    days_ago: int = Field(default=0, ge=0, le=7)

    # [P1-PHOTO-DEDUCTS · 2026-08-07] Ingredientes CONFIRMADOS por el usuario en
    # el modal del escáner, para descontarlos de la Nevera. Antes este modelo no
    # tenía el campo y `extra: "ignore"` lo descartaba en silencio: escanear
    # comida registraba macros y la Nevera no se enteraba. Es el ÚNICO surface
    # donde el cliente sí manda ingredientes — y puede, porque no describen un
    # plato del plan sino lo que el usuario declara haber comido, igual que si
    # lo escribiera en el chat. La confirmación humana en el modal es la
    # autorización; el backend solo la ejecuta.
    #
    # Caps de forma (no de negocio): un plato real no pasa de ~30 componentes,
    # y cada string es "2 huevos", no un párrafo. Sin cap, un payload gigante
    # haría girar el loop de resolución + N queries por item.
    ingredients: Optional[List[str]] = Field(default=None, max_length=40)

    model_config = {"extra": "ignore"}

    @field_validator("ingredients")
    @classmethod
    def _clean_ingredients(cls, v):
        """Descarta entradas vacías/basura y capa el largo por item.

        `_parse_quantity` ya ignora strings de <3 chars, pero filtrarlos acá
        evita que lleguen a `not_in_pantry` y le ensucien al usuario el aviso
        de "no estaban registrados" con ruido que él nunca escribió.
        """
        if v is None:
            return None
        limpio = [str(x).strip()[:120] for x in v if x and len(str(x).strip()) >= 3]
        return limpio or None

    @field_validator("calories", "protein", "carbs", "healthy_fats")
    @classmethod
    def _reject_non_finite(cls, v: float) -> float:
        # [P2-DIARY-NO-PYDANTIC] NaN/Infinity bypassean `ge=0` de Pydantic
        # en algunas versiones (Pydantic v2 los acepta como float válidos).
        # Validator explícito.
        if not math.isfinite(v):
            raise ValueError("Macros deben ser números finitos.")
        return v


class ConsumedFromPlanRequest(BaseModel):
    """[P1-EAT-PLAN-MEAL · 2026-08-07] Payload para
    `POST /api/diary/consumed-from-plan`.

    El cliente manda SOLO coordenadas — jamás el contenido. Nombre, slot,
    macros e ingredientes se leen server-side del `plan_data` del propio
    usuario. Es la misma doctrina que I-Billing-1 (`tier` derivado del
    `plan_id` de PayPal, no de `data.get("tier")`) e I1 (`plan_id` nace del
    INSERT backend): si el cliente pudiera declarar los ingredientes, podría
    descontar de la Nevera lo que quisiera y en la cantidad que quisiera.

    Los límites superiores de los índices son cota de forma, no de negocio:
    rechazan basura estructural antes de tocar la DB. El rango real se valida
    contra el plan.
    """
    plan_id: str = Field(..., min_length=1, max_length=64)
    day_index: int = Field(..., ge=0, le=365)
    meal_index: int = Field(..., ge=0, le=50)
    # Mismo rango que `ConsumedMealRequest.days_ago` y `tools._CONSUMED_MAX_DAYS_AGO`.
    days_ago: int = Field(default=0, ge=0, le=7)

    model_config = {"extra": "ignore"}


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
# advisory lock + roundtrip a la DB tiene costo. 10/60s es generoso
# (UI espera click manual) y restringe spam claro. Mismo patrón que
# `_RECALC_LIMITER` y `_PDF_TELEMETRY_LIMITER` en routers/plans.py.
_PROGRESS_LIMITER = RateLimiter(max_calls=10, period_seconds=60)

# [P1-DIARY-UPLOAD-RATELIMIT · 2026-05-30] Rate-limiter para `/upload`, el
# endpoint MÁS CARO del router (Gemini Vision multimodal + embedding + INSERT a
# Storage POR request). Pre-fix el único gate era `verify_api_quota`, que es
# NO-OP para anónimos (toda su lógica está bajo `if verified_user_id:`). Un guest
# sin token con `user_id=guest` + un JPEG ~1KB podía disparar 1 Gemini Vision
# síncrono (ocupa el event loop en `--workers 1`) + 1 embedding por request, sin
# límite → cost-amplification y degradación de disponibilidad. El hermano
# `/progress` SÍ tenía `_PROGRESS_LIMITER`; el endpoint caro no. Hermano omitido
# del P2-GUEST-LLM-RATELIMIT (que cubrió /swap-meal + /recipe/expand). Bucket por
# user_id (autenticado) o `ip:<client_ip>` (guest) → aísla abusadores. Se APILA
# sobre `verify_api_quota` (no lo reemplaza). Cap 10/60s holgado para el flujo
# real de subir fotos del diario.
_VISION_UPLOAD_LIMITER = RateLimiter(max_calls=10, period_seconds=60)

# [P1-DIARY-EDITABLE · 2026-07-28] Rate-limiter para `DELETE /consumed/{meal_id}`.
# Borrar una comida mal registrada es cero costo LLM (un DELETE filtrado por
# user_id) — aplicar `verify_api_quota` (el paywall mensual: gratis=15,
# basic=50...) bloquearía la corrección de un error de tap justo cuando el
# usuario ya agotó su cap del mes, doctrina idéntica a `_RESTOCK_LIMITER` /
# `_CONSUME_LIMITER` (routers/plans.py, P1-NEVERA-QUOTA-EXEMPT) y
# `_VISION_UPLOAD_LIMITER` arriba. 20/60s: house default de esta clase de
# endpoint (mutación barata, anti-spam sin ánimo de restringir uso legítimo).
# `RateLimiter.__call__` ya envuelve `Depends(get_verified_user_id)` y
# retorna el `verified_user_id` — el endpoint NO necesita un segundo
# `Depends(get_verified_user_id)` aparte (mismo patrón que api_restock).
_DELETE_CONSUMED_LIMITER = RateLimiter(max_calls=20, period_seconds=60)

# [P1-EAT-PLAN-MEAL · 2026-08-07] Rate-limiter para
# `POST /consumed-from-plan`. Quota-exempt por la MISMA doctrina que
# `/restock` e `/inventory/consume` (P1-NEVERA-QUOTA-EXEMPT): marcar como
# comido un plato que el usuario YA pagó al generar el plan no tiene costo
# LLM — es un SELECT sobre `plan_data` + un INSERT + restas en la Nevera.
# Aplicarle `verify_api_quota` sería doblemente absurdo: al llegar al cap el
# usuario no podría registrar lo que come (congelando la Nevera y el diario
# calórico justo cuando más los necesita), y `get_monthly_api_usage` cuenta
# toda fila de `api_usage` sin filtrar endpoint — así que registrar una
# comida quemaría crédito de PLANES. 20/60s es el house default de esta
# clase (mutación barata, anti-spam sin restringir uso legítimo: 5-6 comidas
# al día están holgadamente dentro).
_PLAN_MEAL_LIMITER = RateLimiter(max_calls=20, period_seconds=60)

# [P1-MANUAL-FOOD-LOG · 2026-08-11] Los cuatro writes/reads del registro manual son
# cero costo LLM ⇒ quota-exempt con RateLimiter, la misma doctrina que las filas de
# «Historial-quota-exemption» (CLAUDE.md): al cap, el usuario tiene que poder seguir
# anotando lo que come, y cada anotación NO puede quemar crédito de PLANES.
# `_CONSUMED_WRITE_LIMITER` cierra además una omisión histórica: /consumed era el
# único write del diario SIN limitador.
_CONSUMED_WRITE_LIMITER = RateLimiter(max_calls=20, period_seconds=60)
_MANUAL_MEAL_LIMITER = RateLimiter(max_calls=20, period_seconds=60)
_FOOD_FREQUENT_LIMITER = RateLimiter(max_calls=30, period_seconds=60)
_REPEAT_MEAL_LIMITER = RateLimiter(max_calls=20, period_seconds=60)


# [P3-VISION-UPLOAD-VALIDATION · 2026-05-20] Whitelist de content_types
# permitidos para `/upload`. Cierra el gap F3 del audit
# `docs/gaps-audit-2026-05.md`: pre-fix el endpoint aceptaba el
# `file.content_type` declarado por el cliente Y lo pasaba directo al
# object storage + a `process_image_with_vision`. Vectores cerrados:
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
    purpose: str = Form("diary"),
    tz_offset_mins: int = Form(0),
    # [P1-MEAL-SCAN-GEMMA · 2026-07-12 → P1-VISION-LUNA · 2026-07-28 →
    # P1-VISION-NO-LOCAL · 2026-07-28] Quota-exempt PERMANENTE (doctrina
    # P1-NEVERA-QUOTA-EXEMPT): `verify_api_quota` devolvería 402 al cap
    # mensual Y cada scan quemaría un crédito de planes (`get_monthly_api_usage`
    # cuenta toda fila de api_usage sin filtrar endpoint). Anti-spam:
    # _VISION_UPLOAD_LIMITER (10/60s por user/IP) — el único provider de
    # visión hoy es cloud (`openai_compatible`); el gasto real NO reactiva
    # este paywall — va a `llm_usage_events` (libro de COSTO) vía
    # `log_llm_usage_event`, ver el gate más abajo.
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
    # [P1-DIARY-UPLOAD-RATELIMIT · 2026-05-30] throttle por user_id/IP.
    _rl_vision: Optional[str] = Depends(_VISION_UPLOAD_LIMITER),
):
    try:
        # [P1-PROD-AUDIT-3 · 2026-05-30] Validación de seguridad IDOR.
        # ANTES el guard llevaba un short-circuit `and user_id != session_id`:
        # un atacante autenticado (token propio válido) podía escribir en el
        # visual_diary de una víctima enviando user_id == session_id == <victim_uuid>
        # → la condición `user_id != session_id` era False → el check de ownership
        # NUNCA se evaluaba → actual_user_id = victim → cross-user write (línea
        # background _save_visual_entry_background). Hermano no-guardado de la
        # clase P2-CHAT-WRITE-IDOR; /consumed POST y /consumed/{user_id} GET ya
        # tenían el guard correcto SIN el escape hatch. Se elimina el short-circuit:
        # el ownership se exige SIEMPRE para user_id autenticado.
        # Validación UUID previa (paridad con /consumed/{user_id} GET, línea ~470)
        # — además cierra inyección de path en la Storage key `{actual_user_id}/...`.
        assert_valid_uuid(user_id, allow_guest=True)
        if session_id is not None:
            assert_valid_uuid(session_id, allow_guest=True)
        purpose = (purpose or "diary").strip().lower()
        if purpose not in {"diary", "chat"}:
            raise HTTPException(status_code=422, detail="Propósito de imagen inválido.")
        if purpose == "chat" and not session_id:
            raise HTTPException(status_code=422, detail="El chat requiere session_id.")
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado.")

        actual_user_id = user_id if user_id != "guest" else session_id

        # [P1-DIARY-UPLOAD-GUEST-IDOR · 2026-05-30] Hermano del guest-path que
        # P1-PROD-AUDIT-3 dejó abierto. Aquel fix endureció SOLO la rama
        # autenticada (eliminó el escape `and user_id != session_id`); la rama
        # guest (`actual_user_id = session_id`) seguía confiando en un
        # `session_id` arbitrario SIN lookup de ownership. Un atacante NO
        # autenticado (sin token → `verify_api_quota` es no-op) que envíe
        # `user_id="guest"` + `session_id=<UUID real de la víctima>` lograba que
        # `_save_visual_entry_background` escribiera en el `visual_diary` de la
        # víctima (la tabla NO tiene FK a auth.users → la DB acepta cualquier
        # UUID; SERVICE_ROLE bypassea RLS). Eso contamina el RAG/memoria de la
        # víctima (vector de prompt-injection cross-user persistente). La capa
        # de chat defiende el caso simétrico vía `get_session_owner`
        # (chat.py /message,/stream,POST) — diary no tenía análogo.
        #
        # Fix: en la rama guest, NO atribuir la escritura a una identidad
        # reclamada salvo que el token verificado coincida. Si no hay token (o
        # no coincide), `actual_user_id=None` → se omite la persistencia (el
        # análisis + macros igual se devuelven en la respuesta, así que el scan
        # de comida sigue funcionando para invitados; solo no se persiste a un
        # diario que un invitado tampoco puede leer — la GET exige auth).
        # Tooltip-anchor: P1-DIARY-UPLOAD-GUEST-IDOR.
        if user_id == "guest" and actual_user_id:
            if not verified_user_id or verified_user_id != actual_user_id:
                logger.warning(
                    f"🛡️ [P1-DIARY-UPLOAD-GUEST-IDOR] guest upload con "
                    f"session_id reclamando identidad sin token coincidente "
                    f"(verified={'set' if verified_user_id else 'none'}). "
                    f"Persistencia omitida — posible IDOR cross-user."
                )
                actual_user_id = None

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
        attachment_id = None
        upload_success = False

        # 1. Intentar subir al object storage de visual_diary
        if _storage_client and purpose == "diary":
            try:
                res = await asyncio.to_thread(
                    _storage_client.storage.from_("visual_diary_images").upload,
                    path=unique_filename,
                    file=file_bytes,
                    file_options={"content-type": file.content_type}
                )
                image_url = _storage_client.storage.from_("visual_diary_images").get_public_url(unique_filename)
                upload_success = True
                logger.info(f"☁️ Imagen guardada en object storage: {image_url}")
            except Exception as sb_err:
                logger.error(f"⚠️ Error subiendo al object storage de visual_diary (¿Existe el bucket 'visual_diary_images'?): {sb_err}")
                upload_success = False

        # 2. [P1-MEAL-SCAN-GEMMA · 2026-07-12] Storage OPCIONAL. Tras la
        # migración a Neon `_storage_client = None` (db_core) y este paso
        # abortaba con 500 ANTES de llegar al análisis de visión — el botón
        # "Escanear comida" moría aquí aunque el provider de visión estuviera
        # vivo. La foto se analiza en memoria; sin storage solo se pierde la
        # miniatura del Diario Visual (image_url vacía), no el análisis ni
        # la description.
        if not upload_success and purpose == "diary":
            logger.info(
                "📷 [DIARY-UPLOAD] Sin object storage configurado — "
                "la imagen se analiza en memoria y no se persiste (image_url vacía)."
            )

        # El chat necesita persistencia privada aunque el proveedor externo de
        # object storage no esté configurado. Guardamos el JPEG ya acotado por el
        # cliente en Neon y lo servimos por URL HMAC temporal. Guests siguen
        # fail-closed: no se atribuye media a una identidad no verificada.
        if purpose == "chat" and actual_user_id and session_id:
            await asyncio.to_thread(get_or_create_session, session_id, user_id=actual_user_id)
            attachment_id = await asyncio.to_thread(
                create_chat_attachment,
                session_id,
                actual_user_id,
                file_bytes,
                sniffed_ct,
                file.filename,
            )
            if attachment_id:
                image_url = build_chat_attachment_url(attachment_id)
                upload_success = True
            else:
                raise HTTPException(
                    status_code=503,
                    detail="No pudimos guardar la imagen de forma privada. Reintenta en un momento.",
                )

        # 3. Procesar imagen con Visión SINCRÓNICAMENTE
        logger.info("\n-------------------------------------------------------------")
        logger.info("📸 [VISION AGENT] Procesando nueva imagen subida...")
        vision_result = await process_image_with_vision(file_bytes)
        
        description = vision_result.get("description", "No se pudo analizar la imagen.")
        is_food = vision_result.get("is_food", False)
        calories = vision_result.get("calories", 0)
        # [P2-DIARY-SCAN-MACROS · 2026-05-30] El vision agent ya estima las 4
        # macros + un nombre corto; las exponemos al frontend para el modal
        # "Escanear comida → registrar macros" del Dashboard. NO se persiste
        # nada aquí (la confirmación + INSERT a consumed_meals la hace el
        # usuario vía POST /api/diary/consumed tras revisar/editar).
        # Tooltip-anchor: P2-DIARY-SCAN-MACROS.
        protein = vision_result.get("protein", 0)
        carbs = vision_result.get("carbs", 0)
        healthy_fats = vision_result.get("healthy_fats", 0)
        meal_name = vision_result.get("meal_name", "") or ""
        # [P2-DIARY-SCAN-MACROS · 2026-05-30] Propaga si el analizador falló
        # (timeout / 429 RESOURCE_EXHAUSTED / provider caído) para que el modal
        # distinga "intenta más tarde" de "no es comida".
        analysis_failed = vision_result.get("analysis_failed", False)
        
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
                    # [P2-DIARY-ASYNC-SYNC-DB · 2026-05-30] offload sync DB del
                    # event loop (handler async). Hermano del contrato
                    # P1-ASYNC-SYNC-DB-BLOCKING / P2-PROD-AUDIT-3 (ver chat.py
                    # /tts que ya usa asyncio.to_thread para log_api_usage).
                    profile = await asyncio.to_thread(get_user_profile, user_id)
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
                        # [P2-DIARY-ASYNC-SYNC-DB · 2026-05-30] INSERT síncrono
                        # offloaded del event loop (handler async).
                        await asyncio.to_thread(
                            execute_sql_write,
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

            # [P1-MEAL-SCAN-GEMMA · 2026-07-12 → P1-VISION-LUNA · 2026-07-28
            # → P1-VISION-NO-LOCAL · 2026-07-28] Encender un provider CLOUD
            # pago (Luna) reabría el bug que P1-MEAL-SCAN-GEMMA cerró:
            # `log_api_usage` escribía en `api_usage` (libro de CUOTA de
            # planes) y `get_monthly_api_usage` cuenta TODA fila sin filtrar
            # endpoint — 3 scans/día agotarían el cap gratis (15/mes) en 5
            # días, congelando la generación de planes por fotografiar
            # comida. El gasto real (modelo + tokens si el provider los
            # devuelve + USD) va a `llm_usage_events` (libro de COSTO, NO de
            # cuota) vía `log_llm_usage_event` — sigue siendo auditable
            # (mismo sitio del canario de Luna) sin cobrarle un plan al
            # usuario. `log_llm_usage_event` es best-effort y ya silencia sus
            # propias excepciones — no envolver en otro try.
            #
            # [P1-VISION-NO-LOCAL] El gate ANTES excluía el path LOCAL
            # (`not is_vision_local()`, gemma/Ollama costo cero): sin
            # provider local, esa rama nunca podía ser False, así que el
            # accessor quedaba muerto — eliminado junto con el resto de
            # Ollama. Con un ÚNICO provider (cloud), esta rama SIEMPRE emite
            # la fila cuando hay usuario autenticado; no hay condición de
            # provider que verificar.
            if actual_user_id and actual_user_id != session_id:
                # [P2-DIARY-ASYNC-SYNC-DB · 2026-05-30] offload sync DB del event loop.
                # `_vision_model_name()` es el modelo del ÚNICO provider de
                # visión (`openai_compatible`) — accessor vivo de
                # vision_agent, no un literal hardcodeado que se
                # desincronizaría de MEALFIT_VISION_MODEL.
                await asyncio.to_thread(
                    log_llm_usage_event,
                    user_id=actual_user_id,
                    model=_vision_model_name(),
                    node="vision_scan",
                )

            # 4. Guardar en DB en segundo plano (embedding + insert).
            # `description` se persiste limpio (sin poison_pill) — la
            # signal chrono se canaliza por `pipeline_metrics`.
            # [P1-DIARY-UPLOAD-GUEST-IDOR · 2026-05-30] Gate en actual_user_id:
            # si la rama guest lo anuló (session_id reclamando identidad sin
            # token coincidente), NO persistir — evita el cross-user write a un
            # visual_diary ajeno (y el orphan row con user_id NULL).
            if actual_user_id and purpose == "diary":
                background_tasks.add_task(
                    _save_visual_entry_background,
                    actual_user_id, image_url, description
                )
        else:
            logger.info("➡️ La imagen fue ignorada porque no se detectaron alimentos.")

        return {
            "success": True,
            "is_food": is_food,
            "analysis_failed": analysis_failed,
            # [P1-MEAL-SCAN-GEMMA · 2026-07-12 → P1-VISION-NO-LOCAL · 2026-07-28]
            # `busy=True` distinguía "hay otro scan en vuelo, reintenta en
            # segundos" de "el analizador está caído" bajo el provider LOCAL
            # (single-flight de una sola GPU). Sin provider local, ningún
            # dispatch vuelve a setear `busy` — el key queda SIEMPRE False.
            # Se conserva (no se rompe el contrato de respuesta del frontend,
            # que ya trata `busy` falsy como "no ocupado") en vez de tocar
            # ScanMealModal.jsx fuera del alcance de este P-fix.
            "busy": vision_result.get("busy", False),
            # [P1-CHAT-VISION-GEMMA · 2026-07-12] Clasificación de la foto:
            # 'plato' (macros → diario), 'items' (compra → Nevera vía chat-
            # agent/modify_pantry_inventory), 'otro'. `items` sanitizados.
            "photo_kind": vision_result.get("photo_kind") or ("plato" if is_food else "otro"),
            "items": vision_result.get("items") or [],
            "description": description,
            "image_url": image_url,
            "attachment_id": attachment_id,
            # [P2-DIARY-SCAN-MACROS · 2026-05-30] Macros estimadas + nombre corto
            # para el modal de registro del Dashboard. Cero costo extra (ya se
            # computaron en process_image_with_vision).
            "meal_name": meal_name,
            "macros": {
                "calories": calories,
                "protein": protein,
                "carbs": carbs,
                "healthy_fats": healthy_fats,
            },
            # [P1-DIARY-PROMPT-INJECTION · 2026-05-15] flag estructurado para
            # frontend; reemplaza la inyección de texto en `description`.
            "red_alert": chrono_red_alert,
            "red_alert_meta": {
                "local_hour": chrono_local_hour,
                "schedule_type": chrono_schedule_type,
            } if chrono_red_alert else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/diary/upload: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

def _save_visual_entry_background(user_id: str, image_url: str, description: str):
    """Background task: genera embedding y guarda en la tabla visual_diary."""

    # [P1-COHERE-EMBED-V4] purpose="document": se persiste para retrieval asimétrico.
    embedding = get_multimodal_embedding(description, purpose="document")
    if embedding:
        logger.info(f"📦 Guardando entrada visual en la DB (Vector {len(embedding)}d)...")
        save_visual_entry(user_id=user_id, image_url=image_url, description=description, embedding=embedding)
        logger.info("✅ ¡Imagen registrada en el Diario Visual con éxito!")
    else:
        logger.warning("⚠️ No se pudo vectorizar la imagen. Abortando guardado.")


# [P1-DIARY-EDITABLE · 2026-07-28] Duplicado DELIBERADO de `tools._clamp_days_ago`
# (backend/tools.py) — NO se importa desde `tools.py` acá: ese módulo carga
# LangChain (`langchain_core.tools`), `graph_orchestrator` y `llm_provider`
# completos (deps del chat-agent, ~decenas de imports transitivos) solo para
# reusar 6 líneas de aritmética. Un router HTTP no debería arrastrar la capa
# del agente por esto. Si el rango vuelve a divergir entre el chat y este
# endpoint, actualizar AMBAS constantes (`tools._CONSUMED_MAX_DAYS_AGO` y
# `_DIARY_MAX_DAYS_AGO` abajo) en el mismo commit — hermana textual, misma
# semántica: garbage/negativo/futuro → 0 (hoy), nunca más allá del tope.
_DIARY_MAX_DAYS_AGO = 7


def _clamp_diary_days_ago(days_ago) -> int:
    """0=hoy .. 7=máximo pasado. Garbage/futuro/negativo → 0 (hoy).

    Segunda capa defensiva DETRÁS del `Field(ge=0, le=7)` de
    `ConsumedMealRequest.days_ago` (que ya rechaza con 422 en la frontera
    HTTP). Esta función existe para callers directos (tests, scripts) y para
    que el contrato "nunca en el futuro, nunca más de 7 días atrás" no
    dependa exclusivamente de la validación de Pydantic.
    """
    try:
        d = int(days_ago)
    except (TypeError, ValueError):
        return 0
    return max(0, min(_DIARY_MAX_DAYS_AGO, d))


def _persist_consumed_meal(
    *, user_id: str, meal_name: str, meal_type: str, calories, protein, carbs,
    healthy_fats, ingredients, days_ago, background_tasks: BackgroundTasks,
    source: str, deduct: bool = True,
) -> dict:
    """[P1-MANUAL-FOOD-LOG · 2026-08-11] EL camino de escritura del diario, extraído
    del cuerpo de `/consumed` para que la foto y el componedor manual lo COMPARTAN en
    vez de copiarlo. Aquí viven, una sola vez: el fail-loud de persistencia
    (P1-PROD-AUDIT-3), el sentinel «deduped» del doble-tap (P2-CONSUMED-DEDUP-
    INVENTORY), el atado al ledger (P1-CONSUMPTION-LEDGER), el backdate que preserva
    la hora (P1-DIARY-EDITABLE) y la forma de la respuesta con lo que NO bajó
    (P1-PANTRY-NAME-RESOLUTION). Una copia garantizaría que en tres meses uno de los
    dos endpoints tenga el próximo fix y el otro no.

    `source` viaja al ledger — antes estaba clavado a "photo" en el único caller.

    `deduct=False` con ingredientes: se GUARDAN (el coach los lee) pero se marcan
    `inventory_synced` igual, porque la decisión explícita del usuario de no tocar su
    Nevera debe sobrevivir también a la reconciliación del cierre de chunk — si no,
    el «no» de hoy se convertiría en un descuento silencioso dentro de unos días."""
    from datetime import datetime, timezone, timedelta
    _days_ago = _clamp_diary_days_ago(days_ago)
    consumed_at_override = (
        datetime.now(timezone.utc) - timedelta(days=_days_ago)
    ).isoformat()

    _ingredients = ingredients or None
    _logged_ok = log_consumed_meal(
        user_id, meal_name, int(calories), int(protein), int(carbs),
        int(healthy_fats), ingredients=_ingredients, meal_type=meal_type,
        mark_inventory_synced=bool(_ingredients),
        consumed_at_override=consumed_at_override,
    )
    if not _logged_ok:
        raise HTTPException(
            status_code=500,
            detail="No se pudo registrar la comida. Reintenta en unos segundos.",
        )

    _already_logged = (_logged_ok == "deduped")
    _summary = {}
    if _ingredients and deduct and not _already_logged:
        import db_inventory
        _summary = db_inventory.deduct_consumed_meal_from_inventory(
            user_id, _ingredients,
            consumed_meal_id=(_logged_ok if isinstance(_logged_ok, str) else None),
            source=source,
        ) or {}

    if not _already_logged:
        background_tasks.add_task(trigger_incremental_learning, user_id)

    return {
        "success": True,
        "message": "Comida registrada exitosamente.",
        "already_logged": _already_logged,
        "meal_id": (_logged_ok if isinstance(_logged_ok, str) else None),
        "deducted": _summary.get("succeeded") or [],
        "inferred": _summary.get("inferred") or [],
        "not_in_pantry": _summary.get("not_in_pantry") or [],
        "failed_to_deduct": _summary.get("failed_to_deduct") or [],
    }


@router.post("/consumed")
def api_log_consumed_meal(
    payload: ConsumedMealRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    # [P1-MANUAL-FOOD-LOG · 2026-08-11] Era el único write del diario sin RateLimiter
    # (omisión histórica). `RateLimiter.__call__` envuelve `get_verified_user_id` y
    # devuelve el user_id autenticado — mismo patrón que los vecinos.
    verified_user_id: Optional[str] = Depends(_CONSUMED_WRITE_LIMITER),
):
    """Registra una comida consumida manualmente desde el frontend.

    [P2-DIARY-NO-PYDANTIC · 2026-05-15] Migrado de `data: dict = Body(...)`
    a `ConsumedMealRequest`. Pydantic valida tipos + rangos + NaN/Infinity
    antes del handler (422 con detalle estructurado en lugar de 500
    silencioso).

    [P1-DIARY-EDITABLE · 2026-07-28] `days_ago` (0..7) permite backdatear.

    [P1-MANUAL-FOOD-LOG · 2026-08-11] El cuerpo vive en `_persist_consumed_meal`,
    compartido con el componedor manual. Aquí solo quedan las decisiones de ESTA
    superficie: IDOR, guests y `source="photo"`.
    """
    try:
        user_id = payload.user_id

        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")

        if not user_id or user_id == "guest":
            return {"success": False, "message": "Inicia sesión para registrar comidas."}

        return _persist_consumed_meal(
            user_id=user_id, meal_name=payload.meal_name, meal_type=payload.meal_type,
            calories=payload.calories, protein=payload.protein, carbs=payload.carbs,
            healthy_fats=payload.healthy_fats, ingredients=payload.ingredients,
            days_ago=payload.days_ago, background_tasks=background_tasks,
            source="photo",
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/diary/consumed POST: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/consumed/manual")
def api_log_manual_meal(
    payload: ManualMealRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    verified_user_id: Optional[str] = Depends(_MANUAL_MEAL_LIMITER),
):
    """[P1-MANUAL-FOOD-LOG · 2026-08-11] El componedor: buscar → apilar líneas →
    registrar, sin foto, sin chat y sin gastar un crédito.

    El cliente manda referencias (`dish:`/`food:`/`custom`); la aritmética corre aquí
    UNA vez. Si una ref no resuelve: 422 y NO se registra a medias — el usuario vería
    949 kcal en pantalla y 589 en el diario."""
    try:
        if not verified_user_id:
            return {"success": False, "message": "Inicia sesión para registrar comidas."}

        import food_search
        from shopping_calculator import get_master_ingredients
        catalogo = get_master_ingredients() or []

        resueltas, pantry = [], []
        totales = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fats": 0.0}
        for ln in payload.lines:
            try:
                r = food_search.resolve_line(ln.model_dump(), catalogo)
            except food_search.LineaIrresoluble as li:
                raise HTTPException(status_code=422, detail=f"Línea irresoluble: {li}")
            resueltas.append(r)
            pantry.extend(r["pantry_lines"])
            for k in totales:
                totales[k] += float(r["macros"].get(k) or 0.0)

        pantry = food_search.merge_pantry_lines(pantry)
        meal_name = (payload.meal_name or "").strip() or food_search.derive_meal_name(
            [r["name"] for r in resueltas]
        )

        resp = _persist_consumed_meal(
            user_id=verified_user_id, meal_name=meal_name, meal_type=payload.meal_type,
            calories=totales["kcal"], protein=totales["protein"], carbs=totales["carbs"],
            healthy_fats=totales["fats"], ingredients=(pantry or None),
            days_ago=payload.days_ago, background_tasks=background_tasks,
            source="manual", deduct=bool(payload.deduct_pantry),
        )
        resp["totals"] = {k: int(v) for k, v in totales.items()}
        resp["lines"] = [
            {"name": r["name"], "grams": r["grams"], "kcal": int(r["macros"]["kcal"])}
            for r in resueltas
        ]
        return resp
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/diary/consumed/manual POST: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.get("/foods/frequent")
def api_frequent_foods(
    limit: int = 8,
    verified_user_id: Optional[str] = Depends(_FOOD_FREQUENT_LIMITER),
):
    """[P1-MANUAL-FOOD-LOG · 2026-08-11] «Lo que más registras»: GROUP BY de 30 días
    sobre `consumed_meals`. Sin tabla de favoritos y sin curación manual — pedirle al
    usuario que mantenga una lista es pedirle trabajo por algo que el sistema calcula
    solo. Devuelve además el id de la fila más reciente de cada grupo, que es la
    coordenada que `/consumed/repeat` acepta."""
    try:
        if not verified_user_id:
            return {"items": []}
        limit = max(1, min(20, int(limit or 8)))
        rows = execute_sql_query(
            """
            SELECT meal_name,
                   COUNT(*)::int AS veces,
                   ROUND(AVG(calories))::int AS kcal,
                   ROUND(AVG(protein))::int AS protein,
                   ROUND(AVG(carbs))::int AS carbs,
                   ROUND(AVG(healthy_fats))::int AS fats,
                   (ARRAY_AGG(id::text ORDER BY consumed_at DESC))[1] AS last_meal_id
            FROM consumed_meals
            WHERE user_id = %s AND consumed_at > NOW() - INTERVAL '30 days'
            GROUP BY meal_name
            ORDER BY COUNT(*) DESC, MAX(consumed_at) DESC
            LIMIT %s
            """,
            (verified_user_id, limit),
            fetch_all=True,
        ) or []
        return {"items": rows}
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/diary/foods/frequent: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/consumed/repeat")
def api_repeat_consumed_meal(
    payload: RepeatMealRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    verified_user_id: Optional[str] = Depends(_REPEAT_MEAL_LIMITER),
):
    """[P1-MANUAL-FOOD-LOG · 2026-08-11] «Lo mismo de ayer»: re-registra una fila
    PROPIA por su id. Coordenadas, no contenido. La lectura filtra por dueño y el 404
    es el mismo para «no existe» y «es de otro» (invariante I2 aplicada al diario).

    NO descuenta Nevera: repetir es la vía rápida de anotar, y un descuento implícito
    en una acción de un toque es exactamente la mutación-sin-pedir que el interruptor
    del componedor existe para evitar."""
    try:
        if not verified_user_id:
            return {"success": False, "message": "Inicia sesión para registrar comidas."}

        row = execute_sql_query(
            """
            SELECT meal_name, meal_type, calories, protein, carbs, healthy_fats, ingredients
            FROM consumed_meals WHERE id = %s AND user_id = %s
            """,
            (payload.source_meal_id, verified_user_id),
            fetch_one=True,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Comida no encontrada.")

        return _persist_consumed_meal(
            user_id=verified_user_id, meal_name=row["meal_name"],
            meal_type=(payload.meal_type or row.get("meal_type") or "extra"),
            calories=row.get("calories") or 0, protein=row.get("protein") or 0,
            carbs=row.get("carbs") or 0, healthy_fats=row.get("healthy_fats") or 0,
            ingredients=(row.get("ingredients") or None),
            days_ago=payload.days_ago, background_tasks=background_tasks,
            source="manual", deduct=False,
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/diary/consumed/repeat POST: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/consumed-from-plan")
def api_log_consumed_meal_from_plan(
    payload: ConsumedFromPlanRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    # [P1-EAT-PLAN-MEAL · 2026-08-07] Quota-exempt (ver `_PLAN_MEAL_LIMITER`).
    # `RateLimiter.__call__` ya envuelve `Depends(get_verified_user_id)` y
    # devuelve el user_id autenticado — no hace falta un segundo Depends
    # (mismo patrón que `api_delete_consumed_meal` / `api_restock`).
    verified_user_id: Optional[str] = Depends(_PLAN_MEAL_LIMITER),
):
    """[P1-EAT-PLAN-MEAL · 2026-08-07] "Me comí este plato del plan".

    El camino de consumo MÁS PRECISO del sistema, y el único sin adivinanza:
    el plato del plan ya trae su lista de ingredientes con cantidades (la
    misma que lee `reserve_plan_ingredients`), así que descontar la Nevera es
    aritmética sobre datos que el backend escribió — cero LLM, cero visión,
    cero `_infer_typical_portion`.

    Contrasta con los otros dos surfaces:
      - chat (`tools.log_consumed_meal`): la LLM tiene que ACERTAR la lista de
        ingredientes y sus cantidades a partir de texto libre.
      - foto (`/upload` → `/consumed`): hoy ni siquiera manda ingredientes, así
        que no toca la Nevera.

    **El cliente manda coordenadas, nunca contenido.** `plan_data` se lee
    server-side filtrando `AND user_id = %s` (invariante I2), y de ahí salen
    nombre, slot, macros e ingredientes. Un cliente que pudiera declararlos
    descontaría de la Nevera lo que quisiera.

    Relación con el matcher por SLOT (P1-TODAY-REMAINING): ese heurístico
    DERIVA "ya comiste" comparando `meal_type` del diario contra el slot del
    plan, y se declara ambiguo cuando hay ≥2 slots iguales (2-3 meriendas).
    Este endpoint no compite con él: convierte la inferencia en una
    DECLARACIÓN del usuario sobre un plato concreto, que es justamente el dato
    que al heurístico le falta. El registro sigue guardándose por `meal_type`,
    así que el matcher lo ve y atenúa la card como siempre.

    Tooltip-anchor: P1-EAT-PLAN-MEAL-ENDPOINT
    """
    try:
        assert_valid_uuid(payload.plan_id)

        if not verified_user_id:
            raise HTTPException(status_code=403, detail="Prohibido.")

        # Invariante I2: la lectura filtra por dueño. El 404 es el MISMO para
        # "no existe" y "es de otro" — no filtramos existencia cross-user.
        row = execute_sql_query(
            "SELECT plan_data FROM meal_plans WHERE id = %s AND user_id = %s",
            (payload.plan_id, verified_user_id),
            fetch_one=True,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Plan no encontrado.")

        plan_data = row.get("plan_data") or {}
        if isinstance(plan_data, str):
            try:
                plan_data = json.loads(plan_data)
            except Exception:
                raise HTTPException(status_code=422, detail="El plan está corrupto.")

        days = plan_data.get("days") or []
        if not isinstance(days, list) or payload.day_index >= len(days):
            raise HTTPException(status_code=404, detail="Ese día no existe en tu plan.")

        meals = (days[payload.day_index] or {}).get("meals") or []
        if not isinstance(meals, list) or payload.meal_index >= len(meals):
            raise HTTPException(status_code=404, detail="Ese plato no existe en tu plan.")

        meal = meals[payload.meal_index] or {}
        meal_name = str(meal.get("name") or "").strip()
        if not meal_name:
            # Un plato sin nombre es plan corrupto, no input inválido del
            # cliente: 422 con mensaje propio en vez de registrar basura.
            raise HTTPException(status_code=422, detail="Ese plato no tiene nombre.")

        # `meal` es el SLOT del plan ('Desayuno'/'Almuerzo'/...). El diario lo
        # guarda en `meal_type`, que es la clave por la que el matcher
        # P1-TODAY-REMAINING atenúa la card.
        meal_type = str(meal.get("meal") or "snack").strip().lower() or "snack"

        def _macro(key: str) -> int:
            try:
                v = float(meal.get(key) or 0)
            except (TypeError, ValueError):
                return 0
            return int(v) if math.isfinite(v) and v > 0 else 0

        # El esquema del plato usa `fats`; el diario, `healthy_fats`.
        calories, protein = _macro("cals"), _macro("protein")
        carbs, healthy_fats = _macro("carbs"), _macro("fats")

        ingredients = [
            str(i) for i in (meal.get("ingredients") or [])
            if i and len(str(i).strip()) >= 3
        ]

        from datetime import datetime, timezone, timedelta
        _days_ago = _clamp_diary_days_ago(payload.days_ago)
        consumed_at_override = (
            datetime.now(timezone.utc) - timedelta(days=_days_ago)
        ).isoformat()

        # `mark_inventory_synced` solo si vamos a descontar acto seguido — si
        # no, la reconciliación al cierre del chunk debe poder recogerlo.
        logged = log_consumed_meal(
            verified_user_id, meal_name, calories, protein, carbs, healthy_fats,
            ingredients=ingredients or None,
            meal_type=meal_type,
            mark_inventory_synced=bool(ingredients),
            consumed_at_override=consumed_at_override,
        )
        if not logged:
            # Fail-loud (doctrina P1-PROD-AUDIT-3): sin esto un blip de DB
            # devolvía success:true sobre data fantasma.
            raise HTTPException(
                status_code=500,
                detail="No se pudo registrar la comida. Reintenta en unos segundos.",
            )

        # [P2-CONSUMED-DEDUP-INVENTORY · 2026-05-30] `"deduped"` = doble-tap
        # dentro de la ventana: NO hubo INSERT nuevo, así que descontar otra vez
        # bajaría la Nevera al doble del consumo real.
        already_logged = (logged == "deduped")
        summary = {}
        if ingredients and not already_logged:
            import db_inventory
            summary = db_inventory.deduct_consumed_meal_from_inventory(
                verified_user_id, ingredients,
                consumed_meal_id=(logged if isinstance(logged, str) else None),
                source="plan_meal",
            ) or {}

        if not already_logged:
            background_tasks.add_task(trigger_incremental_learning, verified_user_id)

        return {
            "success": True,
            "already_logged": already_logged,
            "meal_name": meal_name,
            "meal_type": meal_type,
            "calories": calories,
            # [P1-PANTRY-NAME-RESOLUTION · 2026-08-07] `not_in_pantry` viaja al
            # cliente para que la UI pueda decir QUÉ no bajó de la Nevera. Sin
            # esto el usuario ve "registrado" y asume que todo se descontó —
            # exactamente la mentira que aquel P-fix eliminó del lado del chat.
            "deducted": summary.get("succeeded") or [],
            "inferred": summary.get("inferred") or [],
            "not_in_pantry": summary.get("not_in_pantry") or [],
            "failed_to_deduct": summary.get("failed_to_deduct") or [],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/diary/consumed-from-plan: {str(e)}")
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
        
        meals = get_consumed_meals_today(user_id, date_str=date, tz_offset_mins=tzOffset)  # pyright: ignore[reportArgumentType]
        # meals garantizado no-None: get_consumed_meals_today siempre retorna lista
        # (todos los paths con return retornan [] o res:list; el None inferido es
        # un fall-through teórico del loop de retry que el control de flujo no alcanza).
        assert meals is not None

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


@router.get("/consumed-range/{user_id}")
def api_get_consumed_range(
    user_id: str,
    days: int = 14,
    tzOffset: Optional[int] = None,
    verified_user_id: str = Depends(get_verified_user_id),
):
    """[P1-DIARY-HISTORY · 2026-07-31] Totales por DÍA de los últimos N días.

    Existe para que la tira de días del diario signifique algo: sin esto, el
    selector solo puede mostrar fechas y el usuario tiene que ir tocando una a
    una para descubrir cuáles tienen registro. Con esto cada día lleva su kcal
    y su conteo, así que la tira es a la vez navegación y resumen de adherencia.

    La alternativa —pedir los ~14 días con 14 llamadas al endpoint de un día—
    multiplica por 14 el round-trip y la carga para la MISMA información. Aquí
    es UNA agregación en Postgres.

    El día se corta en la zona del USUARIO (`tzOffset` en minutos, la misma
    convención que `/consumed/{user_id}`): agrupar en UTC partiría las cenas
    dominicanas en dos días distintos — en RD (UTC-4) todo lo comido después de
    las 20:00 caería en el día siguiente.

    [I2] Filtra `AND user_id = %s` en SQL además del guard de ownership de
    arriba: defensa en profundidad, igual que el resto de lecturas del diario.
    """
    try:
        assert_valid_uuid(user_id, allow_guest=True)
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
        if not user_id or user_id == "guest":
            return {"days": []}

        # Clamp defensivo: `days` viene del cliente. Sin tope, un `days=99999`
        # convierte un endpoint de UI en un escaneo de tabla.
        try:
            n_days = max(1, min(int(days), 90))
        except (TypeError, ValueError):
            n_days = 14

        # tzOffset es el `getTimezoneOffset()` de JS: minutos que hay que SUMAR
        # a la hora local para obtener UTC (RD = +240). Se resta para pasar de
        # UTC a local, y se clampa al rango real de zonas horarias.
        try:
            off_min = int(tzOffset) if tzOffset is not None else 0
        except (TypeError, ValueError):
            off_min = 0
        off_min = max(-840, min(off_min, 840))

        rows = execute_sql_query(
            """
            SELECT (consumed_at - make_interval(mins => %s))::date AS dia,
                   COALESCE(SUM(calories), 0)      AS calories,
                   COALESCE(SUM(protein), 0)       AS protein,
                   COALESCE(SUM(carbs), 0)         AS carbs,
                   COALESCE(SUM(healthy_fats), 0)  AS healthy_fats,
                   COUNT(*)                        AS meals_count
            FROM consumed_meals
            WHERE user_id = %s
              AND consumed_at >= now() - make_interval(days => %s)
            GROUP BY 1
            ORDER BY 1 DESC
            """,
            (off_min, user_id, n_days),
            fetch_all=True,
        ) or []

        return {
            "days": [
                {
                    "date": str(r.get("dia")),
                    "calories": int(r.get("calories") or 0),
                    "protein": int(r.get("protein") or 0),
                    "carbs": int(r.get("carbs") or 0),
                    "healthy_fats": int(r.get("healthy_fats") or 0),
                    "meals_count": int(r.get("meals_count") or 0),
                }
                for r in rows
            ]
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/diary/consumed-range GET: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.delete("/consumed/{meal_id}")
def api_delete_consumed_meal(
    meal_id: str,
    # [P1-DIARY-EDITABLE · 2026-07-28] `_DELETE_CONSUMED_LIMITER` envuelve
    # `Depends(get_verified_user_id)` internamente y retorna el
    # `verified_user_id` autenticado — cero costo LLM, NO `verify_api_quota`
    # (ver CLAUDE.md "Historial-quota-exemption"). Mismo patrón que
    # `api_restock`/`api_consume_inventory` (routers/plans.py).
    verified_user_id: Optional[str] = Depends(_DELETE_CONSUMED_LIMITER),
):
    """Elimina una comida mal registrada del diario ("Deshacer registro").

    [P1-DIARY-EDITABLE · 2026-07-28] `consumed_meals` era append-only: un
    1.030 kcal mal loggeado (doble-tap, dedo equivocado) vivía para siempre
    hasta purgar la cuenta. El DELETE real vive en
    `db_facts.delete_consumed_meal`, filtrado `AND user_id = %s` (invariante
    I2) — acá solo se valida forma + auth y se traduce el resultado a
    HTTP: 404 si no había fila que borrar (no existía, o pertenecía a otro
    usuario — el mensaje es el mismo para no filtrar existencia cross-user),
    200 si se borró.
    """
    try:
        # [P1-AUDIT-3 · 2026-05-12] Rechaza UUIDs malformados con 400 antes de SQL.
        assert_valid_uuid(meal_id)

        if not verified_user_id:
            raise HTTPException(status_code=403, detail="Prohibido.")

        # [P1-CONSUMPTION-LEDGER · 2026-08-07] Devolver ANTES de borrar.
        #
        # El orden importa y no es simétrico. Si se borrara primero y el revert
        # fallara, la comida se perdería sin rastro visible: el usuario ve el
        # diario limpio y la Nevera baja, sin nada que explique la diferencia.
        # Al revés, si el revert funciona y el DELETE falla, el usuario ve la
        # comida todavía en su diario y reintenta — el revert es idempotente
        # (`reverted_at`), así que el segundo intento no vuelve a sumar y el
        # DELETE se completa. Entre dos fallos parciales gana el recuperable.
        #
        # `revert_consumption_events` ya filtra `AND user_id = %s`, así que un
        # `meal_id` ajeno no devuelve nada — no hace falta pre-verificar
        # propiedad para evitar tocar la Nevera de otro.
        import db_inventory
        _revert = db_inventory.revert_consumption_events(verified_user_id, meal_id)

        deleted = delete_consumed_meal(verified_user_id, meal_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail="Comida no encontrada (o ya fue eliminada).",
            )

        return {
            "success": True,
            "message": "Comida eliminada del diario.",
            # Lo devuelto viaja al cliente para que el toast pueda decirlo: si
            # la Nevera sube y nadie lo explica, parece un bug.
            "returned_to_pantry": _revert.get("reverted") or [],
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/diary/consumed DELETE: {str(e)}")
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

        result_box: dict[str, Optional[list]] = {"weight_history": None}

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

