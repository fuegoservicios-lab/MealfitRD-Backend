"""[P1-VISION-LUNA · 2026-07-28] Redimensionado defensivo de fotos antes de
enviarlas al proveedor de visión (gpt-5.6-luna vía OpenAI-compatible; el
provider LOCAL que existía junto a este, Ollama, fue eliminado en
P1-VISION-NO-LOCAL · 2026-07-28). Bytes → bytes, sin I/O de red, testeable
con imágenes sintéticas.

Por qué existe: una foto de móvil típica (3024x4032) cuesta 12.406 tokens de
entrada ($0,0130/scan) contra 1.374 tokens ($0,0020/scan) a 1024x1024 — 6,6x
más barato, medido contra la API real el 2026-07-28 (detalle:
backend/docs/vision_luna.md). Antes de este fix `vision_agent.py` no
redimensionaba nada y `/api/diary/upload` acepta hasta 20 MB, así que la
foto cruda se enviaba tal cual al proveedor metered.

Import de Pillow GUARDADO a nivel módulo: un VPS sin la dependencia (o sin
wheel para su arquitectura) degrada a fail-open — bytes originales sin tocar,
6,6x más caro en silencio — en vez de reventar el import de `vision_agent`.
"""
from __future__ import annotations

import io
import logging

from knobs import _env_int

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except Exception:  # pragma: no cover - depende de la instalación del host
    Image = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False

_DEFAULT_MAX_SIDE_PX = 1024
_JPEG_QUALITY = 85


def vision_max_side_px() -> int:
    """Knob `MEALFIT_VISION_MAX_SIDE_PX` — lado mayor (px) al que se
    redimensiona una foto antes de enviarla a un proveedor de visión.
    Default 1024, clamp [256, 4096] vía `_env_int(validator=...)`: fuera de
    rango cae al DEFAULT (no clampea al borde — así es como `_env_int`
    trata un `validator` que retorna False, ver knobs.py)."""
    return _env_int(
        "MEALFIT_VISION_MAX_SIDE_PX",
        _DEFAULT_MAX_SIDE_PX,
        validator=lambda v: 256 <= v <= 4096,
    )


def prepare_image_for_vision(raw: bytes, max_side: int | None = None) -> tuple[bytes, dict]:
    """Redimensiona `raw` para que su lado mayor sea `max_side` (o el knob
    `MEALFIT_VISION_MAX_SIDE_PX` si no se pasa ninguno), preservando aspecto,
    y re-encodea a JPEG calidad 85. Si el lado mayor YA es ≤ `max_side`,
    devuelve los bytes intactos — re-encodear algo que ya cabe solo pierde
    calidad sin ahorrar nada.

    Fail-open: ante cualquier error (bytes corruptos, formato no soportado,
    `raw` vacío/None, Pillow ausente) devuelve `raw` sin tocar +
    `skipped_reason` explicando por qué. Un escaneo de comida NUNCA debe
    morir porque el resize falló.

    Retorna `(bytes_a_enviar, telemetria)` con telemetría
    `{"original_bytes", "sent_bytes", "original_wh", "sent_wh", "resized",
    "skipped_reason"}` — el `logger.info` del resize exitoso permite auditar
    el ahorro real en producción.
    """
    original_bytes = len(raw) if isinstance(raw, (bytes, bytearray)) else 0
    info: dict = {
        "original_bytes": original_bytes,
        "sent_bytes": original_bytes,
        "original_wh": None,
        "sent_wh": None,
        "resized": False,
        "skipped_reason": None,
    }

    if not _PIL_AVAILABLE:
        info["skipped_reason"] = "pillow_no_disponible"
        return raw, info

    if not isinstance(raw, (bytes, bytearray)) or not raw:
        info["skipped_reason"] = "bytes_vacios_o_invalidos"
        return raw, info

    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.load()  # fuerza decode completo — detecta corrupción temprano
            w, h = img.size
            info["original_wh"] = (w, h)

            target = max_side if max_side is not None else vision_max_side_px()

            if max(w, h) <= target:
                # Ya cabe: re-encodear solo perdería calidad sin ahorrar nada.
                info["sent_wh"] = (w, h)
                return raw, info

            scale = target / float(max(w, h))
            new_w = max(1, round(w * scale))
            new_h = max(1, round(h * scale))

            # Convertir a RGB ANTES de guardar JPEG — un PNG con canal alpha
            # (RGBA/P con transparencia) revienta el encoder JPEG con OSError.
            resized_img = img.convert("RGB").resize(
                (new_w, new_h), Image.Resampling.LANCZOS
            )
            buf = io.BytesIO()
            resized_img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
            out = buf.getvalue()

        info["sent_bytes"] = len(out)
        info["sent_wh"] = (new_w, new_h)
        info["resized"] = True
        logger.info(
            f"[P1-VISION-LUNA] Foto redimensionada {w}x{h} "
            f"({original_bytes} bytes) -> {new_w}x{new_h} ({len(out)} bytes), "
            f"max_side={target}."
        )
        return out, info
    except Exception as e:
        # Fail-open: bytes corruptos, formato exótico no soportado por Pillow,
        # etc. Nunca debe tumbar el escaneo — se manda la foto original.
        logger.warning(
            f"[P1-VISION-LUNA] prepare_image_for_vision falló, fail-open a "
            f"los bytes originales: {type(e).__name__}: {e}"
        )
        info["skipped_reason"] = f"error_procesando_imagen:{type(e).__name__}"
        info["sent_bytes"] = original_bytes
        info["sent_wh"] = None
        return raw, info
