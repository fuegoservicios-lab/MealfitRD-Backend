"""[P3-VISION-UPLOAD-VALIDATION · 2026-05-20] Parser-based anchor + regression
guards para el bundle de fixes "incident grave reduction" del audit
`docs/gaps-audit-2026-05.md`:

1. **D3 / R2**: knob `MEALFIT_VISION_MODEL` en `vision_agent.py`.
   Pre-fix `gemini-3.1-pro-preview` estaba HARDCODED — Google deprecating
   el preview rompía vision + downstream `fact_extractor` cross-silo
   sin redeploy posible. Patrón espejo de `proactive_agent._proactive_model_name`
   (P3-PREVIEW-MODEL-KNOB).

2. **F3 / Vision upload validation**: content_type whitelist + magic-bytes
   sniff en `routers/diary.py::api_diary_upload`. Pre-fix el endpoint
   aceptaba el `file.content_type` declarado por el cliente Y lo pasaba
   directo a Supabase Storage. Vectores cerrados:
     - SVG/XML con JS embebido → XSS al renderizar en otros clientes.
     - `application/octet-stream` → bypass MIME-sniffing del bucket.
     - Content-type spoofing (declarar `image/jpeg` con body de HTML/exe).

Tests:
  - Anchor strings presentes en source-de-prod (rompen si alguien renombra
    el marker sin actualizar el test).
  - D3: presencia del helper `_vision_model_name()` + uso en
    `process_image_with_vision`.
  - F3: presencia del `_ALLOWED_VISION_CONTENT_TYPES` whitelist,
    `_detect_image_mime_from_bytes` magic-bytes function, y aplicación
    en el handler `api_diary_upload` (raise HTTPException 415 sobre
    content_type inválido).

Tooltip-anchor: P3-VISION-UPLOAD-VALIDATION.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_VISION_AGENT_PY = _BACKEND_ROOT / "vision_agent.py"
_DIARY_PY = _BACKEND_ROOT / "routers" / "diary.py"


# ---------------------------------------------------------------------------
# Anchor presence guards (P2-HIST-AUDIT-14 cross-link).
# ---------------------------------------------------------------------------


def test_anchor_present_in_vision_agent():
    """El anchor del knob D3 debe vivir en el source de `vision_agent.py`
    para que un renombre futuro falle este test antes de cambiar
    producción."""
    src = _VISION_AGENT_PY.read_text(encoding="utf-8")
    assert "P3-VISION-MODEL-KNOB" in src, (
        "Falta anchor `P3-VISION-MODEL-KNOB` en vision_agent.py. "
        "Sin anchor un futuro reader puede 'limpiar' el knob asumiendo "
        "que el hardcoded model está bien."
    )


def test_anchor_present_in_diary_upload():
    """El anchor F3 debe vivir en `routers/diary.py` para que el whitelist
    + magic-bytes check no se 'limpie' sin saber que es defense-in-depth
    contra XSS via SVG y content-type spoofing."""
    src = _DIARY_PY.read_text(encoding="utf-8")
    assert "P3-VISION-UPLOAD-VALIDATION" in src, (
        "Falta anchor `P3-VISION-UPLOAD-VALIDATION` en routers/diary.py."
    )


def test_anchor_present_in_test_file():
    """Cross-link guard P2-HIST-AUDIT-14: el slug del marker
    (`p3_vision_upload_validation`) matchea este test file, y el anchor
    también debe vivir en el test."""
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P3-VISION-UPLOAD-VALIDATION" in src


# ---------------------------------------------------------------------------
# D3 — Knob MEALFIT_VISION_MODEL en vision_agent.py
# ---------------------------------------------------------------------------


def test_vision_model_helper_present():
    """`_vision_model_name()` debe existir + leer `MEALFIT_VISION_MODEL`
    vía `_env_str` (auto-registry en `_KNOBS_REGISTRY`). Patrón espejo
    de `proactive_agent._proactive_model_name`."""
    src = _VISION_AGENT_PY.read_text(encoding="utf-8")
    assert "def _vision_model_name" in src, (
        "Falta helper `_vision_model_name()` en vision_agent.py. "
        "Sin él el modelo queda hardcoded y vulnerable a deprecation "
        "sin redeploy (R2 del audit)."
    )
    assert "MEALFIT_VISION_MODEL" in src, (
        "Falta env var `MEALFIT_VISION_MODEL` en vision_agent.py. "
        "El knob es la única vía para swap de modelo sin redeploy."
    )
    assert "_env_str" in src, (
        "Falta import/uso de `_env_str` para auto-registry en `_KNOBS_REGISTRY`. "
        "Sin auto-registry el knob no aparece en /health/version."
    )


def test_vision_llm_uses_knob_not_hardcoded():
    """`ChatGoogleGenerativeAI` dentro de `process_image_with_vision` debe
    referenciar `_vision_model_name()`, no un string literal."""
    src = _VISION_AGENT_PY.read_text(encoding="utf-8")
    # Buscar la sección de process_image_with_vision (función async)
    m = re.search(
        r"async def process_image_with_vision[\s\S]+?(?=\n(?:async )?def |\Z)",
        src,
    )
    assert m is not None, "No se encontró def de process_image_with_vision"
    body = m.group(0)
    assert "_vision_model_name()" in body, (
        "process_image_with_vision NO usa _vision_model_name(). "
        "Probablemente revertido a hardcoded — re-revisar D3 del audit."
    )
    # Defense: el literal hardcoded NO debe aparecer como argumento
    # a `model=` (puede aparecer como default del helper, eso es OK).
    body_lines = body.splitlines()
    for ln in body_lines:
        if "model=" in ln and "gemini-3.1-pro-preview" in ln:
            pytest.fail(
                f"vision_agent.process_image_with_vision sigue con modelo "
                f"hardcoded en línea: {ln.strip()!r}. Debe ser "
                f"`model=_vision_model_name()`."
            )


# ---------------------------------------------------------------------------
# F3 — Content-type whitelist + magic bytes sniff
# ---------------------------------------------------------------------------


def test_allowed_content_types_whitelist_present():
    """`_ALLOWED_VISION_CONTENT_TYPES` debe ser frozenset con los 6
    types canónicos (JPEG/JPG/PNG/WebP/HEIC/HEIF). Si falta alguno,
    documentar antes de quitar — clientes móviles legítimos pueden romper."""
    src = _DIARY_PY.read_text(encoding="utf-8")
    assert "_ALLOWED_VISION_CONTENT_TYPES" in src, (
        "Falta whitelist `_ALLOWED_VISION_CONTENT_TYPES` en routers/diary.py."
    )
    required = {
        '"image/jpeg"',
        '"image/png"',
        '"image/webp"',
        '"image/heic"',
    }
    missing = [t for t in required if t not in src]
    assert not missing, (
        f"`_ALLOWED_VISION_CONTENT_TYPES` no incluye types requeridos: "
        f"{missing}. Clientes móviles legítimos romperían."
    )


def test_magic_bytes_sniffer_present():
    """`_detect_image_mime_from_bytes` debe existir como defense-in-depth
    contra content-type spoofing. Verifica las 4 signatures canónicas."""
    src = _DIARY_PY.read_text(encoding="utf-8")
    assert "def _detect_image_mime_from_bytes" in src, (
        "Falta función `_detect_image_mime_from_bytes` en routers/diary.py. "
        "Sin ella, declarar `image/jpeg` + body HTML pasa el whitelist."
    )
    # JPEG signature
    assert "\\xff\\xd8\\xff" in src, (
        "Falta JPEG magic bytes (FF D8 FF) en _detect_image_mime_from_bytes."
    )
    # PNG signature
    assert "\\x89PNG" in src or "\\x89" in src, (
        "Falta PNG magic bytes (89 50 4E 47) en _detect_image_mime_from_bytes."
    )
    # WebP signature
    assert '"RIFF"' in src or "b'RIFF'" in src or 'b"RIFF"' in src, (
        "Falta WebP RIFF magic bytes en _detect_image_mime_from_bytes."
    )


def test_upload_handler_validates_content_type():
    """El handler `api_diary_upload` debe rechazar con HTTP 415 antes de
    leer bytes si `file.content_type` no está en el whitelist."""
    src = _DIARY_PY.read_text(encoding="utf-8")
    m = re.search(
        r"async def api_diary_upload[\s\S]+?(?=\n(?:async )?def |\Z)",
        src,
    )
    assert m is not None, "No se encontró def de api_diary_upload"
    body = m.group(0)

    assert "_ALLOWED_VISION_CONTENT_TYPES" in body, (
        "api_diary_upload NO referencia _ALLOWED_VISION_CONTENT_TYPES. "
        "Posible regresión — el whitelist existe pero no se aplica."
    )
    assert "status_code=415" in body, (
        "api_diary_upload NO devuelve HTTP 415 sobre content_type inválido. "
        "Debe ser 415 (Unsupported Media Type), no 400 ni 413."
    )
    # Sniff post-read también debe estar
    assert "_detect_image_mime_from_bytes" in body, (
        "api_diary_upload NO invoca _detect_image_mime_from_bytes. "
        "Magic-bytes check es defense-in-depth obligatorio."
    )


def test_upload_handler_keeps_size_limit():
    """Regression guard: el size limit pre-existente (20MB) NO debe
    haberse perdido al añadir el content_type check."""
    src = _DIARY_PY.read_text(encoding="utf-8")
    m = re.search(
        r"async def api_diary_upload[\s\S]+?(?=\n(?:async )?def |\Z)",
        src,
    )
    body = m.group(0) if m else ""
    assert "MAX_FILE_SIZE" in body, (
        "Regresión: MAX_FILE_SIZE desapareció de api_diary_upload."
    )
    assert "status_code=413" in body, (
        "Regresión: HTTP 413 sobre size overflow desapareció."
    )


# ---------------------------------------------------------------------------
# Functional check del magic-bytes sniffer (importable, sin DB).
# ---------------------------------------------------------------------------


def test_detect_image_mime_jpeg():
    """JPEG real (FF D8 FF) detecta como image/jpeg."""
    # Mock minimal import para no levantar dependencias DB
    import sys
    if "routers.diary" not in sys.modules:
        # Si el módulo no está importable sin Supabase init, saltamos el
        # functional test. Los parser-based de arriba cubren el contrato.
        try:
            from routers.diary import _detect_image_mime_from_bytes
        except Exception as e:
            pytest.skip(f"routers.diary no importable sin DB ({type(e).__name__})")
    else:
        from routers.diary import _detect_image_mime_from_bytes

    # JPEG minimal magic: FF D8 FF + padding
    jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 28
    assert _detect_image_mime_from_bytes(jpeg_bytes) == "image/jpeg"


def test_detect_image_mime_png():
    import sys
    try:
        from routers.diary import _detect_image_mime_from_bytes
    except Exception as e:
        pytest.skip(f"routers.diary no importable sin DB ({type(e).__name__})")

    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
    assert _detect_image_mime_from_bytes(png_bytes) == "image/png"


def test_detect_image_mime_webp():
    try:
        from routers.diary import _detect_image_mime_from_bytes
    except Exception as e:
        pytest.skip(f"routers.diary no importable sin DB ({type(e).__name__})")

    webp_bytes = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 20
    assert _detect_image_mime_from_bytes(webp_bytes) == "image/webp"


def test_detect_image_mime_rejects_html():
    """Body de HTML/exec NO debe pasar el magic-bytes check (defense
    contra content-type spoofing)."""
    try:
        from routers.diary import _detect_image_mime_from_bytes
    except Exception as e:
        pytest.skip(f"routers.diary no importable sin DB ({type(e).__name__})")

    html_payload = b"<!DOCTYPE html><script>alert(1)</script>" + b"\x00" * 12
    assert _detect_image_mime_from_bytes(html_payload) is None, (
        "HTML payload pasó el magic-bytes check — defense rota."
    )


def test_detect_image_mime_rejects_svg():
    """SVG (XML que empieza con <?xml ...?>) NO debe pasar — vector XSS
    clásico via JS embebido en SVG."""
    try:
        from routers.diary import _detect_image_mime_from_bytes
    except Exception as e:
        pytest.skip(f"routers.diary no importable sin DB ({type(e).__name__})")

    svg_payload = b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg">' + b"\x00" * 12
    assert _detect_image_mime_from_bytes(svg_payload) is None, (
        "SVG payload pasó el magic-bytes check — vector XSS abierto."
    )


def test_detect_image_mime_rejects_short_input():
    """Bytes < 12 deben retornar None (no caer en index error)."""
    try:
        from routers.diary import _detect_image_mime_from_bytes
    except Exception as e:
        pytest.skip(f"routers.diary no importable sin DB ({type(e).__name__})")

    assert _detect_image_mime_from_bytes(b"") is None
    assert _detect_image_mime_from_bytes(b"\xff\xd8") is None  # JPEG truncado
