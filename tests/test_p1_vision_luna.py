"""[P1-VISION-LUNA · 2026-07-28] Test ancla.

Doc: backend/docs/vision_luna.md
"""
import asyncio
import io
import logging
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from image_prep import prepare_image_for_vision  # noqa: E402

_BACKEND = os.path.join(os.path.dirname(__file__), "..")


def _src(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as fh:
        return fh.read()


def _png(w, h, mode="RGB"):
    from PIL import Image
    buf = io.BytesIO()
    Image.new(mode, (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def _wh(raw):
    from PIL import Image
    return Image.open(io.BytesIO(raw)).size


def test_foto_de_movil_se_reduce_al_cap():
    """3024x4032 son 12.406 tokens de entrada ($0,0130); a 1024 son 1.374
    ($0,0020). 6,6x, medido contra la API real el 2026-07-28."""
    raw = _png(3024, 4032)
    out, info = prepare_image_for_vision(raw, max_side=1024)
    assert max(_wh(out)) == 1024
    assert info["resized"] is True
    assert len(out) < len(raw)


def test_mantiene_el_aspecto():
    out, _ = prepare_image_for_vision(_png(4000, 2000), max_side=1024)
    w, h = _wh(out)
    assert (w, h) == (1024, 512), (w, h)


def test_imagen_pequena_pasa_intacta():
    """Re-encodear algo que ya cabe solo pierde calidad sin ahorrar nada."""
    raw = _png(800, 600)
    out, info = prepare_image_for_vision(raw, max_side=1024)
    assert out == raw
    assert info["resized"] is False


def test_png_con_alpha_no_revienta_el_encoder_jpeg():
    """RGBA -> JPEG lanza OSError si no se convierte a RGB antes."""
    out, info = prepare_image_for_vision(_png(2000, 2000, mode="RGBA"), max_side=1024)
    assert max(_wh(out)) == 1024
    assert info["resized"] is True


@pytest.mark.parametrize("basura", [b"", b"no soy una imagen", b"\x00" * 100, None])
def test_fail_open_devuelve_los_bytes_originales(basura):
    """Un escaneo NUNCA debe morir porque el resize fallo."""
    out, info = prepare_image_for_vision(basura, max_side=1024)
    assert out == basura
    assert info["resized"] is False
    assert info["skipped_reason"]


def test_knob_max_side_con_clamp():
    from image_prep import vision_max_side_px
    assert vision_max_side_px() == 1024
    os.environ["MEALFIT_VISION_MAX_SIDE_PX"] = "99999"
    try:
        assert vision_max_side_px() == 1024, "fuera de [256,4096] debe caer al default"
    finally:
        os.environ.pop("MEALFIT_VISION_MAX_SIDE_PX", None)


# ---------------------------------------------------------------------------
# Task 2 [P1-VISION-LUNA · 2026-07-28]: cablear el resize + cascada de
# fallback en vision_agent.py. Tests behaviorales (spy sobre los puntos de
# extensión reales del módulo) en vez de grep de substrings — si se invierte
# el comportamiento (orden, cascada, propagación del error), estos deben
# ponerse rojos.
# ---------------------------------------------------------------------------

def _openai_compatible_env(monkeypatch):
    """Config mínima para que `is_vision_enabled()` acepte el provider cloud."""
    monkeypatch.setenv("MEALFIT_VISION_PROVIDER", "openai_compatible")
    monkeypatch.setenv("MEALFIT_VISION_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("MEALFIT_VISION_BASE_URL", "https://example.test/v1")
    monkeypatch.delenv("MEALFIT_VISION_FALLBACK_PROVIDER", raising=False)


def test_resize_invocado_antes_del_despacho_a_cualquier_provider(monkeypatch):
    """Behavioral, NO source-parsing: instrumenta el punto de resize real
    (`prepare_image_for_vision`, importado a vision_agent) y el punto de
    despacho real (`_dispatch_vision_provider`) y verifica el ORDEN de
    invocación en tiempo de ejecución. Si alguien mueve el resize después
    del despacho (o lo condiciona por provider), esto se pone rojo."""
    import vision_agent as va
    monkeypatch.setenv("MEALFIT_VISION_PROVIDER", "ollama")

    calls = []

    def _spy_prepare(raw, max_side=None):
        calls.append("resize")
        return raw, {
            "original_bytes": len(raw) if raw else 0, "sent_bytes": 0,
            "original_wh": None, "sent_wh": None, "resized": False,
            "skipped_reason": None,
        }

    async def _spy_dispatch(provider, image_bytes):
        calls.append(f"dispatch:{provider}")
        return {"is_food": False, "description": "ok", "meal_name": "",
                "calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}

    monkeypatch.setattr(va, "prepare_image_for_vision", _spy_prepare)
    monkeypatch.setattr(va, "_dispatch_vision_provider", _spy_dispatch)

    asyncio.run(va.process_image_with_vision(b"\xff\xd8\xfffake-jpeg"))

    assert calls == ["resize", "dispatch:ollama"], (
        f"el resize debe ocurrir ANTES del despacho a CUALQUIER provider: {calls}"
    )


def test_cascada_no_dispara_sin_knob_configurado(monkeypatch):
    """Sin `MEALFIT_VISION_FALLBACK_PROVIDER`, un primario que falla NUNCA
    debe tocar el otro provider."""
    import vision_agent as va
    _openai_compatible_env(monkeypatch)

    calls = []

    async def _fail_openai(image_bytes):
        calls.append("openai_compatible")
        return {"analysis_failed": True, "description": "boom",
                "is_food": False, "meal_name": "", "calories": 0,
                "protein": 0, "carbs": 0, "healthy_fats": 0}

    def _fail_if_ollama(*a, **k):
        pytest.fail("sin knob configurado NO debe cascar a ollama")

    monkeypatch.setattr(va, "_dispatch_openai_compatible_vision", _fail_openai)
    monkeypatch.setattr(va, "_dispatch_ollama_vision", _fail_if_ollama)

    out = asyncio.run(va.process_image_with_vision(b"\xff\xd8\xfffake"))

    assert calls == ["openai_compatible"]
    assert out.get("analysis_failed") is True


def test_cascada_dispara_exactamente_una_vez_con_knob(monkeypatch):
    """Con el knob puesto, un primario que falla SÍ reintenta con el
    fallback — pero solo UNA vez (sin loop de vuelta al primario)."""
    import vision_agent as va
    _openai_compatible_env(monkeypatch)
    monkeypatch.setenv("MEALFIT_VISION_FALLBACK_PROVIDER", "ollama")

    calls = []

    async def _fail_openai(image_bytes):
        calls.append("openai_compatible")
        return {"analysis_failed": True, "description": "boom",
                "is_food": False, "meal_name": "", "calories": 0,
                "protein": 0, "carbs": 0, "healthy_fats": 0}

    async def _ok_ollama(image_bytes):
        calls.append("ollama")
        return {"is_food": True, "description": "plato ok",
                "meal_name": "Arroz con pollo", "calories": 500,
                "protein": 30, "carbs": 60, "healthy_fats": 10}

    monkeypatch.setattr(va, "_dispatch_openai_compatible_vision", _fail_openai)
    monkeypatch.setattr(va, "_dispatch_ollama_vision", _ok_ollama)

    out = asyncio.run(va.process_image_with_vision(b"\xff\xd8\xfffake"))

    assert calls == ["openai_compatible", "ollama"], (
        f"exactamente UN intento de fallback esperado, sin loops: {calls}"
    )
    assert out["is_food"] is True
    assert not out.get("analysis_failed")


def test_primario_falla_sin_fallback_propaga_el_resultado_original(monkeypatch):
    """Sin fallback configurado, el resultado devuelto debe ser EXACTAMENTE
    el del primario (mismo payload) — no una versión genérica ni mezclada."""
    import vision_agent as va
    _openai_compatible_env(monkeypatch)

    primary_failure = {
        "description": "Error analizando imagen (marca-primario-original).",
        "is_food": False, "analysis_failed": True, "meal_name": "",
        "calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0,
    }

    async def _fail_openai(image_bytes):
        return dict(primary_failure)

    def _fail_if_ollama(*a, **k):
        pytest.fail("sin fallback configurado NUNCA debe tocar ollama")

    monkeypatch.setattr(va, "_dispatch_openai_compatible_vision", _fail_openai)
    monkeypatch.setattr(va, "_dispatch_ollama_vision", _fail_if_ollama)

    out = asyncio.run(va.process_image_with_vision(b"\xff\xd8\xfffake"))

    assert out == primary_failure


def test_cascada_ambos_fallan_devuelve_el_error_del_primario_no_del_fallback(monkeypatch):
    """Si el fallback TAMBIÉN falla, el error que se devuelve (y por tanto
    lo que ve el usuario/log downstream) debe ser el del PRIMARIO, no el
    del fallback — para que apunte al problema real."""
    import vision_agent as va
    _openai_compatible_env(monkeypatch)
    monkeypatch.setenv("MEALFIT_VISION_FALLBACK_PROVIDER", "ollama")

    calls = []
    primary_failure = {
        "description": "primario boom", "is_food": False,
        "analysis_failed": True, "meal_name": "", "calories": 0,
        "protein": 0, "carbs": 0, "healthy_fats": 0,
    }
    fallback_failure = {
        "description": "fallback boom TAMBIEN", "is_food": False,
        "analysis_failed": True, "meal_name": "", "calories": 0,
        "protein": 0, "carbs": 0, "healthy_fats": 0,
    }

    async def _fail_openai(image_bytes):
        calls.append("openai_compatible")
        return dict(primary_failure)

    async def _fail_ollama(image_bytes):
        calls.append("ollama")
        return dict(fallback_failure)

    monkeypatch.setattr(va, "_dispatch_openai_compatible_vision", _fail_openai)
    monkeypatch.setattr(va, "_dispatch_ollama_vision", _fail_ollama)

    out = asyncio.run(va.process_image_with_vision(b"\xff\xd8\xfffake"))

    assert calls == ["openai_compatible", "ollama"]
    assert out == primary_failure, "debe devolver el error del PRIMARIO, no el del fallback"


@pytest.mark.parametrize("fb_value", [
    "disabled", "off", "valor-basura-invalido", "openai_compatible",
])
def test_fallback_invalido_apagado_o_igual_al_primario_es_no_cascada(monkeypatch, fb_value):
    """`disabled`/`off` (apagados a propósito), un valor fuera del choices-set
    (cae al default "" vía `_env_str`, con WARNING) y un fallback IGUAL al
    primario (evita el loop primario→sí-mismo) deben comportarse los tres
    como 'sin cascada' — nunca crashear, nunca reintentar."""
    import vision_agent as va
    _openai_compatible_env(monkeypatch)
    monkeypatch.setenv("MEALFIT_VISION_FALLBACK_PROVIDER", fb_value)

    calls = []

    async def _fail_openai(image_bytes):
        calls.append("openai_compatible")
        return {"analysis_failed": True, "description": "boom",
                "is_food": False, "meal_name": "", "calories": 0,
                "protein": 0, "carbs": 0, "healthy_fats": 0}

    monkeypatch.setattr(va, "_dispatch_openai_compatible_vision", _fail_openai)

    out = asyncio.run(va.process_image_with_vision(b"\xff\xd8\xfffake"))

    assert calls == ["openai_compatible"], (
        f"MEALFIT_VISION_FALLBACK_PROVIDER={fb_value!r} debe comportarse como "
        f"sin cascada (un único intento): {calls}"
    )
    assert out.get("analysis_failed") is True


def test_telemetria_de_resize_se_loguea_a_info(monkeypatch, caplog):
    """La auditoría de ahorro en prod depende de esta línea existiendo y
    siendo greppable — verifica que el marker + los campos de telemetría
    llegan al logger a nivel INFO."""
    import vision_agent as va
    monkeypatch.setenv("MEALFIT_VISION_PROVIDER", "ollama")

    async def _ok_ollama(image_bytes):
        return {"is_food": False, "description": "ok", "meal_name": "",
                "calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}

    monkeypatch.setattr(va, "_dispatch_ollama_vision", _ok_ollama)

    caplog.set_level(logging.INFO, logger="vision_agent")
    asyncio.run(va.process_image_with_vision(_png(64, 64)))

    telemetry = [
        r.message for r in caplog.records
        if r.name == "vision_agent" and "[P1-VISION-LUNA] resize" in r.message
    ]
    assert telemetry, "falta el log de telemetría del resize (marker greppable)"
    line = telemetry[0]
    for key in ("original_bytes=", "sent_bytes=", "original_wh=",
                "sent_wh=", "resized=", "skipped_reason="):
        assert key in line, f"telemetría incompleta, falta {key!r} en: {line}"
