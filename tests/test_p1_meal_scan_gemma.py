"""[P1-MEAL-SCAN-GEMMA · 2026-07-12] "Escanear comida" (Dashboard) conectado a gemma local.

Vivo (owner): el botón "Escanear comida" del card Progreso en Tiempo Real estaba
muerto de punta a punta mientras "Escanear mi nevera" funcionaba con el MISMO env:

  1. `vision_agent._vision_provider()` usaba `_env_str(choices={disabled,
     openai_compatible})` → el valor prod `MEALFIT_VISION_PROVIDER=ollama` caía
     al default `disabled` con WARNING (split-brain del mismo knob: user_data lo
     lee crudo y aceptaba "ollama").
  2. Aunque visión corriera, `/api/diary/upload` abortaba con 500 ANTES del
     análisis: el paso de object storage (`_storage_client = None` tras Neon)
     era obligatorio.
  3. `verify_api_quota` + `log_api_usage("llm_vision")` quemaban crédito del cap
     mensual por un análisis en la GPU del owner (costo LLM cero) — misma clase
     que P1-NEVERA-QUOTA-EXEMPT.

Fix: provider `ollama` en vision_agent (transporte espejo del escáner de Nevera,
mismos knobs), storage opcional en /upload, quota-exempt (RateLimiter se queda),
single-flight COMPARTIDO entre ambos escáneres (misma GPU), y `busy=True` para
distinguir "escáner ocupado" de "analizador caído".
tooltip-anchor: P1-MEAL-SCAN-GEMMA
"""
import asyncio
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "vision_agent.py"), encoding="utf-8") as f:
    _VA = f.read()
with open(os.path.join(_BACKEND, "routers", "diary.py"), encoding="utf-8") as f:
    _DIARY = f.read()
with open(os.path.join(_BACKEND, "routers", "user_data.py"), encoding="utf-8") as f:
    _UD = f.read()


# ---------------------------------------------------------------------------
# 1. Estructura: provider ollama existe y va por to_thread
# ---------------------------------------------------------------------------

def test_ollama_provider_branch_present():
    assert '_VISION_PROVIDER_OLLAMA = "ollama"' in _VA
    assert "asyncio.to_thread(_ollama_meal_scan" in _VA, \
        "el roundtrip httpx síncrono debe salir del event loop (workers=1)"
    # El knob acepta ollama/off — sin esto _env_str degrada 'ollama' a disabled.
    # OJO: anclar en el DEF (el string 'choices={' aparece antes en un comment).
    i = _VA.find("def _vision_provider")
    assert i != -1
    win = _VA[i:i + 500]
    for choice in ("_VISION_PROVIDER_OLLAMA", "_VISION_PROVIDER_OFF"):
        assert choice in win, f"{choice} falta en los choices del knob"


def test_shared_single_flight_lock():
    """Ambos escáneres (comida + nevera) golpean la MISMA GPU local: el lock
    debe ser uno solo, viviendo en vision_agent e importado por user_data."""
    assert "def get_vision_single_flight_lock" in _VA
    assert "from vision_agent import get_vision_single_flight_lock" in _UD, \
        "el escáner de Nevera debe compartir el lock de vision_agent"
    assert "_VISION_SCAN_LOCK" not in _UD, \
        "el lock lazy privado de user_data debía desaparecer (quedaría GPU-race)"


def test_ollama_timeout_mirrors_pantry_knob():
    """gemma local tarda 30-120s/foto: el branch ollama usa MEALFIT_VISION_TIMEOUT_S
    (240s default, como el escáner de Nevera), NO el clamp de 30s del knob cloud."""
    i = _VA.find("def _ollama_timeout_s")
    assert i != -1
    win = _VA[i:i + 500]
    assert "MEALFIT_VISION_TIMEOUT_S" in win
    assert '"240"' in win and "min(600, max(30, v))" in win


# ---------------------------------------------------------------------------
# 2. Funcional: gating por env + coerción pura
# ---------------------------------------------------------------------------

def test_is_vision_enabled_and_local_by_env(monkeypatch):
    import vision_agent as va
    monkeypatch.setenv("MEALFIT_VISION_PROVIDER", "ollama")
    assert va.is_vision_enabled() is True
    assert va.is_vision_local() is True
    monkeypatch.setenv("MEALFIT_VISION_PROVIDER", "disabled")
    assert va.is_vision_enabled() is False
    assert va.is_vision_local() is False
    # openai_compatible sin model/base_url sigue OFF (contrato previo intacto)
    monkeypatch.setenv("MEALFIT_VISION_PROVIDER", "openai_compatible")
    monkeypatch.delenv("MEALFIT_VISION_MODEL", raising=False)
    monkeypatch.delenv("MEALFIT_VISION_BASE_URL", raising=False)
    assert va.is_vision_enabled() is False


def test_coerce_meal_scan_clamps_and_not_food_zeroes():
    from vision_agent import _coerce_meal_scan
    out = _coerce_meal_scan({
        "is_food": True, "meal_name": "Mangu con salami",
        "description": "Plato de mangu",
        "calories": 99999, "protein": -5, "carbs": "310.6", "healthy_fats": None,
    })
    assert out["is_food"] is True
    assert out["calories"] == 10000, "clamp espejo de ConsumedMealRequest"
    assert out["protein"] == 0, "negativo → 0"
    assert out["carbs"] == 311, "string numérica → int redondeado"
    assert out["healthy_fats"] == 0
    assert "Estimación" in out["description"]

    not_food = _coerce_meal_scan({
        "is_food": False, "meal_name": "algo", "description": "un carro",
        "calories": 500, "protein": 20, "carbs": 30, "healthy_fats": 10,
    })
    assert not_food["is_food"] is False
    assert not_food["meal_name"] == ""
    assert not_food["calories"] == 0 and not_food["protein"] == 0


def test_process_image_with_vision_ollama_success(monkeypatch):
    import vision_agent as va
    monkeypatch.setenv("MEALFIT_VISION_PROVIDER", "ollama")
    fake = {
        "description": "Plato de arroz con pollo (Estimación: ...)",
        "is_food": True, "meal_name": "Arroz con pollo",
        "calories": 650, "protein": 42, "carbs": 78, "healthy_fats": 18,
    }
    monkeypatch.setattr(va, "_ollama_meal_scan", lambda b64: fake)
    out = asyncio.run(va.process_image_with_vision(b"\xff\xd8\xfffake-jpeg"))
    assert out["is_food"] is True
    assert out["meal_name"] == "Arroz con pollo"
    assert out["calories"] == 650
    assert not out.get("analysis_failed")
    assert not out.get("busy")


def test_process_image_with_vision_ollama_busy(monkeypatch):
    """Con el lock tomado (otro scan en vuelo) responde busy=True al instante
    en vez de encolar minutos detrás de la GPU."""
    import vision_agent as va
    monkeypatch.setenv("MEALFIT_VISION_PROVIDER", "ollama")
    monkeypatch.setattr(
        va, "_ollama_meal_scan",
        lambda b64: pytest.fail("con lock tomado NO debe llamar a Ollama"),
    )
    lock = va.get_vision_single_flight_lock()
    assert lock.acquire(blocking=False), "precondición: lock libre"
    try:
        out = asyncio.run(va.process_image_with_vision(b"\xff\xd8\xfffake"))
    finally:
        lock.release()
    assert out.get("busy") is True
    assert out.get("analysis_failed") is True


def test_process_image_with_vision_ollama_error_soft_fails(monkeypatch):
    import vision_agent as va
    monkeypatch.setenv("MEALFIT_VISION_PROVIDER", "ollama")

    def _boom(_b64):
        raise RuntimeError("tunnel down")

    monkeypatch.setattr(va, "_ollama_meal_scan", _boom)
    out = asyncio.run(va.process_image_with_vision(b"\xff\xd8\xfffake"))
    assert out.get("analysis_failed") is True
    assert not out.get("busy")
    # El lock debe quedar LIBRE tras el error (finally release).
    lock = va.get_vision_single_flight_lock()
    assert lock.acquire(blocking=False), "lock filtrado tras excepción"
    lock.release()


# ---------------------------------------------------------------------------
# 3. /api/diary/upload: storage opcional + quota-exempt + busy passthrough
# ---------------------------------------------------------------------------

def _upload_body():
    i = _DIARY.find("async def api_diary_upload(")
    assert i != -1
    j = _DIARY.find("\ndef ", i)
    return _DIARY[i:j if j != -1 else len(_DIARY)]


def test_upload_storage_is_optional():
    body = _upload_body()
    assert "Error uploading image to cloud storage." not in body, \
        "el 500 pre-visión debía desaparecer (storage=None es permanente tras Neon)"
    assert "P1-MEAL-SCAN-GEMMA" in body
    # El análisis debe seguir ocurriendo aunque no haya storage.
    assert "process_image_with_vision" in body


def test_upload_quota_exempt_with_ratelimiter():
    body = _upload_body()
    assert "Depends(verify_api_quota)" not in body, \
        "gemma local = costo cero → el paywall 402 congelaba el scan (doctrina P1-NEVERA-QUOTA-EXEMPT)"
    assert "Depends(get_verified_user_id)" in body, "la auth se conserva (IDOR guards intactos)"
    assert "Depends(_VISION_UPLOAD_LIMITER)" in body, "quota-exempt exige RateLimiter anti-spam"


def test_upload_usage_log_gated_on_local_vision():
    body = _upload_body()
    assert "not is_vision_local()" in body, \
        "log_api_usage (cuenta al cap) solo con provider cloud pago"
    # El anchor histórico de P1-DIARY-UPLOAD-GUEST-IDOR se preserva literal.
    assert 'await asyncio.to_thread(log_api_usage, actual_user_id, "llm_vision")' in body


def test_upload_busy_passthrough():
    body = _upload_body()
    assert '"busy": vision_result.get("busy", False)' in body, \
        "el modal distingue 'escáner ocupado' (reintento en segundos) de 'caído'"


# ---------------------------------------------------------------------------
# 4. Frontend: reescalado client-side + manejo de busy
# ---------------------------------------------------------------------------

def test_frontend_downscales_and_handles_busy():
    root = os.path.dirname(_BACKEND)
    with open(os.path.join(root, "frontend", "src", "components", "dashboard",
                           "ScanMealModal.jsx"), encoding="utf-8") as f:
        modal = f.read()
    assert "_downscaleToJpegFile" in modal, \
        "sin reescalado, una foto 4000px tarda minutos por el túnel SSH"
    assert "data.busy" in modal, "manejo del estado 'escáner ocupado'"
    assert "P1-MEAL-SCAN-GEMMA" in modal
