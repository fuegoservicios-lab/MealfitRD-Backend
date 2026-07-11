"""[P1-PANTRY-SCAN-V0 · 2026-07-11] Selector de envase + escáner de nevera por foto.

Feedback del owner (paso 21 en vivo): (1) "quiero las opciones para poder elegir otro
envase — no quiero una lata y quiero un paquete de habichuelas"; (2) "tampoco se
visualiza el botón para subir foto de alimentos".

Contrato:
1. PATCH /api/inventory/items/{id}/unit — cambio de envase ATÓMICO en un statement
   (CTEs): si ya existe row del usuario con nombre+unidad destino (UNIQUE), mergea
   cantidades y borra el origen; si no, actualiza in place. I2 en cada rama.
2. POST /api/inventory/photo-scan — vision provider via knob MEALFIT_VISION_PROVIDER
   (default OFF → 503; el frontend oculta el botón vía photo_scan_enabled del
   pre-flight). READ-ONLY: el scan NUNCA escribe user_inventory — el usuario confirma
   en el checklist y los adds van por POST /inventory/items. Single-flight (409 si
   hay otro scan en vuelo — el modelo local no soporta concurrencia) + cap de tamaño.
3. QPantryBuilder: <select> de envase por fila, botón de escaneo gateado por el flag,
   reescala client-side antes de subir, checklist con preselección solo de items
   confiables Y mapeados al catálogo.

tooltip-anchor: P1-PANTRY-SCAN-V0
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend" / "src"

_UD_SRC = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")
_PLANS_SRC = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_QPB_SRC = (_FRONT / "components" / "assessment" / "questions" / "QPantryBuilder.jsx").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Cambio de envase atómico
# ---------------------------------------------------------------------------

def _unit_endpoint_body():
    i = _UD_SRC.find('@router.patch("/inventory/items/{item_id}/unit")')
    assert i > 0, "endpoint de cambio de envase desapareció"
    return _UD_SRC[i:i + 4000]


def test_unit_change_is_single_atomic_statement():
    body = _unit_endpoint_body()
    assert "WITH src AS" in body and "merged AS" in body and "removed AS" in body, (
        "el cambio de envase debe ser UN statement con CTEs — dos queries separadas "
        "abren race de lost-update contra otra pestaña"
    )
    assert "NOT EXISTS (SELECT 1 FROM dup)" in body, (
        "la rama in-place debe excluirse cuando hay duplicado (el UNIQUE user+nombre+"
        "unidad explotaría el UPDATE)"
    )


def test_unit_change_filters_user_id_everywhere():
    body = _unit_endpoint_body()
    assert body.count("user_id = %s") >= 2, "I2: src y switched deben filtrar user_id"
    assert "HTTPException(status_code=404" in body, "row ajeno/inexistente → 404"


# ---------------------------------------------------------------------------
# 2. Photo-scan: knob, fail-secure, read-only, single-flight
# ---------------------------------------------------------------------------

def _scan_endpoint_body():
    i = _UD_SRC.find('@router.post("/inventory/photo-scan")')
    assert i > 0, "endpoint del escáner desapareció"
    return _UD_SRC[i:i + 6000]


def test_scan_provider_defaults_off():
    m = re.search(r'def vision_scan_provider\(\).*?or "off"', _UD_SRC, re.DOTALL)
    assert m, "MEALFIT_VISION_PROVIDER debe defaultear a 'off' (sin provider = feature apagado)"
    body = _scan_endpoint_body()
    assert 'if provider == "off":' in body and "503" in body


def test_scan_is_read_only():
    body = _scan_endpoint_body()
    for verb in ("INSERT INTO user_inventory", "UPDATE user_inventory", "DELETE FROM user_inventory"):
        assert verb not in body, (
            f"el scan NUNCA escribe inventario ({verb}) — la escritura pasa por el "
            "checklist de confirmación del usuario (POST /inventory/items)"
        )


def test_scan_single_flight_and_size_cap():
    body = _scan_endpoint_body()
    assert "acquire(blocking=False)" in body and "409" in body, (
        "segundo scan simultáneo → 409 (el modelo local de 4GB VRAM no soporta "
        "concurrencia; encolar sería esperar minutos a ciegas)"
    )
    assert "8_000_000" in body and "422" in body, "cap de payload (~6MB imagen)"


def test_scan_ollama_thinking_disabled():
    assert '"think": False' in _UD_SRC, (
        "gemma4 vía Ollama: thinking ON por default deja content VACÍO — el body "
        "debe mandar think=false (lección 2026-07-04)"
    )


def test_feasibility_exposes_scan_flag():
    assert '"photo_scan_enabled": _photo_scan_enabled(),' in _PLANS_SRC
    assert "from routers.user_data import vision_scan_provider" in _PLANS_SRC


# ---------------------------------------------------------------------------
# 3. Frontend: selector + botón + checklist
# ---------------------------------------------------------------------------

def test_frontend_unit_selector_per_row():
    assert "changeUnit(item, e.target.value)" in _QPB_SRC, "selector de envase por fila desapareció"
    assert "`/api/inventory/items/${item.id}/unit`" in _QPB_SRC
    assert "UNIT_OPTIONS" in _QPB_SRC


def test_frontend_scan_button_gated_by_flag():
    assert "feas?.photo_scan_enabled && (" in _QPB_SRC, (
        "el botón de escaneo debe ocultarse sin provider (mostrar un botón que "
        "responde 503 frustra)"
    )
    assert 'capture="environment"' in _QPB_SRC, "móvil debe abrir la cámara trasera"


def test_frontend_downscales_before_upload():
    assert "_downscaleToB64" in _QPB_SRC and "maxSide = 1024" in _QPB_SRC, (
        "sin reescala client-side, una foto de celular de 4000px viaja completa "
        "(payload + tokens de imagen del modelo)"
    )


def test_frontend_confirm_before_add():
    assert "it.selected && it.master_ingredient_id" in _QPB_SRC, (
        "solo entra lo que el usuario marcó Y mapea al catálogo — el scan no "
        "escribe nada por sí solo"
    )
    assert "(it.confidence ?? 0) >= 0.5" in _QPB_SRC, (
        "preselección solo de detecciones confiables (≥0.5) con match"
    )


def test_marker_anchored_in_source():
    assert _UD_SRC.count("P1-PANTRY-SCAN-V0") >= 2
    assert _PLANS_SRC.count("P1-PANTRY-SCAN-V0") >= 1
    assert _QPB_SRC.count("P1-PANTRY-SCAN-V0") >= 3
