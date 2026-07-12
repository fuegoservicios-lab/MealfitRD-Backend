"""[P1-CHAT-VISION-GEMMA · 2026-07-12] Visión gemma en el chat-agent, con clasificación.

Pedido del owner: el Agente debe combinar los dos poderes del escáner en una
sola foto adjunta — (a) PLATO servido → macros → log_consumed_meal, (b) ITEMS
sueltos/compra → lista → modify_pantry_inventory (Nevera). Una sola pasada de
gemma clasifica (`photo_kind`: plato|items|otro) y produce la salida del modo.

Diseño clave:
  - La description del modo items se construye DETERMINISTA desde los items
    sanitizados con el formato EXACTO que documenta modify_pantry_inventory
    ('2 unidades de Manzana') — el chat-agent copia, no reinterpreta.
  - El frontend elige la instrucción por modo con `photo_kind` (decisión de
    gemma, no del LLM del chat) e incluye rama honesta para analizador
    caído/ocupado (antes la description de error se inyectaba como análisis).
  - _sane_item_qty espejo de user_data (lección P1-PANTRY-SCAN-QTY: peso
    impreso leído como cantidad → colapsar envases discretos >12 a 1).
tooltip-anchor: P1-CHAT-VISION-GEMMA
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BACKEND)

from vision_agent import (  # noqa: E402
    _MEAL_VISION_PROMPT,
    _MEAL_VISION_SCHEMA,
    _coerce_meal_scan,
    _fmt_item_phrase,
    _sane_item_qty,
    _vision_disabled_payload,
)


# ---------------------------------------------------------------------------
# 1. Prompt y schema clasifican
# ---------------------------------------------------------------------------

def test_prompt_classifies_photo_kind():
    assert "photo_kind" in _MEAL_VISION_PROMPT
    assert "'plato' si es comida PREPARADA" in _MEAL_VISION_PROMPT
    assert "'items' si son alimentos SUELTOS" in _MEAL_VISION_PROMPT
    assert "NUMERO DE ENVASES O PIEZAS" in _MEAL_VISION_PROMPT, \
        "sin esto gemma pone el peso impreso del empaque como quantity"
    assert "unidad, lb, g, paquete, botella, lata, taza, funda" in _MEAL_VISION_PROMPT, \
        "whitelist de unidades espejo del escaner de Nevera"


def test_schema_has_kind_and_items():
    req = _MEAL_VISION_SCHEMA["required"]
    assert "photo_kind" in req and "items" in req
    assert _MEAL_VISION_SCHEMA["properties"]["photo_kind"]["enum"] == ["plato", "items", "otro"]
    item_req = _MEAL_VISION_SCHEMA["properties"]["items"]["items"]["required"]
    assert set(item_req) == {"name", "quantity", "unit"}


# ---------------------------------------------------------------------------
# 2. Coerción por modo
# ---------------------------------------------------------------------------

def test_items_mode_sanitizes_and_formats_for_pantry_tool():
    out = _coerce_meal_scan({
        "photo_kind": "items", "is_food": True,
        "description": "texto libre de gemma que NO se usa",
        "meal_name": "no aplica", "calories": 999, "protein": 9, "carbs": 9, "healthy_fats": 9,
        "items": [
            {"name": "Avena", "quantity": 500, "unit": "paquete"},   # peso impreso → 1
            {"name": "Manzana", "quantity": 2, "unit": "unidad"},
            {"name": "Pollo", "quantity": 1.5, "unit": "lb"},
            {"name": "", "quantity": 3, "unit": "lata"},              # sin nombre → fuera
        ],
    })
    assert out["photo_kind"] == "items"
    assert out["is_food"] is True
    assert out["meal_name"] == "" and out["calories"] == 0 and out["protein"] == 0, \
        "items NO es un plato: macros en 0 para no contaminar el diario"
    qtys = {i["name"]: i["quantity"] for i in out["items"]}
    assert qtys == {"Avena": 1.0, "Manzana": 2.0, "Pollo": 1.5}, \
        "sanitización espejo de P1-PANTRY-SCAN-QTY (envase discreto >12 → 1)"
    # Formato EXACTO de modify_pantry_inventory: '2 unidades de Manzana'.
    assert "1 paquete de Avena" in out["description"]
    assert "2 unidades de Manzana" in out["description"]
    assert "1.5 lb de Pollo" in out["description"]
    assert out["description"].startswith("Alimentos detectados"), \
        "la description es determinista desde los items sanitizados, no texto libre"


def test_items_mode_empty_degrades_to_otro():
    out = _coerce_meal_scan({"photo_kind": "items", "is_food": True, "items": []})
    assert out["photo_kind"] == "otro"
    assert out["is_food"] is False


def test_plato_mode_keeps_v3_contract():
    out = _coerce_meal_scan({
        "photo_kind": "plato", "is_food": True, "meal_name": "Los tres golpes",
        "description": "mangu, huevo, salami, queso",
        "calories": 750, "protein": 40, "carbs": 70, "healthy_fats": 40,
        "items": [],
    })
    assert out["photo_kind"] == "plato" and out["items"] == []
    assert out["calories"] == 750 and "Estimación" in out["description"]


def test_legacy_output_without_kind_maps_by_is_food():
    assert _coerce_meal_scan({"is_food": True, "calories": 100})["photo_kind"] == "plato"
    assert _coerce_meal_scan({"is_food": False})["photo_kind"] == "otro"


def test_fmt_phrase_pluralizes_containers_only():
    assert _fmt_item_phrase("Manzana", 2, "unidad") == "2 unidades de Manzana"
    assert _fmt_item_phrase("Avena", 1, "paquete") == "1 paquete de Avena"
    assert _fmt_item_phrase("Pollo", 1.5, "lb") == "1.5 lb de Pollo"


def test_sane_item_qty_mirrors_pantry_rules():
    assert _sane_item_qty(500, "paquete") == 1.0
    assert _sane_item_qty(30, "unidad") == 30.0
    assert _sane_item_qty(99, "unidad") == 30.0
    assert _sane_item_qty(0.1, "lb") == 0.25
    assert _sane_item_qty(None, "lata") == 1.0


def test_disabled_payload_has_stable_v4_shape():
    p = _vision_disabled_payload()
    assert p["photo_kind"] == "otro" and p["items"] == []


# ---------------------------------------------------------------------------
# 3. Passthrough y cableado frontend
# ---------------------------------------------------------------------------

def test_diary_upload_passes_kind_and_items():
    with open(os.path.join(_BACKEND, "routers", "diary.py"), encoding="utf-8") as f:
        diary = f.read()
    assert '"photo_kind": vision_result.get("photo_kind")' in diary
    assert '"items": vision_result.get("items") or []' in diary


def test_agentpage_mode_wiring():
    with open(os.path.join(_ROOT, "frontend", "src", "pages", "AgentPage.jsx"),
              encoding="utf-8") as f:
        ap = f.read()
    assert "Analizando tu foto" in ap, \
        "gemma tarda 30-90s: sin señal el usuario mira la nada durante el análisis"
    assert "visionKind === 'items'" in ap, "rama items → ofrecer Nevera"
    assert "visionKind === 'otro'" in ap, "rama sin comida → pedir otra toma"
    assert "modify_pantry_inventory" in ap and "items_to_add" in ap
    assert "NO tienes análisis de la imagen" in ap, \
        "rama honesta: antes la description de error se inyectaba como análisis real"
    assert "!visionFailed" in ap, "un analysis_failed no debe setear visionDescription"


def test_scanmealmodal_redirects_items_to_fridge():
    with open(os.path.join(_ROOT, "frontend", "src", "components", "dashboard",
                           "ScanMealModal.jsx"), encoding="utf-8") as f:
        modal = f.read()
    assert "data.photo_kind === 'items'" in modal, \
        "una compra escaneada como plato precargaria 0 kcal al diario"
