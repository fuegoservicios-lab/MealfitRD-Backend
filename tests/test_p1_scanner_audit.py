"""[P1-PLAN-LOTE-53 · 2026-09-15] Auditoría de los escáneres (encargo nocturno del 15-sep).

Sin la clave de visión en local, lo medido es la capa determinista que va DESPUÉS del modelo
(`scripts/coach_battery/scanner_battery.py`). Tres defectos, dos de ellos vistos en producción:

1. `/api/inventory/photo-scan` (Nevera y paso 21 del formulario) llamaba a Gemini sin
   limitador y sin fila en `llm_usage_events`: gasto sin tope e invisible.
2. `_photo_scan_enabled()` decía True con el provider en «disabled» (el default de
   vision_agent) o sin modelo: la UI ofrecía un botón que siempre fallaba. En producción el
   provider SÍ está (journal: /api/diary/upload 200 los días 4-6 sep); era latente.
3. El nombre de reemplazo de P1-MEAL-NAME-BACKED en los tres escaneos reales: dos salían
   cortados a media frase («…espaguetis guisados con») y uno era un falso positivo
   («Sándwich vegetal» frente a lechuga, tomate y remolacha).
"""
from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest

from constants import derive_meal_name_from_description, meal_name_backed_by_description

_BACKEND = Path(__file__).resolve().parents[1]

# Descripciones copiadas de los logs de producción (04, 05 y 06 sep).
DESC_06 = ("Plato servido con arroz blanco, espaguetis guisados con salsa de tomate y aceitunas, "
           "carne de res guisada y plátano maduro frito.")
DESC_05 = ("Plato servido compuesto por papas asadas en gajos, porción de carne mechada coronada con "
           "huevo escalfado y salsa blanca.")
DESC_04 = ("Sándwich preparado en pan de molde integral relleno de lechuga fresca, rodajas de tomate, "
           "rodajas de remolacha cocida, pepino y queso.")


@pytest.mark.parametrize("desc", [DESC_06, DESC_05, DESC_04])
def test_el_nombre_derivado_no_arrastra_el_marco_ni_cuelga_de_un_conector(desc):
    nombre = derive_meal_name_from_description(desc)
    assert not nombre.lower().startswith("plato servido"), nombre
    ultima = nombre.split()[-1].lower().rstrip(",")
    assert ultima not in {"con", "de", "por", "en", "y", "relleno", "coronada"}, nombre
    assert meal_name_backed_by_description(nombre, desc), "el derivado debe pasar su propio guard"


def test_los_dos_casos_reales_quedan_legibles():
    assert derive_meal_name_from_description(DESC_06) == "Arroz blanco, espaguetis guisados con salsa de tomate"
    assert derive_meal_name_from_description(DESC_05) == "Papas asadas en gajos, porción de carne mechada"


def test_sandwich_vegetal_ya_no_es_falso_positivo():
    assert meal_name_backed_by_description("Sándwich vegetal en pan integral", DESC_04) is True


def test_el_guard_sigue_cazando_lo_inventado():
    # Los dos rechazos CORRECTOS de producción: el rótulo mete habichuelas que no están.
    assert meal_name_backed_by_description("Arroz con espaguetis, carne guisada y habichuelas", DESC_06) is False
    assert meal_name_backed_by_description("Papas con carne mechada, huevo y habichuelas", DESC_05) is False
    # Un descriptor no es un comodín para inventar un alimento.
    assert meal_name_backed_by_description("Sándwich vegetal de pollo", DESC_04) is False


def test_m9_kcal_absurdas_sin_macros_no_se_precargan():
    """Caso M9 exacto de la batería: 99.999 kcal y todo lo demás en 0 / basura."""
    import vision_agent as va
    out = va._coerce_meal_scan({"is_food": True, "photo_kind": "plato", "meal_name": "Pollo guisado",
                                "description": "Pollo guisado con arroz blanco.",
                                "calories": 99999, "protein": -5, "carbs": "mucho", "healthy_fats": None})
    assert out["calories"] == 0 and out.get("low_confidence") is True
    assert "Estimación" not in out["description"], "sin cifra fiable, el coach pide la porción"


@pytest.mark.parametrize("raw,esperadas", [
    ({"calories": 750, "protein": 40, "carbs": 70, "healthy_fats": 40}, 750),   # cuadra (800 ±35 %)
    ({"calories": 3000, "protein": 30, "carbs": 60, "healthy_fats": 20}, 540),  # no cuadra → macros
    ({"calories": 0, "protein": 20, "carbs": 50, "healthy_fats": 10}, 370),     # sin kcal → macros
    ({"calories": 2400, "protein": 0, "carbs": 0, "healthy_fats": 0}, 2400),    # sin macros, verosímil
    ({"calories": 4000, "protein": 150, "carbs": 400, "healthy_fats": 150}, 2500),  # tope por plato
])
def test_las_kcal_de_un_plato_cuadran_con_sus_macros(raw, esperadas):
    import vision_agent as va
    out = va._coerce_meal_scan({"is_food": True, "photo_kind": "plato", "meal_name": "",
                                "description": "Plato.", **raw})
    assert out["calories"] == esperadas


def test_el_tope_por_plato_baja_tambien_las_macros():
    """[P1-PLAN-LOTE-56] Gemini real, foto de dos pizzas: 175 P / 460 C / 70 G (~3.170 kcal) quedaba
    en 2.500 kcal con las macros intactas. Al topar, las macros cuadran con las kcal y se avisa."""
    import vision_agent as va
    out = va._coerce_meal_scan({"is_food": True, "photo_kind": "plato", "meal_name": "Dos pizzas de pepperoni",
                                "description": "Dos pizzas completas de pepperoni.",
                                "calories": 3170, "protein": 175, "carbs": 460, "healthy_fats": 70})
    assert out["calories"] == 2500 and out.get("low_confidence") is True
    kcal = 4 * out["protein"] + 4 * out["carbs"] + 9 * out["healthy_fats"]
    assert abs(kcal - 2500) <= 0.02 * 2500, (out, kcal)
    assert "confirma cuánto comiste" in out["description"]
    # Un plato por debajo del tope no se toca ni se marca.
    normal = va._coerce_meal_scan({"is_food": True, "photo_kind": "plato", "meal_name": "", "description": "Plato.",
                                   "calories": 840, "protein": 30, "carbs": 80, "healthy_fats": 48})
    assert normal["protein"] == 30 and "low_confidence" not in normal


def test_m10_peso_impreso_como_piezas_sigue_saneado():
    import vision_agent as va
    out = va._coerce_meal_scan({"is_food": True, "photo_kind": "items", "description": "Compra.",
                                "items": [{"name": "arroz", "quantity": 500, "unit": "paquete"},
                                          {"name": "huevos", "quantity": 30, "unit": "unidad"}]})
    assert {i["name"]: i["quantity"] for i in out["items"]} == {"arroz": 1.0, "huevos": 30.0}
    assert out["calories"] == 0


def _photo_scan_signature() -> str:
    src = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")
    i = src.index('@router.post("/inventory/photo-scan")')
    return src[i:src.index("):", i)]


def test_photo_scan_tiene_limitador():
    assert "Depends(_PHOTO_SCAN_LIMITER)" in _photo_scan_signature()
    import routers.user_data as ud
    assert ud._PHOTO_SCAN_LIMITER.max_calls == 10


def test_photo_scan_anota_el_coste_con_node_propio(monkeypatch):
    monkeypatch.setenv("MEALFIT_VISION_PROVIDER", "openai_compatible")
    monkeypatch.setenv("MEALFIT_VISION_MODEL", "gemini-3.8-flash")
    import db as db_facade
    import routers.user_data as ud
    import vision_agent as va

    async def _fake_analyze(image_bytes, prompt, schema):
        va._LAST_VISION_USAGE.set({"input_tokens": 1200, "output_tokens": 90})
        return ud._PantryScanResult(items=[])

    anotado = []
    monkeypatch.setattr(va, "analyze_image_structured", _fake_analyze)
    monkeypatch.setattr(db_facade, "log_llm_usage_event", lambda **kw: anotado.append(kw))
    monkeypatch.setattr(db_facade, "execute_sql_query", lambda *a, **k: [])
    body = {"image_b64": base64.b64encode(b"fake-jpeg").decode("ascii")}
    asyncio.run(ud.api_inventory_photo_scan(body=body, verified_user_id="user-123"))
    assert len(anotado) == 1
    fila = anotado[0]
    assert fila["node"] == "pantry_photo_scan" and fila["user_id"] == "user-123"
    assert fila["model"] == "gemini-3.8-flash"
    assert (fila["input_tokens"], fila["output_tokens"]) == (1200, 90)


def test_photo_scan_no_toca_el_libro_de_cuota():
    src = _photo_scan_signature()
    body = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")
    i = body.index('@router.post("/inventory/photo-scan")')
    assert "log_api_usage" not in body[i:i + 7000], "escanear la Nevera no puede quemar crédito de planes"
    assert src


@pytest.mark.parametrize("provider,model,esperado", [
    ("off", "gemini-3.8-flash", False),
    ("disabled", "gemini-3.8-flash", False),
    ("openai_compatible", "", False),
    ("openai_compatible", "gemini-3.8-flash", True),
])
def test_el_boton_solo_aparece_si_la_vision_existe(monkeypatch, provider, model, esperado):
    monkeypatch.setenv("MEALFIT_VISION_PROVIDER", provider)
    monkeypatch.setenv("MEALFIT_VISION_MODEL", model)
    from routers.plans import _photo_scan_enabled
    assert _photo_scan_enabled() is esperado
