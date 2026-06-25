"""[P1-SUPPLEMENT-COMPLETE · 2026-06-25] Magnesio/vit E/fibra ahora tienen plantilla de
suplemento. Antes el panel exhaustivo los marcaba 'bajo' pero `_SUPPLEMENT_TEMPLATES` no
tenía entrada → la tarjeta de suplemento caía en silencio (magnesio sobre todo, que SÍ se
suplementa de forma segura). Decisión del owner: cerrar los gaps que la comida entera no
alcanza con SUPLEMENTACIÓN, no forzando alimentos en el menú (eso lo vuelve monótono/caro
y aun así no llega a vit D). Este test ancla las 3 plantillas + que el builder las surface.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micronutrients import _SUPPLEMENT_TEMPLATES, build_supplement_recommendations

_REQUIRED_KEYS = ("nombre", "dosis", "alimentos", "precaucion")
_NEW = ("magnesium_mg", "vit_e_mg", "fiber_g")


def test_nuevas_plantillas_presentes_y_completas():
    for k in _NEW:
        assert k in _SUPPLEMENT_TEMPLATES, f"falta plantilla de suplemento: {k}"
        tpl = _SUPPLEMENT_TEMPLATES[k]
        for rk in _REQUIRED_KEYS:
            assert tpl.get(rk), f"{k}: falta o vacío el campo '{rk}'"


def test_builder_surface_magnesio_fibra_vite_cuando_bajos():
    report = {"gaps": [
        {"key": "magnesium_mg", "nutriente": "Magnesio", "valor": 406.0, "piso": 420.0,
         "unidad": "mg", "status": "bajo"},
        {"key": "fiber_g", "nutriente": "Fibra", "valor": 35.2, "piso": 38.0,
         "unidad": "g", "status": "bajo"},
        {"key": "vit_e_mg", "nutriente": "Vitamina E", "valor": 13.0, "piso": 15.0,
         "unidad": "mg", "status": "bajo"},
    ]}
    adv = build_supplement_recommendations(report, sex="M", age=20)
    keys = {i["key"] for i in adv["items"]}
    assert {"magnesium_mg", "fiber_g", "vit_e_mg"} <= keys
    assert adv["count"] == 3
    mag = next(i for i in adv["items"] if i["key"] == "magnesium_mg")
    assert "agnesio" in (mag.get("suplemento") or "") and mag.get("dosis_sugerida")


def test_estimado_bajo_tambien_surface():
    report = {"gaps": [
        {"key": "magnesium_mg", "nutriente": "Magnesio", "valor": 300.0, "piso": 420.0,
         "unidad": "mg", "status": "estimado_bajo"},
    ]}
    adv = build_supplement_recommendations(report, sex="M")
    assert adv["count"] == 1 and adv["items"][0]["key"] == "magnesium_mg"


def test_techos_nunca_generan_suplemento():
    # sodio/azúcar 'alto' (ceiling) NO son suplementables — el builder los ignora
    report = {"gaps": [
        {"key": "sodium_mg", "nutriente": "Sodio", "valor": 3000, "techo": 2000,
         "unidad": "mg", "status": "alto"},
        {"key": "free_sugars_g", "nutriente": "Azúcares", "valor": 40, "techo": 25,
         "unidad": "g", "status": "alto"},
    ]}
    adv = build_supplement_recommendations(report, sex="M")
    assert adv["count"] == 0
