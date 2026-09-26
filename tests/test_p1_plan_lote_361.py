# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-361 · 2026-09-26] «Otra…» se aplica como una opción más, sin volver a analizar la foto.

El dueño (captura del escáner): escribió «4 huevos» en «Otra…», tocó «Arepa» en la otra duda… y la foto se volvió a
analizar (el blur del campo la disparaba): el análisis nuevo trajo otras dudas, borró la «Arepa» y los 4 huevos no se
veían aplicados en ningún lado. Ahora la respuesta escrita se convierte en el AJUSTE de esa duda con una llamada de
TEXTO (flash) que ve el plato estimado y las opciones con sus ajustes como referencia («3 huevos» = 0, «4 huevos» =
+72): el resultado es una opción más, elegida y confirmada; las demás dudas no se tocan.
Tooltip-anchor: P1-PLAN-LOTE-361
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]

_REQ = {
    "plato": "Omelet con panecillo y queso",
    "macros": {"calories": 700, "protein": 44, "carbs": 40, "healthy_fats": 38},
    "pregunta": "¿De cuántos huevos hiciste la tortilla?",
    "opciones": [
        {"texto": "2 huevos", "supuesta": False, "ajuste": {"calories": -72, "protein": -6, "carbs": 0, "healthy_fats": -5}},
        {"texto": "3 huevos", "supuesta": True, "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}},
    ],
    "respuesta": "4 huevos",
}


def test_el_prompt_lleva_el_plato_las_opciones_y_la_respuesta_como_dato():
    import ajuste_de_duda as ad
    p = ad.mensaje_para_el_modelo(ad.PeticionAjuste(**_REQ))
    assert "Omelet con panecillo y queso" in p and "700" in p
    assert "3 huevos" in p and "+0" in p and "2 huevos" in p and "-72" in p
    assert '"4 huevos"' in p


def test_el_ajuste_se_acota_y_el_nombre_es_opcional():
    import ajuste_de_duda as ad
    r = ad.normalizar({"calories": 99999, "protein": -900, "carbs": "x", "healthy_fats": 5, "nombre_plato": "  "})
    assert r["ajuste"] == {"calories": 2000, "protein": -200.0, "carbs": 0.0, "healthy_fats": 5.0}
    assert "nombre_plato" not in r
    assert ad.normalizar({"calories": 72, "nombre_plato": "Omelet de 4 huevos"})["nombre_plato"] == "Omelet de 4 huevos"


def test_el_endpoint_devuelve_el_ajuste_y_falla_suave(monkeypatch):
    from routers import diary
    import ajuste_de_duda as ad

    async def bien(pet, idioma, uid):
        return {"calories": 72, "protein": 6, "carbs": 0, "healthy_fats": 5}

    monkeypatch.setattr(ad, "estimar_con_ia", bien)
    r = asyncio.run(diary.api_ajuste_de_duda(ad.PeticionAjuste(**_REQ), verified_user_id="u" * 8))
    assert r["texto"] == "4 huevos" and r["ajuste"]["calories"] == 72

    async def mal(pet, idioma, uid):
        raise TimeoutError("sin IA")

    monkeypatch.setattr(ad, "estimar_con_ia", mal)
    r = asyncio.run(diary.api_ajuste_de_duda(ad.PeticionAjuste(**_REQ), verified_user_id="u" * 8))
    assert r["operation_failed"] is True


def test_exento_de_cuota_con_limitador():
    src = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    i = src.index('@router.post("/scan/ajuste-duda")')
    assert "_AJUSTE_DUDA_LIMITER" in src[i:i + 300] and "verify_api_quota" not in src[i:i + 300]


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 361
