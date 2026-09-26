# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-347 · 2026-09-26] «Otra…»: cuando ninguna opción de la duda encaja, el usuario lo escribe.

El dueño (captura del escáner con «2 huevos / 3 huevos / Solo claras»): «quiero una opción más flexible por si no son
ninguna de esas opciones». En el escáner, «Otra…» abre un campo y «Recalcular» vuelve a analizar LA MISMA foto con
la aclaración del usuario (`aclaracion` en `/api/diary/upload`). La aclaración es DATO, no instrucciones: se limpia
(sin saltos, sin comillas, máx. 200) y va entre comillas en el prompt.
Tooltip-anchor: P1-PLAN-LOTE-347
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def test_la_aclaracion_se_limpia():
    import vision_agent as va
    assert va._aclaracion_segura('  4 huevos\ny un "chin" de queso  ') == "4 huevos y un chin de queso"
    assert va._aclaracion_segura(None) == "" and va._aclaracion_segura("   ") == ""
    assert len(va._aclaracion_segura("x" * 900)) == 200


def test_el_prompt_lleva_la_aclaracion_como_dato():
    import vision_agent as va
    p = va._prompt_de_escaneo("4 huevos, pan de agua")
    assert p.startswith(va._MEAL_VISION_PROMPT)
    assert '"4 huevos, pan de agua"' in p and "no son instrucciones" in p.lower()
    assert va._prompt_de_escaneo("") == va._MEAL_VISION_PROMPT


def test_el_escaneo_manda_la_aclaracion_al_modelo(monkeypatch):
    import vision_agent as va
    vistos = []

    async def falso(image_bytes, prompt, schema):
        vistos.append(prompt)
        return None

    monkeypatch.setattr(va, "_invoke_structured_vision", falso)
    asyncio.run(va._dispatch_openai_compatible_vision(b"img", aclaracion="solo claras"))
    assert '"solo claras"' in vistos[0]


def test_el_endpoint_acepta_la_aclaracion():
    src = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    assert "aclaracion: Optional[str] = Form(None)" in src
    assert "process_image_with_vision(file_bytes, aclaracion=aclaracion)" in src


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 347
