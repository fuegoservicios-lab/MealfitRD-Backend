# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-626 · 2026-09-27] Las dudas de la foto se leen en el idioma del usuario.

Auditoría de idiomas (27-sep): el escáner traducía el plato y sus ingredientes (lote 225) pero las dudas —«¿Cuántos
huevos son?» con sus botones «2 huevos · 3 huevos · 4 huevos»— salían siempre en español, y la opción que cambia el
plato («Arepa») pisaba el nombre ya traducido. La frontera es la de siempre: lo que el motor usa como IDENTIFICADOR
no se toca. `sobre` casa con el ingrediente, `texto` puede renombrarlo y `nombre_plato` es lo que se guarda si la app
no trae el traducido: se quedan en español y la traducción viaja al lado (`texto_mostrar`, `nombre_plato_mostrar`).
La pregunta es solo para leer: se traduce en su sitio. Si la traducción falla, todo sale como antes.

tooltip-anchor: P1-PLAN-LOTE-626
"""
import asyncio
import copy
import inspect

import pytest

DUDAS = [{
    "sobre": "Huevo revuelto",
    "pregunta": "¿Cuántos huevos son?",
    "opciones": [
        {"texto": "2 huevos", "supuesta": False, "ajuste": {"calories": -70}},
        {"texto": "3 huevos", "supuesta": True, "ajuste": {}},
        {"texto": "Arepa", "supuesta": False, "nombre_plato": "Arepa con queso", "ajuste": {"calories": 90}},
    ],
}]


def _falso(llamadas, respuesta="traduce"):
    async def traducir(textos, locale, **kw):
        llamadas.append((list(textos), locale, kw))
        if respuesta is None:
            return None
        return [f"EN {t}" for t in textos]
    return traducir


def test_la_pregunta_se_traduce_y_los_identificadores_se_quedan(monkeypatch):
    import traduccion_para_mostrar as tpm
    from routers import diary
    llamadas = []
    monkeypatch.setattr(tpm, "traducir_para_mostrar", _falso(llamadas))
    out = asyncio.run(diary._dudas_para_mostrar(copy.deepcopy(DUDAS), "en-US", None))
    d = out[0]
    assert d["pregunta"] == "EN ¿Cuántos huevos son?"
    assert d["sobre"] == "Huevo revuelto"
    assert [o["texto"] for o in d["opciones"]] == ["2 huevos", "3 huevos", "Arepa"]
    assert [o.get("texto_mostrar") for o in d["opciones"]] == ["EN 2 huevos", "EN 3 huevos", "EN Arepa"]
    assert d["opciones"][2]["nombre_plato"] == "Arepa con queso"
    assert d["opciones"][2]["nombre_plato_mostrar"] == "EN Arepa con queso"
    # una sola llamada, con la plantilla de frases (no la de nombres de platos)
    assert len(llamadas) == 1 and llamadas[0][2].get("tipo") == "textos"


def test_en_espanol_no_se_llama_al_modelo(monkeypatch):
    import traduccion_para_mostrar as tpm
    from routers import diary
    llamadas = []
    monkeypatch.setattr(tpm, "traducir_para_mostrar", _falso(llamadas))
    assert asyncio.run(diary._dudas_para_mostrar(copy.deepcopy(DUDAS), "es-DO", None)) == DUDAS
    assert llamadas == []


def test_si_la_traduccion_falla_todo_sale_como_antes(monkeypatch):
    import traduccion_para_mostrar as tpm
    from routers import diary
    monkeypatch.setattr(tpm, "traducir_para_mostrar", _falso([], respuesta=None))
    assert asyncio.run(diary._dudas_para_mostrar(copy.deepcopy(DUDAS), "fr-FR", None)) == DUDAS


def test_sin_dudas_no_se_llama(monkeypatch):
    import traduccion_para_mostrar as tpm
    from routers import diary
    llamadas = []
    monkeypatch.setattr(tpm, "traducir_para_mostrar", _falso(llamadas))
    assert asyncio.run(diary._dudas_para_mostrar([], "fr-FR", None)) == []
    assert llamadas == []


def test_el_endpoint_de_la_foto_las_pasa_por_la_traduccion():
    from routers import diary
    src = inspect.getsource(diary)
    assert "_dudas_para_mostrar(" in src
    assert '"dudas": dudas_out' in src


def test_la_traduccion_asincrona_acepta_frases():
    import traduccion_para_mostrar as tpm
    assert "tipo" in inspect.signature(tpm.traducir_para_mostrar).parameters
