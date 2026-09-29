# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-912 · 2026-09-29] «½ de pimiento morrón» en la LISTA → «½ pimiento morrón».

El 429 lo arregló en los PASOS («corta ½ de cebolla») y su comentario daba la lista por arreglada; no lo está: batería real
rdv868 (mujer que pierde grasa), desayuno del día 1, lista «½ de pimiento morrón». Corpus del VPS (5.336 comidas): 21 listas
con «½ de chile poblano», «½ de puerro en rodajas finas», «½ de pimentón»… «½» se lee «media»: «media de pimiento» no es
español; «¼ de cebolla» («un cuarto de») sí, y se queda.
"""
from __future__ import annotations

import pathlib

import lista_sin_de as lsd
import pulido_lineas as pl

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_medio_de_un_alimento_pierde_el_de():
    assert lsd.quitar("½ de pimiento morrón") == "½ pimiento morrón"
    assert lsd.quitar("½ de chile poblano") == "½ chile poblano"
    assert lsd.quitar("½ de puerro en rodajas finas") == "½ puerro en rodajas finas"
    assert lsd.quitar("1½ de tomate") == "1½ tomate"


def test_el_entero_tambien():
    assert lsd.quitar("1 de cebolla") == "1 cebolla"
    assert lsd.quitar("2 de aguacate") == "2 aguacate"


def test_lo_que_se_queda():
    for linea in ("¼ de cebolla",                       # «un cuarto de cebolla»: español correcto
                  "¾ de taza de avena",                 # una medida, no un alimento
                  "½ de la cebolla",                    # con artículo es otra frase
                  "½ taza de avena", "150 g de pechuga de pollo", "Sal al gusto", "0.2 de piña (pulpa) (300g)", ""):
        assert lsd.quitar(linea) == linea, linea
    assert lsd.quitar(None) is None


def test_el_pulido_de_la_lista_lo_aplica():
    assert pl.pulir_linea("½ de pimiento morrón") == "½ pimiento morrón"
    assert pl.pulir_linea("½ de chile poblano") == "½ chile poblano"


def test_lo_que_el_pulido_tomaria_por_especia_se_queda():
    """Replay sobre las 78.830 líneas de lista del corpus: «½ de pimentón» (medio ají morrón) salía «½ pimentón» y, en la
    SEGUNDA pasada del pulido, «Pimentón al gusto»: la regla de especias sin unidad lo tomaba por el polvo."""
    assert lsd.quitar("½ de pimentón") == "½ de pimentón"
    una = pl.pulir_linea("½ de pimentón")
    assert "al gusto" not in una.lower()
    assert pl.pulir_linea(una) == una


def test_la_segunda_pasada_no_cambia_nada():
    for linea in ("½ de cebollita morada", "1 de cebolla", "1 de cebolla picada", "½ de chile poblano", "½ de puerro",
                  "½ de puerro en rodajas finas", "½ de pimiento morrón", "½ de pimentón"):
        una = pl.pulir_linea(linea)
        assert pl.pulir_linea(una) == una, linea


def test_con_el_knob_apagado_nada(monkeypatch):
    monkeypatch.setenv("MEALFIT_LIST_HALF_WITHOUT_DE", "false")
    assert lsd.quitar("½ de pimiento morrón") == "½ de pimiento morrón"
    assert pl.pulir_linea("½ de pimiento morrón") == "½ de pimiento morrón"


def test_ancla_en_el_pulido():
    src = (_BACKEND / "pulido_lineas.py").read_text(encoding="utf-8")
    assert '__import__("lista_sin_de").quitar(out)  # [P1-PLAN-LOTE-912]' in src
    assert src.index('__import__("lista_sin_de").quitar(out)') < src.index("return _mayuscula(out)")
    assert "tooltip-anchor: P1-PLAN-LOTE-912" in (_BACKEND / "lista_sin_de.py").read_text(encoding="utf-8")
