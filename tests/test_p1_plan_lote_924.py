# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-924 · 2026-09-30] La sustitución de la compra única se lleva ENTERA la frase de punto del ave que
escribe el 888, no sólo su temperatura.

Batería real del bloque 2 (rdb932, código vivo 932, owner_like día 8): el grafo entregó «cocina el pollo 6-7 minutos por
lado, hasta que el pollo alcance 74 °C por dentro y dore y esté cocido» (el 888 funde el «hasta que dore y esté cocido»
del modelo con su punto seguro). Desde el día 8 el pollo pasa a sardinas en lata y `_TEMP_SEGURA` quitaba «hasta que el
pollo alcance 74 °C» pero no «por dentro y dore y esté cocido»: el paso decía «calienta las sardinas 2-3 minutos por
dentro y dore y esté cocido».
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import sustitucion_fresca as sf  # noqa: E402

_PASO = ("El Toque de Fuego: calienta el aceite de oliva en sartén a fuego medio-alto y cocina el pollo 6-7 minutos por "
         "lado, hasta que el pollo alcance 74 °C por dentro y dore y esté cocido; calienta el casabe en la misma sartén "
         "1 min por lado.")


def _plato(paso=_PASO):
    return {"name": "Pollo a la plancha con casabe, queso blanco y aguacate",
            "ingredients": ["2 piezas de casabe", "20 g de queso blanco", "72 g de sardinas en lata"],
            "recipe": ["Mise en place: mide 72 g de sardinas en lata y 2 piezas de casabe.", paso,
                       "Montaje: sirve el pollo como plato principal con el casabe."]}


def test_la_frase_de_punto_del_ave_se_va_entera():
    m = _plato()
    sf.reescribir_plato(m, "¼ pechuga de pollo (≈83 g)", "72 g de sardinas en lata", "sardinas en lata")
    paso = m["recipe"][1]
    assert "por dentro" not in paso and "dore" not in paso and "cocido" not in paso, paso
    assert "calienta las sardinas 2-3 minutos; calienta el casabe en la misma sartén 1 min por lado." in paso, paso


def test_sin_sujeto_y_con_otra_clausula_detras():
    m = _plato("El Toque de Fuego: cocina el pollo 5 min por lado, hasta que alcance 74 °C por dentro y esté dorado, "
               "con el orégano por encima.")
    sf.reescribir_plato(m, "¼ pechuga de pollo (≈83 g)", "72 g de sardinas en lata", "sardinas en lata")
    paso = m["recipe"][1]
    assert "por dentro" not in paso and "dorado" not in paso, paso
    assert "con el orégano por encima." in paso, "lo que va detrás de la coma es otra cosa y se queda"


def test_la_cola_fundida_que_no_es_del_ave():
    """Replay sobre la cola viva (Bvivo921): «guisa 8-10 minutos más, hasta que el pollo alcance 74 °C por dentro y el
    guiso espese» y «…por dentro y el pollo esté humeante» (ya con las sardinas)."""
    assert sf._sin_punto_por_dentro("guisa 8-10 minutos más, hasta que el pollo alcance 74 °C por dentro y el guiso "
                                    "espese.") == "guisa 8-10 minutos más, hasta que el guiso espese."
    assert sf._sin_punto_por_dentro("saltea 8-10 minutos, hasta que alcance 74 °C por dentro y las sardinas esté "
                                    "humeante; sazona.") == "saltea 8-10 minutos; sazona."
    assert sf._sin_punto_por_dentro("cocina 8-10 minutos hasta que alcance 74 °C por dentro y retira del fuego.") == (
        "cocina 8-10 minutos y retira del fuego.")
    assert sf._sin_punto_por_dentro("cocina 10-12 minutos hasta 74 °C por dentro; desmenuza.") == (
        "cocina 10-12 minutos; desmenuza."), "embarazo809: «hasta 74 °C por dentro», sin «que»"


def test_knob_apagado_conducta_previa(monkeypatch):
    monkeypatch.setenv("MEALFIT_SUBST_POULTRY_DONENESS_TAIL", "false")
    m = _plato()
    sf.reescribir_plato(m, "¼ pechuga de pollo (≈83 g)", "72 g de sardinas en lata", "sardinas en lata")
    assert "por dentro y dore y esté cocido" in m["recipe"][1], m["recipe"][1]


def test_ancla():
    src = (_BACKEND / "sustitucion_fresca.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-PLAN-LOTE-924" in src
