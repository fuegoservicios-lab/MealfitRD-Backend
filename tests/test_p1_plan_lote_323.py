# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-323 · 2026-09-25] Con «Nada» de tiempo el corrector mira también los PASOS, no solo el `prep_time`.

Batería de cierre, perfil del dueño («Nada», 10 min por comida): «Arepitas de maíz salteadas al wok con filete de pescado
blanco» declara «10 min» y sus pasos sellan el pescado 4-5 min por lado, saltean 3 min y doran las arepitas 2 min por lado
en el mismo wok. El detector del lote 220 solo leía lo declarado y no avisaba al corrector."""
from __future__ import annotations

import pathlib

import graph_orchestrator as go
import tiempo_pasos as tp

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_WOK = ["Mise en place: amasa 135 g de harina de maíz precocida; corta el nabo y la cebolla; mide 260 g de filete de pescado.",
        "El Toque de Fuego: calienta el aceite en un wok; sella el filete de pescado 4-5 minutos por lado hasta 63 °C, retíralo "
        "y deja reposar 3 minutos. En el mismo wok saltea el nabo y la cebolla 3 minutos; aparta y dora las arepitas 2 minutos "
        "por lado.",
        "Montaje: coloca las arepitas como base y encima el pescado."]


def _dia(meal):
    return [{"day": 1, "meals": [dict(meal, meal="Cena")]}]


def test_declara_diez_pero_sus_pasos_suman_quince():
    assert tp.minutos_de_fuego(_WOK) == 15.0                       # 4×2 + 3 + 2×2; el reposo no cuenta
    out = go._detect_prep_time_issues(_dia({"name": "Arepitas salteadas al wok con pescado", "prep_time": "10 min",
                                            "recipe": list(_WOK)}), {"cookingTime": "none"})
    assert len(out) == 1, out
    assert "declara 10 min pero sus pasos suman ~15 min de fuego" in out[0], out[0]


def test_lo_que_de_verdad_cabe_no_se_toca_y_el_texto_de_siempre_sigue():
    rapido = ["El Toque de Fuego: coloca el plátano en el microondas 7-9 min. Mientras tanto, cocina el pescado 3-4 min "
              "por lado.", "Montaje: sirve."]
    assert tp.minutos_de_fuego(rapido) == 7.0                      # en paralelo: el mayor, no la suma
    assert go._detect_prep_time_issues(_dia({"name": "Pescado con plátano", "prep_time": "10 min", "recipe": rapido}),
                                       {"cookingTime": "none"}) == []
    guiso = {"name": "Guiso de lentejas con yuca hervida", "prep_time": "15 min", "recipe": ["Sirve."]}
    out = go._detect_prep_time_issues(_dia(guiso), {"cookingTime": "none"})
    assert out and "declara 15 min y el usuario NO TIENE TIEMPO" in out[0], out
    assert go._detect_prep_time_issues(_dia({"name": "x", "prep_time": "10 min", "recipe": list(_WOK)}),
                                       {"cookingTime": "30min"}) == [], "solo «Nada», como el lote 220"


def test_notas_reposo_y_rangos():
    assert tp.minutos_de_fuego(["⚠️ Seguridad: cocina 10 minutos más.", "Marina el pollo 30 minutos en la nevera.",
                                "Cocina 4-6 minutos."]) == 4.0
    assert tp.minutos_de_fuego(None) == 0.0 and tp.declara(None, 18) == "? min pero sus pasos suman ~18 min de fuego"


def test_el_punto_entre_cifras_es_un_decimal():
    # la frontera de oración del lote 52: «2.5 min por lado» no se parte en «2» y «5 min por lado»
    assert tp.minutos_de_fuego(["Cocina el pollo 2.5 minutos por lado. Sirve."]) == 5.0
    assert tp.minutos_de_fuego(["Hierve 10. Luego saltea 3 min; sirve."]) == 3.0


def test_ancla():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '_fuego = __import__("tiempo_pasos").minutos_de_fuego(m.get("recipe"))  # [P1-PLAN-LOTE-323]' in src
