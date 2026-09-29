# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-862 · 2026-09-29] Un batido siempre dice que se licúa lo que lleva.

Batería real MX de 6d (29-sep, D2): «Batido cremoso de guineo, manzana y leche de soya con queso cottage» sin ningún paso
que licuara el guineo, la manzana ni la leche — sólo «💪 Agrega queso cottage a la licuadora». Corpus: 7 batidos así.
"""
from __future__ import annotations

import pathlib

import batido_licua as bl
import graph_orchestrator as go
import tdf_sin_relleno as tr

_BACKEND = pathlib.Path(__file__).resolve().parents[1]

_BATIDO = {
    "name": "Batido cremoso de guineo, manzana y leche de soya con queso cottage",
    "ingredients": ["½ guineo mediano", "½ manzana mediana", "600 ml de leche de soya sin azúcar", "15 g de avena",
                    "45 g de queso cottage"],
    "recipe": ["Mise en place: pela ½ guineo y córtalo en trozos; lava y corta ½ manzana en cubos; mide 600 ml de leche de "
               "soya sin azúcar y 15 g de avena.",
               "💪 Agrega queso cottage a la licuadora y licúa hasta integrar.",
               "Montaje: sirve frío en un vaso alto; espolvorea una pizca extra de canela."],
}


def test_el_batido_sin_licuar_recibe_su_paso_antes_del_cerrador():
    m = {**_BATIDO, "recipe": list(_BATIDO["recipe"])}
    assert bl.asegurar(m)
    assert m["recipe"][1] == ("El Toque de Fuego: coloca los ingredientes del batido (menos lo que va por encima al "
                              "servir) en la licuadora y licúa a alta velocidad hasta obtener una mezcla homogénea.")
    assert m["recipe"][2].startswith("💪 Agrega queso cottage a la licuadora")
    assert not bl.asegurar(m), "idempotente"


def test_lo_que_ya_licua_o_no_es_batido_no_se_toca():
    ya = {"name": "Batido de lechosa", "recipe": ["Mise en place: corta la lechosa.",
                                                   "El Toque de Fuego: licúa la lechosa con la leche 1 minuto.",
                                                   "Montaje: sirve el batido."]}
    assert not bl.asegurar(ya)
    bowl = {"name": "Bowl cítrico de queso blanco fresco batido, mandarina y chía",
            "recipe": ["Mise en place: pela la mandarina.", "Montaje: sirve el queso en un bowl."]}
    assert not bl.asegurar(bowl), "«queso blanco fresco batido» es adjetivo (lote 734)"
    solo_montaje = {"name": "Licuado de guineo", "recipe": ["Mise en place: pela el guineo.",
                                                             "Montaje: sirve el licuado bien frío."]}
    assert bl.asegurar(solo_montaje), "«sirve el licuado» no licúa nada"
    assert "todos los ingredientes" in solo_montaje["recipe"][1]


def test_el_relleno_sale_pero_la_instruccion_se_queda():
    assert tr.resto("El Toque de Fuego: No requiere cocción.") is None
    assert tr.resto("El Toque de Fuego: No aplica (plato frío).") is None
    assert tr.resto("El Toque de Fuego: sin cocción; coloca el guineo, la manzana y la leche en la licuadora y licúa 1 "
                    "minuto.") == ("El Toque de Fuego: coloca el guineo, la manzana y la leche en la licuadora y licúa 1 "
                                   "minuto.")
    assert tr.resto("El Toque de Fuego: No requiere cocción. Licúa todos los ingredientes hasta que quede cremoso.") == (
        "El Toque de Fuego: licúa todos los ingredientes hasta que quede cremoso.")


def test_el_pase_de_platos_frios_conserva_la_instruccion():
    m = {"name": "Batido de guineo y avena",
         "recipe": ["Mise en place: pela el guineo y mide la avena y la leche.",
                    "El Toque de Fuego: sin cocción; coloca el guineo, la avena y la leche en la licuadora y licúa 1 minuto.",
                    "Montaje: sirve frío."]}
    go._inject_recipe_time_temp_defaults(m)
    assert any("licuadora y licúa" in p for p in m["recipe"]), m["recipe"]
    assert not any("sin cocción" in p for p in m["recipe"]), m["recipe"]
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("batido_licua").asegurar(meal)  # [P1-PLAN-LOTE-862]' in src
