# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-466 · 2026-09-27] En la compra única, el tomate que el plato COCINA pasa a salsa de tomate.

Replay forzado de los días 21+ (88 planes): 95 de 252 tomates sustituidos por zanahoria iban dentro de un guiso o una
salsa («sofríe la cebolla y la zanahoria… salsa»). «Salsa de tomate» es despensa y vive en el catálogo; ~2,5 veces más
concentrada que el fresco (150 g → 60 g). El tomate crudo de una ensalada sigue pasando a zanahoria. Y lo sustituido que
no viene cocinado (batata por plátano) se retoma con su pronombre: «hierve la batata…; escúrrela y májala».
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import compra_unica as cu  # noqa: E402
import graph_orchestrator as go  # noqa: E402
import sustitucion_fresca as sf  # noqa: E402

_REQ = {"need_days": 21, "allow_frozen": False, "freezer_mode": "none"}
_GUISO = ("Guiso criollo. Mise en place: pica 2 tomates y ½ cebolla. El Toque de Fuego: sofríe la cebolla y el tomate "
          "3 min, añade el agua y guisa 10 min.")


def test_el_tomate_del_guiso_pasa_a_salsa():
    r = cu.sustituir_linea("2 tomates", 20, _REQ, plato=_GUISO, gramos_de=lambda t: 300.0)
    assert r and r[0] == "120 g de salsa de tomate" and r[1] == "salsa de tomate", r
    r = cu.sustituir_linea("150 g de tomate", 20, _REQ, plato=_GUISO)
    assert r and r[0] == "60 g de salsa de tomate", r


def test_el_tomate_crudo_sigue_a_zanahoria():
    ensalada = "Ensalada fresca. Mise en place: corta 2 tomates en rodajas. Montaje: sirve con limón."
    r = cu.sustituir_linea("2 tomates", 20, _REQ, plato=ensalada, gramos_de=lambda t: 300.0)
    assert r and r[1] == "zanahoria", r
    assert cu.sustituir_linea("2 tomates", 20, _REQ, gramos_de=lambda t: 300.0)[1] == "zanahoria"


def test_el_plato_se_mide_no_se_pica():
    m = {"name": "Guiso criollo", "ingredients": ["120 g de salsa de tomate", "½ cebolla"],
         "recipe": ["Mise en place: pica 2 tomates en cubos y ½ cebolla.",
                    "El Toque de Fuego: sofríe la cebolla y el tomate picado 3 min, añade el agua y guisa 10 min."]}
    sf.reescribir_plato(m, "2 tomates", "120 g de salsa de tomate", "salsa de tomate")
    assert m["recipe"][0] == "Mise en place: mide 120 g de salsa de tomate y ½ cebolla.", m["recipe"][0]
    assert "sofríe la cebolla y la salsa de tomate 3 min" in m["recipe"][1], m["recipe"][1]


def test_el_pronombre_que_retoma_lo_sustituido():
    m = {"name": "Mangú de plátano verde", "ingredients": ["280 g de batata"],
         "recipe": ["Mise en place: pela y corta 1 plátano verdes en trozos.",
                    "El Toque de Fuego: hierve el plátano en agua con sal durante 15-18 min, hasta que un cuchillo entre "
                    "sin fuerza; escúrrelo reservando un poco del agua y májalo con un tenedor."]}
    sf.reescribir_plato(m, "1 plátano verde", "280 g de batata", "batata")
    assert m["name"] == "Mangú de batata", m["name"]
    assert m["recipe"][0] == "Mise en place: pela y corta 280 g de batata en trozos.", m["recipe"][0]
    assert "hierve la batata" in m["recipe"][1] and "escúrrela" in m["recipe"][1] and "májala" in m["recipe"][1], \
        m["recipe"][1]


def _reescribe(viejo, nueva, sub, paso):
    m = {"name": "x", "ingredients": [nueva], "recipe": [paso]}
    sf.reescribir_plato(m, viejo, nueva, sub)
    return m["recipe"][0]


def test_bateria_real_del_bloque_2():
    """Batería REAL (perfil del dueño, días 4-7 de 30 sin congelador, 27-sep): la IA metió pescado, pollo, res y
    calamar en 7 comidas; cuatro frases salían mal tras la sustitución."""
    # el ajo en polvo no se saltea solo 3 minutos: los vegetales entran después, no «con» el duradero
    p = _reescribe("1 filete de pescado (≈160 g)", "150 g de atún en agua", "atun en agua",
                   "El Toque de Fuego: saltea el pescado con el ajo en polvo durante 3 minutos, añade la cebolla y la "
                   "piña y saltea 3-4 minutos más.")
    assert p.startswith("El Toque de Fuego: saltea el atún con el ajo en polvo"), p
    # la lista «la carne, el tomate y la cebolla»: los vegetales conservan su tiempo; la tortilla conserva su «la»
    p = _reescribe("250 g de carne de res magra en tiras finas", "90 g de sardinas en lata", "sardinas en lata",
                   "El Toque de Fuego: cocina la carne de res, el tomate y la cebolla en una sartén bien caliente durante "
                   "4-5 minutos. Estira la masa en una tortilla fina y cocínala en una sartén seca 2-3 minutos.")
    assert p == ("El Toque de Fuego: cocina el tomate y la cebolla en una sartén bien caliente durante 4-5 minutos; "
                 "añade las sardinas al final y caliéntalas 1-2 min. Estira la masa en una tortilla fina y cocínala en "
                 "una sartén seca 2-3 minutos."), p
    # «opaco y firme», «opaco y completamente cocido»: la cadena entera sale
    p = _reescribe("250 g de calamar limpio", "170 g de atún en agua", "atun en agua",
                   "El Toque de Fuego: añade el calamar, el ajo y los ajíes y cocina 3-4 minutos, removiendo, hasta que el "
                   "calamar esté opaco y firme. Incorpora el limón.")
    assert "firme" not in p and "opaco" not in p, p
    p = _reescribe("280 g de calamar limpio", "90 g de sardinas en lata", "sardinas en lata",
                   "El Toque de Fuego: incorpora el calamar y saltéalo 3–4 minutos hasta que esté opaco y completamente "
                   "cocido; añade el repollo.")
    assert p == "El Toque de Fuego: incorpora las sardinas y saltéalas 2-3 minutos; añade el repollo.", p


def test_el_escudo_pasa_el_plato(monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    monkeypatch.setattr(go, "_resolve_line_food_grams", lambda line, cheap=False: ("tomate", 300.0))

    class _NoopDB:
        def macros_from_ingredient_string(self, s):
            return None
    single = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none",
                           "batch_cooking": "never"}, "diet": {"type": "balanced"}}
    days = [{"day": i + 1, "meals": [{"meal": "Cena", "name": "x", "ingredients": ["1 taza de arroz"],
                                      "ingredients_raw": ["1 taza de arroz"]}]} for i in range(20)]
    days.append({"day": 21, "meals": [{"meal": "Almuerzo", "name": "Guiso criollo",
                                       "ingredients": ["2 tomates", "1 taza de arroz"],
                                       "ingredients_raw": ["2 tomates", "1 taza de arroz"],
                                       "recipe": ["El Toque de Fuego: sofríe el tomate y guisa 10 min."]}]})
    go._single_trip_fresh_substitute(days, db=_NoopDB(), effective=single, diet="balanced", contexto={})
    assert days[-1]["meals"][0]["ingredients"][0] == "120 g de salsa de tomate", days[-1]["meals"][0]["ingredients"]
