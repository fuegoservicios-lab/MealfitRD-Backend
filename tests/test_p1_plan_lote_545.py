# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-545 · 2026-09-27] Con «30 min» o «Nada» de tiempo, la legumbre lenta llega cocida de lata.

Verificador formulario↔plan sobre las baterías reales del 27-sep: 11 perfiles con 30 min o «Nada» recibían habichuelas
SECAS con «remoja 8-12 h y hiérvelas 60-90 min» (50-143 min de pasos contra lo que respondieron).
"""
from __future__ import annotations

import pathlib
import re
import sys
from types import SimpleNamespace

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import legumbre_lista as ll  # noqa: E402


class _Db:
    def lookup(self, nombre):
        n = str(nombre).lower()
        if "habichuela" in n:
            return SimpleNamespace(name="Habichuelas rojas", kcal=333.0)
        if "garbanzo" in n:
            return SimpleNamespace(name="Garbanzos", kcal=364.0)
        if "frijol" in n:
            return SimpleNamespace(name="Frijoles negros", kcal=341.0)
        if "lenteja" in n:
            return SimpleNamespace(name="Lentejas", kcal=352.0)
        return None

    def grams_from_ingredient_string(self, linea):
        import re
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(linea))
        if m:
            return float(m.group(1))
        if "¼ taza" in str(linea):
            return 43.0
        return 90.0 if "½ taza" in str(linea) else None


def _plan(linea):
    return [{"meals": [{"ingredients": ["1 tortilla integral", linea, "30 g de queso blanco"],
                        "ingredients_raw": ["1 tortilla integral", linea, "30 g de queso blanco"],
                        "recipe": ["Mise en place: ten las habichuelas ya cocidas y escurridas."]}]}]


def test_bariatrica_con_30_minutos():
    days = _plan("35 g de habichuelas rojas crudas")
    assert ll.a_lata(days, {"cookingTime": "30min", "batchCooking": "never"}, _Db()) == 1
    m = days[0]["meals"][0]
    linea = m["ingredients"][1]
    assert linea.startswith("90 g de habichuelas rojas cocidas (de lata"), linea
    assert m["ingredients_raw"][1] == linea


def test_con_una_hora_tambien_la_habichuela_seca_llega_de_lata():
    # batería real (estatina + suplementos, «1 hora»): «1 taza de habichuelas rojas secas» → 119 min de pasos
    days = _plan("35 g de habichuelas rojas crudas")
    assert ll.a_lata(days, {"cookingTime": "1hour"}, _Db()) == 1


def test_quien_cocina_por_tandas_o_tiene_tiempo_no_cambia():
    for fd in ({"cookingTime": "30min", "batchCooking": "often"}, {"cookingTime": "plenty"}, {}):
        days = _plan("35 g de habichuelas rojas crudas")
        assert ll.a_lata(days, fd, _Db()) == 0, fd


def test_las_lentejas_solo_con_nada_de_tiempo_y_lo_ya_cocido_no_se_toca():
    days = _plan("40 g de lentejas secas")
    assert ll.a_lata(days, {"cookingTime": "30min"}, _Db()) == 0
    assert ll.a_lata(days, {"cookingTime": "none"}, _Db()) == 1
    assert "lentejas cocidas (de lata" in days[0]["meals"][0]["ingredients"][1]
    days = _plan("150 g de garbanzos cocidos")
    assert ll.a_lata(days, {"cookingTime": "none"}, _Db()) == 0


def test_los_pasos_dejan_de_remojar_la_legumbre():
    # batería real (mujer, perder grasa, 30 min): convertida la línea, quedaba «⚠️ remoja las habichuelas secas y
    # hiérvelas… al menos 10 minutos» y el Mise en place medía «70 g de habichuelas» contra 45 g en la lista
    linea = "17 g de habichuelas rojas secas"
    days = [{"meals": [{"ingredients": ["1 pechuga de pollo (≈200 g)", linea], "ingredients_raw": ["1 pechuga", linea],
                        "recipe": ["Mise en place: mide 55 g de arroz integral y 70 g de habichuelas; si usas habichuelas "
                                   "secas, remójalas y precocéelas hasta que ablanden.",
                                   "El Toque de Fuego: sofríe la cebolla 3 minutos. Cocina habichuelas rojas secas en agua "
                                   "hasta que ablanden e incorpóralas al plato.",
                                   "⚠️ Seguridad alimentaria: remoja las habichuelas secas y hiérvelas a fuego fuerte al "
                                   "menos 10 minutos antes de bajar el fuego; crudas o a medio cocer son tóxicas.",
                                   "⚠️ Seguridad alimentaria: el pollo/cerdo debe cocinarse por completo (~74°C)."]}]}]
    assert ll.a_lata(days, {"cookingTime": "30min"}, _Db()) == 1
    m = days[0]["meals"][0]
    g = m["ingredients"][1].split(" g ")[0]
    txt = " ".join(m["recipe"])
    assert "remoj" not in txt.lower() and "secas" not in txt and "ablanden" not in txt, m["recipe"]
    assert f"mide 55 g de arroz integral y {g} g de habichuelas" in m["recipe"][0], m["recipe"][0]
    assert "Incorpora las habichuelas rojas de lata, enjuagadas y escurridas, al plato." in m["recipe"][1], m["recipe"][1]
    assert "el pollo/cerdo" in txt and len(m["recipe"]) == 3


def test_corpus_lo_listo_no_se_toca_y_el_nombre_sale_limpio():
    # replay sobre 161 platos: «25 g de garbanzos tostados listos para comer» pasaba a «…listos para comer cocidos (de
    # lata)», y «¼ taza de garbanzos secos (43 g), remojados desde la noche anterior» arrastraba su cola al nombre
    days = _plan("25 g de garbanzos tostados listos para comer")
    assert ll.a_lata(days, {"cookingTime": "none"}, _Db()) == 0
    days = _plan("¼ taza de garbanzos secos (43 g), remojados desde la noche anterior")
    days[0]["meals"][0]["recipe"] = ["Mise en place: ten listos ¼ taza de garbanzos secos (60 g) cocidos y pica la cebolla.",
                                     "El Toque de Fuego: Incorpora garbanzos secos , remojados desde la noche anterior en "
                                     "agua hasta que ablanden e incorpóralos al plato."]
    assert ll.a_lata(days, {"cookingTime": "none"}, _Db()) == 1
    m = days[0]["meals"][0]
    assert re.fullmatch(r"\d+ g de garbanzos cocidos \(de lata, enjuagados y escurridos\)", m["ingredients"][1]), m
    g = m["ingredients"][1].split(" g ")[0]
    assert m["recipe"][0] == f"Mise en place: ten listos {g} g de garbanzos cocidos y pica la cebolla.", m["recipe"]
    assert m["recipe"][1] == ("El Toque de Fuego: Incorpora los garbanzos de lata, enjuagados y escurridos, al plato."), \
        m["recipe"]


def test_el_aviso_de_habichuelas_se_va_tambien_con_frijoles():
    days = _plan("60 g de frijoles negros secos")
    days[0]["meals"][0]["recipe"].append("⚠️ Seguridad alimentaria: remoja las habichuelas secas y hiérvelas a fuego fuerte "
                                         "al menos 10 minutos antes de bajar el fuego; crudas o a medio cocer son tóxicas.")
    assert ll.a_lata(days, {"cookingTime": "30min"}, _Db()) == 1
    assert not any("⚠️" in p for p in days[0]["meals"][0]["recipe"]), days[0]["meals"][0]["recipe"]


def test_ancla_en_assemble_y_en_la_cadena_final():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '__import__("legumbre_lista").a_lata(result.get("days") or [], form_data)  # [P1-PLAN-LOTE-545]' in src
    assert '__import__("legumbre_lista").a_lata(days, {"cookingTime": cooking_time}, db)  # [P1-PLAN-LOTE-545]' in src
