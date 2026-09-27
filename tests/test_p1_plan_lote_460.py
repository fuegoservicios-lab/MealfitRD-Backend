# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-460 · 2026-09-27] La sustitución por frescura de la compra única cambia el PLATO, no sólo la lista.

Plan vivo del dueño (6594aae1, 30 días sin congelador): la lista del día 3 traía «150 g de sardinas en lata», pero el plato
seguía siendo «Wok express de **pollo**» y el paso decía «saltea el pollo… hasta que el pollo esté completamente cocido»;
el día 2 mandaba a cocinar «el sardinas en lata blanco a la plancha, 3-4 min por lado». El reemplazo era de UN token
(«pechuga de pollo») y el género quedaba del fresco. Forzando la compra única sobre el corpus: 73 de 150 nombres y 106
recetas nombraban aún la proteína vieja; 47 cocinaban lo enlatado «por lado».
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import sustitucion_fresca as sf  # noqa: E402

SINGLE = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none", "batch_cooking": "never"},
          "diet": {"type": "balanced"}}


class _NoopDB:
    def macros_from_ingredient_string(self, s):
        return None

    def lookup(self, s):
        return None


def _wok():
    return {"meal": "Cena", "name": "Wok express de pollo con tomate y repollo sobre batata",
            "desc": "Pollo salteado al estilo wok con tomate, cebolla y repollo, servido sobre batata tierna.",
            "ingredients": ["¾ pechuga de pollo (≈150 g)", "1 batata mediana", "1 taza de repollo rallado", "2 tomates",
                            "1 cebolla", "1 cdta de aceite de oliva", "1 limón", "Ajo", "Pimienta negra al gusto"],
            "ingredients_raw": ["150 g de pechuga de pollo", "1 taza de repollo rallado", "2 tomates", "1 cebolla",
                                "1 limón", "Ajo", "Pimienta negra al gusto", "1 batata mediana", "1 cdta de aceite de oliva"],
            "recipe": [
                "Mise en place: corta 1 batata mediana (195 g) en cubos pequeños y coloca en un recipiente apto para "
                "microondas; corta ¾ pechuga de pollo (≈150 g), pica 2 tomates, 1 cebolla y 1 diente de Ajo, y mide 1 "
                "taza de repollo rallado.",
                "El Toque de Fuego: cocina la batata tapada en el microondas durante 7-8 min, hasta que esté tierna. A la "
                "vez, calienta 1 cdta de aceite de oliva en una sartén o wok; saltea el pollo con la cebolla, el tomate, el "
                "repollo y Pimienta negra al gusto durante 6-7 min, hasta que el pollo esté completamente cocido.",
                "Montaje: sirve la batata como base y coloca encima el pollo salteado con los vegetales; termina con el "
                "jugo de 1 limón."]}


def _pasos(m):
    return " ".join(p for p in m["recipe"] if not p.lstrip().startswith(("⚠", "💡", "🤰", "⚕")))


def test_el_wok_del_duenyo_ya_no_nombra_el_pollo():
    m = _wok()
    sf.sustituir_en_plato(m, 0, "¾ pechuga de pollo (≈150 g)", "150 g de sardinas en lata", "sardinas en lata")
    texto = (m["name"] + " " + m["desc"] + " " + _pasos(m)).lower()
    assert not re.search(r"\b(?:pollo|pechuga)\b", texto), texto          # «repollo» sí sigue
    assert m["name"] == "Wok express de sardinas con tomate y repollo sobre batata", m["name"]
    assert "escurre 150 g de sardinas en lata" in m["recipe"][0], m["recipe"][0]
    # los vegetales conservan su salteado; las sardinas, que ya vienen cocidas, entran al final
    assert "saltea la cebolla, el tomate, el repollo y Pimienta negra al gusto durante 6-7 min" in m["recipe"][1]
    assert "añade las sardinas al final y caliéntalas 1-2 min" in m["recipe"][1], m["recipe"][1]
    assert "cocido" not in m["recipe"][1].split("A la vez")[1], m["recipe"][1]
    assert "coloca encima las sardinas salteadas" in m["recipe"][2], m["recipe"][2]
    assert m["desc"].startswith("Sardinas salteadas al estilo wok"), m["desc"]


def test_el_pescado_del_duenyo_no_se_cocina_por_lado_ni_a_63_grados():
    m = {"meal": "Almuerzo", "name": "Bowl caribeño de pescado blanco con yautía y aguacate",
         "ingredients": ["1 filete de pescado", "½ tomate"], "ingredients_raw": ["½ tomate", "1 filete de pescado"],
         "recipe": ["Mise en place: pela y corta ½ pedazo de yautía en cubos; seca el filete de pescado blanco de 150 g; "
                    "corta ½ tomate.",
                    "El Toque de Fuego: hierve la yautía hasta que el cuchillo entre sin fuerza y escúrrela. En una sartén "
                    "antiadherente a fuego medio cocina el pescado 3-4 minutos por lado, hasta alcanzar 63 °C en el centro.",
                    "Montaje: sirve la yautía como base del bowl, coloca encima el pescado y los vegetales."]}
    sf.sustituir_en_plato(m, 0, "1 filete de pescado", "150 g de sardinas en lata", "sardinas en lata")
    assert m["name"] == "Bowl caribeño de sardinas con yautía y aguacate", m["name"]
    pasos = _pasos(m)
    assert "pescado" not in pasos and "filete" not in pasos, pasos
    assert "por lado" not in pasos and "°C" not in pasos, pasos
    assert "escurre 150 g de sardinas en lata" in m["recipe"][0], m["recipe"][0]
    assert "calienta las sardinas 2-3 minutos." in m["recipe"][1], m["recipe"][1]
    assert "coloca encima las sardinas y los vegetales" in m["recipe"][2], m["recipe"][2]


def test_concordancia_y_metodos_de_un_crudo():
    casos = [
        ("1 pechuga de pollo (porción) (205 g)", "205 g de atún en agua", "atun en agua",
         "Cocina la pechuga con 1 cdta de aceite de oliva a fuego medio durante 6-8 minutos por lado, o hasta alcanzar "
         "74 °C en la parte más gruesa.",
         "Calienta el atún con 1 cdta de aceite de oliva a fuego medio durante 2-3 minutos."),
        ("150 g de filete de tilapia", "150 g de garbanzos cocidos", "garbanzos cocidos",
         "Asa la tilapia con el ajo, sal y pimienta durante 3-4 min por lado, hasta que esté opaca y se desmenuce "
         "fácilmente.",
         "Calienta los garbanzos con el ajo, sal y pimienta durante 2-3 min."),
        ("200 g de camarones", "200 g de sardinas en lata", "sardinas en lata",
         "Cocina camarones a la plancha o hervidos y sírvelos como proteína del plato.",
         "Calienta sardinas y sírvelas como proteína del plato."),
        ("220 g de pechuga de pollo", "220 g de atún en agua", "atun en agua",
         "Mise en place: corta 220 g de pechuga de pollo en tiras y sazónalas con sal; pica la cebolla.",
         "Mise en place: escurre 220 g de atún en agua y sazónalo con sal; pica la cebolla."),
        ("220 g de pechuga de pollo", "220 g de atún en agua", "atun en agua",
         "Incorpora el pollo y el guineo verde cocido, y calienta 2 minutos más, hasta que el centro del pollo alcance "
         "74 °C.",
         "Incorpora el atún y el guineo verde cocido, y calienta 2 minutos más."),
    ]
    for viejo, nueva, sub, paso, esperado in casos:
        m = {"name": "x", "ingredients": [nueva], "recipe": [paso]}
        sf.reescribir_plato(m, viejo, nueva, sub)
        assert m["recipe"][0] == esperado, (viejo, m["recipe"][0])


def test_el_nombre_pierde_el_metodo_del_crudo_y_concuerda():
    for viejo, sub, nombre, esperado in (
        ("½ pechuga de pollo (≈100 g)", "atun en agua", "Pollo guisado ligero con batata al vapor y bok choy",
         "Atún guisado ligero con batata al vapor y bok choy"),
        ("1 filete de tilapia", "sardinas en lata", "Filete de tilapia a la plancha con batata asada y ensalada",
         "Sardinas con batata asada y ensalada"),
        ("½ pechuga de pollo", "sardinas en lata", "Pollo horneado criollo con arroz y bok choy",
         "Sardinas criollas con arroz y bok choy"),
        ("1 pechuga de pollo", "atun en agua", "Pechuga dorada a la plancha y terminada con limón",
         "Atún terminado con limón"),
    ):
        m = {"name": nombre, "ingredients": ["x"], "recipe": []}
        sf.reescribir_plato(m, viejo, "150 g de x", sub)
        assert m["name"] == esperado, (nombre, m["name"])


def test_lo_fresco_que_no_es_proteina_concuerda_sin_tocar_la_coccion():
    m = {"name": "Tostadas con huevo y aguacate", "ingredients": ["125 g de manzana"],
         "recipe": ["Mise en place: corta el aguacate en láminas finas.", "Montaje: sirve el aguacate fresco encima."]}
    sf.reescribir_plato(m, "½ aguacate", "125 g de manzana", "manzana")
    assert m["name"] == "Tostadas con huevo y manzana", m["name"]
    assert m["recipe"] == ["Mise en place: corta la manzana en láminas finas.", "Montaje: sirve la manzana encima."]


def test_la_nota_de_seguridad_del_crudo_sale_con_el_crudo():
    m = {"name": "Tostadas con pollo", "ingredients": ["¼ pechuga de pollo (≈70 g)", "3 rebanadas de pan integral"],
         "ingredients_raw": ["¼ pechuga de pollo (≈70 g)", "3 rebanadas de pan integral"],
         "recipe": ["Montaje: sirve las tostadas con la pechuga de pollo.",
                    "⚠️ Seguridad alimentaria: cocina pechuga de pollo por completo antes de servir; evita consumirlo "
                    "crudo o poco cocido.",
                    "🤰 Seguridad alimentaria (embarazo/lactancia): cocina el pescado y los mariscos POR COMPLETO."]}
    sf.sustituir_en_plato(m, 0, "¼ pechuga de pollo (≈70 g)", "70 g de atún en agua", "atun en agua")
    assert not any(re.search(r"\b(?:pechuga|pollo)\b", p) for p in m["recipe"]), m["recipe"]
    assert m["recipe"][0] == "Montaje: sirve las tostadas con el atún.", m["recipe"][0]
    assert any(p.startswith("🤰") for p in m["recipe"]), "la nota general del embarazo se queda"


def test_el_reparador_del_huevo_sustituido_no_devuelve_la_pechuga():
    """La marca «huevo->pollo» es de cuando se cambió el huevo; si la compra única cambió después esa pechuga por atún,
    el reparador 426 la devolvía al plato («Cocina la pechuga… 74 °C», «Acompaña con la pechuga de pollo en tiras»)."""
    import pasos_cerrador as pc
    m = {"meal": "Cena", "name": "Tostadas con atún", "_protein_autofix_applied": "huevo->pollo",
         "ingredients": ["3 rebanadas de pan integral", "70 g de atún en agua"],
         "recipe": ["Mise en place: corta 3 rebanadas de pan integral.",
                    "El Toque de Fuego: hornea el pan 8-10 minutos. Calienta atún y sírvelo como proteína del plato.",
                    "Montaje: sirve las tostadas. Acompaña con atún.",
                    "⚠️ Seguridad alimentaria: cocina pechuga de pollo por completo antes de servir; evita consumirlo "
                    "crudo o poco cocido."]}
    antes = list(m["recipe"])
    assert pc.huevo_sustituido(m) == 0
    assert m["recipe"] == antes


def test_de_punta_a_punta_en_el_escudo(monkeypatch):
    """El camino real: `_single_trip_fresh_substitute` (pre-INSERT) en el día 10 de un ciclo de 30 sin congelador."""
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = [{"day": i + 1, "meals": [{"meal": "Cena", "name": "x", "ingredients": ["1 taza de arroz"],
                                      "ingredients_raw": ["1 taza de arroz"]}]} for i in range(9)]
    days.append({"day": 10, "meals": [_wok()]})
    assert go._single_trip_fresh_substitute(days, db=_NoopDB(), effective=SINGLE, diet="balanced", contexto={}) >= 1
    m = days[-1]["meals"][0]
    texto = (m["name"] + " " + _pasos(m)).lower()
    assert not re.search(r"\b(?:pollo|pechuga)\b", texto), texto
    assert not re.search(r"\b(?:el|la) (?:sardinas|garbanzos|lentejas)\b", texto), texto
    assert not re.search(r"\bel atún\b[^;.]*\bcaliéntala\b", texto), texto
    assert "completamente cocido" not in texto and "°c" not in texto, texto
    assert "atún en agua" in m["ingredients"][0] or "sardinas" in m["ingredients"][0] or "garbanzos" in m["ingredients"][0]


def test_el_grafo_delega_en_el_modulo():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def _single_trip_fresh_substitute(")
    cuerpo = src[i:src.index("\ndef ", i + 10)]
    assert '__import__("sustitucion_fresca").sustituir_en_plato(m, idx, text, new_line, sub)' in cuerpo
    assert "hit_tok) +" not in cuerpo, "el reemplazo de UN token no vuelve"
