# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-623 · 2026-09-27] Una alergia escrita con el nombre de OTRO país hispano bloquea el alimento.

La auditoría de países (27-sep) lo midió con `clinical_backstop_for_meal` y país ES: quien declaraba «Melocotón»
recibía «durazno», «Maracuyá» → «chinola», «Calabaza» → «auyama», «Boniato» → «batata», «Judías verdes» →
«vainitas», «Plátano macho» → «plátano verde». Los alérgenos mayores sí bloqueaban (tienen clase con sinónimos);
estos no la tienen, y el lote 225 sólo tradujo los OTROS idiomas al canónico, no el español de otro país. La alergia al
melocotón (LTP) es de las más frecuentes en adultos en España, y el catálogo del motor es dominicano en los 6 países.

`data/food_names_i18n.json` gana dos listas junto a las traducciones:
  - `variantes_regionales`: nombres inequívocos del mismo alimento en otro país (Melocotón = Duraznos). Entran en el
    mismo índice que las traducciones, así que también sirven a los rechazos y a los pools.
  - `variantes_solo_alergias`: nombres AMBIGUOS entre países («Plátano» es banana en España y México y plátano de
    cocinar en RD). Solo amplían la alergia (bloquear de más es la dirección segura); en un rechazo o un buscador
    quitarían la banana al dominicano que no quiere plátano.
Y el canónico que se añade a la búsqueda lleva también su singular: el patrón del escáner tolera el plural del término
(«durazno» → «duraznos») pero no al revés, y el canónico de catálogo suele ir en plural («Duraznos»).

tooltip-anchor: P1-PLAN-LOTE-623
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

BLOQUEAN = [
    ("Melocotón", "1 durazno fresco"),
    ("Melocotón", "150 g de Duraznos"),
    ("Melocotones", "½ taza de durazno en almíbar"),
    ("Maracuyá", "½ taza de jugo de chinola"),
    ("Parchita", "½ taza de jugo de chinola"),
    ("Calabaza", "100 g de auyama"),
    ("Zapallo", "100 g de auyama"),
    ("Boniato", "1 batata mediana"),
    ("Camote", "1 batata mediana"),
    ("Judías verdes", "1 taza de vainitas"),
    ("Ejotes", "1 taza de vainitas"),
    ("Plátano macho", "1 plátano verde"),
    ("Plátano macho", "½ plátano maduro"),
    ("Plátano", "1 guineo mediano"),
    ("Frijoles negros", "1 taza de habichuelas negras"),
    ("Alubias", "½ taza de habichuelas blancas"),
    ("Frijoles", "½ taza de habichuelas rojas"),
    ("Jitomate", "2 tomates"),
    ("Patata", "200 g de papa"),
    ("Güisquil", "1 tayota"),
    ("Pimiento", "½ ají morrón"),
    ("Elote", "½ taza de maíz dulce en granos"),
    ("Puerco", "150 g de cerdo"),
    ("Mandioca", "200 g de yuca"),
    ("Arvejas", "½ taza de guisantes secos"),
    ("Palta", "¼ aguacate"),
    ("Cambur", "1 guineo"),
]


def _backstop(alergia, ingrediente):
    logging.disable(logging.CRITICAL)
    try:
        from graph_orchestrator import clinical_backstop_for_meal
        meal = {"name": "Plato", "meal": "Almuerzo", "ingredients": [ingrediente], "recipe": []}
        return clinical_backstop_for_meal(meal, allergies=[alergia], form_data={"country": "ES"})
    finally:
        logging.disable(logging.NOTSET)


@pytest.mark.parametrize("alergia,ingrediente", BLOQUEAN)
def test_la_alergia_con_el_nombre_de_otro_pais_bloquea(alergia, ingrediente):
    assert _backstop(alergia, ingrediente), f"«{alergia}» dejó pasar «{ingrediente}»"


def test_el_plural_del_canonico_no_esconde_el_singular():
    # «Strawberry» → canónico «Fresas»; el plato puede decir «fresa» en singular
    assert _backstop("Strawberry", "5 fresa picada")


def test_lo_que_ya_no_bloqueaba_sigue_sin_bloquear():
    # «leche» no es «lechosa», «maíz» no es «maní»: la ampliación no abre falsos positivos por prefijo
    assert not _backstop("Leche", "1 taza de lechosa")
    assert not _backstop("Melocotón", "1 manzana")
    assert not _backstop("Calabaza", "1 zanahoria")


def test_lo_ambiguo_no_entra_en_rechazos_ni_buscadores():
    from food_names_i18n import canonicos_para_texto
    # un dominicano que no quiere «plátano» no pierde la banana (Guineo)
    assert "Guineo" not in canonicos_para_texto("plátano")
    # lo inequívoco sí resuelve para todos (rechazos, pools)
    assert "Duraznos" in canonicos_para_texto("melocotón")
    assert "Batata" in canonicos_para_texto("boniato")


def test_cada_variante_apunta_a_una_fila_del_catalogo():
    datos = json.loads((_BACKEND / "data" / "food_names_i18n.json").read_text(encoding="utf-8"))
    alimentos = datos["alimentos"]
    for seccion in ("variantes_regionales", "variantes_solo_alergias"):
        assert isinstance(datos.get(seccion), dict) and datos[seccion], seccion
        for canonico, nombres in datos[seccion].items():
            assert canonico in alimentos, (seccion, canonico)
            assert isinstance(nombres, list) and nombres and all(isinstance(n, str) and n.strip() for n in nombres)
    # lo ambiguo no se repite en lo inequívoco
    for canonico, nombres in datos["variantes_solo_alergias"].items():
        assert not set(nombres) & set(datos["variantes_regionales"].get(canonico, [])), canonico


def test_sin_el_archivo_nada_se_rompe(monkeypatch):
    import food_names_i18n as f
    monkeypatch.setattr(f, "_RUTA", _BACKEND / "data" / "no_existe.json")
    for fn in (f.nombres, f._datos, f._indice, f._indice_solo_alergias, f.canonicos_para_texto):
        fn.cache_clear()
    try:
        assert f.canonicos_para_texto("melocotón") == ()
        assert f.canonicos_que_el_literal_no_alcanza("melocoton", lambda t: t) == []
    finally:
        for fn in (f.nombres, f._datos, f._indice, f._indice_solo_alergias, f.canonicos_para_texto):
            fn.cache_clear()
