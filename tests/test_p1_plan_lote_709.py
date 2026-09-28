# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-709 · 2026-09-28] Lo que siembra el cerrador de micros se usa en el Montaje, no sólo en una nota.

Replay del corpus (425 planes, cola del árbol 665): 229 de 31 236 líneas de la lista nombran un alimento que ningún paso
usa; 176 (77 %) son la siembra del cerrador de micros («10 g de semillas de linaza», «½ zanahoria»), cuya única
instrucción vive en la nota 🌱 («espolvorea semillas de linaza sobre el plato al servir — cierra tu omega-3 del día»).
Quien sigue los pasos no la ve como paso. Ahora el Montaje la incorpora («Espolvorea las semillas de linaza por
encima.») y la nota queda con el porqué («las semillas de linaza cierran tu omega-3 del día»). Sin artículo conocido
para el alimento, no se toca nada. Knob `MEALFIT_SIEMBRA_EN_EL_MONTAJE`.

tooltip-anchor: P1-PLAN-LOTE-709
"""
import siembra_en_el_montaje as sm


def _meal(ings, rec, name="Yogurt con fresas"):
    return {"name": name, "ingredients": ings, "recipe": rec, "_display": {"en-US": {"name": "x"}}}


def test_las_semillas_pasan_al_montaje_y_la_nota_queda_con_el_porque():
    m = _meal(["1 taza de yogurt natural", "10 g de semillas de linaza"],
              ["Mise en place: mide el yogurt.",
               "🌱 Nota del Nutricionista AI: espolvorea semillas de linaza sobre el plato al servir — cierra tu omega-3 del día.",
               "Montaje: sirve el yogurt en un bol."])
    assert sm.integrar(m) == 1
    assert m["recipe"][2] == "Montaje: sirve el yogurt en un bol. Espolvorea las semillas de linaza por encima."
    assert m["recipe"][1] == "🌱 Nota del Nutricionista AI: las semillas de linaza cierran tu omega-3 del día."
    assert "_display" not in m


def test_la_zanahoria_acompana_en_singular():
    m = _meal(["150 g de pollo", "½ zanahoria"],
              ["El Toque de Fuego: cocina el pollo.",
               "🌱 Nota del Nutricionista AI: acompaña el plato con zanahoria rallada — cierra tu vitamina A del día.",
               "Montaje: sirve el pollo"])
    assert sm.integrar(m) == 1
    assert m["recipe"][2] == "Montaje: sirve el pollo. Acompaña con la zanahoria rallada."
    assert m["recipe"][1] == "🌱 Nota del Nutricionista AI: la zanahoria rallada cierra tu vitamina A del día."


def test_bebida_y_sin_montaje_va_al_ultimo_paso():
    m = _meal(["1 taza de leche", "10 g de semillas de girasol sin sal"],
              ["Licúa la leche con la pera.",
               "🌱 Nota del Nutricionista AI: espolvorea semillas de girasol sin sal al servir — cierra tu vitamina E del día."],
              name="Batido de pera")
    assert sm.integrar(m) == 1
    assert m["recipe"][0] == "Licúa la leche con la pera. Espolvorea las semillas de girasol sin sal por encima."


def test_si_un_paso_ya_lo_usa_no_se_toca():
    rec = ["Montaje: sirve y espolvorea la linaza.",
           "🌱 Nota del Nutricionista AI: espolvorea semillas de linaza sobre el plato al servir — cierra tu omega-3 del día."]
    m = _meal(["10 g de semillas de linaza"], list(rec))
    assert sm.integrar(m) == 0 and m["recipe"] == rec


def test_alimento_sin_articulo_conocido_no_se_toca():
    rec = ["Montaje: sirve.",
           "🌱 Nota del Nutricionista AI: espolvorea cacao nibs sobre el plato al servir — cierra tu magnesio del día."]
    m = _meal(["10 g de cacao nibs"], list(rec))
    assert sm.integrar(m) == 0 and m["recipe"] == rec


def test_otras_notas_no_se_tocan():
    rec = ["Montaje: sirve.", "🌱 Nota del Nutricionista AI: esta receta usa solo las claras — NO botes las yemas."]
    m = _meal(["4 claras"], list(rec))
    assert sm.integrar(m) == 0 and m["recipe"] == rec


def test_knob_de_rollback(monkeypatch):
    monkeypatch.setenv("MEALFIT_SIEMBRA_EN_EL_MONTAJE", "false")
    m = _meal(["10 g de semillas de linaza"],
              ["Montaje: sirve.",
               "🌱 Nota del Nutricionista AI: espolvorea semillas de linaza sobre el plato al servir — cierra tu omega-3 del día."])
    assert sm.integrar(m) == 0


def test_el_contrato_lo_engancha_tras_servir_lo_que_sobra():
    src = open(__import__("recipe_contract").__file__, encoding="utf-8").read()
    i = src.index('__import__("pasos_cantidades").servir_lo_que_sobra(meal)')
    j = src.index('__import__("siembra_en_el_montaje").integrar(meal)')
    k = src.index('__import__("pasos_cerrador").acompanamientos_en_una_frase(meal)')
    assert i < j < k


def test_acompanamientos_no_repiten_el_mismo_objeto():
    # rd252 del replay: «Acompaña con yogurt natural entero. Acompaña con yogurt griego entero.», alineadas por el 636 a
    # «yogurt», con la siembra detrás → el 449 las unía en «Acompaña con yogurt y yogurt.»
    import pasos_cerrador as pc
    m = {"recipe": ["Montaje: Acompaña con yogurt. Acompaña con yogurt. Espolvorea las semillas de linaza por encima."]}
    assert pc.acompanamientos_en_una_frase(m) == 1
    assert m["recipe"][0] == "Montaje: Acompaña con yogurt. Espolvorea las semillas de linaza por encima."
    m = {"recipe": ["Montaje: sirve. Acompaña con edamame. Acompaña con la zanahoria rallada. Acompaña con agua."]}
    pc.acompanamientos_en_una_frase(m)
    assert m["recipe"][0] == "Montaje: sirve. Acompaña con edamame, la zanahoria rallada y agua."


def test_la_nota_que_ya_trae_articulo():
    # rd543 del replay: «espolvorea las semillas de linaza sobre el plato al servir»
    m = _meal(["5 g de semillas de linaza"],
              ["Montaje: sirve el parfait.",
               "🌱 Nota del Nutricionista AI: espolvorea las semillas de linaza sobre el plato al servir — cierra tu omega-3 del día."])
    assert sm.integrar(m) == 1
    assert m["recipe"][0] == "Montaje: sirve el parfait. Espolvorea las semillas de linaza por encima."
    assert m["recipe"][1] == "🌱 Nota del Nutricionista AI: las semillas de linaza cierran tu omega-3 del día."
