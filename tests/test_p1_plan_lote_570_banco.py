# backend/tests/test_p1_plan_lote_570_banco.py
"""[P1-PLAN-LOTE-570 · 2026-09-27] Banco de pruebas del analizador: el módulo puro (métricas, Nutrition5k, muestreo)."""
import math

import banco_analizador as ba

FILA = ("dish_1561662216,300.794281,193.000000,12.387489,28.218290,18.633970,"
        "ingr_0000000508,soy sauce,3.398568,1.80124104,0.020391408,0.166529832,0.275284008,"
        "ingr_0000000023,brown rice,68.000000,75.48,0.612,15.64,1.768,"
        "ingr_0000000029,mixed greens,21.339657,5.97510396,0.1,0.9,0.4").split(",")


def _plato(dish_id, kcal, proteina=20.0, carbs=30.0, grasa=10.0, masa=300.0, ingredientes=None):
    return {"dish_id": dish_id, "kcal": kcal, "masa_g": masa, "grasa_g": grasa, "carbs_g": carbs,
            "proteina_g": proteina, "ingredientes": ingredientes or []}


def test_error_relativo_usa_el_piso():
    assert ba.error_relativo(150, 100, ba.PISO_KCAL) == 0.5
    assert ba.error_relativo(5, 2, ba.PISO_MACRO_G) == 0.3      # 3 g de desvío sobre un piso de 10 g, no 150 %


def test_parse_fila_n5k():
    p = ba.parse_fila_n5k(FILA)
    assert p["dish_id"] == "dish_1561662216"
    assert math.isclose(p["kcal"], 300.794281) and p["masa_g"] == 193.0
    assert {"nombre": "brown rice", "gramos": 68.0} in p["ingredientes"]
    assert ba.parse_fila_n5k(["x"]) is None
    assert ba.parse_fila_n5k(["d", "no-numero", "1", "1", "1", "1"]) is None


def test_plato_valido_descarta_etiquetas_incoherentes():
    assert ba.plato_valido(_plato("a", 4 * 20 + 4 * 30 + 9 * 10))
    assert not ba.plato_valido(_plato("b", 50))                        # diminuto
    assert not ba.plato_valido(_plato("c", 900))                       # 900 kcal con 290 de macros: etiqueta rota
    assert not ba.plato_valido(_plato("d", 290, masa=40))


def test_muestrear_es_determinista_y_estratificado():
    platos = []
    for i in range(90):
        for lo in (150, 450, 800):
            kcal = lo + i
            platos.append(_plato(f"dish_{lo}_{i:03d}", kcal, proteina=kcal * 0.25 / 4, carbs=kcal * 0.5 / 4,
                                 grasa=kcal * 0.25 / 9))
    con_foto = {p["dish_id"] for p in platos if not p["dish_id"].endswith("7")}
    a = ba.muestrear(platos, con_foto, n=150, semilla=7)
    b = ba.muestrear(list(reversed(platos)), con_foto, n=150, semilla=7)
    assert [p["dish_id"] for p in a] == [p["dish_id"] for p in b]
    assert len(a) == 150 and all(p["dish_id"] in con_foto for p in a)
    tramos = [sum(1 for p in a if lo <= p["kcal"] < hi) for lo, hi in ba.TRAMOS_KCAL]
    assert tramos == [50, 50, 50]


def test_clase_de_por_palabra_entera():
    assert ba.clase_de("white rice", "en") == "rice"
    assert ba.clase_de("Arroz blanco", "es") == "rice"
    assert ba.clase_de("Repollo", "es") == "leafy"            # «pollo» ⊂ «repollo»: por palabra, no por subcadena
    assert ba.clase_de("Queso fresco", "es") == "cheese"      # «res» ⊂ «fresco»
    assert ba.clase_de("Filete de pescado", "es") == "fish"
    assert ba.clase_de("scrambled eggs", "en") == "egg"
    assert ba.clase_de("Plátano maduro", "es") == "other"


def test_evaluar_plato_clasifica_los_fallos():
    v = _plato("a", 290)
    assert ba.evaluar_plato(v, {"analysis_failed": True})["fallo"] == "error"
    assert ba.evaluar_plato(v, {"is_food": False, "calories": 0})["fallo"] == "no_comida"
    assert ba.evaluar_plato(v, {"is_food": True, "photo_kind": "items", "calories": 0})["fallo"] == "sin_totales"
    assert ba.evaluar_plato(v, None)["fallo"] == "error"


def test_evaluar_plato_mide_errores_y_componentes():
    v = _plato("a", 400, proteina=30, carbs=40, grasa=13.8, masa=300,
               ingredientes=[{"nombre": "white rice", "gramos": 150}, {"nombre": "chicken breast", "gramos": 100},
                             {"nombre": "olive oil", "gramos": 10}])
    est = {"is_food": True, "photo_kind": "plato", "calories": 500, "protein": 27, "carbs": 40, "healthy_fats": 14,
           "items": [{"name": "Arroz blanco"}, {"name": "Brócoli"}]}
    f = ba.evaluar_plato(v, est, latencia_s=4.2)
    assert f["fallo"] is None and f["latencia_s"] == 4.2
    assert f["errores"]["kcal"] == 0.25 and f["errores"]["proteina_g"] == 0.1
    assert f["componentes"]["esperadas"] == ["chicken", "rice"] and f["componentes"]["acertadas"] == ["rice"]


def test_agregar_mediana_p90_y_recall():
    filas = [{"fallo": None, "latencia_s": float(i), "errores": {"kcal": e, "proteina_g": e, "carbs_g": e, "grasa_g": e},
              "componentes": {"esperadas": ["rice"], "acertadas": ["rice"] if i % 2 else [], "otros_pct": 0.0}}
             for i, e in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])]
    r = ba.agregar(filas)
    assert r["n"] == 10 and r["ok"] == 10 and r["valida"] is True
    assert math.isclose(r["kcal"]["mediana"], 0.55) and r["kcal"]["p90"] == 0.9
    assert r["kcal_dentro_20pct"] == 0.2 and r["recall_componentes"] == 0.5


def test_agregar_marca_invalida_con_muchos_fallos():
    ok = {"fallo": None, "latencia_s": 1.0, "errores": {"kcal": 0.1, "proteina_g": 0.1, "carbs_g": 0.1, "grasa_g": 0.1},
          "componentes": {"esperadas": [], "acertadas": [], "otros_pct": 0.0}}
    r = ba.agregar([ok] * 8 + [{"fallo": "error", "latencia_s": None}] * 2)
    assert r["tasa_fallos"] == 0.2 and r["valida"] is False and r["fallos"] == {"error": 2}


def test_verificar_cache_detecta_el_distinto():
    man = ba.manifiesto([_plato("a", 290), _plato("b", 290)], {"a": ba.sha256_de(b"A"), "b": ba.sha256_de(b"B")}, 1, "x")
    datos = {"a": b"A", "b": b"otro"}
    assert ba.verificar_cache(man, datos.get) == ["b"]
    assert ba.verificar_cache(man, {"a": b"A"}.get) == ["b"]      # ausente = distinto


def test_clase_de_con_plurales_y_lexico_dominicano():
    # Revisión final: el vocabulario en español no cubría lo que sí cubre el inglés, y un «other» del ANALIZADOR cuenta
    # como fallo suyo: el recall habría medido el diccionario, no el analizador.
    es = {"Batata asada": "potato", "Tocineta": "pork", "Ají morrón": "vegetable", "Pepinos": "vegetable",
          "Cebollas": "vegetable", "Churrasco": "beef", "Moras": "fruit", "Alitas": "chicken"}
    en = {"sweet potato": "potato", "bell peppers": "vegetable", "tomatoes": "vegetable", "turkey": "chicken"}
    for nombre, clase in es.items():
        assert ba.clase_de(nombre, "es") == clase, nombre
    for nombre, clase in en.items():
        assert ba.clase_de(nombre, "en") == clase, nombre


def test_se_mide_lo_que_el_analizador_dice_y_no_se_clasifica():
    v = _plato("a", 400, proteina=30, carbs=40, grasa=13.8,
               ingredientes=[{"nombre": "white rice", "gramos": 200}])
    est = {"is_food": True, "calories": 400, "protein": 30, "carbs": 40, "healthy_fats": 14,
           "items": [{"name": "Arroz"}, {"name": "Cosa rarísima"}]}
    f = ba.evaluar_plato(v, est)
    assert f["componentes"]["otros_estimado_pct"] == 0.5
    assert ba.agregar([f])["otros_estimado_pct_medio"] == 0.5


def test_clases_para_lo_que_el_banco_real_dejaba_fuera():
    # Medido sobre el manifiesto congelado: 66 de 323 componentes principales caían en «other» (almendras ×18,
    # pizza ×8, ensalada César ×8, coles de Bruselas, yogur, leche, quinoa, tofu, aguacate, ñame…).
    pares = [("almonds", "Almendras", "nuts"), ("pizza", "Pizza de queso", "pizza"),
             ("caesar salad", "Ensalada César", "leafy"), ("brussels sprouts", "Coles de Bruselas", "leafy"),
             ("greek yogurt", "Yogur griego", "dairy"), ("milk", "Leche", "dairy"), ("quinoa", "Quinoa", "grain"),
             ("tofu", "Tofu", "beans"), ("avocado", "Aguacate", "avocado"), ("yam", "Ñame", "potato"),
             ("hash browns", "Papas", "potato")]
    for en, es, clase in pares:
        assert ba.clase_de(en, "en") == clase, en
        assert ba.clase_de(es, "es") == clase, es
