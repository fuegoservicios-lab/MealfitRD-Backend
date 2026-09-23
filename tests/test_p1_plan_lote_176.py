# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-176 · 2026-09-23] Quinta vuelta de la batería REAL del generador (RD), con el 175 dentro: los 4 perfiles
clínicos pasan ya al PRIMER intento (embarazo y lactancia aprobados; HTA sin rechazo; DM2 entregado con aviso). Lo que
queda a la vista:

1. **La costura de los swaps de PROTEÍNA** (la del 175 era la de la fruta): el modelo escribió «Yautía guisada al
   horno con huevos bien cocidos…» —el prompt de embarazo/lactancia pide «huevo bien cocido»— y el diversificador de
   huevo lo cambió por pechuga: el nombre quedó «…con pechuga de pollo bien cocidos…» y la descripción siguió
   «coronada con huevo bien cocido». El pase de honestidad de descripciones no lo reparaba por dos razones: el huevo
   no tenía subgrupo de cambio, y el «bien» entre el alimento y su participio abortaba el cambio.
2. **DM2: plátano maduro → verde.** El revisor lo marcó en las TRES corridas reales de DM2+insulina, y el «como mucho
   una vez cada 3 días» del prompt no se puede cumplir: cada día se genera por separado."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def test_la_descripcion_sigue_al_huevo_cambiado_por_pollo():
    import graph_orchestrator as go
    meal = {"name": "Yautía guisada al horno con pechuga de pollo y cebolla morada",
            "desc": "Cena caliente: yautía horneada en guiso ligero de cebolla y tomate, coronada con huevo bien cocido.",
            "ingredients": ["¾ pedazo de yautía (≈139 g)", "½ pechuga de pollo (≈108 g)", "½ cebolla morada"]}
    assert go._desc_food_honesty_pass([{"meals": [meal]}]) == 1
    assert meal["desc"].endswith("coronada con pollo bien cocido."), meal["desc"]


def test_el_adverbio_no_aborta_y_el_participio_se_reconcuerda():
    import graph_orchestrator as go
    meal = {"desc": "Tostadas con la guayaba bien madura del patio.",
            "ingredients": ["2 rebanadas de pan integral", "1 mango"]}
    go._desc_food_honesty_pass([{"meals": [meal]}])
    assert meal["desc"] == "Tostadas con el mango bien maduro del patio.", meal["desc"]


def test_el_huevo_con_queso_sigue_sin_cambio_generico():
    """El queso es lácteo: «huevo» → «queso» no es un cambio honesto genérico (decisión que se conserva)."""
    import graph_orchestrator as go
    meal = {"desc": "Casabe con huevo bien cocido y tomate.", "ingredients": ["1 casabe", "30 g de queso blanco"]}
    go._desc_food_honesty_pass([{"meals": [meal]}])
    assert "queso" not in meal["desc"]


def test_bien_mas_participio_concuerda_con_el_nucleo():
    import graph_orchestrator as go
    fix = go._fix_name_gender_agreement
    assert fix("Yautía guisada al horno con pechuga de pollo bien cocidos y cebolla morada") == \
        "Yautía guisada al horno con pechuga de pollo bien cocida y cebolla morada"
    assert fix("Arroz con claras bien cocidos") == "Arroz con claras bien cocidas"
    # ya concuerda con el núcleo o con el sustantivo más cercano: las dos lecturas valen
    assert fix("Guiso de lentejas bien cocidas") is None
    assert fix("Pechuga de pollo bien cocido") is None
    assert fix("Tostadas con huevos bien cocidos") is None


def test_dm2_cambia_el_platano_maduro_por_verde():
    import graph_orchestrator as go
    plan = {"days": [{"meals": [{"meal": "Cena", "name": "Mero al horno con plátano maduro asado",
                                 "ingredients": ["150 g de mero", "1 plátano maduro"],
                                 "ingredients_raw": ["150 g de mero", "1 plátano maduro"],
                                 "recipe": ["Asa el plátano maduro en el horno."]}]}]}
    assert go._apply_condition_substitutions(plan, {"medicalConditions": ["Diabetes tipo 2"]}) == 1
    m = plan["days"][0]["meals"][0]
    assert not any("maduro" in i.lower() for i in m["ingredients"])
    assert any("plátano verde" in i.lower() for i in m["ingredients"])
    assert "maduro" not in m["name"].lower(), m["name"]


def test_sin_diabetes_el_maduro_se_queda():
    import graph_orchestrator as go
    plan = {"days": [{"meals": [{"name": "Maduro asado", "ingredients": ["1 plátano maduro"]}]}]}
    go._apply_condition_substitutions(plan, {"medicalConditions": ["Ninguna"]})
    assert plan["days"][0]["meals"][0]["ingredients"] == ["1 plátano maduro"]


def test_el_sustituto_toma_la_caja_del_titulo():
    """Batería real (familia de 4): el 175 escribía el sustituto siempre en minúscula en mitad del nombre, también en un
    título en Title Case: «Tostadas Integrales con Revoltillo de Huevo, Tomate y aguacate»."""
    import dish_naming as dn
    pat = re.compile(r"\bm[aá][nñ]g[oó]\w*", re.IGNORECASE)
    assert dn.sustituir_alimento({"name": "Tostadas Integrales con Revoltillo de Huevo, Tomate y Mango"}, pat,
                                 "Aguacate") == "Tostadas Integrales con Revoltillo de Huevo, Tomate y Aguacate"
    assert dn.sustituir_alimento({"name": "Tostadas con huevo y mango fresco"}, pat,
                                 "Aguacate") == "Tostadas con huevo y aguacate fresco"


def test_media_porcion_en_singular():
    import graph_orchestrator as g

    def _clean(s):
        return g._SINGULAR_ONE_RE.sub(
            lambda m: f"{m.group(1)} {g._SINGULAR_UNIT_MAP.get(m.group(2).lower(), m.group(2))}", s)
    assert _clean("½ porciones de casabe (15 g)") == "½ porción de casabe (15 g)"
    assert _clean("¾ tazas de avena") == "¾ taza de avena"
    assert _clean("1½ tazas de avena") == "1½ tazas de avena", "más de una: plural"
    assert _clean("2 porciones de casabe") == "2 porciones de casabe"


def test_la_palabra_duplicada_del_swap_se_colapsa():
    import graph_orchestrator as g
    dias = [{"meals": [{"name": "Vasito de yogurt con mango",
                        "ingredients": ["¾ taza de Yogurt natural natural sin azúcar", "60 g de mango"],
                        "ingredients_raw": ["¾ taza de Yogurt natural natural sin azúcar", "60 g de mango"]}]}]
    g._polish_finalize_display(dias)
    m = dias[0]["meals"][0]
    assert m["ingredients"][0] == "¾ taza de Yogurt natural sin azúcar"
    assert m["ingredients_raw"][0] == "¾ taza de Yogurt natural sin azúcar"


class _Cat:
    def category_of(self, s):
        return {"berenjena": "Vegetales", "pechuga de pollo": "Proteinas", "quinoa": "Granos"}.get(str(s).lower(), "")


def test_el_protagonista_es_el_que_abre_el_nombre():
    import identidad_plato as ip
    guiso = {"name": "Guiso ligero de berenjena con queso blanco, plátano maduro asado y edamame",
             "ingredients": ["20 g de Queso blanco", "40 g de Plátano maduro", "30 g de Berenjena", "135 g de edamame"]}
    assert ip._protagonista(guiso) == "30 g de Berenjena"
    tortitas = {"name": "Tortitas de quinoa con berenjena guisada", "ingredients": ["30 g de berenjena", "30 g de Quinoa"]}
    assert ip._protagonista(tortitas) == "30 g de Quinoa", "la berenjena aquí acompaña"
    assert ip._piso_protagonista("Berenjena", _Cat()) == 100
    assert ip._piso_protagonista("Quinoa", _Cat()) == 0, "los granos ya tienen su piso por palabra (en seco)"


def test_el_protagonista_solo_sube_lineas_en_gramos():
    src = _src("identidad_plato.py")
    assert "if linea == prota and _EN_GRAMOS.match(linea):" in src
    import identidad_plato as ip
    assert ip._EN_GRAMOS.match("30 g de Berenjena") and not ip._EN_GRAMOS.match("½ tomate mediano")


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 176 and m.group(2) >= "2026-09-23"
