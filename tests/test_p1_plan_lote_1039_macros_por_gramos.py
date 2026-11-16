"""[P1-PLAN-LOTE-1039 · A-41 · 2026-11-16] La proteína de una comida no queda por debajo de la de sus piezas.

Caso real: dos cenas con foto, misma pechuga y mismo bistec. El coach anotó «pechuga de pollo a la plancha (≈170 g)»,
«bistec de res a la plancha (≈170 g)»… y puso el total a ojo: 89 g y, en la cena con MÁS comida, 78 g. Con sus propios
gramos contra el catálogo salen ~102 g y ~92 g (replay en el servidor). El catálogo falso de aquí reproduce, por gramo
servido, lo que devolvió el resolutor real para esas piezas.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import macros_por_gramos as mpg  # noqa: E402

# nombre de fila → (palabra clave en la pieza, proteína por g servido, kcal por g servido)
_FILAS = {
    "Pechuga de pollo": ("pollo", 51.6 / 170, 246.5 / 170),
    "Carne de res": ("res", 49.3 / 170, 322.9 / 170),
    "Zanahoria": ("zanahoria", 1.1 / 120, 53 / 120),
    "Papa": ("papa", 5.4 / 210, 127.7 / 210),
    "Arroz blanco": ("arroz", 5.1 / 200, 260 / 200),
    "Aceite de oliva": ("aceite", 0.0, 9.0),
    "Cerdo": ("cerdo", 28.5 / 100, 229 / 100),
    "Atún en agua": ("atun", 24.7 / 130, 110 / 130),
    "Manzana": ("manzana", 0.3 / 180, 117 / 180),
}


class CatalogoFalso:
    def __init__(self, arroz_en_seco: bool = False):
        self.arroz_en_seco = arroz_en_seco
        self.cadenas = []

    def macros_from_ingredient_string(self, s):
        self.cadenas.append(s)
        m = re.match(r"^(\d+(?:\.\d+)?) g de (.+)$", s)
        if not m:
            return None
        g, nombre = float(m.group(1)), mpg._plano(m.group(2))
        for fila, (clave, p, k) in _FILAS.items():
            if clave in nombre:
                if fila == "Arroz blanco" and self.arroz_en_seco:
                    k, p = 3.6, 0.07
                return {"name": fila, "grams": g, "kcal": round(k * g, 1), "protein": round(p * g, 2),
                        "carbs": 0.0, "fats": 0.0}
        return None


CENA_15 = {"calories": 690, "protein": 89, "carbs": 16, "healthy_fats": 27, "ingredients": [
    "pechuga de pollo a la plancha (≈170 g)", "bistec de res a la plancha (≈170 g)",
    "zanahorias baby cocidas (≈120 g)", "aceite vegetal (≈1 cucharada)"]}
CENA_16 = {"calories": 1250, "protein": 78, "carbs": 99, "healthy_fats": 59, "ingredients": [
    "filete de res asada (≈150 g)", "pechuga de pollo a la plancha (≈120 g)", "puré de papa (≈1 taza, 210 g)",
    "zanahorias baby cocidas con aceite de oliva (≈120 g; 1 cucharadita aceite)",
    "arroz guisado con vegetales (≈1 taza, 200 g)", "ensalada de papa y zanahoria con mayonesa (≈1 taza, 200 g)"]}


# ───────────────────────────────────────────────────────────── el caso que lo pidió
def test_las_dos_cenas_reales_quedan_con_la_proteina_de_sus_piezas():
    a = mpg.reconciliar(CENA_15, CatalogoFalso())
    b = mpg.reconciliar(CENA_16, CatalogoFalso())
    assert a["protein"] == 102 and b["protein"] == 92
    assert a["motivo"] == b["motivo"] == "suelo_de_proteina"
    # solo sube la proteína (y sus 4 kcal/g): carbos y grasa del modelo no se tocan
    assert b["carbs"] == 99 and b["healthy_fats"] == 59 and b["calories"] == 1250 + 4 * (92 - 78)


def test_la_ensalada_con_mayonesa_y_el_aceite_de_la_guarnicion_no_cuentan_como_pieza():
    db = CatalogoFalso()
    r = mpg.resolver(CENA_16["ingredients"], db)
    filas = {c["nombre"][:20]: c["fila"] for c in r["componentes"]}
    assert filas["ensalada de papa y z"] is None, "plato compuesto: su fila sería solo una de sus cosas"
    assert filas["zanahorias baby coci"] == "Zanahoria", "«con aceite de oliva» es la guarnición, no la pieza"
    assert "120 g de zanahorias baby cocidas" in db.cadenas


def test_nunca_baja_la_proteina_ni_toca_lo_que_ya_cuadra():
    alta = dict(CENA_15, protein=130)
    assert mpg.reconciliar(alta, CatalogoFalso()) is None
    cerca = dict(CENA_15, protein=95)       # 95 × 1,10 > 102: dentro del margen
    assert mpg.reconciliar(cerca, CatalogoFalso()) is None


def test_un_cambio_de_menos_de_3_g_es_ruido():
    fruta = {"calories": 95, "protein": 0, "carbs": 25, "healthy_fats": 0, "ingredients": ["1 manzana verde (180 g)"]}
    assert mpg.reconciliar(fruta, CatalogoFalso()) is None


# ───────────────────────────────────────────────────────────── las clases de error que encontró el replay
@pytest.mark.parametrize("ing", [
    "manteca de cerdo usada para freír (14 g)",
    "jamón cocido de cerdo (30 g)",
    "agua de coco (250 g)",
    "pudín de pan (180 g)",
    "2 sándwiches de pollo (360 g)",
    "empanadas de pollo, 2 unidades (240 g)",
    "Chofán de pollo con huevo (350 g)",
    "Pica pollo frito (180 g)",
    "1 presa de pollo frito (150 g)",
    "bistec de res guisado con hueso (150 g)",
    "cerdo asado con cuerito y grasa (150 g)",
    "chuleta de cerdo frita (120 g)",
    "5 albóndigas de res fritas (150 g)",
    "3/4 de barra de chocolate 88% cacao (60 g)",
    "salsa de carne (30 g)",
])
def test_lo_derivado_compuesto_o_con_hueso_no_pone_suelo(ing):
    c = mpg.componente(ing)
    assert c["compuesto"] is True, ing


@pytest.mark.parametrize("ing,pieza", [
    ("atún en aceite escurrido (120 g)", "atún"),
    ("pollo desmenuzado en salsa blanca cremosa (140 g)", "pollo desmenuzado"),
    ("pollo asado sin piel (150 g)", "pollo asado sin piel"),
])
def test_la_cola_con_en_se_corta_y_sin_piel_si_cuenta(ing, pieza):
    c = mpg.componente(ing)
    assert c["pieza"] == pieza and c["compuesto"] is False


def test_el_atun_en_aceite_no_se_resuelve_como_aceite():
    r = mpg.resolver(["atún en aceite escurrido (120 g)"], CatalogoFalso())
    assert r["componentes"][0]["fila"] == "Atún en agua"


def test_una_fila_en_otra_base_no_entra_arroz_contado_en_seco():
    comida = {"calories": 400, "protein": 5, "carbs": 80, "healthy_fats": 2,
              "ingredients": ["arroz con fideos cocido (200 g)"]}
    r = mpg.resolver(comida["ingredients"], CatalogoFalso(arroz_en_seco=True))
    assert r["componentes"][0]["fila"] is None, "720 kcal en 200 g servidos: la fila está en seco"


def test_si_las_piezas_pesan_mas_energia_que_la_comida_no_se_toca():
    rota = dict(CENA_15, calories=300)      # 622 kcal de piezas en una comida «de 300»: algo está mal resuelto
    assert mpg.reconciliar(rota, CatalogoFalso()) is None


@pytest.mark.parametrize("ing,esperado", [
    ("pechuga de pollo a la plancha (≈170 g)", (170.0, "170 g de pechuga de pollo a la plancha")),
    ("puré de papa (≈1 taza, 210 g)", (210.0, "210 g de puré de papa")),
    ("75 g de Aguacate", (75.0, "75 g de Aguacate")),
    ("aceite vegetal (≈1 cucharada)", (None, "1 cucharada de aceite vegetal")),
    ("Huevo frito", (None, "Huevo frito")),
])
def test_lee_los_gramos_del_formato_del_coach(ing, esperado):
    c = mpg.componente(ing)
    assert (c["gramos"], c["cadena"]) == esperado


def test_sin_gramos_explicitos_la_pieza_no_pone_suelo():
    r = mpg.resolver(["aceite vegetal (≈1 cucharada)", "Huevo frito"], CatalogoFalso())
    assert r["simples"] == 0


# ───────────────────────────────────────────────────────────── al registrar
def test_al_registrar_devuelve_las_cifras_nuevas_y_una_nota_para_el_modelo(monkeypatch):
    escritas = []
    import db_core
    monkeypatch.setattr(db_core, "execute_sql_write", lambda q, p, **k: escritas.append((q, p)))
    cal, p, c, f, nota = mpg.al_registrar("u-1", 1250, 78, 99, 59, CENA_16["ingredients"], db=CatalogoFalso())
    assert (cal, p, c, f) == (1306, 92, 99, 59)
    assert "78 → 92 g" in nota and "usa ESTAS cifras" in nota
    assert len(escritas) == 1 and "macros_por_gramos" in escritas[0][0]
    meta = escritas[0][1][1]
    assert "pollo" not in meta and "res" not in meta, "la huella no lleva nombres de comida"


def test_al_registrar_sin_nada_que_corregir_o_con_fallo_deja_todo_igual():
    assert mpg.al_registrar("u-1", 690, 130, 16, 27, CENA_15["ingredients"], db=CatalogoFalso()) == (690, 130, 16, 27, "")

    class Roto:
        def macros_from_ingredient_string(self, s):
            raise RuntimeError("catálogo caído")
    assert mpg.al_registrar("u-1", 690, 89, 16, 27, CENA_15["ingredients"], db=Roto()) == (690, 89, 16, 27, "")


def test_el_knob_lo_apaga(monkeypatch):
    monkeypatch.setenv("MEALFIT_MACROS_POR_GRAMOS", "0")
    assert mpg.reconciliar(CENA_16, CatalogoFalso()) is None


def test_tools_lo_aplica_antes_de_guardar_salvo_etiqueta_o_pote():
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    i = src.index('__import__("macros_por_gramos").al_registrar(')
    assert "if food_facts is None and not _pote and ingredients:" in src[i - 200:i]
    assert i < src.index("result = db_log_consumed_meal(", i), "la corrección va ANTES de guardar"
    assert "msg += _nota_gramos" in src


def test_el_modulo_no_lleva_caracteres_de_control():
    """Un `\\b` escrito por un script de parcheo acabó como retroceso (\\x08): la expresión dejó de coincidir y
    «manteca de cerdo» volvió a contar como carne, con todos los tests de sintaxis en verde."""
    crudo = (_BACKEND / "macros_por_gramos.py").read_text(encoding="utf-8")
    assert not any(ord(ch) < 32 and ch not in "\n\r\t" for ch in crudo)


def test_el_replay_solo_lee():
    fuente = (_BACKEND / "scripts" / "coach_battery" / "replay_macros.py").read_text(encoding="utf-8").upper()
    for verbo in ("INSERT ", "UPDATE ", "DELETE ", "OPENAI", "GEMINI"):
        assert verbo not in fuente


@pytest.mark.parametrize("linea", [
    "170 g de pechuga de pollo a la parrilla", "175 g de pechuga de pollo cocinada", "80 g de pollo al horno",
    "140 g de salmón sellado con limón", "120 g de bistec de res encebollado", "150 g de pollo rostizado",
    "150 g de cerdo a la brasa",
])
def test_toda_coccion_comun_cuenta_la_carne_como_cocida(linea):
    """[A-41] La misma pechuga daba 38 g «a la parrilla» y 52 g «a la plancha»: una contada cruda, la otra cocida."""
    import cocido_en_catalogo as cec
    assert cec._PROT_COCIDO_RX.search(mpg._plano(linea))


# ───────────────────────────────────────────────────────────── paso 3: gramos editables en la ficha
def test_que_renglones_son_ajustables():
    aj = mpg.ajustables(CENA_16["ingredients"], CatalogoFalso())
    assert aj == {0: 150, 1: 120, 2: 210, 3: 120, 4: 200}, "la ensalada con mayonesa (5) es compuesta: no"


def test_subir_el_pollo_mueve_los_totales_lo_que_aporta_esa_pieza():
    n = mpg.ajustar_gramos(CENA_16, 1, 170, CatalogoFalso())
    assert n["ingredients"][1] == "pechuga de pollo a la plancha (≈170 g)"
    assert n["protein"] == round(78 + 51.6 - 120 * 51.6 / 170)
    assert n["calories"] == round(1250 + 246.5 - 120 * 246.5 / 170)
    assert n["ingredients"][0] == CENA_16["ingredients"][0], "los demás renglones no se tocan"


def test_el_numero_que_cambia_es_el_de_los_gramos_no_el_de_las_tazas():
    n = mpg.ajustar_gramos(CENA_16, 2, 250, CatalogoFalso())
    assert n["ingredients"][2] == "puré de papa (≈1 taza, 250 g)"
    assert mpg._con_gramos("75 g de Aguacate", 100) == "100 g de Aguacate"


def test_lo_no_ajustable_o_fuera_de_rango_no_cambia_nada():
    assert mpg.ajustar_gramos(CENA_16, 5, 300, CatalogoFalso()) is None        # ensalada compuesta
    assert mpg.ajustar_gramos(CENA_16, 9, 300, CatalogoFalso()) is None        # no existe
    bajo = mpg.ajustar_gramos(CENA_16, 1, 0, CatalogoFalso())
    assert bajo["gramos"] == mpg.GRAMOS_MIN and bajo["protein"] >= 0


def test_la_ficha_marca_los_renglones_ajustables():
    from ficha_comida import desglose_de_ingredientes
    d = desglose_de_ingredientes(CENA_16["ingredients"], 1250, CatalogoFalso())
    assert [l.get("ajustable_g") for l in d["lineas"]] == [150, 120, 210, 120, 200, None]


class _Fila(dict):
    pass


def _endpoint(monkeypatch, fila, actualizar_ok=True):
    import routers.diary as diary
    import nutrition_db
    import db_facts
    vistas = {"sql": [], "update": []}

    def consulta(sql, params=None, **k):
        vistas["sql"].append((sql, params))
        return dict(fila) if fila and params and params[1] == "u-1" else None
    monkeypatch.setattr(diary, "execute_sql_query", consulta)
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", CatalogoFalso)
    monkeypatch.setattr(db_facts, "update_consumed_meal",
                        lambda uid, mid, **kw: (vistas["update"].append((uid, mid, kw)), "ok" if actualizar_ok else None)[1])
    return diary, vistas


MEAL_ID = "8a2b1c3d-1111-4222-8333-944455556666"


def test_el_endpoint_ajusta_y_devuelve_la_ficha_nueva(monkeypatch):
    diary, vistas = _endpoint(monkeypatch, {**CENA_16, "id": MEAL_ID, "meal_name": "Cena"})
    r = diary.api_ajustar_gramos_de_una_pieza(MEAL_ID, diary.AjusteGramosRequest(indice=1, gramos=170), "u-1")
    assert r["success"] and r["meal"]["protein"] == 93
    assert "AND user_id = %s" in vistas["sql"][0][0] and vistas["sql"][0][1] == (MEAL_ID, "u-1")
    uid, mid, kw = vistas["update"][0]
    assert (uid, mid) == ("u-1", MEAL_ID) and kw["ingredients"][1].endswith("(≈170 g)")


def test_el_endpoint_404_si_es_de_otro_y_409_si_no_es_ajustable(monkeypatch):
    from fastapi import HTTPException
    diary, _ = _endpoint(monkeypatch, {**CENA_16, "id": MEAL_ID})
    with pytest.raises(HTTPException) as e:
        diary.api_ajustar_gramos_de_una_pieza(MEAL_ID, diary.AjusteGramosRequest(indice=1, gramos=170), "otro")
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        diary.api_ajustar_gramos_de_una_pieza(MEAL_ID, diary.AjusteGramosRequest(indice=5, gramos=170), "u-1")
    assert e.value.status_code == 409


def test_el_endpoint_no_cobra_cuota_ni_llama_a_un_modelo():
    src = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    i = src.index('@router.post("/consumed/{meal_id}/gramos")')
    cuerpo = src[i:src.index("\n@router.", i + 10)]
    assert "Depends(_AJUSTE_GRAMOS_LIMITER)" in cuerpo
    for prohibido in ("verify_api_quota", "invoke(", "openai", "gemini"):
        assert prohibido not in cuerpo


# ───────────────────────────────────────────────────────────── paso 2: porciones habituales como referencia
def test_porciones_habituales_son_la_mediana_de_las_piezas_proteicas():
    filas = [{"ingredients": CENA_15["ingredients"]}, {"ingredients": CENA_16["ingredients"]},
             {"ingredients": ["pechuga de pollo a la plancha (≈160 g)", "puré de papa (210 g)"]}]
    p = mpg.porciones_habituales("u-1", CatalogoFalso(), filas=filas)
    assert ("Pechuga de pollo", 160, 3) in p and ("Carne de res", 160, 2) in p
    assert all(f != "Papa" and f != "Zanahoria" for f, _, _ in p), "solo piezas proteicas"


def test_una_sola_vez_no_es_habitual_y_el_invitado_no_tiene():
    assert mpg.porciones_habituales("u-1", CatalogoFalso(), filas=[{"ingredients": CENA_15["ingredients"]}]) == []
    assert mpg.porciones_habituales("guest", CatalogoFalso(), filas=[{"ingredients": CENA_15["ingredients"]}] * 3) == []


def test_la_referencia_es_una_referencia_no_una_regla():
    t = mpg.referencia_de_porciones([("Pechuga de pollo", 160, 3)])
    assert "pechuga de pollo ≈160 g (3 registros)" in t and "no es una regla" in t
    assert "si se ve claramente más grande o más pequeña, estima lo que ves" in t
    assert mpg.referencia_de_porciones([]) == ""


def test_solo_entra_en_los_turnos_con_foto_de_plato():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    i = src.index("_largo_antes_foto = len(system_prompt)")
    tramo = src[i:i + 800]
    # la línea literal la fijan P3-I18N-PROMPT-VISION-CLIENTE y P1-NEVERA-OPCIONAL: una vez, tras las dos ramas
    assert "system_prompt += build_vision_context(vision, nevera_activa=_nevera_on)" in tramo
    assert 'if "FOTO DE PLATO (PRIORIDAD MÁXIMA)" in system_prompt[_largo_antes_foto:]:' in tramo
    assert "referencia_de_porciones(_mpg.porciones_habituales(user_id))" in tramo
    from prompts.chat_agent import _PLATO_INSTRUCCION
    assert _PLATO_INSTRUCCION.startswith("FOTO DE PLATO (PRIORIDAD MÁXIMA)"), "el marcador que busca agent.py"
