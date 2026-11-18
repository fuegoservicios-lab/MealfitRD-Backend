"""[P1-PLAN-LOTE-1080 · 2026-11-18] Dos cierres que contradecían lo que el usuario ve (cuenta de prueba, 17-nov):
1. 12:47, anotó desayuno y almuerzo y el coach cerró «No me mandaste el desayuno de hoy; mándame una foto»;
2. 1380 de 1300 kcal (+6 %, «meta cumplida» en la app) y le dijo «superaste tu meta» con consejo para mañana."""
from pathlib import Path

import pytest

from cierre_del_dia import nota_comidas_al_dia, nota_kcal_en_meta

SRC = (Path(__file__).resolve().parents[1] / "tools.py").read_text(encoding="utf-8")


def test_caso_12_47_desayuno_y_almuerzo_en_un_mensaje():
    n = nota_comidas_al_dia(["desayuno", "almuerzo"], 12.78)
    assert "COMIDAS YA ANOTADAS" in n and ": desayuno." in n and "NO digas «No me mandaste…»" in n


@pytest.mark.parametrize("tipos,hora,esperado", [
    (["desayuno", "almuerzo", "merienda"], 19.5, "desayuno, almuerzo y merienda"),
    (["almuerzo", "desayuno"], 16.2, "desayuno y almuerzo"),
    (["Breakfast"], 13.0, ": desayuno."),
])
def test_lista_las_franjas_pasadas_anotadas(tipos, hora, esperado):
    assert esperado in nota_comidas_al_dia(tipos, hora)


@pytest.mark.parametrize("tipos,hora", [
    (["desayuno"], 10.0),          # su hora no ha pasado: el bloque no le pedía nada
    (["cena"], 22.0),              # la cena no «pasa» dentro del día
    (["desayuno"], 2.0),           # madrugada: el bloque no afirma nada
    (["desayuno"], None), ([], 14.0), (None, 14.0),
])
def test_nada_que_aclarar(tipos, hora):
    assert nota_comidas_al_dia(tipos, hora) == ""


def test_hasta_mas_10_por_ciento_cerro_en_su_meta():
    n = nota_kcal_en_meta({"kcal": 1300}, {"kcal": 1380})
    assert "CERRÓ EN SU META" in n and "~1380 de ~1300 kcal" in n and "~80 por encima" in n
    assert "sin «superaste»" in n and "sin consejos para mañana" in n
    assert nota_kcal_en_meta({"kcal": 1300}, {"kcal": 1430}) != ""     # 10 % justo


@pytest.mark.parametrize("meta,lleva", [(1300, 1431), (1300, 1300), (1300, 1200), (0, 900), (None, 900)])
def test_fuera_del_10_por_ciento_o_sin_pasarse_no_aplica(meta, lleva):
    assert nota_kcal_en_meta({"kcal": meta}, {"kcal": lleva}) == ""
    assert nota_kcal_en_meta(None, None) == ""


def test_enganchan_en_el_total_del_dia_y_el_aviso_de_983_sigue_para_mas_del_10():
    assert '__import__("cierre_del_dia").nota_comidas_al_dia(' in SRC
    i = SRC.index('if "kcal: YA SE PASÓ" in _m:')
    bloque = SRC[i:i + 600]
    assert '__import__("cierre_del_dia").nota_kcal_en_meta(_metas, _c)' in bloque
    assert "díselo SIEMPRE en una frase con la cifra" in bloque
