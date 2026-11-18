"""[P1-PLAN-LOTE-1076 · 2026-11-17] Dos ajustes del coach pedidos por el dueño tras su día del 17-nov:
1. de noche (21:00+) no se cierra con «cuando cenes, cuéntame»: se pregunta en pasado («¿y la cena?»);
2. con la cena anotada, la proteína muy corta (68 de 134 g) se dice una vez, sin regañar, con una idea para mañana."""
from pathlib import Path

from cierre_del_dia import nota_proteina_corta

METAS = {"kcal": 2050, "protein_g": 134}


def test_caso_del_duenno_68_de_134_con_la_cena_a_las_22():
    n = nota_proteina_corta(METAS, {"protein_g": 68}, ["almuerzo", "merienda", "cena"], 22.43)
    assert "~68 de ~134 g" in n and "~66 g" in n
    assert "sin regañar" in n and "SIN proponer comer más ahora" in n and "mañana" in n


def test_no_aplica_sin_cena_temprano_con_proteina_suficiente_o_sin_meta():
    assert nota_proteina_corta(METAS, {"protein_g": 68}, ["almuerzo", "merienda"], 22.4) == ""   # sin cena
    assert nota_proteina_corta(METAS, {"protein_g": 68}, ["cena"], 18.5) == ""                   # aún no es de noche
    assert nota_proteina_corta(METAS, {"protein_g": 90}, ["cena"], 22.0) == ""                   # 67 % ≥ 60 %
    assert nota_proteina_corta({"protein_g": None}, {"protein_g": 10}, ["cena"], 22.0) == ""
    assert nota_proteina_corta(None, None, None, None) == ""
    assert nota_proteina_corta(METAS, {"protein_g": 30}, ["cena"], 1.5) != ""                    # madrugada = mismo día


def test_engancha_en_el_total_del_dia_y_de_noche_la_cena_va_en_pasado():
    src = (Path(__file__).resolve().parents[1] / "tools.py").read_text(encoding="utf-8")
    assert '__import__("cierre_del_dia").nota_proteina_corta(' in src
    i = src.index('if _prox == "cena" and (float(_ahora) >= 21.0 or float(_ahora) < 4.0):')
    assert "«¿y la cena? Si ya cenaste, cuéntame qué fue y lo anoto»" in src[i:i + 200]
