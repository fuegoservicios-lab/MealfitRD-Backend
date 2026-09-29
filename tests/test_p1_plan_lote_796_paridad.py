"""[P1-PLAN-LOTE-796 · paridad · 2026-09-29] Lo que el registry DECLARA de una fila tiene que coincidir con lo que el
escáner DECIDE, también para los platos que esconden la clase (`vocabulario_alergenos.OCULTOS`).

El gate del 29-sep cayó en `test_las_dos_capas_de_lacteo_coinciden` (lee el catálogo real): el 796 enseñó al escáner
que «Wafles», «Pan de maíz» y «Aderezo ranch» llevan leche, y `allergen_classes_for` —que alimenta el snapshot del
registry y la identidad dietaria de cada fila del catálogo (`food_identity`)— no buscaba esos términos. Este test fija
la paridad sin necesitar la base de datos.
"""
import pytest

import dish_registry as DR


@pytest.mark.parametrize("nombre", ["Wafles", "Pan de maíz", "Aderezo ranch"])
def test_la_fila_que_esconde_leche_se_declara_lactea(nombre):
    assert "lacteos" in DR.allergen_classes_for([nombre]), nombre


@pytest.mark.parametrize("nombre,clase", [("Pan de maíz", "huevo"), ("Aderezo ranch", "huevo"), ("Wafles", "huevo"),
                                          ("Nueces mixtas", "mani")])
def test_los_demas_ocultos_tambien_se_declaran(nombre, clase):
    assert clase in DR.allergen_classes_for([nombre]), (nombre, clase)


@pytest.mark.parametrize("nombre", ["Leche de coco", "Mantequilla de maní", "Yogur de coco", "Tortilla de maíz",
                                    "Arepa de maíz", "Maíz dulce"])
def test_sin_falsos_lacteos(nombre):
    assert "lacteos" not in DR.allergen_classes_for([nombre]), nombre
