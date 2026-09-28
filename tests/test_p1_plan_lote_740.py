# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-740 · 2026-09-28] (G67) La línea base de micros del catálogo corre siempre, y la «fila-cascarón» no lo era.

G67 decía dos cosas. (1) «Sazón con culantro y achiote» (0 kcal, 0 macros, 17 000 mg de sodio, `usda`, fdc 172242)
«parece verificada y no lo es». Comprobado contra la API pública de USDA el 28-sep: FDC 172242 es «Seasoning mix, dry,
sazon, coriander & annatto» (SR Legacy) y trae EXACTAMENTE 0 kcal, 0 g de proteína, grasa, carbohidrato y fibra y
17 000 mg de sodio. El `fdc_id` es una afirmación correcta: la fila es fiel a su fuente, como «Sal». (2) La línea base de
`test_p2_micros_beta_backfill.py` saltaba entera sin DB, y fuera de FastAPI el catálogo sale vacío: nunca corría. Ahora usa
la DB si hay filas y, si no, una foto versionada del catálogo (`tests/fixtures/catalogo_micros_2026_09_28.json`, SELECT de
solo lectura sobre prod, 349 filas). La foto se refresca con el mismo SELECT; el test dice de dónde leyó.

tooltip-anchor: P1-PLAN-LOTE-740
"""
import importlib


def _base():
    return importlib.import_module("test_p2_micros_beta_backfill")


def test_la_linea_base_no_salta_sin_db():
    filas, origen = _base().cargar_filas()
    assert origen in ("db", "foto")
    assert len(filas) >= 300


def test_el_sazon_es_fiel_a_fdc_172242():
    filas, _ = _base().cargar_filas()
    sazon = next(r for r in filas if r["name"] == "Sazón con culantro y achiote")
    assert sazon["fdc_id"] == 172242 and sazon["nutrition_source"] == "usda"
    assert float(sazon["kcal_per_100g"]) == 0 and float(sazon["sodium_mg_per_100g"]) == 17000
    assert _base()._FORMA_CASCARON["Sazón con culantro y achiote"].startswith("CORRECTA")
