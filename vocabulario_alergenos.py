# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-252 · 2026-09-25] Alimentos de las demás clases de alergia que el escáner no reconocía.

Misma sonda que el lote 250 (pescados y mariscos), clase por clase: con el chip marcado, estas líneas pasaban limpias
por el backstop de alérgenos, el de rechazos y, en huevo/lácteos, el de dieta vegana. Dos eran filas del catálogo que la
IA puede servir: «Granola» (avena, y la avena ya cuenta como gluten) y «Natilla» (lleva yema); la salsa de soya lleva
TRIGO (la fuente escondida de gluten de manual; «sin gluten» la excusa, el tamari es de soya) y «wrap» es alias de
las tortillas de trigo del catálogo.

Lo consumen `graph_orchestrator._ALLERGEN_SYNONYMS` (y `_DIET_EGG_TERMS` / `_DIET_DAIRY_TERMS`, cuya paridad exige
`test_paridad_dieta_alergeno_bidireccional`) y el filtro de catálogo `constants._get_fast_filtered_catalogs` (cuya
paridad exige `test_paridad_filtro_vs_escaner_canonico`). Formas sin tilde y en singular (el escáner compara sin acentos,
con frontera de palabra y plural español). Fuera a propósito «harina» sola: 46 líneas reales son «harina de maíz».
tooltip-anchor: P1-PLAN-LOTE-252-VOCABULARIO-ALERGENOS
"""

EXTRA = {
    "gluten": ("granola", "teriyaki", "wrap", "espelta", "kamut", "farro", "triticale", "semolina", "pita", "croissant",
               "cruasan", "brioche", "yaniqueque", "crepa", "crepe",
               # [P1-PLAN-LOTE-268] «Harina de Negrito» = crema de trigo (farina): fila del catálogo que se le
               # OFRECÍA al celíaco; la cazó el revisor de IA, no el escáner (batería final, mariscos + gluten).
               "harina de negrito", "negrito", "farina"),
    "huevo": ("revoltillo", "omelette", "omelet", "frittata", "huevito", "flan", "natilla",
              # [P1-PLAN-LOTE-268] con mayonesa: «Ensalada de macarrones» (alias «ensalada de pasta con mayonesa»)
              "ensalada de macarrones", "ensalada de coditos", "ensalada rusa",
              "cesar"),  # [P1-PLAN-LOTE-270] el aderezo César lleva yema
    "lacteos": ("lactosuero", "cesar"),  # [P1-PLAN-LOTE-270] el aderezo César lleva parmesano
    "frutos secos": ("macadamia", "pecana", "castana"),
    "soya": ("tamari", "shoyu"),
    "sesamo": ("tahin", "zaatar"),
}

# Términos que se BUSCAN en el plato cuando la clase ya está declarada, pero que NO resuelven una declaración a esa clase.
# La resolución declaración→clase es bidireccional por palabra completa: con «salsa de soya» en gluten, quien marca el
# chip «Soya» se quedaba sin pan, pasta ni avena (lo cazó `test_paridad_filtro_vs_escaner_canonico[Soya]`). Por lo mismo
# no entran «tortilla de papas / española» (quien rechaza «papas» o «tortilla» perdería el huevo). Los consume
# `graph_orchestrator._expand_allergy_declarations`, `_ALLERGEN_GLUTEN_TERM_SET` (la excusa «sin gluten») y el filtro de
# catálogo de `constants`.
OCULTOS = {
    "gluten": ("salsa de soya", "salsa de soja"),
}
