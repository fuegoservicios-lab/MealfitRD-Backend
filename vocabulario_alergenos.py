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

# [P1-PLAN-LOTE-796 · 2026-09-28] Quesos por su nombre suelto y dulces de leche: alias del catálogo («gouda», «cheddar»,
# «provolone», «edam», «quesito panela», «cajeta» y «manjar blanco» de Arequipe, «suero atollabuey») que la IA escribe sin
# la palabra «queso» o «leche» y el escáner no veía. Van a lácteos Y a lactosa (la clase lactosa ya cuenta «queso»).
QUESOS_Y_DULCES_DE_LECHE = ("gouda", "cheddar", "provolone", "edam", "manchego", "emmental", "gruyere", "mascarpone",
                            "feta", "brie", "camembert", "burrata", "pecorino", "quesito", "manjar blanco", "cajeta",
                            "atollabuey")
LACTOSA_EXTRA = ("cottage", "biscuit", *QUESOS_Y_DULCES_DE_LECHE)  # el biscuit (US) se hace con mantequilla y suero

EXTRA = {
    "gluten": ("granola", "teriyaki", "wrap", "espelta", "kamut", "farro", "triticale", "semolina", "pita", "croissant",
               "cruasan", "brioche", "yaniqueque", "crepa", "crepe",
               # [P1-PLAN-LOTE-268] «Harina de Negrito» = crema de trigo (farina): fila del catálogo que se le
               # OFRECÍA al celíaco; la cazó el revisor de IA, no el escáner (batería final, mariscos + gluten).
               "harina de negrito", "negrito", "farina",
               # [P1-PLAN-LOTE-796] alias del catálogo que el resolutor acepta (Harina de trigo, Bulgur, Galletas de soda,
               # Pan sobao, Panecillos, Masa para pie, Bacalaítos) y nombres de trigo de US/PR/CO. «cuchuco» y «cracker»
               # llevan excusa de base (de maíz / de arroz) en `graph_orchestrator._ALLERGEN_TERM_BASE_EXCUSES`.
               "harina blanca", "harina de todo uso", "harina todo uso", "harina para todo uso", "harina multiuso",
               "harina quaker", "burgol", "bulghur", "cuchuco", "sobao", "muffin", "biscuit", "hotcake", "gravy",
               "corteza de pie", "cracker", "saltine", "quipe", "kipe", "tipili"),  # quipe/tipilí (DO): bulgur
    "huevo": ("revoltillo", "omelette", "omelet", "frittata", "huevito", "flan", "natilla",
              # [P1-PLAN-LOTE-268] con mayonesa: «Ensalada de macarrones» (alias «ensalada de pasta con mayonesa»)
              "ensalada de macarrones", "ensalada de coditos", "ensalada rusa",
              "cesar",  # [P1-PLAN-LOTE-270] el aderezo César lleva yema
              "ajoaceite"),  # [P1-PLAN-LOTE-796] alias de la fila «Alioli»
    "lacteos": ("lactosuero", "cesar", "biscuit", *QUESOS_Y_DULCES_DE_LECHE),  # [P1-PLAN-LOTE-270] el aderezo César lleva parmesano
    # [P1-PLAN-LOTE-796] «frutos secos» / «fruto seco» es como la IA escribe la mezcla («10 g de frutos secos mixtos sin
    # sal», batería rdfinal) y el alias de la fila «Nueces mixtas»: sin el nombre GENÉRICO el pool del esqueleto, las
    # sugerencias del prompt y el backstop se lo daban al alérgico. Más los nombres locales: cajuil (DO), pajuil (PR),
    # cajú, pistache (MX), pacana (ES), y los platos cuyo nombre esconde el fruto seco (nogada, romesco, ajoblanco).
    "frutos secos": ("macadamia", "pecana", "castana", "frutos secos", "fruto seco", "cajuil", "pajuil", "caju",
                     "pistache", "pacana", "nogada", "romesco", "ajoblanco"),
    "soya": ("tamari", "shoyu"),
    "sesamo": ("tahin", "zaatar", "humus"),  # [P1-PLAN-LOTE-796] «humus»: grafía del alias de la fila «Hummus»
}

# Términos que se BUSCAN en el plato cuando la clase ya está declarada, pero que NO resuelven una declaración a esa clase.
# La resolución declaración→clase es bidireccional por palabra completa: con «salsa de soya» en gluten, quien marca el
# chip «Soya» se quedaba sin pan, pasta ni avena (lo cazó `test_paridad_filtro_vs_escaner_canonico[Soya]`). Por lo mismo
# no entran «tortilla de papas / española» (quien rechaza «papas» o «tortilla» perdería el huevo). Los consume
# `graph_orchestrator._expand_allergy_declarations`, `_ALLERGEN_GLUTEN_TERM_SET` (la excusa «sin gluten») y el filtro de
# catálogo de `constants`.
OCULTOS = {
    "gluten": ("salsa de soya", "salsa de soja", "frituras de bacalao",  # [P1-PLAN-LOTE-796] ver abajo
               "salsa cremosa de salchicha"),  # [P1-PLAN-LOTE-796] alias de «Salsa de salchicha» (gravy con harina)
    # [P1-PLAN-LOTE-796] la tortilla de papas ES huevo; se busca sin que «papas» o «tortilla» declaren la clase.
    "huevo": ("tortilla espanola", "tortilla de papa", "tortilla de patata"),
    # [P1-PLAN-LOTE-796] alias de «Bacalaítos» (masa de trigo, arriba) y de «Vieira»: como compuestos resolverían
    # «bacalao» a gluten y «concha» (el pan) a mariscos, así que se BUSCAN sin declarar la clase.
    "mariscos": ("concha de abanico",),
}
