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
# [P1-PLAN-LOTE-796 · revisión] Lácteos por el nombre del PREPARADO: el quesillo (flan de DO / queso Oaxaca de MX), el
# kumis (CO), el capuchino y el latte, la bechamel. Sin «queso» ni «leche» en la línea, el escáner no los veía.
LACTEOS_PREPARADOS = ("quesillo", "kumis", "capuchino", "cappuccino", "latte", "bechamel")

# [P1-PLAN-LOTE-796 · ronda 3 · 2026-09-28] Los ALIAS del lote (nombres locales, préstamos, alias del catálogo, quesos por
# su nombre suelto) separados de los nombres de siempre de `EXTRA`: el escáner los busca igual (entran en `EXTRA`), pero
# la línea dura del prompt, que corta por alergia, los nombra DETRÁS de los de siempre — cortos como «brie», «kipe» o
# «caju», desplazaban a «mayonesa», «pistacho» o «helado» (`prompts.day_generator.allergy_hard_line`).
# tooltip-anchor: P1-PLAN-LOTE-796-ALIAS-AL-FINAL
ALIAS = {
    # alias del catálogo que el resolutor acepta (Harina de trigo, Bulgur, Galletas de soda, Pan sobao, Panecillos,
    # Masa para pie, Bacalaítos) y nombres de trigo de US/PR/CO. «cuchuco» y «cracker» llevan excusa de base (de maíz /
    # de arroz) en `graph_orchestrator._ALLERGEN_TERM_BASE_EXCUSES`. [revisión] préstamos que en DO/MX/CO/PR se escriben
    # así («waffles» y «spaghetti» son alias de Wafles y Espaguetis) y el pastelito dominicano (masa de trigo; el de yuca
    # lo excusa la base). [ronda 3] el pan rallado japonés, la masa de hojaldre, la magdalena y la galletita (el
    # diminutivo no lo alcanza el plural de «galleta»).
    "gluten": ("harina blanca", "harina de todo uso", "harina todo uso", "harina para todo uso", "harina multiuso",
               "harina quaker", "burgol", "bulghur", "cuchuco", "sobao", "muffin", "biscuit", "hotcake", "gravy",
               "corteza de pie", "cracker", "saltine", "quipe", "kipe", "tipili",  # quipe/tipilí (DO): bulgur
               "waffle", "spaghetti", "pancake", "hot cake", "hot-cake", "panqueca", "pastelito",
               "panko", "hojaldre", "magdalena", "galletita"),
    "huevo": ("ajoaceite",),  # alias de la fila «Alioli»
    # el biscuit (US): mantequilla y suero
    "lacteos": ("biscuit", *QUESOS_Y_DULCES_DE_LECHE, *LACTEOS_PREPARADOS),
    # nombres locales: cajuil (DO), pajuil (PR), cajú, pistache (MX), pacana (ES), y los platos cuyo nombre esconde el
    # fruto seco (nogada, romesco, ajoblanco); [revisión] el alfóncigo, el pistacho del Reglamento UE 1169/2011 (anexo II)
    "frutos secos": ("frutos secos", "fruto seco", "cajuil", "pajuil", "caju", "pistache", "pacana", "nogada", "romesco",
                     "ajoblanco", "alfoncigo"),
    # «humus»: grafía del alias de la fila «Hummus»; [revisión] el za'atar con su apóstrofo (el escáner no quita la
    # puntuación y «zaatar» no casaba con «za'atar»)
    "sesamo": ("humus", "za'atar", "za’atar", "za atar"),
    # [revisión] «manises» (el plural caribeño: el patrón da manis/manies) y el pollo encacahuatado
    "mani": ("manises", "encacahuatado", "encacahuatada"),
}
LACTOSA_EXTRA = ("cottage", *ALIAS["lacteos"])

EXTRA = {
    "gluten": ("granola", "teriyaki", "wrap", "espelta", "kamut", "farro", "triticale", "semolina", "pita", "croissant",
               "cruasan", "brioche", "yaniqueque", "crepa", "crepe",
               # [P1-PLAN-LOTE-268] «Harina de Negrito» = crema de trigo (farina): fila del catálogo que se le
               # OFRECÍA al celíaco; la cazó el revisor de IA, no el escáner (batería final, mariscos + gluten).
               "harina de negrito", "negrito", "farina", *ALIAS["gluten"]),  # [P1-PLAN-LOTE-796]
    "huevo": ("revoltillo", "omelette", "omelet", "frittata", "huevito", "flan", "natilla",
              # [P1-PLAN-LOTE-268] con mayonesa: «Ensalada de macarrones» (alias «ensalada de pasta con mayonesa»)
              "ensalada de macarrones", "ensalada de coditos", "ensalada rusa",
              "cesar",  # [P1-PLAN-LOTE-270] el aderezo César lleva yema
              *ALIAS["huevo"]),
    "lacteos": ("lactosuero", "cesar", *ALIAS["lacteos"]),  # [P1-PLAN-LOTE-270] el César lleva parmesano
    # [P1-PLAN-LOTE-796] «frutos secos» / «fruto seco» es como la IA escribe la mezcla («10 g de frutos secos mixtos sin
    # sal», batería rdfinal) y el alias de la fila «Nueces mixtas»: sin el nombre GENÉRICO el pool del esqueleto, las
    # sugerencias del prompt y el backstop se lo daban al alérgico.
    "frutos secos": ("macadamia", "pecana", "castana", *ALIAS["frutos secos"]),
    "soya": ("tamari", "shoyu"),
    "sesamo": ("tahin", "zaatar", *ALIAS["sesamo"]),
    "mani": ALIAS["mani"],
}

# Términos que se BUSCAN en el plato cuando la clase ya está declarada, pero que NO resuelven una declaración a esa clase.
# La resolución declaración→clase es bidireccional por palabra completa: con «salsa de soya» en gluten, quien marca el
# chip «Soya» se quedaba sin pan, pasta ni avena (lo cazó `test_paridad_filtro_vs_escaner_canonico[Soya]`). Por lo mismo
# no entran «tortilla de papas / española» (quien rechaza «papas» o «tortilla» perdería el huevo). Los consume
# `graph_orchestrator._expand_allergy_declarations`, `_ALLERGEN_GLUTEN_TERM_SET` (la excusa «sin gluten») y el filtro de
# catálogo de `constants`; desde la revisión del 796 también `dish_registry._clases_ocultas` (constituyentes y nombre de
# la plantilla) y `termino_compartido` (la excusa «sin gluten» no absuelve un término que otra clase declarada prohíbe).
# [P1-PLAN-LOTE-796 · revisión] Los PLATOS que esconden el alérgeno en la preparación (nadie los declara como alergia):
# el empanizado/rebozado/milanesa/croqueta lleva pan rallado o harina Y huevo; la croqueta y la pizza, lácteo; el MOLE, la
# pasta con cacahuate, ajonjolí, almendra y pan o galleta (el «de olla» no: lo excusa `EXCUSAS_DE_BASE`); la salsa
# macha, cacahuate y ajonjolí. Y la MEZCLA de frutos secos para el alérgico al maní: «Nueces mixtas» es «mixed nuts, dry
# roasted, with peanuts» (su alta de catálogo) y las mezclas de DO/US/ES suelen llevar maní — el fruto seco suelto
# («almendras», «nueces») no entra aquí. La granola de la fila (FDC 171646, «granola, homemade»: 11 mg de vitamina E
# por 100 g, la firma de la almendra y la semilla) y la comercial llevan fruto seco.
_EMPANIZADOS = ("croqueta", "milanesa", "empanado", "empanizado", "empanizada", "apanado", "apanada", "rebozado",
                "rebozada")
# [P1-PLAN-LOTE-796 · ronda 3 · 2026-09-28] Más platos que esconden el alérgeno. Las MASAS de desayuno llevan huevo (y
# las de trigo ya eran gluten): pancake, hot cake, waffle, muffin, crepa, bizcocho, magdalena, la tortita (el pancake de
# ES y la de papa/atún de MX; la de arroz o maíz inflado no, `EXCUSAS_DE_BASE`) — el muffin INGLÉS es pan sin huevo
# (`termino_compartido.excusa_de_su_clase`: sólo para el huevo; al celíaco se le sigue marcando). La tortilla «de» verduras, queso,
# yuca o plátano es la de huevo en DO/ES/PR/CO (la de maíz o harina no está aquí). Las albóndigas llevan huevo y pan
# rallado; el aderezo ranch, suero de mantequilla y yema; el pan de maíz (cornbread), huevo y leche; la ensaladilla, la
# salsa rosada y la tártara, mayonesa; pandebono, almojábana y buñuelo (CO), queso y huevo; la quesadilla, queso. El
# César va también aquí para que el registry lo busque en el NOMBRE de la plantilla (`dish_registry._clases_ocultas`).
# Las mezclas de nueces para el maní, como «nueces mixtas» (`constants.normalize_ingredient_for_tracking` las resuelve a
# «nueces/almendras», cuyo alias lleva maní). Nada de esto entra en la dieta vegana: su versión vegetal existe.
_MASAS_CON_HUEVO = ("pancake", "panqueque", "panqueca", "hot cake", "hot-cake", "hotcake", "waffle", "wafle", "muffin",
                    "crepa", "crepe", "bizcocho", "magdalena", "tortita")
_TORTILLAS_DE_HUEVO = ("tortilla de vegetales", "tortilla de verduras", "tortilla de espinaca", "tortilla de queso",
                       "tortilla de yuca", "tortilla de platano", "tortilla de calabacin", "tortilla de atun",
                       "tortilla de bacalao")
_PANES_Y_ADEREZOS_LACTEOS = ("ranch", "pan de maiz", "cornbread", "pandebono", "almojabana", "bunuelo")
OCULTOS = {
    "gluten": ("salsa de soya", "salsa de soja", "frituras de bacalao",  # [P1-PLAN-LOTE-796] ver abajo
               "salsa cremosa de salchicha",  # [P1-PLAN-LOTE-796] alias de «Salsa de salchicha» (gravy con harina)
               *_EMPANIZADOS, "bechamel", "pizza", "mole",  # [revisión]
               # [ronda 3] «sandwich de …»: el «jamón de sándwich» (fila del catálogo) no es un sándwich
               "albondiga", "burrito", "sandwich de", "sandwich integral", "empanadilla", "pastelillo"),
    # [P1-PLAN-LOTE-796] la tortilla de papas ES huevo; se busca sin que «papas» o «tortilla» declaren la clase.
    # [revisión] la francesa (sólo huevo, ES) y el quesillo dominicano (flan: huevo y leche).
    "huevo": ("tortilla espanola", "tortilla de papa", "tortilla de patata", "tortilla francesa", "quesillo",
              *_EMPANIZADOS, *_MASAS_CON_HUEVO, *_TORTILLAS_DE_HUEVO, *_PANES_Y_ADEREZOS_LACTEOS,  # [ronda 3]
              "albondiga", "ensaladilla", "salsa rosada", "salsa tartara", "cesar"),
    # [revisión] la bechamel ya es lácteo por su nombre (LACTEOS_PREPARADOS). [ronda 3] el waffle: la fila «Wafles» del
    # catálogo (US) es el congelado, con leche y huevo, y el registry la sirve sin nombrar la leche.
    "lacteos": ("croqueta", "pizza", *_PANES_Y_ADEREZOS_LACTEOS, "quesadilla", "cesar", "wafle", "waffle"),
    "lactosa": ("croqueta", "pizza", *_PANES_Y_ADEREZOS_LACTEOS, "quesadilla", "wafle", "waffle"),
    # [ronda 3] La granola NO va aquí, a propósito: es cereal con fruto seco, no una mezcla de frutos secos; la de la
    # fila (FDC 171646, «granola, homemade») es avena, almendra y semilla, sin maní, y la comercial con maní lo nombra
    # («granola con maní», que el término «maní» ya caza). La mezcla sí: la fila «Nueces mixtas» lleva maní.
    "mani": ("frutos secos", "fruto seco", "nueces mixtas", "nueces surtidas", "mixed nuts", "mole", "salsa macha",
             "mezcla de nueces", "mix de nueces", "nueces variadas", "nueces y semillas", "trail mix"),
    "frutos secos": ("granola", "mole"),
    "sesamo": ("mole", "salsa macha"),
    # [P1-PLAN-LOTE-796] alias de «Bacalaítos» (masa de trigo, arriba) y de «Vieira»: como compuestos resolverían
    # «bacalao» a gluten y «concha» (el pan) a mariscos, así que se BUSCAN sin declarar la clase.
    "mariscos": ("concha de abanico",),
    # [ronda 3] el nombre de la plantilla «Ensalada César» (anchoas) y la sierra (el pescado del «escabeche de sierra»,
    # DO/PR/MX): `vocabulario_mar` la deja fuera de los sinónimos a propósito (también es la herramienta y el monte), así
    # que se BUSCA sólo cuando el pescado está declarado y no entra en la dieta ni en los rechazos.
    "pescado": ("cesar", "sierra"),
}
# [P1-PLAN-LOTE-796 · ronda 3] Los PLATOS no se excusan por el adjetivo vegetal que los sigue (`graph_orchestrator.
# _PLANT_ADJ_EXCUSE_RX`, pensado para el ANÁLOGO: «leche de coco», «carne de soya»): «pancakes de avena», «croquetas de
# arroz» o «mole de almendra» siguen llevando el huevo, el pan rallado o el cacahuate. «pizza vegana» o «pancakes sin
# huevo» se excusan sólo para las clases animales (`termino_compartido.excusa_de_su_clase`).
PLATOS = frozenset(t for ts in OCULTOS.values() for t in ts)

# [P1-PLAN-LOTE-796 · revisión] Formas de DECLARAR la clase que no son nombres de comida (las consume
# `graph_orchestrator._ALLERGEN_DECLARATION_ALIASES`): «frutos de cáscara» es el nombre LEGAL de la clase en España
# (Reglamento UE 1169/2011, el que sale en cada etiqueta) y «frutos de casca rija» el de Portugal; APLV/PLV/CMPA, la
# alergia a la proteína de la leche de vaca.
DECLARACIONES = {
    "frutos secos": ("frutos de cascara", "fruto de cascara", "frutos de casca rija", "fruto de casca rija",
                     "frutos de casca dura"),
    "lacteos": ("aplv", "plv", "cmpa", "proteina de leche de vaca", "proteina de la leche de vaca"),
}

# [P1-PLAN-LOTE-796 · revisión] Excusa de BASE acotada al término (`graph_orchestrator._ALLERGEN_TERM_BASE_EXCUSES`):
# el mole de olla y el de caderas son caldos sin pasta; el pastelito de yuca (catibía) no lleva trigo. [ronda 3] la tortita
# de arroz, maíz o quinoa inflados (ES) no lleva huevo (la de trigo o harina sí puede: es el pancake).
EXCUSAS_DE_BASE = {"mole": ("olla", "caderas"), "pastelito": ("yuca", "platano", "maiz", "casabe"),
                   "tortita": ("arroz", "maiz", "quinoa", "mijo", "amaranto", "espelta", "centeno")}
