# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-198 · 2026-09-24] Plantillas del plan de EMERGENCIA (fallback determinista), fuera del god-file.

Datos puros movidos TAL CUAL desde graph_orchestrator.py (que estaba a 1 línea de su tope de 52.240): el genérico
de 3 comidas y el curado bariátrico de 6 (P1-FALLBACK-BARIATRIC-CURATED). graph_orchestrator los re-exporta con el
mismo nombre, así que `go._FALLBACK_MEAL_POOLS` sigue funcionando para todo el que lo lea.
"""


# Pools ordenados por slot: la PRIMERA entrada reproduce el menú histórico
# (restricted vacío → comportamiento idéntico); la ÚLTIMA es neutral (sin
# tokens). Cada entrada: (name, frozenset(tokens), desc, [ingredients]).
_FALLBACK_MEAL_POOLS = {
    "Desayuno": [
        ("Huevos y Avena", frozenset({"egg", "oats", "gluten"}),
         "Huevos revueltos con avena cocida y fruta.",
         ["2 huevos", "1/2 taza de avena cocida", "1 fruta de temporada"]),
        ("Avena con Frutas y Semillas", frozenset({"oats", "gluten"}),
         "Avena cocida con frutas frescas y semillas.",
         ["1/2 taza de avena", "frutas variadas", "1 cda de semillas de chía"]),
        ("Frutas Frescas con Semillas", frozenset(),
         "Bowl de frutas frescas de temporada con semillas.",
         ["frutas variadas de temporada", "semillas de girasol o chía", "agua"]),
    ],
    "Almuerzo": [
        ("Pollo y Arroz", frozenset({"chicken"}),
         "Pechuga a la plancha con arroz blanco y vegetales.",
         ["pechuga de pollo a la plancha", "arroz blanco", "vegetales al gusto"]),
        ("Carne de Res y Arroz", frozenset({"beef"}),
         "Carne de res magra con arroz y vegetales.",
         ["carne de res magra", "arroz blanco", "vegetales al gusto"]),
        ("Pescado con Arroz", frozenset({"fish"}),
         "Filete de pescado con arroz y ensalada.",
         ["filete de pescado", "arroz blanco", "ensalada verde"]),
        ("Lentejas con Arroz", frozenset({"legume"}),
         "Lentejas guisadas con arroz y vegetales.",
         ["lentejas guisadas", "arroz blanco", "vegetales al gusto"]),
        ("Arroz con Vegetales y Aguacate", frozenset(),
         "Arroz con vegetales salteados y aguacate.",
         ["arroz blanco", "vegetales salteados", "1/2 aguacate"]),
    ],
    "Cena": [
        ("Pescado y Batata", frozenset({"fish"}),
         "Filete de pescado al horno con batata asada.",
         ["filete de pescado al horno", "batata asada", "vegetales al vapor"]),
        ("Pollo con Vegetales", frozenset({"chicken"}),
         "Pechuga de pollo con vegetales al vapor y batata.",
         ["pechuga de pollo", "vegetales al vapor", "batata asada"]),
        ("Carne de Res con Vegetales", frozenset({"beef"}),
         "Carne de res magra con vegetales y batata.",
         ["carne de res magra", "vegetales al vapor", "batata asada"]),
        ("Garbanzos con Vegetales", frozenset({"legume"}),
         "Garbanzos guisados con vegetales y batata.",
         ["garbanzos guisados", "vegetales al vapor", "batata asada"]),
        ("Ensalada de Vegetales con Aguacate", frozenset(),
         "Ensalada abundante de vegetales con aguacate y aceite de oliva.",
         ["vegetales variados", "1/2 aguacate", "aceite de oliva"]),
    ],
}


# [P1-FALLBACK-BARIATRIC-CURATED · 2026-06-28] Pool de 6 slots clínicamente VETADO por construcción (review adversaria
# código+clínico ASMBS). Formato 4-tuple idéntico al genérico: (name, frozenset(tokens), desc, [ingredients]). Reglas
# garantizadas: porciones ≤caps (queso30/yogurt≤120/fruta≤80/aguacate≤30/volumen≤300g comida·≤200g merienda), proteína
# densa MOIST (pescado/huevo/yogurt griego/pollo GUISADO — no seco), sin crudos, coherencia nombre↔ingredientes, sin
# condimentos densos (caps cross-día = no-op). NO usa cottage (el cap de queso ("queso") lo amputaría a 30g → yogurt
# griego en su lugar). Ingredientes con GRAMOS resolubles en el catálogo Neon (verificado: 26/28; galleta/caldo NO
# resuelven → casabe/auyama). Último item de cada slot = frozenset() neutral (vegano, fail-safe bajo multi-alergia).
# Tokens en inglés (egg/dairy/fish/oats/gluten/chicken/legume) consistentes con _detect_restricted_tokens.
_FALLBACK_MEAL_POOLS_BARIATRIC = {
    "Desayuno": [
        ("Huevos Revueltos con Yuca", frozenset({"egg"}),
         "2 huevos revueltos con yuca cocida. Proteína densa, sin crudos.",
         ["2 unidades huevo", "100g yuca cocida"]),
        ("Yogurt Griego con Avena", frozenset({"dairy", "oats", "gluten"}),
         "Yogurt griego sin azúcar con avena cocida. Anti-dumping.",
         ["115g yogurt griego", "30g avena cocida"]),
        ("Yuca con Aguacate", frozenset(),
         "Yuca cocida con aguacate. Suave, bajo volumen, sin crudos.",
         ["120g yuca cocida", "25g aguacate"]),
    ],
    "Merienda AM": [
        ("Sardinas con Casabe", frozenset({"fish"}),
         "Sardinas en agua escurridas con casabe. Proteína MOIST.",
         ["60g sardinas en agua escurridas", "20g casabe"]),
        ("Queso Blanco con Manzana", frozenset({"dairy"}),
         "Queso blanco con manzana. Porción medida.",
         ["30g queso blanco", "60g manzana"]),
        ("Zanahoria Cocida", frozenset(),
         "Zanahoria cocida al vapor. Merienda ligera, sin crudos.",
         ["120g zanahoria cocida"]),
    ],
    "Almuerzo": [
        ("Mero a la Plancha con Papa y Brócoli", frozenset({"fish"}),
         "Mero a la plancha con papa cocida y brócoli al vapor. Proteína densa primero.",
         ["150g mero a la plancha", "80g papa cocida", "60g brocoli al vapor"]),
        ("Pollo Guisado con Yuca y Tayota", frozenset({"chicken"}),
         "Pollo guisado húmedo con yuca cocida y tayota. Proteína MOIST (guisada, no seca).",
         ["120g pollo guisado", "90g yuca cocida", "70g tayota cocida"]),
        ("Lentejas Guisadas con Arroz", frozenset({"legume"}),
         "Lentejas guisadas con arroz blanco. Proteína vegetal, volumen controlado.",
         ["120g lentejas guisadas", "80g arroz blanco cocido"]),
        ("Auyama Guisada con Yuca", frozenset(),
         "Auyama guisada con yuca cocida. Opción vegetal suave, fácil de digerir.",
         ["120g auyama cocida", "80g yuca cocida"]),
    ],
    "Merienda PM": [
        ("Atún con Casabe", frozenset({"fish"}),
         "Atún en agua escurrido con casabe. Proteína MOIST, bajo volumen.",
         ["80g atun en agua escurrido", "20g casabe"]),
        ("Yogurt Griego con Fresa", frozenset({"dairy"}),
         "Yogurt griego sin azúcar con fresa. Lácteo suave, anti-dumping.",
         ["115g yogurt griego", "60g fresa"]),
        ("Garbanzos Cocidos", frozenset({"legume"}),
         "Garbanzos cocidos. Proteína vegetal, porción de merienda.",
         ["100g garbanzos cocidos"]),
        ("Zanahoria con Aguacate", frozenset(),
         "Zanahoria cocida con aguacate. Merienda ligera.",
         ["100g zanahoria cocida", "25g aguacate"]),
    ],
    "Cena": [
        ("Tilapia al Horno con Batata", frozenset({"fish"}),
         "Tilapia al horno con batata cocida y vainitas. Cena ligera, sin crudos.",
         ["150g tilapia al horno", "75g batata cocida", "65g vainitas cocidas"]),
        ("Huevos con Berenjena Asada", frozenset({"egg"}),
         "Huevos cocidos con berenjena asada. Proteína MOIST blanda, bajo volumen.",
         ["2 unidades huevo", "100g berenjena asada"]),
        # [P0-FALLBACK-CENA-ARROZ · 2026-07-01] Era "Habichuelas Guisadas con Arroz" (80g arroz blanco cocido):
        # arroz en la CENA — la violación cultural #1 del slot-appropriateness — servida DETERMINISTA el día 3/7
        # de todo plan fallback bariátrico multi-día (rotación P2-FALLBACK-DAY-ROTATION), porque el fallback
        # crítico bypassa assemble (sin _night_rice_autofix ni gate). Tubérculo nocturno en su lugar; "yuca
        # cocida" ya está verificada en el catálogo por este mismo pool. tooltip-anchor: P0-FALLBACK-CENA-ARROZ
        ("Habichuelas Guisadas con Yuca", frozenset({"legume"}),
         "Habichuelas guisadas con yuca cocida. Proteína vegetal.",
         ["80g habichuelas guisadas", "80g yuca cocida"]),
        ("Auyama con Vainitas", frozenset(),
         "Auyama cocida con vainitas. Cena vegetal suave.",
         ["120g auyama cocida", "80g vainitas cocidas"]),
    ],
    "Merienda Nocturna": [
        ("Yogurt Griego Natural", frozenset({"dairy"}),
         "Yogurt griego sin azúcar. Ultra-ligero, evita dumping nocturno.",
         ["115g yogurt griego"]),
        ("Atún en Agua", frozenset({"fish"}),
         "Atún en agua escurrido, porción pequeña. Proteína lenta nocturna.",
         ["60g atun en agua escurrido"]),
        ("Zanahoria Cocida", frozenset(),
         "Zanahoria cocida, porción pequeña. Merienda nocturna ligera.",
         ["100g zanahoria cocida"]),
    ],
}


# [P1-PLAN-LOTE-240 · 2026-09-25] Vocabulario propio del fallback (antes en el grafo; el escáner SSOT lo complementa desde el lote 237).
_FALLBACK_ALLERGEN_KEYWORDS = {
    "egg":       ("huevo", "huevos", "egg", "clara de huevo"),
    "chicken":   ("pollo", "chicken", "pechuga de pollo", "gallina"),
    "fish":      ("pescado", "pescados", "fish", "atun", "salmon", "tilapia",
                  "bacalao", "sardina", "mero"),
    "shellfish": ("marisco", "mariscos", "camaron", "camarones", "langosta",
                  "cangrejo", "shellfish", "shrimp", "ostra", "calamar", "pulpo"),
    "beef":      ("carne de res", "ternera", "vacuno", "beef"),
    "pork":      ("cerdo", "puerco", "pork", "tocino", "jamon", "chorizo",
                  "salchicha", "embutido"),
    "dairy":     ("leche", "lacteo", "lacteos", "lactosa", "dairy", "queso",
                  "yogur", "yogurt", "mantequilla"),
    "peanut":    ("mani", "peanut", "cacahuate", "cacahuete"),
    "soy":       ("soya", "soja", "tofu", "edamame"),
    "gluten":    ("gluten", "trigo", "wheat", "celiaco", "celiaca"),
    "oats":      ("avena", "oat"),
    "legume":    ("lenteja", "lentejas", "garbanzo", "garbanzos", "frijol",
                  "frijoles", "habichuela", "habichuelas", "legumbre", "legumbres"),
    "nuts":      ("nuez", "nueces", "almendra", "almendras", "frutos secos",
                  "tree nut", "anacardo", "merey", "pistacho"),
}
