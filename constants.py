# backend/constants.py

DOMINICAN_PROTEINS = [
    "Pollo", "Cerdo", "Res", "Pescado", "Atún", "Huevos", "Queso de Freír",
    "Salami Dominicano", "Camarones", "Chuleta", "Longaniza", "Berenjena",
    "Habichuelas Rojas", "Habichuelas Negras", "Gandules", "Lentejas", "Garbanzos", "Soya/Tofu"
]

DOMINICAN_CARBS = [
    "Plátano Verde", "Plátano Maduro", "Yuca", "Batata", "Arroz Blanco", 
    "Arroz Integral", "Avena", "Pan Integral", "Papas", "Guineítos Verdes", "Ñame", "Yautía"
]

PROTEIN_SYNONYMS = {
    "pollo": ["pollo", "pechuga", "muslo", "alitas", "chicharrón de pollo", "filete de pollo"],
    "cerdo": ["cerdo", "masita", "chicharrón de cerdo", "lomo", "pernil", "costilla"],
    "res": ["res", "carne molida", "bistec", "filete", "churrasco", "vaca", "picadillo", "carne de res"],
    "pescado": ["pescado", "dorado", "chillo", "mero", "salmón", "tilapia", "filete de pescado"],
    "atún": ["atún", "atun"],
    "huevos": ["huevos", "huevo", "tortilla", "revoltillo"],
    "queso de freír": ["queso de freír", "queso de freir", "queso frito", "queso de hoja"],
    "salami dominicano": ["salami dominicano", "salami", "salchichón"],
    "camarones": ["camarones", "camarón", "camaron"],
    "chuleta": ["chuleta", "chuletas", "chuleta frita", "chuleta al horno"],
    "longaniza": ["longaniza", "longanizas"],
    "berenjena": ["berenjena", "berenjenas", "berenjena rellena"],
    "habichuelas rojas": ["habichuelas rojas", "frijoles rojos", "habichuela roja"],
    "habichuelas negras": ["habichuelas negras", "frijoles negros", "habichuela negra"],
    "gandules": ["gandules", "guandules", "gandul", "guandul"],
    "lentejas": ["lentejas", "lenteja"],
    "garbanzos": ["garbanzos", "garbanzo"],
    "soya/tofu": ["soya", "tofu", "carne de soya"]
}

CARB_SYNONYMS = {
    "plátano verde": ["plátano verde", "platano verde", "mangú", "mangu", "tostones", "fritos verdes", "mangú de plátano", "mangu de platano"],
    "plátano maduro": ["plátano maduro", "platano maduro", "maduros", "plátano al caldero", "fritos maduros"],
    "yuca": ["yuca", "casabe", "arepitas de yuca", "puré de yuca"],
    "arroz blanco": ["arroz blanco", "arroz"],
    "arroz integral": ["arroz integral"],
    "avena": ["avena", "avena en hojuelas", "overnight oats"],
    "pan integral": ["pan integral", "pan", "tostada integral", "tostada"],
    "papas": ["papas", "papa", "puré de papas", "papa hervida"],
    "guineítos verdes": ["guineítos", "guineitos", "guineos verdes", "guineito verde", "guineitos verdes"],
    "ñame": ["ñame", "name", "ñame hervido"],
    "yautía": ["yautía", "yautia", "yautía hervida"],
    "batata": ["batata", "puré de batata", "batata hervida", "boniato"]
}

DOMINICAN_VEGGIES_FATS = [
    "Aguacate", "Berenjena", "Tayota", "Repollo", "Zanahoria", 
    "Molondrones", "Brócoli", "Coliflor", "Tomate", "Vainitas",
    "Aceitunas", "Cebolla", "Ajíes", "Aceite de Oliva", "Nueces/Almendras"
]

VEGGIE_FAT_SYNONYMS = {
    "aguacate": ["aguacate", "palta"],
    "berenjena": ["berenjena", "berenjenas"],
    "tayota": ["tayota", "chayote"],
    "repollo": ["repollo"],
    "zanahoria": ["zanahoria", "zanahorias"],
    "molondrones": ["molondrones", "molondrón", "okra"],
    "brócoli": ["brócoli", "brocoli"],
    "coliflor": ["coliflor"],
    "tomate": ["tomate", "tomates", "pico de gallo"],
    "vainitas": ["vainitas", "judías verdes", "ejotes"],
    "aceitunas": ["aceitunas", "aceituna"],
    "cebolla": ["cebolla", "cebollas"],
    "ajíes": ["ajíes", "ají", "pimientos", "pimiento"],
    "aceite de oliva": ["aceite de oliva", "aceite verde"],
    "nueces/almendras": ["nueces", "almendras", "maní"]
}

def get_reverse_synonyms_map():
    """Crea un diccionario inverso donde la clave es la variante ('pechuga') y el valor es el término base ('pollo')."""
    reverse_map = {}
    for synonyms_dict in [PROTEIN_SYNONYMS, CARB_SYNONYMS, VEGGIE_FAT_SYNONYMS]:
        for base, variants in synonyms_dict.items():
            for variant in variants:
                reverse_map[variant.lower()] = base.lower()
    return reverse_map

GLOBAL_REVERSE_MAP = get_reverse_synonyms_map()
# Ordenamos las variantes de mayor a menor longitud para no reemplazar subpalabras accidentalmente.
# Por ejemplo, reemplazar "habichuelas rojas" antes de "habichuelas".
SORTED_VARIANTS = sorted(GLOBAL_REVERSE_MAP.keys(), key=len, reverse=True)

def apply_synonyms(text: str) -> str:
    """Reemplaza variantes por sus términos base en un texto usando expresiones regulares."""
    import re
    text = text.lower()
    for variant in SORTED_VARIANTS:
        if variant in text:
            # Usar \b para límites de palabra y evitar reemplazos parciales
            text = re.sub(rf'\b{re.escape(variant)}\b', GLOBAL_REVERSE_MAP[variant], text)
    return text
