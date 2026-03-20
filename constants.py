# backend/constants.py

DOMINICAN_PROTEINS = [
    "Pollo", "Cerdo", "Res", "Pescado", "Atún", "Huevos", "Queso de Freír",
    "Salami Dominicano", "Camarones", "Chuleta", "Longaniza",
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
    "berenjena": ["berenjena", "berenjenas", "berenjena rellena"],
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

DOMINICAN_FRUITS = [
    "Guineo", "Mango", "Piña", "Lechosa", "Chinola",
    "Limón", "Fresa", "Naranja", "Sandía", "Melón"
]

FRUIT_SYNONYMS = {
    "guineo": ["guineo", "guineo maduro", "banana", "banano"],
    "mango": ["mango", "mangos", "mango maduro"],
    "piña": ["piña", "pina", "piña natural"],
    "lechosa": ["lechosa", "papaya"],
    "chinola": ["chinola", "maracuyá", "maracuya", "parcha"],
    "limón": ["limón", "limon", "lima", "jugo de limón"],
    "fresa": ["fresa", "fresas", "frutilla"],
    "naranja": ["naranja", "naranjas", "jugo de naranja"],
    "sandía": ["sandía", "sandia", "patilla"],
    "melón": ["melón", "melon"]
}

def get_reverse_synonyms_map():
    """Crea un diccionario inverso donde la clave es la variante ('pechuga') y el valor es el término base ('pollo')."""
    reverse_map = {}
    for synonyms_dict in [PROTEIN_SYNONYMS, CARB_SYNONYMS, VEGGIE_FAT_SYNONYMS, FRUIT_SYNONYMS]:
        for base, variants in synonyms_dict.items():
            for variant in variants:
                reverse_map[variant.lower()] = base.lower()
    return reverse_map

GLOBAL_REVERSE_MAP = get_reverse_synonyms_map()
# Ordenamos las variantes de mayor a menor longitud para no reemplazar subpalabras accidentalmente.
# Por ejemplo, reemplazar "habichuelas rojas" antes de "habichuelas".
SORTED_VARIANTS = sorted(GLOBAL_REVERSE_MAP.keys(), key=len, reverse=True)
import re
# Pre-compilar patrones regex con \b al cargar el módulo (O(1) por llamada posterior)
_SYNONYM_PATTERNS = [
    (re.compile(rf'\b{re.escape(variant)}\b'), GLOBAL_REVERSE_MAP[variant])
    for variant in SORTED_VARIANTS
]

def apply_synonyms(text: str) -> str:
    """Reemplaza variantes por sus términos base en un texto usando patrones pre-compilados."""
    text = text.lower()
    for pattern, base in _SYNONYM_PATTERNS:
        text = pattern.sub(base, text)
    return text

import unicodedata

# Regex pre-compilado para stripear cantidades/unidades al inicio de un string de ingrediente.
# Captura patrones como: "200g", "1/2", "3 cdas", "1 lb", "2 tazas de", "100 ml de", etc.
_QUANTITY_PATTERN = re.compile(
    r'^[\d\s/.,]+'                           # Números, fracciones, espacios iniciales
    r'(?:'
        r'g\b|gr\b|kg\b|mg\b|ml\b|l\b|lb\b|lbs\b|oz\b'   # Unidades métricas/imperiales
        r'|cdas?\b|cdtas?\b|cucharadas?\b|cucharaditas?\b'  # Cucharadas
        r'|tazas?\b|vasos?\b|porciones?\b|unidades?\b'      # Medidas de volumen
        r'|rodajas?\b|rebanadas?\b|lascas?\b|tiras?\b'      # Formas de corte
        r'|pizcas?\b|puñados?\b|ramitas?\b'                  # Cantidades imprecisas
        r'|libras?\b|medias?\b|media\b'                      # Libras / fracciones
    r')?'
    r'\s*(?:de\s+)?',                        # "de " opcional que conecta cantidad con ingrediente
    re.IGNORECASE
)

def normalize_ingredient_for_tracking(raw: str) -> str:
    """Normaliza un string crudo de ingrediente para frequency tracking.
    
    Pipeline:  "200g Pechuga de Pollo deshuesada" 
            →  strip accents  →  "200g pechuga de pollo deshuesada"
            →  strip quantities →  "pechuga de pollo deshuesada"
            →  apply synonyms  →  "pollo"
    
    Retorna el término base canónico (ej: "pollo", "platano verde", "aguacate").
    
    ⚠️ DIFERENTE a graph_orchestrator._normalize(), que normaliza NOMBRES DE PLATOS
    para anti-repetición y preserva las técnicas de cocción (ej: "plancha", "guisado")
    para poder distinguir preparaciones diferentes del mismo ingrediente.
    """
    if not raw or not raw.strip():
        return ""
    
    # 1. Normalizar acentos y minúsculas
    text = ''.join(
        c for c in unicodedata.normalize('NFD', raw)
        if unicodedata.category(c) != 'Mn'
    ).lower().strip()
    
    # 2. Stripear cantidades y unidades del inicio
    text = _QUANTITY_PATTERN.sub('', text).strip()
    
    # Si quedó vacío tras stripear (ej: el input era solo "200g"), devolver el original limpio
    if not text:
        text = ''.join(
            c for c in unicodedata.normalize('NFD', raw)
            if unicodedata.category(c) != 'Mn'
        ).lower().strip()
    
    # 3. Aplicar mapa de sinónimos para colapsar a término base
    #    Usamos n-gramas (de mayor a menor) contra el GLOBAL_REVERSE_MAP
    #    para detectar multipalabra como "pechuga de pollo" → "pollo"
    words = text.split()
    for n in range(min(4, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            if ngram in GLOBAL_REVERSE_MAP:
                return GLOBAL_REVERSE_MAP[ngram]
    
    # 4. Fallback: si no matcheó ningún sinónimo, devolver el texto limpio
    return text
