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
    "soya/tofu": ["soya/tofu", "soya", "tofu", "carne de soya", "tofu/soya", "tofu/soya firme", "tofu firme", "soya firme"]
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
    "guineítos verdes": ["guineítos verdes", "guineítos", "guineitos", "guineos verdes", "guineito verde", "guineitos verdes"],
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
    "nueces/almendras": ["nueces/almendras", "nueces", "almendras", "maní"]
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

import unicodedata
import re

def strip_accents(s: str) -> str:
    """Remueve acentos de un string para comparaciones normalizadas.
    Esta es la definición CANÓNICA. Importar desde aquí en todos los módulos."""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def get_reverse_synonyms_map():
    """Crea un diccionario inverso donde la clave es la variante ('pechuga') y el valor es el término base ('pollo').
    Incluye tanto las versiones con acento como sin acento para máxima resiliencia."""
    reverse_map = {}
    for synonyms_dict in [PROTEIN_SYNONYMS, CARB_SYNONYMS, VEGGIE_FAT_SYNONYMS, FRUIT_SYNONYMS]:
        for base, variants in synonyms_dict.items():
            for variant in variants:
                lowered = variant.lower()
                reverse_map[lowered] = base.lower()
                # Para asegurar match con textos sin acentos (e.g. "pure de papas"), agregamos la versión limpia:
                stripped = strip_accents(lowered)
                if stripped not in reverse_map:
                    reverse_map[stripped] = base.lower()
    return reverse_map

GLOBAL_REVERSE_MAP = get_reverse_synonyms_map()
# Ordenamos las variantes de mayor a menor longitud para no reemplazar subpalabras accidentalmente.
# Por ejemplo, reemplazar "habichuelas rojas" antes de "habichuelas".
SORTED_VARIANTS = sorted(GLOBAL_REVERSE_MAP.keys(), key=len, reverse=True)

# Pre-compilar patrones regex con \b al cargar el módulo (O(1) por llamada posterior)
_SYNONYM_PATTERNS = [
    (re.compile(rf'\b{re.escape(variant)}\b'), GLOBAL_REVERSE_MAP[variant])
    for variant in SORTED_VARIANTS
]

def apply_synonyms(text: str) -> str:
    """Reemplaza variantes por sus términos base en un texto usando patrones pre-compilados."""
    # Ahora que `_SYNONYM_PATTERNS` incluye sinónimos CON y SIN acento, si
    # enviamos un texto sin acento ("pure de papas") sí que hará match. 
    text = text.lower()
    for pattern, base in _SYNONYM_PATTERNS:
        text = pattern.sub(base, text)
    return text

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
            →  strip quantities →  "pechuga de pollo deshuesada"
            →  strip accents  →  "pechuga de pollo deshuesada" (sin acentos)
            →  apply synonyms  →  "pollo"
    
    Retorna el término base canónico (ej: "pollo", "platano verde", "aguacate").
    
    ⚠️ DIFERENTE a graph_orchestrator._normalize(), que normaliza NOMBRES DE PLATOS
    para anti-repetición y preserva las técnicas de cocción (ej: "plancha", "guisado")
    para poder distinguir preparaciones diferentes del mismo ingrediente.
    """
    if not raw or not raw.strip():
        return ""
    
    # 1. Normalizar minúsculas solamente, ¡NO quitar acentos todavía!
    # De esta manera _QUANTITY_PATTERN sí atrapa 'puñados' (con ñ).
    text = raw.lower().strip()
    
    # 2. Stripear cantidades y unidades del inicio
    text = _QUANTITY_PATTERN.sub('', text).strip()
    
    # Si quedó vacío tras stripear (ej: el input era solo "200g"), devolver el original
    if not text:
         text = raw.lower().strip()
    
    # 3. Ahora SÍ quitamos todos los acentos usando nuestra función canónica
    text = strip_accents(text)
    
    # 4. Aplicar mapa de sinónimos para colapsar a término base
    #    Como text ahora NO tiene acento ("pure de papas"), 
    #    hará match perfecto con la versión sin acentuar en GLOBAL_REVERSE_MAP
    #    Usamos n-gramas (de mayor a menor) para detectar multipalabra.
    words = text.split()
    for n in range(min(4, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            if ngram in GLOBAL_REVERSE_MAP:
                return GLOBAL_REVERSE_MAP[ngram]
    
    # 5. Fallback: si no matcheó ningún sinónimo, devolver el texto limpio
    return text



# ============================================================
# TÉCNICAS DE COCCIÓN Y SUPLEMENTOS
# ============================================================
TECHNIQUE_FAMILIES = {
    "seca": [
        "Horneado Saludable",
        "En Airfryer Crujiente",
        "Asado a la Parrilla",
        "A la Plancha con Cítricos"
    ],
    "húmeda": [
        "Guiso o Estofado Ligero",
        "En Salsa a base de Vegetales Naturales"
    ],
    "transformada": [
        "Desmenuzado (Ropa Vieja)",
        "En Puré o Majado",
        "Croquetas o Tortitas al Horno",
        "Relleno (Ej. Canoas, Vegetales rellenos)"
    ],
    "fresca": [
        "Estilo Ceviche o Fresco",
        "Salteado tipo Wok",
        "Al Vapor con Finas Hierbas"
    ],
    "fusión": [
        "Estilo Fusión Criolla",
        "Estilo Bowl/Poke Tropical",
        "Wrap o Burrito Dominicano"
    ]
}

ALL_TECHNIQUES = [t for techs in TECHNIQUE_FAMILIES.values() for t in techs]

TECH_TO_FAMILY = {}
for family, techs in TECHNIQUE_FAMILIES.items():
    for t in techs:
        TECH_TO_FAMILY[t] = family

SUPPLEMENT_NAMES = {
    "whey_protein": "Proteína Whey",
    "creatine": "Creatina Monohidrato",
    "bcaa": "Aminoácidos BCAA",
    "glutamine": "Glutamina",
    "omega3": "Omega-3 (Aceite de Pescado)",
    "multivitamin": "Multivitamínico Completo",
    "vitamin_d": "Vitamina D3",
    "magnesium": "Magnesio (Citrato o Glicinato)",
    "pre_workout": "Pre-Entreno (Cafeína + Beta-Alanina)",
    "collagen": "Colágeno Hidrolizado",
}

def _get_fast_filtered_catalogs(allergies: tuple, dislikes: tuple, diet: str):
    """Filtra el catálogo dominicano basado en restricciones del usuario O(N) sin Cache Thrashing volátil."""
    filtered_proteins = DOMINICAN_PROTEINS.copy()
    filtered_carbs = DOMINICAN_CARBS.copy()
    filtered_veggies = DOMINICAN_VEGGIES_FATS.copy()
    filtered_fruits = DOMINICAN_FRUITS.copy()
    
    restrictions = list(allergies) + list(dislikes)
    
    if diet in ["vegano", "vegan"]:
        restrictions.extend(["pollo", "cerdo", "res", "pescado", "atún", "huevos", "queso", "salami", "camarones", "chuleta", "longaniza", "carne", "marisco", "lácteo", "leche"])
    elif diet in ["vegetariano", "vegetarian"]:
        restrictions.extend(["pollo", "cerdo", "res", "pescado", "atún", "salami", "camarones", "chuleta", "longaniza", "carne", "marisco"])
    elif diet in ["pescetariano", "pescatarian"]:
        restrictions.extend(["pollo", "cerdo", "res", "salami", "chuleta", "longaniza", "carne"])
        
    if not restrictions:
        return filtered_proteins, filtered_carbs, filtered_veggies, filtered_fruits
        
    normalized_restrictions = [strip_accents(r.lower()) for r in restrictions]
    
    # Reglas genéricas CATCH-ALL de mariscos y carnes
    if any(r in ["mariscos", "seafood", "marisco"] for r in normalized_restrictions):
        normalized_restrictions.extend(["camaron", "camarones", "pescado", "atun"])
    if any(r in ["carne", "carnes", "meat"] for r in normalized_restrictions):
        normalized_restrictions.extend(["pollo", "cerdo", "res", "chuleta", "longaniza", "salami"])
        
    # [OPTIMIZACIÓN O(1)] Compilar un único patrón maestro ultra veloz
    import re
    or_pattern = '|'.join(map(re.escape, normalized_restrictions))
    fast_regex = re.compile(rf'\b({or_pattern})\b')
    
    def is_allowed(item):
        item_normalized = strip_accents(item.lower())
        return not fast_regex.search(item_normalized)
        
    filtered_proteins = [p for p in filtered_proteins if is_allowed(p)]
    filtered_carbs = [c for c in filtered_carbs if is_allowed(c)]
    filtered_veggies = [v for v in filtered_veggies if is_allowed(v)]
    filtered_fruits = [f for f in filtered_fruits if is_allowed(f)]
    
    return filtered_proteins, filtered_carbs, filtered_veggies, filtered_fruits

# ============================================================
# BASE DE DATOS CLINICA / DIGESTIVA (RAG CULINARIO)
# ============================================================
CULINARY_KNOWLEDGE_BASE = """
<biblioteca_culinaria_local>
[BASE DE DATOS CLÍNICA DE PLATOS DOMINICANOS]
Mofongo: 5-6 horas de digestión (Fritura profusa + almidón denso). Peligro de reflujo nocturno y pico insulínico si se consume antes de dormir.
Mangú (Los Tres Golpes): 4-5 horas de digestión aguda. Altísima carga de grasas saturadas (salami/queso frito) combinadas con carbohidrato puro.
La Bandera (arroz, habichuela, carne, concón): 4.5 horas. Demasiada carga glucémica para horarios sin desgaste físico posterior.
Yaroa: 6+ horas de digestión. Bomba de grasas trans/saturadas y carbohidratos fritos. Arruina el ciclo REM del sueño.
Pica Pollo / Chimi: 5+ horas. Exceso de aceites hidrogenados e irritantes gástricos.
Sancocho: 4-5 horas. Extrema condensación de viandas pesadas y caldos grasos.
</biblioteca_culinaria_local>
"""
