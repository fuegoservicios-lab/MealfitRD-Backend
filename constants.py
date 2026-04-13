import logging
import os
import math
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

# --- VECTOR SEARCH CACHE ---
_embedding_model = None
_embedding_cache = {}
_pantry_embeddings_cache = {}

def get_embedding(text: str) -> List[float]:
    global _embedding_model
    if not _embedding_model:
        _embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview", 
            google_api_key=os.environ.get("GEMINI_API_KEY")
        )
    if text not in _embedding_cache:
        emb = _embedding_model.embed_query(text)
        _embedding_cache[text] = emb
    return _embedding_cache[text]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)
# ---------------------------

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
    "atún": ["atún", "atun", "atun en agua", "atun en lata"],
    "sardina": ["sardina", "sardinas", "sardina en lata"],
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
    "avena": ["avena", "avena en hojuelas", "overnight oats", "avena instantanea"],
    "pasta": ["pasta", "espagueti", "espaguetis", "spaghetti", "macarrones", "coditos", "fideos"],
    "quinoa": ["quinoa", "quinua"],
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
    "tayota": ["tayota", "chayote", "tayotas", "chayotes", "cidra"],
    "espinaca": ["espinaca", "espinacas", "baby spinach"],
    "pepino": ["pepino", "pepinos"],
    "lechuga": ["lechuga", "lechugas", "lechuga romana", "lechuga iceberg"],
    "cilantro": ["cilantro", "culantro", "verdura", "recao"],
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
    "guineo": ["guineo", "guineo maduro", "banana", "banano", "cambur"],
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
        r'|paquetes?\b|paqueticos?\b|fundas?\b|latas?\b|sobres?\b|sobrecitos?\b|chin\b|toques?\b|chorritos?\b|hojitas?\b' # Términos dominicanos y extremos
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

def _to_base_unit(qty: float, unit: str):
    unit = unit.lower() if unit else 'unidad'
    # Weights -> g
    if unit in ['g', 'gr', 'gramos', 'gramo']: return qty, 'g'
    if unit in ['kg', 'kilo', 'kilos', 'kilogramos', 'kilogramo']: return qty * 1000.0, 'g'
    if unit in ['lb', 'lbs', 'libra', 'libras']: return qty * 453.592, 'g'
    if unit in ['oz', 'onza', 'onzas']: return qty * 28.3495, 'g'
    
    # Volumes -> ml
    if unit in ['ml', 'mililitro', 'mililitros']: return qty, 'ml'
    if unit in ['l', 'litro', 'litros']: return qty * 1000.0, 'ml'
    if unit in ['taza', 'tazas']: return qty * 236.588, 'ml'
    if unit in ['cda', 'cucharada', 'cucharadas']: return qty * 14.7868, 'ml'
    if unit in ['cdta', 'cucharadita', 'cucharaditas', 'cdita']: return qty * 4.92892, 'ml'
    
    # Extreme Abstract Dominican Terms -> nominal weight
    if unit in ['chin', 'pizca', 'pizcas', 'toque', 'toques', 'chorrito', 'chorritos', 'puñado', 'puñados', 'ramita', 'ramitas', 'hojita', 'hojitas']: return qty * 5.0, 'g'
    
    # Informal Containers -> always track as units for delta
    if unit in ['paquete', 'paquetes', 'paquetico', 'paqueticos', 'funda', 'fundas', 'sobre', 'sobres', 'sobrecito', 'sobrecitos', 'lata', 'latas', 'pote', 'potes']: return qty, 'unidad'
    
    return qty, unit

def _format_unit_qty(base_qty: float, base_unit: str) -> str:
    """Para mensajes de error legibles."""
    if base_unit == 'g':
        if base_qty >= 1000: return f"{round(base_qty/1000.0, 2)} kg"
        if base_qty >= 226: return f"{round(base_qty/453.592, 2)} lbs"
        return f"{round(base_qty)} g"
    if base_unit == 'ml':
        if base_qty >= 1000: return f"{round(base_qty/1000.0, 2)} L"
        if base_qty >= 220: return f"{round(base_qty/236.588, 1)} tazas"
        return f"{round(base_qty)} ml"
    return f"{round(base_qty, 2)} {base_unit}"

VOLUMETRIC_DENSITIES = {
    # Carbohidratos (Crudos y cocidos en volumen)
    "arroz blanco": 0.845,
    "arroz integral": 0.845,
    "arroz": 0.845,
    "avena": 0.380,
    "harina": 0.550,
    "harina de maiz": 0.550,
    "pasta": 0.420,
    "quinoa": 0.720,
    
    # Granos (Leguminosas en volumen)
    "lentejas": 0.810,
    "habichuelas rojas": 0.840,
    "habichuelas negras": 0.840,
    "garbanzos": 0.840,
    "habichuelas": 0.840,
    "gandules": 0.840,
    
    # Proteínas (Picadas/Desmenuzadas/Líquidas)
    "pollo": 0.634,
    "res": 0.634,
    "cerdo": 0.634,
    "pescado": 0.600,
    "carne molida": 0.850,
    "atun": 0.600,
    "tofu": 0.950,
    "clara de huevo": 1.03,
    
    # Lácteos y Similares
    "queso de freir": 0.500,
    "queso rallado": 0.450,
    "queso": 0.500,
    "yogurt": 1.050,
    "yogurt griego": 1.050,
    "leche": 1.030,
    "leche en polvo": 0.500,
    "queso crema": 0.950,
    "ricotta": 0.950,
    "cottage": 0.950,
    
    # Grasas y Semillas
    "aceite": 0.920,
    "aceite de oliva": 0.920,
    "aceite de coco": 0.920,
    "mantequilla": 0.960,
    "mayonesa": 0.960,
    "mantequilla de mani": 0.960,
    "mani": 0.600,
    "nueces": 0.600,
    "almendras": 0.600,
    "nueces/almendras": 0.600,
    "semillas de chia": 0.650,
    "semillas de linaza": 0.650,
    
    # Vegetales y Frutas (Picados en taza/volumen)
    "cebolla": 0.600,
    "tomate": 0.600,
    "aji": 0.600,
    "pimiento": 0.600,
    "espinaca": 0.200,
    "lechuga": 0.150,
    "fruta picada": 0.650,
    "pina": 0.650,
    "melon": 0.650,
    "manzana": 0.650,
    "salsa de tomate": 1.050,
    "pasta de tomate": 1.050,
    "salsa de soya": 1.050,
}

UNIT_WEIGHTS = {
    # Víveres y Carbohidratos (Unidad entera o porción estándar)
    "platano verde": 280.0,
    "platano maduro": 280.0,
    "guineito verde": 100.0,
    "yuca": 400.0,
    "batata": 220.0,
    "papa": 150.0,
    "papas": 150.0,
    "name": 300.0,
    "yautia": 250.0,
    "pan integral": 30.0,
    "pan": 30.0,
    "pan de agua": 60.0,
    "casabe": 20.0,
    "tortilla": 45.0,
    "wrap": 45.0,
    "plantilla": 45.0,

    # Proteínas (Unidades y raciones rápidas)
    "huevos": 50.0,
    "huevo": 50.0,
    "chuleta": 150.0,
    "longaniza": 100.0,
    "salami": 40.0,
    "salami dominicano": 40.0,
    "queso": 25.0,
    "queso crema": 226.0, # 8pz
    "queso cottage": 453.0, # 16oz
    "yogurt griego": 453.0, # 16oz
    "yogurt": 453.0,
    "jamon": 20.0,
    "pechuga de pavo": 20.0,
    "pechuga de pollo": 200.0,
    "filete de pescado": 150.0,
    "lata de atun": 120.0,
    "atun": 120.0,
    "sardina": 106.0,

    # Frutas y Vegetales (Unidades enteras)
    "guineo": 120.0,
    "guineo maduro": 120.0,
    "manzana": 150.0,
    "naranja": 130.0,
    "limon": 60.0,
    "chinola": 90.0,
    "aguacate": 250.0,
    "tomate": 120.0,
    "cebolla": 110.0,
    "ajo": 5.0,
    "diente de ajo": 5.0,
    "aji": 100.0,
    "pimiento": 100.0,
    "zanahoria": 75.0,
    "pepino": 200.0,
    "berenjena": 250.0,
    "tayota": 250.0,
    "molondron": 15.0,
    "molondrones": 15.0,

    # Frutas tropicales (Unidades enteras - Mercado Dominicano)
    "mango": 300.0,
    "mango maduro": 300.0,
    "pina": 1500.0,
    "piña": 1500.0,
    "lechosa": 800.0,
    "papaya": 800.0,
    "fresa": 15.0,
    "sandia": 3000.0,
    "melon": 1200.0,

    # Vegetales grandes / Crucíferas (Unidades enteras)
    "auyama": 500.0,
    "brocoli": 300.0,
    "coliflor": 500.0,
    "repollo": 600.0,
    "vainitas": 200.0,
    "habichuelas verdes": 200.0,
    "aji dulce": 10.0,
    "ajies": 100.0,
}

def _get_converted_quantity(req_qty: float, req_unit: str, dispo_unit: str, base_name: str) -> float | None:
    """Convierte matemáticamente entre familias de unidades incompatibles (Masa/Vol/Unidad)."""
    if not base_name: return None
    density = VOLUMETRIC_DENSITIES.get(base_name)
    unit_weight = UNIT_WEIGHTS.get(base_name)
    
    if density is None:
        for k, v in VOLUMETRIC_DENSITIES.items():
            if k in base_name or base_name in k:
                density = v
                break
    if unit_weight is None:
        for k, v in UNIT_WEIGHTS.items():
            if k in base_name or base_name in k:
                unit_weight = v
                break

    if req_unit == 'g' and dispo_unit == 'ml' and density: return req_qty / density
    if req_unit == 'ml' and dispo_unit == 'g' and density: return req_qty * density
    if req_unit == 'g' and dispo_unit == 'unidad' and unit_weight: return req_qty / unit_weight
    if req_unit == 'unidad' and dispo_unit == 'g' and unit_weight: return req_qty * unit_weight
    if req_unit == 'ml' and dispo_unit == 'unidad' and density and unit_weight: return (req_qty * density) / unit_weight
    if req_unit == 'unidad' and dispo_unit == 'ml' and density and unit_weight: return (req_qty * unit_weight) / density
    return None

def validate_ingredients_against_pantry(generated_ingredients: list, pantry_ingredients: list) -> bool | str:
    """
    Función guardrail estricta y matemática. Comprueba:
    1. Que todos los ingredientes generados estén en la despensa.
    2. Que las CANTIDADES generadas no superen el Ledger de la despensa.
    """
    if not pantry_ingredients:
        logger.debug("⚠️ [PANTRY GUARD] Lista de despensa vacía — guardrail desactivado.")
        return True
        
    try:
        from shopping_calculator import _parse_quantity
    except ImportError:
        logger.error("❌ Falló import de _parse_quantity")
        _parse_quantity = lambda x: (0, 'u', x)

    pantry_ledger = {}
    pantry_bases = set()
    
    for p in pantry_ingredients:
        norm = normalize_ingredient_for_tracking(p)
        if norm: pantry_bases.add(norm)
        pantry_bases.add(strip_accents(p.lower().strip()))
        
        qty, unit, name = _parse_quantity(p)
        base_norm = normalize_ingredient_for_tracking(name) or strip_accents(name.lower().strip())
        
        if not base_norm: continue
            
        base_qty, base_unit = _to_base_unit(qty, unit)
        
        if base_norm not in pantry_ledger:
            pantry_ledger[base_norm] = {}
        if base_unit not in pantry_ledger[base_norm]:
            pantry_ledger[base_norm][base_unit] = 0.0
            
        pantry_ledger[base_norm][base_unit] += base_qty

    if not pantry_bases:
        return True
        
    unauthorized = []
    over_limit = []
    
    for item in generated_ingredients:
        item = item.strip()
        if not item: continue
        
        gen_qty, gen_unit, gen_name = _parse_quantity(item)
        gen_base_qty, gen_base_unit = _to_base_unit(gen_qty, gen_unit)
        base = normalize_ingredient_for_tracking(gen_name) or strip_accents(gen_name.lower().strip())
        
        item_lower = strip_accents(item.lower())
        allowed_condiments = {
            "sal", "pimienta", "agua", "ajo", "oregano", "cilantro", 
            "limon", "aceite", "soya", "canela", "vinagre"
        }
        
        if base in allowed_condiments or any(c in item_lower for c in allowed_condiments):
            continue
            
        matched_pantry_key = None
        if base in pantry_ledger:
            matched_pantry_key = base
        else:
            # 1. Intentar Regex/Subcadena tradicional
            for pb in pantry_bases:
                if pb and len(pb) > 2 and (pb in item_lower or pb in base):
                    matched_pantry_key = pb if pb in pantry_ledger else None
                    if not matched_pantry_key:
                        for k in pantry_ledger.keys():
                            if k in pb or pb in k:
                                matched_pantry_key = k
                                break
                    break
                    
            # 2. Si falló el match tradicional, intentamos Similitud Coseno (Mejora 4)
            if not matched_pantry_key and len(base) > 2:
                try:
                    gen_emb = get_embedding(base)
                    best_match = None
                    best_score = -1.0
                    
                    for p_key in pantry_ledger.keys():
                        if p_key not in _pantry_embeddings_cache:
                            _pantry_embeddings_cache[p_key] = get_embedding(p_key)
                        
                        p_emb = _pantry_embeddings_cache[p_key]
                        score = cosine_similarity(gen_emb, p_emb)
                        
                        if score > best_score:
                            best_score = score
                            best_match = p_key
                            
                    if best_score > 0.85: # Threshold estricto para evitar falsos positivos
                        logger.debug(f"🧠 [VECTOR MATCH] '{base}' -> '{best_match}' (score: {best_score:.3f})")
                        matched_pantry_key = best_match
                except Exception as e:
                    logger.warning(f"Error en Vector Search para '{base}': {e}")
                    
        if not matched_pantry_key:
            unauthorized.append(item)
            continue
            
        if gen_qty > 0 and matched_pantry_key in pantry_ledger:
            available_units_for_item = pantry_ledger[matched_pantry_key]
            
            # Tolerancia Inteligente: Si el inventario para TODAS las unidades de este ítem es 0,
            # no es un "exceso" sino un "faltante". Lo dejamos pasar — aparecerá en la lista de compras.
            total_available = sum(available_units_for_item.values())
            if total_available <= 0.01:
                logger.debug(f"🛒 [PANTRY GUARD] '{gen_name}' tiene inventario 0 — skip (se comprará).")
                continue
            
            if gen_base_unit in available_units_for_item:
                available_qty = available_units_for_item[gen_base_unit]
                if available_qty <= 0.01:
                    logger.debug(f"🛒 [PANTRY GUARD] '{gen_name}' en {gen_base_unit} tiene stock 0 — skip.")
                    continue
                if gen_base_qty > (available_qty * 1.15):
                    formatted_req = _format_unit_qty(gen_base_qty, gen_base_unit)
                    formatted_avail = _format_unit_qty(available_qty, gen_base_unit)
                    over_limit.append(f"[{item}] (Pediste {formatted_req}, límite: {formatted_avail})")
                else:
                    available_units_for_item[gen_base_unit] -= gen_base_qty
            else:
                # Conversión Matemática Activa (Mejora 1)
                converted = False
                for dispo_unit, available_qty in available_units_for_item.items():
                    if available_qty <= 0.01:
                        converted = True
                        break
                    req_qty_in_dispo_unit = _get_converted_quantity(gen_base_qty, gen_base_unit, dispo_unit, matched_pantry_key)
                    if req_qty_in_dispo_unit is not None:
                        if req_qty_in_dispo_unit > (available_qty * 1.15):
                            formatted_req = _format_unit_qty(gen_base_qty, gen_base_unit)
                            formatted_avail = _format_unit_qty(available_qty, dispo_unit)
                            over_limit.append(f"[{item}] (Pediste {formatted_req}, convertido dinámicamente excede tu inventario de {formatted_avail})")
                        else:
                            available_units_for_item[dispo_unit] -= req_qty_in_dispo_unit
                        converted = True
                        break 
                
                if not converted:
                    # Solución 3: Conversor por default (5g aprox por porción imprecisa: "pizca", "ramita", "chorrito")
                    fallback_g = gen_base_qty * 5.0
                    for dispo_unit, available_qty in available_units_for_item.items():
                        req_qty_in_dispo_unit = None
                        
                        # Conversiones directas de masa desde fallback a la unidad de la despensa
                        if dispo_unit in ['g', 'kg', 'lb', 'oz', 'ml', 'unidad']:
                            req_qty_in_dispo_unit = _get_converted_quantity(fallback_g, 'g', dispo_unit, matched_pantry_key)
                            if req_qty_in_dispo_unit is None:
                                if dispo_unit == 'g': req_qty_in_dispo_unit = fallback_g
                                elif dispo_unit == 'kg': req_qty_in_dispo_unit = fallback_g / 1000.0
                                elif dispo_unit == 'lb': req_qty_in_dispo_unit = fallback_g / 453.592
                                elif dispo_unit == 'oz': req_qty_in_dispo_unit = fallback_g / 28.3495
                                elif dispo_unit == 'ml': req_qty_in_dispo_unit = fallback_g # Asumir densidad agua para cosas raras
                        
                        if req_qty_in_dispo_unit is not None:
                            logger.debug(f"🔧 [PANTRY GUARD] Aplicando fallback de 5g/ut para '{gen_name}' ({gen_base_qty} {gen_base_unit} -> {req_qty_in_dispo_unit:.2f} {dispo_unit})")
                            if req_qty_in_dispo_unit > (available_qty * 1.15):
                                formatted_req = _format_unit_qty(gen_base_qty, gen_base_unit)
                                formatted_avail = _format_unit_qty(available_qty, dispo_unit)
                                over_limit.append(f"[{item}] (Pediste {formatted_req}, convertido con fallback [~{fallback_g}g] excede tu inventario de {formatted_avail})")
                            else:
                                available_units_for_item[dispo_unit] -= req_qty_in_dispo_unit
                            converted = True
                            break
                            
                    if not converted:
                        logger.debug(f"⚠️ [PANTRY GUARD] Unidades asintóticas TOTALMENTE irresolubles para {gen_name} (req: {gen_base_unit}). Aprobación flexible.")
            
    if unauthorized or over_limit:
        error_msg = "ERRORES DE DESPENSA HALLADOS OBLIGANDO A CORREGIR:\n"
        if unauthorized:
            error_msg += f"- Ingredientes COMPLETAMENTE INEXISTENTES en inventario: {', '.join(unauthorized)}.\n"
        if over_limit:
            error_msg += f"- Excediste tus CANTIDADES (Tu inventario restringe esto matemáticamente): {', '.join(over_limit)}.\n"
            
        error_msg += "Corrige tu respuesta bajando las porciones estrictamente numéricas al límite exacto, O eliminando/sustituyendo ingredientes."
        logger.warning(f"🚨 [PANTRY GUARD] RECHAZO | unauthorized={len(unauthorized)} | over_limit={len(over_limit)}")
        return error_msg
        
    logger.debug(f"✅ [PANTRY GUARD] APROBADO (Cantidades & Confiabilidad validadas)")
    return True
