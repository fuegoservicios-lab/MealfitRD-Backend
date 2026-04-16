import re
import math
import os
from collections import defaultdict
import logging
from fractions import Fraction
from db_core import supabase, connection_pool, execute_sql_query

import time as _time

_master_cache = None
_master_cache_ts = 0
_MASTER_CACHE_TTL = 300  # 5 minutos de TTL para que aliases nuevos se refresquen
_semantic_cache = None

def invalidate_master_cache():
    """Invalida el caché de master_ingredients para forzar recarga desde DB."""
    global _master_cache, _master_cache_ts, _semantic_cache
    _master_cache = None
    _master_cache_ts = 0
    _semantic_cache = None

def get_semantic_cache():
    global _semantic_cache
    if _semantic_cache is not None:
        return _semantic_cache
        
    master_list = get_master_ingredients()
    if not master_list:
        return None
        
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview", google_api_key=api_key)
        
        texts = [f"{m['name']} - Categoría: {m.get('category','')}. Alias: {', '.join(m.get('aliases') or [])}" for m in master_list]
        vectors = embeddings.embed_documents(texts)
        
        _semantic_cache = {
            "master_list": master_list,
            "vectors": vectors,
            "embeddings_client": embeddings
        }
        logging.info("🧠 Caché semántico local inicializado con éxito por primera vez.")
        return _semantic_cache
    except Exception as e:
        logging.error(f"Error inicializando caché semántico: {e}")
        return None

def cosine_similarity(v1, v2):
    dot = sum(a*b for a,b in zip(v1, v2))
    mag1 = math.sqrt(sum(a*a for a in v1))
    mag2 = math.sqrt(sum(a*a for a in v2))
    if mag1 == 0 or mag2 == 0: return 0
    return dot / (mag1 * mag2)

def get_master_ingredients():
    global _master_cache, _master_cache_ts
    now = _time.time()
    if _master_cache is None or (now - _master_cache_ts) > _MASTER_CACHE_TTL:
        if connection_pool:
            try:
                res = execute_sql_query("SELECT * FROM master_ingredients", fetch_all=True)
                _master_cache = res or []
                _master_cache_ts = now
            except Exception as e:
                logging.error(f"Error fetching master_ingredients via pool: {e}")
                if _master_cache is None:
                    _master_cache = []
        else:
            logging.error("No connection_pool available to fetch master_ingredients")
            if _master_cache is None:
                _master_cache = []
    return _master_cache

DEFAULT_G_PER_TAZA = 150

def parse_fraction(val: str) -> float:
    val = val.strip()
    try:
        if ' ' in val:
            parts = val.split(' ')
            if '/' in parts[1]:
                num, den = parts[1].split('/')
                return float(parts[0]) + float(num)/float(den)
        if '/' in val:
            num, den = val.split('/')
            return float(num)/float(den)
        return float(val)
    except Exception:
        return 0.0

def normalize_name(orig_name: str) -> str:
    n = str(orig_name).lower().strip()
    n = re.sub(r'\(.*?\)', '', n).strip()
    # Limpieza de prefijos contenedores o medidas informales
    n = re.sub(r'^(cda|cdta|cdita|cucharada|cucharadita|taza|vaso|pizca|chorrito|puñado|atado|manojo|scoop|lonja|loncha|paquete|paquetico|funda|lata|sobre|sobrecito|chin|toque)(s)?\s*(de\s+|del\s+)?', '', n, flags=re.IGNORECASE)
    # Nueva mejora: Limpieza estricta de pseudo-unidades anatómicas LATINAS SOLO si están seguidas de 'de'
    n = re.sub(r'^(pechuga|filete|muslo|trozo|chuleta|pieza|corte|ración|racion|porción|porcion|filetico|medallón|medallones|carne)(s)?\s+(de\s+|del\s+)', '', n, flags=re.IGNORECASE)
    n = re.sub(r'^(de\s+|del\s+)', '', n, flags=re.IGNORECASE)
    
    stops = ['cortado', 'cortada', 'cortados', 'cortadas', 'picado', 'picada', 'picados', 'picadas', 'picadito', 'picadita', 'picaditos', 'picaditas', 'pelado', 'pelada', 'pelados', 'peladas', 'hervido', 'hervida', 'hervidos', 'hervidas', 'cocido', 'cocida', 'cocidos', 'cocidas', 'asado', 'asada', 'asados', 'asadas', 'crudo', 'cruda', 'crudos', 'crudas', 'horneado', 'horneada', 'horneados', 'horneadas', 'desmenuzado', 'desmenuzada', 'desmenuzados', 'desmenuzadas', 'rallado', 'rallada', 'rallados', 'ralladas', 'guisado', 'guisada', 'guisados', 'guisadas', 'frito', 'frita', 'fritos', 'fritas', 'majado', 'majada', 'majados', 'majadas', 'triturado', 'triturada', 'triturados', 'trituradas', 'hecha puré', 'hecho puré', 'puré', 'en julianas', 'en tiras', 'en cubos', 'en hojuelas', 'en dados', 'en aros', 'en trozos', 'en rodajas', 'en porciones', 'en lonjas', 'en lonja', 'finamente', 'muy', 'pequeño', 'pequeña', 'pequeños', 'pequeñas', 'grande', 'grandes', 'mediano', 'mediana', 'medianos', 'medianas', 'maduro', 'madura', 'maduros', 'maduras', 'fresco', 'fresca', 'frescos', 'frescas', 'firme', 'firmes', 'entero', 'entera', 'enteros', 'enteras', 'fina', 'finas', 'gruesa', 'gruesas', 'magro', 'magra', 'magros', 'magras', 'natural', 'naturales', 'bajo en grasa', 'bajas en grasa', 'bajos en grasa', 'bajo en sodio', 'bajas en sodio', 'bajos en sodio', 'descremado', 'descremada', 'descremados', 'descremadas', 'sin sal', 'con sal', 'sin piel', 'sin hueso', 'para rebozar', 'al gusto', 'pizca de', 'rodajas de', 'de la despensa', 'ralladura y jugo de 1/2', 'la', 'el', 'los', 'las']
    clean_n = n
    for s in stops:
        clean_n = re.sub(r'\b' + s + r'\b', '', clean_n, flags=re.IGNORECASE)
        
    # Limpiar conjunciones o preposiciones que quedan colgadas al quitar los stops al inicio o al final
    clean_n = re.sub(r'^\s*(y|e|o|en|con|de|del|para)\b', '', clean_n, flags=re.IGNORECASE)
    clean_n = re.sub(r'\b(y|e|o|en|con|de|del|para)\s*$', '', clean_n, flags=re.IGNORECASE)
    clean_n = re.sub(r'\s+', ' ', clean_n).replace(',', '').strip()
    
    master_list = get_master_ingredients()
    from constants import strip_accents
    
    n_stripped = strip_accents(n)
    clean_n_stripped = strip_accents(clean_n)
    
    # Recolectar todos los aliases + nombres canónicos para búsqueda,
    # ordenados por longitud (más largos primero) para evitar que 
    # 'platano' se trague 'platano maduro' o 'queso' se trague 'queso cottage'
    all_aliases = []
    for master in master_list:
        # El nombre canónico también cuenta como alias para búsqueda exacta
        master_name = master["name"]
        all_aliases.append((strip_accents(master_name.strip().lower()), master_name))
        for alias in (master.get("aliases") or []):
            all_aliases.append((strip_accents(alias.strip().lower()), master_name))
            
    all_aliases.sort(key=lambda x: len(x[0]), reverse=True)

    # ── INTENTO 1: Match Exacto sobre el texto RAW (sin mutilar por stops) ──
    # Esto es CRÍTICO porque los stops eliminan palabras como 'natural', 'descremado',
    # 'bajo en grasa' que son parte de aliases legítimos como 'yogurt griego natural'.
    for alias_stripped, master_name in all_aliases:
        if n_stripped == alias_stripped:
            return master_name

    # ── INTENTO 2: Regex sobre el texto RAW (sin mutilar) ──
    # Buscar "queso mozzarella bajo en grasa" dentro de "queso mozzarella bajo en grasa rallado"
    for alias_stripped, master_name in all_aliases:
        if re.search(r'\b' + re.escape(alias_stripped) + r'\b', n_stripped, flags=re.IGNORECASE):
            return master_name

    # ── INTENTO 3: Match Exacto sobre clean_n (texto limpio, fallback) ──
    for alias_stripped, master_name in all_aliases:
        if clean_n_stripped == alias_stripped:
            return master_name

    # ── INTENTO 4: Regex sobre clean_n (último recurso antes de semántica) ──
    for alias_stripped, master_name in all_aliases:
        if re.search(r'\b' + re.escape(alias_stripped) + r'\b', clean_n_stripped, flags=re.IGNORECASE):
            return master_name

    # Intento 3: Búsqueda de Similitud Semántica Vectorial (Fallback Local)
    # Solo vale la pena gastar un request si la palabra no fue encontrada en absoluto y tiene suficiente longitud
    if len(n) > 3:
        cache = get_semantic_cache()
        if cache:
            try:
                # Calculamos el vector del texto no reconocido
                query_vector = cache["embeddings_client"].embed_query(n)
                best_score = -1.0
                best_match = None
                
                # Buscamos matemáticamente contra toda la tabla en milisegundos de RAM
                for i, master_vector in enumerate(cache["vectors"]):
                    score = cosine_similarity(query_vector, master_vector)
                    if score > best_score:
                        best_score = score
                        best_match = cache["master_list"][i]["name"]
                
                # Umbral de confianza estricto (0.70 o 70% de similitud)
                if best_score >= 0.70:
                    logging.info(f"🧠 [Semantic Search] Resuelto: '{orig_name}' -> '{best_match}' con score {best_score:.3f}")
                    return best_match
            except Exception as e:
                logging.error(f"Error en búsqueda semántica de '{orig_name}': {e}")

    if len(clean_n) > 0:
        return clean_n[0].upper() + clean_n[1:]
    return n

def _preprocess_nlp_quantities(s: str) -> str:
    s_lower = str(s).lower().strip()
    
    # Soporte nativo para fracciones Unicode al inicio
    fraction_map = {
        u"\u00BD": "1/2",  # ½
        u"\u00BC": "1/4",  # ¼
        u"\u00BE": "3/4",  # ¾
        u"\u2153": "1/3",  # ⅓
        u"\u2154": "2/3",  # ⅔
        u"\u2155": "1/5"   # ⅕
    }
    for k, v in fraction_map.items():
        if s_lower.startswith(k):
            s_lower = s_lower.replace(k, v + " ", 1)
            
    replacements = [
        (r'^un cuarto de\b', '1/4 de'),
        (r'^un cuarto\b', '1/4'),
        (r'^1 cuarto de\b', '1/4 de'),
        (r'^1 cuarto\b', '1/4'),
        (r'^tres cuartos de\b', '3/4 de'),
        (r'^tres cuartos\b', '3/4'),
        (r'^3 cuartos de\b', '3/4 de'),
        (r'^3 cuartos\b', '3/4'),
        (r'^un tercio de\b', '1/3 de'),
        (r'^un tercio\b', '1/3'),
        (r'^1 tercio de\b', '1/3 de'),
        (r'^1 tercio\b', '1/3'),
        (r'^media\b', '1/2'),
        (r'^medio\b', '1/2'),
        (r'^mitad de\b', '1/2 de'),
        (r'^mitad\b', '1/2'),
        (r'^un octavo de\b', '1/8 de'),
        (r'^un octavo\b', '1/8'),
        (r'^(cantidad necesaria|al gusto|al ojo)\s+(de\s+)?', '1 pizca de '),
        (r'^(un\s+)?chin\s+(de\s+)?', '1 chin de '),
        (r'^(un\s+)?chorrito\s+(de\s+)?', '1 chorrito de '),
        (r'^(un\s+)?toque\s+(de\s+)?', '1 toque de '),
        (r'^(una\s+)?pizca\s+(de\s+)?', '1 pizca de '),
        (r'^una\b', '1'),
        (r'^un\b', '1'),
        (r'^uno\b', '1'),
        (r'^dos\b', '2'),
        (r'^tres\b', '3'),
        (r'^cuatro\b', '4'),
        (r'^cinco\b', '5'),
        (r'^seis\b', '6'),
        (r'^siete\b', '7'),
        (r'^ocho\b', '8'),
        (r'^nueve\b', '9'),
        (r'^diez\b', '10')
    ]
    
    for pattern, repl in replacements:
        new_s = re.sub(pattern, repl, s_lower, count=1)
        if new_s != s_lower:
            return new_s.strip()
            
    return s.strip()

def _calculate_yield_multiplier(raw_name: str) -> float:
    n = raw_name.lower()
    # 1. Pastas y Granos cocidos (Expanden, necesitas menos crudo)
    if bool(re.search(r'\b(cocid[oa]|hervid[oa])\b', n)) and bool(re.search(r'\b(arroz|pasta|quinoa|lenteja|habichuela|frijol|guandul)\b', n)):
        return 0.35
    
    # 2. Proteínas cocidas (Se encogen por humedad, necesitas más crudo)
    if bool(re.search(r'\b(cocid[oa]|hervid[oa]|asad[oa]|hornead[oa]|desmenuzad[oa]|frit[oa])\b', n)) and bool(re.search(r'\b(pollo|carne|res|pescado|cerdo|camar|pavo|salm[oó]n|filete)\b', n)):
        return 1.35
        
    # 3. Merma de Cáscara/Limpieza (Víveres y Mariscos pelados)
    if bool(re.search(r'\b(pelad[oa]|limpi[oa]|sin piel|sin c[aá]scara)\b', n)) and bool(re.search(r'\b(yuca|platano|pl[aá]tano|batata|papa|guineo|camar[oó]n|manzana|pera)\b', n)):
        return 1.30
        
    # 4. Merma de Hueso (comprar sin hueso es más carne, pero si la receta pide carne magra y el ingrediente en lista es estándar)
    if bool(re.search(r'\b(sin hueso|deshuesad[oa])\b', n)) and bool(re.search(r'\b(pollo|muslo|carne|chuleta)\b', n)):
        return 1.40
        
    return 1.0

def _parse_quantity(s):
    if isinstance(s, dict):
        qty = float(s.get("quantity", 0))
        unit = s.get("unit", "unidad")
        if unit:
            unit = str(unit).strip().lower()
        if not unit:
            unit = "unidad"
        name_raw = s.get("name") or s.get("ingredient_name") or s.get("item_name") or "Desconocido"
        return qty, unit, normalize_name(name_raw).strip()

    s_lower = str(s).lower().strip()
    
    # Mejora 3: Si contiene términos puramente informales SIN NÚMEROS (ej: "sal al gusto")
    # los mandaremos como nominal 0.0 para no alterar matemáticamente la despensa pero sí listarlos.
    abstract_terms = ['al gusto', 'al ojo', 'cantidad necesaria']
    for term in abstract_terms:
        if term in s_lower and not any(char.isdigit() for char in s_lower):
            clean_s = s_lower.replace(term, '').replace(' de ', ' ').strip()
            return 0.0, 'pizca', normalize_name(clean_s).strip()
            
    s = _preprocess_nlp_quantities(s)
    # Limpieza previa: si el AI genera "1 Ud." o "2 Uds.", limpiar el punto
    s = re.sub(r'\b([Uu]ds?)\.', r'\1', s)
    match = re.search(r'^(\d+(?:\s+\d+\/\d+|\/\d+|\.\d+)?)\s*(?:de\s+)?([a-zA-ZáéíóúÁÉÍÓÚñÑ]+)?(?:\s+(.*))?$', s)
    if not match:
        return 0.0, 'cantidad necesaria', normalize_name(s).strip()
    
    qty_str = match.group(1)
    unit_str = match.group(2)
    rest_str = match.group(3) or ""
    
    raw_qty = parse_fraction(qty_str)
    
    yield_mult = _calculate_yield_multiplier(rest_str)
    qty = raw_qty * yield_mult
    
    if unit_str:
        u = unit_str.lower()
        if u in ['g', 'gr', 'gramos', 'gramo']: unit_str = 'g'
        elif u in ['kg', 'kilo', 'kilos', 'kilogramos', 'kilogramo']: unit_str = 'kg'
        elif u in ['lb', 'lbs', 'libra', 'libras']: unit_str = 'lb'
        elif u in ['oz', 'onza', 'onzas']: unit_str = 'oz'
        elif u in ['ml', 'mililitro', 'mililitros']: unit_str = 'ml'
        elif u in ['l', 'litro', 'litros']: unit_str = 'l'
        elif u in ['taza', 'tazas']: unit_str = 'taza'
        elif u in ['cda', 'cucharada', 'cucharadas']: unit_str = 'cda'
        elif u in ['cdta', 'cucharadita', 'cucharaditas', 'cdita']: unit_str = 'cdta'
        elif u in ['diente', 'dientes']: unit_str = 'diente'
        elif u in ['lata', 'latas']: unit_str = 'lata'
        elif u in ['paquete', 'paquetes', 'paquetico', 'paqueticos', 'pqte', 'paq', 'funda', 'fundas']: unit_str = 'paquete'
        elif u in ['sobre', 'sobres', 'sobrecito', 'sobrecitos']: unit_str = 'sobre'
        elif u in ['chin', 'pizca', 'pizcas', 'toque', 'toques', 'chorrito', 'chorritos', 'puñado', 'puñados', 'ramita', 'ramitas', 'hojita', 'hojitas']: unit_str = 'pizca'
        elif u in ['pote', 'potes', 'tarro']: unit_str = 'pote'
        elif u in ['botella', 'botellas', 'frasco']: unit_str = 'botella'
        elif u in ['cabeza', 'cabezas']: unit_str = 'cabeza'
        elif u in ['hoja', 'hojas']: unit_str = 'hoja'
        elif u in ['rebanada', 'rebanadas', 'lonja', 'lonjas']: unit_str = 'rebanada'
        elif u in ['unidad', 'unidades', 'ud', 'uds', 'unid']: unit_str = 'unidad'
        else:
            rest_str = unit_str + (" " + rest_str if rest_str else "")
            unit_str = 'unidad'
    else:
        unit_str = 'unidad'
        
    return qty, unit_str, normalize_name(rest_str).strip()
    
def get_plural_unit(num, u):
    if num <= 1 or not u: return u
    u_lower = u.lower()
    PLURALS = {
        'lb': 'lbs', 'lbs': 'lbs',
        'paquete': 'paquetes', 'pote': 'potes', 'unidad': 'unidades',
        'lata': 'latas', 'cabeza': 'cabezas', 'diente': 'dientes',
        'cartón': 'cartones', 'carton': 'cartones',
        'sobre': 'sobres', 'sobrecito': 'sobrecitos',
        'botella': 'botellas', 'frasco': 'frascos',
        'fundita': 'funditas', 'mazo': 'mazos', 'envase': 'envases',
        'rebanada': 'rebanadas', 'hoja': 'hojas',
        'cda': 'cdas', 'cdta': 'cdtas', 'taza': 'tazas',
        'ud.': 'Uds.',
    }
    result = PLURALS.get(u_lower, u)
    # Preservar capitalización del input: si "Pote" → "Potes", si "pote" → "potes"
    if len(result) > 0 and u[0].isupper() and result[0].islower():
        result = result[0].upper() + result[1:]
    return result

# Mínimos comprables en mercado/colmado dominicano
MARKET_MINIMUMS = {
    "lb": 0.25,       # No se vende menos de 1/4 lb
    "lbs": 0.25,
    "pote": 1,        # No puedes comprar "medio pote"
    "paquete": 1,     # Siempre se compra entero  
    "fundita": 1,
    "mazo": 1,
    "lata": 1,
    "sobre": 1,
    "sobrecito": 1,
    "frasco": 1,
    "botella": 1,
    "cartón": 1,
    "carton": 1,
    "envase": 1,
    "cabeza": 1,
    "ud.": 1,
    "ud": 1,
}

# Mapeo canónico de categorías DB → categorías de display para PDF
DISPLAY_CATEGORY_MAP = {
    "Proteínas":        "PROTEÍNAS",
    "Lácteos":          "LÁCTEOS",
    "Frutas":           "FRUTAS",
    "Vegetales":        "VEGETALES",
    "Víveres":          "VÍVERES",
    "Despensa":         "DESPENSA",
    "Despensa y Granos": "DESPENSA",
    "Especias":         "ESPECIAS",
    "Suplementos":      "SUPLEMENTOS",
}

def _get_display_category(db_category: str, name: str = "") -> str:
    """Resuelve la categoría de display para el PDF. Server-side, elimina regex del frontend."""
    if db_category in DISPLAY_CATEGORY_MAP:
        return DISPLAY_CATEGORY_MAP[db_category]
    # Fallback NLP para ingredientes sin categoría en DB
    n = name.lower()
    if re.search(r'pollo|carne|pescado|\bres\b|cerdo|huevo|camar|at[uú]n|sardina|pavo|jam[oó]n|tocineta|salchicha|longaniza|salami', n):
        return "PROTEÍNAS"
    if re.search(r'queso|leche|yogur|crema|ricotta|cottage|mozzarella|mantequilla|margarina', n):
        return "LÁCTEOS"
    if re.search(r'manzana|guineo|naranja|fresa|chinola|mango|pi[ñn]a|lechosa|aguacate|lim[oó]n|pera|uva|mel[oó]n|sand[ií]a|kiwi|cereza|durazno|banana', n):
        return "FRUTAS"
    if re.search(r'tomate|cebolla|aj[ií]|zanahoria|br[oó]coli|espinaca|lechuga|pepino|ajo|cilantro|apio|repollo|coliflor|tayota|berenjena|vainita|molondr|auyama|jengibre|r[aá]bano|pimiento|habichuel[ií]ta', n):
        return "VEGETALES"
    if re.search(r'pl[aá]tano|papa|yuca|batata|yaut[ií]a|[ñn]ame|guine[ií]to', n):
        return "VÍVERES"
    if re.search(r'arroz|pasta|avena|harina|habichuela|frijol|lenteja|garbanzo|quinoa|guand[uú]l|\bpan\b', n):
        return "DESPENSA"
    if re.search(r'aceite|\bsal\b|pimienta|or[eé]gano|canela|comino|vinagre|miel|salsa|semilla|almendra|nuez|man[ií]|ch[ií]a|az[uú]car|caf[eé]|saz[oó]n', n):
        return "DESPENSA"
    return "OTROS"

# ═══════════════════════════════════════════════════════════════
# Helpers para SKU-Aware Sizing (P3)
# ═══════════════════════════════════════════════════════════════
def _find_best_sku(g_total: float, available_sizes_g: list, anti_waste_pct: float = 0.10):
    """Encuentra la combinación óptima de SKUs para minimizar desperdicio.
    
    Estrategias (en orden de prioridad):
      1. Single-SKU: paquete más pequeño que cubre la necesidad (≤20% waste, ≤2x tamaño)
      2. Best-Fit Multi: prueba TODOS los tamaños, elige el que minimiza desperdicio
    
    Returns: (count, size_g) — cuántos paquetes de qué tamaño
    """
    import math
    sizes = sorted([float(s) for s in available_sizes_g])  # ascendente
    
    # Estrategia 1: Un solo paquete que cubre la necesidad
    # Tolerancia muy ajustada (5%) para obligar escalar visualmente cuando aumentan personas.
    SINGLE_PKG_TOLERANCE = 0.05
    for size in sizes:
        if size >= g_total and size <= g_total * 2:
            waste_pct = (size - g_total) / size
            if waste_pct <= SINGLE_PKG_TOLERANCE:
                return 1, size
    
    # Estrategia 2: Prueba cada tamaño disponible, elige el mejor
    # Criterio: mínimo desperdicio con mínimo conteo de paquetes
    best_result = None
    best_waste = float('inf')
    
    for size in sizes:
        if size < g_total * 0.15:  # Skip tamaños ridículamente pequeños
            continue
        raw_count = g_total / size
        floor_count = math.floor(raw_count)
        frac = raw_count - floor_count
        
        # Anti-desperdicio: redondear abajo si sobra poco
        if frac <= anti_waste_pct and floor_count >= 1:
            count = floor_count
            total_g = count * size
            waste = max(0, g_total - total_g)  # Under-buy waste (aceptable si <10%)
        else:
            count = max(1, math.ceil(raw_count))
            total_g = count * size
            waste = total_g - g_total  # Over-buy waste
        
        waste_score = waste / g_total if g_total > 0 else 0
        # Penalizar conteos altos exponencialmente: 1 paquete siempre > N paquetes
        # count^1.5: 1→0.04, 2→0.11, 3→0.21, 4→0.32, 5→0.45
        score = waste_score + (count ** 1.5 * 0.04)
        
        if score < best_waste:
            best_waste = score
            best_result = (count, size)
    
    return best_result if best_result else (1, sizes[0])

def to_unicode_fraction(frac_str: str) -> str:
    mapping = {"1/4": "¼", "1/2": "½", "3/4": "¾"}
    return mapping.get(frac_str, frac_str)


def _sku_size_label(size_g: float, unit_hint: str = None) -> str:
    """Convierte gramos a etiqueta legible de mercado dominicano.
    
    453g → '1lb', 908g → '2lb', 473g → '473ml', 946g → '946ml', 200g → '200g'
    Con soporte especial para potes/frascos en onzas fluidas.
    """
    if size_g is None:
        return ""
    size_g = float(size_g)
    if unit_hint and unit_hint.lower() in ['cartón', 'carton', 'botella', 'ml', 'l', 'galón', 'envase', 'lata']:
        # Tamaños de volumen conocidos (leche, jugos — se venden por ml, no por peso)
        VOLUME_LABELS = {250: "250ml", 473: "473ml", 946: "946ml", 1000: "1L", 1892: "1/2 Galón"}
        for vol_g, label in VOLUME_LABELS.items():
            if abs(size_g - vol_g) < 10:
                return label
            
    if unit_hint and unit_hint.lower() in ['pote', 'frasco']:
        # Mapeos típicos de onzas para potes (yogurt, queso crema, aceitunas)
        if abs(size_g - 453.592) < 15: return "16 oz"
        if abs(size_g - 226.796) < 15: return "8 oz"
        if abs(size_g - 340.194) < 15: return "12 oz"
    
    lbs = size_g / 453.592
    # Libras enteras limpias — threshold estricto (±2%) para no confundir 473g con 1lb
    if abs(lbs - round(lbs)) < 0.05 and round(lbs) >= 1:
        return f"{round(lbs)} lb" if round(lbs) == 1 else f"{round(lbs)} lbs"
    # Media libra
    if abs(lbs - 0.5) < 0.05:
        return "½ lb"
    if abs(lbs - 0.25) < 0.05:
        return "¼ lb"
        
    # Mejorar la etiqueta para pesos de mega frutas o porciones grandes (ej. 800g -> ~1.8 lbs)
    if lbs > 1.2:
        return f"{round(lbs, 1):g} lbs"
        
    # Todo lo demás en gramos
    return f"{int(size_g)}g"


def apply_smart_market_units(name: str, weight_in_lbs: float, unit_str: str, raw_qty: float, master_item: dict = None):
    """Motor determinístico de unidades de mercado dominicano.
    
    Flujo de resolución (4 bloques, sin hardcoded weights):
      1. DB Container: market_container + container_weight_g → Potes, Paquetes, Cartones, etc.
         1a. SKU-Aware: si hay available_sizes_g, optimiza tamaño de empaque
      2. DB Density:   density_g_per_unit → Unidades físicas (frutas, vegetales, huevos)
      3. Dominican Lbs: Fracciones de libra (1/4, 1/2, 3/4) para carnes, quesos, granel
      4. Raw Fallback:  Cantidades crudas del AI sin conversión
    
    Returns dict con confidence_score (1.0=DB+SKU, 0.95=DB, 0.85=density, 0.75=lbs, 0.5=raw)
    """
    import math
    from constants import UNIT_WEIGHTS
    import unicodedata
    n_lower = name.lower()
    
    if master_item is None:
        master_item = {}
        
    cat = (master_item.get("category") or "").lower()
    density_per_u = master_item.get("density_g_per_unit")
    if density_per_u is not None:
        density_per_u = float(density_per_u)

    # Fallback Semántico si no hay densidad en Supabase
    if not density_per_u:
        from constants import UNIT_WEIGHTS
        n_clean = ''.join(c for c in unicodedata.normalize('NFD', n_lower) if unicodedata.category(c) != 'Mn')
        # Búsqueda exacta o como palabra entera para evitar "agua" == "pan de agua"
        for k, v in UNIT_WEIGHTS.items():
            if k == n_clean or (re.search(rf'\b{re.escape(k)}(s|es)?\b', n_clean)):
                density_per_u = v
                break
        # Fallback plurales multi-palabra: "guineitos verdes" → "guineito verde"
        if not density_per_u:
            n_singular = re.sub(r'(es|s)\b', '', n_clean).strip()
            for k, v in UNIT_WEIGHTS.items():
                if k == n_singular or n_singular.startswith(k) or k.startswith(n_singular):
                    density_per_u = v
                    break

    # Autocorrección de Alucinaciones (unidades líquidas para sólidos)
    if unit_str.lower() in ['ml', 'l', 'lt', 'oz', 'onzas'] and re.search(r'queso|pollo|cerdo|carne|arroz|avena|lenteja|habichuela|almendra', n_lower):
        if weight_in_lbs <= 0 and raw_qty > 0:
            weight_in_lbs = raw_qty / 453.59 if unit_str.lower() in ['g', 'ml'] else raw_qty / 16.0
        unit_str = 'lb'
        
    was_unitarized = False
    display_qty = ""
    market_qty = weight_in_lbs if weight_in_lbs > 0 else raw_qty
    market_unit = "lbs" if weight_in_lbs > 0 else unit_str
    confidence = 0.5  # Default: raw fallback
    sku_label = None   # None = no SKU optimization applied

    # Guards mínimos para Bloques 2 y 3 (solo 2 regex, eliminados los 15+ anteriores)
    is_meat_seafood = bool(re.search(r'\b(pollo|cerdo|carne|res|pescado|camar[oó]n|camarones|mariscos?|filetes?|chuletas?|longanizas?|salamis?|jam[oó]n|pavo|tocineta|bacon|salchichas?)\b', n_lower))
    is_cheese = bool(re.search(r'\b(quesos?|mozzarella|cheddar|parmesano|gouda|dan[eé]s)\b', n_lower)) and not re.search(r'\b(crema|mantequilla)\b', n_lower)

    # Nuevas clasificaciones Nivel de Producción (Actualizado con plurales y más alimentos)
    is_native_countable = bool(re.search(r'\b(pl[aá]tanos?|guineos?|lim[oó]n|limones|huevos?|manzanas?|naranjas?|peras?|chinolas?|mandarinas?|kiwis?|duraznos?)\b', n_lower))
    is_mega_fruit = bool(re.search(r'\b(aguacates?|pi[ñn]as?|sand[ií]as?|mel[oó]n|melones|lechosas?|papayas?)\b', n_lower))
    is_native_weighable = bool(re.search(r'\b(zanahorias?|tomates?|aj[ií]es?|cebollas?|papas?|yucas?|batatas?|berenjenas?|tayotas?|remolachas?|calabac[ií]nes?|calabac[ií]n|auyamas?|vegetales|[ñn]ames?|yaut[ií]as?|pimientos?|chiles?)\b', n_lower))
    is_native_cabeza = bool(re.search(r'\b(br[oó]colis?|coliflor|repollos?|lechugas?)\b', n_lower))
    is_herb_mazo = bool(re.search(r'\b(cilantro|cilantrico|puerro|perejil|menta|albahaca|romero|verdura|verdurita|recao|eneldo)\b', n_lower))

    # ═══════════════════════════════════════════════════════════════
    # BLOQUE 1: Resolución Data-Driven (PRIORIDAD MÁXIMA)
    # Usa market_container + container_weight_g directamente de la DB.
    # Cubre: Lácteos(Pote/Cartón), Despensa(Paquete/Fundita/Botella),
    #         Especias(Sobre), Vegetales(Mazo/Cabeza/Lata), etc.
    # Anti-desperdicio (Ahora estricto): 2% de colchón para errores de coma flotante. 
    # Forzará compras mayores a la mínima escalada matemática (Ej: 4 personas vs 6).
    ANTI_WASTE_THRESHOLD = 0.02

    db_container = master_item.get("market_container")
    db_container_weight_g = master_item.get("container_weight_g")
    available_sizes = master_item.get("available_sizes_g")
    
    if db_container and db_container_weight_g and weight_in_lbs > 0:
        g_total = weight_in_lbs * 453.592
        
        # ── SKU-Aware Path: múltiples tamaños disponibles ──
        if available_sizes and isinstance(available_sizes, list) and len(available_sizes) > 1:
            sku_count, sku_size_g = _find_best_sku(g_total, available_sizes, ANTI_WASTE_THRESHOLD)
            sku_label = _sku_size_label(sku_size_g, db_container)
            display_qty = f"{sku_count} {get_plural_unit(sku_count, db_container)} ({sku_label})"
            market_qty = sku_count
            market_unit = db_container
            was_unitarized = True
            confidence = 1.0
        else:
            # ── Standard Path: tamaño único de envase ──
            container_weight_g = float(db_container_weight_g)
            if container_weight_g > 0:
                raw_units = g_total / container_weight_g
                floor_units = math.floor(raw_units)
                frac = raw_units - floor_units
                if frac <= ANTI_WASTE_THRESHOLD and floor_units >= 1:
                    units_needed = floor_units
                else:
                    units_needed = max(1, math.ceil(raw_units))
                
                sku_label = _sku_size_label(container_weight_g, db_container)
                display_qty = f"{units_needed} {get_plural_unit(units_needed, db_container)} ({sku_label})"
                market_qty = units_needed
                market_unit = db_container
                was_unitarized = True
                confidence = 0.95

    # ═══════════════════════════════════════════════════════════════
    # BLOQUE 1.5: Intercepción de Hierbas Flexibles (Nivel 5)
    # Siempre se compran por mazo o atadito en RD, evitando "1/4 lb" o "15g"
    # ═══════════════════════════════════════════════════════════════
    if not was_unitarized and is_herb_mazo:
        g_total = (weight_in_lbs * 453.592) if weight_in_lbs > 0 else 0
        if unit_str.lower() in ['mazo', 'mazos', 'atado', 'atados']:
            units_needed = max(1, math.ceil(raw_qty))
        else:
            units_needed = max(1, math.ceil(g_total / 50.0))  # 1 mazo ≈ 50g
            
        display_qty = f"{units_needed} {'Mazo' if units_needed == 1 else 'Mazos'}"
        market_qty = units_needed
        market_unit = "Mazo"
        was_unitarized = True
        confidence = 0.90

    # ═══════════════════════════════════════════════════════════════
    # BLOQUE 2: Conversión Matemática → Unidades Físicas
    # Para items vendidos por unidad con density_g_per_unit (frutas,
    # vegetales unitarios, huevos, plátanos, etc.)
    # Excluye carnes/quesos (se venden por peso en RD).
    # Guard anti-absurdo: items muy pequeños (vainitas 10g, molondrones 15g)
    # con conteos altos → mejor por libra.
    # ═══════════════════════════════════════════════════════════════
    if not was_unitarized and weight_in_lbs > 0 and density_per_u and not re.search(r'lata|envase|ud|frasco|pote|caja', unit_str.lower()):
        if not is_meat_seafood and not is_cheese:
            g_total = weight_in_lbs * 453.592
            raw_count = g_total / density_per_u
            floor_count = math.floor(raw_count)
            frac = raw_count - floor_count
            # Anti-desperdicio: si necesitas <10% de una unidad extra, no comprarla
            if frac <= ANTI_WASTE_THRESHOLD and floor_count >= 1:
                units_count = floor_count
            else:
                units_count = max(1, math.ceil(raw_count))
            
            # Guard: "20 vainitas" no tiene sentido → "1/2 lb de vainitas"
            # También, si la densidad es extremadamente baja (<= 15g) como vainitas, molondrones, fresas,
            # nunca debería venderse por unidad a menos que sea ajo (que se calcula por cabeza/diente).
            is_absurd = (units_count > 6 and density_per_u < 50) or (density_per_u <= 15 and "ajo" not in n_lower)
            
            if not is_absurd:
                if is_native_weighable:
                    # Enfoque Híbrido Priorizado a Peso: "1 lb (~5 Uds)"
                    lbs_for_weighable = (units_count * density_per_u) / 453.592
                    whole = math.floor(lbs_for_weighable)
                    frac_w = lbs_for_weighable - whole
                    fraction_str = ""
                    if frac_w < 0.15: fraction_str = ""
                    elif frac_w <= 0.35: fraction_str = "1/4"
                    elif frac_w <= 0.65: fraction_str = "1/2"
                    elif frac_w <= 0.85: fraction_str = "3/4"
                    else: 
                        fraction_str = ""
                        whole += 1
                        
                    if whole == 0 and not fraction_str:
                        # Si es muy ligero, forzar a "1/4 lb" o unidades puras si es excepcionalmente pequeño
                        unit_text = "Ud." if units_count == 1 else "Uds."
                        display_qty = f"{units_count} {unit_text}"
                        market_qty = units_count
                        market_unit = "Ud."
                        sku_label = None
                    else:
                        if whole > 0 and fraction_str: 
                            weight_lbl = f"{whole} {to_unicode_fraction(fraction_str)} lbs"
                            market_qty_str = f"{whole} {fraction_str}"
                        elif whole > 0: 
                            weight_lbl = f"{whole} {'lb' if whole == 1 else 'lbs'}"
                            market_qty_str = whole
                        else: 
                            weight_lbl = f"{to_unicode_fraction(fraction_str)} lb"
                            market_qty_str = fraction_str
                            
                        # Limpiamos visualmente
                        display_qty = f"{weight_lbl} (~{units_count} {'Ud.' if units_count == 1 else 'Uds.'})"
                        market_qty = market_qty_str
                        market_unit = "lb" if whole <= 1 and not (whole==1 and fraction_str) else "lbs"
                        sku_label = None
                        
                    was_unitarized = True
                    confidence = 0.85

                else:
                    unit_text = "Ud." if units_count == 1 else "Uds."
                    if is_native_cabeza or re.search(r'\bajo\b', n_lower): unit_text = "Cabeza" if units_count == 1 else "Cabezas"
                    
                    if is_native_countable:
                        # Sin sufijo para "plátanos" o "huevos"
                        sku_label = None
                    else:
                        # Mega Frutas y demás tendrán su etiqueta de peso estimado (~X lbs)
                        approx_weight_label = _sku_size_label(density_per_u * units_count)
                        if approx_weight_label:
                            sku_label = f"~{approx_weight_label}"
                        else:
                            sku_label = None

                    display_qty = f"{units_count} {unit_text}"
                    if sku_label: display_qty += f" ({sku_label})"

                    market_qty = units_count
                    market_unit = "Ud." if "Cabeza" not in unit_text else "Cabeza"
                    was_unitarized = True
                    confidence = 0.85

    # ═══════════════════════════════════════════════════════════════
    # BLOQUE 3: Escala Mercado Dominicano para Pesos
    # Para carnes, quesos, y cualquier item sin envase estándar.
    # Redondea a fracciones de libra reales: 1/4, 1/2, 3/4
    # ═══════════════════════════════════════════════════════════════
    if not was_unitarized and weight_in_lbs > 0:
        if weight_in_lbs < 0.23:
            # Mínimo comprable en colmado dominicano: 1/4 lb
            display_qty = "¼ lb"
            market_qty = "1/4"
            market_unit = "lb"
            confidence = 0.75
        else:
            whole = math.floor(weight_in_lbs)
            frac = weight_in_lbs - whole
            fraction_str = ""
            
            if frac < 0.15: fraction_str = ""
            elif frac <= 0.35: fraction_str = "1/4"
            elif frac <= 0.65: fraction_str = "1/2"
            elif frac <= 0.85: fraction_str = "3/4"
            else: 
                fraction_str = ""
                whole += 1
                
            if whole > 0 and fraction_str:
                display_qty = f"{whole} {to_unicode_fraction(fraction_str)} lbs"
                market_qty = f"{whole} {fraction_str}"
                market_unit = "lbs"
            elif whole > 0:
                display_qty = f"{whole} {'lb' if whole == 1 else 'lbs'}"
                market_qty = whole
                market_unit = "lb" if whole == 1 else "lbs"
            elif fraction_str:
                display_qty = f"{to_unicode_fraction(fraction_str)} lb"
                market_qty = fraction_str
                market_unit = "lb"
            else:
                display_qty = "¼ lb"
                market_qty = "1/4"
                market_unit = "lb"
            confidence = 0.75

    # ═══════════════════════════════════════════════════════════════
    # BLOQUE 4: Fallback para formatos crudos sin peso aplicable
    # ═══════════════════════════════════════════════════════════════
    if not display_qty:
        if raw_qty > 0:
            if unit_str in ['unidad', 'unidades', 'paquete', 'paquetes', 'lata', 'latas', 'sobre', 'sobres', 'frasco', 'pote', 'potes', 'cartón', 'carton', 'botella', 'botellas', 'envase', 'envases', 'funda', 'fundas', 'fundita', 'funditas', 'mazo', 'mazos', 'cabeza', 'cabezas']:
                q_rounded = f"{math.ceil(raw_qty)}"
            else:
                q_rounded = f"{raw_qty:.2f}".rstrip('0').rstrip('.')
            if q_rounded == "": q_rounded = "1"
            
            if unit_str == 'unidad' or unit_str == 'unidades':
                if db_container:
                     display_qty = f"{q_rounded} {get_plural_unit(float(q_rounded) if '.' in q_rounded else int(q_rounded), db_container)}"
                     market_unit = db_container
                     sku_label = _sku_size_label(db_container_weight_g, db_container)
                     if sku_label: display_qty += f" ({sku_label})"
                else:
                     display_qty = f"{q_rounded} {'Ud.' if str(q_rounded) == '1' else 'Uds.'}"
                     market_unit = "Ud."
            else:
                display_qty = f"{q_rounded} {get_plural_unit(raw_qty, unit_str)}"
                
            market_qty = float(q_rounded) if '.' in q_rounded else int(q_rounded)
        else:
            display_qty = "Al gusto"
            market_qty = 0
            market_unit = "Al gusto"

    # ═══ Formato Final ═══
    if "Al gusto" in display_qty or "Pizca" in display_qty:
        final_str = f"{display_qty} de {name}"
    elif market_unit in ["Ud.", "Uds.", "Cabeza", "Cabezas", "Mazo", "Mazos"]:
        final_str = f"{display_qty} {name}"
    else:
        final_str = f"{display_qty} de {name}"

    final_str = final_str.replace(" de de ", " de ")

    formatted_market_qty = str(market_qty) if isinstance(market_qty, str) else round(market_qty, 2) if isinstance(market_qty, (float, int)) else market_qty

    def _parse_market_qty(mq):
        if isinstance(mq, (int, float)):
            return float(mq)
        if isinstance(mq, str) and '/' in mq:
            try:
                parts = mq.strip().split()
                if len(parts) == 2:
                    num, den = parts[1].split('/')
                    return float(parts[0]) + float(num)/float(den)
                else:
                    num, den = mq.strip().split('/')
                    return float(num)/float(den)
            except:
                return 0.0
        return 0.0

    numeric_qty = _parse_market_qty(formatted_market_qty)

    # Enforcement de mínimos comprables interactuando con reglas culturales
    if numeric_qty > 0 and market_unit.lower() in MARKET_MINIMUMS:
        min_qty = MARKET_MINIMUMS[market_unit.lower()]
        
        # Nivel de Producción: Carnes crudas mínimo 1/2 libra (excepto embutidos/deli)
        if market_unit.lower() in ['lb', 'lbs'] and is_meat_seafood and not re.search(r'\b(jam[oó]n|tocineta|bacon|salami|longaniza)\b', n_lower):
            min_qty = 0.5
            
        if numeric_qty < min_qty:
            formatted_market_qty = min_qty
            market_qty = min_qty
            if market_unit.lower() in ['lb', 'lbs']:
                frac_str = ""
                whole_min = math.floor(min_qty)
                frac_min = min_qty - whole_min
                if abs(frac_min - 0.25) < 0.1: frac_str = "1/4"
                elif abs(frac_min - 0.5) < 0.1: frac_str = "1/2"
                elif abs(frac_min - 0.75) < 0.1: frac_str = "3/4"
                
                if whole_min > 0 and frac_str: display_qty = f"{whole_min} {to_unicode_fraction(frac_str)} lbs"
                elif whole_min > 0: display_qty = f"{whole_min} {'lb' if whole_min == 1 else 'lbs'}"
                elif frac_str: display_qty = f"{to_unicode_fraction(frac_str)} lb"
                else: display_qty = f"{min_qty} lb"
                
                # Resincronizar market_qty fraccionado si aplica
                if whole_min == 0 and frac_str: formatted_market_qty = frac_str
                elif whole_min > 0 and frac_str: formatted_market_qty = f"{whole_min} {frac_str}"
                
            else:
                display_qty = f"{int(min_qty)} {market_unit}"
                
            if market_unit.lower() in ["ud.", "uds.", "cabeza", "cabezas", "mazo", "mazos"]:
                final_str = f"{display_qty} {name}"
            else:
                final_str = f"{display_qty} de {name}"

    # Preservar la cadena híbrida construida a la perfección (ej: "1/2 lb (~5 Uds.)") 
    # El código antiguo sobreescribía esta variable robando inteligencia.
    display_qty_final = display_qty
        
    # Nivel de Producción: Si logramós extraer un sku_size_label útil (tamaño paquete o aprox peso), anexarlo
    if sku_label and f"({sku_label})" not in display_qty_final:
        display_qty_final = f"{display_qty_final} ({sku_label})"

    result = {
        "name": name,
        "market_qty": formatted_market_qty,
        "market_unit": market_unit,
        "display_qty": display_qty_final,
        "display_string": final_str,
        "confidence_score": confidence,
        "shelf_life_days": master_item.get("shelf_life_days") if master_item else None
    }
    if sku_label:
        result["sku_size_label"] = sku_label
    return result


def aggregate_and_deduct_shopping_list(plan_ingredients: list[str], consumed_ingredients: list[str] = None, categorize: bool = False, structured: bool = False, multiplier: float = 1.0):
    aggregated = defaultdict(lambda: defaultdict(float))
    
    if consumed_ingredients is None:
        consumed_ingredients = []
    
    plan_names = set()
    for item in plan_ingredients:
        if not item or len(item) < 3: continue
        qty, unit, name = _parse_quantity(item)
        if not name: continue
        if name.lower() in ["ola", "olas"]: name = "Cebolla"
        aggregated[name][unit] += float(qty) * float(multiplier)
        plan_names.add(name)

    logging.info(f"🛒 [AGGREGATE] {len(plan_ingredients)} raw items → {len(plan_names)} unique names: {sorted(plan_names)[:30]}...")

    for item in consumed_ingredients:
        if not item or len(item) < 3: continue
        qty, unit, name = _parse_quantity(item)
        if not name: continue
        if name.lower() in ["ola", "olas"]: name = "Cebolla"
        aggregated[name][unit] -= float(qty)

    # --- RESOLUCIÓN DE FRICCIÓN DE UNIDADES (Híbridas) ---
    master_list = get_master_ingredients()
    # Mapeo por nombre canónico + aliases para resolución robusta
    master_map = {}
    for m in master_list:
        master_map[m["name"]] = m
        # Indexar todos los aliases para resolución fuzzy
        for alias in (m.get("aliases") or []):
            master_map[alias.strip().lower()] = m
            # También indexar con capitalización Title
            master_map[alias.strip().title()] = m

    # ── Re-agrupación por Nombre Canónico ──
    # Si el LLM devolvió "Huevo", "Huevos" y "Huevos enteros", el agregador original
    # los tiene como 3 llaves. Aquí los fusionamos en la llave canónica oficial ("Huevos")
    # para que su volumen se sume correctamente antes de calcular empaques comerciales.
    canonical_aggregated = defaultdict(lambda: defaultdict(float))
    for name, units in aggregated.items():
        m_item = master_map.get(name) or master_map.get(name.lower()) or master_map.get(name.title())
        canonical_name = m_item["name"] if m_item else name

        # Consolidación dura para huevos y sus derivados
        _can_lower = canonical_name.lower()
        if re.search(r'^(huevo|huevos|clara de huevo|claras de huevo|claras de huevo hervidas|clara de huevo hervida|yema de huevo|yemas de huevo|huevo hervido|huevos hervidos|huevos enteros|huevo entero)$', _can_lower, re.IGNORECASE):
            canonical_name = 'Huevo'

        # Consolidación: Ñame variantes (blanco, amarillo, etc.) → Ñame
        if re.search(r'^[ñn]ame\b', _can_lower):
            canonical_name = 'Ñame'

        # Consolidación: Miel variantes → Miel
        if re.search(r'^miel\b', _can_lower):
            canonical_name = 'Miel'

        # Consolidación: Ajo con preparación (majado, triturado) o "diente de ajo" → Ajo (no ajo en polvo)
        if (re.search(r'^ajo\b', _can_lower) or re.search(r'dientes?\s+de\s+ajo', _can_lower)) and 'polvo' not in _can_lower:
            canonical_name = 'Ajo'

        # Consolidación: Pechuga de pavo variantes (en lonjas, picadita, etc.) → Jamón de pavo
        if re.search(r'pavo', _can_lower) and re.search(r'(pechuga|lonjas?|rebanada|picadit[oa])', _can_lower):
            canonical_name = 'Jamón de pavo'

        # Consolidación: Fresas variantes (congeladas, frescas) → Fresas
        if re.search(r'^fresas?\b', _can_lower):
            canonical_name = 'Fresas'

        # Consolidación: Almendras variantes → Almendras fileteadas
        if re.search(r'^almendras?\b', _can_lower) and 'mantequilla' not in _can_lower:
            canonical_name = 'Almendras fileteadas'

        # Consolidación: Orégano variantes (seco, dominicano) → Orégano dominicano
        if re.search(r'^or[eé]gano\b', _can_lower):
            canonical_name = 'Orégano dominicano'

        # Consolidación: Tortilla/Tortillas integral/integrales → Tortilla integral
        if re.search(r'^tortillas?\s+integral', _can_lower):
            canonical_name = 'Tortilla integral'

        for u, q in units.items():
            canonical_aggregated[canonical_name][u] += q
            
    aggregated = canonical_aggregated

    for name, units in aggregated.items():
        master_item = master_map.get(name) or master_map.get(name.lower()) or master_map.get(name.title()) or {}
        
        # --- Normalización Universal por Peso ---
        # Si un ingrediente se contabilizó en conteos/volúmenes o incluso en contenedores (pote, lata)
        # pero tenemos constancia en BD de su peso (density/container), lo sumamos hacia el gramo
        # para que fluya hacia el Bloque 1/2 y asigne empaques matemáticamente exactos.
        g_per_taza = float(master_item.get("density_g_per_cup") or 0)
        g_per_u = float(master_item.get("density_g_per_unit") or 0)
        
        # [Fallback] Si no hay densidad en la BD, buscamos en constants
        if g_per_u <= 0 or g_per_taza <= 0:
            from constants import UNIT_WEIGHTS, strip_accents, VOLUMETRIC_DENSITIES
            n_clean = strip_accents(name.lower())
            
            if g_per_u <= 0:
                for k, v in UNIT_WEIGHTS.items():
                    if k == n_clean or (re.search(rf'\b{re.escape(k)}(s|es)?\b', n_clean)):
                        g_per_u = v
                        break
                # Fallback para plurales multi-palabra: singularizar cada palabra del input
                # Ej: "guineitos verdes" → "guineito verde" para matchear UNIT_WEIGHTS
                if g_per_u <= 0:
                    n_singular = re.sub(r'(es|s)\b', '', n_clean).strip()
                    for k, v in UNIT_WEIGHTS.items():
                        if k == n_singular or n_singular.startswith(k) or k.startswith(n_singular):
                            g_per_u = v
                            break
                        
            if g_per_taza <= 0:
                for k, v in VOLUMETRIC_DENSITIES.items():
                    if k == n_clean or (re.search(rf'\b{re.escape(k)}(s|es)?\b', n_clean)):
                        # VOLUMETRIC_DENSITIES es g/ml, 1 taza = 236.588 ml
                        g_per_taza = v * 236.588
                        break
        
        if g_per_taza <= 0:
            g_per_taza = DEFAULT_G_PER_TAZA

        container_weight_g = float(master_item.get("container_weight_g") or 0)
        db_container = (master_item.get("market_container") or "").lower()
        
        # Guardamos llaves en lista para modificar diccionario on-the-fly
        
        # Consolidation para Ajo
        if name.lower() == 'ajo':
            u_dientes = 0
            for k in list(units.keys()):
                if k.strip().lower() in ['diente', 'dientes', 'diente.', 'dientes.']:
                    u_dientes += units.pop(k)
            if u_dientes > 0:
                units['cabeza'] = units.get('cabeza', 0) + (u_dientes / 10.0)
                
        # Empaque comercial mínimo para Huevos (Cartones en RD)
        # PRE-PASO: Convertir cualquier peso/volumen de huevos a unidades
        # (ej: "150ml de claras de huevo" ≈ 5 huevos, "100g de huevo" ≈ 2 huevos)
        # Esto evita que claras generen una entrada duplicada por el bloque de peso.
        if name.lower() in ['huevo', 'huevos']:
            egg_weight_g = 50  # 1 huevo entero ≈ 50g
            egg_white_ml = 30  # 1 clara ≈ 30ml
            extra_eggs_from_weight = 0
            
            for k in list(units.keys()):
                k_lower = k.strip().lower()
                if k_lower == 'g':
                    extra_eggs_from_weight += units.pop(k) / egg_weight_g
                elif k_lower == 'ml':
                    extra_eggs_from_weight += units.pop(k) / egg_white_ml
                elif k_lower == 'kg':
                    extra_eggs_from_weight += (units.pop(k) * 1000) / egg_weight_g
                elif k_lower == 'oz':
                    extra_eggs_from_weight += (units.pop(k) * 28.35) / egg_weight_g
                elif k_lower == 'lb':
                    extra_eggs_from_weight += (units.pop(k) * 453.592) / egg_weight_g
                elif k_lower == 'taza':
                    extra_eggs_from_weight += (units.pop(k) * g_per_taza) / egg_weight_g
                elif k_lower in ['cda', 'cdas', 'cucharada', 'cucharadas']:
                    extra_eggs_from_weight += (units.pop(k) * (g_per_taza / 16.0)) / egg_weight_g
                    
            if extra_eggs_from_weight > 0:
                units['unidad'] = units.get('unidad', 0) + math.ceil(extra_eggs_from_weight)
            
            # Ahora consolidar TODAS las unidades en cartones
            u_qty = 0
            for k in list(units.keys()):
                if k.strip().lower() in ['unidad', 'unidades', 'ud', 'uds', 'ud.', 'uds.', 'u', 'u.', 'pieza', 'piezas']:
                    u_qty += units.pop(k)
                elif hasattr(k, 'lower') and 'ud' in k.lower():
                    # Fallback agresivo para atrapar ' Uds.' o cualquier sufijo
                    u_qty += units.pop(k)
            if u_qty > 0:
                if u_qty <= 6:
                    units['cartón (6 uds.)'] = units.get('cartón (6 uds.)', 0) + 1
                elif u_qty <= 15:
                    units['medio cartón (15 uds.)'] = units.get('medio cartón (15 uds.)', 0) + 1
                else:
                    units['cartón (30 uds.)'] = units.get('cartón (30 uds.)', 0) + math.ceil(u_qty / 30.0)

        for u in list(units.keys()):
            q = units[u]
            u_lower = u.lower()
            mapped_to_g = False
            
            # 1. Volúmenes
            if u_lower == 'taza':
                units['g'] = units.get('g', 0) + q * g_per_taza
                mapped_to_g = True
            elif u_lower in ['cda', 'cdas', 'cucharada', 'cucharadas']:
                units['g'] = units.get('g', 0) + q * (g_per_taza / 16.0)
                mapped_to_g = True
            elif u_lower in ['cdta', 'cdtas', 'cdita', 'cucharadita']:
                units['g'] = units.get('g', 0) + q * (g_per_taza / 48.0)
                mapped_to_g = True
                
            # 2. Unidades Físicas
            elif u_lower in ['unidad', 'unidades', 'ud', 'uds']:
                if g_per_u > 0:
                    units['g'] = units.get('g', 0) + q * g_per_u
                    mapped_to_g = True
            elif u_lower in ['rebanada', 'rebanadas', 'lonja', 'lonjas']:
                r_weight = 25 if 'pan' in name.lower() else (g_per_u if g_per_u > 0 else 25)
                units['g'] = units.get('g', 0) + q * r_weight
                mapped_to_g = True
                
            # 3. Contenedores Estándar (si sabemos cuánto pesa el contenedor)
            else:
                if container_weight_g > 0 and (u_lower == db_container or u_lower in ['paquete', 'paquetes', 'pote', 'potes', 'lata', 'latas', 'cartón', 'carton', 'cartones', 'envase', 'envases', 'botella', 'botellas', 'funda', 'fundas', 'fundita', 'funditas']):
                    units['g'] = units.get('g', 0) + q * container_weight_g
                    mapped_to_g = True
            
            # Borrar la unidad original si logramos migrarla a gramos
            if mapped_to_g:
                del units[u]

    results = []
    categorized_results = defaultdict(list)
    total_estimated_cost = 0.0
    
    PANTRY_STAPLES = {
        'Sal y ajo en polvo', 'Aceite de oliva', 'Aceite de coco', 
        'Aceite de sésamo o maní', 'Salsa de soya', 'Orégano', 
        'Canela', 'Pimienta', 'Sal', 'Vinagre', 'Ajo en polvo'
    }
    IGNORE_SHOPPING = {'agua', 'hielo', 'agua potable', 'cubos de hielo'}
    
    for name, units in aggregated.items():
        master_item = master_map.get(name) or master_map.get(name.lower()) or master_map.get(name.title()) or {}
        
        # Evitar líquidos comunes/ilimitados en casa
        from constants import strip_accents
        if strip_accents(name.lower()) in IGNORE_SHOPPING:
            continue
            
        weight_in_lbs = 0.0
        has_weight = False
        cat = master_item.get("category") or "Otros"
        display_cat = _get_display_category(cat, name)
        
        price_per_lb = float(master_item.get("price_per_lb", 0) or 0)
        price_per_unit = float(master_item.get("price_per_unit", 0) or 0)
        
        if 'g' in units:
            weight_in_lbs += units['g'] / 453.592
            has_weight = True
            del units['g']
        if 'kg' in units:
            weight_in_lbs += units['kg'] * 2.20462
            has_weight = True
            del units['kg']
        if 'oz' in units:
            weight_in_lbs += units['oz'] / 16.0
            has_weight = True
            del units['oz']
        if 'lb' in units:
            weight_in_lbs += units['lb']
            has_weight = True
            del units['lb']
        # Líquidos: ml ≈ gramos (densidad ≈ 1 para leche, jugos, aceites)
        # Esto permite que 450ml de leche → peso → Bloque 1 → "1 Cartón"
        if 'ml' in units:
            weight_in_lbs += units['ml'] / 453.592  # 1ml ≈ 1g
            has_weight = True
            del units['ml']
        if 'l' in units:
            weight_in_lbs += (units['l'] * 1000) / 453.592
            has_weight = True
            del units['l']
            
        added = False
        
        # DEDUP: Si el ingrediente tiene cantidades reales (peso ó unidades concretas),
        # eliminar las entradas nominales (pizca, al gusto) porque son redundantes.
        nominal_units = {'pizca', 'al gusto', 'cantidad necesaria', 'chin', 'toque', 'chorrito'}
        has_real_qty = has_weight or any(
            u not in nominal_units and q > 0.0001 
            for u, q in list(units.items())
        )
        if has_real_qty:
            # Tiene cantidades reales → borrar las nominales redundantes
            for nom_u in list(units.keys()):
                if nom_u in nominal_units:
                    del units[nom_u]
        
        # Si SOLO quedan nominales (pizca, al gusto) y no hay peso → saltar ingrediente
        # No aporta a una lista de compras real
        remaining_real = any(u not in nominal_units for u in units) or has_weight
        if not remaining_real:
            continue
            
        if has_weight:
            if weight_in_lbs > 0.0001:
                _n_lower = name.lower()
                if any(kw in _n_lower for kw in ['pechuga', 'pavo', 'yogurt', 'lechosa', 'aguacate', 'arroz']):
                    logging.info(f"  🔬 [RAW LBS] {name}: {weight_in_lbs:.4f} lbs (mult={multiplier})")
                item_cost = weight_in_lbs * price_per_lb
                total_estimated_cost += item_cost
                market_obj = apply_smart_market_units(name, weight_in_lbs, 'lb', 0.0, master_item)
                market_obj["category"] = cat
                market_obj["display_category"] = display_cat
                market_obj["is_staple"] = False
                market_obj["estimated_cost_rd"] = round(item_cost, 2) if item_cost > 0 else None
                item_val = market_obj if structured else market_obj["display_string"]
                results.append(item_val)
                categorized_results[display_cat].append(item_val)
                added = True
                
        for u, q in list(units.items()):
            # Saltar entradas nominales
            if u in nominal_units:
                continue
            if q > 0.0001:
                # DEDUP: Si este ingrediente ya fue agregado por peso (has_weight path),
                # y esta unidad residual es 'unidad/uds' que no se pudo convertir a gramos,
                # NO agregarlo de nuevo — ya está representado en la entrada de peso.
                if added and u.lower() in ['unidad', 'unidades', 'ud', 'uds', 'ud.', 'uds.']:
                    logging.info(f"🔀 [DEDUP] Saltando entrada duplicada por unidad para '{name}' (ya tiene entrada por peso)")
                    continue
                item_cost = 0.0
                if u in ['unidad', 'unidades', 'lata', 'latas', 'paquete', 'paquetes']:
                    item_cost = q * price_per_unit
                    total_estimated_cost += item_cost
                market_obj = apply_smart_market_units(name, 0.0, u, q, master_item)
                market_obj["category"] = cat
                market_obj["display_category"] = display_cat
                market_obj["is_staple"] = False
                market_obj["estimated_cost_rd"] = round(item_cost, 2) if item_cost > 0 else None
                item_val = market_obj if structured else market_obj["display_string"]
                results.append(item_val)
                categorized_results[display_cat].append(item_val)
                added = True
                
        # Removido el PANTRY_STAPLES force-add ("Disponible"). 
        # Si un alimento (incluyendo los estables) se deduce al 100%, 
        # no debe irrumpir en la lista de compras del supermercado.

    results.sort(key=lambda x: x["display_string"] if structured else x)
    
    result_names = [r["name"] if structured and isinstance(r, dict) else str(r) for r in results]
    logging.info(f"🛒 [AGGREGATE FINAL] {len(results)} output items: {result_names[:20]}...")
    
    if categorize:
        for k in categorized_results:
            categorized_results[k].sort(key=lambda x: x["display_string"] if structured else x)
        return dict(categorized_results)
        
    return results

def aggregate_shopping_list(ingredients_list: list[str]) -> list[str]:
    return aggregate_and_deduct_shopping_list(ingredients_list, [])

def get_aggregated_shopping_list_for_plan(plan_result: dict) -> list[str]:
    return get_realtime_pantry(plan_result, [])

def get_shopping_list_delta(user_id: str, plan_result: dict, is_new_plan: bool = False, categorize: bool = False, structured: bool = False, multiplier: float = 1.0):
    """Calcula el verdadero Delta: Ingredientes Totales del Plan - Inventario Físico Actual - (Opcional) Consumidos."""
    all_ingredients = []
    days = plan_result.get("days", [])
    if not days and plan_result.get("meals"):
        days = [{"day": 1, "meals": plan_result.get("meals")}] 
    if not days and plan_result.get("perfectDay"):
        days = [{"day": 1, "meals": plan_result.get("perfectDay")}]

    # Si hay 3 días generados, representan un ciclo rotativo. Promediamos por día y proyectamos a 7 días.
    num_days = max(1, len(days))
    base_duration_scale = 7.0 / num_days
    
    effective_multiplier = multiplier * base_duration_scale
    
    logging.info(f"🔄 [SHOPPING MATH] days_len={num_days} base_scale={base_duration_scale} raw_mult={multiplier} eff_mult={effective_multiplier}")


    meal_count = 0
    for day in days:
        for meal in day.get("meals", []):
            if "suplemento" in meal.get("meal", "").lower():
                continue
            meal_count += 1
            ingredients = meal.get("ingredients", [])
            if not ingredients:
                # Fallback: check if ingredients are inside a 'recipe' dict
                recipe = meal.get("recipe")
                if isinstance(recipe, dict):
                    ingredients = recipe.get("ingredients", [])
            for i in ingredients:
                if isinstance(i, str):
                    all_ingredients.append(i)
                elif isinstance(i, dict):
                    q = i.get("quantity", 0)
                    u = i.get("unit", "unidad")
                    n = i.get("name") or i.get("item_name") or i.get("display_name") or "Desconocido"
                    if q > 0 or u in ['pizca', 'al gusto', 'cantidad necesaria', 'chin', 'toque', 'chorrito']:
                        all_ingredients.append(f"{q} {u} de {n}")
                    else:
                        all_ingredients.append(n)
                    
    logging.info(f"🛒 [SHOPPING EXTRACT] {len(days)} days, {meal_count} meals, {len(all_ingredients)} raw ingredients")
    physical_inventory = []
    consumed_ingredients = []
    
    # JIT Rolling Windows: La lista de compras ahora SIEMPRE se calcula como un verdadero delta
    # contra el inventario físico. En el modelo JIT, el usuario siempre tiene ingredientes remanentes
    # de la ventana anterior, por lo que un plan nuevo debe descontar la despensa para evitar desperdicio.
    if user_id and user_id != "guest":
        try:
            from db_inventory import get_raw_user_inventory
            from datetime import datetime
            raw_inventory = get_raw_user_inventory(user_id)
            if raw_inventory:
                master_list = get_master_ingredients()
                master_map = {m["name"]: m for m in master_list}
                PANTRY_STAPLES = {
                    'Sal y ajo en polvo', 'Aceite de oliva', 'Aceite de coco', 
                    'Aceite de sésamo o maní', 'Salsa de soya', 'Orégano', 
                    'Canela', 'Pimienta', 'Sal', 'Vinagre', 'Ajo en polvo'
                }
                
                for item in raw_inventory:
                    qty = float(item.get("quantity", 0))
                    if qty <= 0: continue
                    name = item.get("ingredient_name", "")
                    
                    is_expired = False
                    if name not in PANTRY_STAPLES:
                        created_at_str = item.get("created_at")
                        if created_at_str:
                            try:
                                item_date = datetime.strptime(created_at_str[:10], "%Y-%m-%d").date()
                                days_old = (datetime.now().date() - item_date).days
                                shelf_life = master_map.get(name, {}).get("shelf_life_days", 14)
                                if (shelf_life - days_old) < 0:
                                    is_expired = True
                            except Exception:
                                pass
                                
                    if not is_expired:
                        physical_inventory.append(item)
                
            # Solución 2: Excluir las comidas ya consumidas durante el período del plan (solo para planes en curso)
            if not is_new_plan:
                from db_plans import get_latest_meal_plan_with_id
                from db_facts import get_consumed_meals_since
                
                plan_record = get_latest_meal_plan_with_id(user_id)
                if plan_record and plan_record.get("plan_data"):
                    plan_created_at = plan_record.get("created_at")
                    if plan_created_at:
                        consumed_meals = get_consumed_meals_since(user_id, plan_created_at)
                        for cm in consumed_meals:
                            ings = cm.get("ingredients") or []
                            if isinstance(ings, list):
                                consumed_ingredients.extend(ings)
                            
        except Exception as e:
            logging.error(f"Error extrayendo inventario/consumidos en delta: {e}")
            
    items_to_deduct = []
    if physical_inventory:
        items_to_deduct.extend([f"{item.get('quantity', 0)} {item.get('unit', 'unidad')} de {item.get('ingredient_name')}" for item in physical_inventory])
    if consumed_ingredients:
        items_to_deduct.extend(consumed_ingredients)
        
    return aggregate_and_deduct_shopping_list(all_ingredients, items_to_deduct, categorize=categorize, structured=structured, multiplier=effective_multiplier)

def get_realtime_pantry(plan_result: dict, consumed_ingredients: list[str]) -> list[str]:
    all_ingredients = []
    days = plan_result.get("days", [])
    if not days and plan_result.get("meals"):
        days = [{"day": 1, "meals": plan_result.get("meals")}] 
    if not days and plan_result.get("perfectDay"):
        days = [{"day": 1, "meals": plan_result.get("perfectDay")}]


    for day in days:
        for meal in day.get("meals", []):
            if "suplemento" in meal.get("meal", "").lower():
                continue
            ingredients = meal.get("ingredients", [])
            for i in ingredients:
                if isinstance(i, str):
                    all_ingredients.append(i)
                elif isinstance(i, dict):
                    q = i.get("quantity", 0)
                    u = i.get("unit", "unidad")
                    n = i.get("name") or i.get("item_name") or i.get("display_name") or "Desconocido"
                    if q > 0 or u in ['pizca', 'al gusto', 'cantidad necesaria', 'chin', 'toque', 'chorrito']:
                        all_ingredients.append(f"{q} {u} de {n}")
                    else:
                        all_ingredients.append(n)
                    
    return aggregate_and_deduct_shopping_list(all_ingredients, consumed_ingredients)
