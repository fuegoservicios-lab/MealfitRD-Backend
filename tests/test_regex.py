import re

def normalize_name(orig_name: str) -> str:
    n = str(orig_name).lower().strip()
    n = re.sub(r'\(.*?\)', '', n).strip()
    # Limpieza de prefijos contenedores o medidas informales
    n = re.sub(r'^(cda|cdta|cdita|cucharada|cucharadita|taza|vaso|pizca|chorrito|puñado|atado|manojo|scoop|lonja|loncha|paquete|paquetico|funda|lata|sobre|sobrecito|chin|toque)(s)?\s*(de\s+|del\s+)?', '', n, flags=re.IGNORECASE)
    n = re.sub(r'^(pechuga|filete|muslo|trozo|chuleta|pieza|corte|ración|racion|porción|porcion|filetico|medallón|medallones|carne)(s)?\s+(de\s+|del\s+)', '', n, flags=re.IGNORECASE)
    n = re.sub(r'^(de\s+|del\s+)', '', n, flags=re.IGNORECASE)
    
    stops = ['cortado', 'cortada', 'cortados', 'cortadas', 'picado', 'picada', 'picados', 'picadas', 'pelado', 'pelada', 'pelados', 'peladas', 'hervido', 'hervida', 'hervidos', 'hervidas', 'cocido', 'cocida', 'cocidos', 'cocidas', 'asado', 'asada', 'asados', 'asadas', 'crudo', 'cruda', 'crudos', 'crudas', 'horneado', 'horneada', 'horneados', 'horneadas', 'desmenuzado', 'desmenuzada', 'desmenuzados', 'desmenuzadas', 'rallado', 'rallada', 'rallados', 'ralladas', 'guisado', 'guisada', 'guisados', 'guisadas', 'frito', 'frita', 'fritos', 'fritas', 'hecha puré', 'hecho puré', 'puré', 'en julianas', 'en tiras', 'en cubos', 'en hojuelas', 'en dados', 'en aros', 'en trozos', 'en rodajas', 'en porciones', 'finamente', 'muy', 'pequeño', 'pequeña', 'pequeños', 'pequeñas', 'grande', 'grandes', 'mediano', 'mediana', 'medianos', 'medianas', 'maduro', 'madura', 'maduros', 'maduras', 'fresco', 'fresca', 'frescos', 'frescas', 'firme', 'firmes', 'entero', 'entera', 'enteros', 'enteras', 'fina', 'finas', 'gruesa', 'gruesas', 'magro', 'magra', 'magros', 'magras', 'natural', 'naturales', 'bajo en grasa', 'bajas en grasa', 'bajos en grasa', 'bajo en sodio', 'bajas en sodio', 'bajos en sodio', 'descremado', 'descremada', 'descremados', 'descremadas', 'sin sal', 'con sal', 'sin piel', 'sin hueso', 'para rebozar', 'al gusto', 'pizca de', 'rodajas de', 'de la despensa', 'ralladura y jugo de 1/2', 'la', 'el', 'los', 'las']
    clean_n = n
    for s in stops:
        clean_n = re.sub(r'\b' + s + r'\b', '', clean_n, flags=re.IGNORECASE)
        
    print("After stops:", repr(clean_n))
        
    # Limpiar conjunciones o preposiciones que quedan colgadas al quitar los stops
    clean_n = re.sub(r'\b(y|en|con|de|del|para)\b', '', clean_n, flags=re.IGNORECASE)
    clean_n = re.sub(r'\s+', ' ', clean_n).replace(',', '').strip()
    
    print("After regex:", repr(clean_n))
    return clean_n

print(normalize_name("Papas y"))
print(normalize_name("Papas peladas y picadas"))
