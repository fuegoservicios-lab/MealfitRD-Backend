import re
import unicodedata
import difflib
from constants import strip_accents, _QUANTITY_PATTERN

_STOPWORDS_RE = re.compile(r'\b(con|de|y|al|a|la|el|en|las|los|del|para|por|tipo|estilo)\b')
_WHITESPACE_RE = re.compile(r'\s+')


def _contar_grupo_regex(items, synonyms_map, history_normalized):
    """Compila UNA sola regex de alternancia por grupo en vez de O(items × syns) scans."""
    out = {it: 0 for it in items}
    if not items or not history_normalized:
        return out
    syn_to_items = {}
    for it in items:
        for syn in synonyms_map.get(it.lower(), [it.lower()]):
            norm = strip_accents(syn.lower()).strip()
            if norm:
                syn_to_items.setdefault(norm, []).append(it)
    if not syn_to_items:
        return out
    # Ordenar por longitud descendente para evitar que un prefijo robe un compuesto.
    alts = sorted(syn_to_items.keys(), key=len, reverse=True)
    pat = re.compile(r'\b(?:' + '|'.join(re.escape(a) for a in alts) + r')\b')
    for m in pat.finditer(history_normalized):
        tok = m.group(0)
        for it in syn_to_items.get(tok, ()):
            out[it] += 1
    return out


def _calcular_frecuencias_regex_cpu_bound(
    history_normalized, 
    filtered_proteins, protein_synonyms, 
    filtered_carbs, carb_synonyms, 
    filtered_veggies, veggie_fat_synonyms, 
    filtered_fruits, fruit_synonyms
):
    """
    Tarea CPU-Bound: cuenta frecuencias de sinónimos sobre el historial en 1 pasada por grupo.
    """
    protein_freq = _contar_grupo_regex(filtered_proteins, protein_synonyms, history_normalized)
    carb_freq = _contar_grupo_regex(filtered_carbs, carb_synonyms, history_normalized)
    veggie_freq = _contar_grupo_regex(filtered_veggies, veggie_fat_synonyms, history_normalized)
    fruit_freq = _contar_grupo_regex(filtered_fruits, fruit_synonyms, history_normalized)
    return protein_freq, carb_freq, veggie_freq, fruit_freq


def _normalize_meal_name(s: str) -> str:
    """
    Normaliza NOMBRES DE PLATOS para anti-repetición (Jaccard/SequenceMatcher).
    Preserva técnicas de cocción pero elimina stopwords largas para que Jaccard se enfoque.
    También stripea cantidades/unidades para evitar falsos negativos (ej: "200g Pollo" vs "Pollo").
    """
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = s.lower()
    # Stripear cantidades/unidades del inicio (ej: "200g", "1/2 lb de")
    s = _QUANTITY_PATTERN.sub('', s).strip()
    s = _STOPWORDS_RE.sub('', s)
    return _WHITESPACE_RE.sub(' ', s).strip()

def _validar_repeticiones_cpu_bound(recent_meal_names, days_plan):
    """
    Tarea CPU-Bound: Ejecuta el macheo de Jaccard Token Sets y Difflib SequenceMatcher.
    Si la base de datos de historial de comidas crece a 30 días, este loop es brutal.
    """
    if not recent_meal_names:
        return []

    recent_data = []
    for name in recent_meal_names:
        if name:
            norm = _normalize_meal_name(name)
            recent_data.append({
                "norm": norm,
                "tokens": set(norm.split())
            })
    
    MAIN_MEAL_JACCARD = 0.85
    MAIN_MEAL_SEQ = 0.75
    SNACK_JACCARD = 0.95
    SNACK_SEQ = 0.90
    
    repeated_meals = []
    
    # Manejar fallbacks a la forma del diccionario si plan = { "meals": [...] } en generacion
    if not days_plan and isinstance(days_plan, list) == False:
        pass # Not applicable here directly, days_plan should be a list of days
        
    for day_obj in days_plan:
        for meal in day_obj.get("meals", []):
            meal_type = meal.get("meal", "").lower()
            
            if meal_type in ["desayuno", "almuerzo", "cena"]:
                jaccard_threshold = MAIN_MEAL_JACCARD
                seq_threshold = MAIN_MEAL_SEQ
            elif meal_type in ["merienda", "snack", "merienda am", "merienda pm"]:
                jaccard_threshold = SNACK_JACCARD
                seq_threshold = SNACK_SEQ
            else:
                continue
            
            raw_name = meal.get("name", "")
            new_norm = _normalize_meal_name(raw_name)
            if not new_norm:
                continue
            
            new_tokens = set(new_norm.split())
            is_repeated = False
            
            for recent in recent_data:
                if new_tokens and recent["tokens"]:
                    intersection = new_tokens.intersection(recent["tokens"])
                    # Default overlap guard division by zero
                    if len(new_tokens) > 0 and len(recent["tokens"]) > 0:
                        overlap1 = len(intersection) / len(new_tokens)
                        overlap2 = len(intersection) / len(recent["tokens"])
                        
                        if max(overlap1, overlap2) >= jaccard_threshold:
                            is_repeated = True
                            break
                        
                if abs(len(new_norm) - len(recent["norm"])) <= 5:
                    if difflib.SequenceMatcher(None, new_norm, recent["norm"]).ratio() >= seq_threshold:
                        is_repeated = True
                        break
                        
            if is_repeated:
                repeated_meals.append(raw_name)
                
    return repeated_meals
