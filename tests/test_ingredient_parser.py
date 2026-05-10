import re

def _parse_ingredient_fallback(raw: str):
    raw_lower = raw.lower().strip()
    
    fractions = {'½': '0.5', '⅓': '0.33', '⅔': '0.67', '¼': '0.25', '¾': '0.75'}
    for f_char, f_val in fractions.items():
        raw_lower = raw_lower.replace(f_char, f_val)
        
    word_to_num = {'un ': '1 ', 'una ': '1 ', 'unos ': '1 ', 'unas ': '1 ', 'medio ': '0.5 ', 'media ': '0.5 ', 'dos ': '2 ', 'tres ': '3 '}
    for w, n in word_to_num.items():
        if raw_lower.startswith(w):
            raw_lower = raw_lower.replace(w, n, 1)

    qty = 1.0
    # Modificamos para aceptar espacio opcional antes de letras
    qty_match = re.match(r'^([\d\.,/]+(?:\s*-\s*[\d\.,/]+)?)(?:\s+|$|(?=[a-zñA-ZÑ]))', raw_lower)
    if qty_match:
        qty_str = qty_match.group(1).strip()
        raw_lower = raw_lower[len(qty_match.group(0)):].strip()
        if '-' in qty_str:
            qty_str = qty_str.split('-')[-1].strip()
        qty_str = qty_str.replace(',', '.')
        try:
            if '/' in qty_str:
                num, den = qty_str.split('/')
                qty = float(num) / float(den)
            else:
                qty = float(qty_str)
        except ValueError:
            pass

    units_regex = r'^(g|ml|lb|lbs|tazas?|cda?s?|cucharadas?|cditas?|cucharaditas?|oz|onzas?|dientes?(?:\s+de)?|puñado|pizca|rebanadas?|filetes?|porción|porcion|unidades?|piezas?)\b'
    unit_match = re.match(units_regex, raw_lower)
    unit = ""
    if unit_match:
        unit = unit_match.group(1).strip()
        raw_lower = raw_lower[len(unit_match.group(0)):].strip()
        if unit.endswith(" de"):
            unit = unit[:-3]
            
    if raw_lower.startswith("de "):
        raw_lower = raw_lower[3:].strip()
        
    name = raw_lower.strip()
    return qty, unit, name

test_cases = [
    "½ taza de arroz",
    "un puñado de nueces",
    "2-3 dientes de ajo",
    "150g pollo",
    "150 g de pollo",
    "1/4 lb carne molida",
    "1 cucharada aceite de oliva"
]

for t in test_cases:
    print(f"'{t}' ->", _parse_ingredient_fallback(t))
