import re
import math

# Diccionario de equivalencias para medidas caseras dominicanas
# key: string base simplificado
# value: dict con:
#   - weight: peso en gramos o ml por unidad
#   - singular: etiqueta singular
#   - plural: etiqueta plural
#   - is_liquid: si es líquido (para usar tazas/cdas en vez de onzas)
#   - slices: si se mide en lonjas/rebanadas
DOMINICAN_HOUSEHOLD_MEASURES = {
    # Víveres enteros
    "platano verde": {"weight": 280.0, "singular": "plátano verde", "plural": "plátanos verdes"},
    "platano maduro": {"weight": 280.0, "singular": "plátano maduro", "plural": "plátanos maduros"},
    "guineito verde": {"weight": 100.0, "singular": "guineíto verde", "plural": "guineítos verdes"},
    "guineos verdes": {"weight": 100.0, "singular": "guineíto verde", "plural": "guineítos verdes"},
    "yuca": {"weight": 400.0, "singular": "pedazo mediano de yuca", "plural": "pedazos medianos de yuca"},
    "batata": {"weight": 220.0, "singular": "batata mediana", "plural": "batatas medianas"},
    "papa": {"weight": 150.0, "singular": "papa mediana", "plural": "papas medianas"},
    "papas": {"weight": 150.0, "singular": "papa mediana", "plural": "papas medianas"},
    "name": {"weight": 300.0, "singular": "pedazo de ñame", "plural": "pedazos de ñame"},
    "yautia": {"weight": 250.0, "singular": "pedazo de yautía", "plural": "pedazos de yautía"},
    "casabe": {"weight": 20.0, "singular": "torta pequeña de casabe", "plural": "tortas pequeñas de casabe"},

    # Frutas y Vegetales
    "aguacate": {"weight": 250.0, "singular": "aguacate mediano", "plural": "aguacates medianos"},
    "guineo maduro": {"weight": 120.0, "singular": "guineo maduro", "plural": "guineos maduros"},
    "manzana": {"weight": 150.0, "singular": "manzana", "plural": "manzanas"},
    "naranja": {"weight": 130.0, "singular": "naranja", "plural": "naranjas"},
    "limon": {"weight": 60.0, "singular": "limón", "plural": "limones"},
    "fresa": {"weight": 15.0, "singular": "fresa", "plural": "fresas"},
    "fresas frescas": {"weight": 15.0, "singular": "fresa fresca", "plural": "fresas frescas"},
    "tomate": {"weight": 120.0, "singular": "tomate", "plural": "tomates"},
    "cebolla": {"weight": 110.0, "singular": "cebolla", "plural": "cebollas"},
    "diente de ajo": {"weight": 5.0, "singular": "diente de ajo", "plural": "dientes de ajo"},
    "aji": {"weight": 100.0, "singular": "ají", "plural": "ajíes"},
    "pimiento": {"weight": 100.0, "singular": "pimiento", "plural": "pimientos"},
    "zanahoria": {"weight": 75.0, "singular": "zanahoria", "plural": "zanahorias"},

    # Huevos
    "huevo": {"weight": 50.0, "singular": "huevo", "plural": "huevos"},
    "huevos enteros": {"weight": 50.0, "singular": "huevo entero", "plural": "huevos enteros"},
    "clara de huevo": {"weight": 33.0, "singular": "clara de huevo", "plural": "claras de huevo"},
    "claras de huevo": {"weight": 33.0, "singular": "clara de huevo", "plural": "claras de huevo"},

    # Carnes y Proteínas (porciones o unidades)
    "chuleta": {"weight": 150.0, "singular": "chuleta", "plural": "chuletas"},
    "longaniza": {"weight": 100.0, "singular": "pedazo de longaniza", "plural": "pedazos de longaniza"},
    "pechuga de pollo": {"weight": 200.0, "singular": "pechuga de pollo (porción)", "plural": "pechugas de pollo"},
    "filete de pescado": {"weight": 150.0, "singular": "filete de pescado", "plural": "filetes de pescado"},

    # Panes
    "pan integral": {"weight": 30.0, "singular": "rebanada de pan integral", "plural": "rebanadas de pan integral"},
    "pan": {"weight": 30.0, "singular": "rebanada de pan", "plural": "rebanadas de pan"},
    "tortilla": {"weight": 45.0, "singular": "tortilla", "plural": "tortillas"},

    # Rebanadas/Lonjas (Salami, Quesos, Jamón)
    "salami": {"weight": 40.0, "singular": "rueda de salami", "plural": "ruedas de salami"},
    "salami dominicano": {"weight": 40.0, "singular": "rueda de salami dominicano", "plural": "ruedas de salami dominicano"},
    "queso": {"weight": 25.0, "singular": "lonja/pedazo de queso", "plural": "lonjas/pedazos de queso"},
    "queso de freir": {"weight": 25.0, "singular": "lonja de queso de freír", "plural": "lonjas de queso de freír"},
    "queso blanco": {"weight": 25.0, "singular": "lonja de queso blanco", "plural": "lonjas de queso blanco"},
    "queso blanco fresco": {"weight": 25.0, "singular": "lonja de queso blanco fresco", "plural": "lonjas de queso blanco fresco"},
    "queso ricotta": {"weight": 15.0, "singular": "cda de queso ricotta", "plural": "cdas de queso ricotta"},
    "jamon": {"weight": 20.0, "singular": "lonja de jamón", "plural": "lonjas de jamón"},
    "pechuga de pavo": {"weight": 20.0, "singular": "lonja de pechuga de pavo", "plural": "lonjas de pechuga de pavo"}
}

# Regex pre-compilado para extraer cantidad, unidad y nombre
_QUANTITY_PATTERN = re.compile(
    r'^([\d\s/.,]+)'                           # Números, fracciones, espacios iniciales
    r'(?:'
        r'(g|gr|kg|mg|ml|l|lb|lbs|oz)'         # Unidades métricas/imperiales (strict)
    r')?'
    r'\s*(?:de\s+)?',                        # "de " opcional que conecta cantidad con ingrediente
    re.IGNORECASE
)

def number_to_fraction_str(num: float) -> str:
    """Convierte un número a fracción legible (½, ¼) o string redondeado."""
    if num <= 0:
        return ""
    
    # Redondeo a cuartos más cercanos
    quarter_rounded = round(num * 4) / 4.0
    
    if quarter_rounded == 0:
        return "¼" # mínimo
        
    whole = int(quarter_rounded)
    remainder = quarter_rounded - whole
    
    fractions = {
        0.25: "¼",
        0.5: "½",
        0.75: "¾"
    }
    
    if remainder == 0:
        return str(whole)
    elif whole == 0:
        return fractions.get(remainder, str(round(num, 1)))
    else:
        return f"{whole}{fractions.get(remainder, '')}"

def strip_accents(s: str) -> str:
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def humanize_ingredient(raw_ingredient: str) -> str:
    """
    Convierte un ingrediente en gramos/ml a medidas caseras si aplica.
    Ej: '100 g de plátano verde' -> '½ plátano verde'
    """
    raw_lower = raw_ingredient.lower().strip()
    
    # 1. Parsear cantidad original
    match = _QUANTITY_PATTERN.match(raw_lower)
    if not match:
        return raw_ingredient # No pudimos extraer cantidad/unidad
        
    qty_str = match.group(1).strip()
    unit = match.group(2)
    name = raw_ingredient[match.end():].strip()
    
    if not unit:
        # Ya está en unidades (ej. "2 huevos"), no tocar
        return raw_ingredient
        
    unit = unit.lower()
    
    # 2. Convertir qty_str a float
    qty = 1.0
    qty_str = qty_str.replace(',', '.')
    try:
        if '/' in qty_str:
            parts = qty_str.split('/')
            qty = float(parts[0]) / float(parts[1])
        else:
            qty = float(qty_str)
    except (ValueError, ZeroDivisionError):
        # [P3-HUMANIZE-ZERODIV · 2026-05-30] Capturar también ZeroDivisionError:
        # una hallucination del LLM como "1/0 lb de arroz" lanzaba ZeroDivisionError
        # NO capturada que, vía el call-site sin try/except per-item en
        # humanize_plan_ingredients, abortaba la humanización del plan COMPLETO
        # (las ~21 comidas perdían las medidas caseras dominicanas por un solo
        # ingrediente malformado). Ahora degrada de forma aislada — devuelve el
        # raw de ESE ítem y deja el resto del plan humanizado. Simétrico con el
        # hermano parse_fraction (shopping_calculator.py) que usa `except Exception`.
        return raw_ingredient
        
    # Convertir todo a gramos o ml
    base_qty = qty
    if unit in ['kg']: base_qty = qty * 1000.0
    elif unit in ['lb', 'lbs']: base_qty = qty * 453.592
    elif unit in ['oz']: base_qty = qty * 28.3495
    elif unit in ['l']: base_qty = qty * 1000.0

    # 3. Buscar correspondencia en diccionario de medidas
    name_clean = strip_accents(name.lower().strip())
    
    best_match_key = None
    # Prioridad: match exacto
    if name_clean in DOMINICAN_HOUSEHOLD_MEASURES:
        best_match_key = name_clean
    else:
        # Match por sufijo/prefijo
        for key in sorted(DOMINICAN_HOUSEHOLD_MEASURES.keys(), key=len, reverse=True):
            if key in name_clean:
                best_match_key = key
                break
                
    if best_match_key:
        measure = DOMINICAN_HOUSEHOLD_MEASURES[best_match_key]
        weight_per_unit = measure["weight"]
        
        # Calcular unidades
        units = base_qty / weight_per_unit
        
        # Si las unidades son menores a 0.25 (muy poquito) o mayores a 10 (muchísimo), mejor dejarlo en taza/cda o gramos
        if 0.25 <= units <= 10:
            fraction_str = number_to_fraction_str(units)
            label = measure["singular"] if units <= 1.0 else measure["plural"]
            
            # Reemplazar la base del nombre pero preservar adjetivos
            # Ej: Si name es "plátano verde hervido", y label es "plátano verde",
            # el resultado debe ser "½ plátano verde hervido"
            
            # Simple approach: Return the fraction + label, and optionally append leftover words
            # This avoids complex regex replacement for now.
            return f"{fraction_str} {label}"

    # 4. Fallback a tazas y cucharadas para cosas genéricas si es líquido o granulado
    # Para arroz, avena, líquidos, grasas, especias
    
    # Aceites/Grasas líquidas (1 cda = 15g, 1 cdta = 5g)
    if any(x in name_clean for x in ['aceite', 'mantequilla', 'vinagre', 'salsa de soya', 'miel']):
        if base_qty <= 10:
            return f"{number_to_fraction_str(base_qty / 5.0)} cdta de {name}"
        elif base_qty <= 60:
            return f"{number_to_fraction_str(base_qty / 15.0)} cda de {name}"
            
    # Granos crudos / cocidos (arroz, avena, lentejas, yogurt) (1 taza = ~200-240g)
    if any(x in name_clean for x in ['arroz', 'avena', 'lentejas', 'habichuela', 'garbanzo', 'yogurt', 'pasta', 'quinoa', 'pure']):
        if base_qty >= 50:
            tazas = base_qty / 200.0 # Aproximación genérica
            return f"{number_to_fraction_str(tazas)} taza de {name}"

    # Si no hubo match, devolver el original
    return raw_ingredient

def humanize_plan_ingredients(plan_result: dict) -> dict:
    """
    Recorre el plan completo y humaniza los ingredientes en cada comida.
    Solo afecta la lista `ingredients` de las `meals`, no toca recipe ni macros.

    [P1-4] Preserva la lista pre-humanización en `meal["ingredients_raw"]`
    para que el aggregator de la lista de compras (`get_shopping_list_delta`)
    pueda re-procesar el plan persistido sin perder las unidades métricas.

    ANTES, si un plan ya humanizado ("1 pechuga de pollo (porción)") se
    re-aggregaba después (regenerar con `/shift-plan`, recalcular shopping
    al cambiar householdSize, etc), `_parse_quantity` no encontraba la
    unidad métrica original ("200g pechuga") y caía a `unit='unidad'` →
    el aggregator consolidaba "1 unidad pollo" en lugar del peso real.
    Para listas escaladas ×4 mensual, "4 pechugas" llegaban con cantidad
    derivada del `density_g_per_unit` del master (que puede divergir
    significativamente del peso semántico real del plato), produciendo
    listas de compras descuadradas.

    AHORA `ingredients_raw` se setea SOLO en la primera invocación
    (idempotente: si ya existe, no se sobrescribe — protege contra
    re-humanización que perdería el original).
    """
    for day in plan_result.get("days", []):
        for meal in day.get("meals", []):
            if "ingredients" in meal:
                # [P1-4] Preservar el original SOLO si no existe.
                # Idempotente ante re-llamadas.
                if "ingredients_raw" not in meal:
                    meal["ingredients_raw"] = list(meal["ingredients"])
                humanized_ingredients = []
                for ing in meal["ingredients"]:
                    humanized = humanize_ingredient(ing)
                    humanized_ingredients.append(humanized)
                meal["ingredients"] = humanized_ingredients
    return plan_result
