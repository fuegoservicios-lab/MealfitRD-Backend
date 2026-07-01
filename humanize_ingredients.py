import re
import math

try:
    from knobs import _env_bool
except Exception:  # fail-safe: si knobs no resuelve, el fix queda ON por default (es backstop seguro)
    def _env_bool(_name, _default=False):
        import os
        return (os.environ.get(_name, "1" if _default else "0") or "").strip().lower() in ("1", "true", "yes", "on")

# [P1-INGREDIENT-DOUBLE-FRACTION · 2026-06-29] Backstop de DISPLAY: neutraliza cantidades malformadas con DOBLE fracción
# ("0.5 jugo de 0.5 limón" → "jugo de 0.5 limón"). Si tras la cantidad líder el resto del nombre YA contiene su propia
# cantidad+sustantivo contable ("de 0.5 limón"), la líder es un prepend espurio. Default True; flip a False revierte.
INGREDIENT_DOUBLE_FRACTION_FIX = _env_bool("MEALFIT_INGREDIENT_DOUBLE_FRACTION_FIX", True)
# cantidad interna "de <num> <sustantivo>": fracción legítima del nombre (jugo de ½ limón).
_INNER_QTY_NOUN_RE = re.compile(r'\bde\s+(?:\d+(?:[.,]\d+)?|½|¼|¾)\s+\w+', re.IGNORECASE)
# cantidad líder (acepta decimal o fracción unicode) seguida de un resto que contiene "de <num> ...".
_LEAD_PLUS_INNER_RE = re.compile(r'^\s*(?:\d+(?:[.,]\d+)?|½|¼|¾)\s+(.*\bde\s+(?:\d|½|¼|¾).+)$', re.IGNORECASE)


def _collapse_double_fraction(ing) -> str:
    """[P1-INGREDIENT-DOUBLE-FRACTION] "0.5 jugo de 0.5 limón" → "jugo de 0.5 limón" (la cantidad líder es un prepend
    espurio sobre un nombre que ya trae su fracción). Idempotente (re-pasar "jugo de 0.5 limón" no matchea → intacto) y
    fail-safe (no-string / excepción → devuelve el original). NO toca strings normales ("150g de arroz", "2 huevos")."""
    try:
        s = str(ing).strip()
        m = _LEAD_PLUS_INNER_RE.match(s)
        if m and _INNER_QTY_NOUN_RE.search(m.group(1)):
            return m.group(1)  # conserva el nombre con su fracción legítima; quita la líder espuria
        return ing
    except Exception:
        return ing

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

def _polish_countunit_display(raw_ingredient: str, qty_str: str, name: str) -> str:
    """[P1-RECIPE-STEP-INGREDIENT-COHERENCE · 2026-06-28] Pulido cosmético del ingrediente sin unidad métrica (el quantize
    no lo toca porque "cda"/count no son unidades métricas). Display-only (corre dentro de humanize_plan_ingredients, POST
    macros/shopping → no altera datos). Dos reglas deterministas:
      (a) "0.25 cda"/"¼ cda"/"1/4 cda" de X → "1 cdta de X" (¼ cucharada es impráctica).
      (b) "1 <plural>" → "1 <singular>" usando DOMINICAN_HOUSEHOLD_MEASURES (match EXACTO del plural → evita romper
          gramática con adjetivos no mapeados). Ej: "1 huevos enteros"→"1 huevo entero", "1 huevos"→"1 huevo".
    Cualquier otro caso: devuelve el original intacto."""
    try:
        qty = 1.0
        _qs = (qty_str or "").strip().replace(',', '.')
        if '/' in _qs:
            _p = _qs.split('/')
            qty = float(_p[0]) / float(_p[1])
        elif _qs:
            qty = float(_qs)
    except (ValueError, ZeroDivisionError, IndexError):
        return raw_ingredient

    name_l = strip_accents((name or "").lower().strip())

    # (a) ¼ cda → 1 cdta
    m_spoon = re.match(r'^(cda|cucharada)s?\b\s*(?:de\s+)?(.*)$', name_l)
    if m_spoon and 0 < qty <= 0.34:
        rest = re.sub(r'^\s*(?:cda|cucharada)s?\b\s*(?:de\s+)?', '', name, flags=re.IGNORECASE).strip()
        return f"1 cdta de {rest}" if rest else raw_ingredient

    # (b) "1 <plural>" → "1 <singular>" (match exacto del plural)
    if abs(qty - 1.0) < 1e-6 and name_l:
        for meas in sorted(DOMINICAN_HOUSEHOLD_MEASURES.values(),
                           key=lambda mm: len(mm.get("plural", "")), reverse=True):
            pl = strip_accents((meas.get("plural") or "").lower())
            if pl and name_l == pl:
                return f"1 {meas['singular']}"

    return raw_ingredient


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
        # [P1-RECIPE-STEP-INGREDIENT-COHERENCE · 2026-06-28] Caso SIN unidad métrica (count/cucharada): el quantize lo
        # deja como vino ("0.25 cda de aceite", "1 huevos enteros"). Pulido cosmético display-only (corre POST-macros →
        # no toca compras/macros): (a) "0.25 cda"→"1 cdta"; (b) "1 <plural>"→"1 <singular>". Resto intacto.
        return _polish_countunit_display(raw_ingredient, qty_str, name)
        
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
                    # [P1-INGREDIENT-DOUBLE-FRACTION] backstop de display: colapsa "0.5 jugo de 0.5 limón" antes de humanizar.
                    ing_clean = _collapse_double_fraction(ing) if INGREDIENT_DOUBLE_FRACTION_FIX else ing
                    humanized = humanize_ingredient(ing_clean)
                    humanized_ingredients.append(humanized)
                meal["ingredients"] = humanized_ingredients
                # [P2-STEP-HOUSEHOLD-SYNC · 2026-07-01] armoniza las menciones de los PASOS con la
                # medida casera recién aplicada a la lista (ver docstring del helper).
                try:
                    sync_recipe_steps_to_household(meal)
                except Exception:
                    pass
    return plan_result


# Captura del alimento ACOTADA a ≤3 palabras con lookahead de conectores (y/con/para/hasta/en/o) →
# "150 g de arroz y lávalo" captura solo "arroz" y el resto del paso queda intacto tras el replace.
_STEP_GRAMS_MENTION_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos)\s+de\s+"
    r"([a-záéíóúñü][\wáéíóúñü]*(?:\s+(?!y\b|con\b|para\b|hasta\b|en\b|o\b)[\wáéíóúñü]+){0,2})",
    re.IGNORECASE)


def sync_recipe_steps_to_household(meal: dict) -> int:
    """[P2-STEP-HOUSEHOLD-SYNC · 2026-07-01] (audit v2 recetas GAP-5, batch P2-AUDIT-V2-BATCH)
    Armoniza las UNIDADES entre lista y pasos: `humanize_ingredient` convierte la LISTA a medida
    casera ("¾ taza de arroz") pero los PASOS quedaban en gramos post-qty-sync ("pesa 80 g de
    arroz") → el usuario leía dos unidades distintas para el mismo alimento. Reescribe la mención
    del paso a la medida casera CON los gramos entre paréntesis — "¾ taza de arroz (80 g)" — ambas
    unidades visibles (cocinable con taza O balanza).

    Diseño conservador (espejo anti-falso-positivo de _sync_recipe_step_quantities):
      - Solo menciones `<qty> g de <alimento>` cuya qty COINCIDE con la cantidad líder RAW del
        ingrediente (no toca cantidades parciales tipo "espolvorea 10 g de arroz").
      - El alimento se matchea por token principal (primera palabra ≥4 chars); tokens compartidos
        por dos ingredientes se descartan (ambiguo).
      - Solo ingredientes cuya forma humanizada DIFIERE del raw (si humanize no convirtió, no hay
        nada que armonizar). Notas ⚠/💡/⚕ intactas. Idempotente (la mención reescrita ya no
        matchea el patrón `g de`). Fail-safe → 0. Muta `meal`. tooltip-anchor: P2-STEP-HOUSEHOLD-SYNC"""
    try:
        ings = meal.get("ingredients")
        raws = meal.get("ingredients_raw")
        recipe = meal.get("recipe")
        if not (isinstance(ings, list) and isinstance(raws, list) and isinstance(recipe, list)
                and len(ings) == len(raws) and recipe):
            return 0
        token_map = {}
        ambiguous = set()
        for h, r in zip(ings, raws):
            h_s, r_s = str(h).strip(), str(r).strip()
            if not h_s or h_s == r_s:
                continue
            m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos)\s+(?:de\s+)?(.+)$", r_s, re.IGNORECASE)
            if not m:
                continue
            raw_qty = m.group(1).replace(",", ".")
            food = strip_accents(m.group(2).strip().lower())
            toks = [t for t in re.split(r"[^\wáéíóúñü]+", food) if len(t) >= 4]
            if not toks:
                continue
            tok = toks[0]
            if tok in token_map:
                ambiguous.add(tok)
                continue
            token_map[tok] = (raw_qty, h_s)
        for tok in ambiguous:
            token_map.pop(tok, None)
        if not token_map:
            return 0
        fixed = 0
        new_steps = []
        for step in recipe:
            s = str(step)
            if not isinstance(step, str) or ("⚠" in s) or ("💡" in s) or ("⚕" in s):
                new_steps.append(step)
                continue

            def _sub(mm):
                nonlocal fixed
                food_m = strip_accents(mm.group(2).strip().lower())
                toks_m = [t for t in re.split(r"[^\wáéíóúñü]+", food_m) if len(t) >= 4]
                for ft in toks_m[:2]:
                    entry = token_map.get(ft)
                    if entry:
                        raw_qty, human = entry
                        if mm.group(1).replace(",", ".") == raw_qty:
                            fixed += 1
                            return f"{human} ({mm.group(1)} g)"
                        break
                return mm.group(0)

            new_steps.append(_STEP_GRAMS_MENTION_RE.sub(_sub, step))
        if fixed:
            meal["recipe"] = new_steps
        return fixed
    except Exception:
        return 0
