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

    # (c) [P1-RECIPE-POLISH-5 · 2026-07-12] "1 <unidad-plural> de X" → singular la PALABRA-UNIDAD
    # ("1 tazas de yogurt griego", "1 dientes de ajo" — emitidos así por el day-gen, vivos en el
    # plan 1bfda745). Solo cantidad EXACTA 1 y solo la primera palabra si es unidad conocida;
    # el resto de la línea queda intacto (no tocamos adjetivos ni el alimento).
    if abs(qty - 1.0) < 1e-6 and name_l:
        m_unit = re.match(
            r'^(tazas|dientes|rebanadas|rodajas|tajadas|lonjas|hojas|unidades|pedazos|latas|potes|paquetes|fundas|mazos|cucharadas)\b',
            name_l)
        if m_unit:
            _uw = m_unit.group(1)
            return f"1 {_uw[:-1]}{name[len(_uw):]}"

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
                # [P2-DISPLAY-FRACTIONS · 2026-07-01] (batch P1-DISH-REALISM-BATCH) pulido final de
                # display: "0.5 papa"→"½ papa", "1.75 cdta"→"1¾ cdta", "1 cdas"→"1 cda", "1 tallos"→
                # "1 tallo". Display-only (ingredients_raw intacto para compras).
                meal["ingredients"] = [_prettify_quantity_display(h) for h in humanized_ingredients]
                # [P2-DISPLAY-NAME-SPECIFICITY · 2026-07-05] el display puede perder calificadores
                # del alimento ("145 g de queso" vs raw "145g de queso cottage cocido", plan
                # 7e4e5570 — compras BIEN, usuario veía el genérico). Restaura el nombre específico
                # desde raw por índice (display-only).
                meal["ingredients"] = _restore_gram_name_specificity(
                    meal["ingredients"], meal.get("ingredients_raw"))
                # [P2-STEP-HOUSEHOLD-SYNC · 2026-07-01] armoniza las menciones de los PASOS con la
                # medida casera recién aplicada a la lista (ver docstring del helper).
                try:
                    sync_recipe_steps_to_household(meal)
                except Exception:
                    pass
                # [P3-STEP-FRACTIONS · 2026-07-05] "2.5 cdas"/"0.5 taza"/"1/4 de la masa" en pasos
                # → fracciones unicode (paridad visual con la lista). Display-only, fail-safe.
                try:
                    _rec_pf = meal.get("recipe")
                    if isinstance(_rec_pf, list):
                        meal["recipe"] = [prettify_step_fractions(_s) if isinstance(_s, str) else _s
                                          for _s in _rec_pf]
                except Exception:
                    pass
    return plan_result


# [P2-DISPLAY-FRACTIONS · 2026-07-01] unidades/sustantivos que se singularizan tras "1 " (curado,
# anti-falso-positivo: nada de despluralizar palabras arbitrarias).
_DISPLAY_SINGULAR = {
    "cdas": "cda", "cdtas": "cdta", "tazas": "taza", "tallos": "tallo", "rebanadas": "rebanada",
    "hojas": "hoja", "dientes": "diente", "latas": "lata", "paquetes": "paquete", "unidades": "unidad",
    "lonjas": "lonja", "cucharadas": "cucharada", "cucharaditas": "cucharadita",
}
# [P2-PLURAL-CONCORDANCE · 2026-07-05] (screenshots plan 3aa6e58a: "2 huevo", "3 hoja de laurel",
# "3 cebolla grande") El singularizador solo cubría "1 <plural>" → "1 <singular>"; el camino
# inverso (N>1 con sustantivo singular) quedaba agramatical. Mapa curado: unidades (inverso del
# de arriba) + alimentos contables frecuentes. Solo aplica a conteos ENTEROS >1.
_DISPLAY_PLURAL = {v: k for k, v in _DISPLAY_SINGULAR.items()}
_DISPLAY_PLURAL.update({
    "huevo": "huevos", "cebolla": "cebollas", "tomate": "tomates", "zanahoria": "zanahorias",
    "papa": "papas", "kiwi": "kiwis", "limon": "limones", "limón": "limones",
    "guineo": "guineos", "manzana": "manzanas", "fresa": "fresas", "pepino": "pepinos",
    # [P3-DISPLAY-GRAMMAR · 2026-07-05] (review visual #6: "2½ tomate", "1 chuletas",
    # "1 Lechosa mediano") + alimentos contables frecuentes del catálogo.
    "chuleta": "chuletas", "tortilla": "tortillas", "lechosa": "lechosas", "batata": "batatas",
    "mango": "mangos", "naranja": "naranjas", "pera": "peras", "aguacate": "aguacates",
    "clara": "claras", "sardina": "sardinas", "arepita": "arepitas", "mandarina": "mandarinas",
    # [P1-VEG-GHOST-RAW-SYNC · 2026-07-06] (review #9: "8½ aceitunas verde") + contables vistos.
    "aceituna": "aceitunas", "casabe": "casabes", "molondron": "molondrones",
    # [P2-ORGAN-MEAT-CAP · 2026-07-06] "2½ puerro mediano" (review #10).
    "puerro": "puerros",
    # [P2-STEM-FILLER-TOKENS · 2026-07-06] review #11: "½ guayabas fresco", "1 filetes de pescado".
    "guayaba": "guayabas", "filete": "filetes",
    # [P2-BLANCH-INGREDIENT-TRUTH batch · 2026-07-06] review #12: "20 g de Maní fileteadas"
    # (renombre almendras→maní dejó el adjetivo femenino plural sobre sustantivo masc sing).
    "maní": "maníes", "mani": "manies",
    # [P2-COUNTFRUIT-GRAMMAR · 2026-07-06] (review #14: "1 ciruelas (50 g)") frutas CONTABLES
    # que el singularizador de "1 <plural>" no cubría (solo tenía unidades). Concordancia
    # bidireccional: "1 ciruelas"→"1 ciruela", "2 ciruela"→"2 ciruelas".
    "ciruela": "ciruelas", "uva": "uvas", "durazno": "duraznos", "mandarina": "mandarinas",
    "chinola": "chinolas", "guineo": "guineos", "melocoton": "melocotones", "higo": "higos",
})
# [P2-COUNTFRUIT-GRAMMAR · 2026-07-06] las frutas contables también al singularizador "1 <plural>"
# (el mapa base _DISPLAY_SINGULAR solo tenía unidades; el inverso auto-generado no las incluía
# porque se definieron en el .update de _DISPLAY_PLURAL, no en el base).
_DISPLAY_SINGULAR.update({
    "ciruelas": "ciruela", "uvas": "uva", "duraznos": "durazno", "mandarinas": "mandarina",
    "chinolas": "chinola", "melocotones": "melocoton", "higos": "higo",
})

# [P3-DISPLAY-GRAMMAR · 2026-07-05] Concordancia número/género de leads con FRACCIÓN/mixto
# ("2½ tomate" → "2½ tomates", "½ cdas" → "½ cda", "1 chuletas" → "1 chuleta") + adjetivo
# ADYACENTE curado ("2 guineos mediano" → "medianos"; "1 Lechosa mediano" → "1 lechosa mediana")
# + minúscula de alimentos conocidos capitalizados. El lead decimal ya lo cubría el prettify;
# esta capa ve fracciones unicode y mixtos que _DISPLAY_LEAD_RE no parsea. Curado y conservador:
# adjetivo desconocido en -o/-a adyacente → el sustantivo NO se toca (jamás crear un mismatch).
_GRAMMAR_FRAC_MAP = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1.0 / 3.0, "⅔": 2.0 / 3.0}
_GRAMMAR_LEAD_RE = re.compile(
    r"^\s*(?P<lead>\d+(?:[.,]\d+)?\s?[½¼¾⅓⅔]?|[½¼¾⅓⅔])\s+"
    r"(?P<word>[A-Za-zÁÉÍÓÚÑÜáéíóúñü][\wáéíóúñü]*)(?P<rest>.*)$")
# [review #12] + unidades de peso/volumen: "20 g de Maní fileteadas" necesitaba la
# rama de unidad para concordar el alimento tras "de".
_GRAMMAR_UNITS = {"cda", "cdta", "taza", "cucharada", "cucharadita", "g", "gr", "gramos", "ml", "kg"}
_GRAMMAR_PLURAL_TO_SING = {v: k for k, v in _DISPLAY_PLURAL.items()}
_GRAMMAR_ADJ_GENDER_STEMS = ("median", "pequeñ", "pequen", "madur", "fresc", "magr", "pelad",
                             "picad", "rallad", "cocid", "hervid", "tostad", "asad", "enter",
                             # [review #12] "Maní fileteadas" → "Maní fileteado".
                             "filetead")
_GRAMMAR_ADJ_INVARIANT = ("grande", "verde")
# [P3-RECIPE-POLISH-4 · 2026-07-06] invariantes de género con plural en -es ("1 tortilla
# integrales" vivo → "integral"; "integrales" no se deriva con rstrip('s')).
_GRAMMAR_ADJ_INVARIANT_ES = ("integral",)
_GRAMMAR_ADJ_ADJACENT_RE = re.compile(r"^(\s+)([a-záéíóúñü]+)(\b.*)$")
_GRAMMAR_CONNECTORS = {"de", "del", "con", "en", "al", "para", "y", "o", "u", "sin"}


def _grammar_lead_value(lead: str):
    lead = str(lead).strip()
    if lead in _GRAMMAR_FRAC_MAP:
        return _GRAMMAR_FRAC_MAP[lead]
    m = re.match(r"^(\d+(?:[.,]\d+)?)\s?([½¼¾⅓⅔])?$", lead)
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    if m.group(2):
        v += _GRAMMAR_FRAC_MAP[m.group(2)]
    return v


def _grammar_inflect_adj(adj_low: str, plural: bool, feminine: bool):
    """Adjetivo del set curado concordado en género/número; None si no es del set."""
    # [P3-RECIPE-POLISH-4 · 2026-07-06] invariantes con plural en -es ("integral"/"integrales").
    _base_es = adj_low[:-2] if adj_low.endswith("es") else adj_low
    if _base_es in _GRAMMAR_ADJ_INVARIANT_ES:
        return _base_es + ("es" if plural else "")
    base = adj_low.rstrip("s")
    if base in _GRAMMAR_ADJ_INVARIANT:
        return base + ("s" if plural else "")
    for stem in _GRAMMAR_ADJ_GENDER_STEMS:
        if base in (stem + "o", stem + "a"):
            return stem + ("a" if feminine else "o") + ("s" if plural else "")
    return None


def _fix_display_grammar(s: str) -> str:
    """[P3-DISPLAY-GRAMMAR · 2026-07-05] Ver comment del bloque. Display-only, fail-safe."""
    try:
        m = _GRAMMAR_LEAD_RE.match(s)
        if not m:
            return s
        val = _grammar_lead_value(m.group("lead"))
        if val is None or val <= 0:
            return s
        word, rest = m.group("word"), m.group("rest")
        w_orig_low = word.lower()
        _known = (w_orig_low in _DISPLAY_PLURAL or w_orig_low in _GRAMMAR_PLURAL_TO_SING
                  or w_orig_low.rstrip("s") in _GRAMMAR_UNITS)
        # 0) alimento/unidad conocidos capitalizados a mitad de línea → minúscula
        _case_fixed_word = w_orig_low if (word[:1].isupper() and _known) else word
        plural_target = val > 1.0 + 1e-9
        is_unit = w_orig_low.rstrip("s") in _GRAMMAR_UNITS
        new_word = _case_fixed_word
        # [review #12] unidades de peso/volumen INVARIANTES: "20 g" jamás "20 gs".
        _UNIT_INVARIANT = {"g", "gr", "gramos", "ml", "kg"}
        if is_unit and w_orig_low in _UNIT_INVARIANT:
            new_word = w_orig_low
        elif is_unit:
            new_word = w_orig_low.rstrip("s") + ("s" if plural_target else "")
        elif plural_target and w_orig_low in _DISPLAY_PLURAL:
            new_word = _DISPLAY_PLURAL[w_orig_low]
        elif not plural_target and w_orig_low in _GRAMMAR_PLURAL_TO_SING:
            new_word = _GRAMMAR_PLURAL_TO_SING[w_orig_low]
        new_rest = rest
        if is_unit:
            # [P3-RECIPE-POLISH-4 · 2026-07-06] "2 tazas de Lechosa frescos" — concordancia del
            # ALIMENTO tras la unidad: minúscula del alimento conocido + adjetivo con el número/
            # género del alimento (no de la unidad).
            m_uf = re.match(r"^(\s+de\s+)(?P<f>[A-Za-zÁÉÍÓÚÑÜáéíóúñü][\wáéíóúñü]*)"
                            r"(?P<a>\s+[a-záéíóúñü]+)?(?P<tail>.*)$", rest)
            if m_uf:
                _fw = m_uf.group("f")
                _fw_low = _fw.lower()
                _known_f = _fw_low in _DISPLAY_PLURAL or _fw_low in _GRAMMAR_PLURAL_TO_SING
                _new_fw = _fw_low if (_fw[:1].isupper() and _known_f) else _fw
                _adj_part = m_uf.group("a") or ""
                _new_adj_part = _adj_part
                if _known_f and _adj_part:
                    _f_plural = _fw_low in _GRAMMAR_PLURAL_TO_SING
                    _f_sing = _GRAMMAR_PLURAL_TO_SING.get(_fw_low, _fw_low)
                    _adj_low2 = _adj_part.strip().lower()
                    _infl2 = _grammar_inflect_adj(_adj_low2, _f_plural, _f_sing.endswith("a"))
                    if _infl2 and _infl2 != _adj_low2:
                        _new_adj_part = " " + _infl2
                if _new_fw != _fw or _new_adj_part != _adj_part:
                    new_rest = m_uf.group(1) + _new_fw + _new_adj_part + m_uf.group("tail")
        if not is_unit:
            _sing_form = _GRAMMAR_PLURAL_TO_SING.get(new_word.lower(), new_word.lower())
            _feminine = _sing_form.endswith("a")
            _noun_plural = new_word.lower() in _GRAMMAR_PLURAL_TO_SING
            m_adj = _GRAMMAR_ADJ_ADJACENT_RE.match(rest)
            if m_adj and m_adj.group(2) not in _GRAMMAR_CONNECTORS:
                _adj_low = m_adj.group(2).lower()
                _infl = _grammar_inflect_adj(_adj_low, _noun_plural, _feminine)
                if _infl is not None:
                    if _infl != _adj_low:
                        new_rest = m_adj.group(1) + _infl + m_adj.group(3)
                elif (re.fullmatch(r"[a-záéíóúñü]+[oa]s?", _adj_low)
                      and new_word.lower() != w_orig_low):
                    # adjetivo DESCONOCIDO en -o/-a adyacente y cambio de NÚMERO planeado:
                    # pluralizar el sustantivo crearía un mismatch nuevo ("2½ tomates picado")
                    # → se revierte el número; el lowercase-fix sí se queda.
                    new_word = _case_fixed_word
        if new_word == word and new_rest == rest:
            return s
        return s[:m.start("word")] + new_word + new_rest
    except Exception:
        return s


# [P3-STEP-FRACTIONS · 2026-07-05] (review visual #6: "2.5 cdas de mantequilla", "0.5 taza de
# agua tibia", "vierte 1/4 de la masa" en PASOS mientras la lista muestra fracciones unicode)
# Decimales de fracción y fracciones ASCII en pasos → unicode. Lookarounds anti-rango/fecha
# ("8-10 minutos", "10/12" intactos); temperaturas enteras no matchean.
_STEP_DEC_TAIL = {".5": "½", ",5": "½", ".25": "¼", ",25": "¼", ".75": "¾", ",75": "¾",
                  ".33": "⅓", ",33": "⅓", ".67": "⅔", ",67": "⅔", ".66": "⅔", ",66": "⅔"}
_STEP_ASCII_FRAC_MAP = {"1/2": "½", "1/4": "¼", "3/4": "¾", "1/3": "⅓", "2/3": "⅔"}
_STEP_ASCII_FRAC_RE = re.compile(r"(?<![\d/.,])(1/2|1/4|3/4|1/3|2/3)(?![\d/])")
_STEP_DECIMAL_RE = re.compile(r"(?<![\d/.,])(\d*)([.,](?:25|33|5|66|67|75))(?=[\s)])")


def prettify_step_fractions(step: str) -> str:
    """[P3-STEP-FRACTIONS · 2026-07-05] Pulido cosmético de fracciones en pasos de receta.
    Display-only (los pasos son texto humano; qty-sync ya soporta mixtos unicode). Fail-safe."""
    try:
        s = str(step)
        s = _STEP_ASCII_FRAC_RE.sub(lambda mm: _STEP_ASCII_FRAC_MAP[mm.group(1)], s)

        def _dec(mm):
            frac = _STEP_DEC_TAIL.get(mm.group(2))
            if frac is None:
                return mm.group(0)
            whole = mm.group(1)
            return (whole if whole and whole != "0" else "") + frac

        return _STEP_DECIMAL_RE.sub(_dec, s)
    except Exception:
        return str(step)
# [P2-UNITLESS-SPOON-LEAD · 2026-07-05] ("Cda de miel (opcional)", "Cdta de miel") — unidad de
# cuchara/taza SIN número líder: se asume 1 ("1 cda de miel"). Solo unidades de medida (jamás
# alimentos); "Sal al gusto" no matchea.
_UNITLESS_SPOON_LEAD_RE = re.compile(r"^\s*(cda|cdta|cucharada|cucharadita|taza)(s?)\s+de\s+",
                                     re.IGNORECASE)
_DISPLAY_LEAD_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s+(\S+)(.*)$")


# [P2-DISPLAY-THIRDS · 2026-07-05] (screenshots del plan vivo 23c958bb) Decimales INTERNOS
# "jugo de 0.5 limón" → "jugo de ½ limón" (el lead-prettify no los veía: esa línea no tiene
# cantidad líder) + tercios en el lead ("0.33 taza de harina" → "⅓ taza"). Display-only.
_DEC_FRAC_MAP = {"0.25": "¼", "0,25": "¼", "0.33": "⅓", "0,33": "⅓", "0.5": "½", "0,5": "½",
                 "0.66": "⅔", "0,66": "⅔", "0.67": "⅔", "0,67": "⅔", "0.75": "¾", "0,75": "¾",
                 # [P2-INNER-ASCII-FRAC · 2026-07-05] "Jugo de 1/2 limón" / "de 1/4 limón"
                 # (plan 7e4e5570) — el prettify interno solo cubría decimales.
                 "1/2": "½", "1/4": "¼", "3/4": "¾", "1/3": "⅓", "2/3": "⅔"}
_INNER_DECIMAL_FRAC_RE = re.compile(
    r"\bde\s+(0[.,](?:25|33|5|66|67|75)|1/2|1/4|3/4|1/3|2/3)(?=\s)")
_LEAD_THIRDS = ((1.0 / 3.0, "⅓"), (2.0 / 3.0, "⅔"))
# [P2-CDAS-TO-CUPS · 2026-07-05] "11 cdas de harina de trigo (77g)" (plan 7e4e5570): medible pero
# ridículo — ≥6 cdas se promueve a tazas (conversión de VOLUMEN exacta: 16 cdas = 1 taza, sin
# densidad), snap a cuartos/tercios. Display-only (raw intacto para compras).
_CUPS_SNAP = tuple(sorted({k / 4.0 for k in range(1, 13)} | {k / 3.0 for k in range(1, 10)}))
_CUPS_FRAC = {0.25: "¼", 1.0 / 3.0: "⅓", 0.5: "½", 2.0 / 3.0: "⅔", 0.75: "¾"}


def _cups_from_spoons(qty_cdas: float) -> str:
    """Convierte cdas → string de tazas snapeado (¼/⅓/½/⅔/¾ + enteros). '' si no snapea bien."""
    cups = qty_cdas / 16.0
    best = min(_CUPS_SNAP, key=lambda c: abs(c - cups))
    if abs(best - cups) > 0.09:
        return ""
    whole = int(best + 1e-9)
    rem = best - whole
    frac = ""
    for fv, fc in _CUPS_FRAC.items():
        if abs(rem - fv) < 0.02:
            frac = fc
            break
    if rem > 0.02 and not frac:
        return ""
    return (str(whole) if whole else "") + frac


def _restore_gram_name_specificity(display_list, raw_list):
    """[P2-DISPLAY-NAME-SPECIFICITY · 2026-07-05] Reparación mecánica por índice: mismo lead en
    gramos y el food del raw EXTIENDE al del display → el display adopta el nombre específico
    (limpiando el sufijo ' cocido/a' — ruido de catálogo). La lista de compras (raw) ya estaba
    bien; esto alinea lo que el usuario VE con lo que compra. Display-only, fail-safe."""
    if not (isinstance(display_list, list) and isinstance(raw_list, list)
            and len(display_list) == len(raw_list)):
        return display_list
    _g_re = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*g\s+de\s+(.+)$", re.IGNORECASE)
    out = []
    for d, r in zip(display_list, raw_list):
        try:
            md, mr = _g_re.match(str(d)), _g_re.match(str(r))
            if md and mr and md.group(1).replace(",", ".") == mr.group(1).replace(",", "."):
                fd = md.group(2).strip().rstrip(".")
                fr = re.sub(r"\s+cocid[oa]s?\s*$", "", mr.group(2).strip().rstrip("."), flags=re.IGNORECASE)
                if fr.lower().startswith(fd.lower()) and len(fr) > len(fd):
                    out.append(f"{md.group(1)} g de {fr}")
                    continue
        except Exception:
            pass
        out.append(d)
    return out


def _prettify_quantity_display(s: str) -> str:
    """[P2-DISPLAY-FRACTIONS · 2026-07-01] (batch P1-DISH-REALISM-BATCH) Pulido cosmético del lead:
    (a) decimales de cuarto (0.25/0.5/0.75/1.5/1.75...) → fracción unicode vía number_to_fraction_str
    ("0.5 papa mediana"→"½ papa mediana", "1.75 cdta"→"1¾ cdta"); (b) concordancia "1 <plural>" →
    "1 <singular>" para el set curado ("1 cdas de aceite"→"1 cda de aceite"). Display-only, fail-safe
    (cualquier duda → string intacto). NO toca strings que ya llevan fracción unicode.
    [P2-DISPLAY-THIRDS · 2026-07-05] + tercios en el lead (0.33→⅓, 0.67→⅔) y decimales INTERNOS
    ("de 0.5 limón" → "de ½ limón") aunque la línea no tenga cantidad líder."""
    try:
        s0 = str(s)
        s0 = _INNER_DECIMAL_FRAC_RE.sub(lambda mm: "de " + _DEC_FRAC_MAP[mm.group(1)], s0)
        # [P2-UNITLESS-SPOON-LEAD] "Cda de miel (opcional)" → "1 cda de miel (opcional)".
        if _UNITLESS_SPOON_LEAD_RE.match(s0):
            s0 = _UNITLESS_SPOON_LEAD_RE.sub(
                lambda mm: f"1 {mm.group(1).lower()} de ", s0, count=1)
        # [P3-RECIPE-POLISH-4 · 2026-07-06] "½ de cebolla" → "½ cebolla" (el 'de' sobra ante un
        # contable conocido; "¼ de taza" — partitivo legítimo — queda intacto por el lookahead).
        s0 = re.sub(r"^(\s*[½¼¾⅓⅔])\s+de\s+(?=(?:cebolla|tomate|huevo|papa|zanahoria|"
                    r"lim[oó]n|guineo|manzana|aguacate|lechosa|batata|mandarina|pepino|"
                    r"aj[ií]|pl[aá]tano)s?\b)", r"\1 ", s0)
        # [P3-DISPLAY-GRAMMAR · 2026-07-05] concordancia número/género de leads fraccionarios
        # ("2½ tomate"→"2½ tomates", "½ cdas"→"½ cda", "1 Lechosa mediano"→"1 lechosa mediana").
        s0 = _fix_display_grammar(s0)
        m = _DISPLAY_LEAD_RE.match(s0)
        if not m:
            return s0
        qty_str, word, rest = m.group(1), m.group(2), m.group(3)
        qty = float(qty_str.replace(",", "."))
        frac_part = qty - int(qty)
        out_qty = qty_str
        if 0 < qty < 10 and (abs(frac_part - 0.25) < 1e-6 or abs(frac_part - 0.5) < 1e-6
                             or abs(frac_part - 0.75) < 1e-6):
            try:
                pretty = number_to_fraction_str(qty)
                if pretty:
                    out_qty = pretty
            except Exception:
                pass
        elif 0 < qty < 10:
            # [P2-DISPLAY-THIRDS] 0.33→⅓, 1.67→1⅔ (redondeo a cuartos no los cubre).
            for _tv, _tc in _LEAD_THIRDS:
                if abs(frac_part - _tv) < 0.02:
                    out_qty = (str(int(qty)) if int(qty) else "") + _tc
                    break
        out_word = word
        if abs(qty - 1.0) < 1e-6 and word.lower() in _DISPLAY_SINGULAR:
            out_word = _DISPLAY_SINGULAR[word.lower()]
        # [P2-PLURAL-CONCORDANCE] "2 huevo" → "2 huevos" (solo conteos ENTEROS >1, mapa curado).
        elif qty > 1.0 + 1e-6 and abs(frac_part) < 1e-6 and word.lower() in _DISPLAY_PLURAL:
            out_word = _DISPLAY_PLURAL[word.lower()]
        # [P2-CDAS-TO-CUPS · 2026-07-05] ≥6 cdas → tazas (16 cdas = 1 taza, volumen exacto).
        if word.lower() in ("cda", "cdas", "cucharada", "cucharadas") and qty >= 6:
            _cups = _cups_from_spoons(qty)
            if _cups:
                _cup_word = "taza" if _cups in ("¼", "⅓", "½", "⅔", "¾", "1") else "tazas"
                return f"{_cups} {_cup_word}{rest}"
        # [P2-TSP-TO-TBSP · 2026-07-05] ≥3 cdta → cdas ("8¾ cdta de miel" → "3 cdas de miel";
        # 3 cdta = 1 cda, volumen exacto). Snap a cuartos/tercios; sin snap limpio → intacto.
        if word.lower() in ("cdta", "cdtas", "cucharadita", "cucharaditas") and qty >= 3:
            _tbsp_val = qty / 3.0
            _tbsp_best = min(_CUPS_SNAP, key=lambda c: abs(c - _tbsp_val))
            if abs(_tbsp_best - _tbsp_val) <= 0.09:
                _tw = int(_tbsp_best + 1e-9)
                _tr = _tbsp_best - _tw
                _tf = next((fc for fv, fc in _CUPS_FRAC.items() if abs(_tr - fv) < 0.02), "")
                if _tr <= 0.02 or _tf:
                    _tbsp_str = (str(_tw) if _tw else "") + _tf
                    _tbsp_word = "cda" if _tbsp_best <= 1.0 else "cdas"
                    return f"{_tbsp_str} {_tbsp_word}{rest}"
        if out_qty == qty_str and out_word == word:
            return s0
        return f"{out_qty} {out_word}{rest}"
    except Exception:
        return s


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
