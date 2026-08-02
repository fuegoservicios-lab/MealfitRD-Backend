"""[P1-shop-coh-1 · 2026-05-07] SSOT de normalización de unidades.

Antes existían dos maps independientes:
  - shopping_calculator._parse_quantity (cadena if/elif sobre `unit_str`)
  - db_inventory._CANONICAL_UNIT_MAP (dict)

Si un alias se añadía a uno y no al otro, el aggregator sumaba en una
unidad y la deducción de inventario operaba en otra → divergencia
silenciosa entre Σ(recetas) y Σ(lista de compras). Este módulo es la
única fuente de verdad. Ambos clientes leen de aquí.
"""

CANONICAL_UNIT_MAP: dict = {
    # Peso
    'g': 'g', 'gr': 'g', 'gramo': 'g', 'gramos': 'g',
    'kg': 'kg', 'kilo': 'kg', 'kilos': 'kg',
    'kilogramo': 'kg', 'kilogramos': 'kg',
    'lb': 'lb', 'lbs': 'lb', 'libra': 'lb', 'libras': 'lb',
    'oz': 'oz', 'onza': 'oz', 'onzas': 'oz',

    # Volumen
    'ml': 'ml', 'mililitro': 'ml', 'mililitros': 'ml',
    'l': 'l', 'litro': 'l', 'litros': 'l',
    'taza': 'taza', 'tazas': 'taza',

    # Cucharas
    'cda': 'cda', 'cdas': 'cda',
    'cucharada': 'cda', 'cucharadas': 'cda',
    'cdta': 'cdta', 'cdtas': 'cdta',
    'cdita': 'cdta', 'cditas': 'cdta',
    'cucharadita': 'cdta', 'cucharaditas': 'cdta',

    # Containers
    'paquete': 'paquete', 'paquetes': 'paquete',
    'paquetico': 'paquete', 'paqueticos': 'paquete',
    'pqte': 'paquete', 'paq': 'paquete',
    'funda': 'paquete', 'fundas': 'paquete',
    'fundita': 'paquete', 'funditas': 'paquete',
    'cartón': 'paquete', 'carton': 'paquete', 'cartones': 'paquete',
    'caja': 'caja', 'cajas': 'caja',
    'bolsa': 'bolsa', 'bolsas': 'bolsa',
    'bolsita': 'bolsa', 'bolsitas': 'bolsa',
    'tetra': 'tetra', 'tetrapak': 'tetra',
    'galón': 'galón', 'galon': 'galón', 'galones': 'galón',
    'jarra': 'jarra', 'jarras': 'jarra',
    'sobre': 'sobre', 'sobres': 'sobre',
    'sobrecito': 'sobre', 'sobrecitos': 'sobre',
    'lata': 'lata', 'latas': 'lata',
    'pote': 'pote', 'potes': 'pote', 'tarro': 'pote',
    'envase': 'pote', 'envases': 'pote',
    'botella': 'botella', 'botellas': 'botella',
    'frasco': 'botella', 'frascos': 'botella',

    # Discretas
    'unidad': 'unidad', 'unidades': 'unidad',
    'ud': 'unidad', 'uds': 'unidad', 'unid': 'unidad',
    'diente': 'diente', 'dientes': 'diente',
    'cabeza': 'cabeza', 'cabezas': 'cabeza',
    'hoja': 'hoja', 'hojas': 'hoja',
    'rebanada': 'rebanada', 'rebanadas': 'rebanada',
    'lonja': 'rebanada', 'lonjas': 'rebanada',
    'mazo': 'mazo', 'mazos': 'mazo',
    'atado': 'mazo', 'atados': 'mazo',
    'manojo': 'mazo', 'manojos': 'mazo',

    # Pizca / abstractos
    'pizca': 'pizca', 'pizcas': 'pizca',
    'chin': 'pizca', 'toque': 'pizca', 'toques': 'pizca',
    'chorrito': 'pizca', 'chorritos': 'pizca',
    'puñado': 'pizca', 'puñados': 'pizca',
    'ramita': 'pizca', 'ramitas': 'pizca',
    'hojita': 'pizca', 'hojitas': 'pizca',
    'al gusto': 'pizca',
}


def canonicalize_unit(raw):
    """Normaliza un alias a su unidad canónica.

    Args:
        raw: string crudo (case-insensitive, opcional punto final).

    Returns:
        Unidad canónica si `raw` matchea el SSOT, o None si es desconocido.
        El caller decide el fallback (típicamente: rebobinar al name y usar 'unidad').
    """
    if not raw:
        return None
    key = str(raw).strip().lower().rstrip('.')
    return CANONICAL_UNIT_MAP.get(key)


# ──────────────────────────────────────────────────────────────────────────
# [P1-NEW-10 · 2026-05-11] Conversor de unidades dentro del mismo sistema.
# ──────────────────────────────────────────────────────────────────────────
# Mapa canonical_unit → (base_unit, factor_to_base). Solo conversiones
# SEGURAS dentro de un mismo sistema físico (peso↔peso, volumen↔volumen).
# Cross-system (taza→g requiere densidad por alimento) queda OUT-OF-SCOPE
# de esta capa — el caller que necesite densidad debe usar otro helper
# (futuro P2: `DENSITY_BY_FOOD` + `to_grams_for_food(qty, unit, food)`).
#
# Por qué existe (audit 2026-05-11):
#   `compare_expected_vs_aggregated` iteraba unidades por nombre literal,
#   tratando `kg` y `g` del mismo alimento como dos entradas distintas
#   ({kg: 1.0} en expected vs {g: 1000.0} en aggregated → ambos lados
#   reportaban fantasma). Sin conversor, el guard quedaba "frágil bajo
#   prompt drift": cualquier modelo nuevo que normalice "1 kg" a "1000 g"
#   (o viceversa) dispararía falsos positivos masivos. Hoy LLM normaliza
#   simétricamente — el fix es preventivo (knob canary off por default).
#
# Factores estándar de cocina (US/EU comunes en RD):
#   kg = 1000 g            lb = 453.592 g       oz = 28.3495 g
#   l  = 1000 ml           taza = 240 ml        cda = 15 ml      cdta = 5 ml
#
# Unidades discretas (unidad/diente/cabeza/hoja/...), containers (paquete/
# caja/...), y abstractas (pizca/chin/...) NO se convierten — no tienen
# magnitud universal sin contexto de alimento. Se devuelven tal cual del
# helper para que el caller las preserve.
UNIT_TO_BASE_FACTOR: dict = {
    # Peso → base 'g'
    'g':    ('g', 1.0),
    'kg':   ('g', 1000.0),
    'lb':   ('g', 453.592),
    'oz':   ('g', 28.3495),
    # Volumen → base 'ml'
    'ml':   ('ml', 1.0),
    'l':    ('ml', 1000.0),
    'taza': ('ml', 240.0),
    'cda':  ('ml', 15.0),
    'cdta': ('ml', 5.0),
}


def to_base_amount(qty, unit_raw):
    """[P1-NEW-10 · 2026-05-11] Convierte `(qty, unit)` a `(qty_base, base_unit)`.

    Solo opera sobre unidades convertibles del mismo sistema físico
    (peso↔peso, volumen↔volumen). Unidades fuera del mapa o no convertibles
    (`unidad`, `diente`, `pizca`, `paquete`, etc.) se devuelven tal cual.

    Args:
        qty: cantidad numérica (int/float/string castable a float).
        unit_raw: string crudo de unidad (se canonicaliza internamente).

    Returns:
        Tupla `(qty_base, base_unit_or_original)`:
          - Si unit es convertible: `(qty * factor, base_unit)`.
          - Si unit no es convertible (ej. 'unidad'): `(qty, canonical_unit)`.
          - Si unit es desconocida: `(qty, raw)` (caller decide qué hacer).
        Inputs no numéricos retornan `(qty, raw)` intactos.

    Garantías:
      - Idempotente: `to_base_amount(*to_base_amount(q, u)) == to_base_amount(q, u)`
        (g→g, ml→ml ya están en base).
      - NO cross-system: nunca convierte taza→g sin densidad explícita.
      - Determinístico: misma entrada produce misma salida (sin float drift
        más allá de los factores hardcoded).
    """
    try:
        qty_f = float(qty)
    except (TypeError, ValueError):
        return (qty, unit_raw)
    canonical = canonicalize_unit(unit_raw) if unit_raw else None
    if canonical is None:
        # Unidad desconocida: devolver intacta para que el caller decida.
        return (qty_f, unit_raw)
    entry = UNIT_TO_BASE_FACTOR.get(canonical)
    if entry is None:
        # Canonical conocida pero no convertible (unidad/diente/pizca/etc).
        return (qty_f, canonical)
    base_unit, factor = entry
    return (qty_f * factor, base_unit)


# ──────────────────────────────────────────────────────────────────────────
# [P1-CONTAINER-SERVABLE · 2026-08-02] EL ENVASE NO ES EL CONTENIDO.
# ──────────────────────────────────────────────────────────────────────────
# Forense del 2026-08-02: `user_inventory` del owner tenía "Huevo: 2 cartón
# (20 uds.)" — el pantry guard (`constants.validate_ingredients_against_pantry`)
# y su espejo de regenerate-day (`routers/plans._inventory_grams_ledger`)
# comparaban unidades-de-PLATO contra unidades-de-ENVASE sin expandir: "2
# cartón" se contaba como "2 huevos sueltos" (el ENVASE, no su CONTENIDO), así
# que un plato de 3 huevos veía "límite: 2" con ~40 huevos reales en la
# nevera. El mismo gap golpea envases por PESO sin anotación (`_inventory_
# grams_ledger` no tenía NINGÚN tratamiento de contenedor: "1 pote (1.96 kg)
# de Yogurt" caía a `to_grams('unidad')` sin `density_g_per_unit` → 0g
# disponibles con ~2kg reales).
#
# Este helper es el SSOT de la expansión envase→contenido SERVABLE, usado por
# AMBOS espejos:
#   - cartón (huevos): parsea "(N uds.)" del string si está presente; sin
#     anotación, tabla canónica RD por nombre (hoy solo huevo=15, el tamaño
#     mediano entre los 12-30 verificados in-store — ampliar aquí si aparece
#     otro envase-por-unidades real).
#   - paquete/pote/lata/funda/botella/caja/bolsa/tetra/frasco/sobre (peso):
#     parsea "(N g/kg/oz/ml/l/lb/lbs)" del string si está presente; si no,
#     `container_weight_g` del SSOT `master_ingredients` (vía
#     `nutrition_db.IngredientNutritionDB`, `allow_category_fallback=True`) o,
#     en último caso, el fallback conservador por categoría que YA usa el
#     aggregator de la lista de compras (`shopping_calculator.
#     _fallback_container_weight_g` — un SOLO número por categoría, no un 3er
#     estimado inventado aquí).
#
# Fail-open POR DISEÑO: cualquier caso que no pueda expandirse con confianza
# retorna `None` — el caller conserva su comportamiento ACTUAL (mejor
# conservador que inventar un peso/conteo). `allow_category_fallback=False`
# (default) además desactiva el fallback genérico por categoría — lo usa
# `constants.validate_ingredients_against_pantry`, que YA maneja
# correctamente el caso "envase CON peso anotado" desde antes de este fix
# (P3-PANTRY-GUARD-UNICODE-FRACTIONS) y no debe cambiar el comportamiento de
# los ~15 tests existentes que ejercitan envases SIN anotación (esos siguen
# cayendo al `_to_base_unit` legacy, "1 unidad abstracta" — correcto para ese
# caller). `_inventory_grams_ledger` (routers/plans.py) no tenía NINGÚN
# tratamiento previo de contenedor, así que ahí `allow_category_fallback=True`
# es estrictamente una mejora (antes: 0g; ahora: un estimado conservador).
_STRIP_ACCENTS_MAP = str.maketrans(
    "áéíóúÁÉÍÓÚñÑüÜ", "aeiouAEIOUnNuU"
)


def _strip_accents_for_container(s) -> str:
    """Sin depender de `constants.strip_accents` (evitar acoplar este módulo,
    hoy sin imports, a un ciclo con `constants.py`). Cobertura suficiente para
    el vocabulario de envases es-DO (vocales acentuadas + ñ/ü)."""
    return str(s or "").translate(_STRIP_ACCENTS_MAP)


import re as _re_container

# Fracción unicode → decimal (mismo set que `constants._UNICODE_FRACTION_MAP`,
# duplicado deliberadamente para no importar `constants` desde aquí).
_CONTAINER_UNICODE_FRACTIONS = {
    "½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3,
    "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875, "⅙": 1 / 6, "⅚": 5 / 6,
}

# "(20 uds.)" / "(1.96 kg)" / "(½ lb)" — grupo 1 = cantidad, grupo 2 = unidad.
_CONTAINER_QTY_RE = _re_container.compile(
    r"\(([¼½¾⅓⅔⅛⅜⅝⅞⅙⅚]|\d+(?:[.,]\d+)?)\s*(uds?\.?|unidades?|u\.?|g|gr|kg|oz|ml|l|lb|lbs)\)",
    _re_container.IGNORECASE,
)

_CONTAINER_MASS_VOL_TO_G = {
    "g": 1.0, "gr": 1.0, "kg": 1000.0, "oz": 28.3495, "lb": 453.592, "lbs": 453.592,
    # [P1-CONTAINER-SERVABLE] ml/l tratados como ~agua (1 g/ml) para el estimado del
    # ENVASE — mismo supuesto conservador que el resto del repo usa para "cosas raras"
    # sin densidad propia (ver constants.py fallback 5g/unidad de abstractos).
    "ml": 1.0, "l": 1000.0,
}

# Tabla canónica RD: envases contados por UNIDADES (no peso) sin anotación "(N uds.)"
# en el string. Hoy solo huevo — ampliar aquí (NO crear una 4ª tabla en otro archivo)
# si aparece otro envase-por-unidades real medido in-store.
_DEFAULT_UNITS_PER_CONTAINER_BY_NAME_KEYWORD = {
    "huevo": 15.0,
}

_CONTAINER_GRAMS_KEYWORDS = (
    "paquete", "pote", "lata", "funda", "botella", "caja", "bolsa",
    "tetra", "frasco", "sobre", "envase", "tarro",
)


def _parse_container_frac_or_num(raw: str) -> float:
    raw = (raw or "").strip()
    if raw in _CONTAINER_UNICODE_FRACTIONS:
        return _CONTAINER_UNICODE_FRACTIONS[raw]
    try:
        return float(raw.replace(",", "."))
    except (TypeError, ValueError):
        return 0.0


def expand_container_to_servable(name, qty, unit, allow_category_fallback: bool = False, db=None):
    """[P1-CONTAINER-SERVABLE · 2026-08-02] Envase→contenido SERVABLE.

    `unit` puede ser el string de unidad crudo ("cartón (20 uds.)") O la línea completa
    del inventario ("2 cartones (20 uds.) de Huevo") — la búsqueda es por keyword/regex,
    no exige match exacto, así que ambos callers (unit column vs. línea formateada) sirven
    tal cual sin pre-procesar.

    Returns:
        `(qty_servable, unit_servable)` con `unit_servable` ∈ {'unidad', 'g'}, o `None` si
        no se puede expandir con confianza (fail-open — ver comentario del módulo arriba).

    Ejemplos verificados (forense 2026-08-02):
      expand_container_to_servable("Huevo", 2, "2 cartones (20 uds.) de Huevo") → (40.0, 'unidad')
      expand_container_to_servable("Huevo", 1, "cartón") → (15.0, 'unidad')  # fallback RD
      expand_container_to_servable("Yogurt griego", 1, "pote (1.96 kg)", allow_category_fallback=True)
          → (1960.0, 'g')
    """
    try:
        qty_f = float(qty)
    except (TypeError, ValueError):
        return None
    if qty_f <= 0:
        return None
    unit_str = str(unit or "")
    if not unit_str.strip():
        return None
    unit_low = _strip_accents_for_container(unit_str.lower())
    name_low = _strip_accents_for_container(str(name or "").lower())

    is_carton = bool(_re_container.search(r"\bcarton", unit_low))
    is_grams_container = any(
        _re_container.search(r"\b" + kw, unit_low) for kw in _CONTAINER_GRAMS_KEYWORDS
    )
    if not is_carton and not is_grams_container:
        return None

    m = _CONTAINER_QTY_RE.search(unit_str)

    if is_carton:
        if m:
            tok = m.group(2).strip(". ").lower()
            if tok.startswith("u"):  # uds/ud/unidad/unidades/u
                per_container = _parse_container_frac_or_num(m.group(1))
                if per_container > 0:
                    return (qty_f * per_container, "unidad")
        for kw, default_units in _DEFAULT_UNITS_PER_CONTAINER_BY_NAME_KEYWORD.items():
            if kw in name_low:
                return (qty_f * default_units, "unidad")
        # cartón sin "(N uds.)" reconocible y sin match en la tabla RD (ej. cartón de
        # leche con anotación en kg/l) → cae al path de gramos de abajo, NO retorna aún.

    if m:
        tok = m.group(2).strip(". ").lower()
        if tok in _CONTAINER_MASS_VOL_TO_G:
            per_container_g = _parse_container_frac_or_num(m.group(1)) * _CONTAINER_MASS_VOL_TO_G[tok]
            if per_container_g > 0:
                return (qty_f * per_container_g, "g")

    if not allow_category_fallback:
        return None

    # SSOT `master_ingredients` (container_weight_g) + fallback conservador por
    # categoría — lazy imports (canonical_units.py no importa nada a nivel de módulo;
    # `nutrition_db` importa ESTE módulo a nivel de módulo, así que el import inverso
    # solo es seguro DIFERIDO — mismo patrón que `constants.py` usa con
    # `shopping_calculator._parse_quantity`).
    try:
        if db is None:
            from nutrition_db import IngredientNutritionDB
            db = IngredientNutritionDB()
        info = db.lookup(name)
        if info is not None and getattr(info, "container_weight_g", None):
            return (qty_f * float(info.container_weight_g), "g")
        category = db.category_of(name) if hasattr(db, "category_of") else None
        from shopping_calculator import _fallback_container_weight_g
        fallback_g = _fallback_container_weight_g(category)
        if fallback_g and fallback_g > 0:
            return (qty_f * float(fallback_g), "g")
    except Exception:
        pass

    return None
