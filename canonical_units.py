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
