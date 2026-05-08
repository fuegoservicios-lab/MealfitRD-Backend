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
