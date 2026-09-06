"""[P1-ARQ25-F7-CULTURE · 2026-09-05 · subfase G «despensa duradera»] SSOT de cuánto AGUANTA un alimento y de
qué cabe en cada día de un ciclo de UNA sola compra según el congelador del usuario.

Por qué existe: `master_ingredients.shelf_life_days` es un relleno (lechuga, fresas, carne y repollo valen 14 por
igual), así que ni el registry ni el validador podían saber qué plato es «de despensa». Aquí la durabilidad es una
tabla de REGLAS por nombre (y default por categoría), y de ella cuelgan:

  - el registry (`dish_registry.derive_logistics` → `days_fresh_min`, `days_with_freezer_min`, `pantry_only`),
  - el filtro de candidatos del blueprint y del prompt (`template_candidates(..., need_days, allow_frozen)`),
  - el validador de fidelidad (`horizon.fresh_beyond_horizon_issues`),
  - la ventana de congelación por modo (`freeze_window_days`: sin congelador 0 · limitado 14 · completo el ciclo).

Cuatro clases:
  pantry     — seco, enlatado, curado, aceites, especias: 90-365 días;
  cold       — refrigerado duradero (huevo, quesos curados, raíces, repollo, cítricos, manzana): 21-90 días;
  freezable  — proteína fresca (pollo, res, cerdo, pescado, mariscos): 3 días en fresco, 90 congelada;
  frozen     — congelado DE FÁBRICA (edamame, papas ralladas, wafles): 1 día fuera, 365 dentro. Exige congelador;
  fresh      — hojas, hierbas, frutas blandas, tomate, aguacate, lácteos frescos: 3-10 días.

Puro y determinista. Fail-open: un nombre desconocido cae al default de su categoría (o a «fresh» 7 días).
"""
from __future__ import annotations

import unicodedata
from typing import Any, Iterable, Optional

FRESH_HORIZON_DAYS = 7
FROZEN_DAYS = 90
FACTORY_FROZEN_DAYS = 365   # [ARQ27-P1-09] lo que venía congelado de fábrica, dentro del congelador
LIMITED_FREEZE_WINDOW_DAYS = 14

# (tokens, clase, días en fresco). Primer match gana; los tokens se buscan como subcadena del nombre normalizado.
_RULES: tuple[tuple[tuple[str, ...], str, int], ...] = (
    # ── excepciones frescas (contienen tokens de otras clases: «sin azúcar», «rellenos», «de mantequilla») ──
    (("huevos rellenos", "panecillos de mantequilla", "pan de maiz", "ensalada de macarrones", "hummus", "bacalaitos"), "fresh", 5),
    (("yogur", "yogurt", "kefir", "suero de mantequilla", "suero costeno", "queso blanco", "queso de hoja", "queso cottage",
      "queso ricotta", "requeson", "queso crema", "cuajada", "nata", "crema", "natilla"), "cold", 14),
    # ── despensa ────────────────────────────────────────────────────────────────────────────
    (("atun en agua", "sardinas en lata", "anchoas", "arenque", "bacalao", "atun"), "pantry", 180),
    (("frijoles refritos", "frijoles horneados", "chili con carne", "maiz dulce en granos", "durazno en almibar",
      "aceitunas", "alcaparrado", "palmito", "leche evaporada", "leche de coco", "leche de cabra en polvo",
      "leche de almendras", "leche de avena", "leche de soya", "soya texturizada"), "pantry", 365),
    # [ARQ27-P1-09 · 2026-09-06] Estos cuatro son CONGELADOS DE FÁBRICA y hasta hoy se clasificaban
    # `pantry` 90: el módulo decía que un paquete de edamame aguanta tres meses en la alacena. En un
    # ciclo de una sola compra SIN congelador pasaban el guard el día 30 sin que nada avisara. Su
    # clase es `frozen`: 1 día fuera del congelador, 365 dentro — y exigen congelador aunque se hayan
    # comprado ya congelados, que es literalmente el criterio de cierre del gap.
    (("papas ralladas", "bolitas de papa", "wafles", "edamame"), "frozen", 1),
    (("arroz", "pasta", "fideos", "coditos", "avena", "harina", "lenteja", "garbanzo", "habichuelas", "frijol", "judias", "alubias",
      "gandules", "guisantes secos", "habas", "quinoa", "bulgur", "cebada", "semola", "casabe", "galletas",
      "pan rallado", "pretzel", "granola", "mezcla para panqueques", "masa para pie"), "pantry", 180),
    (("aceite", "vinagre", "sal", "azucar", "miel", "panela", "jarabe", "salsa", "ketchup", "kétchup", "mostaza",
      "adobo", "sazon", "comino", "oregano", "canela", "pimienta", "laurel", "tomillo", "pimenton", "curcuma",
      "curry", "achiote", "chile en polvo", "chile guajillo", "chile ancho", "chile pasilla", "chile mulato",
      "chile chipotle", "chile de arbol", "sazonador", "vainilla", "polvo de hornear", "cacao", "chocolate de mesa",
      "flor de jamaica", "ron de cocina", "especias", "albahaca seca", "ajo en polvo", "cebolla en polvo",
      "semillas", "chia", "linaza", "ajonjoli", "mani", "nuez", "nueces", "almendra", "pistacho", "merey", "pinones",
      "pasas", "datiles", "ciruela pasa", "tamarindo", "membrillo dulce", "mazapan", "turron", "malvaviscos",
      "mantequilla de mani", "mantequilla de almendras", "arequipe", "nori", "guascas", "sofrito", "pique"), "pantry", 180),
    # ── refrigerado duradero ─────────────────────────────────────────────────────────────
    (("huevo", "clara de huevo", "yema"), "cold", 35),
    (("queso de papa", "queso gouda", "queso cheddar", "queso parmesano", "queso provolone", "queso mozzarella",
      "queso en hebras", "mantequilla"), "cold", 45),
    (("jamon serrano", "jamon iberico", "lomo embuchado", "chorizo espanol", "salami", "pepperoni", "cecina",
      "sobrasada", "panceta iberica", "chistorra", "butifarra"), "cold", 30),
    (("chorizo mexicano", "chorizo verde", "chorizo santarrosano", "longaniza", "salchicha italiana", "morcilla"), "freezable", 5),
    (("chuleta ahumada", "tocineta", "jamon de cocinar", "jamon de sandwich", "jamon de pavo", "salchichas",
      "chicharron", "pavochon", "tofu"), "cold", 14),
    (("ajo", "cebolla", "jengibre"), "cold", 60),
    (("papa", "auyama", "repollo", "zanahoria", "remolacha", "manzana"), "cold", 45),
    (("batata", "name", "coco", "granada", "membrillo", "naranja", "limon", "toronja", "yautia", "tortilla de maiz",
      "tortilla de trigo", "tortilla integral"), "cold", 30),
    (("yuca", "mapuey", "nabo", "tayota", "jicama", "mandarina", "kiwi", "pera", "chinola", "uchuva", "borojo",
      "xoconostle", "chontaduro", "arracacha", "panapen"), "cold", 21),
    (("apio", "puerro", "coles de bruselas", "chile jalapeno", "chile serrano", "chile poblano", "chile habanero",
      "berenjena", "alcachofa", "plátano verde", "platano verde", "calabacin", "nopal", "aji morron", "aji cubanela",
      "coliflor", "brocoli", "bok choy", "uva", "arandanos", "sandia", "melon"), "cold", 10),
    # ── proteína congelable ──────────────────────────────────────────────────────────────
    (("pollo", "pechuga", "muslo", "gallina", "pavo", "carne de res", "res molida", "carne molida", "bistec",
      "cerdo", "pernil", "costilla", "chivo", "cordero", "conejo", "higado", "tilapia", "salmon", "trucha", "mero",
      "filete de pescado", "pescado", "camaron", "gambas", "almejas", "mejillones", "calamar", "pulpo", "cangrejo",
      "vieira", "percebes", "boquerones"), "freezable", 3),
    # ── fresco ───────────────────────────────────────────────────────────────────────────
    (("lechuga", "berro", "rucula", "espinaca", "acelga", "kale", "cilantro", "perejil", "albahaca", "recao",
      "cebollin", "epazote", "hoja santa", "champinon", "huitlacoche", "esparrago", "vainitas", "pepino", "tomate",
      "aguacate", "platano maduro", "guineo", "lechosa", "mango", "pina", "fresa", "mora", "frambuesa", "guayaba",
      "guanabana", "higo", "nispero", "durazno", "ciruela", "lulo", "curuba", "feijoa", "granadilla", "tuna de nopal",
      "leche", "pan ", "pan de agua", "pan sobao", "panecillo", "bagel", "cundeamor", "molondrones", "rabano",
      "bacalaitos frescos"), "fresh", 7),
)
_CATEGORY_DEFAULT = {"despensa": ("pantry", 180), "viveres": ("cold", 21), "vegetales": ("fresh", 7), "frutas": ("fresh", 7),
                     "lacteos": ("fresh", 10), "proteinas": ("freezable", 3)}


# [P1-DURABILITY-FRESH-STATE · 2026-09-05] La tabla de arriba resuelve por el ALIMENTO y no mira la palabra
# de estado, y para cuatro pescados eso invierte el resultado: su nombre es a la vez el de la conserva y el del
# pescado del mostrador. Medido contra este mismo módulo:
#
#     atún fresco      → despensa, 180 días        (la regla de «atun en agua»)
#     bacalao fresco   → despensa, 180 días
#     arenque fresco   → despensa, 180 días
#     anchoas frescas  → despensa, 180 días
#
# Consecuencia real: en un ciclo de 30 días SIN congelador, un plato con atún fresco pasaba el guard el día 25.
#
# Lo que NO se toca, porque no está mal: «arroz cocido» y «claras de huevo» siguen siendo despensa 180 y frío 35.
# Este módulo responde «¿cuánto aguanta lo que el usuario COMPRA?», y lo que compra es arroz y huevos; el plato se
# cocina o se separa ese día. Tratarlos como sobras de nevera bloquearía platos correctos. La conservación de lo
# ya cocinado es otra pregunta y pertenece al modo «cocino por tandas», que este módulo todavía no representa.
_FRESH_QUALIFIERS = ("fresco", "fresca", "frescos", "frescas", "crudo", "cruda", "crudos", "crudas",
                     "del dia", "de mostrador")

# [ARQ27-P1-09 · 2026-09-06] Estado del ENVASE. «Una bebida estable cerrada no conserva el mismo
# horizonte después de abrirse» —criterio de cierre del gap— y hasta hoy la tabla resolvía por el
# alimento y punto: una leche vegetal era 365 días, abierta o no.
#
# El mecanismo es el mismo que el de `fresh_state`: el calificativo del NOMBRE manda sobre la tabla.
# Lo que este módulo NO hace todavía —y conviene decirlo en vez de fingirlo— es deducir solo que un
# cartón abierto el día 1 ya no sirve el día 20 de la misma compra. Eso necesita saber en qué días se
# usa cada ingrediente, y esa pregunta pertenece al modo «cocino por tandas», que este módulo aún no
# representa. Igual que la nota de «arroz cocido» de arriba: la frontera queda declarada, no borrosa.
_OPENED_QUALIFIERS = ("abierto", "abierta", "abiertos", "abiertas", "empezado", "empezada",
                      "destapado", "destapada")
# Alimentos estables SOLO mientras el envase está cerrado. Abiertos son refrigerados de pocos días.
_SHELF_STABLE_UNTIL_OPENED = (
    ("leche de coco", 4), ("leche de almendras", 7), ("leche de avena", 7), ("leche de soya", 7),
    ("leche evaporada", 4), ("frijoles refritos", 4), ("frijoles horneados", 4),
    ("maiz dulce en granos", 4), ("aceitunas", 14), ("alcaparrado", 14), ("palmito", 5),
    ("durazno en almibar", 5), ("salsa de tomate", 7), ("chili con carne", 4),
    ("atun en agua", 2), ("sardinas en lata", 2),
)
# Pescados cuyo nombre desnudo significa CONSERVA en la tabla. Calificados de frescos, son proteína de 3 días.
_PRESERVED_FISH = ("atun", "bacalao", "arenque", "anchoa", "sardina")


def _frozen_days(cls: str, days: int) -> int:
    """Cuánto aguanta CONGELADO. `freezable` (proteína fresca que el usuario congela) 90; `frozen`
    (congelado de fábrica) 365, porque ya venía así y su cadena de frío nunca se rompió."""
    if cls == "freezable":
        return FROZEN_DAYS
    if cls == "frozen":
        return FACTORY_FROZEN_DAYS
    return int(days)


def _norm(s: Any) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return " ".join(s.split())


def classify(name: Any, category: Optional[str] = None) -> dict:
    """{cls, days_fresh, days_frozen, rule} de un alimento por su nombre (canónico o alias).

    `rule` dice de dónde salió el plazo —el token que casó, `fresh_state` o `category_default`— para que un
    plazo raro se pueda auditar sin releer la tabla entera. tooltip-anchor: P1-DURABILITY-FRESH-STATE"""
    n = " " + _norm(name) + " "
    # [P1-DURABILITY-FRESH-STATE] El calificativo manda sobre la tabla, y solo para los pescados cuyo nombre
    # desnudo es el de la conserva: «atún fresco» es proteína de 3 días, no una lata de 180.
    if any((" " + q) in n for q in _FRESH_QUALIFIERS) and any((" " + f) in n for f in _PRESERVED_FISH):
        return {"cls": "freezable", "days_fresh": 3, "days_frozen": FROZEN_DAYS, "rule": "fresh_state"}
    # [ARQ27-P1-09] Envase abierto: la estabilidad era del envase CERRADO, no del alimento.
    if any((" " + q) in n for q in _OPENED_QUALIFIERS):
        for tok, dias in _SHELF_STABLE_UNTIL_OPENED:
            if (" " + _norm(tok)) in n:
                return {"cls": "cold", "days_fresh": int(dias), "days_frozen": int(dias),
                        "rule": "opened_package"}
    for tokens, cls, days in _RULES:
        for tok in tokens:
            t = _norm(tok)
            if not t:
                continue
            hit = (" " + t + " ") in n if len(t) <= 4 else (" " + t) in n   # cortos exactos («sal»≠«salmon»), largos por prefijo («yogur»→«yogurt»)
            if hit:
                return {"cls": cls, "days_fresh": int(days),
                        "days_frozen": _frozen_days(cls, days), "rule": t}
    cls, days = _CATEGORY_DEFAULT.get(_norm(category), ("fresh", 7))
    return {"cls": cls, "days_fresh": int(days), "days_frozen": _frozen_days(cls, days),
            "rule": "category_default"}


def durability_of(constituents: Iterable[Any], categories: Optional[dict] = None) -> dict:
    """Mínimos de una lista de constituyentes (dicts con canonical/name o strings):
    `days_fresh_min` (sin congelador), `days_with_freezer_min` (proteína congelable = 90 días), `pantry_only`
    (todo aguanta ≥ 21 días sin congelar) y las clases presentes."""
    fresh_min, frozen_min, classes = None, None, set()
    for c in constituents or ():
        name = c if isinstance(c, str) else ((c or {}).get("canonical") or (c or {}).get("name"))
        if not name:
            continue
        cat = (categories or {}).get(_norm(name)) if categories else None
        d = classify(name, cat)
        classes.add(d["cls"])
        fresh_min = d["days_fresh"] if fresh_min is None else min(fresh_min, d["days_fresh"])
        frozen_min = d["days_frozen"] if frozen_min is None else min(frozen_min, d["days_frozen"])
    return {"days_fresh_min": fresh_min, "days_with_freezer_min": frozen_min,
            "pantry_only": bool(fresh_min is not None and fresh_min >= 21), "classes": sorted(classes)}


def freeze_window_days(freezer_mode: Any, total_days: int) -> int:
    """Hasta qué día (exclusivo, 0-based) vale una proteína congelada el día de la compra: sin congelador 0,
    limitado 14 (una semana de frescos + una de congelados), completo el ciclo entero."""
    m = _norm(freezer_mode) or "limited"
    total = max(0, int(total_days or 0))
    if m == "none":
        return 0
    if m == "full":
        return total
    return min(LIMITED_FREEZE_WINDOW_DAYS, total) if total else LIMITED_FREEZE_WINDOW_DAYS


def template_fits(days_fresh_min: Optional[int], days_with_freezer_min: Optional[int], need_days: int, allow_frozen: bool) -> bool:
    """¿Un plato cuyos constituyentes aguantan `days_fresh_min` (o `days_with_freezer_min` congelando) sirve
    para el día `need_days` del ciclo? Sin datos ⇒ True (fail-open)."""
    if days_fresh_min is None:
        return True
    if days_fresh_min >= int(need_days):
        return True
    return bool(allow_frozen and days_with_freezer_min is not None and days_with_freezer_min >= int(need_days))


def single_trip_requirements(effective: Optional[dict], day_index: Optional[int]) -> Optional[dict]:
    """Exigencia de durabilidad para un día (0-based) de un ciclo de UNA sola compra; None si el usuario repone
    frescos, el ciclo es semanal o el día cae en la semana de frescos."""
    try:
        shopping = (effective or {}).get("shopping") or {}
        cycle = int(shopping.get("main_cycle_days") or 0)
        if cycle <= FRESH_HORIZON_DAYS or shopping.get("fresh_topup_days") or day_index is None:
            return None
        d = int(day_index)
        if d < FRESH_HORIZON_DAYS:
            return None
        window = freeze_window_days(shopping.get("freezer_mode"), cycle)
        return {"need_days": d + 1, "allow_frozen": d < window, "freezer_mode": _norm(shopping.get("freezer_mode")) or "limited",
                "freeze_window_days": window}
    except Exception:
        return None


def ingredient_issue_beyond_horizon(name: Any, abs_day_index: int, allow_frozen: bool) -> Optional[str]:
    """Código de issue si `name` no aguanta hasta `abs_day_index` (0-based) en un ciclo de una sola compra."""
    d = classify(name)
    need = int(abs_day_index) + 1
    if d["cls"] == "frozen":
        # [ARQ27-P1-09] Código propio: el consejo NO es el mismo. A la proteína fresca se le ofrece
        # una alternativa de despensa; a un congelado de fábrica hay que decirle que sin congelador
        # ese plato no cabe en su compra única.
        return None if (allow_frozen or d["days_fresh"] >= need) else "frozen_needs_freezer"
    if d["cls"] == "freezable":
        return None if (allow_frozen or d["days_fresh"] >= need) else "protein_beyond_freeze_window"
    return None if d["days_fresh"] >= need else "fresh_beyond_horizon"


__all__ = ["FRESH_HORIZON_DAYS", "FROZEN_DAYS", "FACTORY_FROZEN_DAYS", "LIMITED_FREEZE_WINDOW_DAYS",
           "classify", "durability_of", "_FRESH_QUALIFIERS", "_PRESERVED_FISH",
           "_OPENED_QUALIFIERS", "_SHELF_STABLE_UNTIL_OPENED",
           "freeze_window_days", "template_fits", "single_trip_requirements", "ingredient_issue_beyond_horizon"]
