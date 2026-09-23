# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-46 · 2026-09-14] El ingrediente que da nombre al plato no se quita.

Segunda prueba RD del dueño (plan 63eedc6b, 14-sep, día determinista): «Guacamole criollo…» sin aguacate (el re-trim de
grasas del guardado lo dejó en nada), «Maní tostado con pasas…» sin maní, «…pasas y dátiles» sin dátiles y «Avena cocida
con leche evaporada…» sin leche evaporada. Cada ajuste de macros tenía su motivo; ninguno miraba qué hace a ese plato ese
plato, y los pasos seguían mandando tostar un maní que ya no estaba.

Alcance: platos con receta congelada (`_recipe_source == "library"` y `_template_id`), porque ahí la plantilla dice qué
lleva y cuánto. IDENTIDAD = los constituyentes que el NOMBRE del plato nombra, más el más pesado de la plantilla (el
guacamole no dice «aguacate»). Dos piezas:

  · `protege_linea(meal, linea)`: los recortes de macros (re-trim de grasas y de carbohidratos) no tocan esas líneas y
    recortan de las demás fuentes.
  · `restaurar_identidad(days)`: el respaldo. Si aun así el alimento FALTA, vuelve con `PISO_FRACCION` de los gramos de la
    plantilla × `_scale_factor`. Sólo lo que falta: medido sobre el plan 63eedc6b, subir también lo que quedó pequeño (el
    salami de 5 g a 41 g) disparaba la grasa del día al 116-122 % y el sodio.

No se toca lo que otro pase SUSTITUYÓ a propósito (autofix de proteína, sustitución por presupuesto, cambio por sodio) ni
lo que choca con una alergia declarada. La línea entra en `ingredients` y en `ingredients_raw` (la lista de compras lee
raw) y los macros se re-miden con el truth-up del repo. Knob `MEALFIT_DISH_IDENTITY_FLOOR` (True).
tooltip-anchor: P1-PLAN-LOTE-46-IDENTIDAD
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

PISO_FRACCION = 0.25
GRAMOS_MIN_IDENTIDAD = 10.0      # canela, orégano o sal de una plantilla no dan identidad
PISO_MIN_G = 5
_PAISES = ("DO", "ES", "US", "MX", "PR", "CO")
_VACIAS = {"de", "del", "la", "el", "los", "las", "con", "y", "en", "al", "a", "e"}


def enabled() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_DISH_IDENTITY_FLOOR", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _sa(s) -> str:
    return unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()


def _palabras(s) -> list:
    return [w for w in re.findall(r"[a-z]+", _sa(s)) if len(w) >= 3 and w not in _VACIAS]


def _variantes(w: str) -> set:
    v = {w, w + "s", w + "es"}
    if w.endswith("es"):
        v.add(w[:-2])
    if w.endswith("s"):
        v.add(w[:-1])
    return v


def _contiene(alimento, texto) -> bool:
    """Todas las palabras con contenido del alimento están en el texto, como palabras enteras y en singular o plural."""
    pal = _palabras(alimento)
    if not pal:
        return False
    en_texto = set(_palabras(texto))
    return all(_variantes(p) & en_texto for p in pal)


def nombrado(alimento, nombre_plato) -> bool:
    """¿El NOMBRE del plato nombra este alimento? («Leche evaporada» en «Avena cocida con leche evaporada y maní»)."""
    return _contiene(alimento, nombre_plato)


def plantilla(template_id) -> Optional[dict]:
    """La plantilla del registro, en la biblioteca que la tenga. `None` si no está."""
    if not template_id:
        return None
    try:
        import dish_registry as dr
    except Exception:                                                          # noqa: BLE001
        return None
    for cc in _PAISES:
        try:
            t = (dr.templates_by_id(cc) or {}).get(str(template_id))
        except Exception:                                                      # noqa: BLE001
            t = None
        if isinstance(t, dict):
            return t
    return None


def identidad(meal: dict, tpl: dict) -> list:
    """`[(nombre del constituyente, gramos de la plantilla)]` que dan identidad al plato: los que el nombre nombra y el más
    pesado de la plantilla."""
    cons = []
    for c in (tpl or {}).get("constituents") or []:
        nom = c.get("canonical") or c.get("name")
        try:
            g = float(c.get("grams") or 0)
        except (TypeError, ValueError):
            g = 0.0
        if nom and g >= GRAMOS_MIN_IDENTIDAD:
            cons.append((str(nom), g))
    if not cons:
        return []
    nombre = str((meal or {}).get("name") or "")
    out = [c for c in cons if nombrado(c[0], nombre)]
    pesado = max(cons, key=lambda c: c[1])
    if pesado not in out:
        out.append(pesado)
    return out


def _identidad_de(meal) -> list:
    if not enabled() or not isinstance(meal, dict) or meal.get("_recipe_source") != "library":
        return []
    tpl = plantilla(meal.get("_template_id") or meal.get("_recipe_template_id"))
    return identidad(meal, tpl) if tpl else []


def protege_linea(meal, linea) -> bool:
    """¿Esta línea es de un alimento que da identidad al plato? Los recortes de macros no la tocan. Plato de biblioteca:
    la plantilla lo dice. Plato del MODELO (sin plantilla): lo dice su NOMBRE (`nombrada_en_el_nombre`, P1-PLAN-LOTE-172)."""
    try:
        if isinstance(meal, dict) and meal.get("_recipe_source") == "library":
            return any(_contiene(nom, linea) for nom, _g in _identidad_de(meal))
        return llm_on() and nombrada_en_el_nombre(meal, linea)
    except Exception:                                                          # noqa: BLE001
        return False


# ─────────────── [P1-PLAN-LOTE-172 · 2026-09-23] la identidad de los platos del MODELO ───────────────
# Batería real del generador: «Maní tostado con pasas…» con 1,44 g de maní (el solver lo dejó en su cota inferior),
# «Panqueques de avena» con 15 g de avena y «Panqueques de trigo» con 10 g de harina. Este módulo sólo protegía los platos
# de biblioteca, porque sólo ellos traen plantilla; en los del modelo la identidad está escrita en el NOMBRE. La palabra
# con contenido del alimento (sin cantidades, unidades ni descriptores como «blanco» o «tostado») que aparece en el
# nombre lo marca. Knob `MEALFIT_DISH_IDENTITY_LLM` (True). tooltip-anchor: P1-PLAN-LOTE-172-IDENTIDAD-DEL-MODELO
_UNIDADES = frozenset({
    "g", "gr", "gramo", "gramos", "kg", "ml", "mililitros", "litro", "litros", "oz", "lb", "taza", "tazas", "cda", "cdas",
    "cdta", "cdtas", "cucharada", "cucharadas", "cucharadita", "cucharaditas", "unidad", "unidades", "lonja", "lonjas",
    "rebanada", "rebanadas", "pieza", "piezas", "torta", "tortas", "diente", "dientes", "lata", "latas", "pote",
    "potes", "paquete", "paquetes", "pizca", "puñado", "punado", "pedazo", "pedazos", "porcion", "porciones", "rodaja",
    "rodajas", "tira", "tiras", "hoja", "hojas", "ramita", "ramitas", "sobre", "sobres", "vaso", "vasos",
})
_DESCRIPTORES = frozenset({
    "blanco", "blanca", "blancos", "blancas", "verde", "verdes", "fresco", "fresca", "frescos", "frescas", "natural",
    "naturales", "integral", "integrales", "tostado", "tostada", "tostados", "tostadas", "cocido", "cocida", "cocidos",
    "cocidas", "crudo", "cruda", "rojo", "roja", "rojos", "rojas", "negro", "negra", "negros", "negras", "grande",
    "grandes", "mediano", "mediana", "medianos", "medianas", "pequeno", "pequena", "pequenos", "pequenas", "picado",
    "picada", "picados", "picadas", "rallado", "rallada", "entero", "entera", "enteros", "enteras", "light", "bajo",
    "baja", "grasa", "sin", "azucar", "sal", "molido", "molida", "seco", "seca", "secos", "secas", "maduro", "madura",
    "maduros", "maduras", "dulce", "dulces", "criollo", "criolla", "dominicano", "dominicana", "casero", "casera",
    "extra", "virgen", "polvo", "hojuelas", "trozos", "cubos", "griego", "griega", "descremada", "descremado",
    "desnatada", "semidescremada", "ligero", "ligera", "suave", "tierno", "tierna", "fina", "fino", "cruda",
})


def llm_on() -> bool:
    try:
        from knobs import _env_bool
        return enabled() and _env_bool("MEALFIT_DISH_IDENTITY_LLM", True)
    except Exception:                                                          # noqa: BLE001
        return enabled()


def _palabras_del_alimento(linea) -> list:
    toks = re.findall(r"[a-z]+", re.sub(r"\([^)]*\)", " ", _sa(linea)))
    while toks and (toks[0] in _UNIDADES or toks[0] in _VACIAS):
        toks.pop(0)
    return [t for t in toks if len(t) >= 3 and t not in _VACIAS and t not in _DESCRIPTORES and t not in _UNIDADES]


def nombrada_en_el_nombre(meal_o_nombre, linea) -> bool:
    """¿El nombre del plato nombra el alimento de esta línea? («1.44 g de maní tostado» en «Maní tostado con pasas»)."""
    if not enabled():
        return False
    nombre = meal_o_nombre.get("name") if isinstance(meal_o_nombre, dict) else meal_o_nombre
    en_nombre = set(_palabras(nombre))
    if not en_nombre:
        return False
    return any(_variantes(p) & en_nombre for p in _palabras_del_alimento(linea))


def _sustituidos(meal: dict) -> set:
    """Palabras de los alimentos que otro pase quitó a propósito (autofix de proteína, presupuesto): no vuelven."""
    out: set = set()
    pa = meal.get("_protein_autofix_applied")
    if isinstance(pa, str) and "->" in pa:
        out |= set(_palabras(pa.split("->", 1)[0]))
    for s in meal.get("_budget_substitutions") or []:
        out |= set(_palabras(str(s).split("→")[0]))
    return out


def _choca_alergia(alimento, alergias: Iterable) -> bool:
    pal: set = set()
    for p in _palabras(alimento):
        pal |= _variantes(p)
    return any(set(_palabras(a)) & pal for a in (alergias or ()) if str(a or "").strip())


def _alergeno_de_plantilla(tpl: dict, alergias: Iterable) -> bool:
    """Cinturón y tirantes: si la plantilla declara un alérgeno que el usuario declaró, el plato no se toca."""
    decl = {_sa(a) for a in ((tpl or {}).get("intrinsic_risk_attributes") or {}).get("allergens", []) if a}
    return bool(decl) and any(_sa(a) in decl for a in (alergias or ()) if str(a or "").strip())


def _canonico(nombre: str, index: dict) -> Optional[str]:
    """El nombre con el que el catálogo resuelve este alimento (el mismo resolutor que el contrato de la lista)."""
    try:
        from recipe_contract import _cantidades_lista
        claves = list(_cantidades_lista([f"100 g de {nombre}"], index))
        return str(claves[0][0]) if claves else None
    except Exception:                                                          # noqa: BLE001
        return None


def _remedir(meal: dict, db) -> None:
    if db is None:
        return
    try:
        from graph_orchestrator import _truth_up_meal_macros_from_strings as _tu
        _tu(meal, db)
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-46] re-medición tras restaurar la identidad no-op: {type(e).__name__}: {e}")


# ─────────────── [P1-PLAN-LOTE-49 · 2026-09-14] lo presente pero pobre, hasta el final ───────────────
# Quinta prueba RD del dueño (plan a059d7bb): «Guacamole criollo» con 5 g de aguacate (la plantilla pide 100 × 1,525),
# «Maní tostado con pasas» con 5 g de maní y el casabe con mantequilla de maní con 2,7 g. Los recortes de grasa lo dejaron
# en migajas y este módulo sólo devolvía lo que FALTABA: 5 g cuentan como «presente». Y el día 1 terminó al 71 % de su
# grasa y al 89 % de sus kcal: el recorte ni siquiera hacía falta. En la cola del guardado (después de TODOS los recortes)
# lo que quedó por debajo del piso sube al piso si el día tiene sitio; el sitio se mide (lote 46: subir sin medir llevaba la
# grasa al 116-122 %). Knob `MEALFIT_DISH_IDENTITY_RAISE` (True).
KCAL_TECHO = 1.05
GRASA_TECHO = 1.05


def subir_on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_DISH_IDENTITY_RAISE", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _num(v) -> float:
    m = re.search(r"-?\d+(?:[.,]\d+)?", str(v if v is not None else ""))
    return float(m.group(0).replace(",", ".")) if m else 0.0


def objetivos_de(plan_data) -> Optional[dict]:
    """`{"kcal", "grasa"}` del plan (`calories` y `macros.fats`); `None` si falta alguno."""
    if not isinstance(plan_data, dict):
        return None
    kcal, grasa = _num(plan_data.get("calories")), _num((plan_data.get("macros") or {}).get("fats"))
    return {"kcal": kcal, "grasa": grasa} if kcal > 0 and grasa > 0 else None


def _margen_del_dia(meals, objetivos) -> Optional[dict]:
    """Lo que al día le queda hasta su techo de kcal y de grasa; `None` sin objetivos o con el knob apagado."""
    if not objetivos or not subir_on():
        return None
    kcal = sum(_num(m.get("cals") or m.get("calories")) for m in meals if isinstance(m, dict))
    grasa = sum(_num(m.get("fats")) for m in meals if isinstance(m, dict))
    return {"kcal": float(objetivos["kcal"]) * KCAL_TECHO - kcal, "grasa": float(objetivos["grasa"]) * GRASA_TECHO - grasa}


def _lineas_de(lineas, canon, index) -> tuple:
    """`(índices, gramos)` de las líneas del alimento `canon`, por alimento (el resolutor del contrato de la lista)."""
    from recipe_contract import _cantidades_lista
    idx, gramos = [], 0.0
    for i, s in enumerate(lineas or []):
        if not isinstance(s, str):
            continue
        c = _cantidades_lista([s], index)
        if any(k[0] == canon for k in c):
            idx.append(i)
            gramos += sum(v for k, v in c.items() if k[0] == canon and k[1] == "g")
    return idx, gramos


def _subir_linea(meal, canon, piso, index, db, margen) -> Optional[str]:
    """El alimento está, pero por debajo de su piso: sube AL piso, en la lista y en la compra, si al día le cabe."""
    if db is None or margen is None:
        return None
    ings, raw = meal.get("ingredients"), meal.get("ingredients_raw")
    i_d, _g = _lineas_de(ings, canon, index)
    i_r, _g = _lineas_de(raw, canon, index) if isinstance(raw, list) else ([], 0.0)
    if len(i_d) != 1 or len(i_r) > 1:
        return None                      # dos líneas del mismo alimento: no se adivina cuál sube
    # Los gramos, con el lector de la base (el de la lista del contrato leía «57.2 g» como 2 g y «subía» la soya a 18).
    g_cur = 0.0
    for linea_act in ([raw[i_r[0]]] if i_r else []) + [ings[i_d[0]]]:
        try:
            g_cur = float(db.grams_from_ingredient_string(str(linea_act)) or 0)
        except Exception:                                                      # noqa: BLE001
            g_cur = 0.0
        if g_cur > 0:
            break
    if g_cur <= 0 or g_cur >= piso:
        return None                      # sin gramos legibles no se toca; y nunca se baja
    mac = db.macros_from_ingredient_string(f"{piso - g_cur:.0f} g de {canon}") or {}
    dk, dg = float(mac.get("kcal") or 0), float(mac.get("fats") or 0)
    if dk <= 0 or dk > margen["kcal"] or dg > margen["grasa"]:
        logger.info(f"🧩 [P1-PLAN-LOTE-49] «{str(meal.get('name'))[:40]}»: {canon} en {g_cur:.0f} g (piso {piso}) y el día "
                    f"no tiene sitio (+{dk:.0f} kcal / +{dg:.1f} g de grasa; quedan {margen['kcal']:.0f} y {margen['grasa']:.1f})")
        return None
    linea = f"{piso} g de {canon}"
    # Por ALIMENTO, nunca por índice (la familia `raw[idx]`): se sustituye la línea de `canon` —ya se comprobó que es una.
    from recipe_contract import _cantidades_lista

    def _es_de(s) -> bool:
        return isinstance(s, str) and any(k[0] == canon for k in _cantidades_lista([s], index))
    vieja = next(x for x in ings if _es_de(x))
    ings.insert(ings.index(vieja), linea)
    ings.remove(vieja)
    if i_r:
        meal["ingredients_raw"] = [linea if _es_de(r) else r for r in raw]
    elif isinstance(raw, list):
        raw.append(linea)
    margen["kcal"] -= dk
    margen["grasa"] -= dg
    return f"↑{g_cur:.0f}→{piso} g de {canon}"


# ─────────────── [P1-PLAN-LOTE-174 · 2026-09-23] lo pobre de los platos del MODELO, también hasta el final ───────────────
# Batería real (DM2 + insulina): «Casabe crujiente con queso blanco fresco, maní y huevo» con 0,01 g de maní, «Mandarina
# con almendras…» con 0 g de almendras, «…y sardinas en lata» con 1,97 g de sardinas: el día 3 entregó 1.450 kcal y 110 g
# de proteína (meta 1.750/153). Los platos del modelo no tienen plantilla, así que su piso sale del TIPO de alimento; la
# identidad, de su nombre (`nombrada_en_el_nombre`). Se sube sólo lo PRESENTE y sólo si al día le cabe (el mismo margen de
# kcal y grasa del lote 49). Knob `MEALFIT_DISH_IDENTITY_LLM` (el de la identidad del modelo, lote 172).
# tooltip-anchor: P1-PLAN-LOTE-174-PISO-DEL-MODELO
_PISO_POR_PALABRA = (
    ("chia", 5), ("linaza", 5), ("ajonjoli", 5), ("semilla", 5),
    ("mani", 10), ("almendra", 10), ("nuez", 10), ("nueces", 10), ("pistacho", 10), ("maranon", 10),
    ("mantequilla de mani", 10),
    ("avena", 30), ("quinoa", 30), ("harina", 25), ("casabe", 20), ("pan", 30), ("arroz", 40), ("pasta", 40),
    ("espagueti", 40), ("bulgur", 30), ("cebada", 30),
    ("queso", 20), ("ricotta", 30), ("cottage", 40), ("yogur", 80), ("leche", 100),
)
_PISO_POR_CATEGORIA = {"proteinas": 60, "viveres": 60, "frutas": 60, "vegetales": 30, "lacteos": 20}


# Hierbas, aromáticos y condimentos dan nombre («al cilantro», «al ajillo») pero no piden ración: sin piso.
_SIN_PISO = re.compile(r"\b(cilantro|culantro|perejil|oregano|albahaca|hierbabuena|menta|cebollin|puerro|ajo|ajillo|"
                       r"jengibre|canela|comino|pimienta|pimenton|sal|limon|lima|vinagre|aceite|mostaza|salsa|vainilla|"
                       r"laurel|tomillo|romero|curcuma|curry|adobo|sazon|agua|hielo|cafe|te)\b")


def _piso_de(canon: str, db) -> int:
    n = _sa(canon)
    # La CABEZA del alimento decide si es hierba/condimento: «almendras tostadas sin sal» no es sal.
    _cab = (_palabras_del_alimento(canon) or [""])[0]
    if _SIN_PISO.search(_cab):
        return 0
    for w, g in _PISO_POR_PALABRA:
        if re.search(rf"\b{w}", n):
            return g
    try:
        cat = _sa(db.category_of(canon) or "")
    except Exception:                                                          # noqa: BLE001
        cat = ""
    return int(_PISO_POR_CATEGORIA.get(cat, 0))


# «0 g de almendras…»: el lector de la lista no la resuelve (sin gramos no hay alimento), así que sube por su propio camino.
_CERO = re.compile(r"^\s*0+(?:[.,]0+)?\s*(?:g|gr|gramos)\s+de\s+(.+)$", re.IGNORECASE)


def _rescatar_cero(meal: dict, alimento: str, db, margen, allergies) -> Optional[str]:
    alimento = alimento.strip()
    if _choca_alergia(alimento, allergies):
        return None
    piso = _piso_de(alimento, db)
    if not piso:
        return None
    nueva = f"{piso} g de {alimento}"
    mac = db.macros_from_ingredient_string(nueva) or {}
    dk, dg = float(mac.get("kcal") or 0), float(mac.get("fats") or 0)
    if dk <= 0 or dk > margen["kcal"] or dg > margen["grasa"]:
        return None
    for campo in ("ingredients", "ingredients_raw"):
        ls = meal.get(campo)
        if isinstance(ls, list):
            meal[campo] = [nueva if (isinstance(x, str) and _CERO.match(x)
                                     and _sa(_CERO.match(x).group(1)).strip() == _sa(alimento)) else x for x in ls]
    margen["kcal"] -= dk
    margen["grasa"] -= dg
    return f"↑0→{piso} g de {alimento}"


def _es_proteico(alimento: str, db) -> bool:
    """≥10 g de proteína por 100 g: la fase que va primero cuando el margen del día es corto."""
    try:
        info = db.lookup(alimento)
        return bool(info) and float(getattr(info, "protein", 0) or 0) >= 10.0
    except Exception:                                                          # noqa: BLE001
        return False


def _subir_identidad_del_modelo(meal: dict, index: dict, *, db=None, allergies=None, margen=None, fase=None) -> list:
    if not llm_on() or margen is None or db is None:
        return []
    from recipe_contract import _cantidades_lista
    hechos = []
    for linea in list(meal.get("ingredients") or []):
        if not isinstance(linea, str) or not nombrada_en_el_nombre(meal, linea):
            continue
        _m0 = _CERO.match(linea)
        if _m0:
            if fase is not None and _es_proteico(_m0.group(1), db) != (fase == "proteina"):
                continue
            sub = _rescatar_cero(meal, _m0.group(1), db, margen, allergies)
            if sub:
                hechos.append(sub)
            continue
        claves = list(_cantidades_lista([linea], index))
        if len(claves) != 1:
            continue
        canon = str(claves[0][0])
        if _choca_alergia(canon, allergies):
            continue
        if fase is not None and _es_proteico(canon, db) != (fase == "proteina"):
            continue
        piso = _piso_de(canon, db)
        if not piso:
            continue
        sub = _subir_linea(meal, canon, piso, index, db, margen)
        if sub:
            hechos.append(sub)
    if hechos:
        meal["_identidad_restaurada"] = list(meal.get("_identidad_restaurada") or []) + hechos
        meal.pop("_display", None)
        _remedir(meal, db)
    return hechos


def restaurar_meal(meal: dict, index: dict, *, db=None, allergies=None, margen=None, fase=None) -> list:
    """Un plato. Devuelve lo que añadió o subió (`["+38 g de Aguacate"]`, `["↑5→38 g de Aguacate"]`); `[]` si no tocó
    nada. Sin `margen` sólo vuelve lo que FALTA (lote 46); con `margen` (lo que al día le queda hasta su techo de kcal y de
    grasa, `_margen_del_dia`) sube además lo presente por debajo del piso, si cabe. [P1-PLAN-LOTE-174] Un plato del
    modelo (sin plantilla) sube lo que su nombre nombra hasta el piso de su tipo de alimento."""
    if not isinstance(meal, dict) or meal.get("_sodium_autofix_applied"):
        return []
    if meal.get("_recipe_source") != "library":
        return _subir_identidad_del_modelo(meal, index, db=db, allergies=allergies, margen=margen, fase=fase)
    if fase == "proteina":
        return []            # la biblioteca va en la segunda pasada, como siempre
    tpl = plantilla(meal.get("_template_id") or meal.get("_recipe_template_id"))
    ings = meal.get("ingredients")
    if not tpl or not isinstance(ings, list) or _alergeno_de_plantilla(tpl, allergies):
        return []
    from recipe_contract import _cantidades_lista
    subs = _sustituidos(meal)
    try:
        factor = float(meal.get("_scale_factor") or 1.0)
    except (TypeError, ValueError):
        factor = 1.0
    presentes = {k[0] for k in _cantidades_lista(ings, index)}
    hechos = []
    for nom, g_tpl in identidad(meal, tpl):
        if set(_palabras(nom)) & subs or _choca_alergia(nom, allergies):
            continue
        canon = _canonico(nom, index)
        if not canon:
            continue            # sin resolver en el catálogo no se inventa
        piso = max(PISO_MIN_G, int(round(PISO_FRACCION * g_tpl * factor)))
        if canon in presentes:
            sub = _subir_linea(meal, canon, piso, index, db, margen) if margen is not None else None
            if sub:
                hechos.append(sub)
            continue            # presente: sin margen del día (o sin sitio en él) no se toca
        linea = f"{piso} g de {canon}"
        ings.append(linea)
        raw = meal.get("ingredients_raw")
        if isinstance(raw, list):
            raw.append(linea)
        presentes.add(canon)
        hechos.append(f"+{linea}")
        if margen is not None and db is not None:
            mac = db.macros_from_ingredient_string(linea) or {}
            margen["kcal"] -= float(mac.get("kcal") or 0)
            margen["grasa"] -= float(mac.get("fats") or 0)
    if hechos:
        meal["_identidad_restaurada"] = hechos
        meal.pop("_display", None)        # la capa de traducción espeja `ingredients` por índice: se regenera
        _remedir(meal, db)
    return hechos


def restaurar_identidad(days, *, db=None, index=None, allergies=None, objetivos=None) -> int:
    """Todos los días. Devuelve cuántos platos tocó. Fail-open: sin catálogo no hace nada. Con `objetivos` (`objetivos_de`
    del plan: la cola del guardado, que corre después de todos los recortes) sube también lo presente por debajo del piso
    cuando el día tiene sitio. tooltip-anchor: P1-PLAN-LOTE-49-IDENTIDAD-HASTA-EL-FINAL"""
    if not enabled() or not isinstance(days, list):
        return 0
    if index is None:
        try:
            from recipe_contract import _index_default
            index = _index_default(db)
        except Exception:                                                      # noqa: BLE001
            index = {}
    if not index:
        return 0
    tocados = 0
    for d in days:
        meals = [m for m in ((d.get("meals") or []) if isinstance(d, dict) else []) if isinstance(m, dict)]
        margen = _margen_del_dia(meals, objetivos)
        # [P1-PLAN-LOTE-174] con margen, dos pasadas: primero lo PROTEICO (el déficit que el revisor rechaza), después
        # el resto; sin margen (lote 46) una sola, como siempre.
        for fase in (("proteina", "resto") if margen is not None else (None,)):
            for m in meals:
                try:
                    hechos = restaurar_meal(m, index, db=db, allergies=allergies, margen=margen, fase=fase)
                except Exception as e:                                         # noqa: BLE001
                    logger.debug(f"[P1-PLAN-LOTE-46] identidad no-op en {str((m or {}).get('name'))[:40]}: {e!r}")
                    hechos = []
                if hechos:
                    tocados += 1
                    logger.info(f"🧩 [P1-PLAN-LOTE-46] identidad del plato restaurada en «{str(m.get('name'))[:48]}»: "
                                f"{', '.join(hechos)}")
    return tocados
