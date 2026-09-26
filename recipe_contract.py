# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-23 · 2026-09-12] (C2 · CUL-P0-03) El contrato sobre la receta FINAL: la última palabra la tiene la lista.

Qué defecto cierra. Los pasos de una receta nombran cantidades («casca 6 huevos», «1 diente de ajo», «mide 90 g de
maní») y la lista de ingredientes es la que compra, cuenta macros y llena la Nevera. Entre generar el plato y
persistirlo, la lista la MUTAN media docena de reparadores (caps de huevo y de porciones absurdas, pisos de porción
subservible, el motor de macros que reescala, sustituciones por alergia, el reconciliador display↔raw…). El repo ya
tenía sincronizadores de pasos (`_sync_recipe_step_quantities`, `_egg_count_step_sync`, `_rewrite_recipe_steps_after_subs`),
pero corren en puntos fijos: **una reparación que corre DESPUÉS del sincronizador deja lista y pasos desincronizados**.

Medido en el corpus fijo del 09-12 (5 planes, 64 comidas): 47 comidas con V7a (la lista compra más piezas de las que
los pasos usan), 38 con V7e (un paso pide más piezas de las que la lista compra), 11 V6 y 4 V4 — 100 de 111 hallazgos
de capa 1 son cantidades. Re-ejecutar el sincronizador existente sobre el corpus sólo bajaba V7e de 38 a 35: no lee
piezas desnudas ni alimentos de menos de 4 letras («ajo», «pan»), y sí corrigió «casca 6 huevos» → 3, lo que prueba
que al persistir NO corrió después del cap. De las 68 comidas que el dueño marcó con defecto en el golden set, 54 lo
describen en su nota: «no coinciden las cantidades entre ingredientes y preparación».

Qué hace `reconcile_step_quantities`: lee la lista con los MISMOS parsers del contrato determinista
(`culinary_coherence`: gramos V4, unidades contables V6, piezas desnudas V7) y reescribe en los pasos la cantidad que
contradice a la lista, familia por familia (gramos↔gramos, «diente»↔«diente», pieza↔pieza; nunca cruza familias:
eso es V7b, sin densidad no hay conversión). Reglas, por orden:

  · la lista es la autoridad (nutrición y compra salen de ella); los pasos la siguen;
  · una sola mención numérica del alimento en los pasos ⇒ se alinea en las DOS direcciones;
  · dos o más menciones ⇒ sólo se RECORTA la que pide más de lo comprado (un reparto entre pasos no se adivina:
    subir una mención duplicaría el ingrediente);
  · tolerancias de las capas que miden: 25 % en gramos (V4), 5 % o 0,06 en conteos (V6/V7e) — lo que la medición
    tolera, el reparador no lo toca;
  · piezas desnudas que cruzan el singular/plural («1 tomate» ↔ 2,5) NO se reescriben cuando el paso pide MENOS de lo
    comprado: cambiar el número sin cambiar el sustantivo produce «2,5 tomate»; se informa como `gramatical`, y V7a
    lo sigue viendo. [P1-PLAN-LOTE-31] Cuando el paso pide MÁS de lo comprado («los 2 plátanos verdes» con «½ plátano
    verde» en la lista) SÍ se reescribe, con concordancia: número, artículo, sustantivo y adjetivos («el ½ plátano
    verde»), y si la lista compra exactamente 1, el número sobra («divide las 2 ciruelas» → «divide la ciruela»);
    lo que no sabe concordar sin dejar un residuo peor («1½ tostones de casabe») sigue en `gramatical`;
  · rangos («1–2 mandarinas»), aproximaciones («≈ 30 g»), notas ⚠/💡 y notas de procedencia («se reemplazó…») quedan
    fuera; `ingredients_raw` no se toca (sólo `recipe`).

Idempotente: la segunda pasada no cambia nada (test). Fail-open: jamás lanza; ante duda, no toca. PURO: el catálogo
entra como índice (`build_culinary_index`), sin DB ni env. El caller (los dos finalizadores del persist boundary) decide
el knob y el orden: ÚLTIMO, después de todo lo que muta la lista. tooltip-anchor: P1-PLAN-LOTE-23-RECIPE-CONTRACT

[P1-PLAN-LOTE-24 · 2026-09-12] (C3 · CUL-P0-04) Dos cosas más. (1) El contrato conoce las TRES FORMAS del huevo
(`reconcile_meal`: la lista nombra la forma, los pasos la siguen, y sólo después las cantidades) — ver la sección
«Las tres formas del huevo». (2) «ÚLTIMO» se midió y no lo era: dentro de `finalize_plan_data_coherence` quedaban
detrás el band-closer, los caps de realismo, el tope diario de huevos enteros, el piso de proteína y el re-cuadre de
conteos (`db_plans._finalize_plan_data_for_insert`), y en swap y chat-modify el re-cuadre del día y el motor de macros.
El contrato corre ahora también en la COLA de esos tres chokepoints (`P1-PLAN-LOTE-24-FINAL-CONTRACT-TAIL`); los ganchos
de los finalizadores se conservan (idempotente: la segunda pasada no cambia nada).
"""
from __future__ import annotations

import logging
import re
from collections import Counter

from constants import strip_accents
from culinary_coherence import (
    V4_TOLERANCIA, _V4_APPROX_LEAD_RE, _V4_GRAMS_RE, _V6_MASA_RE, _V6_RE, _V7_MEDIDA_RE, _V7_PIEZA_RE,
    _V7_RANGO_RE, _catalog_food_spans, _norm, _v4_grams_by_food, _v6_cuentas, _v6_valor, _v7_piezas,
    build_culinary_index, clause_bounds, find_catalog_foods, grams_owner,
)

logger = logging.getLogger(__name__)

#: Unidades contables (singular ↔ plural) que V6 reconoce; el reconciliador concuerda el número gramatical al reescribir.
UNIDAD_SINGULAR = {
    "unidades": "unidad", "dientes": "diente", "rebanadas": "rebanada", "lonjas": "lonja", "tazas": "taza",
    "cucharadas": "cucharada", "cucharaditas": "cucharadita", "cdas": "cda", "cdtas": "cdta", "gajos": "gajo",
    "ramitas": "ramita", "filetes": "filete", "pedazos": "pedazo", "hojas": "hoja", "tallos": "tallo",
    "latas": "lata", "paquetes": "paquete",
}
UNIDAD_PLURAL = {v: k for k, v in UNIDAD_SINGULAR.items()}
_FRACCIONES = ((0.125, "⅛"), (0.25, "¼"), (1 / 3, "⅓"), (0.5, "½"), (2 / 3, "⅔"), (0.75, "¾"))
_YOGURT_LEN_RE = re.compile(r"\byogurt(s?)\b")
_NOTA_PROCEDENCIA = ("se reemplaz", "sustituy", "seguridad alimentaria")


def _es_nota(paso: str) -> bool:
    """Notas deterministas (⚠/💡) y de procedencia («se reemplazó X por Y» cita el alimento ORIGINAL a propósito)."""
    s = str(paso or "")
    if "⚠" in s or "💡" in s or "🌱" in s:   # [P1-PLAN-LOTE-24] la nota del nutricionista (🌱) tampoco es un paso
        return True
    low = strip_accents(s.lower())
    return any(t in low for t in _NOTA_PROCEDENCIA)


def _mapa_posiciones(paso: str) -> "list | None":
    """`mapa[pos_norm] = pos_original`. Se detecta sobre `_norm(paso)` —el MISMO texto que leen V4/V6/V7, para que el
    reparador atribuya igual que el medidor— y se reescribe el original. `_norm` conserva la longitud (minúsculas, sin
    acentos) salvo por la «t» de «yogurt», que quita: el mapa salta esas posiciones. `None` si la premisa de longitud
    no se cumple (entonces no se toca nada)."""
    base = strip_accents(str(paso or "").lower())
    if len(base) != len(paso):
        return None
    quitadas = {m.start() + 5 for m in _YOGURT_LEN_RE.finditer(base)}
    mapa = [i for i in range(len(base)) if i not in quitadas]
    mapa.append(len(base))
    return mapa


#: A qué alimento pertenece un «N g»: por GRAMÁTICA, compartido con V4 (`culinary_coherence.grams_owner`) para que el
#: reparador atribuya EXACTAMENTE igual que el medidor — la cercanía daba los 265 g del tomate al nabo.
_atribuir_gramos = grams_owner


def _unidad_norm(u: str) -> str:
    u = strip_accents(str(u or "").lower())
    return UNIDAD_SINGULAR.get(u, u)


def formatear_cantidad(v: float) -> str:
    """«3», «½», «1½», «2.5» — como escriben los pasos del repo (fracciones unicode cuando la cifra es una de cocina)."""
    if v is None:
        return ""
    if abs(v - round(v)) < 0.01:
        return str(int(round(v)))
    entero = int(v)
    resto = v - entero
    for val, simbolo in _FRACCIONES:
        if abs(resto - val) < 0.02:
            return f"{entero}{simbolo}" if entero else simbolo
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _tolera(familia: str, paso_val: float, lista_val: float) -> bool:
    """Lo que la medición tolera, el reparador no lo toca."""
    if familia == "g":
        return abs(paso_val - lista_val) <= V4_TOLERANCIA * max(lista_val, 1e-9)
    return abs(paso_val - lista_val) <= max(0.06, 0.05 * lista_val)


def _cantidades_lista(ings: list, index: dict) -> dict:
    """{(alimento, familia): total} de la lista. familia ∈ {"g", "pieza", ("u", unidad)}. Los gramos y las piezas se
    SUMAN entre líneas (dos líneas del mismo alimento compran la suma); las unidades contables también."""
    out: dict = {}
    for ing in ings:
        s = str(ing)
        for food, g in _v4_grams_by_food(_norm(s), index).items():
            out[(food, "g")] = out.get((food, "g"), 0.0) + g
        for food, pares in _v6_cuentas(s, index).items():
            for uni, val in pares:
                k = (food, ("u", _unidad_norm(uni)))
                out[k] = out.get(k, 0.0) + val
        for food, n in _v7_piezas(s, index).items():
            out[(food, "pieza")] = out.get((food, "pieza"), 0.0) + n
    return out


def _menciones_paso(paso: str, index: dict) -> list:
    """Las menciones numéricas de un paso, con sus posiciones en el paso ORIGINAL:
    {"familia", "food", "valor", "ini", "fin", "unidad", "u_ini", "u_fin", "food_ini", "food_fin"}. Se detecta sobre
    `_norm(paso)` (lo que leen V4/V6/V7) y se traducen las posiciones con `_mapa_posiciones`. [P1-PLAN-LOTE-31]
    `food_ini`/`food_fin` acotan el NOMBRE del alimento en el paso (None si no se pudo situar): la concordancia
    singular/plural y el colapso de repeticiones necesitan saber dónde termina la mención, no sólo dónde empieza."""
    mapa = _mapa_posiciones(paso)
    if mapa is None:
        return []
    blob = _norm(paso)
    if len(blob) + 1 != len(mapa):
        return []
    menciones = []
    ocupados = []

    def _libre(a, b):
        return not any(a < e and s < b for s, e in ocupados)

    def _orig(a, b):
        return mapa[a], mapa[b]

    # (b) unidades contables — antes que las piezas, para que «2 dientes de ajo» no cuente como 2 piezas
    for m in _V6_RE.finditer(blob):
        val = _v6_valor(m.group(1))
        unidad = m.group(2) or ""
        if val is None or _V6_MASA_RE.fullmatch(unidad):
            continue
        crudos = list(find_catalog_foods(m.group(3), index))
        normas = {f: _norm(f) for f in crudos}
        crudos = [f for f in crudos if not any(f != o and normas[f] in normas[o] for o in crudos)]
        if len(crudos) != 1 or not _libre(m.start(1), m.end(2)):
            continue
        a, b = _orig(m.start(1), m.end(1))
        ua, ub = _orig(m.start(2), m.end(2))
        fa, fb = _span_alimento(m.group(3), crudos[0], index, m.start(3), _orig)
        menciones.append({"familia": ("u", _unidad_norm(unidad)), "food": crudos[0], "valor": val,
                          "ini": a, "fin": b, "unidad": paso[ua:ub], "u_ini": ua, "u_fin": ub, "food_ini": fa, "food_fin": fb})
        ocupados.append((m.start(1), m.end(2)))
    # (a) gramos, por cláusula y por GRAMÁTICA (el alimento que sigue al «N g de», o el que lo precede pegado)
    for c_ini, c_fin in clause_bounds(blob):
        clause = blob[c_ini:c_fin]
        foods = _catalog_food_spans(clause, index)
        if not foods:
            continue
        for m in _V4_GRAMS_RE.finditer(clause):
            if _V4_APPROX_LEAD_RE.search(clause[:m.start()]):
                continue
            food = _atribuir_gramos(clause, m.start(), m.end(), foods)
            if food is None or not _libre(c_ini + m.start(1), c_ini + m.end(1)):
                continue
            a, b = _orig(c_ini + m.start(1), c_ini + m.end(1))
            fs = next(((s, e) for s, e, n in foods if n == food), None)
            fa, fb = _orig(c_ini + fs[0], c_ini + fs[1]) if fs else (None, None)
            menciones.append({"familia": "g", "food": food, "valor": float(m.group(1).replace(",", ".")),
                              "ini": a, "fin": b, "unidad": None, "u_ini": None, "u_fin": None, "food_ini": fa, "food_fin": fb})
            ocupados.append((c_ini + m.start(1), c_ini + m.end(1)))
    # (c) piezas desnudas — las mismas guardas que `_v7_piezas`
    for m in _V7_PIEZA_RE.finditer(blob):
        val = _v6_valor(m.group(1))
        cola = m.group(2) or ""
        if val is None or _V7_RANGO_RE.search(blob[:m.start()]):
            continue
        medida = _V7_MEDIDA_RE.search(cola)
        crudos = list(find_catalog_foods(cola, index))
        if len(crudos) != 1:
            continue
        if medida:
            spans = _catalog_food_spans(cola, index)
            if not spans or medida.start() < spans[0][0]:
                continue
        if not _libre(m.start(1), m.end(1)):
            continue
        a, b = _orig(m.start(1), m.end(1))
        fa, fb = _span_alimento(cola, crudos[0], index, m.start(2), _orig)
        menciones.append({"familia": "pieza", "food": crudos[0], "valor": val, "ini": a, "fin": b,
                          "unidad": None, "u_ini": None, "u_fin": None, "food_ini": fa, "food_fin": fb})
        ocupados.append((m.start(1), m.end(1)))
    return menciones


def _span_alimento(texto_norm: str, food: str, index: dict, desplaz: int, _orig) -> tuple:
    """[P1-PLAN-LOTE-31] (ini, fin) en el paso ORIGINAL del nombre de `food` dentro de `texto_norm` (un trozo del blob
    normalizado que empieza en `desplaz`). (None, None) si el alimento no se sitúa."""
    try:
        sp = next(((s, e) for s, e, n in _catalog_food_spans(texto_norm, index) if n == food), None)
        return _orig(desplaz + sp[0], desplaz + sp[1]) if sp else (None, None)
    except Exception:
        return (None, None)


# ─────────────────────────────────────────────────────────
# [P1-PLAN-LOTE-31 · 2026-09-13] Concordancia singular/plural cuando el paso pide MÁS piezas que la lista.
#
# El contrato de C2 dejaba en `gramatical` toda pieza desnuda cuyo número cruzaba el 1 («2 plátanos» ↔ ½): cambiar sólo
# el número produce «½ plátanos verdes». Medido en el bench real (3 planes recién generados): 7 de los 10 V7e «de fábrica»
# eran exactamente eso — el motor de macros o un cerrador bajó la lista de 2 a ½ (o a 1) y el paso siguió diciendo «pela
# los 2 plátanos verdes», «divide las 2 ciruelas», «calienta las 2 tortillas integrales», «pica los 2 cebollines». En esa
# dirección seguir el paso rompe la nutrición y la Nevera (se usa lo que no se compró), así que se reescribe CON
# concordancia: el número, el artículo, el sustantivo (token a token contra la clave del índice: «plátanos verdes» →
# «plátano verde», «cebollines» → «cebollín», sin inventar acentos), los adjetivos que le siguen («ciruelas frescas» →
# «ciruela fresca») y, si la cláusula sólo habla de ese alimento, los clíticos («córtalos» → «córtalo», sin «cada uno»).
# Si la lista compra exactamente 1 y hay artículo, el número sobra: «divide las 2 ciruelas» → «divide la ciruela».
# La dirección contraria (el paso pide MENOS: «pica 1 tomate» con 3 en la lista, 21 menciones en el corpus) sigue en
# `gramatical`: es la decisión V7a que el dueño tiene en su hoja. Lo que no sabe concordar sin dejar un residuo peor
# («1½ tostones de casabe»: el singular pide un acento que el texto no trae) tampoco se toca y se declara.
# tooltip-anchor: P1-PLAN-LOTE-31-CONCORDANCIA

#: plurales que se vuelven singular RECORTANDO letras (sin inventar acentos): {plural normalizado: letras que sobran}.
#: Sólo adjetivos de receta y «tostadas» (la pieza de casabe/pan). Sustantivos de forma («rodajas», «trozos») NO: «2
#: rodajas de tomate» cuenta rodajas, no tomates, y reescribirlas cambiaría la receta.
_PLURAL_RECORTE = {
    **{w: 1 for w in (
        "verdes", "maduros", "maduras", "frescos", "frescas", "medianos", "medianas", "grandes", "pequenos", "pequenas",
        "enteros", "enteras", "crudos", "crudas", "cocidos", "cocidas", "picados", "picadas", "cortados", "cortadas",
        "pelados", "peladas", "rallados", "ralladas", "tostados", "tostadas", "hervidos", "hervidas", "duros", "duras",
        "firmes", "tiernos", "tiernas", "finos", "finas", "gruesos", "gruesas", "sancochados", "sancochadas", "fritos",
        "fritas", "secos", "secas", "rojos", "rojas", "amarillos", "amarillas", "blancos", "blancas", "negros", "negras",
        "morados", "moradas", "dulces", "criollos", "criollas", "dominicanos", "dominicanas", "limpios", "limpias",
        "lavados", "lavadas", "troceados", "troceadas", "desmenuzados", "desmenuzadas", "asados", "asadas", "horneados",
        "horneadas", "reservados", "reservadas", "restantes", "descongelados", "descongeladas", "escurridos", "escurridas",
        "listos", "listas", "calientes", "frios", "frias", "tibios", "tibias", "suaves", "abiertos", "abiertas", "partidos",
        "partidas", "batidos", "batidas", "escalfados", "escalfadas", "revueltos", "revueltas", "molidos", "molidas",
        "machacados", "machacadas", "majados", "majadas", "dorados", "doradas", "crujientes", "jugosos", "jugosas",
        "aplastados", "aplastadas", "hidratados", "hidratadas", "remojados", "remojadas", "tostaditas", "tostaditos",
        "maduritos", "maduritas", "pequenitos", "pequenitas", "medianitos", "medianitas", "picaditos", "picaditas",
    )},
    **{w: 2 for w in ("integrales", "naturales", "azules", "especiales", "tradicionales", "artesanales", "vegetales",
                      "normales", "adicionales", "individuales", "principales", "comunes", "jovenes")},
}
_ART_ANTES_RE = re.compile(r"\b(el|la|los|las|unos|unas)\s+$", re.IGNORECASE)
#: imperativo + clítico plural con tilde («córtalos», «resérvalas», «escúrrelas»); «ponlos» no lleva tilde y no se toca
_CLITICO_PLURAL_RE = re.compile(r"(?<![\wáéíóúñü])([\wáéíóúñü]*[áéíóú][\wáéíóúñü]*[aeiou]l[oa])s\b")
_CADA_UNO_RE = re.compile(r"\s+cada\s+un[oa]\b", re.IGNORECASE)
_PALABRA_TRAS_RE = re.compile(r"(\s+)([\wáéíóúñü]+)")


def _plural_a_singular(palabra: str) -> "str | None":
    n = _PLURAL_RECORTE.get(_norm(palabra))
    return palabra[:-n] if n else None


def _singular_span(span: str, index: dict, food: str) -> "str | None":
    """«plátanos verdes» → «plátano verde», «cebollines» → «cebollín»: token a token contra el NOMBRE CANÓNICO del catálogo
    (`food`, con sus acentos: el plural «cebollines» no lleva tilde y el singular sí — recortar letras la perdía). Si el span es
    un alias que no casa con el canónico, se recorta «s»/«es» sobre el texto; un alias que ya es plural («plátanos verdes» está
    en el índice tal cual) no dice cuál es su singular y devuelve None. Conserva mayúsculas del texto."""
    sn = _norm(span)
    toks = span.split()
    canon = _norm(food)
    ttoks = None
    if canon in index and index[canon]["rx"].fullmatch(sn):
        ktoks, ttoks = canon.split(), str(index[canon]["name"]).split()
        if len(ttoks) != len(ktoks):
            ttoks = None
    else:
        clave = next((k for k in sorted(index, key=len, reverse=True) if index[k]["rx"].fullmatch(sn)), None)
        if clave is None:
            return None
        ktoks = clave.split()
    if len(toks) != len(ktoks):
        return None
    out = []
    for j, (t, k) in enumerate(zip(toks, ktoks)):
        tn = _norm(t)
        if tn == k:
            if k.endswith("s") and ttoks is None:
                return None                                  # alias en plural: su singular no está escrito en ningún sitio
            out.append(t)
        elif tn in (k + "s", k + "es"):
            if ttoks is not None:
                s = ttoks[j].lower()
                out.append(s[:1].upper() + s[1:] if t[:1].isupper() else s)
            else:
                out.append(t[:-1] if tn == k + "s" else t[:-2])
        elif k.endswith("s") and tn in (k[:-1], k[:-2]):
            out.append(t)                                    # la clave es plural y el texto ya viene en singular
        else:
            return None
    return " ".join(out)


def _singulariza_adjetivos_tras(texto: str, pos: int, maximo: int = 2) -> str:
    """Hasta `maximo` palabras seguidas a partir de `pos` que sean plurales del léxico pasan al singular."""
    for _ in range(maximo):
        m = _PALABRA_TRAS_RE.match(texto, pos)
        if not m:
            break
        s = _plural_a_singular(m.group(2))
        if s is None:
            break
        texto = texto[:m.start(2)] + s + texto[m.end(2):]
        pos = m.start(2) + len(s)
    return texto


def _concordar_pieza(paso: str, m: dict, objetivo: float, index: dict) -> "str | None":
    """«pela los 2 plátanos verdes y córtalos en 4 trozos cada uno» con «½ plátano verde» en la lista →
    «pela el ½ plátano verde y córtalo en 4 trozos». Sólo hacia el singular (objetivo ≤ 1). None cuando no sabe hacerlo
    sin dejar un residuo peor: entonces el llamador lo declara `gramatical` y no toca nada."""
    fi, ff = m.get("food_ini"), m.get("food_fin")
    if fi is None or ff is None or objetivo > 1.0 or fi < m["fin"]:
        return None
    entre = []
    for w in paso[m["fin"]:fi].split():
        if _norm(w) in ("de", "del") and entre:
            entre.append(w)                              # «2 tostadas DE casabe»: el «de» sigue a la forma
            continue
        s = _plural_a_singular(w)
        if s is None:
            return None                                  # «1½ tostones de casabe»: no se inventa el acento
        entre.append(s)
    sing = _singular_span(paso[fi:ff], index, m["food"])
    if sing is None:
        return None
    prefijo = paso[:m["ini"]]
    art = _ART_ANTES_RE.search(prefijo)
    uno = abs(objetivo - 1.0) < 0.01
    cabeza = ""
    if art:
        a = art.group(1)
        if a.lower() in ("unos", "unas"):
            prefijo = prefijo[:art.start(1)]                   # «unas 2 ciruelas» → «½ ciruela»
        else:
            nuevo_art = _ARTICULO_SINGULAR.get(a.lower(), a)
            prefijo = prefijo[:art.start(1)] + (nuevo_art.capitalize() if a[:1].isupper() else nuevo_art) + " "
            if uno:
                cabeza = ""                                      # «divide las 2 ciruelas» → «divide la ciruela»
    if not (art and a.lower() not in ("unos", "unas") and uno):
        cabeza = formatear_cantidad(objetivo) + " "
    cuerpo = cabeza + "".join(w + " " for w in entre) + sing
    nuevo = prefijo + cuerpo
    pos_fin = len(nuevo)
    nuevo += paso[ff:]
    nuevo = _singulariza_adjetivos_tras(nuevo, pos_fin)          # «ciruelas frescas» → «ciruela fresca»
    # la cláusula que sigue, si sólo habla de este alimento: «córtalos en 4 trozos cada uno» → «córtalo en 4 trozos»
    try:
        ini_cl = max([a_ for a_, b_ in clause_bounds(nuevo) if a_ <= pos_fin] or [0])
        fin_cl = next((b_ for a_, b_ in clause_bounds(nuevo) if a_ <= pos_fin < b_), len(nuevo))
        foods = {n for _, _, n in _catalog_food_spans(nuevo[ini_cl:fin_cl], index)}
        if foods == {m["food"]}:
            seg = nuevo[pos_fin:fin_cl]
            seg2 = _CADA_UNO_RE.sub("", _CLITICO_PLURAL_RE.sub(r"\1", seg))
            if seg2 != seg:
                for mc in list(re.finditer(r"[\wáéíóúñü]*[áéíóú][\wáéíóúñü]*[aeiou]l[oa]\b", seg2)):
                    seg2 = _singulariza_adjetivos_tras(seg2, mc.end(), 1)   # «resérvala enteras» → «resérvala entera»
                nuevo = nuevo[:pos_fin] + seg2 + nuevo[fin_cl:]
    except Exception:
        pass
    return nuevo if nuevo != paso else None


def _cruza_plural(a: float, b: float) -> bool:
    return (a <= 1.0) != (b <= 1.0)


_ARTICULO_RE = re.compile(r"(\b)(las|los|la|el)(\s+)$", re.IGNORECASE)
_ARTICULO_PLURAL = {"la": "las", "el": "los", "las": "las", "los": "los"}
_ARTICULO_SINGULAR = {"las": "la", "los": "el", "la": "la", "el": "el"}


def _concordar_articulo(paso: str, ini: int, valor: float) -> str:
    """El artículo que precede al número sigue al número: «las 2 rebanadas» → «la 1 rebanada»."""
    m = _ARTICULO_RE.search(paso[:ini])
    if not m:
        return paso
    art = m.group(2)
    nuevo = (_ARTICULO_PLURAL if valor > 1.0 else _ARTICULO_SINGULAR).get(art.lower(), art)
    if art[:1].isupper():
        nuevo = nuevo[:1].upper() + nuevo[1:]
    return paso[:m.start(2)] + nuevo + paso[m.end(2):]


# ─────────────── [P1-PLAN-LOTE-182 · 2026-09-23] el peso aproximado de la lista ───────────────
# Batería real (rd9 + rd10, 111 comidas): 12 con un paso que no cuadra con su lista; en 7 de 13 menciones el paso daba
# GRAMOS de un alimento que la lista cuenta en piezas con su peso: «corta 275 g de pechuga de pollo» con «1 pechuga de
# pollo (≈134 g)». Ese «(≈N g)» lo pone el humanizador y es el PESO con el que se miden los macros
# (`nutrition_db.grams_from_ingredient_string` prefiere el paréntesis), pero el paso nunca se comparaba con él: V4 no
# toma lo aproximado como contrato de MEDICIÓN y aquí no había otro peso con el que alinear el paso. Quien sigue la
# receta cocinaba el doble de pollo del que el plan cuenta.
# Sólo en la dirección que el contrato ya repara sin discusión (el paso pide MÁS de lo que la lista pesa: se recorta);
# «el paso pide MENOS» sigue siendo la decisión V7a del dueño y no se toca.
# Knob `MEALFIT_CONTRACT_APPROX_GRAMS` (True). tooltip-anchor: P1-PLAN-LOTE-182-PESO-APROXIMADO
_HINT_APROX_RE = re.compile(r"\(\s*≈\s*(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos)\b[^)]*\)", re.IGNORECASE)


def approx_grams_on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_CONTRACT_APPROX_GRAMS", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _gramos_aproximados(ings: list, index: dict, lista: dict) -> dict:
    """{(alimento, "g"): gramos} del «(≈N g)» de las líneas contables de la lista, sólo para alimentos SIN gramos
    exactos en ella. Se suman entre líneas, como la lista."""
    out: dict = {}
    for ing in ings:
        s = str(ing)
        mh = _HINT_APROX_RE.search(s)
        if not mh:
            continue
        foods = list(find_catalog_foods(_norm(_HINT_APROX_RE.sub(" ", s)), index))
        if len(foods) != 1 or (foods[0], "g") in lista:
            continue
        clave = (foods[0], "g")
        out[clave] = out.get(clave, 0.0) + float(mh.group(1).replace(",", "."))
    return out


def reconcile_step_quantities(meal: dict, index: dict) -> dict:
    """Reescribe en `meal["recipe"]` las cantidades que contradicen a `meal["ingredients"]`. Muta `meal` in-place.
    Devuelve el informe: {"reescritas", "familias", "sin_reparar": {"gramatical", "reparto"}, "cambios": [...]}."""
    informe = {"reescritas": 0, "familias": Counter(), "sin_reparar": Counter(), "cambios": [], "concordancia": 0}
    try:
        if not isinstance(meal, dict) or not index:
            return informe
        ings = [str(x) for x in (meal.get("ingredients") or []) if str(x).strip()]
        rec = meal.get("recipe")
        if not ings or not isinstance(rec, list) or not rec:
            return informe
        lista = _cantidades_lista(ings, index)
        if not lista:
            return informe
        # [P1-PLAN-LOTE-182] el «(≈N g)» de una línea contable es el techo de los gramos del paso
        aprox = _gramos_aproximados(ings, index, lista) if approx_grams_on() else {}
        objetivos = {**lista, **aprox}
        # todas las menciones de todos los pasos, para saber cuántas veces aparece cada (alimento, familia)
        por_paso = []
        conteo = Counter()
        for i, paso in enumerate(rec):
            if not isinstance(paso, str) or _es_nota(paso):
                por_paso.append([])
                continue
            ms = [m for m in _menciones_paso(paso, index) if (m["food"], m["familia"]) in objetivos]
            por_paso.append(ms)
            for m in ms:
                conteo[(m["food"], m["familia"])] += 1
        for i, ms in enumerate(por_paso):
            if not ms:
                continue
            paso = rec[i]
            # de derecha a izquierda para que los offsets anteriores sigan valiendo
            for m in sorted(ms, key=lambda x: x["ini"], reverse=True):
                clave = (m["food"], m["familia"])
                objetivo = objetivos[clave]
                if clave in aprox and m["valor"] <= objetivo and (conteo[clave] >= 2 or not __import__("v7a_plural").activo()):
                    continue                                   # [P1-PLAN-LOTE-182] contra el peso aproximado, sólo se recorta
                    #                                            [P1-PLAN-LOTE-212] V7a: con UNA mención, también se sube
                if _tolera(m["familia"] if m["familia"] == "g" else "n", m["valor"], objetivo):
                    continue
                if conteo[clave] >= 2 and m["valor"] < objetivo:
                    informe["sin_reparar"]["reparto"] += 1      # varias menciones: sólo se recorta el exceso
                    continue
                if m["familia"] == "pieza" and _cruza_plural(m["valor"], objetivo):
                    # [P1-PLAN-LOTE-31] el paso pide MÁS que la lista ⇒ se reescribe con concordancia al singular;
                    # [P1-PLAN-LOTE-212] pide MENOS ⇒ al plural (V7a, decisión del dueño del 24-sep); lo que no sabe ⇒ gramatical
                    nuevo_paso = (_concordar_pieza(paso, m, objetivo, index) if m["valor"] > objetivo
                                  else __import__("v7a_plural").pluralizar_pieza(paso, m, objetivo, index))
                    if nuevo_paso is None:
                        informe["sin_reparar"]["gramatical"] += 1
                        continue
                    informe["reescritas"] += 1
                    informe["familias"]["pieza"] += 1
                    informe["concordancia"] += 1
                    informe["cambios"].append({"paso": i, "food": m["food"], "de": m["valor"], "a": objetivo, "familia": "pieza",
                                               "concordancia": True, "antes": paso[max(0, m["ini"] - 30):m["fin"] + 40],
                                               "despues": nuevo_paso[max(0, m["ini"] - 30):m["ini"] + 44]})
                    paso = nuevo_paso
                    continue
                if m["familia"] == "pieza" and m["valor"] < objetivo and (m["food"], "g") in lista:
                    # «1 cebolla (25 g)» en la lista y «pica ¼ de cebolla» en el paso: el conteo de la lista es
                    # de COMPRA y los gramos son lo que se usa (el hint gana en el parser de nutrición). Subir el
                    # paso al conteo cuadruplicaría la cebolla del plato. Sólo se recorta el exceso.
                    informe["sin_reparar"]["conteo_con_gramos"] += 1
                    continue
                nuevo_num = formatear_cantidad(objetivo)
                antes = paso
                if m["familia"] == "pieza":
                    # «¼ de cebolla» → «1 cebolla»: con un entero, el «de» partitivo sobra
                    cola_ini = m["fin"]
                    if objetivo >= 1 and abs(objetivo - round(objetivo)) < 0.01 and paso[cola_ini:cola_ini + 4].lower() == " de ":
                        paso = paso[:m["ini"]] + nuevo_num + paso[cola_ini + 3:]
                    else:
                        paso = paso[:m["ini"]] + nuevo_num + paso[m["fin"]:]
                elif m["familia"] != "g" and m["unidad"]:
                    uni_txt = m["unidad"]
                    base = _unidad_norm(uni_txt)
                    nueva_uni = (UNIDAD_PLURAL.get(base, uni_txt) if objetivo > 1.0 else base)
                    if uni_txt[:1].isupper():
                        nueva_uni = nueva_uni[:1].upper() + nueva_uni[1:]
                    paso = paso[:m["ini"]] + nuevo_num + paso[m["fin"]:m["u_ini"]] + nueva_uni + paso[m["u_fin"]:]
                else:
                    paso = paso[:m["ini"]] + nuevo_num + paso[m["fin"]:]
                # concordancia del artículo: «tuesta las 2 rebanadas» → «tuesta la 1 rebanada»; «el 1 huevo» → «los 3 huevos»
                if m["familia"] != "g":
                    paso = _concordar_articulo(paso, m["ini"], objetivo)
                if paso != antes:
                    informe["reescritas"] += 1
                    fam = m["familia"] if isinstance(m["familia"], str) else f"u:{m['familia'][1]}"
                    informe["familias"][fam] += 1
                    informe["cambios"].append({"paso": i, "food": m["food"], "de": m["valor"], "a": objetivo,
                                               "familia": fam,
                                               "antes": antes[max(0, m["ini"] - 30):m["fin"] + 30],
                                               "despues": paso[max(0, m["ini"] - 30):m["ini"] + len(nuevo_num) + 30]})
            rec[i] = paso
        informe["familias"] = dict(informe["familias"])
        informe["sin_reparar"] = dict(informe["sin_reparar"])
        return informe
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-23] reconcile_step_quantities fail-open ({type(e).__name__}: {e})")
        informe["familias"] = dict(informe["familias"])
        informe["sin_reparar"] = dict(informe["sin_reparar"])
        return informe


def reconcile_days(days: list, index: dict) -> dict:
    """El contrato sobre todas las comidas de `days`. Devuelve el agregado (comidas tocadas, reescritas, familias)."""
    agg = {"comidas": 0, "comidas_tocadas": 0, "reescritas": 0, "familias": Counter(), "sin_reparar": Counter(), "concordancia": 0}
    for d in days or []:
        for m in (d.get("meals") or []) if isinstance(d, dict) else []:
            if not isinstance(m, dict):
                continue
            agg["comidas"] += 1
            r = reconcile_step_quantities(m, index)
            if r["reescritas"]:
                agg["comidas_tocadas"] += 1
                agg["reescritas"] += r["reescritas"]
            agg["familias"].update(r["familias"])
            agg["sin_reparar"].update(r["sin_reparar"])
            agg["concordancia"] += int(r.get("concordancia") or 0)
    agg["familias"] = dict(agg["familias"])
    agg["sin_reparar"] = dict(agg["sin_reparar"])
    return agg


# ─────────────────────────────────────────────────────────
# [P1-PLAN-LOTE-24 · 2026-09-12] (C3 · CUL-P0-04) Las tres formas del huevo: entero, clara y yema NO son intercambiables.
#
# Medido en el corpus fijo del 09-12: 14 de 64 comidas llevan huevo; 6 de ellas salieron del tope diario de enteros
# (`_cap_daily_whole_eggs`: «6 huevos» → «3 huevos» + «3 claras de huevo» EN LA LISTA) y en las 6 los pasos seguían
# diciendo «casca 6 huevos» — la forma cambió en la lista y no en la receta. Peor: el contrato de cantidades de C2,
# ciego a la forma, reescribía «casca 6 huevos» → «casca 3 huevos» y las claras desaparecían de la preparación.
# Nutrición y compra ya distinguen las tres formas (`Clara de huevo` 33 g/ud y 0,1 g de grasa por 100 g; `Huevo` 50 g y
# 9,5 g; `Yema de huevo` 17 g y 26,5 g): la receta era la única capa que las confundía.
#
# Dos reglas, ambas con la LISTA como autoridad:
#   1. La lista dice la forma con su nombre: «N huevos sin yema» / «N huevos (solo claras)» → «N claras de huevo»;
#      «N huevos sin clara» → «N yemas de huevo». Sin esto «12 huevos sin yema» hereda los macros de 12 enteros
#      (medido: 600 g y 57 g de grasa contra 396 g y 0,7 g). Cambia la lista ⇒ el llamador re-mide los macros.
#   2. Los pasos siguen a la forma de la lista. Si la lista trae claras (o yemas), toda mención «N huevos» de los pasos
#      que no coincida con los enteros comprados se reescribe con el reparto real («3 huevos y 2 claras de huevo»);
#      si NO hay enteros, «N huevo(s)» y «el/los huevo(s)» pasan a la forma comprada («1 clara de huevo», «la clara») y
#      la nota de seguridad deja de exigir «yema y clara firmes». Si las claras de la lista no aparecen en NINGÚN paso,
#      la primera mención de huevo recibe el reparto. Nunca al revés: una lista de enteros con pasos que hablan de
#      «la clara cuajada» es técnica, no contradicción — no se toca. Nada convierte claras en yemas.
# tooltip-anchor: P1-PLAN-LOTE-24-EGG-FORMS

HUEVO, CLARA, YEMA = "Huevo", "Clara de huevo", "Yema de huevo"
_EGG_LIST_SIN_YEMA_RE = re.compile(
    r"^(\s*(?:\d+(?:[.,]\d+)?|[¼½¾⅓⅔⅛])\s+)huevos?\s*\(?\s*(?:sin\s+(?:la\s+|las\s+)?yemas?|s[oó]lo\s+(?:la\s+|las\s+)?claras?)"
    r"\s*\)?\s*(?:\(\s*\d+(?:[.,]\d+)?\s*g\s*\))?\s*$", re.IGNORECASE)
_EGG_LIST_SIN_CLARA_RE = re.compile(
    r"^(\s*(?:\d+(?:[.,]\d+)?|[¼½¾⅓⅔⅛])\s+)huevos?\s*\(?\s*(?:sin\s+(?:la\s+|las\s+)?claras?|s[oó]lo\s+(?:la\s+|las\s+)?yemas?)"
    r"\s*\)?\s*(?:\(\s*\d+(?:[.,]\d+)?\s*g\s*\))?\s*$", re.IGNORECASE)
_EGG_NOUN_AFTER_NUM_RE = re.compile(r"\s+huevos?(?:\s+enteros?)?\b", re.IGNORECASE)
_EGG_DEFINITE_RE = re.compile(
    r"\b(el|los)\s+huevos?\b(?!\s+(?:duros?|fritos?|revueltos?|cocidos?|hervidos?|estrellados?|escalfados?|pasados?|batidos?)\b)",
    re.IGNORECASE)
_EGG_BARE_NOUN_RE = re.compile(r"(?<!\bde )(?<![\d½¼¾⅓⅔]\s)(?:\b(?:el|los|un|unos)\s+)?\bhuevos?\b(?!\s+(?:duros?|fritos?|revueltos?|cocidos?|hervidos?)\b)",
                               re.IGNORECASE)
# [P1-PLAN-LOTE-307 · 2026-09-25] El huevo tiene su propio detector numérico. «bate 4 huevos con el comino» con la lista
# «3 huevos + 1 clara» salía «bate 4 3 huevos y 1 clara de huevo»: `_menciones_paso` no veía el «4 huevos» (la cola
# «huevos con el comino» nombra DOS alimentos del catálogo y V7 la descarta por ambigua) y el respaldo sustituía sólo la
# palabra «huevos», dejando el 4 delante. Además: si el paso ya nombra las claras, sólo cambia el conteo de enteros (antes
# «bate 3 huevos y 1 clara de huevo y 1 clara»), y con la lista sólo de claras «bate 1 huevo con ajo» pasa a «bate 1 clara
# de huevo con ajo» (quedaba el huevo entero que la lista no compra). El respaldo ya no casa justo detrás de un número.
# tooltip-anchor: P1-PLAN-LOTE-307-HUEVO-NUMERICO
_EGG_NUM_RE = re.compile(
    r"(?<![\w.,/])(?:(?:los|las|unos)\s+)?(\d+(?:[.,]\d+)?|[½¼¾⅓⅔])\s+huevos?(?:\s+enteros?)?\b(?!\s+de\s+codorniz)",
    re.IGNORECASE)
_EGG_SAFETY_WHOLE_TXT = "yema y clara firmes, sin partes líquidas"
_EGG_SAFETY_WHITES_TXT = "la clara firme, sin partes líquidas"


def _plural_huevo(n: float, forma: str) -> str:
    uno = abs(n - 1.0) < 0.01
    if forma == HUEVO:
        return f"{formatear_cantidad(n)} {'huevo' if uno else 'huevos'}"
    if forma == CLARA:
        return f"{formatear_cantidad(n)} {'clara' if uno else 'claras'} de huevo"
    return f"{formatear_cantidad(n)} {'yema' if uno else 'yemas'} de huevo"


def egg_forms_in_list(ings: list, index: dict) -> dict:
    """{"Huevo": a, "Clara de huevo": b, "Yema de huevo": c} en PIEZAS, leído con el mismo parser que V7."""
    lista = _cantidades_lista([str(x) for x in (ings or []) if str(x).strip()], index)
    formas = {f: float(lista.get((f, "pieza"), 0.0)) for f in (HUEVO, CLARA, YEMA)}
    # [P1-PLAN-LOTE-307] la forma que la lista da SÓLO en gramos («60 g de huevo») también cuenta, a 50/33/17 g la pieza:
    # sin esto «60 g de huevo» + «6 claras» se leía «sólo claras» y «añade el huevo» pasaba a «añade las claras»
    for f, g_pieza in ((HUEVO, 50.0), (CLARA, 33.0), (YEMA, 17.0)):
        g = float(lista.get((f, "g"), 0.0))
        if formas[f] <= 0 and g > 0:
            formas[f] = float(max(1, round(g / g_pieza)))
    return formas


def canonicalize_egg_form_lines(meal: dict) -> int:
    """Regla 1: la lista nombra la forma. Reescribe `ingredients` y la línea IGUAL de `ingredients_raw` (por texto,
    nunca por índice). Devuelve nº de líneas reescritas. Nunca convierte claras en enteros ni en yemas."""
    n = 0
    try:
        ings = meal.get("ingredients")
        if not isinstance(ings, list):
            return 0
        raw = meal.get("ingredients_raw") if isinstance(meal.get("ingredients_raw"), list) else None
        for i, s in enumerate(ings):
            if not isinstance(s, str):
                continue
            nuevo = None
            m = _EGG_LIST_SIN_YEMA_RE.match(s)
            if m:
                nuevo = f"{m.group(1)}claras de huevo" if not _uno(m.group(1)) else f"{m.group(1)}clara de huevo"
            else:
                m = _EGG_LIST_SIN_CLARA_RE.match(s)
                if m:
                    nuevo = f"{m.group(1)}yemas de huevo" if not _uno(m.group(1)) else f"{m.group(1)}yema de huevo"
            if nuevo is None or nuevo == s:
                continue
            ings[i] = nuevo
            if raw is not None:
                hits = [j for j, r in enumerate(raw) if isinstance(r, str) and r.strip() == s.strip()]
                if len(hits) == 1:
                    raw[hits[0]] = nuevo
            n += 1
        return n
    except Exception:
        return n


def _uno(lead: str) -> bool:
    try:
        return abs(float(str(lead).strip().replace(",", ".")) - 1.0) < 0.01
    except ValueError:
        return False


def egg_forms_step_sync(meal: dict, index: dict) -> dict:
    """Regla 2: los pasos siguen a la forma de la lista. Muta `meal["recipe"]`. Informe: {"reescritas", "cambios"}."""
    informe = {"reescritas": 0, "cambios": []}
    try:
        if not isinstance(meal, dict) or not index:
            return informe
        rec = meal.get("recipe")
        if not isinstance(rec, list) or not rec:
            return informe
        formas = egg_forms_in_list(meal.get("ingredients") or [], index)
        a, b, c = formas[HUEVO], formas[CLARA], formas[YEMA]
        if b <= 0 and c <= 0:
            return informe                       # lista de enteros: «hasta que la clara cuaje» es técnica
        frase = " y ".join(_plural_huevo(n, f) for n, f in ((a, HUEVO), (b, CLARA), (c, YEMA)) if n > 0)
        pasos_txt = strip_accents(" ".join(p for p in rec if isinstance(p, str) and not _es_nota(p)).lower())
        claras_en_pasos = bool(re.search(r"\bclaras?\b", pasos_txt)) if b > 0 else True
        yemas_en_pasos = bool(re.search(r"\byemas?\b", pasos_txt)) if c > 0 else True
        primera_pendiente = not (claras_en_pasos and yemas_en_pasos)
        for i, paso in enumerate(rec):
            if not isinstance(paso, str):
                continue
            antes = paso
            if _es_nota(paso):
                if a <= 0 and c <= 0 and _EGG_SAFETY_WHOLE_TXT in paso:
                    paso = paso.replace(_EGG_SAFETY_WHOLE_TXT, _EGG_SAFETY_WHITES_TXT)
            else:
                # [P1-PLAN-LOTE-307] menciones numéricas del huevo con su propio detector (ver `_EGG_NUM_RE`)
                otra_forma = bool(re.search(r"\b(?:claras?|yemas?)\b", strip_accents(paso.lower())))
                cambios_paso = []
                for mm in _EGG_NUM_RE.finditer(paso):
                    val = _v6_valor(mm.group(1))
                    if val is None:
                        continue
                    coincide = a > 0 and _tolera("n", val, a)
                    if coincide and not primera_pendiente:
                        continue
                    if a <= 0:
                        if otra_forma:
                            continue                      # «2 huevos y 2 claras» con sólo claras en la lista: no se adivina
                        nuevo = _plural_huevo(b if b > 0 else c, CLARA if b > 0 else YEMA)
                    elif otra_forma:
                        nuevo = _plural_huevo(a, HUEVO)   # el paso ya nombra las claras: sólo el conteo de enteros
                    else:
                        nuevo = frase
                    cambios_paso.append((mm.start(), mm.end(), nuevo))
                    primera_pendiente = False
                for ini_c, fin_c, nuevo in reversed(cambios_paso):
                    paso = paso[:ini_c] + nuevo + paso[fin_c:]
                if a <= 0:
                    forma_def = CLARA if b > 0 else YEMA
                    n_def = b if b > 0 else c
                    def _def(mm, _n=n_def, _f=forma_def):
                        uno = abs(_n - 1.0) < 0.01
                        art = ("la" if uno else "las")
                        if mm.group(1)[:1].isupper():
                            art = art.capitalize()
                        sust = ("clara" if _f == CLARA else "yema") + ("" if uno else "s")
                        return f"{art} {sust}"
                    paso = _EGG_DEFINITE_RE.sub(_def, paso)
            if paso != antes:
                rec[i] = paso
                informe["reescritas"] += 1
                informe["cambios"].append({"paso": i, "food": HUEVO, "familia": "huevo_forma", "de": None, "a": frase,
                                           "antes": antes[:120], "despues": paso[:120]})
        if primera_pendiente and a > 0:
            # Lista mixta cuyas claras no aparecen en NINGÚN paso y sin una sola mención numérica («Cocina huevo a la
            # plancha»): la primera mención desnuda del huevo recibe el reparto comprado.
            for i, paso in enumerate(rec):
                if not isinstance(paso, str) or _es_nota(paso):
                    continue
                m = _EGG_BARE_NOUN_RE.search(paso)
                if not m:
                    continue
                nuevo = paso[:m.start()] + frase + paso[m.end():]
                rec[i] = nuevo
                informe["reescritas"] += 1
                informe["cambios"].append({"paso": i, "food": HUEVO, "familia": "huevo_forma", "de": None, "a": frase,
                                           "antes": paso[:120], "despues": nuevo[:120]})
                break
        return informe
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-24] egg_forms_step_sync fail-open ({type(e).__name__}: {e})")
        return informe


def _reparar_estructura(meal: dict) -> dict:
    """[P1-PLAN-LOTE-29] (CUL-P1-04) paso (4): la crema que no espesa, el wrap que no cierra, la tortilla con el vegetal
    crudo — sólo texto de los pasos, jamás la lista. Fail-open."""
    try:
        from recipe_repair import reparar_estructura
        return reparar_estructura(meal)
    except Exception:
        return {"aplicado": [], "descartado": [], "cambios": []}


def _colapsar_repeticiones(meal: dict, index: dict) -> dict:
    """[P1-PLAN-LOTE-31] Paso (6): la misma mención numérica repetida en cadena dentro de un paso («6 claras de huevo y 6
    claras de huevo y 6 claras») se deja una vez (`recipe_repair.colapsar_repeticiones`). Fail-open."""
    try:
        from recipe_repair import colapsar_repeticiones
        return colapsar_repeticiones(meal, index)
    except Exception:
        return {"aplicado": [], "descartado": [], "cambios": []}


def _retirar_sin_lista(meal: dict, index: dict) -> dict:
    """[P1-PLAN-LOTE-30] Paso (5): los pasos dejan de nombrar lo que la lista no trae (`recipe_repair.retirar_sin_lista`).
    Fail-open: sin índice o con error, nada cambia y nada se anota."""
    try:
        from recipe_repair import retirar_sin_lista
        return retirar_sin_lista(meal, index)
    except Exception:
        return {"aplicado": [], "descartado": [], "cambios": []}


def reconcile_meal(meal: dict, index: dict) -> dict:
    """El contrato completo sobre UN plato, en orden: (1) la lista nombra la forma del huevo, (2) los pasos siguen a
    esa forma, (3) las cantidades de los pasos siguen a la lista (C2), (4) la ESTRUCTURA del plato se repara sin tocar la
    lista (`recipe_repair`, P1-PLAN-LOTE-29), (5) los pasos dejan de nombrar lo que la lista NO trae (V5 — nace cuando un
    cerrador quita una línea; `recipe_repair.retirar_sin_lista`, P1-PLAN-LOTE-30), (6) la misma mención numérica repetida
    en cadena dentro de un paso se deja una vez (`recipe_repair.colapsar_repeticiones`, P1-PLAN-LOTE-31: el LLM escribió
    «6 claras de huevo y 6 claras de huevo y 6 claras»). Informe agregado; `lista_reescrita` > 0
    avisa al llamador de que la lista cambió y los macros hay que re-medirlos; `estructura` cuenta las reparaciones de (4),
    `sin_lista` los alimentos retirados de los pasos en (5), `repeticiones` las cadenas colapsadas en (6) y `concordancia`
    las piezas que (3) reescribió cambiando de número gramatical; lo que (5)/(6) no pudo va a `sin_reparar[...]`."""
    lista_n = canonicalize_egg_form_lines(meal)
    huevo = egg_forms_step_sync(meal, index)
    r = reconcile_step_quantities(meal, index)
    est = _reparar_estructura(meal)
    r["estructura"] = len(est.get("aplicado") or [])
    if r["estructura"]:
        r["cambios_estructura"] = list(est.get("cambios") or [])
    sl = _retirar_sin_lista(meal, index)                                 # (5) [P1-PLAN-LOTE-30]
    r["sin_lista"] = len(sl.get("aplicado") or [])
    if sl.get("descartado"):
        r["sin_reparar"] = dict(r.get("sin_reparar") or {})
        r["sin_reparar"]["sin_lista"] = len(sl["descartado"])
    if r["sin_lista"]:
        r["cambios_sin_lista"] = list(sl.get("cambios") or [])
    rp = _colapsar_repeticiones(meal, index)                              # (6) [P1-PLAN-LOTE-31]
    r["repeticiones"] = len(rp.get("aplicado") or [])
    if rp.get("descartado"):
        r["sin_reparar"] = dict(r.get("sin_reparar") or {})
        r["sin_reparar"]["repeticion"] = len(rp["descartado"])
    if r["repeticiones"]:
        r["cambios_repeticion"] = list(rp.get("cambios") or [])
    r["lista_reescrita"] = lista_n
    r["reescritas"] += huevo["reescritas"]
    if huevo["reescritas"]:
        r["familias"] = dict(r.get("familias") or {})
        r["familias"]["huevo_forma"] = huevo["reescritas"]
    r["cambios"] = list(huevo["cambios"]) + list(r.get("cambios") or [])
    return r


_INDEX_CACHE: dict = {"index": None, "n": -1}


def index_for_catalog(catalog: list) -> dict:
    """Índice culinario con caché por tamaño de catálogo (el catálogo cambia rara vez; el índice cuesta construirlo)."""
    try:
        n = len(catalog or [])
        if _INDEX_CACHE["index"] is None or _INDEX_CACHE["n"] != n:
            _INDEX_CACHE["index"] = build_culinary_index(catalog or [])
            _INDEX_CACHE["n"] = n
        return _INDEX_CACHE["index"] or {}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────────────────────
# El gancho: ÚLTIMO en los dos finalizadores del persist boundary

MODOS = ("off", "shadow", "repair")
TELEMETRIA_KEY = "_recipe_contract_final"


def final_contract_mode() -> str:
    """Knob `MEALFIT_RECIPE_FINAL_CONTRACT`: `repair` (default — la lista tiene la última palabra y los pasos se
    reescriben), `shadow` (sólo anota en `_recipe_contract_final` lo que habría reescrito: para medir en una flota
    nueva antes de tocar), `off`. Vuelta atrás sin redeploy: `MEALFIT_RECIPE_FINAL_CONTRACT=shadow`."""
    try:
        from knobs import _env_str
        m = (_env_str("MEALFIT_RECIPE_FINAL_CONTRACT", "repair") or "repair").strip().lower()
        return m if m in MODOS else "repair"
    except Exception:
        return "repair"


def _index_default(db=None) -> dict:
    """El índice del catálogo real (`master_ingredients`); `{}` si no hay catálogo (fuera de FastAPI sin pool abierto).
    Sin índice el contrato no toca nada — igual que el scan: fail-open, pero se ve en la telemetría."""
    try:
        from shopping_calculator import get_master_ingredients
        return index_for_catalog(get_master_ingredients() or [])
    except Exception:
        return {}


def _remedir_macros(meal: dict, db) -> None:
    """[P1-PLAN-LOTE-24] Si el contrato reescribió una línea de la LISTA («12 huevos sin yema» → «12 claras de huevo»),
    los macros del plato se re-miden con el mismo truth-up del repo. Import perezoso (graph_orchestrator importa este
    módulo). Fail-open."""
    if db is None:
        return
    try:
        from graph_orchestrator import _truth_up_meal_macros_from_strings as _tu
        _tu(meal, db)
    except Exception as e:
        logger.debug(f"[P1-PLAN-LOTE-24] re-medición tras canonizar la forma del huevo no-op: {type(e).__name__}: {e}")


def _aplicar_meal(meal: dict, index: dict, mode: str, db=None) -> int:
    """Un plato: en `repair` reescribe y anota; en `shadow` mide sobre una copia y anota lo que habría hecho.
    La anotación (`_recipe_contract_final`) sólo se escribe cuando hay algo que decir, para no engordar cada plato."""
    import copy as _copy
    if mode == "shadow":
        sombra = _copy.deepcopy(meal)
        r = reconcile_meal(sombra, index)
    else:
        __import__("pasos_cantidades").quitar_trazas(meal)  # [P1-PLAN-LOTE-319] «0.95 ml de leche»: antes de C2, cuyo V5 limpia los pasos
        r = reconcile_meal(meal, index)
        __import__("pasos_cantidades").sincronizar_exacto(meal)   # [P1-PLAN-LOTE-308] el contrato tolera ±25 %
        __import__("pasos_cantidades").pesos_de_la_lista(meal)    # [P1-PLAN-LOTE-310] «¾ manzana (≈120 g)»
        __import__("pasos_cantidades").lo_que_dice_la_lista(meal)  # [P1-PLAN-LOTE-328..331] piezas, porciones, pizcas
        __import__("avena_liquido").completar(meal)               # [P1-PLAN-LOTE-311] la avena cocida lleva líquido
        __import__("pasos_cantidades").decimales_de_cocina(meal)  # [P1-PLAN-LOTE-312] «2.22 cdas» → «2¼ cdas»
        __import__("pasos_cantidades").frases_repetidas(meal)     # [P1-PLAN-LOTE-316] «Acompaña con X. Acompaña con X.»
        if r.get("lista_reescrita"):
            _remedir_macros(meal, db)
    if r["reescritas"] or r["sin_reparar"] or r.get("lista_reescrita") or r.get("estructura") or r.get("sin_lista") or r.get("repeticiones"):
        meal[TELEMETRIA_KEY] = {"modo": mode, "reescritas": r["reescritas"], "familias": r["familias"],
                                "sin_reparar": r["sin_reparar"]}
        if r.get("lista_reescrita"):
            meal[TELEMETRIA_KEY]["lista_reescrita"] = r["lista_reescrita"]   # [P1-PLAN-LOTE-24] sólo si la lista cambió
        if r.get("estructura"):
            meal[TELEMETRIA_KEY]["estructura"] = r["estructura"]             # [P1-PLAN-LOTE-29] sólo si se reparó estructura
        if r.get("sin_lista"):
            meal[TELEMETRIA_KEY]["sin_lista"] = r["sin_lista"]               # [P1-PLAN-LOTE-30] sólo si se retiró algo de los pasos
        if r.get("repeticiones"):
            meal[TELEMETRIA_KEY]["repeticiones"] = r["repeticiones"]         # [P1-PLAN-LOTE-31] sólo si se colapsó una cadena
        if r.get("concordancia"):
            meal[TELEMETRIA_KEY]["concordancia"] = r["concordancia"]         # [P1-PLAN-LOTE-31] piezas reescritas cambiando de número
    elif not meal.get(TELEMETRIA_KEY):
        # [P1-PLAN-LOTE-30] una pasada que no tiene nada que decir no borra lo que dijo una anterior: el bench en modo real
        # vio 8 sellos del pipeline quedar en 5 tras el INSERT — la flota perdía la cuenta de lo que el contrato SÍ hizo.
        meal.pop(TELEMETRIA_KEY, None)
    return r["reescritas"] if mode == "repair" else 0


def apply_final_contract(days: list, db=None) -> str:
    """Para `finalize_plan_data_coherence` y la cola del persist boundary (`db_plans._finalize_plan_data_for_insert`,
    swap, chat-modify): el contrato sobre TODAS las comidas, al final de todo. Devuelve el trozo del resumen
    (`recipe_contract=<n>` o `recipe_contract_shadow=<n>`), «» si no hubo nada. Idempotente: correr dos veces no
    cambia nada. Jamás lanza."""
    try:
        mode = final_contract_mode()
        if mode == "off" or not isinstance(days, list) or not days:
            return ""
        index = _index_default(db)
        if not index:
            return "recipe_contract=sin_catalogo"
        n = 0
        sombra = 0
        estructura = 0
        sin_lista = 0
        repeticiones = 0
        concordancia = 0
        for d in days:
            for m in (d.get("meals") or []) if isinstance(d, dict) else []:
                if isinstance(m, dict):
                    k = _aplicar_meal(m, index, mode, db)
                    n += k
                    estructura += int((m.get(TELEMETRIA_KEY) or {}).get("estructura") or 0)
                    sin_lista += int((m.get(TELEMETRIA_KEY) or {}).get("sin_lista") or 0)
                    repeticiones += int((m.get(TELEMETRIA_KEY) or {}).get("repeticiones") or 0)
                    concordancia += int((m.get(TELEMETRIA_KEY) or {}).get("concordancia") or 0)
                    if mode == "shadow" and m.get(TELEMETRIA_KEY, {}).get("reescritas"):
                        sombra += m[TELEMETRIA_KEY]["reescritas"]
        if estructura and mode == "repair":
            logger.info(f"🧱 [P1-PLAN-LOTE-29] estructura del plato reparada en {estructura} comida(s) (sin tocar la lista)")
        if sin_lista and mode == "repair":
            logger.info(f"🧹 [P1-PLAN-LOTE-30] {sin_lista} alimento(s) que la lista no trae retirado(s) de los pasos "
                        f"(la lista tiene la última palabra también cuando pierde una línea)")
        if repeticiones and mode == "repair":
            logger.info(f"🔁 [P1-PLAN-LOTE-31] {repeticiones} cadena(s) de la misma mención repetida colapsada(s) en los pasos")
        if concordancia and mode == "repair":
            logger.info(f"🔠 [P1-PLAN-LOTE-31] {concordancia} pieza(s) reescrita(s) con concordancia singular/plural "
                        f"(el paso pedía más de lo que la lista compra)")
        if mode == "shadow":
            return f"recipe_contract_shadow={sombra}" if sombra else ""
        if n:
            logger.info(f"📐 [P1-PLAN-LOTE-23] contrato final de receta: {n} cantidad(es) de pasos re-alineada(s) con la lista")
        return f"recipe_contract={n}" if n else ""
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-23] apply_final_contract fail-open ({type(e).__name__}: {e})")
        return ""


def apply_final_contract_meal(meal: dict, db=None) -> int:
    """Para `finalize_single_meal_recipe_coherence` (swap, chat-modify, regenerar día, recipe-expand): el mismo
    contrato sobre UN plato. Devuelve nº de cantidades reescritas (0 en shadow/off). Jamás lanza."""
    try:
        mode = final_contract_mode()
        if mode == "off" or not isinstance(meal, dict):
            return 0
        index = _index_default(db)
        if not index:
            return 0
        return _aplicar_meal(meal, index, mode, db)
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-23] apply_final_contract_meal fail-open ({type(e).__name__}: {e})")
        return 0


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Extraído de graph_orchestrator.py (techo de líneas: 53.100) — misma conducta, mismo nombre, mismos tests.

def _ground_meat_step_noun_sync(days) -> int:
    """[P1-GROUND-MEAT-STEP-NOUN · 2026-07-28] Ingredientes dicen "pollo MOLIDO" pero los
    pasos hablan de "PECHUGA de pollo" ×3 (caso vivo, plan ab2b0a16 cena miércoles: el swap
    cambió el sustantivo en un lado y no en el otro — la clase del 'yogur cocido'). Cuando
    la especie está en forma MOLIDA en ingredientes y NINGUNA línea trae pechuga/filete de
    esa especie, los pasos que digan "pechuga de <especie>" se reescriben a "el <especie>
    molido". Display-only (cero macros). Fail-open por comida.
    [P1-PLAN-LOTE-23] Vive aquí (extraído de graph_orchestrator, que lo re-exporta con el mismo nombre): es un
    sincronizador lista→pasos, la familia del contrato final. tooltip-anchor: P1-GROUND-MEAT-STEP-NOUN
    """
    if not isinstance(days, list):
        return 0
    fixed = 0
    for day in days:
        for meal in (day.get("meals") or []) if isinstance(day, dict) else []:
            try:
                if not isinstance(meal, dict):
                    continue
                ings_sa = strip_accents(" ".join(str(i) for i in (meal.get("ingredients") or [])).lower())
                rec = meal.get("recipe")
                if not isinstance(rec, list):
                    continue
                _tocado = False
                for _esp in ("pollo", "pavo", "res", "cerdo"):
                    if f"{_esp} molido" not in ings_sa and f"{_esp} molida" not in ings_sa:
                        continue
                    if f"pechuga de {_esp}" in ings_sa or f"filete de {_esp}" in ings_sa:
                        continue
                    _rx = re.compile(rf"(?:la\s+|el\s+)?[Pp]echuga\s+de\s+{_esp}\b")
                    for j, p in enumerate(rec):
                        if isinstance(p, str) and _rx.search(p):
                            rec[j] = _rx.sub(f"el {_esp} molido", p)
                            _tocado = True
                if _tocado:
                    fixed += 1
                    logger.info(f"🔤 [P1-GROUND-MEAT-STEP-NOUN] '{str(meal.get('name'))[:40]}': "
                                f"pasos re-sincronizados a la forma molida del ingrediente.")
            except Exception:
                continue
    return fixed
