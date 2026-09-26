# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-27 · 2026-09-12] C5 (segunda parte) · CUL-P1-02: estructura culinaria del plato como contrato LIGERO de
evaluación — familia, componentes, relaciones de cantidades sensibles — derivado de lo que el plato YA declara (nombre,
lista, pasos), con confianza y trazabilidad. Metadata auxiliar de EVALUACIÓN: no escribe el texto del usuario
(`canonical_recipe.render_line` sigue reservado a compras/macros, como dejó dicho la revisión).

## La evidencia, medida antes de fijar un solo umbral (biblioteca curada DO, 193 recetas · corpus fijo, 64 comidas)

· Familias en la biblioteca: guiso 55, batido/crema 30, tortilla/revuelto 18, ensalada 13, panqueque 7, tostada/wrap 5,
  bowl 1. En el corpus: «otro» 33 de 64, batido/crema 11, panqueque 8, tostada/wrap 6.
· Wraps/tostadas curados: relleno ÷ pan = 2,4 · 2,5 · 3,8 (mediana 2,5). El umbral de «desproporcionado» se fija en
  5,0 — un 30 % por encima del máximo curado — y no en un número redondo inventado.
· Batidos/cremas curados: sólidos ÷ líquido — mediana 0,90, mínimo 0,26 (los tres valores ≈ 11 son avenas con poca
  leche, otra familia). Una «crema» que promete espesor con menos de 0,20 de sólidos por ml y sin proceso que espese
  (reducir, cocer hasta espesar, avena/chía/guineo/yogur griego/aguacate/mantequilla de maní) es la crema de 10 g de
  legumbre y cientos de ml de leche del backlog. En el corpus: «Avena cremosa de mango y chinola» — 30 g de avena en
  360 ml de leche (0,08) — es el caso real.
· Tortilla/revoltillo con vegetales de agua (tomate, calabacín, espinaca, champiñón, cebolla…): en 12 de 14 recetas
  curadas los vegetales se sofríen/saltean/escurren ANTES del huevo. Sin ese paso, la tortilla de claras suelta agua.

## Qué produce

`contract(meal)` → `{"familia", "componentes": {principal, soporte, liquidos_ml, solidos_g, vegetales_agua}, "relaciones":
[{"tipo", "detalle", "evidencia"}], "confianza", "fuente"}`. `relaciones` son los TRES hallazgos mecanizables del backlog:
`crema_sin_espesante`, `wrap_desproporcionado`, `tortilla_vegetales_crudos`. El escáner culinario las emite como **V9**
(`minor`, no reparable): el reparador que adapta relleno o técnica es CUL-P1-04.

Puro; nunca lanza. tooltip-anchor: P1-PLAN-LOTE-27-DISH-STRUCTURE
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

FAMILIAS = ("tortilla_revuelto", "panqueque", "bowl", "guiso", "ensalada", "tostada_wrap", "batido_crema", "otro")

_FAM_RE = {
    "tortilla_revuelto": re.compile(r"\b(tortilla|revoltillo|revuelt\w*|omelet\w*|frittata)\b"),
    "panqueque": re.compile(r"\b(panqueques?|pancakes?|tortitas?|arepitas?|crepes?|crepas?|waffles?|yaniqueques?)\b"),
    "bowl": re.compile(r"\b(bowl|tazon)\b"),
    "guiso": re.compile(r"\b(guis\w+|estofad\w+|asopao|sancocho|caldo|sopa|locrio|moro)\b"),
    "ensalada": re.compile(r"\bensaladas?\b"),
    "tostada_wrap": re.compile(r"\b(tostadas?|wraps?|sandwich\w*|sanduches?|burritos?|tacos?|pita|emparedados?|bocadillos?)\b"),
    "batido_crema": re.compile(r"\b(batid[oa]s?|licuados?|smoothies?|cremas?|pures?|jugos?)\b"),
}
#: la tortilla «de trigo/maíz» es el PAN del wrap, no la tortilla de huevo
_TORTILLA_PAN_RE = re.compile(r"\btortillas?\s+(?:de\s+)?(?:trigo|maiz|harina|integral)")
_PROMESA_ESPESA_RE = re.compile(r"\b(cremos[oa]s?|espes[oa]s?|espesa\w*|denso|densa|cremas?)\b")
_ESPESANTE_RE = re.compile(r"\b(avena|chia|guineo|platano|banana|yogur|yogurt|aguacate|mantequilla de mani|mani|"
                           r"almendra|nuez|nueces|semillas?|leche en polvo|queso|cottage|proteina|gelatina|maicena|"
                           r"harina|arroz|batata|auyama|lentejas?|garbanzos?|habichuelas?)\b")
_PROCESO_ESPESA_RE = re.compile(r"\b(reduc\w+|cocina hasta que espese|cuece hasta que espese|hierve hasta|deja que espese|"
                                r"a fuego bajo hasta|hasta que tome cuerpo|hasta que cuaje|espesa con|anade (?:la |el )?(?:avena|chia|maicena))\b")
_LIQUIDO_RE = re.compile(r"\b(leche|agua|caldo|jugo|zumo|bebida|te|cafe|kefir)\b")
_PAN_RE = re.compile(r"\b(pan|tortillas?|wraps?|pita|casabe|arepas?|tostadas?|galletas?)\b")
_VEG_AGUA_RE = re.compile(r"\b(tomates?|calabacin\w*|zucchini|espinacas?|champinon\w*|hongos?|setas?|cebollas?|"
                          r"pimientos?|ajies|aji|berenjenas?|repollo|pepinos?|tayota|vainitas?|brocoli)\b")
_VEG_PREP_RE = re.compile(r"\b(sofrie\w*|saltea\w*|escurr\w*|dora\w*|cocina (?:primero )?(?:la|el|los|las) |"
                          r"seca\w*|exprime\w*|deshidrat\w*|precocin\w*|blanquea\w*)")
_HUEVO_RE = re.compile(r"\b(huevos?|claras?|yemas?)\b")
_CANT_G = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:g|gr|grs|gramos?)\b\s*(?:de\s+)?(.+)$")
_CANT_ML = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:ml|mililitros?)\b\s*(?:de\s+)?(.+)$")
_CANT_L = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:l|litros?)\b\s*(?:de\s+)?(.+)$")
_PARENT_G = re.compile(r"\((\d+(?:[.,]\d+)?)\s*g\)")

UMBRALES = {
    # sólidos/ml por debajo de esto una «crema» no espesa sola (biblioteca: mediana 0,90, mínimo 0,26)
    "crema_solidos_por_ml_min": 0.20,
    # relleno/pan por encima de esto el wrap no cierra (biblioteca: 2,4–3,8; 5,0 = máximo curado + 30 %)
    "wrap_relleno_por_pan_max": 5.0,
}
EVIDENCIA = {
    "crema_sin_espesante": "biblioteca curada DO: sólidos/líquido mediana 0,90, mínimo 0,26 (n=23)",
    "wrap_desproporcionado": "biblioteca curada DO: relleno/pan 2,4–3,8, mediana 2,5 (n=3)",
    "tortilla_vegetales_crudos": "biblioteca curada DO: 12 de 14 tortillas/revoltillos sofríen o escurren el vegetal antes del huevo",
}


def _norm(s) -> str:
    t = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return " ".join("".join(c if c.isalnum() else " " for c in t).split())


def familia(meal) -> str:
    """La familia del plato por su NOMBRE (y, si el nombre calla, por los pasos para tortilla/revoltillo)."""
    try:
        n = _norm((meal or {}).get("name"))
        es_wrap, n = _nombre_de_familia_378(n)      # [P1-PLAN-LOTE-378] la tortilla que acompaña no hace wrap
        if es_wrap:
            return "tostada_wrap"
        for f in ("tortilla_revuelto", "panqueque", "tostada_wrap", "batido_crema", "guiso", "ensalada", "bowl"):
            if _FAM_RE[f].search(n):
                return f
        pasos = _norm(" . ".join(str(p) for p in ((meal or {}).get("recipe") or [])))
        if re.search(r"\b(revuelve los huevos|bate los huevos|cuaja)\b", pasos) and _HUEVO_RE.search(n or pasos):
            return "tortilla_revuelto"
        return "otro"
    except Exception:
        return "otro"


def _linea(ing) -> tuple:
    """`(nombre_norm, gramos, ml)` de una línea de la lista; los paréntesis «(45 g)» ganan como gramos."""
    s = str(ing or "").strip()
    n = _norm(s)
    g = ml = 0.0
    nombre = n
    m = _CANT_G.match(n)
    if m:
        g, nombre = float(m.group(1).replace(",", ".")), m.group(2)
    else:
        m = _CANT_ML.match(n)
        if m:
            ml, nombre = float(m.group(1).replace(",", ".")), m.group(2)
        else:
            m = _CANT_L.match(n)
            if m:
                ml, nombre = float(m.group(1).replace(",", ".")) * 1000, m.group(2)
            else:
                nombre = re.sub(r"^[\d.,/½¼¾⅓⅔]+\s*(?:unidades?|tazas?|cdas?|cdtas?|cucharadas?|cucharaditas?|pizca|rebanadas?|lonjas?)?\s*(?:de\s+)?", "", n)
    mp = _PARENT_G.search(s)
    if mp and not g:
        g = float(mp.group(1).replace(",", "."))
        nombre = re.sub(r"\(.*?\)", "", nombre).strip()
    return nombre.strip(), g, ml


def componentes(meal) -> dict:
    """Principal (la línea con más gramos que no es pan ni líquido), soporte (pan/base), líquidos (ml), sólidos (g) y
    vegetales de agua presentes. Sólo cuenta lo que la lista declara en g/ml: una pieza sin gramos no inventa peso."""
    out = {"principal": None, "soporte": None, "liquidos_ml": 0.0, "solidos_g": 0.0, "vegetales_agua": [], "lineas": 0}
    try:
        lineas = [(nm, g, ml) for nm, g, ml in (_linea(i) for i in ((meal or {}).get("ingredients") or [])) if nm]
        out["lineas"] = len(lineas)
        mejor = None
        for nm, g, ml in lineas:
            if _VEG_AGUA_RE.search(nm) and nm not in out["vegetales_agua"]:
                out["vegetales_agua"].append(nm)
            if _LIQUIDO_RE.search(nm) and (ml or g):
                out["liquidos_ml"] += ml if ml else g          # la leche en gramos pesa ~ lo que mide
                continue
            if ml and not g:
                out["liquidos_ml"] += ml
                continue
            out["solidos_g"] += g
            if _es_pan_378(nm) and g:                    # [P1-PLAN-LOTE-378] «almendras tostadas» no es pan
                if out["soporte"] is None or g > out["soporte"][1]:
                    out["soporte"] = (nm, g)
                continue
            if g and (mejor is None or g > mejor[1]):
                mejor = (nm, g)
        out["principal"] = mejor
    except Exception:
        return out
    return out


def relaciones(meal, comp: Optional[dict] = None, fam: Optional[str] = None) -> list:
    """Los tres hallazgos mecanizables. Cada uno lleva su evidencia; sin cifras suficientes no se acusa."""
    out = []
    try:
        fam = fam or familia(meal)
        comp = comp or componentes(meal)
        n = _norm((meal or {}).get("name"))
        pasos = _norm(" . ".join(str(p) for p in ((meal or {}).get("recipe") or [])))
        ings = _norm(" . ".join(str(i) for i in ((meal or {}).get("ingredients") or [])))
        if fam == "batido_crema" and _PROMESA_ESPESA_RE.search(n + " " + pasos) and comp["liquidos_ml"] >= 150:
            ratio = comp["solidos_g"] / comp["liquidos_ml"] if comp["liquidos_ml"] else None
            espesante_en_lista = any(_ESPESANTE_RE.search(_linea(i)[0]) and not _linea(i)[1] for i in ((meal or {}).get("ingredients") or []))
            if (ratio is not None and ratio < UMBRALES["crema_solidos_por_ml_min"] and not _PROCESO_ESPESA_RE.search(pasos)
                    and not espesante_en_lista and not _CREMA_APARTE_RE.search(pasos)):
                out.append({"tipo": "crema_sin_espesante",
                            "detalle": (f"promete espesor con {comp['solidos_g']:g} g de sólidos en {comp['liquidos_ml']:g} ml "
                                        f"({ratio:.2f} g/ml, umbral {UMBRALES['crema_solidos_por_ml_min']}) y ningún paso reduce ni espesa"),
                            "evidencia": EVIDENCIA["crema_sin_espesante"]})
        if (fam == "tostada_wrap" and comp["soporte"] and comp["soporte"][1] > 0 and not _ABIERTA_RE.search(n)  # [P1-PLAN-LOTE-344]
                and _soporte_del_nombre_378(n, comp["soporte"][0])):                     # [P1-PLAN-LOTE-378]
            relleno = comp["solidos_g"] - comp["soporte"][1]
            ratio = relleno / comp["soporte"][1]
            if ratio > UMBRALES["wrap_relleno_por_pan_max"] and not _WRAP_APARTE_RE.search(pasos):
                out.append({"tipo": "wrap_desproporcionado",
                            "detalle": (f"{relleno:g} g de relleno para {comp['soporte'][1]:g} g de {comp['soporte'][0]} "
                                        f"({ratio:.1f}×, umbral {UMBRALES['wrap_relleno_por_pan_max']:g}×)"),
                            "evidencia": EVIDENCIA["wrap_desproporcionado"]})
        if fam == "tortilla_revuelto" and comp["vegetales_agua"] and _HUEVO_RE.search(ings):
            dentro = [v for v in comp["vegetales_agua"] if _veg_va_dentro(v, pasos)]
            i_huevo = _posicion_huevo_cocinado(pasos)
            prep_antes = not dentro
            for m in _VEG_PREP_RE.finditer(pasos):
                if i_huevo < 0 or m.start() < i_huevo or "antes" in pasos[max(0, m.start() - 40):m.start()]:
                    prep_antes = True
                    break
            if not prep_antes:
                out.append({"tipo": "tortilla_vegetales_crudos",
                            "detalle": (f"lleva {', '.join(comp['vegetales_agua'][:3])} y ningún paso los sofríe, saltea o escurre "
                                        f"antes del huevo: la tortilla suelta agua"),
                            "evidencia": EVIDENCIA["tortilla_vegetales_crudos"]})
    except Exception:
        return out
    return out


#: [P1-PLAN-LOTE-29] lo servido APARTE con intención (el resto del relleno como ensalada; el líquido sobrante como bebida) no
#: es un defecto de estructura: es la reparación de `recipe_repair`, y reconocerla es lo que la hace idempotente.
# [P1-PLAN-LOTE-344 · 2026-09-26] Una TOSTADA es un plato abierto: el topping va encima y no hay nada que «cerrar». La
# bariátrica de la batería del 25-sep recibía «Montaje: rellena la pan integral 30 g con lo que cierra… un wrap que se
# puede cerrar» sobre una «Tostada Integral con Hummus y Tomate». Solo el sustantivo «tostada(s)» (no el adjetivo
# «tostado»: un «wrap tostado» sigue siendo un wrap). tooltip-anchor: P1-PLAN-LOTE-344
_ABIERTA_RE = re.compile(r"\btostadas?\b|\bbruschettas?\b|\bcrostinis?\b")
_WRAP_APARTE_RE = re.compile(r"resto del relleno|relleno[^.]{0,40}\bal lado\b|como ensalada")
_CREMA_APARTE_RE = re.compile(r"\d+\s*ml[^.]{0,40}\brestantes\b|como bebida|ml[^.]{0,30}\bal lado\b")
_HUEVO_COCINA_RE = re.compile(r"\b(vierte|anade|agrega|incorpora|echa|cuaja|revuelve|mezcla|pon|vuelca)\b[^.]{0,50}?\b(huevos?|claras?)\b|"
                              r"\b(huevos?|claras?)\b[^.]{0,40}?\b(en la sarten|a la sarten|al sarten|cuaj\w+|revuelv\w+)")
_VEG_FUERA_RE = re.compile(r"\b(ensalada|alina\w*|acompana\w*|sirve con|al lado|de guarnicion|aparte)\b")


def _posicion_huevo_cocinado(pasos_norm: str) -> int:
    """Dónde el huevo entra al fuego («vierte el huevo», «cuaja»); si no se dice, la ÚLTIMA mención (la del mise en place
    no cuenta: batir el huevo en un bol no es cocinarlo)."""
    m = _HUEVO_COCINA_RE.search(pasos_norm)
    if m:
        return m.start()
    ult = [x.start() for x in _HUEVO_RE.finditer(pasos_norm)]
    return ult[-1] if ult else -1


def _veg_va_dentro(veg: str, pasos_norm: str) -> bool:
    """False si el vegetal sólo aparece en cláusulas de ensalada/acompañamiento («aliña el repollo», «sirve con»)."""
    cabeza = veg.split()[0]
    vistas = 0
    from culinary_coherence import clause_bounds        # [P1-PLAN-LOTE-52] la MISMA frontera: «0.5 taza» no parte la oración
    for c_ini, c_fin in clause_bounds(pasos_norm):
        cl = pasos_norm[c_ini:c_fin]
        if re.search(r"\b" + re.escape(cabeza), cl):
            vistas += 1
            if not _VEG_FUERA_RE.search(cl):
                return True
    return vistas == 0      # no se nombra en los pasos: se asume dentro (la lista lo trae para la tortilla)


def contract(meal) -> dict:
    """El contrato ligero del plato, con confianza: `alta` si la lista trae gramos/ml en ≥ 2/3 de sus líneas, `media` si
    en la mitad, `baja` si menos (una pieza sin gramos no pesa nada aquí y las relaciones lo saben)."""
    try:
        fam = familia(meal)
        comp = componentes(meal)
        con_cifra = sum(1 for i in ((meal or {}).get("ingredients") or []) if _linea(i)[1] or _linea(i)[2])
        n = comp["lineas"] or 1
        conf = "alta" if con_cifra >= 2 * n / 3 else ("media" if con_cifra >= n / 2 else "baja")
        return {"familia": fam, "componentes": comp, "relaciones": relaciones(meal, comp, fam), "confianza": conf,
                "fuente": "nombre+lista+pasos", "umbrales": dict(UMBRALES)}
    except Exception:
        return {"familia": "otro", "componentes": {}, "relaciones": [], "confianza": "baja", "fuente": "error"}



# ── [P1-PLAN-LOTE-378 · 2026-09-26] El wrap es el plato que se ENVUELVE ─────────────────────────────────────────────────
# Batería REAL sobre el 376 (perfil del dueño, día 3): «Wok rápido de pollo y auyama con tortilla integral» recibía un
# SEGUNDO Montaje —«rellena la casabe con lo que cierra (unos 150 g del relleno)… un wrap que se puede cerrar»—. Tres
# fallos del detector del lote 27: (1) una «tortilla integral/de trigo/de maíz» en CUALQUIER lugar del nombre hacía wrap
# al plato, también cuando ACOMPAÑA («… con tortilla integral», «… y tortillas de maíz») o es la BASE («… sobre tortilla
# integral»): sólo es la vasija cuando el nombre EMPIEZA por ella o la dice con «en»/«de» («… en tortilla integral»,
# «Wrap de tortilla integral»); si acompaña, el plato es lo demás («Revoltillo … con tortilla integral» es un revoltillo);
# (2) el soporte era el pan con más gramos de la lista —el casabe que el cerrador añadió de guarnición—, no el pan que el
# nombre dice; (3) «almendras tostadas» y «soya tostada» contaban como pan (y como tostada del nombre). En el corpus de
# 315 planes hay 15 reparaciones así y 14 van sobre algo que no se cierra. tooltip-anchor: P1-PLAN-LOTE-378
_VASIJA_378 = ("en", "de")
_ADJ_PAN_378_RE = re.compile(r"^\s*(?:tostad[ao]s?|dorad[ao]s?|tibi[ao]s?|calientes?|crujientes?|horneadas?)\b")
_ADJ_TOSTADA_378_RE = re.compile(r"\b(?!(?:pan|tortillas?|casabe|arepas?)\b)[a-z]+\s+tostad[ao]s?\b")
_PAN_NOMBRE_378_RE = re.compile(r"\b(pan|tortillas?|pita|casabe|arepas?)\b")


def _nombre_de_familia_378(n: str):
    """`(es_wrap, nombre_sin_acompañantes)`: la tortilla-pan es la vasija sólo al empezar el nombre o tras «en»/«de»; si
    acompaña o es la base, sale del nombre (con su adjetivo) y el plato es lo demás. «X tostada(s)» con X que no es pan
    tampoco es una tostada."""
    while True:
        m = _TORTILLA_PAN_RE.search(n)
        if not m:
            break
        antes = n[:m.start()].split()
        if not antes or antes[-1] in _VASIJA_378:
            return True, n
        n = n[:m.start()] + " " + _ADJ_PAN_378_RE.sub(" ", n[m.end():])
    return False, _ADJ_TOSTADA_378_RE.sub(" ", n)


def _es_pan_378(nm: str) -> bool:
    """Una línea es pan si nombra uno; «tostada(s)» sólo cuando ES la línea («tostadas integrales»), no su adjetivo
    («almendras tostadas», «soya tostada»)."""
    for m in _PAN_RE.finditer(str(nm or "")):
        if m.group(1).startswith("tostada") and m.start() > 0:
            continue
        return True
    return False


def _soporte_del_nombre_378(n: str, soporte: str) -> bool:
    """Si el nombre dice su pan, el soporte ES ese pan; sin nombrar pan («Wrap de pollo»), vale cualquiera."""
    nombrados = {w[:-1] if w.endswith("s") else w for w in _PAN_NOMBRE_378_RE.findall(str(n or ""))}
    return not nombrados or any(re.search(r"\b" + re.escape(w), str(soporte or "")) for w in nombrados)
