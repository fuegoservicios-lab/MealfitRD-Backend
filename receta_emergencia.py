# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-446 · 2026-09-27] Los pasos del plan de EMERGENCIA (el día determinista que sustituye a un día que la IA
no pudo generar): cocinan lo que la lista compra, con su nombre y su punto, y no cocinan lo que se come crudo.

Antes vivían en `graph_orchestrator._fallback_recipe_steps` (P2-FALLBACK-RECIPE-SLOT-TEMPLATE) con tres defectos, vistos
en el replay de 322 planes: (1) «Huevos y Avena» (16 desayunos) entraba por la rama de la avena y sus 2 huevos —«Huevos
revueltos con avena cocida», dice la plantilla— no los cocinaba ningún paso; la nota ⚠️ «cocina el huevo por completo»
hablaba de un paso que no existía. (2) Toda proteína iba a «sazona la proteína… a la plancha 6-8 minutos por lado»: el
pescado 12-16 minutos, el «atún en agua escurrido» de la plantilla bariátrica a la plancha, el «pollo guisado» a la
plancha, sin nombre ni punto de cocción, y «el víver/carbohidrato», «la ensalada/vegetales» con barras. (3) Lo demás caía
en «cocina los ingredientes principales a fuego medio 10-12 minutos»: el yogur griego, la fruta, el queso con manzana y
la ensalada de aguacate, «cocinados» 10-12 minutos. Aquí cada proteína se cocina con su método y su punto (ave 74 °C,
carne 71 °C, pescado 63 °C), lo enlatado se escurre, lo listo para comer no se cocina, los acompañamientos se nombran y
la avena cocida se calienta (como el lote 428). Contrato de siempre: Mise en place / El Toque de Fuego / Montaje.
tooltip-anchor: P1-PLAN-LOTE-446
"""
from __future__ import annotations

import re
import unicodedata

_MISE = "Mise en place: lava, pica y pesa cada ingrediente según las cantidades listadas."
_GENERICO = [_MISE, "El Toque de Fuego: cocina los ingredientes principales a fuego medio 10-12 minutos hasta que estén "
                    "tiernos.", "Montaje: sirve y ajusta sal al gusto."]
_CANTIDAD_RE = re.compile(r"^\s*(?:[\d½¼¾⅓⅔/.,]+\s*(?:g|gr|kg|ml|unidades?|tazas?|cdas?|cdtas?)?\s*(?:de\s+)?)?")
_PROT_RE = re.compile(r"\b(pollo|pavo|res|cerdo|pescado|tilapia|mero|atun|sardinas?|camaron(?:es)?)\b")
_ENLATADO_RE = re.compile(r"\s+(?:en agua|en lata|enlatad\w*|escurrid\w*)\b.*$")
#: Lo que se come como viene: no pasa por el fuego.
_LISTO_RE = re.compile(r"\b(?:yogur\w*|queso\w*|frutas?|manzanas?|fresas?|guineos?|lechosa|mango|pina|aguacate|semillas?|"
                       r"chia|casabe|agua|aceite|ensalada|vegetales variados|nuez|nueces|almendras?|mani)\b")
_FRUTA_RE = re.compile(r"\b(?:frutas?|manzanas?|fresas?|guineos?|lechosa|mango|pina|aguacate)\b")
_FEM = {"carne", "leche"}


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _nombre(ing: str) -> str:
    """«150g mero a la plancha» → «mero a la plancha»; «1/2 taza de avena cocida» → «avena cocida»."""
    return _CANTIDAD_RE.sub("", str(ing or "")).strip()


def _gn(n: str) -> tuple:
    import pasos_cerrador as pc
    pl, fem = pc._genero_numero(n)
    return pl, fem or (_sa(n).split() or [""])[0] in _FEM


def _art(n: str) -> str:
    pl, fem = _gn(n)
    return ("las" if fem else "los") if pl else ("la" if fem else "el")


def _pron(n: str) -> str:
    pl, fem = _gn(n)
    return ("las" if fem else "los") if pl else ("la" if fem else "lo")


def _tiern(n: str) -> str:
    pl, fem = _gn(n)
    return ("estén " if pl else "esté ") + "tiern" + (("as" if fem else "os") if pl else ("a" if fem else "o"))


def _x(n: str) -> str:
    return f"{_art(n)} {n}"


def _lista(xs: list) -> str:
    xs = [x for x in xs if x]
    return xs[0] if len(xs) == 1 else (", ".join(xs[:-1]) + " y " + xs[-1]) if xs else ""


def _proteina(linea: str) -> tuple:
    """(frase de cocción, nombre con artículo) de la proteína de `linea`."""
    n = _nombre(linea)
    t = _sa(n)
    if _ENLATADO_RE.search(n):
        b = _ENLATADO_RE.sub("", n).strip()
        return f"escurre bien {_x(b)} y desmenúza{_pron(b)} con un tenedor", _x(b)
    b = re.sub(r"\s+(?:a la plancha|al horno|guisad[oa]s?|magr[oa]s?)\b.*$", "", n).strip() or n
    xb, pr = _x(b), _pron(b)
    if re.search(r"\b(?:pollo|pavo)\b", t):
        if "guisad" in t:
            return (f"guisa {xb} con cebolla y ajo a fuego medio-bajo, tapado, 20-25 minutos, hasta que alcance 74 °C "
                    f"por dentro", xb)
        return (f"sazona {xb} con sal, ajo y orégano y cocína{pr} a la plancha a fuego medio-alto 6-7 minutos por lado, "
                f"hasta que alcance 74 °C en la parte más gruesa", xb)
    if re.search(r"\b(?:res|cerdo)\b", t):
        return (f"sazona {xb} con sal, ajo y orégano y cocína{pr} a la plancha a fuego medio-alto 4-5 minutos por lado, "
                f"hasta que no quede rosada por dentro (71 °C al centro)", xb)
    if re.search(r"\bcamaron", t):
        return f"saltea {xb} con unas gotas de aceite 2-3 minutos por lado, hasta que estén rosados y opacos", xb
    if "horno" in t:
        return f"hornea {xb} a 200 °C 12-15 minutos, hasta que se desmenuce fácilmente (63 °C al centro)", xb
    return (f"sazona {xb} con sal, ajo y limón y cocína{pr} a la plancha a fuego medio 3-4 minutos por lado, hasta que "
            f"se desmenuce fácilmente (63 °C al centro)", xb)


def _item(linea: str) -> tuple:
    """(frase de cocción o «», nombre con artículo) de un ingrediente que no es la proteína."""
    n = _nombre(linea)
    t = _sa(n)
    if _LISTO_RE.search(t):
        return "", _x(n)
    if re.search(r"\b(?:batata|papa|yuca)\s+asad", t):
        b = re.sub(r"\s+asad[oa]s?\b", "", n)
        return f"asa {_x(b)} en el horno a 200 °C 25-30 minutos, hasta que el cuchillo entre sin fuerza", _x(n)
    if re.search(r"\bal vapor\b", t):
        return f"cocina {_x(n)} 5-7 minutos, hasta que {_tiern(n)}", _x(n)
    if re.search(r"\b(?:cocid[oa]s?|asad[oa]s?|guisad[oa]s?)\b", t):
        return f"calienta {_x(n)} a fuego medio 2-3 minutos, removiendo", _x(n)
    if re.search(r"\barroz\b", t):
        return f"cocina {_x(n)} en agua con sal 15-20 minutos, hasta que esté tierno y suelto", _x(n)
    if re.search(r"\bsaltead", t):
        b = re.sub(r"\s+saltead[oa]s?\b", "", n)
        return f"saltea {_x(b)} con unas gotas de aceite 3-4 minutos", _x(b)
    if re.search(r"\b(?:vegetales|brocoli|vainitas|zanahoria|tayota|berenjena)\b", t):
        return f"saltea {_x(n)} con unas gotas de aceite 3-4 minutos", _x(n)
    return "", _x(n)


def _sin_fuego(items: list, lineas: list) -> str:
    frutas = [x for (f, x), l in zip(items, lineas) if _FRUTA_RE.search(_sa(l))]
    return f"El Toque de Fuego: no lleva cocción; corta {_lista(frutas)} en trozos." if frutas else \
        "El Toque de Fuego: no lleva cocción; sírvelo frío."


def pasos(meal_type: str, ingredients: list) -> list:
    """Los tres pasos de la receta de emergencia para `ingredients`. Fail-safe: el genérico de siempre."""
    try:
        ings = [str(i) for i in (ingredients or [])]
        txt = _sa(" ".join(ings))
        # avena ANTES del heurístico de batido: avena+leche es AVENA COCIDA, no un licuado.
        if "avena" in txt:
            av = next((i for i in ings if "avena" in _sa(i)), "")
            if re.search(r"\bavena\s+cocida\b", _sa(av)):
                avena = "calienta la avena cocida a fuego medio 2-3 minutos, removiendo hasta que esté cremosa"
            else:
                avena = "cocina la avena con el líquido a fuego medio 5 minutos, removiendo hasta cremosa"
            encima = _lista([_x(_nombre(i)) for i in ings if not re.search(r"\b(?:avena|huevos?|agua|leche)\b", _sa(i))])
            if "huevo" in txt:
                return [_MISE, ("El Toque de Fuego: bate los huevos con una pizca de sal y cuájalos en una sartén con unas "
                                "gotas de aceite a fuego medio 3-4 minutos, removiendo, hasta que yema y clara estén "
                                f"firmes. Aparte, {avena}."),
                        "Montaje: sirve la avena en un bowl" + (f", corónala con {encima}" if encima else "")
                        + " y acompaña con los huevos revueltos."]
            return [_MISE, f"El Toque de Fuego: {avena}.",
                    "Montaje: sirve la avena en un bowl" + (f" y corónala con {encima}." if encima else ".")]
        if any(t in txt for t in ("batido", "licuado")) or (
                any(t in txt for t in ("yogur", "leche")) and any(t in txt for t in ("guineo", "fruta", "fresa", "mango"))):
            todo = _lista([_x(_nombre(i)) for i in ings if not re.search(r"\b(?:batido|licuado)\b", _sa(i))])
            return [_MISE, f"El Toque de Fuego: licúa {todo or 'todos los ingredientes'} 1 minuto a velocidad alta hasta "
                           f"quedar homogéneo.", "Montaje: sirve frío de inmediato."]
        if "huevo" in txt:
            # lo que acompaña va primero a la sartén (salteado): un revoltillo no suelta agua (dish_structure, lote 29)
            otros = [i for i in ings if "huevo" not in _sa(i)]
            antes = [f"saltea {_x(_nombre(i))} con unas gotas de aceite 2-3 minutos" for i in otros
                     if not _LISTO_RE.search(_sa(_nombre(i)))]
            huevos = ("bate los huevos y cuájalos en " + ("la misma sartén" if antes else "una sartén con unas gotas de aceite")
                      + " a fuego medio 3-4 minutos, hasta que yema y clara estén firmes")
            return [_MISE, "El Toque de Fuego: " + "; ".join(antes + [huevos]) + ".",
                    "Montaje: sirve los huevos" + (f" con {_lista([_x(_nombre(i)) for i in otros])}" if otros else "") + "."]
        prot = next((i for i in ings if _PROT_RE.search(_sa(i))), None)
        if prot:
            coccion, xp = _proteina(prot)
            otros = [_item(i) for i in ings if i is not prot]
            fuego = "; ".join([coccion] + [f for f, _ in otros if f])
            return [_MISE, f"El Toque de Fuego: {fuego}.",
                    f"Montaje: sirve {xp}" + (f" con {_lista([x for _, x in otros])}" if otros else "") + "."]
        comida = [i for i in ings if not re.search(r"^(?:agua|aceite\b.*)$", _sa(_nombre(i)))]
        items = [_item(i) for i in comida]
        fuego = "; ".join(f for f, _ in items if f)
        toque = f"El Toque de Fuego: {fuego}." if fuego else _sin_fuego(items, comida)
        montaje = f"Montaje: sirve {_lista([x for _, x in items])}"
        if any(_sa(_nombre(i)).startswith("aceite") for i in ings):
            montaje += "; aliña con el aceite de oliva"
        if any(_sa(_nombre(i)) == "agua" for i in ings):
            montaje += "; acompaña con agua"
        return [_MISE, toque, montaje + "."]
    except Exception:
        return list(_GENERICO)
