# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-806 · 2026-09-29] El huevo que ningún paso cuaja recibe su paso de cocción, no sólo una nota.

Batería real (embarazo, 29-sep, con 802-805 desplegados): «Tortilla bien cuajada con tomate y aguacate, guayaba fresca,
yogurt griego entero» — «bate 3 huevos y 1 clara de huevo… El Toque de Fuego: calienta el aceite…; cocina el tomate, la
cebolla y el ajo 3 minutos. Añade huevos al lado para acompañar. Montaje: sirve la tortilla…». Ningún paso cuajaba los
huevos. El detector de seguridad del huevo (`_scan_raw_egg_violations`, P3-FOOD-SAFETY) SÍ lo vio (`no_cook`), pero para
el huevo «cocinable» su mitigación era sólo la nota «⚠️ cocina el huevo por completo»: a una embarazada le llegaba una
tortilla sin instrucción de cuajarla. V7f no ayudaba: daba el huevo por cocido porque «bate» lo mete en una «mezcla» y
cualquier cocción posterior (el sofrito) se le acreditaba — y endurecer V7f rompía revoltillos buenos («remueve hasta que
cuajen»), medido sobre 4.969 veredictos.

Aquí, cuando el detector dice `no_cook` (y no es una masa: panqueques, arepitas…), se comprueba con un criterio ESTRICTO
si algún paso cocina el huevo de verdad —una cláusula que nombra el huevo con un verbo de cocción, o que lo cuaja— y, si
ninguno lo hace, se inserta antes del Montaje «💪 Vierte los huevos batidos en la sartén… hasta que cuajen por completo».
El criterio estricto evita el falso positivo del detector (sobre-detecta a propósito: con una nota daba igual; con un
paso, duplicaría la cocción). Knob `MEALFIT_RAW_EGG_COOK_STEP` (True). tooltip-anchor: P1-PLAN-LOTE-806
"""
from __future__ import annotations

import re
import unicodedata

_HUEVO_RE = re.compile(r"\b(?:huevos?|claras?|yemas?|revoltillo|batidos?\s+de\s+huevo)\b")
_COCCION_RE = re.compile(r"\b(?:cocin\w*|cuec\w*|coce\w*|cocid[oa]s?|hierv\w*|hirv\w*|herv\w*|dur[oa]s?|fri[eo]\w*|frei\w*|"
                         r"horne\w*|horno|plancha|sarten|saltea\w*|sofri\w*|dora\w*|revuelv\w*|revolv\w*|escalfa\w*|"
                         r"pocha\w*|airfryer|vapor|microondas|asa\b|asal\w*|guisa\w*|wok)")
_CUAJA_RE = re.compile(r"\bcuaj\w*")
#: «cocínalas», «fríelos», «hornéalas»: la cocción de lo que nombró la cláusula anterior
_ENCLITICO_RE = re.compile(r"\b(?:cocin|cuec|hierv|fri|horne|salte|dor|sofri|asa|guis)\w*(?:lo|la|los|las)\b")
#: el huevo entra en una preparación (masa, mezcla, tortitas…) que luego se cuece entera
_MEZCLA_RE = re.compile(r"\b(?:mezcl\w*|integr(?!al)\w*|incorpor\w*|combin\w*|une\b|amas\w*|masa|tortitas?|bocaditos?|"
                        r"croquetas?|arepitas?|panqueques?|bollitos?|albondigas?|hamburguesas?|empanad\w*)")
#: notas que NO son pasos (seguridad, clínica); la «💡 Cocción previa» SÍ es un paso de cocción
_NOTA_RE = re.compile(r"^\s*(?:⚠|🤰|⚕|🌱|🛡|nota\b|💡(?!\s*cocci[oó]n\s+previa))", re.IGNORECASE)


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_RAW_EGG_COOK_STEP", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def cocido_en_pasos(meal: dict) -> bool:
    """¿Algún paso (no nota) cocina el huevo? Una cláusula que lo cuaja; o que nombra el huevo con un verbo o un estado de
    cocción («2 huevos bien cocidos (10 minutos en agua hirviendo)»); o que lo retoma con un pronombre justo después de
    nombrarlo («sazona las claras…; cocínalas en la sartén»); o el huevo entró en una mezcla/masa y un paso posterior
    cuece («mezcla el pescado con el huevo…; forma tortitas y hornéalas»)."""
    en_mezcla = previa_huevo = False
    for p in (meal.get("recipe") or []):
        if not isinstance(p, str) or _NOTA_RE.search(p):
            continue
        for cl in re.split(r"(?<=[.;])\s+", _sa(p)):   # [P1-PLAN-LOTE-52] con espacio: no corta «1.5 tazas»
            huevo = bool(_HUEVO_RE.search(cl))
            if _CUAJA_RE.search(cl) or (huevo and _COCCION_RE.search(cl)):
                return True
            if (previa_huevo and _ENCLITICO_RE.search(cl)) or (en_mezcla and _COCCION_RE.search(cl)):
                return True
            if huevo and _MEZCLA_RE.search(cl):
                en_mezcla = True
            previa_huevo = huevo
    return False


def _cuenta(lista: str, rx: str) -> int:
    n = 0
    for m in re.finditer(r"(\d+)\s+" + rx, lista):
        n += int(m.group(1))
    return n


def paso(meal: dict) -> str:
    """El paso de cocción con lo que la lista compra: «los huevos», «la clara», «los huevos y las claras»."""
    lista = " ; ".join(_sa(x) for x in (meal.get("ingredients") or []) if isinstance(x, str))
    n_cl = _cuenta(lista, r"claras?\b")
    sin_formas = re.sub(r"(?:claras?|yemas?)\s+de\s+huevos?", " ", lista)
    n_en = _cuenta(sin_formas, r"huevos?\b") or (1 if re.search(r"\bhuevos?\b", sin_formas) else 0)
    partes = []
    if n_en:
        partes.append("el huevo" if n_en == 1 else "los huevos")
    if n_cl:
        partes.append("la clara" if n_cl == 1 else "las claras")
    if not partes:
        partes = ["los huevos"]
    objeto = " y ".join(partes)
    fem = partes == ["la clara"] or partes == ["las claras"]
    plural = len(partes) > 1 or objeto.startswith(("los ", "las "))
    suf = ("as" if fem else "os") if plural else ("a" if fem else "o")
    pron = ("las" if fem else "los") if plural else ("la" if fem else "lo")
    mise = " ".join(_sa(p) for p in (meal.get("recipe") or []) if isinstance(p, str))
    if re.search(r"\bbat[eiao]\w*", mise):
        return (f"💪 Vierte {objeto} batid{suf} en la sartén caliente con un poco de aceite y cocína{pron}, removiendo, "
                f"3-4 minutos, hasta que cuajen por completo (sin partes líquidas).")
    return (f"💪 Bate {objeto}, viérte{pron} en la sartén caliente con un poco de aceite y cocína{pron}, removiendo, "
            f"3-4 minutos, hasta que cuajen por completo (sin partes líquidas).")


def necesita_paso(meal: dict) -> bool:
    return on() and isinstance(meal, dict) and not cocido_en_pasos(meal)
