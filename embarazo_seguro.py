# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-172 · 2026-09-23] Embarazo y lactancia: lo que el revisor médico tiene que LEER para aprobar.

Batería real del generador (perfil «Embarazo», RD): el revisor médico rechazó el plan dos veces con severidad CRÍTICA —
«el pescado no está identificado por especie, no se puede verificar que sea bajo en mercurio» y «el queso blanco fresco
no está confirmado como pasteurizado»— y el usuario recibió el plan de EMERGENCIA («Huevos y Avena», «Pollo y Arroz»,
kcal −10/−14 % y la lista de compras VACÍA). El revisor tenía razón: en embarazo la especie del pescado y la
pasteurización del queso fresco son la regla de seguridad, no un detalle. Y la sustitución de pescados altos en
mercurio de `condition_rules` los cambiaba por «Filete de pescado blanco»: un nombre GENÉRICO, justo lo que el revisor
rechaza.

Este pase escribe lo que el plan ya debía decir, sin cambiar el plato:
  · pescado genérico («pescado», «pescado blanco», «filete(s) de pescado [blanco]») → tilapia, especie baja en mercurio
    que el catálogo resuelve («Tilapia», alias «filete de tilapia»), en `ingredients` y en `ingredients_raw` (la compra);
    y en el nombre del plato. Los pasos no cambian de pescado: la lista ya identifica la especie.
  · queso fresco/blando sin la palabra «pasteurizado» → se le añade. Los quesos curados/duros no la necesitan.
  · [P1-PLAN-LOTE-173] también la LECHE de origen animal, y el queso en los PASOS (el revisor leía «desmenuza el queso
    blanco fresco» en la receta aunque la lista dijera «pasteurizado»). Lo llama `etiquetas_clinicas.etiquetar`, al
    sustituir por condición y AL FINAL del escudo: el cerrador de proteína añade el cottage después de la sustitución.

  · [P1-PLAN-LOTE-175] el atún «claro» (la especie baja en mercurio), el edamame «cocido» y, si hay yuca, el paso de
    hervirla del todo y botar el agua. Y la etiqueta corre también JUSTO antes del revisor (`review_plan_node`): en la
    batería del 23-sep dos planes (embarazo y lactancia) perdieron un intento entero por queso sin «pasteurizado» que
    un paso posterior a la sustitución clínica había vuelto a escribir.

Sólo con la regla `pregnancy` activa (`condition_rules.detect_active_rules`). Idempotente. Knob
`MEALFIT_PREGNANCY_LABELS` (True). tooltip-anchor: P1-PLAN-LOTE-172-EMBARAZO-ETIQUETAS
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# «filete(s) de pescado [blanco] [fresco]», «pescado blanco [fresco]» o «pescado» a secas; NO «pescado azul», NO si ya
# nombra especie («pescado (tilapia)», «pescado de mar»…: lo que venga detrás que no sea blanco/fresco lo deja estar).
_PESCADO_LINEA = re.compile(r"\b(filetes?\s+de\s+)?pescado(?:\s+blanco)?(?:\s+fresco)?\b(?!\s+(?:azul|\())", re.IGNORECASE)
_PESCADO_NOMBRE = re.compile(r"\b(?:filetes?\s+de\s+)?pescado(?:\s+blanco)?\b(?!\s+(?:azul|\())", re.IGNORECASE)
# Adjetivos que siguen al pescado en un nombre y cambian de género con «tilapia».
_ADJ_FEM = {"dorado": "dorada", "horneado": "horneada", "asado": "asada", "guisado": "guisada", "frito": "frita",
            "empanizado": "empanizada", "apanado": "apanada", "sellado": "sellada", "marinado": "marinada",
            "gratinado": "gratinada", "desmenuzado": "desmenuzada", "cocido": "cocida", "tierno": "tierna",
            "jugoso": "jugosa", "criollo": "criolla", "blanco": "blanca", "fresco": "fresca", "ligero": "ligera"}
_QUESO_FRESCO = re.compile(r"\b(?:queso(?!\s+(?:cheddar|parmesano|gouda|provolone|edam|de\s+papa|de\s+bola|amarillo|"
                           r"suizo|manchego|curado|azul))|ricotta|cottage|reques[oó]n|mozzarella|"
                           # [P1-PLAN-LOTE-173] el yogur también: el revisor lo pidió a una madre lactante
                           r"yogur(?:t)?(?!\s+de\s+(?:coco|soya|soja|almendras?)))\b[^,;()]*", re.IGNORECASE)


def enabled() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_PREGNANCY_LABELS", True)
    except Exception:                                                          # noqa: BLE001
        return True


def aplica(form_data) -> bool:
    try:
        from condition_rules import detect_active_rules
        return any(getattr(r, "id", "") == "pregnancy" for r in detect_active_rules(form_data or {}))
    except Exception:                                                          # noqa: BLE001
        return False


# [P1-PLAN-LOTE-173] El mero (FDA «buena elección», una ración a la semana) junto al atún se lo rechazó CRÍTICO el revisor
# a una madre lactante («mero o tilapia» + 175 g de atún). En embarazo/lactancia el pescado blanco es tilapia.
_MERO = re.compile(r"\bmero(?:\s+o\s+tilapia)?\b|\btilapia\s+o\s+mero\b", re.IGNORECASE)


def _linea_pescado(s: str) -> str:
    def _sub(m):
        return (m.group(1) or "") + "tilapia"
    return _MERO.sub("tilapia", _PESCADO_LINEA.sub(_sub, s))


def _nombre_pescado(nombre: str) -> str:
    m = _PESCADO_NOMBRE.search(nombre)
    if not m:
        return nombre
    cab = "Tilapia" if m.start() == 0 else "tilapia"
    if m.group(0).lower().startswith("filete"):
        cab = ("Filete" if m.group(0)[0].isupper() else "filete") + (" de tilapia" if not m.group(0).lower().startswith("filetes") else "s de tilapia")
        return nombre[:m.start()] + cab + nombre[m.end():]
    resto = nombre[m.end():]
    ma = re.match(r"(\s+)([a-záéíóúñ]+)", resto)
    if ma and ma.group(2).lower() in _ADJ_FEM:
        resto = ma.group(1) + _ADJ_FEM[ma.group(2).lower()] + resto[ma.end():]
    return nombre[:m.start()] + cab + resto


def _linea_queso(s: str) -> str:
    if "pasteuriz" in s.lower():
        return s
    m = _QUESO_FRESCO.search(s)
    if not m:
        return s
    fin = m.end()
    while fin > m.start() and s[fin - 1] == " ":
        fin -= 1
    return s[:fin] + _concuerda(m.group(0)) + s[fin:]


# [P1-PLAN-LOTE-173] La leche de origen animal también (el revisor: «no se especifica que la leche ni el queso sean
# pasteurizados»). Las bebidas vegetales, la evaporada, la condensada y la de polvo no la necesitan.
_LECHE_LINEA = re.compile(r"\bleche\b(?!\s+(?:de\s+(?:coco|almendras?|soya|soja|avena|arroz|cabra)\b|evaporada|condensada|"
                          r"en\s+polvo|materna))[^,;()]*", re.IGNORECASE)


def _linea_leche(s: str) -> str:
    if "pasteuriz" in s.lower():
        return s
    m = _LECHE_LINEA.search(s)
    if not m:
        return s
    fin = m.end()
    while fin > m.start() and s[fin - 1] == " ":
        fin -= 1
    return s[:fin] + " pasteurizada" + s[fin:]


# [P1-PLAN-LOTE-173] En los PASOS, sólo la frase del alimento (sin arrastrar el resto de la oración) y en cada mención
# que aún no lo diga: el revisor leía «desmenuza el queso blanco fresco» en la receta aunque la lista dijera «pasteurizado».
_QUESO_PASO = re.compile(r"\b(queso\s+blanco\s+fresco|queso\s+(?:blanco|fresco|de\s+hoja|cottage|ricotta|crema|mozzarella)|"
                         r"ricotta|cottage|reques[oó]n)\b(?!\s+(?:fresco\s+)?pasteuriz)", re.IGNORECASE)


def _concuerda(frase: str) -> str:
    return " pasteurizada" if frase.strip().lower().startswith("ricotta") else " pasteurizado"


def _paso_queso(s: str) -> str:
    return _QUESO_PASO.sub(lambda m: m.group(1) + _concuerda(m.group(1)), s)


# [P1-PLAN-LOTE-175] Yuca en embarazo/lactancia: el revisor rechazó CRÍTICO «no se especifica que la yuca se hierva
# completamente ni que se descarte el agua de cocción» (la yuca mal cocida libera cianuro). El pase de víveres crudos
# (`P1-RAW-VIVER-SAFETY`) sólo actúa si la yuca va CRUDA; aquí se deja escrito el paso aunque la receta ya la hierva.
_YUCA = re.compile(r"\byucas?\b", re.IGNORECASE)
_PASO_YUCA = ("⚠️ Seguridad alimentaria: hierve la yuca hasta que esté completamente blanda por dentro y desecha el agua "
              "de cocción antes de servirla o majarla.")


# [P1-PLAN-LOTE-175] Mismo revisor, embarazo, batería real: «el atún en agua no especifica que sea atún claro (light o
# skipjack)» y «el edamame no se indica como cocido» — CRÍTICO, al segundo intento, y plan de emergencia. El catálogo
# resuelve igual «atún claro en agua» y «edamame cocido» (probado contra el matcher de producción: misma fila, mismas
# macros), así que la etiqueta no mueve ni la compra ni los números.
_ATUN = re.compile(r"\bat[uú]n\b(?!\s+(?:claro|rojo|blanco|fresco|aleta|patudo|albacora))", re.IGNORECASE)
_EDAMAME = re.compile(r"\b(edamames?)\b(?![^,;()]*\b(?:cocid|hervid|al\s+vapor|salteado))", re.IGNORECASE)


def _linea_atun_edamame(s: str) -> str:
    s = _ATUN.sub(lambda m: m.group(0) + " claro", s, count=1)
    return _EDAMAME.sub(lambda m: m.group(1) + (" cocidos" if m.group(1).lower().endswith("s") else " cocido"), s, count=1)


# Sólo la yuca que se HIERVE: la harina, el almidón, el casabe y la yuca rallada de las arepitas no llevan este paso.
_YUCA_PROCESADA = re.compile(r"\b(?:harina|almid[oó]n|casabe)\s+de\s+yucas?\b|\byucas?\s+rallad[ao]s?\b", re.IGNORECASE)
_HIERVE = re.compile(r"\b(?:hierv\w*|herv\w*|sancoch\w*|cuec\w*|cocid[ao]s?|cocin\w*\s+(?:la\s+yuca\s+)?en\s+agua)\b",
                     re.IGNORECASE)


def _paso_yuca(meal: dict) -> bool:
    ings = _YUCA_PROCESADA.sub(" ", " ".join(str(x) for x in (meal.get("ingredients") or [])))
    if not _YUCA.search(ings):
        return False
    pasos = meal.get("recipe")
    if not isinstance(pasos, list):
        return False
    texto = " ".join(str(p) for p in pasos).lower()
    if not _HIERVE.search(texto):
        return False
    if "desecha el agua" in texto or "bota el agua" in texto or "descarta el agua" in texto:
        return False
    pasos.append(_PASO_YUCA)
    return True


# [P1-PLAN-LOTE-177] Yautía/malanga/ñame: el revisor rechazó CRÍTICO a una madre lactante «no se indica que la yautía
# quede bien cocida» (oxalato de calcio: cruda o a medio cocer irrita). Sirve para cualquier técnica; la harina no cuenta.
_YAUTIA = re.compile(r"\b(?:yaut[ií]as?|malangas?|ñames?)\b", re.IGNORECASE)
_YAUTIA_PROCESADA = re.compile(r"\b(?:harina|almid[oó]n)\s+de\s+(?:yaut[ií]a|malanga|ñame)s?\b", re.IGNORECASE)
_PASO_YAUTIA = ("⚠️ Seguridad alimentaria: cocina la yautía (o malanga, o ñame) hasta que esté completamente blanda por "
                "dentro; cruda o a medio cocer irrita la boca y la garganta.")


def _paso_vianda(meal: dict) -> bool:
    ings = _YAUTIA_PROCESADA.sub(" ", " ".join(str(x) for x in (meal.get("ingredients") or [])))
    pasos = meal.get("recipe")
    if not _YAUTIA.search(ings) or not isinstance(pasos, list):
        return False
    texto = " ".join(str(p) for p in pasos).lower()
    if "completamente blanda" in texto and "yaut" in texto:
        return False
    pasos.append(_PASO_YAUTIA)
    return True


def etiquetar(plan: dict, form_data) -> int:
    """Devuelve cuántas comidas tocó. Muta `plan` (display, raw y nombre)."""
    if not (enabled() and isinstance(plan, dict) and aplica(form_data)):
        return 0
    tocadas = 0
    for d in plan.get("days") or []:
        for m in (d.get("meals") or []) if isinstance(d, dict) else []:
            if not isinstance(m, dict):
                continue
            cambio = False
            for campo in ("ingredients", "ingredients_raw"):
                lineas = m.get(campo)
                if not isinstance(lineas, list):
                    continue
                nuevas = [(_linea_atun_edamame(_linea_leche(_linea_queso(_linea_pescado(x)))) if isinstance(x, str) else x)
                          for x in lineas]
                if nuevas != lineas:
                    m[campo] = nuevas
                    cambio = True
            pasos = m.get("recipe")
            if isinstance(pasos, list):
                nuevos = [(_paso_queso(p) if isinstance(p, str) else p) for p in pasos]
                if nuevos != pasos:
                    m["recipe"] = nuevos
                    cambio = True
            if _paso_yuca(m):
                cambio = True
            if _paso_vianda(m):
                cambio = True
            nombre = m.get("name")
            if isinstance(nombre, str):
                n2 = _nombre_pescado(nombre)
                if n2 != nombre:
                    m["name"] = n2
                    cambio = True
            if cambio:
                m.pop("_display", None)      # la capa de traducción espeja `ingredients` por índice: se regenera
                m["_pregnancy_labels"] = True
                tocadas += 1
    if tocadas:
        logger.info(f"🤰 [P1-PLAN-LOTE-172] embarazo: especie del pescado y pasteurización del queso escritas en "
                    f"{tocadas} comida(s)")
    return tocadas
