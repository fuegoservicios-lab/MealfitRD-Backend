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
    y en el nombre del plato. Los pasos no se tocan: la lista ya identifica la especie.
  · queso fresco/blando sin la palabra «pasteurizado» → se le añade. Los quesos curados/duros no la necesitan.

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
                           r"suizo|manchego|curado|azul))|ricotta|cottage|reques[oó]n|mozzarella)\b[^,;()]*", re.IGNORECASE)


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


def _linea_pescado(s: str) -> str:
    def _sub(m):
        return (m.group(1) or "") + "tilapia"
    return _PESCADO_LINEA.sub(_sub, s)


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
    return s[:fin] + " pasteurizado" + s[fin:]


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
                nuevas = [(_linea_queso(_linea_pescado(x)) if isinstance(x, str) else x) for x in lineas]
                if nuevas != lineas:
                    m[campo] = nuevas
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
