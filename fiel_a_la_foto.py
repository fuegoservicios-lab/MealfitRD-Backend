# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-990 · 2026-11-12] Lo que se registra desde una foto es lo que muestra la foto.

Caso vivo del dueño (12-nov 11:43): el escáner leyó bien el plato —«1 huevo frito, 2 rebanadas de pan tostado en
triángulos (2 mitades con mantequilla de maní y 2 con queso derretido) y frutas picadas (lechoza, piña y sandía)»—, el
usuario contestó la duda con «Queso derretido», y el coach registró «Omelet de 4 huevos con 2 arepitas de maíz cubiertas
de queso derretido»: el «omelet de 4 huevos» venía de sus hechos guardados («desayuna omelet de 4 huevos») y las arepitas
de ninguna parte; la mantequilla de maní y la fruta desaparecieron.

`log_consumed_meal` en un turno con UNA foto de plato compara el registro con la descripción por FAMILIAS de alimento:
  · inventado: una familia del registro que no está en la foto (ni en sus dudas) ni la dijo el usuario en este turno;
  · omitido: una familia de la foto que el registro no lleva;
  · huevos: la foto cuenta N huevos y el registro otro número que el usuario no dijo.
Rechaza UNA vez por usuario en 120 s con la lista exacta; si el modelo reintenta igual, pasa (el registro jamás se
pierde por esto). Knob `MEALFIT_PHOTO_FIDELITY_GUARD` (True). tooltip-anchor: P1-PLAN-LOTE-990
"""
from __future__ import annotations

import re
import time
import unicodedata
from typing import Iterable, Optional

# familia → patrones (texto normalizado sin acentos). El orden importa poco: cada familia se busca por separado.
_FAMILIAS = {
    "huevo": r"huevos?|omelets?|omelettes?|revoltillo|claras? de huevo|\d+ claras?",
    "pan": r"pan(?:es)?|panecillos?|tostadas?|sandwich(?:es)?|emparedados?|bagels?|croissants?|pan de agua",
    "arepa": r"arepas?|arepitas?",
    "mangu": r"mangu",
    "casabe": r"casabe",
    "platano": r"platanos?|tostones?|maduros?|\d+ fritos|fritos verdes|fritos maduros|con fritos",
    "yuca": r"yuca",
    "batata": r"batatas?",
    "papa": r"papas?",
    "arroz": r"arroz|locrio|moro",
    "legumbre": r"habichuelas?|frijoles?|lentejas?|garbanzos?|guandules?",
    "pasta": r"pasta|espaguetis?|spaghetti|fideos?|macarrones?|ramen|noodles?",
    "pollo": r"pollo|pechuga",
    "res": r"res|carne molida|bistec|churrasco|steak",
    "cerdo": r"cerdo|chuleta|longaniza|chicharron",
    "salami": r"salami",
    "jamon": r"jamon",
    "pescado": r"pescado|atun|salmon|bacalao|sardinas?",
    "queso": r"queso",
    "mani": r"mani|cacahuate",
    "fruta": r"frutas?|lechosa|lechoza|pina|sandia|melon|mango|fresas?|guineos?|manzana|uvas?|chinola|naranja|guanabana|morir sonando|coco|limon",
    "avena": r"avena",
    "yogur": r"yogurt?|yogur",
    "cereal": r"granola|cereal",
    "aguacate": r"aguacate",
    "galleta": r"galletas?",
    "panqueque": r"panqueques?|pancakes?|waffles?",
}
_PATRONES = {f: re.compile(r"\b(?:" + p + r")\b") for f, p in _FAMILIAS.items()}
_NUM = {"un": 1, "una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6}
_HUEVOS = re.compile(r"\b(\d+|un|una|dos|tres|cuatro|cinco|seis)\s+(?:huevos?|claras?)\b")
_VENTANA_S = 120.0
_YA_AVISADO: dict = {}


def _norm(t) -> str:
    t = unicodedata.normalize("NFKD", str(t or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", t).strip()


# Lo que el usuario dice en lugar de lo que se veía («Salmón» donde la foto decía pollo) sustituye dentro de su GRUPO.
_GRUPO = {
    **{f: "proteina" for f in ("huevo", "pollo", "res", "cerdo", "salami", "jamon", "pescado")},
    **{f: "base" for f in ("pan", "arepa", "mangu", "casabe", "platano", "yuca", "batata", "papa", "arroz", "pasta",
                           "panqueque", "avena", "cereal", "galleta")},
    **{f: "lacteo" for f in ("queso", "yogur")},
}


def familias(texto) -> set:
    n = _norm(texto)
    out = {f for f, p in _PATRONES.items() if p.search(n)}
    if "mangu" in out:           # el mangú ES plátano majado: el que dice «mangú» también dice «plátano»
        out.add("platano")
    if "jamon" in out:           # «jamón cocido de cerdo» no es un cerdo aparte
        out.discard("cerdo")
    return out


_ALTERNATIVA = re.compile(r"([a-z]+(?: [a-z]+)?) o ([a-z]+(?: [a-z]+)?)")


def _opcionales(desc: str) -> list:
    """«plátano maduro o batata majada» → [{'platano', 'batata'}]: con que el registro lleve una, basta."""
    out = []
    for m in _ALTERNATIVA.finditer(_norm(desc)):
        a, b = familias(m.group(1)), familias(m.group(2))
        if a and b and a != b:
            out.append(a | b)
    return out


def huevos(texto) -> Optional[int]:
    total, visto = 0, False
    for m in _HUEVOS.finditer(_norm(texto)):
        v = m.group(1)
        total += int(v) if v.isdigit() else _NUM.get(v, 0)
        visto = True
    return total if visto else None


def _texto_registro(meal_name, ingredientes: Iterable) -> str:
    partes = [str(meal_name or "")]
    for ing in ingredientes or []:
        partes.append(str(ing.get("name") if isinstance(ing, dict) else ing))
    return " · ".join(partes)


def diferencias(meal_name, ingredientes, foto_desc: str, foto_desc_completa: str = "", texto_usuario: str = "") -> dict:
    reg = _texto_registro(meal_name, ingredientes)
    f_reg = familias(reg)
    f_foto = familias(foto_desc)
    permitidas = f_foto | familias(foto_desc_completa) | familias(texto_usuario)
    f_usr = familias(texto_usuario)
    sustitutas = {_GRUPO.get(g) for g in (f_usr & f_reg) - f_foto} - {None}
    omitidos = {f for f in f_foto - f_reg if _GRUPO.get(f) not in sustitutas}
    for alternativas in _opcionales(foto_desc):
        if alternativas & f_reg:
            omitidos -= alternativas
    out = {"inventados": sorted(f_reg - permitidas), "omitidos": sorted(omitidos), "huevos": None}
    # los huevos del registro se cuentan en sus ingredientes (el nombre los repite); sin ingredientes, en el nombre
    h_reg = huevos(_texto_registro("", ingredientes))
    h_reg = huevos(meal_name) if h_reg is None else h_reg
    h_foto = huevos(foto_desc)
    if h_foto is not None and h_reg is not None and h_foto != h_reg and huevos(texto_usuario) is None:
        out["huevos"] = (h_reg, h_foto)
    return out


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_PHOTO_FIDELITY_GUARD", True)
    except Exception:                                                          # noqa: BLE001
        return True


_NOMBRE = {"mani": "mantequilla de maní / maní", "res": "carne de res", "legumbre": "habichuelas/legumbres",
           "fruta": "la fruta", "pan": "pan/tostadas", "yogur": "yogur"}


def rechazo(user_id, meal_name, ingredientes, fotos: list, texto_usuario: str = "", *, ahora: float = None):
    """Texto del rechazo, o None (= registrar). Solo con UNA foto de plato en el turno; una vez por usuario en 120 s."""
    if not activo() or not isinstance(fotos, list) or len(fotos) != 1 or not isinstance(fotos[0], dict):
        return None
    foto = fotos[0]
    desc = str(foto.get("desc") or "")
    if not desc:
        return None
    # solo si el registro ES la comida de la foto (comparten algún alimento): unos bagels del desayuno registrados en
    # el mismo turno que la foto del almuerzo no se comparan con ella
    if not (familias(_texto_registro(meal_name, ingredientes)) & familias(desc)):
        return None
    dif = diferencias(meal_name, ingredientes, desc, str(foto.get("desc_completa") or ""), texto_usuario)
    if not (dif["inventados"] or dif["omitidos"] or dif["huevos"]):
        return None
    t = time.time() if ahora is None else ahora
    for k, ts in list(_YA_AVISADO.items()):
        if t - ts > _VENTANA_S:
            _YA_AVISADO.pop(k, None)
    clave = str(user_id)
    if clave in _YA_AVISADO:
        return None
    _YA_AVISADO[clave] = t
    lineas = []
    if dif["inventados"]:
        lineas.append("lleva lo que NO está en la foto ni dijo el usuario: "
                      + ", ".join(_NOMBRE.get(f, f) for f in dif["inventados"]))
    if dif["omitidos"]:
        lineas.append("le falta lo que SÍ se ve en la foto (inclúyelo salvo que el usuario haya dicho que eso no se lo "
                      "comió): " + ", ".join(_NOMBRE.get(f, f) for f in dif["omitidos"]))
    if dif["huevos"]:
        lineas.append(f"pone {dif['huevos'][0]} huevo(s) y la foto muestra {dif['huevos'][1]}")
    return ("⚠️ NO REGISTRADO TODAVÍA: el registro no coincide con la foto del usuario — " + "; ".join(lineas)
            + f". La foto muestra: «{desc[:300]}». Registra EXACTAMENTE eso (con la respuesta del usuario a la duda, "
            "si la dio), con cantidad en cada ingrediente y las calorías y macros recalculadas, y vuelve a llamar a "
            "`log_consumed_meal`. No uses lo que suele desayunar ni platos de otros días. No le preguntes nada al "
            "usuario por esto ni le digas que hubo un rechazo.")
