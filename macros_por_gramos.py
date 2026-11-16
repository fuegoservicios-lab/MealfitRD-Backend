# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1039 · A-41 · 2026-11-16] Los macros de una comida salen de sus gramos, no del modelo.

Caso real (16-nov): misma pechuga y mismo bistec en dos cenas con foto. El coach anotó los gramos de cada pieza
(«pechuga de pollo a la plancha (≈170 g)») pero el TOTAL lo puso a ojo y quedó ~13 g de proteína por debajo de sus
propios gramos en las dos; la cena con más comida salió con menos proteína (78 g contra 89 g) y el usuario lo vio.

Aquí cada componente del plato se resuelve contra el catálogo con el MISMO resolutor que el plan y el contador de
micros (`IngredientNutritionDB.macros_from_ingredient_string`, que ya pasa los gramos cocinados a la base de la fila
— `cocido_en_catalogo`), y se aplica UN solo arreglo: el SUELO de proteína. Las piezas simples que resuelven con
gramos explícitos (pollo, res, huevo, puré…) suman una proteína que el plato entero no puede no alcanzar; si el total
del modelo queda >10 % por debajo, sube hasta ese suelo (y sus kcal, 4 kcal/g). Nunca se baja nada.

Por qué NO se reemplaza el plato entero por la suma (replay de 302 comidas reales, 16-nov): la lista de ingredientes
guardada suele estar INCOMPLETA respecto al plato («Pechuga de pollo cocida (100 g)» en una comida de 640 kcal) y no
trae la grasa de freír (plátano frito: grasa 36 → 6 g). La suma solo es fiable como cota INFERIOR.

Lo que NO se resuelve se deja al modelo: platos compuestos («ensalada de papa con mayonesa», «arroz con habichuelas
y carne»), filas del catálogo que no comparten palabra con el componente, o gramos fuera de rango. El caso que lo
exige: «zanahorias cocidas con aceite de oliva (120 g)» resolvía a 120 g de ACEITE (1.080 kcal); por eso la cola
«con …» se corta antes de buscar (es la guarnición, no la pieza) y la fila tiene que compartir palabra.

Puro: sin red ni base (el `db` se inyecta). Fail-open: cualquier excepción ⇒ `None` (se queda lo del modelo).
Knob `MEALFIT_MACROS_POR_GRAMOS=0` lo apaga.
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Optional

UMBRAL_SUELO = 1.10          # el suelo solo actúa si el modelo queda >10 % por debajo
GRAMOS_MIN, GRAMOS_MAX = 5.0, 1500.0

_RE_PAREN = re.compile(r"\(([^)]*)\)")
_RE_GRAMOS = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:g|gr|grs|gramos)\b", re.I)
_RE_ML = re.compile(r"(\d+(?:[.,]\d+)?)\s*ml\b", re.I)
_RE_CASERA = re.compile(r"^\s*(\d+(?:[.,/]\d+)?\s*[a-záéíóúñ]+)", re.I)
# Un plato hecho de varias cosas: su fila del catálogo sería solo UNA de ellas.
_RE_COMPUESTO = re.compile(
    r"\b(?:y|e|ensalada|sopa|sancocho|guiso|guisado\s+de|mixt[oa]s?|relleno|rellena|sandwich|sándwich|wrap|"
    r"pizza|hamburguesa|tacos?|burrito|lasa[ñn]a|pastel[oó]n|moro|locrio|mofongo|mang[uú])\b",
    re.I,
)
_PALABRAS_VACIAS = {"de", "del", "la", "el", "los", "las", "con", "sin", "al", "a", "en", "y", "e"}
# Productos DERIVADOS cuya fila del catálogo sería la del alimento de origen (replay 16-nov): «manteca de cerdo» →
# Cerdo, «jamón de cerdo» → Cerdo, «agua de coco» → Coco, «harina … preparada» → Harina, «pudín de pan» → Pan.
_RE_DERIVADO = re.compile(
    r"\b(?:manteca|aceite|jamon|salami|salchich\w*|embutido|chorizo|longaniza|agua|jugo|batida|batido|licuado|"
    r"leche\s+de|harina|pudin|bizcocho|galleta\w*|caldo|salsa|sopa|crema|yogur\w*|proteina\s+en\s+polvo|"
    r"whey|barra|cereal|granola|chocolate|empanad\w*|empanizad\w*|rebozad\w*|apanad\w*|nuggets?|"
    r"pica\s+pollo|alitas?|chofan|chaufa|pastelit\w*|quipes?|kipes?|croquetas?|sandwich\w*)\b"
)
# El peso incluye hueso, piel o grasa, o la pieza lleva relleno (pan, harina): su fila de carne magra la SOBRESTIMA
# (replay 16-nov: «bistec guisado con hueso», «cerdo asado con cuerito y grasa», «1 presa de pollo frito», «albóndigas»).
_RE_PESO_NO_MAGRO = re.compile(
    r"\b(?:hueso|huesos|cuerito|chicharr\w*|presas?|chuletas?|albondig\w*|tortitas?|costillas?|alas)\b|(?<!sin )\b(?:piel|grasa)\b"
)
MIN_DELTA_PROTEINA = 3.0     # por debajo, es ruido de redondeo (frutas: 0 → 0,3 g)
# kcal por gramo SERVIDO por encima de las cuales la fila no es la del plato servido (arroz contado en seco: 3,6;
# frutos secos, mantequilla de maní, aceites). Carnes cocinadas ≈ 1,5-2,5; queso ≈ 3.
KCAL_POR_GRAMO_MAX = 3.2


def activo() -> bool:
    return (os.getenv("MEALFIT_MACROS_POR_GRAMOS", "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def _plano(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t or "") if not unicodedata.combining(c)).lower()


def _num(t: str) -> float:
    return float(t.replace(",", "."))


def componente(ing: Any) -> Optional[dict]:
    """«pechuga de pollo a la plancha (≈170 g)» → {nombre, pieza, gramos, cadena, compuesto}.

    `pieza` es el nombre sin su guarnición («zanahorias cocidas con aceite» → «zanahorias cocidas»); `cadena` es la
    línea en el formato del catálogo («170 g de pechuga de pollo a la plancha»), o None si no hay cantidad."""
    if not isinstance(ing, str) or not ing.strip():
        return None
    s = " ".join(ing.replace("≈", "").replace("~", "").split())
    m = _RE_PAREN.search(s)
    nombre = (s[:m.start()] if m else s).strip(" ,;:-")
    dentro = m.group(1) if m else ""
    if not m:  # «75 g de Aguacate»: la cantidad va delante
        lider = _RE_GRAMOS.match(s)
        if lider:
            dentro = lider.group(0)
            nombre = re.sub(r"^(?:de|del)\s+", "", s[lider.end():].strip(), flags=re.I).strip(" ,;:-")
    # «zanahorias … con aceite», «atún en aceite»: la cola es la guarnición o el medio, no la pieza.
    pieza = re.split(r"\s+(?:con|en)\s+", nombre, maxsplit=1, flags=re.I)[0].strip(" ,;:-")
    gramos = None
    cadena = None
    g = _RE_GRAMOS.search(dentro)
    if g:
        gramos = _num(g.group(1))
        cadena = f"{gramos:g} g de {pieza}"
    elif m:
        ml = _RE_ML.search(dentro)
        casera = _RE_CASERA.match(dentro.split(",")[0].split(";")[0])
        if ml:
            cadena = f"{_num(ml.group(1)):g} ml de {pieza}"
        elif casera:
            cadena = f"{casera.group(1)} de {pieza}"
    else:
        cadena = s  # ya viene como «150 g de X» o «1 taza de X»
    return {
        "nombre": nombre, "pieza": pieza, "gramos": gramos, "cadena": cadena,
        "compuesto": bool(_RE_COMPUESTO.search(_plano(pieza)) or _RE_DERIVADO.search(_plano(pieza))
                          or _RE_PESO_NO_MAGRO.search(_plano(nombre))),
    }


def _comparte_palabra(fila: str, pieza: str) -> bool:
    p = set(re.findall(r"[a-zñ]{3,}", _plano(pieza)))
    p |= {w[:-1] for w in p if w.endswith("s")} | {w[:-2] for w in p if w.endswith("es")}
    for w in re.findall(r"[a-zñ]{3,}", _plano(fila)):
        if w in _PALABRAS_VACIAS:
            continue
        if w in p or (w.endswith("s") and w[:-1] in p):
            return True
    return False


def resolver(ingredients: Any, db) -> Optional[dict]:
    """Resuelve cada componente. {componentes: [...], simples: n, total: n, completo: bool, suma: {...}}."""
    if not isinstance(ingredients, list) or not ingredients or db is None:
        return None
    filas = []
    for ing in ingredients:
        c = componente(ing)
        if c is None:
            continue
        r = None
        if c["cadena"] and not c["compuesto"]:
            try:
                r = db.macros_from_ingredient_string(c["cadena"])
            except Exception:
                r = None
            if r and not _comparte_palabra(str(r.get("name") or ""), c["pieza"]):
                r = None
            if r and not (GRAMOS_MIN <= float(r.get("grams") or 0) <= GRAMOS_MAX * 1.5):
                r = None
            # gramos SERVIDOS explícitos y densidad de plato servido: si no, la fila puede estar en otra base
            if r and (c["gramos"] is None or not (GRAMOS_MIN <= c["gramos"] <= GRAMOS_MAX)
                      or float(r.get("kcal") or 0) / c["gramos"] > KCAL_POR_GRAMO_MAX):
                r = None
        filas.append({**c, "fila": (r or {}).get("name"), "macros": r})
    if not filas:
        return None
    simples = [f for f in filas if f["macros"]]
    suma = {k: round(sum(float(f["macros"].get(k) or 0) for f in simples), 1) for k in ("kcal", "protein", "carbs", "fats")}
    return {"componentes": filas, "simples": len(simples), "total": len(filas),
            "completo": len(simples) == len(filas), "suma": suma}


def reconciliar(meal: dict, db) -> Optional[dict]:
    """Corrección para una comida {calories, protein, carbs, healthy_fats, ingredients} o None si no hay nada que tocar.

    Devuelve {calories, protein, carbs, healthy_fats, motivo, antes, suelo} con los valores NUEVOS."""
    if not activo() or not isinstance(meal, dict):
        return None
    try:
        r = resolver(meal.get("ingredients"), db)
        if not r or not r["simples"]:
            return None
        antes = {k: float(meal.get(k) or 0) for k in ("calories", "protein", "carbs", "healthy_fats")}
        s = r["suma"]
        suelo = s["protein"]
        if suelo <= 0 or antes["protein"] * UMBRAL_SUELO >= suelo or suelo - antes["protein"] < MIN_DELTA_PROTEINA:
            return None
        # Si las piezas resueltas ya pesan MÁS energía que la comida entera, algo se resolvió mal (o el total del
        # modelo está roto por otro lado): no se toca.
        if antes["calories"] > 0 and s["kcal"] > antes["calories"] * 1.3:
            return None
        proteina = round(suelo)
        delta = proteina - antes["protein"]     # con la cifra que se VE, así kcal y proteína cuadran entre sí
        return {"calories": round(antes["calories"] + 4 * delta), "protein": proteina,
                "carbs": round(antes["carbs"]), "healthy_fats": round(antes["healthy_fats"]),
                "motivo": "suelo_de_proteina", "antes": antes, "suelo": suelo}
    except Exception:
        return None


_DB = None


def _db():
    global _DB
    if _DB is None:
        from nutrition_db import IngredientNutritionDB
        _DB = IngredientNutritionDB()
    return _DB


def al_registrar(user_id: Optional[str], calories, protein, carbs, healthy_fats, ingredients, db=None):
    """Para `tools.log_consumed_meal`: devuelve (calories, protein, carbs, healthy_fats, nota_para_el_modelo).

    Sin corrección devuelve los mismos valores y nota "". Con corrección deja huella en `pipeline_metrics` (node
    `macros_por_gramos`: motivo y deltas; sin nombres de comida ni datos de salud) y una nota para que el coach no
    contradiga la cifra guardada. Fail-open."""
    sin_cambio = (calories, protein, carbs, healthy_fats, "")
    try:
        c = reconciliar({"calories": calories, "protein": protein, "carbs": carbs, "healthy_fats": healthy_fats,
                         "ingredients": ingredients}, db if db is not None else _db())
    except Exception:
        return sin_cambio
    if not c:
        return sin_cambio
    try:
        if user_id and user_id != "guest":
            import json
            from db_core import execute_sql_write
            execute_sql_write(
                "INSERT INTO pipeline_metrics (user_id, session_id, node, duration_ms, retries, tokens_estimated, "
                "confidence, metadata) VALUES (%s, NULL, 'macros_por_gramos', 0, 0, 0, 0, %s::jsonb)",
                (user_id, json.dumps({"motivo": c["motivo"],
                                      "delta_proteina": round(c["protein"] - c["antes"]["protein"], 1),
                                      "delta_kcal": round(c["calories"] - c["antes"]["calories"], 1)})),
            )
    except Exception:
        pass
    nota = (f" (Para el asistente: la proteína se ajustó a los gramos de cada pieza que anotaste — "
            f"{int(round(c['antes']['protein']))} → {c['protein']} g, {c['calories']} kcal —; usa ESTAS cifras.)")
    return c["calories"], c["protein"], c["carbs"], c["healthy_fats"], nota


# ───────────────────────────────────────────────────────────── [A-41 · paso 3] gramos editables en la ficha
# La foto no sabe cuánto pesa la pechuga; el usuario sí. Un toque de −/+ en la ficha cambia los gramos de UNA pieza y
# los totales se mueven exactamente lo que esa pieza aporta según el catálogo (sin modelo). Solo piezas simples con
# gramos explícitos —las mismas que ponen el suelo—: en un plato compuesto no hay forma honesta de repartir el cambio.
PASO_GRAMOS = 10

_RE_GRAMOS_EDITABLE = re.compile(r"(\d+(?:[.,]\d+)?)(\s*(?:g|gr|grs|gramos)\b)", re.I)


def ajustables(ingredients: Any, db) -> dict:
    """{índice del renglón: gramos} de los renglones que se pueden ajustar."""
    out = {}
    try:
        r = resolver(ingredients, db)
    except Exception:
        return out
    if not r:
        return out
    textos = [x for x in ingredients if isinstance(x, str) and x.strip()]
    if len(textos) != len(r["componentes"]):
        return out
    for i, comp in enumerate(r["componentes"]):
        if comp["macros"] and comp["gramos"]:
            out[i] = int(round(comp["gramos"]))
    return out


def _con_gramos(texto: str, gramos: int) -> str:
    """El renglón con sus gramos cambiados: el número dentro del paréntesis («(≈170 g)») o el de delante («75 g de…»)."""
    m = _RE_PAREN.search(texto)
    if m:
        dentro = _RE_GRAMOS_EDITABLE.sub(lambda g: f"{gramos}{g.group(2)}", m.group(1), count=1)
        return texto[:m.start(1)] + dentro + texto[m.end(1):]
    return _RE_GRAMOS_EDITABLE.sub(lambda g: f"{gramos}{g.group(2)}", texto, count=1)


def ajustar_gramos(row: dict, indice: int, gramos_nuevos: Any, db) -> Optional[dict]:
    """Nueva {calories, protein, carbs, healthy_fats, ingredients} tras poner `gramos_nuevos` en el renglón `indice`, o
    None si ese renglón no es ajustable. Los totales se mueven la DIFERENCIA de esa pieza (y nunca bajan de 0)."""
    try:
        textos = [x for x in (row.get("ingredients") or []) if isinstance(x, str) and x.strip()]
        indice = int(indice)
        if not (0 <= indice < len(textos)):
            return None
        if indice not in ajustables(textos, db):
            return None
        g = int(round(float(gramos_nuevos)))
        g = int(min(GRAMOS_MAX, max(GRAMOS_MIN, g)))
        comp = componente(textos[indice])
        antes = db.macros_from_ingredient_string(comp["cadena"])
        despues = db.macros_from_ingredient_string(f"{g} g de {comp['pieza']}")
        if not antes or not despues:
            return None
        delta = {k: float(despues.get(k) or 0) - float(antes.get(k) or 0) for k in ("kcal", "protein", "carbs", "fats")}
        nuevos = list(textos)
        nuevos[indice] = _con_gramos(textos[indice], g)
        return {
            "calories": max(0, round(float(row.get("calories") or 0) + delta["kcal"])),
            "protein": max(0, round(float(row.get("protein") or 0) + delta["protein"])),
            "carbs": max(0, round(float(row.get("carbs") or 0) + delta["carbs"])),
            "healthy_fats": max(0, round(float(row.get("healthy_fats") or 0) + delta["fats"])),
            "ingredients": nuevos,
            "gramos": g,
        }
    except Exception:
        return None
