# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-187 · 2026-09-23] Embarazo y lactancia: el pescado y el marisco, 227–340 g por semana (FDA/EPA, 2-3 raciones).

Batería real: el revisor rechazó CRÍTICO «545 g de pescado y mariscos en 3 días» (rd5 → plan de EMERGENCIA) y «460 g de
pescado (120 g de atún claro y 340 g de tilapia) en tres días» (rd15). La regla está en el prompt desde el lote 175 y el
modelo la incumple: cada día se genera por separado y no ve los otros, así que el total sólo se puede cumplir DESPUÉS.

Se recorren los días en orden y, cuando el pescado acumulado pasaría del tope de la ventana (340 g por cada 7 días), el
pescado de ESE plato se cambia por la misma cantidad de una proteína que ese día no se repita (pechuga de pollo,
pechuga de pavo, carne de res, cerdo; ninguna que choque con alergias o rechazos declarados). Sustituir y no recortar:
recortar abre DÉFICIT de proteína (medido en el lote 184). Sólo dieta omnívora — una pescetariana no tiene a qué
cambiar — y sin sustituto limpio el plato se queda como está. Lista, compra, nombre y pasos; los macros los re-mide el
truth-up. Knob `MEALFIT_PREGNANCY_FISH_CAP_G` (340; 0 = apagado). tooltip-anchor: P1-PLAN-LOTE-187-PESCADO-EMBARAZO
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# «dorado» sólo como «filete de dorado»: suelto es también adjetivo («plátano dorado», «pollo dorado»); «bonito», fuera.
_PESCADO = re.compile(r"\b(?:filetes?\s+de\s+dorado|(?:filetes?\s+de\s+)?(?:pescado(?:\s+blanco)?|tilapia|at[uú]n(?:\s+claro)?|"
                      r"salm[oó]n|mero|chillo|pargo|merluza|bacalao|arenque|sardinas?|caballa|corvina|camarones|camar[oó]n|"
                      r"langostinos?|calamar(?:es)?|pulpo|mejillones|langosta|cangrejo))\b", re.IGNORECASE)
_SUSTITUTOS = ("Pechuga de pollo", "Pechuga de pavo", "Carne de res", "Cerdo")

# [P1-PLAN-LOTE-783 · 2026-09-28] El pescado de LATA se come tal cual; su sustituto, no. Con el tope, «Mezcla el atún en agua
# (ya viene cocido) con el tomate… sirve frío» quedaba «mezcla pechuga de pavo en agua (ya viene cocido)… sirve frío», y la
# batería real de lactancia servía «Acompaña con pechuga de pavo en agua» en un revoltillo «…y pavo en agua»: carne CRUDA,
# fría, a una embarazada o lactante (el aviso de 74 °C del 737 no dice CUÁNDO cocinarla). Si el pescado cambiado era de
# lata, la frase del envase se va entera (nombre, descripción y pasos), «(ya viene cocido)» se cae, «escurre» pasa a
# «desmenuza» y, si ningún paso cocina el sustituto, lleva la «💡 Cocción previa» del lote 407 (74 °C, tras el Mise en
# place). El pescado fresco sigue como estaba: sus pasos ya lo cocinan. Knob `MEALFIT_PREGNANCY_FISH_SUB_COOK`.
# tooltip-anchor: P1-PLAN-LOTE-783
_SUFIJOS_ENVASE = (" en agua", " en aceite", " en salmuera", " en lata", " de lata",
                   " enlatado", " enlatada", " enlatados", " enlatadas")
_PESCADO_ENVASE = re.compile("(?:" + _PESCADO.pattern + r")(?:\s+(?:en\s+(?:agua|aceite|salmuera|lata)|de\s+lata|"
                             r"enlatad[oa]s?)\b)?", re.IGNORECASE)
_LISTO_RE = re.compile(r"\ben\s+(?:agua|aceite|salmuera)\b|\blatas?\b|enlatad", re.IGNORECASE)
_LATA_POR_DEFECTO_RE = re.compile(r"\bat[uú]n\b|\bsardinas?\b|\barenque", re.IGNORECASE)
_FRESCO_RE = re.compile(r"\bfresc[oa]s?\b|\bfiletes?\b|\bpostas?\b|\blomos?\b|\brodajas?\b", re.IGNORECASE)
_YA_VIENE_RE = re.compile(r"\s*\((?:ya\s+viene|viene\s+ya)\s+cocid[oa]s?\)", re.IGNORECASE)
_CLAVE_SUSTITUTO = {"pechuga de pollo": "pollo", "pechuga de pavo": "pavo", "carne de res": "res", "cerdo": "cerdo"}


def _sub_cocina_on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_PREGNANCY_FISH_SUB_COOK", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _es_de_lata(linea) -> bool:
    """¿Pescado que se come tal cual? «en agua/aceite», lata; el atún, la sardina y el arenque lo son salvo que la línea
    diga fresco, filete, posta, lomo o rodaja."""
    s = str(linea or "")
    return bool(_LISTO_RE.search(s) or (_LATA_POR_DEFECTO_RE.search(s) and not _FRESCO_RE.search(s)))


def _cocinar_sustituto(meal: dict, nombre: str) -> int:
    """Tras cambiar pescado de lata por `nombre` (crudo): fuera «(ya viene cocido)», «escurre» → «desmenuza» y, si ningún
    paso lo cocina, la «💡 Cocción previa» del 407 tras el Mise en place. Nº de cambios; 0 ante cualquier error."""
    try:
        import pasos_cantidades as pc
        rec = meal.get("recipe")
        clave = _CLAVE_SUSTITUTO.get(str(nombre or "").lower())
        if not isinstance(rec, list) or not rec or not clave:
            return 0
        nucleo = re.escape(str(nombre).lower().split(" de ")[0])        # «pechuga», «carne», «cerdo»
        sub = r"(?:(?:el|la|los|las)\s+)?" + re.escape(str(nombre).lower())
        n = 0
        nuevos = []
        for p in rec:
            q = p
            if isinstance(p, str) and not pc._es_nota(p) and re.search(nucleo, p, re.IGNORECASE):
                q = _YA_VIENE_RE.sub("", p)
                if q.strip().lower().startswith("mise en place"):
                    # «escurre el atún» del Mise en place: crudo no se escurre, y la cocción previa ya lo desmenuza
                    q = re.sub(r"\b[Ee]scurre\s+" + sub + r"\s*(?:y\s+|;\s*|,\s*)", "", q, flags=re.IGNORECASE)
                    q = re.sub(r"(?:\s*[,;]\s*|\s+y\s+)[Ee]scurre\s+" + sub + r"\b", "", q, flags=re.IGNORECASE)
                    q = re.sub(r"\b[Ee]scurre\s+" + sub + r"\b\s*", "", q, flags=re.IGNORECASE)
                    if re.fullmatch(r"\s*mise en place\s*:?\s*\.?\s*", q, re.IGNORECASE):
                        n += 1
                        continue
                q = re.sub(r"\b([Ee])scurre\s+e\s+incorpora\b", lambda m: "Incorpora" if m.group(1) == "E" else "incorpora", q)
                q = re.sub(r"\b([Ee])scurre\b(?=[^.;]{0,20}" + nucleo + ")",
                           lambda m: "Desmenuza" if m.group(1) == "E" else "desmenuza", q)
                n += q != p
            nuevos.append(q)
        pasos = " . ".join(pc._sa(str(p).lower()) for p in nuevos if isinstance(p, str) and not pc._es_nota(p))
        if not re.search(pc._VERBO_COCCION_407 + pc._SINONIMOS_407.get(clave, clave), pasos):
            clase, art = pc._clase_407(clave, str(nombre).lower())
            nota = pc._NOTA_PROT_407[clase].format(n=art)
            if nota not in nuevos:
                i_mise = next((i for i, x in enumerate(nuevos)
                               if isinstance(x, str) and x.strip().lower().startswith("mise en place")), None)
                pos = (i_mise + 1) if i_mise is not None else 0
                nuevos[pos:pos] = [nota]
                n += 1
        if n:
            meal["recipe"] = nuevos
            logger.info(f"🐟 [P1-PLAN-LOTE-783] {str(meal.get('name'))[:44]!r}: el {str(nombre).lower()} que sustituye al "
                        f"pescado de lata se cocina ({n} cambio(s))")
        return n
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-783] no-op: {type(e).__name__}: {e}")
        return 0


def tope_g() -> int:
    try:
        from knobs import _env_int
        return max(0, _env_int("MEALFIT_PREGNANCY_FISH_CAP_G", 340))
    except Exception:                                                          # noqa: BLE001
        return 340


def aplica(form_data) -> bool:
    try:
        from nutrition_calculator import _is_pregnancy_or_lactation
        from constants import canonicalize_diet_type
        fd = form_data or {}
        return bool(_is_pregnancy_or_lactation(fd)) and canonicalize_diet_type(fd.get("dietType")) == "balanced"
    except Exception:                                                          # noqa: BLE001
        return False


def _gramos(linea: str, db) -> float:
    try:
        return float(db.grams_from_ingredient_string(linea) or 0.0)
    except Exception:                                                          # noqa: BLE001
        return 0.0


def _lineas_pescado(meal: dict) -> list:
    return [i for i, x in enumerate(meal.get("ingredients") or []) if isinstance(x, str) and _PESCADO.search(x)]


def _sustituto(dia: dict, meal: dict, vetos: list, go):
    """El primer sustituto cuya proteína no aparece en OTRA comida del día (mismo SSOT que la puerta de variedad) y que
    no choca con alergias ni rechazos. `None` si no hay ninguno limpio."""
    from constants import strip_accents
    usados: set = set()
    for m in dia.get("meals") or []:
        if m is meal or not isinstance(m, dict):
            continue
        blob = strip_accents((str(m.get("name") or "") + " " + " ".join(str(x) for x in m.get("ingredients") or [])).lower())
        usados |= go._protein_gate_labels_in_text(blob)
    for nombre in _SUSTITUTOS:
        if go._protein_gate_labels_in_text(strip_accents(nombre.lower())) & usados:
            continue
        if vetos and go._allergen_pool_item_banned(nombre, vetos):
            continue
        return nombre
    return None


# [P1-PLAN-LOTE-785 · 2026-09-28] El punto del PESCADO no sirve para el ave ni la carne. Con el tope, «forma tortitas y
# hornéalas… hasta que el huevo esté cocido y el pescado alcance 63 °C» quedaba «…y pechuga de pavo alcance 63 °C» (corpus
# de la cola 744, embarazo): el ave pide 74 °C y la carne 71 °C — las cifras de las notas del 407 y del 737. En la frase
# que nombra al sustituto: 60-69 °C → su temperatura, y «hasta que se desmenuce / esté opaco» → «hasta que no quede
# rosada por dentro». Las demás cifras (el horno a 200 °C) no se tocan. Knob `MEALFIT_PREGNANCY_FISH_SUB_COOK`.
# tooltip-anchor: P1-PLAN-LOTE-785
_TEMP_PESCADO_RE = re.compile(r"\b6\d\s*°\s*[Cc]\b")
_PUNTO_PESCADO_RE = re.compile(r"\b(?:se\s+desmenuce(?:\s+(?:f[aá]cilmente|con\s+(?:un|el)\s+tenedor))?|"
                               r"(?:est[eé]n?|quede[n]?)\s+opac[oa]s?(?:\s+y\s+firmes?)?)", re.IGNORECASE)


def _punto_del_sustituto(meal: dict, nombre: str) -> int:
    """En los pasos (no en las notas), la frase que nombra al sustituto deja el punto del pescado por el suyo. Nº de pasos
    cambiados; 0 ante cualquier error."""
    try:
        import pasos_cantidades as pc
        rec = meal.get("recipe")
        clave = _CLAVE_SUSTITUTO.get(str(nombre or "").lower())
        if not isinstance(rec, list) or not rec or not clave:
            return 0
        grados = "74 °C" if clave in ("pollo", "pavo") else "71 °C"
        rosado = "rosado" if clave == "cerdo" else "rosada"
        nucleo = re.compile(r"\b(?:" + re.escape(str(nombre).lower().split(" de ")[0]) + "|" + re.escape(clave) + r")\b",
                            re.IGNORECASE)
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or pc._es_nota(p) or not nucleo.search(p):
                continue
            partes = re.split(r"((?<!\d)[.;](?!\d))", p)
            for j, cl in enumerate(partes):
                if nucleo.search(cl):
                    cl2 = _TEMP_PESCADO_RE.sub(grados, cl)
                    cl2 = _PUNTO_PESCADO_RE.sub(f"no quede {rosado} por dentro", cl2)
                    partes[j] = cl2
            q = "".join(partes)
            if q != p:
                rec[i] = q
                n += 1
        if n:
            meal["recipe"] = rec
            logger.info(f"🌡️ [P1-PLAN-LOTE-785] {str(meal.get('name'))[:44]!r}: el {str(nombre).lower()} lleva su punto "
                        f"({grados}), no el del pescado")
        return n
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-785] no-op: {type(e).__name__}: {e}")
        return 0

def limitar_pescado(plan, form_data, db=None) -> int:
    """Devuelve cuántas comidas cambió. Muta `plan`."""
    cap = tope_g()
    if not (cap and isinstance(plan, dict) and aplica(form_data)):
        return 0
    dias = [d for d in (plan.get("days") or []) if isinstance(d, dict) and d.get("meals")]
    if not dias:
        return 0
    try:
        import dish_naming
        import graph_orchestrator as go
        from constants import alergias_y_rechazos
        if db is None:
            from nutrition_db import IngredientNutritionDB
            db = IngredientNutritionDB()
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-187] tope de pescado no-op: {type(e).__name__}: {e}")
        return 0
    tope = cap * max(1.0, len(dias) / 7.0)
    vetos = alergias_y_rechazos(form_data)
    acumulado, tocadas = 0.0, 0
    for dia in sorted(dias, key=lambda d: int(d.get("day") or 0)):
        for meal in dia.get("meals") or []:
            if not isinstance(meal, dict):
                continue
            idx = _lineas_pescado(meal)
            if not idx:
                continue
            gramos = sum(_gramos(meal["ingredients"][i], db) for i in idx)
            if acumulado + gramos <= tope:
                acumulado += gramos
                continue
            nombre = _sustituto(dia, meal, vetos, go)
            if nombre is None:
                acumulado += gramos                  # sin sustituto limpio el plato se queda como está
                continue
            try:
                listo = _sub_cocina_on() and any(_es_de_lata(meal["ingredients"][i]) for i in idx)  # [P1-PLAN-LOTE-783]
                tokens: set = set()            # los pasos dicen «la tilapia», no «filete de tilapia»: los tres
                for i in idx:
                    g = _PESCADO.search(meal["ingredients"][i]).group(0).lower()
                    base = re.sub(r"^filetes?\s+de\s+", "", g)
                    tokens |= {g, base, base.split()[0]}
                if listo:                      # [P1-PLAN-LOTE-783] la frase del envase se va entera
                    tokens |= {t + suf for t in list(tokens) for suf in _SUFIJOS_ENVASE}
                for campo in ("ingredients", "ingredients_raw"):
                    lineas = meal.get(campo)
                    if not isinstance(lineas, list):
                        continue
                    meal[campo] = [(f"{round(_gramos(x, db)) or 100} g de {nombre.lower()}"
                                    if isinstance(x, str) and _PESCADO.search(x) else x) for x in lineas]
                meal["name"] = dish_naming.sustituir_alimento(meal, _PESCADO_ENVASE if listo else _PESCADO,
                                                              nombre.split(" de ")[-1].capitalize()
                                                              if nombre.startswith("Pechuga") else nombre)
                go._rewrite_recipe_steps_after_subs(meal, [(sorted(tokens, key=len, reverse=True), nombre.lower())])
                if listo:
                    _cocinar_sustituto(meal, nombre)  # [P1-PLAN-LOTE-783] crudo: se cocina, no se «escurre»
                if _sub_cocina_on():
                    _punto_del_sustituto(meal, nombre)  # [P1-PLAN-LOTE-785] 74 °C el ave, no los 63 °C del pescado
                meal.pop("_display", None)
                meal["_embarazo_pescado_cap"] = nombre
                try:
                    go._truth_up_meal_macros_from_strings(meal, db)
                except Exception:                                              # noqa: BLE001
                    pass
                tocadas += 1
            except Exception as e:                                             # noqa: BLE001
                logger.debug(f"[P1-PLAN-LOTE-187] pescado no-op en {str(meal.get('name'))[:40]}: {e!r}")
    if tocadas:
        logger.info(f"🐟 [P1-PLAN-LOTE-187] embarazo: {tocadas} plato(s) de pescado de más → otra proteína "
                    f"(tope {round(tope)} g en {len(dias)} día(s))")
    return tocadas
