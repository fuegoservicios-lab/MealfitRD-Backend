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
    """¿Esta línea es de un alimento que da identidad a un plato de biblioteca? Los recortes de macros no la tocan."""
    try:
        return any(_contiene(nom, linea) for nom, _g in _identidad_de(meal))
    except Exception:                                                          # noqa: BLE001
        return False


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


def restaurar_meal(meal: dict, index: dict, *, db=None, allergies=None, margen=None) -> list:
    """Un plato. Devuelve lo que añadió o subió (`["+38 g de Aguacate"]`, `["↑5→38 g de Aguacate"]`); `[]` si no tocó
    nada. Sin `margen` sólo vuelve lo que FALTA (lote 46); con `margen` (lo que al día le queda hasta su techo de kcal y de
    grasa, `_margen_del_dia`) sube además lo presente por debajo del piso, si cabe."""
    if not isinstance(meal, dict) or meal.get("_recipe_source") != "library" or meal.get("_sodium_autofix_applied"):
        return []
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
        for m in meals:
            try:
                hechos = restaurar_meal(m, index, db=db, allergies=allergies, margen=margen)
            except Exception as e:                                             # noqa: BLE001
                logger.debug(f"[P1-PLAN-LOTE-46] identidad no-op en {str((m or {}).get('name'))[:40]}: {e!r}")
                hechos = []
            if hechos:
                tocados += 1
                logger.info(f"🧩 [P1-PLAN-LOTE-46] identidad del plato restaurada en «{str(m.get('name'))[:48]}»: "
                            f"{', '.join(hechos)}")
    return tocados
