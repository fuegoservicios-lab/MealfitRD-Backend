# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-214/215 · 2026-09-24] La compra única de 30 días, determinista.

El dueño (24-sep): «si elijo 30 días y quiero hacer la compra de golpe, debería darme alimentos para comprar una sola
vez durante todo el mes […] de manera determinista». La política ya existía (`horizon.single_trip_policy`: ciclo > 7
días y sin reposición de frescos), pero medido en su plan real de 30 días (`6594aae1`, sin congelador) la lista del día 1
extrapolaba ×10 los 3 días generados: 2 lb de pechuga y 32 oz de pescado que aguantan 3 días, 60 huevos «alcanza ~13 de
30 días», y NINGUNA proteína de despensa para los días 8-30, que la política obliga a cocinar con duraderos. El usuario
compraba una vez y a mitad de mes le faltaba comida.

Dos piezas, un solo SSOT de «qué aguanta y por qué se cambia»:

1. **La sustitución, segura y variada (lote 214).** Lo que no aguanta hasta su día se cambia por su equivalente duradero
   (`graph_orchestrator._single_trip_fresh_substitute`, P1-STEP14-SHOPPING-COOKING, que ahora delega aquí la decisión
   por línea). Toda carne y todo pescado pasaban a «atún en agua» SIN mirar las alergias —y eso corre en el escudo final,
   después del revisor: un alérgico al pescado con un plan de 30 días sin congelador recibía atún del día 8 en adelante,
   la misma clase que el yogur del lote 189— y 23 días de atún. Ahora la proteína rota por día y comida (atún, sardinas,
   garbanzos; vegetariano: garbanzos, lentejas) y cada candidato pasa por el backstop clínico
   (`clinical_backstop_for_meal`: alergias, dieta y mercurio en embarazo); el primero seguro gana y, si ninguno lo es,
   NO se sustituye (un aviso de durabilidad es mejor que un alérgeno). Los víveres y el pan que no llegan al fin del
   ciclo tienen ahora su duradero (plátano y yuca → batata, pan → casabe). La cantidad respeta el peso de la línea
   («1 pechuga (≈200 g)» → «200 g de …», antes «1 de atún»). Knob `MEALFIT_SINGLE_TRIP_SAFE_SUBSTITUTE`.

2. **La proyección del ciclo (lote 215).** `dias_de_la_compra(plan_data, reales)`: los días reales + los que faltan hasta
   el fin del ciclo, copiados en rueda y pasados por la MISMA sustitución que recibirán sus bloques; lo que en la copia
   sigue sin aguantar su día y no tiene duradero no se compra para ese día. Vive DENTRO de
   `shopping_calculator.shopping_source_days`, el SSOT de «desde qué días se agrega la lista», así que la lista, el lado
   esperado del guard de coherencia y su base de días leen el MISMO mes: el guard no ve fantasmas. Con el ciclo
   proyectado la lista del ciclo es la Σ exacta de sus días (multiplicador 1), no «3 días × 10»; lo fresco se queda en su
   semana y la despensa cubre el resto. La lista lleva el sello `_compra_unica` (días del ciclo) y la híbrida lo honra:
   en una sola compra no hay «perecederos de la semana», todo sale con la cantidad del ciclo, y lo ya comprado no vuelve
   a pedirse hasta que acabe el ciclo. Knob `MEALFIT_SINGLE_TRIP_CYCLE_PROJECTION`.

Un bloque suelto (el `result` de un chunk, `_days_offset > 0`) NO se proyecta: su lista es transitoria y el worker la
rehace sobre el plan completo (T2). Puro salvo el backstop de alergias; nunca lanza.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)

PROTEINA_TABLA = "atun en agua"          # el sustituto genérico de la proteína fresca (tabla de abajo)
PROTEINA_VEGETAL = "garbanzos cocidos"
_ROTACION_OMNIVORA = ("atun en agua", "sardinas en lata", "garbanzos cocidos")
_ROTACION_VEGETAL = ("garbanzos cocidos", "lentejas cocidas")

# (tokens, duradero). Primer match gana; los tokens se buscan como palabra (singular/plural) en la línea sin acentos.
# Era `graph_orchestrator._FRESH_SUBSTITUTES` (P1-STEP14-SHOPPING-COOKING); allí queda un alias.
SUSTITUTOS = (
    (("lechuga", "berro", "rucula", "arugula", "espinaca", "acelga", "kale", "col rizada"), "repollo"),
    (("tomate cherry", "tomate"), "zanahoria"),
    (("pepino", "calabacin", "zucchini", "brocoli", "coliflor", "vainitas", "habichuelas verdes", "esparrago", "champinon", "hongos", "setas"), "zanahoria"),
    (("cilantro", "perejil", "albahaca", "menta", "cebollin", "cebollino"), "oregano"),
    (("fresa", "frambuesa", "mora", "arandano", "uva", "lechosa", "papaya", "mango", "pina", "melon", "sandia", "guineo", "banana", "durazno", "melocoton", "pera", "kiwi", "cereza", "mamey", "nispero", "aguacate"), "manzana"),
    (("pescado", "tilapia", "salmon", "mero", "chillo", "dorado", "bacalao fresco", "merluza", "camaron", "camarones", "mariscos", "calamar", "pulpo", "cangrejo", "langosta", "lambi"), PROTEINA_TABLA),
    (("pechuga de pollo", "pollo", "muslo", "pavo", "carne de res", "res molida", "res", "bistec", "cerdo", "chuleta", "lomo", "chivo", "conejo", "higado"), PROTEINA_TABLA),
    # lácteos: solo la leche tiene sustituto duradero honesto (UHT, misma unidad de volumen); yogurt, cottage y queso
    # fresco se dejan al prompt (bloque 5 vivo: «305 ml de queso parmesano», «¾ taza de queso parmesano»)
    (("leche descremada", "leche entera", "leche"), "leche UHT"),
    # [P1-PLAN-LOTE-215] víveres y pan que no llegan al fin de una compra única (plátano 7-10 días, yuca 21, pan 7)
    (("platano verde", "platano maduro", "platano", "rulo"), "batata"),
    (("yuca",), "batata"),
    (("pan de agua", "pan sobao", "pan integral", "pan de molde", "pan", "panecillo", "bagel"), "casabe"),
)
# tokens que NO se sustituyen aunque no aguanten: sin equivalente duradero coherente
SIN_SUSTITUTO = ("yogur", "yogurt", "cottage", "ricotta", "requeson", "queso fresco", "queso blanco", "queso de freir",
                 "leche de coco", "leche de almendra")
# la línea ya dice que es de despensa
_DURADERO_EN_TEXTO = ("en lata", "enlatad", "congelad", "seco", "secos", "en polvo", "deshidratad")

_RX_CANTIDAD = re.compile(
    r"^\s*([\d.,/½¼¾⅓⅔]+\s*(?:g|gr|gramos|ml|taza|tazas|cda|cdas|cdta|cdtas|unidad|unidades)?)\s+(?:de\s+)?",
    re.IGNORECASE)
_RX_GRAMOS = re.compile(r"≈\s*(\d+(?:[.,]\d+)?)\s*g\b", re.IGNORECASE)


def activo() -> bool:
    """tooltip-anchor: MEALFIT_SINGLE_TRIP_SAFE_SUBSTITUTE"""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_SINGLE_TRIP_SAFE_SUBSTITUTE", True)
    except Exception:
        return True


def proyeccion_activa() -> bool:
    """tooltip-anchor: MEALFIT_SINGLE_TRIP_CYCLE_PROJECTION"""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_SINGLE_TRIP_CYCLE_PROJECTION", True)
    except Exception:
        return True


def _sa(texto) -> str:
    try:
        from constants import strip_accents
        return strip_accents(str(texto or "")).lower()
    except Exception:
        return str(texto or "").lower()


# ─────────────────────────────────────────────────────────────── 214 · la sustitución segura

def alergias_de(contexto) -> list:
    """Alergias declaradas en un formulario / contexto clínico (`allergies` + `otherAllergies`)."""
    fd = contexto if isinstance(contexto, dict) else {}
    out = []
    for k in ("allergies", "otherAllergies"):
        v = fd.get(k)
        out.extend(v if isinstance(v, list) else ([v] if isinstance(v, str) and v.strip() else []))
    return [a for a in out if a]


def es_seguro(nombre: str, alergias=None, *, dieta=None, contexto=None) -> bool:
    """El mismo backstop que protege los platos del modelo (alergias, dieta dura y mercurio en embarazo).
    Fail-closed: si no se puede verificar, no es seguro. Sin alergias, dieta ni contexto no hay nada que verificar
    («balanced» no restringe nada: no cuenta como dieta)."""
    if dieta in (None, "", "balanced"):
        dieta = None
    if not alergias and not dieta and not contexto:
        return True
    try:
        go = sys.modules.get("graph_orchestrator")
        if go is None:
            import graph_orchestrator as go
        return not go.clinical_backstop_for_meal(
            {"meal": "Almuerzo", "name": nombre, "ingredients": [f"100 g de {nombre}"]},
            allergies=list(alergias or []), diet_type=dieta,
            form_data=contexto if isinstance(contexto, dict) else None)
    except Exception:
        return False


def sustituto_seguro(sub: str, semilla: int, vegetal: bool, alergias=None, *, dieta=None, contexto=None) -> Optional[str]:
    """El duradero para `sub`: la proteína rota por `semilla` (día absoluto + comida) y salta lo que choca con una
    alergia o la dieta; el resto de la tabla solo se comprueba. None si ninguno es seguro."""
    if not activo():
        return PROTEINA_VEGETAL if (sub == PROTEINA_TABLA and vegetal) else sub
    if sub == PROTEINA_TABLA:
        rot = _ROTACION_VEGETAL if vegetal else _ROTACION_OMNIVORA
        candidatos = [rot[(int(semilla) + k) % len(rot)] for k in range(len(rot))]
    else:
        candidatos = [sub]
    for c in candidatos:
        if es_seguro(c, alergias, dieta=dieta, contexto=contexto):
            return c
    logger.info(f"🧳 [P1-PLAN-LOTE-214] sin duradero seguro para «{sub}» (alergias {list(alergias or [])}): "
                f"se deja el fresco con su aviso de durabilidad")
    return None


def cantidad_de(texto: str) -> str:
    """El prefijo de cantidad de la línea para su sustituto. El peso aproximado manda sobre la pieza: «1 pechuga de
    pollo (≈200 g)» → «200 g de »; sin él, la cantidad inicial con su unidad («150 g de », «1 taza de »)."""
    t = str(texto or "")
    mg = _RX_GRAMOS.search(t)
    if mg:
        return f"{mg.group(1).replace(',', '.')} g de "
    mm = _RX_CANTIDAD.match(t)
    return (mm.group(1).strip() + " de ") if mm else ""


def _aguanta(texto: str, dia_abs: int, req: dict) -> bool:
    low = _sa(texto)
    if any(h in low for h in _DURADERO_EN_TEXTO):
        return True
    from pantry_durability import ingredient_issue_beyond_horizon
    return not ingredient_issue_beyond_horizon(str(texto), int(dia_abs), bool((req or {}).get("allow_frozen")))


def sustituir_linea(texto, dia_abs: int, req: Optional[dict], *, vegetal: bool = False, vegano: bool = False,
                    alergias=None, dieta=None, contexto=None, semilla: Optional[int] = None):
    """(línea nueva, sustituto, token que casó) si `texto` no aguanta hasta el día `dia_abs` (0-based) de la compra
    única y tiene un duradero seguro; None si aguanta, no tiene equivalente o ninguno es seguro.
    tooltip-anchor: P1-PLAN-LOTE-214-SUSTITUIR-LINEA"""
    if not req:
        return None
    text = str(texto or "")
    if not text or _aguanta(text, dia_abs, req):
        return None
    low = _sa(text)
    if any(t in low for t in SIN_SUSTITUTO):
        return None
    sub, hit = None, None
    for toks, rep in SUSTITUTOS:
        for t in toks:
            if re.search(r"\b" + re.escape(t) + r"s?\b", low):
                sub, hit = rep, t
                break
        if sub:
            break
    if not sub:
        return None
    if sub == "queso parmesano" and vegano:
        sub = PROTEINA_VEGETAL
    sub = sustituto_seguro(sub, int(dia_abs if semilla is None else semilla), vegetal, alergias,
                           dieta=dieta, contexto=contexto)
    if not sub:
        return None
    nueva = f"{cantidad_de(text)}{sub}"
    if nueva == text:
        return None
    return nueva, sub, hit


# ─────────────────────────────────────────────────────────────── 215 · la proyección del ciclo

def _politica(plan_data):
    """(política efectiva, offset del bloque). La del plan persistido; si no la lleva (el `result` de una corrida,
    antes de sellarse), la del formulario de la corrida (`nevera_exigida` la fija en un ContextVar)."""
    pp = plan_data.get("_plan_policy") if isinstance(plan_data, dict) else None
    if isinstance(pp, dict) and isinstance(pp.get("effective"), dict) and pp["effective"]:
        return pp["effective"], 0
    try:
        fd = sys.modules["nevera_exigida"]._FD.get() if "nevera_exigida" in sys.modules else None
    except Exception:
        fd = None
    if isinstance(fd, dict) and isinstance(fd.get("_plan_policy_effective"), dict):
        try:
            off = int(fd.get("_days_offset") or 0)
        except (TypeError, ValueError):
            off = 0
        return fd["_plan_policy_effective"], off
    return None, 0


def ciclo_de(plan_data) -> Optional[int]:
    """Días del ciclo de compra única de este plan, o None (no es compra única, knob apagado o bloque suelto)."""
    if not proyeccion_activa():
        return None
    eff, off = _politica(plan_data)
    if off or not isinstance(eff, dict):
        return None
    try:
        from horizon import single_trip_policy
        if not single_trip_policy(eff):
            return None
        return int((eff.get("shopping") or {}).get("main_cycle_days") or 0) or None
    except Exception:
        return None


def _texto_linea(ing) -> str:
    if isinstance(ing, str):
        return ing
    if isinstance(ing, dict):
        q = ing.get("quantity", 0)
        u = ing.get("unit", "unidad")
        n = ing.get("name") or ing.get("item_name") or ing.get("display_name") or ""
        try:
            return f"{q} {u} de {n}" if (q and float(q) > 0) else str(n)
        except (TypeError, ValueError):
            return str(n)
    return ""


def _lineas(meal: dict) -> list:
    ings = meal.get("ingredients_raw") or meal.get("ingredients") or []
    if not ings and isinstance(meal.get("recipe"), dict):
        ings = meal["recipe"].get("ingredients") or []
    return [t for t in (_texto_linea(x) for x in ings) if t]


_MEMO: "OrderedDict[str, list]" = OrderedDict()
_MEMO_MAX = 32


def _clave(reales, ciclo, eff) -> str:
    filas = [[[m.get("meal"), m.get("name"), _lineas(m)] for m in (d.get("meals") or []) if isinstance(m, dict)]
             for d in reales]
    blob = json.dumps([filas, ciclo, eff.get("shopping"), eff.get("diet")], ensure_ascii=False, sort_keys=True,
                      default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def dias_de_la_compra(plan_data, reales: list) -> list:
    """Los días de los que sale la lista: los `reales` y, si el plan es de compra única y aún no tiene todo el ciclo
    generado, los que faltan PROYECTADOS (copia en rueda de los reales, en su día absoluto, con la sustitución de
    duraderos). No mutar los días devueltos: los proyectados se memoizan.
    tooltip-anchor: P1-PLAN-LOTE-215-PROYECCION"""
    try:
        reales = [d for d in (reales or []) if isinstance(d, dict)]
        ciclo = ciclo_de(plan_data)
        n = len(reales)
        if not ciclo or n == 0 or n >= ciclo:
            return reales
        eff, _off = _politica(plan_data)
        clave = _clave(reales, ciclo, eff)
        proyectados = _MEMO.get(clave)
        if proyectados is None:
            proyectados = _proyectar(reales, ciclo, eff)
            _MEMO[clave] = proyectados
            while len(_MEMO) > _MEMO_MAX:
                _MEMO.popitem(last=False)
        else:
            _MEMO.move_to_end(clave)
        return reales + list(proyectados)
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-215] proyección no-op (fail-open): {type(e).__name__}: {e}")
        return [d for d in (reales or []) if isinstance(d, dict)]


def _proyectar(reales: list, ciclo: int, eff: dict) -> list:
    from pantry_durability import single_trip_requirements
    try:
        from constants import canonicalize_diet_type
        dieta = canonicalize_diet_type(((eff.get("diet") or {}).get("type")))
    except Exception:
        dieta = None
    vegetal, vegano = dieta in ("vegan", "vegetarian"), dieta == "vegan"
    alergias = [a for a in ((eff.get("diet") or {}).get("allergies") or []) if a]
    n = len(reales)
    out, cambios, fuera = [], 0, 0
    # La rueda de la proteína avanza con cada sustitución, no con el día: con 3 días reales y 3 duraderos, «día + comida»
    # le daba SIEMPRE el mismo duradero a cada día copiado (todo el pollo → sardinas, todo el pescado → garbanzos).
    rueda = 0
    for j in range(n, int(ciclo)):
        base = reales[j % n]
        req = single_trip_requirements(eff, j)
        meals = []
        for mi, m in enumerate(base.get("meals") or []):
            if not isinstance(m, dict):
                continue
            nuevas = []
            for t in _lineas(m):
                if req:
                    r = sustituir_linea(t, j, req, vegetal=vegetal, vegano=vegano, alergias=alergias,
                                        dieta=dieta, semilla=rueda)
                    if r:
                        t = r[0]
                        cambios += 1
                        if r[1] in _ROTACION_OMNIVORA or r[1] in _ROTACION_VEGETAL:
                            rueda += 1
                    elif not _aguanta(t, j, req):
                        fuera += 1
                        continue      # no aguanta su día y no tiene duradero: no se compra para ese día
                nuevas.append(t)
            meals.append({"meal": m.get("meal"), "name": m.get("name"), "ingredients": nuevas,
                          "ingredients_raw": list(nuevas), "_proyectado": True})
        out.append({"day": j + 1, "meals": meals, "_proyectado": True})
    logger.info(f"🧳 [P1-PLAN-LOTE-215] compra única de {ciclo} días: {n} días reales + {len(out)} proyectados "
                f"({cambios} líneas a su duradero, {fuera} que no aguantan fuera de la compra)")
    return out


def sellar_lista(res, plan_data):
    """Sella cada ítem de una lista de compra única con los días de su ciclo (`_compra_unica`). Muta y devuelve."""
    try:
        ciclo = ciclo_de(plan_data)
        if not ciclo:
            return res
        items = res if isinstance(res, list) else (
            [x for v in res.values() if isinstance(v, list) for x in v] if isinstance(res, dict) else [])
        for it in items:
            if isinstance(it, dict):
                it["_compra_unica"] = int(ciclo)
    except Exception:
        pass
    return res


def ciclo_de_lista(items) -> int:
    """Los días del ciclo si la lista es de una compra única (sello `_compra_unica`), 0 si no."""
    try:
        return max([int(i.get("_compra_unica") or 0) for i in (items or []) if isinstance(i, dict)] or [0])
    except Exception:
        return 0
