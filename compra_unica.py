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
# [P1-PLAN-LOTE-465 · 2026-09-27] Los garbanzos cocidos tienen ~9 g de proteína por 100 g; la pechuga que sustituyen,
# ~23 crudos. En la rueda de tres, un tercio de las sustituciones de un omnívoro caía en garbanzos y el día perdía
# proteína que ningún pase podía devolver (los topes de realismo cortan la legumbre antes): replay del chain completo
# forzando la compra única, días con garbanzos a 0,57-0,71 del objetivo de proteína. Para un omnívoro la rueda es de
# pescado en lata (atún en agua, sardinas) y los garbanzos quedan de RESERVA: sólo si los dos pescados ya están en el día
# o no son seguros (alergia al pescado, embarazo con el tope de mercurio, rechazo). tooltip-anchor: P1-PLAN-LOTE-465
_ROTACION_OMNIVORA = ("atun en agua", "sardinas en lata")
# [P1-PLAN-LOTE-495 · 2026-09-27] …y con alergia al pescado la rueda entera caía en garbanzos: el band-closer los trata
# como carbohidrato y los recorta. Batería real del 27-sep (alérgico al pescado, días 4-7 de 30 sin congelador): la cena
# «1¾ pechugas de pollo» (71 g de proteína) salió con 70 g de garbanzos (6 g); el día, al 43 % de su proteína. Las claras
# pasteurizadas (botella de 400 g del súper, 35 días en nevera) van ANTES: proteína casi pura que el piso sí puede subir.
# Hasta MAX_EGG_WHITES_PER_MEAL por comida; con «Nada» de tiempo no (se cocinan). tooltip-anchor: P1-PLAN-LOTE-495
CLARAS = "claras de huevo"
_RESERVA_OMNIVORA = (CLARAS, "garbanzos cocidos")
_RESERVA_SIN_ROTAR = (CLARAS,)        # su proteína manda sobre la variedad del día: no se aparta por `evitar`
_G_CLARA = 33.0
_ROTACION_VEGETAL = ("garbanzos cocidos", "lentejas cocidas")
# [P1-PLAN-LOTE-496] los duraderos que sustituyen una PROTEÍNA: dentro de un plato, todas sus proteínas van al mismo
PROTEINAS_DURADERAS = frozenset(_ROTACION_OMNIVORA + _RESERVA_OMNIVORA + _ROTACION_VEGETAL)
# [P1-PLAN-LOTE-216] cómo se reconoce en una línea que el día YA lleva ese duradero
_CLAVE_ROTACION = {"atun en agua": "atun", "sardinas en lata": "sardina", "garbanzos cocidos": "garbanzo",
                   "lentejas cocidas": "lenteja", CLARAS: "clara"}
# [P1-PLAN-LOTE-286 · 2026-09-25] Con «Nada» de tiempo la legumbre duradera se compra LISTA. «200 g de garbanzos
# cocidos» dejaba en la lista del dueño (30 días, sin congelador, «Nada») 1 funda de garbanzos SECOS: remojo de una noche
# y una hora de olla para quien declaró 5 minutos. La línea dice «de lata, escurridos» y la lista (lote 285) compra la
# lata; la identidad del duradero (`sub`, la rueda, `evitar`) no cambia. tooltip-anchor: P1-PLAN-LOTE-286-DURADERO-LISTO
_LISTO_SIN_TIEMPO = {"garbanzos cocidos": "garbanzos de lata, escurridos",
                     "lentejas cocidas": "lentejas de lata, escurridas"}


def sin_tiempo(contexto=None, plan_data=None) -> bool:
    """¿El usuario declaró «Nada» de tiempo? El formulario (`contexto`), el sello del plan (`_cooking_time`, lote 220)
    o, en una corrida sin sellar, el formulario que fijó `nevera_exigida`."""
    try:
        for fuente in (contexto, plan_data):
            if isinstance(fuente, dict):
                v = fuente.get("cookingTime") if "cookingTime" in fuente else fuente.get("_cooking_time")
                if v is not None:
                    return str(v).strip().lower() == "none"
        fd = sys.modules["nevera_exigida"]._FD.get() if "nevera_exigida" in sys.modules else None
        return isinstance(fd, dict) and str(fd.get("cookingTime") or "").strip().lower() == "none"
    except Exception:
        return False


def duraderos_del_dia(lineas) -> set:
    """Los duraderos de la rueda que ya aparecen en estas líneas (para no repetirlos el mismo día)."""
    presentes = set()
    for t in lineas or ():
        low = _sa(t)
        for sub, clave in _CLAVE_ROTACION.items():
            if clave in low:
                presentes.add(sub)
    return presentes

# (tokens, duradero). Primer match gana; los tokens se buscan como palabra (singular/plural) en la línea sin acentos.
# Era `graph_orchestrator._FRESH_SUBSTITUTES` (P1-STEP14-SHOPPING-COOKING); allí queda un alias.
SUSTITUTOS = (
    # [P1-PLAN-LOTE-497 · 2026-09-27] el bok choy aguanta ~10 días: el día 15 de la batería real seguía en la lista
    (("lechuga", "berro", "rucula", "arugula", "espinaca", "acelga", "kale", "col rizada", "bok choy", "pak choi"),
     "repollo"),
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
    # [P1-PLAN-LOTE-468 · 2026-09-27] el edamame del súper es congelado de fábrica: sin congelador no llega al día 4
    # (batería real del 27-sep, alérgico al pescado: 300 g tres días seguidos). La legumbre de despensa lo sustituye.
    (("edamame",), "garbanzos cocidos"),
    (("pan de agua", "pan sobao", "pan integral", "pan de molde", "pan", "panecillo", "bagel"), "casabe"),
)
# tokens que NO se sustituyen aunque no aguanten: sin equivalente duradero coherente
SIN_SUSTITUTO = ("yogur", "yogurt", "cottage", "ricotta", "requeson", "queso fresco", "queso blanco", "queso de freir",
                 "leche de coco", "leche de almendra")
# [P1-PLAN-LOTE-460 · 2026-09-27] cómo se ESCRIBE el duradero en la línea que ve el usuario: la identidad (`sub`, la rueda,
# `evitar`) sigue sin tilde, pero «150 g de atun en agua» y «oregano» llegaban así a su lista de ingredientes.
_VISIBLE = {"atun en agua": "atún en agua", "oregano": "orégano"}
# [P1-PLAN-LOTE-466 · 2026-09-27] El tomate que el plato COCINA (guiso, salsa, sofrito) pasa a salsa de tomate de
# despensa —«Salsa de tomate» vive en el catálogo—, no a zanahoria: replay forzado de los días 21+ (88 planes), 95 de 252
# tomates sustituidos iban dentro de un guiso o una salsa y salían «salsa de zanahoria». La salsa es ~2,5 veces más
# concentrada: 150 g de tomate fresco → 60 g de salsa. El tomate crudo (ensalada) sigue a zanahoria.
# tooltip-anchor: P1-PLAN-LOTE-466-TOMATE-COCINADO
SALSA_TOMATE = "salsa de tomate"
_FACTOR_SALSA = 0.4
_TOMATE_COCINADO = re.compile(
    r"\b(?:sofr[ií]\w*|sofrito|guis\w*|salsa|estofad\w*|sancocho|locrio|moro)\b"
    r"|\bcocina[^.;]{0,60}\btomate|\btomate[^.;]{0,60}\b(?:cocina\w*|sofr\w*|guis\w*|hierv\w*|cuece|salte\w*|reduc\w*)",
    re.IGNORECASE)


def texto_plato(meal) -> str:
    """Nombre + pasos del plato: el contexto con el que el tomate decide si va a salsa."""
    try:
        if not isinstance(meal, dict):
            return ""
        pasos = [str(p) for p in (meal.get("recipe") or []) if isinstance(p, str)]
        return " ".join([str(meal.get("name") or "")] + pasos)
    except Exception:
        return ""


# la línea ya dice que es de despensa
_DURADERO_EN_TEXTO = ("en lata", "enlatad", "congelad", "seco", "secos", "en polvo", "deshidratad")

_RX_CANTIDAD = re.compile(
    r"^\s*([\d.,/½¼¾⅓⅔]+\s*(?:g|gr|gramos|ml|taza|tazas|cda|cdas|cdta|cdtas|unidad|unidades)?)\s+(?:de\s+)?",
    re.IGNORECASE)
_RX_GRAMOS = re.compile(r"≈\s*(\d+(?:[.,]\d+)?)\s*g\b", re.IGNORECASE)
# [P1-PLAN-LOTE-462] una MEDIDA (peso o volumen) se conserva tal cual: «2 tazas de lechuga» → «2 tazas de repollo»
_RX_MEDIDA = re.compile(
    r"^\s*([\d.,/½¼¾⅓⅔]+)\s+((?:g|gr|gramos|kg|ml|l|litros?|tazas?|cdas?|cdtas?|cucharadas?|cucharaditas?|onzas?|oz|lb|"
    r"libras?|pizcas?|puñados?|latas?|sobres?)\.?)\s+(?:de\s+)?", re.IGNORECASE)


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


def sustituto_seguro(sub: str, semilla: int, vegetal: bool, alergias=None, *, dieta=None, contexto=None,
                     evitar=(), excluir=()) -> Optional[str]:
    """El duradero para `sub`: la proteína rota por `semilla` (día absoluto + comida) y salta lo que choca con una
    alergia o la dieta; lo que el día ya lleva (`evitar`) va al final de la rueda. El resto de la tabla solo se
    comprueba. None si ninguno es seguro."""
    if not activo():
        return PROTEINA_VEGETAL if (sub == PROTEINA_TABLA and vegetal) else sub
    if sub == PROTEINA_TABLA:
        rot = _ROTACION_VEGETAL if vegetal else _ROTACION_OMNIVORA
        candidatos = [rot[(int(semilla) + k) % len(rot)] for k in range(len(rot))]
        reserva = [] if vegetal else [c for c in _RESERVA_OMNIVORA if c not in candidatos]   # [P1-PLAN-LOTE-465]
        reserva = [c for c in reserva if c not in (excluir or ())]                          # [P1-PLAN-LOTE-495]
        fijas = [c for c in reserva if c in _RESERVA_SIN_ROTAR]
        reserva = [c for c in reserva if c not in _RESERVA_SIN_ROTAR]
        candidatos = ([c for c in candidatos if c not in evitar] + fijas + [c for c in reserva if c not in evitar]
                      + [c for c in candidatos if c in evitar] + [c for c in reserva if c in evitar])
    else:
        candidatos = [sub]
    for c in candidatos:
        if es_seguro(c, alergias, dieta=dieta, contexto=contexto):
            return c
    logger.info(f"🧳 [P1-PLAN-LOTE-214] sin duradero seguro para «{sub}» (alergias {list(alergias or [])}): "
                f"se deja el fresco con su aviso de durabilidad")
    return None


# [P1-PLAN-LOTE-469 · 2026-09-27] «1 rebanada de pan integral familiar» → «1 torta pequeña de casabe». La «rebanada» no es
# una unidad que el catálogo pese: sin gramos, la línea salía «1 de casabe» (display «1 casabe») y el paso «ten listas
# 1 rebanada de casabe familiar» (batería real del 27-sep). La torta pequeña (≈30 g) pesa lo que una rebanada.
# tooltip-anchor: P1-PLAN-LOTE-469-CASABE-EN-TORTAS
_FRACCION = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3}


def _cuenta(num: str) -> Optional[float]:
    m = re.fullmatch(r"(\d+(?:[.,]\d+)?)?\s*([½¼¾⅓⅔])?", str(num or "").strip())
    if not m or not (m.group(1) or m.group(2)):
        return None
    return float((m.group(1) or "0").replace(",", ".")) + _FRACCION.get(m.group(2) or "", 0.0)


def casabe_en_tortas(texto: str) -> Optional[str]:
    """«N piezas» de pan sin peso conocido → «N torta(s) pequeña(s) de casabe»; None si la línea trae peso o medida."""
    t = str(texto or "")
    if _RX_GRAMOS.search(t) or _RX_MEDIDA.match(t):
        return None
    mm = _RX_CANTIDAD.match(t)
    if not mm:
        return None
    num = mm.group(1).strip()
    val = _cuenta(num)
    if not val or val <= 0:
        return None
    return f"{num} {'torta pequeña' if val <= 1 else 'tortas pequeñas'} de casabe"


# [P1-PLAN-LOTE-490 · 2026-09-27] La hierba fresca pasa a orégano SECO, unas tres veces más fuerte: «¼ taza de cilantro
# picado» salía «¼ taza de orégano», «75 g de cilantro» «75 g de orégano» y «½ ramita de cilantro» «½ de orégano», sin
# unidad (replay forzado de los días 21+, 322 planes: 1.061 hierbas sustituidas; 45 en tazas, 18 en gramos, 94 en
# ramitas). Regla de cocina: 1 cda de hierba fresca = 1 cdta de seca; tope 1 cda por línea.
# tooltip-anchor: P1-PLAN-LOTE-490-OREGANO-SECO
_CDTAS_FRESCA = {"taza": 48.0, "tazas": 48.0, "cda": 3.0, "cdas": 3.0, "cucharada": 3.0, "cucharadas": 3.0,
                 "cdta": 1.0, "cdtas": 1.0, "cucharadita": 1.0, "cucharaditas": 1.0, "ramita": 1.5, "ramitas": 1.5,
                 "tallo": 6.0, "tallos": 6.0, "hoja": 0.25, "hojas": 0.25, "punado": 12.0, "punados": 12.0,
                 "g": 3.0, "gr": 3.0, "gramos": 3.0}
_CDTAS_PIEZA_HIERBA = 6.0      # «2½ cebollín picado»: la pieza entera, ≈ 2 cdas picada
_QUEBRADO = {0.0: "", 0.25: "¼", 0.5: "½", 0.75: "¾"}


def oregano_seco(texto: str) -> Optional[str]:
    """«2 cdas de cilantro picado» → «2 cdtas de orégano»; None si la línea no dice cuánto («Cilantro fresco»)."""
    m = re.match(r"^\s*([\d.,]*\s*[½¼¾⅓⅔]?)\s*([a-záéíóúñ]+)?", str(texto or ""), re.IGNORECASE)
    if not m or not m.group(1).strip():
        return None
    val = _cuenta(m.group(1))
    if not val or val <= 0:
        return None
    unidad = _sa(m.group(2) or "")
    frescas = val * _CDTAS_FRESCA.get(unidad, _CDTAS_PIEZA_HIERBA)
    secas = min(3.0, max(0.25, round(frescas / 3.0 * 4) / 4))
    if secas >= 3.0:
        return "1 cda de orégano"
    entero = int(secas)
    cifra = (str(entero) if entero else "") + _QUEBRADO.get(round(secas - entero, 2), "")
    return f"{cifra} {'cdta' if secas <= 1 else 'cdtas'} de orégano"


def _max_claras() -> int:
    go = sys.modules.get("graph_orchestrator")
    try:
        return max(1, int(getattr(go, "MAX_EGG_WHITES_PER_MEAL", 6)))
    except Exception:
        return 6


def claras_de(texto: str, gramos: Optional[float] = None) -> str:
    """[P1-PLAN-LOTE-495] «1¾ pechugas de pollo (≈279 g)» → «6 claras de huevo»: una clara por cada ~33 g de la
    proteína que sustituye, entre 3 y el tope por comida (sin peso conocido, 4)."""
    t = str(texto or "")
    g = None
    mg = _RX_GRAMOS.search(t)
    md = _RX_MEDIDA.match(t)
    try:
        if mg:
            g = float(mg.group(1).replace(",", "."))
        elif md and md.group(2).lower().rstrip(".") in ("g", "gr", "gramos"):
            g = float(md.group(1).replace(",", "."))
        elif gramos:
            g = float(gramos)
    except (TypeError, ValueError):
        g = None
    n = int(round(g / _G_CLARA)) if g and g > 0 else 4
    return f"{max(3, min(_max_claras(), n))} {CLARAS}"


def candidatos_del_dia(cands, dia, form_data=None):
    """[P1-PLAN-LOTE-521 · 2026-09-27] Las proteínas que el cerrador de proteína puede AÑADIR a un plato del día `dia`
    (el dict del día; su «day» es absoluto también en los bloques): en la compra única sin congelador, desde el día 4 sólo
    las que aguantan hasta ese día. Batería real del 27-sep (alérgico al pescado, días 8-11): el cerrador añadía «1¼
    pechugas de pollo (≈255 g)» y «y pechuga de pollo» al nombre del día 8, y la sustitución de la compra única la cambiaba
    después por claras topadas a 6 (22 g de proteína donde el cerrador había contado 59). Eligiendo aquí lo que dura (atún,
    sardinas, claras, huevo…), el cerrador cuenta la densidad de lo que de verdad se sirve. Sin política de compra única,
    antes del día 4 o si nada de la lista aguanta, la lista de siempre. tooltip-anchor: P1-PLAN-LOTE-521"""
    try:
        if not cands or not isinstance(dia, dict):
            return cands
        n = int(dia.get("day") or 0) - 1
        if n < 3:
            return cands
        fd = form_data if isinstance(form_data, dict) else {}
        eff = fd.get("_plan_policy_effective") if isinstance(fd.get("_plan_policy_effective"), dict) else None
        if eff is None:
            eff, _off = _politica({})
        if not isinstance(eff, dict):
            return cands
        from pantry_durability import single_trip_requirements
        req = single_trip_requirements(eff, n)
        if not req:
            return cands
        out = [c for c in cands if _aguanta(f"100 g de {c}", n, req)]
        return out or cands
    except Exception:
        return cands


def _redondea(gramos: float) -> str:
    g = float(gramos)
    return str(int(round(g))) if g < 20 else str(int(5 * round(g / 5.0)))


def cantidad_de(texto: str, gramos: Optional[float] = None) -> str:
    """El prefijo de cantidad de la línea para su sustituto. El peso aproximado manda sobre la pieza: «1 pechuga de
    pollo (≈200 g)» → «200 g de »; sin él, la medida inicial («150 g de », «1 taza de »).
    [P1-PLAN-LOTE-462 · 2026-09-27] Una PIEZA sin peso («1¼ filetes de pescado», «2 tomates») no se copia como número
    suelto: «1¼ de sardinas en lata» no dice cuánto, la receta lo leía como UNA sardina y la lista compraba por unidad
    de lata. Con `gramos` (el peso de la pieza, del resolvedor del catálogo) sale «190 g de »; sin él, como antes."""
    t = str(texto or "")
    mg = _RX_GRAMOS.search(t)
    if mg:
        return f"{mg.group(1).replace(',', '.')} g de "
    md = _RX_MEDIDA.match(t)
    if md:
        return f"{md.group(1)} {md.group(2)} de "
    try:
        if gramos and float(gramos) > 0:
            return f"{_redondea(gramos)} g de "
    except (TypeError, ValueError):
        pass
    mm = _RX_CANTIDAD.match(t)
    return (mm.group(1).strip() + " de ") if mm else ""


def _gramos_de_linea(texto: str) -> Optional[float]:
    """Peso de la línea según el resolvedor del catálogo del grafo, si está cargado (el proceso del backend); None si
    no. Sin red nueva: el grafo memoiza por línea y la lista hace el mismo parseo justo después."""
    try:
        go = sys.modules.get("graph_orchestrator")
        if go is None or not hasattr(go, "_resolve_line_food_grams"):
            return None
        g = go._resolve_line_food_grams(str(texto))[1]
        return float(g) if g else None
    except Exception:
        return None


def _aguanta(texto: str, dia_abs: int, req: dict) -> bool:
    low = _sa(texto)
    pistas = [h for h in _DURADERO_EN_TEXTO if h in low]
    # [P1-PLAN-LOTE-289 · 2026-09-25] «congelado» es despensa SOLO con congelador, y «previamente congelado» /
    # «descongelado» ya no lo es: «250 g de salmón previamente congelado» del plan del dueño (sin congelador) pasaba por
    # duradero y la proyección lo copiaba en los 30 días — 2,5 kg de salmón en la lista del mes.
    # tooltip-anchor: P1-PLAN-LOTE-289-CONGELADO-SIN-CONGELADOR
    if pistas and (any(h != "congelad" for h in pistas) or (
            bool((req or {}).get("allow_frozen")) and not re.search(r"previamente congelad|descongelad", low))):
        return True
    from pantry_durability import ingredient_issue_beyond_horizon
    return not ingredient_issue_beyond_horizon(str(texto), int(dia_abs), bool((req or {}).get("allow_frozen")))


def sustituir_linea(texto, dia_abs: int, req: Optional[dict], *, vegetal: bool = False, vegano: bool = False,
                    alergias=None, dieta=None, contexto=None, semilla: Optional[int] = None, evitar=(),
                    listo: Optional[bool] = None, gramos_de=None, plato: str = "", forzar: Optional[str] = None):
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
    # [P1-PLAN-LOTE-491 · 2026-09-27] «½ taza de caldo de pescado o agua» no es pescado: el caldo (cubito, tetrabrik) es
    # despensa. Salía «½ taza de atún en agua» y el guiso «incorpora las sardinas y el caldo… hasta que el atún se
    # impregne» (replay forzado de los días 21+). Igual una crema de leche o un jugo hechos DEL alimento.
    # tooltip-anchor: P1-PLAN-LOTE-491-PRODUCTO-DE
    if re.search(r"\b(?:caldo|consome|fondo|cubitos?|crema|jugo|zumo|pasta|pure|salsa|sopa|polvo|harina|esencia|"
                 r"extracto)\s+(?:[a-z]+\s+)?de\s+(?:[a-z]+\s+)?" + re.escape(hit) + r"(?:s|es)?\b", low):
        return None
    if sub == "queso parmesano" and vegano:
        sub = PROTEINA_VEGETAL
    if hit in ("tomate", "tomate cherry") and plato and _TOMATE_COCINADO.search(_sa(plato)):   # [P1-PLAN-LOTE-466]
        sub = SALSA_TOMATE
    if listo is None:
        listo = sin_tiempo(contexto)
    # [P1-PLAN-LOTE-496 · 2026-09-27] la segunda proteína de un plato recibe el duradero de la primera (`forzar`): antes la
    # rueda le daba otro y el plato quedaba «Atún con zanahoria… Acompaña con atún y sardinas en lata».
    # tooltip-anchor: P1-PLAN-LOTE-496-FORZAR
    if (sub == PROTEINA_TABLA and forzar in PROTEINAS_DURADERAS and not (listo and forzar == CLARAS)
            and es_seguro(forzar, alergias, dieta=dieta, contexto=contexto)):
        sub = forzar
    else:
        sub = sustituto_seguro(sub, int(dia_abs if semilla is None else semilla), vegetal, alergias,
                               dieta=dieta, contexto=contexto, evitar=evitar,
                               excluir=(CLARAS,) if listo else ())                            # [P1-PLAN-LOTE-495]
    if not sub:
        return None
    gramos = None
    if gramos_de is not None and not _RX_GRAMOS.search(text) and not _RX_MEDIDA.match(text):   # [P1-PLAN-LOTE-462]
        try:
            gramos = gramos_de(text)
        except Exception:
            gramos = None
    visible = (_LISTO_SIN_TIEMPO.get(sub) if listo else None) or _VISIBLE.get(sub, sub)     # [P1-PLAN-LOTE-460]
    nueva = f"{cantidad_de(text, gramos)}{visible}"  # [P1-PLAN-LOTE-286]
    if sub == "casabe" and not gramos:                                                        # [P1-PLAN-LOTE-469]
        nueva = casabe_en_tortas(text) or nueva
    if sub == "oregano":                                                                      # [P1-PLAN-LOTE-490]
        nueva = oregano_seco(text) or nueva
    if sub == CLARAS:                                                                         # [P1-PLAN-LOTE-495]
        nueva = claras_de(text, gramos)
    if sub == SALSA_TOMATE:                                                                    # [P1-PLAN-LOTE-466]
        g = None
        mg = _RX_GRAMOS.search(text)
        md = _RX_MEDIDA.match(text)
        if mg:
            g = float(mg.group(1).replace(",", "."))
        elif md and md.group(2).lower().rstrip(".") in ("g", "gr", "gramos"):
            try:
                g = float(md.group(1).replace(",", "."))
            except ValueError:
                g = None
        if g is None and gramos_de is not None:
            try:
                g = gramos_de(text)
            except Exception:
                g = None
        nueva = (f"{_redondea(g * _FACTOR_SALSA)} g de {SALSA_TOMATE}" if g and g > 0
                 else f"¼ taza de {SALSA_TOMATE}")
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
        listo = sin_tiempo(None, plan_data)                      # [P1-PLAN-LOTE-286]
        clave = _clave(reales, ciclo, eff) + ("|listo" if listo else "")
        proyectados = _MEMO.get(clave)
        if proyectados is None:
            proyectados = _proyectar(reales, ciclo, eff, listo=listo)
            _MEMO[clave] = proyectados
            while len(_MEMO) > _MEMO_MAX:
                _MEMO.popitem(last=False)
        else:
            _MEMO.move_to_end(clave)
        return reales + list(proyectados)
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-215] proyección no-op (fail-open): {type(e).__name__}: {e}")
        return [d for d in (reales or []) if isinstance(d, dict)]


def _proyectar(reales: list, ciclo: int, eff: dict, listo: bool = False) -> list:
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
        presentes = duraderos_del_dia(t for m in (base.get("meals") or []) if isinstance(m, dict) for t in _lineas(m))
        for mi, m in enumerate(base.get("meals") or []):
            if not isinstance(m, dict):
                continue
            nuevas = []
            for t in _lineas(m):
                if req:
                    r = sustituir_linea(t, j, req, vegetal=vegetal, vegano=vegano, alergias=alergias,
                                        dieta=dieta, semilla=rueda, evitar=presentes, listo=listo,
                                        gramos_de=_gramos_de_linea,  # [P1-PLAN-LOTE-462]
                                        plato=str(m.get("name") or ""))  # [P1-PLAN-LOTE-466]
                    if r:
                        t = r[0]
                        cambios += 1
                        presentes.add(r[1])
                        if r[1] in _ROTACION_OMNIVORA or r[1] in _ROTACION_VEGETAL or r[1] in _RESERVA_OMNIVORA:
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
        # [P1-PLAN-LOTE-221 · 2026-09-24] …y si el ciclo va sin congelador, para que la nota de cobertura de un fresco
        # diga cuánto aguanta EN LA NEVERA (el filete de 32 oz salía «alcanza ~14 días» sin congelador).
        sin_congelador = False
        try:
            from pantry_durability import freeze_window_days
            eff, _off = _politica(plan_data)
            sin_congelador = freeze_window_days(((eff or {}).get("shopping") or {}).get("freezer_mode"), int(ciclo)) <= 0
        except Exception:
            sin_congelador = False
        items = res if isinstance(res, list) else (
            [x for v in res.values() if isinstance(v, list) for x in v] if isinstance(res, dict) else [])
        for it in items:
            if isinstance(it, dict):
                it["_compra_unica"] = int(ciclo)
                if sin_congelador:
                    it["_compra_unica_sin_congelador"] = True
    except Exception:
        pass
    return res


def ciclo_de_lista(items) -> int:
    """Los días del ciclo si la lista es de una compra única (sello `_compra_unica`), 0 si no."""
    try:
        return max([int(i.get("_compra_unica") or 0) for i in (items or []) if isinstance(i, dict)] or [0])
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────── 216 · la Nevera virtual
# [P1-PLAN-LOTE-216 · 2026-09-24] La lista del día 1 alcanza para el ciclo (215), pero nada obligaba a los bloques
# siguientes a COCINAR con lo comprado: sin Nevera (vacía porque el usuario no marcó «Ya compré», o apagada) el bloque
# se generaba libre, podía traer alimentos que no se compraron y la lista del ciclo crecía a mitad de mes. En una compra
# única, lo comprado ES la Nevera: el bloque 2+ recibe como Nevera la compra del ciclo (los nombres de su lista, sin
# cantidades) y el revisor la exige como a cualquier Nevera (espejo `nevera_exigida.lista`), así que el sembrador
# (regla b) y los cerradores (lote 199) eligen de ella. Con la Nevera real en uso (con alimentos), manda la real.
# Sin cantidades no hay reservas que medir: las guardas de Nevera la eximen (`_pantry_gate_waiver_reason`).
# Knob `MEALFIT_SINGLE_TRIP_VIRTUAL_PANTRY`.

_SQL_LISTAS_DEL_CICLO = """
SELECT mp.plan_data->'aggregated_shopping_list' AS activa,
       mp.plan_data->'aggregated_shopping_list_monthly' AS mensual,
       mp.plan_data->'aggregated_shopping_list_biweekly' AS quincenal
  FROM plan_chunk_queue q JOIN meal_plans mp ON mp.id = q.meal_plan_id
 WHERE q.id = %s AND mp.user_id = %s
"""


def nevera_virtual_activa() -> bool:
    """tooltip-anchor: MEALFIT_SINGLE_TRIP_VIRTUAL_PANTRY"""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_SINGLE_TRIP_VIRTUAL_PANTRY", True)
    except Exception:
        return True


def _lista_del_ciclo(fila: dict, ciclo: int) -> list:
    """La lista de la compra del ciclo: la que lleva el sello `_compra_unica` de ESTE ciclo; si ninguna, la del
    periodo que le corresponde (30 → mensual, 15 → quincenal)."""
    candidatas = [fila.get("activa"), fila.get("mensual"), fila.get("quincenal")]
    for lista in candidatas:
        if isinstance(lista, list) and ciclo_de_lista(lista) == int(ciclo):
            return lista
    lista = fila.get("mensual") if int(ciclo) >= 30 else fila.get("quincenal")
    return lista if isinstance(lista, list) else []


def nevera_virtual(form_data, task_id=None, user_id=None, consultar=None):
    """El bloque 2+ de una compra única sin Nevera real recibe como Nevera la compra del ciclo. Muta y devuelve
    `form_data`; fail-open (sin cambios) ante cualquier duda. tooltip-anchor: P1-PLAN-LOTE-216-NEVERA-VIRTUAL"""
    try:
        if not nevera_virtual_activa() or not isinstance(form_data, dict) or form_data.get("_pantry_paused"):
            return form_data
        eff = form_data.get("_plan_policy_effective")
        from horizon import single_trip_policy
        if not single_trip_policy(eff):
            return form_data
        if int(form_data.get("_days_offset") or 0) <= 0:
            return form_data
        real = [x for x in (form_data.get("current_pantry_ingredients") or []) if x]
        if real and not form_data.get("_nevera_apagada"):
            return form_data            # la Nevera real manda
        if not task_id or not user_id or user_id == "guest":
            return form_data
        if consultar is None:
            from db import execute_sql_query as consultar
        fila = consultar(_SQL_LISTAS_DEL_CICLO, (task_id, user_id), fetch_one=True) or {}
        ciclo = int(((eff or {}).get("shopping") or {}).get("main_cycle_days") or 0)
        # [P1-PLAN-LOTE-221 · 2026-09-24] Sólo lo que LLEGA a este bloque: el yogur o el pescado del día 1 no están en la
        # nevera del día 22 (sin congelador el filete aguanta 3 días). Mismo criterio que la proyección
        # (`_aguanta` + `single_trip_requirements`); sin exigencia para el día del bloque, todo vale.
        # tooltip-anchor: P1-PLAN-LOTE-221-NEVERA-VIRTUAL-LO-QUE-LLEGA
        dia0 = int(form_data.get("_days_offset") or 0)
        try:
            from pantry_durability import single_trip_requirements
            req = single_trip_requirements(eff, dia0)
        except Exception:
            req = None
        nombres, vistos = [], set()
        for it in _lista_del_ciclo(fila, ciclo):
            if not isinstance(it, dict) or str(it.get("category") or "").startswith("🚨"):
                continue
            n = str(it.get("name") or "").strip()
            k = _sa(n)
            if n and req and not _aguanta(n, dia0, req):
                continue
            if n and k not in vistos:
                vistos.add(k)
                nombres.append(n)
        if len(nombres) < 4:
            return form_data
        form_data["current_pantry_ingredients"] = nombres
        form_data["_fresh_pantry_source"] = "compra_unica_virtual"
        form_data["_nevera_virtual"] = True
        form_data.pop("_pantry_advisory_only", None)   # la compra del ciclo SÍ se exige: es lo que hay en casa
        logger.info(f"🧳 [P1-PLAN-LOTE-216] compra única sin Nevera real (user {str(user_id)[:8]}, bloque desde el día "
                    f"{int(form_data.get('_days_offset') or 0) + 1}): el bloque cocina con la compra del ciclo "
                    f"({len(nombres)} alimentos).")
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-216] Nevera virtual no-op (fail-open): {type(e).__name__}: {e}")
    return form_data
