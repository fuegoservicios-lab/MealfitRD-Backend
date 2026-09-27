# backend/banco_analizador.py
"""[P1-PLAN-LOTE-570 · 2026-09-27] Banco de pruebas del analizador de fotos — la parte PURA.

Spec: docs/superpowers/specs/2026-09-27-analizador-banco-de-pruebas-design.md. Aquí no hay red ni DB: métricas, lectura
de Nutrition5k, el muestreo congelado y el emparejado de componentes POR CLASE (palabra entera, nunca subcadena: «pollo»
vive dentro de «repollo» y «res» dentro de «fresco»). La corrida vive en scripts/banco_analizador_correr.py.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Callable, Iterable, Optional

SEMILLA = 20260927
MANIFIESTO = Path(__file__).resolve().parent / "data" / "banco_analizador" / "manifest.json"
PISO_KCAL = 100.0
PISO_MACRO_G = 10.0
UMBRAL_COMPONENTE = 0.15
UMBRAL_FALLOS_VALIDA = 0.10
TRAMOS_KCAL = ((100.0, 300.0), (300.0, 600.0), (600.0, math.inf))
# (clave en la verdad, clave en la respuesta del analizador, piso del error relativo)
MACROS = (("kcal", "calories", PISO_KCAL), ("proteina_g", "protein", PISO_MACRO_G),
          ("carbs_g", "carbs", PISO_MACRO_G), ("grasa_g", "healthy_fats", PISO_MACRO_G))

# Orden = prioridad: la primera clase cuyo vocabulario aparezca como PALABRA en el nombre gana.
_CLASES = (
    ("pizza", {"en": {"pizza"}, "es": {"pizza"}}),
    ("chicken", {"en": {"chicken", "turkey"}, "es": {"pollo", "pechuga", "muslo", "alitas", "gallina", "pavo"}}),
    ("fish", {"en": {"fish", "salmon", "tuna", "cod", "tilapia", "shrimp", "halibut", "trout", "seafood", "crab"},
              "es": {"pescado", "salmon", "atun", "bacalao", "tilapia", "camaron", "mariscos", "merluza",
                     "dorado", "chillo", "trucha", "sardina", "pulpo", "calamar", "langosta", "cangrejo", "jaiba"}}),
    ("pork", {"en": {"pork", "bacon", "ham", "sausage", "prosciutto", "chorizo", "pepperoni"},
              "es": {"cerdo", "puerco", "tocino", "tocineta", "jamon", "chuleta", "salchicha", "longaniza",
                     "chorizo", "pernil", "chicharron"}}),
    ("beef", {"en": {"beef", "steak", "burger", "meatball", "meatballs", "brisket"},
              "es": {"res", "carne", "bistec", "churrasco", "entrana", "hamburguesa", "albondiga", "picadillo",
                     "filete"}}),
    ("egg", {"en": {"egg", "eggs", "omelet", "omelette"}, "es": {"huevo", "huevos", "omelet", "omelette", "revoltillo"}}),
    ("cheese", {"en": {"cheese", "mozzarella", "parmesan", "cheddar", "feta", "ricotta"},
                "es": {"queso", "quesos", "mozzarella", "parmesano", "cheddar", "ricotta"}}),
    ("dairy", {"en": {"yogurt", "milk"}, "es": {"yogur", "yogurt", "leche"}}),
    ("beans", {"en": {"beans", "bean", "lentils", "lentil", "chickpeas", "chickpea", "hummus", "edamame", "tofu"},
               "es": {"habichuela", "habichuelas", "frijol", "frijoles", "lenteja", "lentejas", "garbanzo",
                      "garbanzos", "hummus", "gandules", "guandules", "tofu"}}),
    ("nuts", {"en": {"almond", "walnut", "peanut", "cashew", "pecan", "pistachio", "nuts", "nut"},
              "es": {"almendra", "nuez", "nueces", "mani", "cacahuate", "maranon", "pistacho"}}),
    ("rice", {"en": {"rice", "risotto"}, "es": {"arroz", "moro", "locrio", "risotto"}}),
    ("pasta", {"en": {"pasta", "spaghetti", "noodles", "macaroni", "penne", "lasagna"},
               "es": {"pasta", "espagueti", "espaguetis", "spaghetti", "fideos", "macarrones", "lasana"}}),
    ("grain", {"en": {"quinoa", "bulgur", "wheat", "pilaf", "couscous", "oats", "oatmeal", "granola"},
               "es": {"quinoa", "bulgur", "trigo", "cuscus", "avena", "granola"}}),
    ("potato", {"en": {"potato", "potatoes", "fries", "yam", "hash"}, "es": {"papa", "patata", "batata", "name", "yuca"}}),
    ("bread", {"en": {"bread", "toast", "bagel", "bun", "roll", "pita", "croissant", "tortilla"},
               "es": {"pan", "tostada", "tostadas", "bagel", "arepa", "casabe", "croissant", "pita"}}),
    ("leafy", {"en": {"lettuce", "greens", "spinach", "arugula", "kale", "cabbage", "chard", "salad", "brussels",
                      "sprouts", "bok", "choy"},
               "es": {"lechuga", "espinaca", "espinacas", "rucula", "repollo", "col", "kale", "acelga", "berro",
                      "hojas", "ensalada", "bruselas"}}),
    ("vegetable", {"en": {"broccoli", "carrot", "carrots", "cauliflower", "zucchini", "squash", "pepper", "peppers",
                          "tomato", "tomatoes", "cucumber", "cucumbers", "corn", "asparagus", "beets", "celery",
                          "mushroom", "mushrooms", "eggplant", "peas", "onion", "onions"},
                   "es": {"brocoli", "zanahoria", "coliflor", "calabacin", "auyama", "pimiento", "aji", "morron",
                          "tomate", "pepino", "maiz", "esparrago", "remolacha", "apio", "champinon", "hongo",
                          "berenjena", "guisante", "cebolla", "cebollin", "puerro", "vainita", "ajo", "vegetal",
                          "verdura"}}),
    ("avocado", {"en": {"avocado", "guacamole"}, "es": {"aguacate", "guacamole"}}),
    ("fruit", {"en": {"apple", "apples", "banana", "bananas", "berries", "strawberries", "strawberry", "blueberries",
                      "raspberries", "grapes", "melon", "cantaloupe", "honeydew", "pineapple", "orange", "oranges",
                      "watermelon", "fruit", "mango", "pear", "kiwi", "peach", "cherry", "cherries", "papaya"},
               "es": {"manzana", "guineo", "banana", "fresa", "uva", "melon", "pina", "naranja", "sandia",
                      "arandano", "mango", "lechosa", "pera", "kiwi", "fruta", "mora", "frambuesa", "cereza",
                      "durazno", "melocoton", "papaya", "chinola"}}),
)


def _palabras(texto: str) -> set[str]:
    """Las palabras del nombre, sin acentos, y también en singular («pepinos» → pepino, «tomatoes» → tomato): el
    vocabulario va en singular y el analizador dice plurales (revisión final: el recall medía el diccionario)."""
    t = unicodedata.normalize("NFKD", str(texto or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    palabras = set(re.findall(r"[a-z]+", t))
    for w in list(palabras):
        if len(w) > 4 and w.endswith("es"):
            palabras.add(w[:-2])
        if len(w) > 3 and w.endswith("s"):
            palabras.add(w[:-1])
    return palabras


def clase_de(nombre: str, idioma: str) -> str:
    palabras = _palabras(nombre)
    for clase, vocab in _CLASES:
        if palabras & vocab.get(idioma, set()):
            return clase
    return "other"


def error_relativo(estimado: float, verdad: float, piso: float) -> float:
    return round(abs(float(estimado) - float(verdad)) / max(float(verdad), piso), 4)


def parse_fila_n5k(campos: list[str]) -> Optional[dict]:
    """Una fila de `dish_metadata_cafe{1,2}.csv`: 6 columnas del plato y bloques de 7 por ingrediente
    (`ingr_id, ingr_name, grams, calories, fat, carb, protein`)."""
    if len(campos) < 6 or not str(campos[0]).strip():
        return None
    try:
        plato = {"dish_id": str(campos[0]).strip(), "kcal": float(campos[1]), "masa_g": float(campos[2]),
                 "grasa_g": float(campos[3]), "carbs_g": float(campos[4]), "proteina_g": float(campos[5]),
                 "ingredientes": []}
        resto = campos[6:]
        for i in range(0, len(resto) - 6, 7):
            nombre, gramos = str(resto[i + 1]).strip(), float(resto[i + 2])
            if nombre and gramos > 0:
                plato["ingredientes"].append({"nombre": nombre, "gramos": gramos})
    except (TypeError, ValueError):
        return None
    return plato


def plato_valido(p: dict) -> bool:
    """Fuera los platos diminutos y las etiquetas incoherentes (kcal a más de un 50 % de 4P+4C+9G)."""
    if p["masa_g"] < 100 or p["kcal"] < 100:
        return False
    atwater = 4 * p["proteina_g"] + 4 * p["carbs_g"] + 9 * p["grasa_g"]
    return abs(atwater - p["kcal"]) <= 0.5 * p["kcal"]


def muestrear(platos: Iterable[dict], con_foto: set[str], n: int = 150, semilla: int = SEMILLA) -> list[dict]:
    """Muestra ESTRATIFICADA por tramo de calorías y determinista: mismo resultado sea cual sea el orden de entrada."""
    candidatos = sorted((p for p in platos if p["dish_id"] in con_foto and plato_valido(p)),
                        key=lambda p: p["dish_id"])
    rng = random.Random(semilla)
    cuota = n // len(TRAMOS_KCAL)
    elegidos, sobrantes = [], []
    for lo, hi in TRAMOS_KCAL:
        grupo = [p for p in candidatos if lo <= p["kcal"] < hi]
        rng.shuffle(grupo)
        elegidos.extend(grupo[:cuota])
        sobrantes.extend(grupo[cuota:])
    rng.shuffle(sobrantes)
    elegidos.extend(sobrantes[:max(0, n - len(elegidos))])
    return sorted(elegidos, key=lambda p: p["dish_id"])


def _motivo_fallo(estimado) -> str:
    if not isinstance(estimado, dict) or estimado.get("analysis_failed"):
        return "error"
    if estimado.get("is_food") is False:
        return "no_comida"
    return "sin_totales"


def _totales(estimado) -> Optional[dict]:
    if not isinstance(estimado, dict) or estimado.get("analysis_failed") or estimado.get("is_food") is False:
        return None
    try:
        tot = {k: float(estimado.get(k_est) or 0) for k, k_est, _ in MACROS}
    except (TypeError, ValueError):
        return None
    return tot if tot["kcal"] > 0 else None


def _componentes(verdad: dict, estimado: dict) -> dict:
    masa = sum(i["gramos"] for i in verdad["ingredientes"]) or verdad["masa_g"] or 1.0
    esperadas, masa_principal, masa_otros = set(), 0.0, 0.0
    for ing in verdad["ingredientes"]:
        if ing["gramos"] / masa < UMBRAL_COMPONENTE:
            continue
        masa_principal += ing["gramos"]
        c = clase_de(ing["nombre"], "en")
        if c == "other":
            masa_otros += ing["gramos"]
        else:
            esperadas.add(c)
    clases_dichas = [clase_de(it.get("name", ""), "es") for it in (estimado.get("items") or []) if isinstance(it, dict)]
    vistas = set(clases_dichas)
    return {"esperadas": sorted(esperadas), "acertadas": sorted(esperadas & vistas),
            "otros_pct": round(masa_otros / masa_principal, 3) if masa_principal else 0.0,
            "otros_estimado_pct": round(clases_dichas.count("other") / len(clases_dichas), 3) if clases_dichas else 0.0}


def evaluar_plato(verdad: dict, estimado: Optional[dict], latencia_s: Optional[float] = None) -> dict:
    fila = {"dish_id": verdad["dish_id"], "latencia_s": latencia_s}
    tot = _totales(estimado)
    if tot is None:
        fila["fallo"] = _motivo_fallo(estimado)
        return fila
    fila["fallo"] = None
    fila["estimado"] = tot
    fila["errores"] = {k: error_relativo(tot[k], verdad[k], piso) for k, _, piso in MACROS}
    fila["componentes"] = _componentes(verdad, estimado)
    return fila


def percentil(valores: list[float], q: float) -> Optional[float]:
    """Rango más cercano: el valor real que deja por debajo al q de la muestra (sin interpolar)."""
    v = sorted(valores)
    if not v:
        return None
    return v[max(1, math.ceil(q * len(v))) - 1]


def agregar(filas: list[dict]) -> dict:
    ok = [f for f in filas if not f.get("fallo")]
    fallos: dict = {}
    for f in filas:
        if f.get("fallo"):
            fallos[f["fallo"]] = fallos.get(f["fallo"], 0) + 1
    n = len(filas)
    r = {"n": n, "ok": len(ok), "fallos": fallos, "tasa_fallos": round((n - len(ok)) / n, 4) if n else 0.0}
    for k, _, _ in MACROS:
        errs = [f["errores"][k] for f in ok]
        r[k] = {"mediana": round(statistics.median(errs), 4) if errs else None, "p90": percentil(errs, 0.9)}
    r["kcal_dentro_20pct"] = round(sum(1 for f in ok if f["errores"]["kcal"] <= 0.20) / len(ok), 4) if ok else None
    esperadas = sum(len(f["componentes"]["esperadas"]) for f in ok)
    acertadas = sum(len(f["componentes"]["acertadas"]) for f in ok)
    r["recall_componentes"] = round(acertadas / esperadas, 4) if esperadas else None
    r["otros_pct_medio"] = round(statistics.mean([f["componentes"]["otros_pct"] for f in ok]), 4) if ok else None
    r["otros_estimado_pct_medio"] = (round(statistics.mean([f["componentes"].get("otros_estimado_pct", 0.0) for f in ok]), 4)
                                     if ok else None)
    lat = [f["latencia_s"] for f in filas if f.get("latencia_s") is not None]
    r["latencia_s"] = {"p50": percentil(lat, 0.5), "p90": percentil(lat, 0.9)}
    r["valida"] = bool(n) and r["tasa_fallos"] <= UMBRAL_FALLOS_VALIDA
    return r


# [P1-PLAN-LOTE-601 · 2026-09-27] Regla de aceptación PAREADA. Medido: la MISMA base corrida dos veces dio 26,2 % y
# 28,8 % de mediana en calorías (grasa 31,7 % y 36,2 %) y una lectura cambia una mediana de 6,8 % entre corridas, así
# que comparar las medianas de UNA corrida contra otra con 2 puntos de tolerancia rechazaba la base contra sí misma.
MIN_BASES_PAREADO = 2
TOPE_ERROR_PAREADO = 1.0      # un plato disparatado (2.500 % de error) no decide la media
UMBRAL_PAREADO = 0.02         # 2 puntos
REMUESTRAS_PAREADO = 2000
LATENCIA_MAX_PAREADO = 1.25


def comparar_pareado(bases: list[list[dict]], candidata: list[dict], *, remuestras: int = REMUESTRAS_PAREADO,
                     semilla: int = SEMILLA) -> dict:
    """Candidata contra ≥ 2 corridas de la base, plato a plato (solo los que salen bien en TODAS las corridas).

    Por plato: error de la candidata − media del error en las bases, cada error topado a TOPE_ERROR_PAREADO. Por
    macro: la media de esas diferencias y su intervalo bootstrap al 90 % (remuestreando platos). Entra si calorías o
    proteína mejoran ≥ 2 puntos con el intervalo entero por debajo de 0, ninguna macro empeora > 2 puntos con el
    intervalo entero por encima de 0, no hay más fallos que en la peor base y la latencia p50 no sube más de un 25 %."""
    if len(bases) < MIN_BASES_PAREADO:
        return {"n": 0, "macros": {}, "acepta": False,
                "motivo": f"hacen falta al menos {MIN_BASES_PAREADO} corridas de la base (una sola varía ±2-4 puntos)"}

    def _ok(filas):
        return {f["dish_id"]: f for f in filas if not f.get("fallo") and f.get("errores")}

    idx_b = [_ok(b) for b in bases]
    idx_c = _ok(candidata)
    platos = sorted(set(idx_c).intersection(*idx_b))
    if not platos:
        raise ValueError("sin platos comunes entre la candidata y las bases")
    rng = random.Random(semilla)
    muestras = [[rng.randrange(len(platos)) for _ in platos] for _ in range(remuestras)]
    macros: dict = {}
    for k, _, _ in MACROS:
        difs = [min(idx_c[d]["errores"][k], TOPE_ERROR_PAREADO)
                - statistics.fmean(min(b[d]["errores"][k], TOPE_ERROR_PAREADO) for b in idx_b) for d in platos]
        medias = sorted(sum(difs[i] for i in m) / len(m) for m in muestras)
        macros[k] = {"media": round(statistics.fmean(difs), 4),
                     "ic90": [round(medias[int(0.05 * remuestras)], 4), round(medias[int(0.95 * remuestras) - 1], 4)],
                     "mejor": sum(1 for x in difs if x < -UMBRAL_PAREADO),
                     "peor": sum(1 for x in difs if x > UMBRAL_PAREADO)}

    def _tasa(filas):
        return sum(1 for f in filas if f.get("fallo")) / len(filas) if filas else 0.0

    def _p50(filas):
        return percentil([f["latencia_s"] for f in filas if f.get("latencia_s") is not None], 0.5)

    mejora = [k for k in ("kcal", "proteina_g")
              if macros[k]["media"] <= -UMBRAL_PAREADO and macros[k]["ic90"][1] < 0]
    peores = [k for k, m in macros.items() if m["media"] > UMBRAL_PAREADO and m["ic90"][0] > 0]
    motivos = []
    if not mejora:
        motivos.append("no mejora calorías ni proteína (≥ 2 puntos, con el intervalo entero por debajo de 0)")
    if peores:
        motivos.append("empeora " + ", ".join(peores))
    if _tasa(candidata) > max(_tasa(b) for b in bases):
        motivos.append("más fallos que la base")
    lat_b = [x for x in (_p50(b) for b in bases) if x is not None]
    lat_c = _p50(candidata)
    if lat_b and lat_c is not None and lat_c > LATENCIA_MAX_PAREADO * statistics.fmean(lat_b):
        motivos.append("latencia p50 sube más de un 25 %")
    return {"n": len(platos), "macros": macros, "acepta": not motivos,
            "motivo": "; ".join(motivos) if motivos else "mejora " + ", ".join(mejora)}


def sha256_de(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def manifiesto(platos: list[dict], hashes: dict, semilla: int, fuente: str) -> dict:
    return {"version": 1, "fuente": fuente, "semilla": semilla, "n": len(platos),
            "platos": [{**p, "sha256_png": hashes[p["dish_id"]]} for p in platos]}


def verificar_cache(man: dict, leer_bytes: Callable[[str], Optional[bytes]]) -> list[str]:
    """Los `dish_id` cuya foto en caché falta o no es la congelada. Vacío = se puede correr."""
    malos = []
    for p in man["platos"]:
        datos = leer_bytes(p["dish_id"])
        if datos is None or sha256_de(datos) != p["sha256_png"]:
            malos.append(p["dish_id"])
    return malos
