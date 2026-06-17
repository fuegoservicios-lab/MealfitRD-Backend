"""[P2-PRICES-ENGINE-1 · 2026-06-16] Scraper autónomo de precios de Supermercados
Nacional (supermercadosnacional.com) → precios BASE en master_ingredients.

DESCUBRIMIENTO (prueba en vivo 2026-06-16): el sitio es Magento server-rendered —
los precios vienen en el HTML crudo (`data-price-amount="84"` + nombre en
`product-item-name`), SIN JS. Por eso `requests` puro basta y el motor es AUTÓNOMO
(no depende de un navegador headless ni de herramientas externas).

Diseño:
  - `CATEGORIES`: URLs de categoría reales (verificadas). Se fetchea cada una UNA vez.
  - `MATCH`: slug de master_ingredient → categorías + keywords. Un producto matchea
    por keyword (accent-insensitive). El precio representativo = MEDIANA de los matches
    (robusto a outliers premium tipo "Free Farm").
  - Columna destino por `master.default_unit`: 'lb' → price_per_lb_base; resto →
    price_per_unit_base (mismo contrato que lee shopping_calculator).
  - Confianza por categoría: Despensa=high (staple estable), Lácteos=medium,
    perecederos (Proteínas frescas/Frutas/Vegetales/Víveres)=low.

POLÍTICA: correr como SCRIPT supervisado (no cron prod) — un cambio de HTML no debe
corromper precios en silencio. El cron de inflación (price_engine) mantiene los vivos
frescos entre scrapes; re-scrapear re-ancla la base.

Uso:
    PYTHONPATH=backend python backend/scripts/fetch_nacional_prices.py --out precios.csv
    PYTHONPATH=backend python backend/scripts/fetch_nacional_prices.py --apply   # importa + reescala
"""
import argparse
import csv
import html as ihtml
import os
import re
import sys
import time
import unicodedata
from datetime import date, datetime, timezone
from statistics import median

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")  # consola Windows cp1252 → utf-8
except Exception:
    pass

# backend/ al path para importar price_engine/db_core sin depender de PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

import psycopg

BASE = "https://supermercadosnacional.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MealfitRD-price-bot/1.0)"}

# Categorías reales verificadas (2026-06-16). Clave lógica → ruta. Se paginan con ?p=N.
CATEGORIES = {
    # Carnes / pescados
    "res":        "/carnes-pescados-y-mariscos/carnes/res",
    "pollo":      "/carnes-pescados-y-mariscos/carnes/pollo",
    "cerdo":      "/carnes-pescados-y-mariscos/carnes/cerdo",
    "pavo":       "/carnes-pescados-y-mariscos/carnes/pavo",
    "sustitutos": "/carnes-pescados-y-mariscos/carnes/sustitutos-carnicos",
    "pescados":   "/carnes-pescados-y-mariscos/pescados-y-mariscos/pescados",
    "camarones":  "/carnes-pescados-y-mariscos/pescados-y-mariscos/camarones",
    # Quesos y embutidos
    "charcuteria":"/quesos-y-embutidos/charcuteria-y-embutidos",
    "quesos":     "/quesos-y-embutidos/quesos",
    # Lácteos / huevos
    "leches":     "/lacteos-y-huevos/leches",
    "yogures":    "/lacteos-y-huevos/yogures",
    "mantequilla":"/lacteos-y-huevos/mantequilla-y-margarina",
    "huevos":     "/lacteos-y-huevos/huevos",
    # Frutas / vegetales / víveres
    "frutas":     "/frutas-y-vegetales/frutas",
    "vegetales":  "/frutas-y-vegetales/vegetales-y-hortalizas",
    "hierbas":    "/frutas-y-vegetales/vegetales-y-hortalizas/hierbas-aromaticas",
    "viveres":    "/frutas-y-vegetales/viveres",
    # Despensa
    "arroz":      "/despensa/arroz-cereales-y-legumbres/arroz",
    "habichuelas":"/despensa/arroz-cereales-y-legumbres/habichuelas",
    "granos":     "/despensa/arroz-cereales-y-legumbres",
    "aceites":    "/despensa/aceites",
    "pastas":     "/despensa/pastas-y-salsas",
    "conservas":  "/despensa/conservas-y-encurtidos",
    "condimentos":"/despensa/condimentos-y-especias",
    "especias":   "/despensa/condimentos-y-especias/especias",
    "miel":       "/despensa/desayuno/miel-y-melaza",
    "aderezos":   "/despensa/aderezos-y-salsas",
    "untables":   "/despensa/mermeladas-y-untables",
    "desayuno":   "/despensa/desayuno",
    "galletas":   "/despensa/galletas",
    "reposteria": "/despensa/reposteria",
    "azucar":     "/despensa/azucar-y-edulcorantes",
    "picaderas":  "/despensa/picaderas",
}

# slug de master_ingredient → (categorías, keywords-include [, keywords-exclude]).
# Keywords accent-insensitive. Un producto matchea si CONTIENE algún include y NINGÚN exclude.
MATCH = {
    # ── Proteínas frescas ──
    "pechuga-de-pollo": (["pollo"], ["pechuga"]),
    "carne-de-res":     (["res"], ["res", "bistec", "lomo"], ["molida", "hamburguesa"]),
    "carne-molida":     (["res"], ["molida"]),
    "cerdo":            (["cerdo"], ["cerdo", "chuleta", "lomo", "costilla"]),
    "bacalao":          (["pescados"], ["bacalao"]),
    "filete-pescado":   (["pescados"], ["filete", "tilapia", "merluza", "pescado", "salmon", "mero", "dorado"]),
    "camarones":        (["camarones"], ["camaron"]),
    "tofu":             (["sustitutos"], ["tofu"]),
    "huevo":            (["huevos"], ["huevo"], ["codorniz"]),
    "clara-de-huevo":   (["huevos"], ["clara"]),
    "yema-de-huevo":    (["huevos"], ["yema"]),
    # ── Charcutería / embutidos (proteínas procesadas) ──
    "salami":           (["charcuteria"], ["salami"]),
    "longaniza":        (["charcuteria"], ["longaniza"]),
    "jamon-de-pavo":    (["charcuteria"], ["pavo"]),  # jamón/pechuga de pavo deli
    # ── Enlatados (conservas) ──
    "atun":             (["conservas"], ["atun"]),
    "maiz-dulce":       (["conservas"], ["maiz"]),
    "gandules":         (["granos", "conservas"], ["guandul", "gandul"]),
    "aceitunas":        (["conservas"], ["aceituna"]),
    "salsa-de-tomate":  (["conservas", "pastas"], ["salsa de tomate", "pasta de tomate"]),
    # ── Despensa: granos ──
    "arroz-blanco":     (["arroz"], ["arroz"], ["integral"]),
    "arroz-integral":   (["arroz"], ["integral"]),
    "habichuelas-rojas":(["habichuelas"], ["roja"]),
    "habichuelas-negras":(["habichuelas"], ["negra"]),
    "habichuelas-blancas":(["habichuelas"], ["blanca"]),
    "lentejas":         (["habichuelas", "granos"], ["lenteja"]),
    "garbanzos":        (["habichuelas", "granos"], ["garbanzo"]),
    "quinoa":           (["granos", "desayuno"], ["quinoa"]),
    # ── Despensa: aceites ──
    "aceite-vegetal":   (["aceites"], ["vegetal", "soya", "girasol", "maiz"]),
    "aceite-de-oliva":  (["aceites"], ["oliva"]),
    "aceite-de-coco":   (["aceites"], ["coco"]),
    "aceite-de-sesamo": (["aceites"], ["sesamo", "ajonjoli"]),
    # ── Despensa: pastas/harinas/repostería ──
    "pasta-integral":   (["pastas"], ["integral"]),
    "harina-de-trigo":  (["reposteria"], ["harina de trigo", "harina todo uso", "harina leudante"]),
    "harina-maiz":      (["reposteria", "conservas"], ["harina de maiz", "maiz precocida", "harina precocida"]),
    "extracto-vainilla":(["reposteria"], ["vainilla"]),
    # ── Despensa: condimentos/especias (leaf 'especias' = especias puras) ──
    "ajo-en-polvo":     (["especias", "condimentos"], ["ajo en polvo", "ajo molido", "ajo granulado"]),
    "oregano":          (["especias", "condimentos"], ["oregano"]),
    "pimenton":         (["especias", "condimentos"], ["pimenton", "paprika"]),
    "pimienta-negra":   (["especias", "condimentos"], ["pimienta negra", "pimienta"]),
    "canela":           (["especias", "condimentos", "reposteria"], ["canela"]),
    "albahaca-seca":    (["especias", "condimentos"], ["albahaca"]),
    "sal":              (["condimentos"], ["sal "], ["salsa", "salami", "balsamico", "ensalada", "trufa"]),
    # ── Despensa: aderezos ──
    "mostaza":          (["aderezos"], ["mostaza"]),
    "salsa-soya":       (["aderezos"], ["soya", "soja"]),
    "vinagre-blanco":   (["aderezos", "condimentos"], ["vinagre blanco"]),
    "vinagre-de-manzana":(["aderezos", "condimentos"], ["vinagre de manzana", "vinagre organico de manzana"]),
    # ── Despensa: untables / miel ──
    "mantequilla-de-mani":(["untables"], ["mani", "cacahuate", "peanut"]),
    "miel":             (["miel", "untables"], ["miel"], ["mielda"]),
    # ── Despensa: desayuno ──
    "avena":            (["desayuno"], ["avena"]),
    "granola":          (["desayuno"], ["granola"]),
    "semillas-chia":    (["desayuno", "reposteria"], ["chia"]),
    # ── Despensa: galletas ──
    "galletas-de-soda": (["galletas"], ["soda"]),
    "galletas-arroz":   (["galletas"], ["galleta de arroz", "arroz"]),
    # ── Despensa: edulcorantes / frutos secos ──
    "estevia":          (["azucar"], ["estevia", "stevia"]),
    "almendras":        (["picaderas", "reposteria"], ["almendra"]),
    "mani":             (["picaderas"], ["mani"], ["mantequilla"]),
    # ── Lácteos ──
    "leche":            (["leches"], ["leche"], ["descremada", "deslactosada", "almendra", "coco", "evaporada", "condensada"]),
    "leche-descremada": (["leches"], ["descremada"]),
    "leche-evaporada":  (["leches"], ["evaporada"]),
    "mantequilla":      (["mantequilla"], ["mantequilla"], ["mani", "cacahuate"]),
    "yogurt-griego-entero": (["yogures"], ["griego"]),
    "yogurt-griego":    (["yogures"], ["griego"]),
    # ── Quesos ──
    "queso-blanco":     (["quesos"], ["queso blanco", "queso fresco"]),
    "queso-de-hoja":    (["quesos"], ["hoja"]),
    "queso-mozzarella": (["quesos"], ["mozzarella"]),
    "queso-crema":      (["quesos"], ["crema"]),
    "queso-cottage":    (["quesos"], ["cottage"]),
    "queso-parmesano":  (["quesos"], ["parmesano", "parmigiano"]),
    "queso-ricotta":    (["quesos"], ["ricotta"]),
    # ── Frutas (perecederos) ──
    "guineo":           (["frutas"], ["guineo"], ["verde"]),
    "manzana":          (["frutas"], ["manzana"]),
    "aguacate":         (["frutas"], ["aguacate"]),
    "limon":            (["frutas"], ["limon"]),
    "naranja":          (["frutas"], ["naranja"]),
    "pina":             (["frutas"], ["pina"]),
    "lechosa":          (["frutas"], ["lechosa", "papaya"]),
    "fresas":           (["frutas"], ["fresa"]),
    "mango":            (["frutas"], ["mango"]),
    "chinola":          (["frutas"], ["chinola", "maracuya"]),
    "melon":            (["frutas"], ["melon"]),
    "sandia":           (["frutas"], ["sandia", "patilla"]),
    # ── Vegetales ──
    "tomate":           (["vegetales"], ["tomate"]),
    "cebolla":          (["vegetales"], ["cebolla"]),
    "ajo":              (["vegetales"], ["ajo"], ["en polvo"]),
    "zanahoria":        (["vegetales"], ["zanahoria"]),
    "brocoli":          (["vegetales"], ["brocoli"]),
    "lechuga":          (["vegetales"], ["lechuga"]),
    "pepino":           (["vegetales"], ["pepino"]),
    "pimiento-morron":  (["vegetales"], ["pimiento", "morron"]),
    "espinacas":        (["vegetales"], ["espinaca"]),
    "repollo":          (["vegetales"], ["repollo"]),
    "berenjena":        (["vegetales"], ["berenjena"]),
    "tayota":           (["vegetales"], ["tayota"]),
    "vainitas":         (["vegetales"], ["vainita"]),
    "aji-cubanela":     (["vegetales"], ["cubanela", "aji"]),
    "cilantro":         (["hierbas", "vegetales"], ["cilantro", "cilantrico"]),
    "coliflor":         (["vegetales"], ["coliflor"]),
    "jengibre":         (["vegetales", "viveres"], ["jengibre"]),
    "molondrones":      (["vegetales"], ["molondron"]),
    # ── Víveres ──
    "platano-verde":    (["viveres"], ["platano verde"]),
    "platano-maduro":   (["viveres"], ["maduro"]),
    "guineo-verde":     (["viveres"], ["guineo verde"]),
    "yuca":             (["viveres"], ["yuca"]),
    "batata":           (["viveres"], ["batata"]),
    "papa":             (["viveres"], ["papa"]),
    "auyama":           (["vegetales", "viveres"], ["auyama"]),
    "name":             (["viveres"], ["ñame", "name"]),
    "yautia":           (["viveres"], ["yautia"]),
}

# Tipo de unidad del catálogo → cómo interpretar el precio scrapeado.
_WEIGHT_UNITS = {"lb", "g", "kg", "oz"}
_PIECE_UNITS = {"unidad", "cabeza", "diente", "mazo", "hoja", "rebanada", "atado", "manojo"}
# El resto (botella, pote, lata, paquete, sobre, frasco, caja, cartón, funda, tetra,
# galón, jarra, bolsa) son ENVASES: el precio listado del envase ES el precio por-unidad.

CONFIDENCE_BY_CATEGORY = {
    "despensa": "high", "lácteos": "medium", "lacteos": "medium",
    "proteínas": "low", "proteinas": "low", "frutas": "low",
    "vegetales": "low", "víveres": "low", "viveres": "low",
}

_PRICE_RE = re.compile(r'data-price-amount="([\d.]+)"\s+data-price-type="finalPrice"')
_NAME_RE = re.compile(r'class="product\s+name\s+product-item-name"[^>]*>\s*<a[^>]*>([^<]+)</a>')


def _norm(s: str) -> str:
    """lower + strip accents para matching robusto."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


def _parse_page(html_text: str) -> list[tuple[str, float]]:
    prices = _PRICE_RE.findall(html_text)
    names = [ihtml.unescape(" ".join(n.split())) for n in _NAME_RE.findall(html_text)]
    pairs = []
    for n, p in zip(names, prices):
        try:
            pairs.append((n, float(p)))
        except ValueError:
            pass
    return pairs


def fetch_category(path: str, max_pages: int = 5) -> list[tuple[str, float]]:
    """Devuelve [(name, price), ...] de una categoría Magento, PAGINANDO con ?p=N.

    `product_list_limit` lo ignora el sitio (siempre ~16/página), así que recorremos
    páginas hasta que una no traiga productos NUEVOS (Magento clampa a la última página
    repitiéndola) o se acabe. [] si 404/error.
    """
    all_pairs: list[tuple[str, float]] = []
    seen: set[str] = set()
    for p in range(1, max_pages + 1):
        url = f"{BASE}{path}?p={p}"
        # Retry: el sitio ocasionalmente devuelve non-200 o una página vacía bajo carga.
        page = None
        for attempt in range(3):
            try:
                r = requests.get(url, headers=HEADERS, timeout=30)
            except Exception as e:
                if attempt == 2:
                    print(f"  ERR {path} p{p}: {type(e).__name__}: {str(e)[:60]}")
                time.sleep(1.5)
                continue
            if r.status_code != 200:
                if attempt == 2 and p == 1:
                    print(f"  WARN {path}: HTTP {r.status_code}")
                time.sleep(1.5)
                continue
            parsed = _parse_page(r.text)
            if parsed or p > 1:  # página 1 vacía → reintenta (probable hiccup)
                page = parsed
                break
            time.sleep(1.5)
        if page is None:
            break
        new = [(n, pr) for (n, pr) in page if n not in seen]
        if not new:
            break  # página repetida (clamp) o vacía → fin
        for n, _ in new:
            seen.add(n)
        all_pairs.extend(new)
        if len(page) < 16:
            break  # última página parcial
        time.sleep(0.4)
    return all_pairs


def _matches(name: str, include: list[str], exclude: list[str]) -> bool:
    n = _norm(name)
    if exclude and any(_norm(x) in n for x in exclude):
        return False
    return any(_norm(x) in n for x in include)


_RANGE_LB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:a|-)\s*(\d+(?:\.\d+)?)\s*libra")
_SINGLE_LB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:lb|libra)")
# Nacional escribe el peso como "800 Gr", "8 Onz", "1 Kg" — no solo "g"/"oz".
_GRAMS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:gramos|grs|gr|g)\b")
_KG_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kilogramos?|kilos?|kg)\b")
_OZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:onzas?|onz|oz)\b")
_CONTAINER_WORDS = ("malla", "funda", "paquete", "saco", "caja", "paq")


def to_per_lb(name: str, price: float) -> float:
    """Normaliza el precio de un producto a RD$/libra.

    Distingue dos modos por el nombre (verificado en vivo 2026-06-16):
      - PER-LB:   "Yuca…, Lb (Aprox…)", "…/lb", "…Por Libra" → el precio YA es por libra.
      - PAQUETE:  "Arroz La Garza 10 Lb", "Papas Malla (Aprox 3 a 5 Libras Por Paquete)"
                  → el precio es del paquete → dividir por su peso.
    Marca PER-LB tiene prioridad sobre un peso aprox descriptivo ("Peso Aprox. 2 Libra").
    """
    n = _norm(name)
    # Peso explícito (rango → promedio; si no, primer "N lb/libra").
    weight = None
    rm = _RANGE_LB_RE.search(n)
    if rm:
        weight = (float(rm.group(1)) + float(rm.group(2))) / 2.0
    else:
        sm = _SINGLE_LB_RE.search(n)
        if sm:
            weight = float(sm.group(1))
        else:
            gm = _GRAMS_RE.search(n) or _KG_RE.search(n)
            if _KG_RE.search(n):
                weight = float(_KG_RE.search(n).group(1)) * 2.20462
            elif _GRAMS_RE.search(n):
                weight = float(_GRAMS_RE.search(n).group(1)) / 453.592
            elif _OZ_RE.search(n):
                weight = float(_OZ_RE.search(n).group(1)) / 16.0

    # Marca de venta-por-libra (", Lb" / "/lb" / "Por Libra") GANA: cualquier peso en
    # el nombre es entonces el peso aprox. de la pieza, no del paquete. Si no hay marca y
    # sí hay un peso explícito (lb O gramos/kg/oz), es un paquete → dividir.
    per_lb_marker = (", lb" in n) or ("/lb" in n) or ("por libra" in n)
    if per_lb_marker:
        return price
    if weight and weight > 0:
        return price / weight
    return price               # sin señal de peso → best-effort


_UNITS_PER_LB_RE = re.compile(r"(\d+)\s*(?:-|a)?\s*(\d+)?\s*unidad(?:es)?\s+por\s+libra")
_UNITS_PER_PACK_RE = re.compile(r"(\d+)\s*(?:-|a)?\s*(\d+)?\s*unidad(?:es)?(?:\s+por\s+paquete)?")
_UND_PER_PAQ_RE = re.compile(r"(\d+)\s*und\s*/?\s*paq")


def _avg2(g1, g2):
    return (float(g1) + float(g2)) / 2.0 if g2 else float(g1)


def _units_per_lb(n: str):
    m = _UNITS_PER_LB_RE.search(n)
    return _avg2(m.group(1), m.group(2)) if m else None


def _units_per_pack(n: str):
    m = _UND_PER_PAQ_RE.search(n)
    if m:
        return float(m.group(1))
    # "Aprox. 4 Unidades" / "6-8 Unidades Por Paquete" dentro de un paquete
    m = _UNITS_PER_PACK_RE.search(n)
    return _avg2(m.group(1), m.group(2)) if m else None


def estimate_prices(products: list[tuple[str, float]]):
    """Estima (precio_por_libra, precio_por_pieza) a partir de los productos matcheados.

    Pobla AMBOS cuando es posible (el calculador usa la columna que aplique según cómo
    la receta exprese la cantidad). Maximiza cobertura sin inflar:
      - per-lb: SÓLO de productos con señal de peso (marca ", Lb"/"por libra" o "N lb"),
        normalizando paquetes-por-peso vía to_per_lb. Un "…, Und" NO cuenta como per-lb.
      - per-pieza: de "…, Und", de paquetes con conteo ("4 Und/Paq", "Aprox 4 Unidades"),
        o derivado de per-lb ÷ (unidades por libra). Bultos por peso sin conteo se ignoran.
    """
    per_lb_vals, per_unit_vals = [], []
    for (name, price) in products:
        n = _norm(name)
        is_pkg = any(w in n for w in _CONTAINER_WORDS) or "/paq" in n
        upp = _units_per_pack(n)
        upl = _units_per_lb(n)
        has_weight = ((", lb" in n) or ("lb (" in n) or ("/lb" in n) or ("por libra" in n)
                      or bool(_RANGE_LB_RE.search(n)) or bool(_SINGLE_LB_RE.search(n))
                      or bool(_GRAMS_RE.search(n)) or bool(_KG_RE.search(n)) or bool(_OZ_RE.search(n)))
        if has_weight:
            per_lb_vals.append(to_per_lb(name, price))
        if (", und" in n or "/und" in n or n.endswith(" und")) and not is_pkg:
            per_unit_vals.append(price)
        elif is_pkg and upp:
            per_unit_vals.append(price / upp)
        elif upl and (", lb" in n or "por libra" in n):
            per_unit_vals.append(to_per_lb(name, price) / upl)
    per_lb = round(float(median(per_lb_vals)), 2) if per_lb_vals else None
    per_unit = round(float(median(per_unit_vals)), 2) if per_unit_vals else None
    return per_lb, per_unit


def main():
    ap = argparse.ArgumentParser(description="Scrapea precios base de Supermercados Nacional.")
    ap.add_argument("--out", default=None, help="Ruta CSV de salida (compatible con import_prices.py).")
    ap.add_argument("--period", default=None, help="price_base_period YYYY-MM (default: mes actual).")
    ap.add_argument("--apply", action="store_true", help="Importa a DB (price_engine) + reescala. Requiere NEON.")
    ap.add_argument("--max-pages", type=int, default=5, dest="max_pages", help="Páginas (?p=N) a recorrer por categoría.")
    args = ap.parse_args()

    period = args.period or datetime.now(timezone.utc).strftime("%Y-%m")
    today = date.today().isoformat()

    NEON = os.environ.get("NEON_DATABASE_URL") or os.environ.get("NEON_DATABASE_URL_POOLED")
    if not NEON:
        print("FATAL: NEON_DATABASE_URL(_POOLED) no está en .env")
        sys.exit(1)

    # 1. Catálogo real (slug → name, category, default_unit).
    with psycopg.connect(NEON) as conn:
        master = {
            r[0]: {"name": r[1], "category": r[2], "default_unit": r[3]}
            for r in conn.execute(
                "SELECT slug, name, category, default_unit FROM master_ingredients"
            ).fetchall()
        }
    print(f"master_ingredients: {len(master)} | mapping: {len(MATCH)} ingredientes | período {period}")

    # 2. Fetch de categorías (una vez cada una).
    needed = sorted({c for rule in MATCH.values() for c in rule[0]})
    cat_products: dict[str, list[tuple[str, float]]] = {}
    for key in needed:
        path = CATEGORIES.get(key)
        if not path:
            print(f"  (sin URL para categoría {key})")
            continue
        pairs = fetch_category(path, args.max_pages)
        cat_products[key] = pairs
        print(f"  {key}: {len(pairs)} productos")
        time.sleep(0.5)  # cortesía

    # 3. Matching → precio representativo por ingrediente.
    rows = []
    for slug, rule in MATCH.items():
        if slug not in master:
            continue
        cats, include = rule[0], rule[1]
        exclude = rule[2] if len(rule) > 2 else []
        unit = (master[slug]["default_unit"] or "").lower()
        matched = []
        for c in cats:
            for (name, price) in cat_products.get(c, []):
                if _matches(name, include, exclude):
                    matched.append((name, price))
        if not matched:
            continue
        if unit in _WEIGHT_UNITS:
            # Peso: precio por libra (el calculador agrega estos ingredientes en peso).
            per_lb, _ = estimate_prices(matched)
            lb_base, unit_base = per_lb, None
        elif unit in _PIECE_UNITS:
            # Pieza suelta (manzana/ajo/huevo): deriva la pieza + per-lb si está limpio.
            lb_base, unit_base = estimate_prices(matched)
        else:
            # Envase (botella/pote/lata/paquete/…): el precio del envase ES el por-unidad.
            prices = [p for (_, p) in matched]
            unit_base = round(float(median(prices)), 2) if prices else None
            lb_base = None
        if lb_base is None and unit_base is None:
            continue
        cat = _norm(master[slug]["category"])
        confidence = CONFIDENCE_BY_CATEGORY.get(cat, "medium")
        rows.append({
            "slug": slug, "name": master[slug]["name"],
            "price_per_lb_base": lb_base, "price_per_unit_base": unit_base,
            "price_base_period": period, "price_source": "nacional_online",
            "price_confidence": confidence, "price_captured_at": today,
            "_n_matched": len(matched),
        })
        shown = f"{lb_base}/lb" if lb_base is not None else ""
        shown += (" " if shown else "") + (f"{unit_base}/ud" if unit_base is not None else "")
        print(f"  ✓ {slug:<22} RD${shown:<18} ({len(matched)} matches, {confidence})")

    print(f"\nResueltos {len(rows)}/{len(MATCH)} ingredientes con precio real.")

    # 4. Salida.
    if args.out:
        cols = ["slug", "name", "price_per_lb_base", "price_per_unit_base",
                "price_base_period", "price_source", "price_confidence", "price_captured_at"]
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in cols})
        print(f"CSV escrito: {args.out}")

    if args.apply:
        # El connection_pool de db_core no se auto-abre en scripts standalone.
        from db_core import connection_pool as _pool
        if _pool is not None:
            try:
                _pool.open(); _pool.wait(timeout=15)
            except Exception as _e:
                print(f"  (pool open: {_e})")
        import price_engine as pe
        res = pe.import_base_prices(rows)
        print(f"import (BASE): {res['matched']} matched, {res['unmatched']} unmatched")
        # NO force: recompute respeta MEALFIT_PRICES_ENABLED. Así los precios VIVOS
        # (los que muestra shopping_calculator) sólo se publican con el feature ON.
        rc = pe.recompute_adjusted_prices()
        print(f"recompute (vivos): {rc}")
        if rc.get("status") == "disabled":
            print("  ℹ Precios BASE importados, pero los VIVOS no se publicaron porque "
                  "MEALFIT_PRICES_ENABLED=false. Actívalo (+ ingiere índice BCRD) para "
                  "que el cron publique los precios y aparezcan en la lista de compras.")


if __name__ == "__main__":
    main()
