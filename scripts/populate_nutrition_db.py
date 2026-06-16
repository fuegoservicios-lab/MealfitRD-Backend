#!/usr/bin/env python
"""[P1-FOOD-DB-NUTRITION · 2026-06-13] Pobla master_ingredients.{kcal,protein,carbs,
fats,fiber,sodium}_per_100g desde USDA FoodData Central (cimiento del solver
determinista del cerebro dividido MDDA).

Uso (one-shot, NO corre en runtime de generación):
    export USDA_API_KEY=<tu key gratis de fdc.nal.usda.gov>   # o DEMO_KEY (30/hr)
    conda activate mealfit
    python scripts/populate_nutrition_db.py            # pobla todo
    python scripts/populate_nutrition_db.py --dry-run  # solo muestra, no escribe
    python scripts/populate_nutrition_db.py --only "Pechuga de pollo"

Estrategia: USDA cubre el grueso (alimentos comunes, datos Foundation/SR Legacy con
trazabilidad fdc_id). Los procesados/DD que USDA no tiene bien se curan manual
(MANUAL_MACROS, flag is_dominican_cultivar). kcal = Atwater 4/4/9 (consistente con el
solver). Mapeo es-DO → query USDA en USDA_QUERY (la curación es-DO→inglés es el grueso).

[P2-LOGGER-EXEMPT: script CLI one-shot, salida a stdout intencional]
"""
import argparse
import os
import sys
import time

import requests
import dotenv
import psycopg

try:  # consola Windows cp1252 rompe en emoji; forzar UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

USDA_KEY = os.environ.get("USDA_API_KEY", "DEMO_KEY")
USDA_SEARCH = "https://api.nal.usda.gov/fdc/v1/foods/search"
NEON_URL = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")

# --- Mapeo es-DO → query USDA (Foundation/SR Legacy). Forma "raw"/cruda canónica. ---
USDA_QUERY = {
    # Despensa
    "Aceite de coco": "coconut oil", "Aceite de oliva": "olive oil",
    "Aceite de sésamo": "sesame oil", "Aceite vegetal": "soybean oil",
    "Aceitunas": "olives ripe canned", "Ajo en polvo": "garlic powder",
    "Albahaca seca": "basil dried", "Almendras fileteadas": "almonds",
    "Arroz blanco": "rice white long-grain raw", "Arroz integral": "rice brown long-grain raw",
    "Avena": "oats", "Canela en polvo": "cinnamon ground",
    "Extracto de vainilla": "vanilla extract", "Galletas de arroz": "rice cakes",
    "Galletas de soda": "crackers saltines", "Gandules": "pigeon peas mature seeds raw",
    "Garbanzos": "chickpeas garbanzo mature seeds raw", "Habichuelas blancas": "beans white mature seeds raw",
    "Habichuelas negras": "beans black mature seeds raw", "Habichuelas rojas": "beans kidney red mature seeds raw",
    "Harina de maíz precocida": "cornmeal degermed", "Harina de trigo": "wheat flour white all-purpose enriched",
    "Lentejas": "lentils raw", "Mantequilla de maní": "peanut butter smooth",
    "Maíz dulce en granos": "corn sweet yellow raw", "Miel": "honey",
    "Mostaza": "mustard prepared yellow", "Orégano dominicano": "oregano dried",
    "Pan de agua": "bread white commercially prepared", "Pan integral": "bread whole wheat commercially prepared",
    "Pasta integral": "pasta whole wheat dry", "Pimentón": "paprika",
    "Pimienta negra": "spices pepper black", "Quinoa": "quinoa uncooked",
    "Salsa de soya baja en sodio": "soy sauce reduced sodium", "Salsa de tomate": "tomato sauce canned",
    "Semillas de chía": "seeds chia dried", "Tortilla integral": "tortilla whole wheat",
    "Vinagre de manzana": "vinegar cider",
    # [P1-RESOLVER-COVERAGE · 2026-06-16] alimentos atómicos añadidos (gap medido vs planes reales).
    "Granola": "granola homemade", "Maní": "peanuts all types dry-roasted without salt",
    # Frutas
    "Aguacate": "avocado raw", "Chinola": "passion-fruit purple raw", "Fresas": "strawberries raw",
    "Guineo": "banana raw", "Lechosa": "papaya raw", "Limón": "lime raw",
    "Mango": "mango raw", "Melón": "melon cantaloupe raw", "Naranja": "orange raw",
    "Piña": "pineapple raw", "Sandía": "watermelon raw", "Manzana": "apples raw with skin",
    # Lácteos
    "Leche": "milk whole", "Leche evaporada": "milk canned evaporated",
    "Mantequilla": "butter salted", "Queso blanco": "cheese queso fresco",
    "Queso cottage": "cheese cottage lowfat", "Queso crema": "cream cheese",
    "Queso mozzarella": "cheese mozzarella whole milk", "Queso parmesano": "cheese parmesan grated",
    "Queso ricotta": "cheese ricotta whole milk", "Yogurt griego sin azúcar": "yogurt greek plain nonfat",
    # Proteínas
    "Atún en agua": "fish tuna light canned in water", "Bacalao": "fish cod atlantic raw",
    "Camarones": "crustaceans shrimp raw", "Carne de res": "beef round eye of round raw",
    "Carne de res molida": "ground beef 85 lean 15 fat raw", "Cerdo": "pork loin raw",
    "Filete de pescado blanco": "fish tilapia raw", "Huevo": "egg whole raw fresh",
    "Jamón de pavo": "turkey ham cured", "Pechuga de pollo": "chicken breast boneless skinless raw",
    "Tofu": "tofu raw firm",
    # [P1-RESOLVER-COVERAGE · 2026-06-16 follow-up] clara separada del huevo entero (precisión).
    "Clara de huevo": "egg white raw fresh",
    # [P2-1/P2-2/P2-3 · 2026-06-16] filas nuevas separadas del genérico (yema≠huevo entero,
    # leche descremada≠entera, yogur griego entero≠nonfat) — datos USDA reales, no inventados.
    "Yema de huevo": "egg yolk raw fresh",
    "Leche descremada": "milk nonfat fat free skim",
    "Yogurt griego entero": "yogurt greek plain whole milk",
    # Vegetales
    "Ajo": "garlic raw", "Ají cubanela": "peppers sweet green raw", "Berenjena": "eggplant raw",
    "Brócoli": "broccoli raw", "Cebolla": "onions raw", "Cilantro": "coriander cilantro leaves raw",
    "Coliflor": "cauliflower raw", "Espinacas": "spinach raw", "Jengibre": "ginger root raw",
    "Lechuga": "lettuce green leaf raw", "Molondrones": "okra raw", "Pimiento morrón": "peppers sweet red raw",
    "Repollo": "cabbage raw", "Tayota": "chayote raw", "Tomate": "tomatoes red ripe raw",
    "Vainitas": "beans snap green raw", "Zanahoria": "carrots raw", "Pepino": "cucumber with peel raw",
    # Víveres (viandas DD → USDA tiene la especie; flag is_dominican_cultivar)
    "Auyama": "pumpkin raw", "Batata": "sweet potato raw", "Guineo verde": "plantains green raw",
    "Papa": "potatoes raw", "Plátano maduro": "plantains ripe raw", "Plátano verde": "plantains green raw",
    "Yautía": "taro raw", "Yuca": "cassava raw", "Ñame": "yams raw",
}
# [P2-CULTIVAR-PROVENANCE · 2026-06-15] (gap-audit G17) PROVENIENCIA de los cultivares dominicanos: estas 9
# viandas se pueblan con la ESPECIE USDA más cercana (proxy botánico: Yautía→taro, Ñame→yams, Plátano→
# plantains, Auyama→pumpkin, Yuca→cassava, Batata→sweet potato), NO con datos CURADOS es-DO. Son
# aproximaciones plausibles (especie correcta) pero el error vs el cultivar local NO está cuantificado.
# Quedan marcadas `is_dominican_cultivar=TRUE` + `nutrition_source='usda'` (trazable a fdc_id, honesto en el
# footer de proveniencia del PDF). CURACIÓN PENDIENTE (bloqueada por recurso externo): validar contra una
# fuente Caribe (INCAP/LATINFOODS/FAO INFOODS) + revisión de nutricionista → ahí cambiarían a
# `nutrition_source='incap'`/'curado'. Doc: backend/docs/food_db_integration.md. NO asumir que son curadas.
_VIVERES_DD = {"Auyama", "Batata", "Guineo verde", "Papa", "Plátano maduro",
               "Plátano verde", "Yautía", "Yuca", "Ñame"}

# --- Manual: procesados/DD que USDA no cubre bien. (kcal,P,C,F,fiber,sodium_mg) por 100g. ---
MANUAL_MACROS = {
    "Casabe":              (338, 1.6, 84.0, 0.5, 1.5, 12, True),
    "Estevia":             (0, 0, 0, 0, 0, 0, False),
    "Proteína en polvo":   (380, 80.0, 8.0, 5.0, 0, 200, False),
    "Sal":                 (0, 0, 0, 0, 0, 38758, False),
    "Vinagre blanco":      (18, 0, 0.04, 0, 0, 2, False),
    "Queso de hoja":       (300, 20.0, 3.0, 23.0, 0, 600, True),
    "Longaniza dominicana":(310, 15.0, 2.0, 27.0, 0, 800, True),
    "Salami dominicano":   (320, 16.0, 4.0, 27.0, 0, 1100, True),
}

# [P1-CEILING-COVERAGE-AWARE · 2026-06-15] (gap-audit G5) Micros de las filas MANUAL de alto interés clínico
# que USDA no cubre, ESTIMADOS conservadoramente desde el análogo USDA más cercano (caveat: NO trazables a
# fdc_id; el panel los reporta y el techo coverage-aware ya no los oculta). Antes eran NULL → un plan rico en
# embutidos reportaba satfat 'ok' a un dislipidémico (falso-negativo silencioso) + subestimaba hierro/B12 hemo.
# (sat_fat_g, cholesterol_mg, iron_mg, b12_mcg) por 100g. Análogos: Salami→"Salami, dry/hard, pork";
# Longaniza→"Sausage, pork, fresh, cooked"; Queso de hoja→"Cheese, queso fresco/blanco".
MANUAL_MICROS = {
    "Salami dominicano":    (11.0, 79.0, 1.5, 1.6),
    "Longaniza dominicana": (10.0, 75.0, 1.3, 1.4),
    "Queso de hoja":        (14.0, 70.0, 0.5, 0.8),
}

# --- Overrides post-auditoría [P2-MDDA-NUTRITION-AUDIT · 2026-06-13] ---
# El heurístico "primer resultado de búsqueda" eligió alimentos errados para
# algunos ingredientes (el ranking USDA no siempre pone el correcto primero:
# "Avena"→"Oil, oat", "Leche"→mozzarella, "Carne de res molida"→pavo). Dos fixes:
#   FDC_PIN: fetch por fdc_id EXACTO (determinista, sin ranking) — el fix robusto.
#   QUERY_OVERRIDE: re-buscar con un query más específico (gana sobre USDA_QUERY).
# Poblados con los hallazgos del workflow de auditoría nutricional.
# fdc_ids verificados vía detail endpoint contra la referencia del auditor.
# Cierran los mis-matches del ranking de búsqueda (search no fiable: comas/paréntesis
# rompen el query, "oats"→sorgo, "sweet potato raw"→puffs congelados).
FDC_PIN: dict = {
    "Avena":               173904,  # Cereals, oats (era "Oil, oat" 900kcal)
    "Batata":              168482,  # Sweet potato, raw, unprepared (era hojas)
    "Chinola":             169108,  # Passion-fruit, purple, raw (era jugo)
    "Leche":               171265,  # Milk, whole, 3.25% (era queso mozzarella)
    "Naranja":             169097,  # Oranges, raw, all commercial (era cáscara/jugo)
    "Guineo verde":        173944,  # Bananas, raw (tenía ficha de plátano 152kcal)
    "Mantequilla de maní": 172470,  # Peanut butter, smooth, without salt (era reduced-fat)
}
QUERY_OVERRIDE: dict = {}  # {name: "query más específica"}


def _nut(food, *names):
    """Extrae un nutriente por 100g. Maneja AMBOS shapes de USDA:
    search (`nutrientName`/`value`) y detail /food/{id} (`nutrient.name`/`amount`)."""
    for n in food.get("foodNutrients", []):
        nm = n.get("nutrientName") or (n.get("nutrient") or {}).get("name")
        if nm in names:
            v = n.get("value")
            if v is None:
                v = n.get("amount")
            if v is not None:
                return float(v)
    return 0.0


def _extract_all(food):
    """[P3-MICRONUTRIENTS] Extrae macros + micros por 100g de un food USDA → dict.
    Vit D: prefiere mcg ("Vitamin D (D2 + D3)"); si solo hay IU, convierte (1 mcg = 40 IU).
    Micros None si la fila no los reporta (frecuente en SR Legacy) → degradación grácil."""
    vit_d = _nut(food, "Vitamin D (D2 + D3)")
    if vit_d == 0.0:
        iu = _nut(food, "Vitamin D (D2 + D3), International Units")
        vit_d = round(iu / 40.0, 3) if iu else 0.0
    return {
        "p": _nut(food, "Protein"),
        "c": _nut(food, "Carbohydrate, by difference"),
        "fat": _nut(food, "Total lipid (fat)"),
        "fiber": _nut(food, "Fiber, total dietary"),
        "sodium": _nut(food, "Sodium, Na"),
        "vit_d": vit_d,
        "calcium": _nut(food, "Calcium, Ca"),
        "iron": _nut(food, "Iron, Fe"),
        "b12": _nut(food, "Vitamin B-12"),
        "sugars": _nut(food, "Sugars, total including NLEA", "Sugars, total", "Total Sugars"),
        "potassium": _nut(food, "Potassium, K"),
        "fdc": food.get("fdcId"),
        "desc": food.get("description"),
    }


def fetch_usda_by_fdc(fdc_id):
    """Fetch EXACTO por fdcId (endpoint /food/{id}) — determinista, sin ranking de
    búsqueda. Para pinear el alimento correcto cuando search eligió mal. → dict de _extract_all."""
    url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"
    for attempt in range(3):
        r = requests.get(url, params={"api_key": USDA_KEY}, timeout=25)
        if r.status_code == 429:
            if USDA_KEY == "DEMO_KEY":
                return "RATE_LIMITED"
            print("   ⏳ 429 rate-limit; espera 65s…")
            time.sleep(65)
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return _extract_all(r.json())
    return None


def fetch_usda(query):
    """Devuelve dict de _extract_all por 100g, o None si no hay match. Trae 5 candidatos
    y elige el primero con macros reales (P+C+F>0): hay filas Foundation con todos los
    nutrientes en 0 (e.g. Oil, soybean fdc#748366)."""
    params = {"query": query, "api_key": USDA_KEY, "dataType": "Foundation,SR Legacy", "pageSize": 5}
    for attempt in range(3):
        r = requests.get(USDA_SEARCH, params=params, timeout=25)
        if r.status_code == 429:
            if USDA_KEY == "DEMO_KEY":
                return "RATE_LIMITED"  # 30/hr + 50/día: no esperar, marcar y seguir
            print(f"   ⏳ 429 rate-limit; espera 65s…")
            time.sleep(65)
            continue
        r.raise_for_status()
        foods = r.json().get("foods") or []
        if not foods:
            return None
        best = None
        for f in foods:
            cand = _extract_all(f)
            if best is None:
                best = cand
            if (cand["p"] + cand["c"] + cand["fat"]) > 0:  # primera fila con macros reales gana
                return cand
        return best  # todas en 0 (e.g. agua) → la primera
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--names", default=None, help="lista coma-separada de nombres a re-correr")
    args = ap.parse_args()
    print(f"USDA key: {'DEMO_KEY (30/hr — lento)' if USDA_KEY=='DEMO_KEY' else 'custom (1000/hr)'}")

    conn = psycopg.connect(NEON_URL, autocommit=True)
    cur = conn.cursor()
    cur.execute("SELECT name FROM master_ingredients ORDER BY name")
    names = [r[0] for r in cur.fetchall()]
    if args.only:
        names = [n for n in names if n == args.only]
    if args.names:
        wanted = {x.strip() for x in args.names.split(",")}
        names = [n for n in names if n in wanted]

    ok = manual = miss = 0
    for name in names:
        try:
            # Precedencia: FDC_PIN (exacto) > MANUAL > QUERY_OVERRIDE/USDA_QUERY.
            # micros = {vit_d, calcium, iron, b12, sugars, potassium} (None para manual).
            if name in FDC_PIN:
                res = fetch_usda_by_fdc(FDC_PIN[name])
                if res == "RATE_LIMITED":
                    print(f"  ⏳ {name}: DEMO_KEY agotado"); miss += 1; continue
                if not res:
                    print(f"  ❌ {name}: fdc#{FDC_PIN[name]} sin data"); miss += 1; continue
                p, c, fat, fiber, sodium = res["p"], res["c"], res["fat"], res["fiber"], res["sodium"]
                vit_d, calcium, iron, b12, sugars, potassium = (
                    res["vit_d"], res["calcium"], res["iron"], res["b12"], res["sugars"], res["potassium"])
                fdc, desc, dd, src = res["fdc"], res["desc"], name in _VIVERES_DD, "usda"
                time.sleep(0.8 if USDA_KEY != "DEMO_KEY" else 2.2)
            elif name in MANUAL_MACROS:
                kcal_src = MANUAL_MACROS[name]
                p, c, fat, fiber, sodium, dd = kcal_src[1], kcal_src[2], kcal_src[3], kcal_src[4], kcal_src[5], kcal_src[6]
                # Manual: sin micros USDA → NULL (degradación grácil en el validador).
                vit_d = calcium = sugars = potassium = None
                # [P1-CEILING-COVERAGE-AWARE · 2026-06-15] (G5) hierro/B12 estimados para los embutidos hemo
                # (antes NULL → subestimaba el panel de anemia). satfat/colesterol se escriben aparte (no están
                # en el UPDATE compartido). Si no hay estimado, queda None (degradación grácil).
                _mm = MANUAL_MICROS.get(name)
                iron = _mm[2] if _mm else None
                b12 = _mm[3] if _mm else None
                fdc, src, desc = None, "manual", "manual"
            elif name in QUERY_OVERRIDE or name in USDA_QUERY:
                q = QUERY_OVERRIDE.get(name) or USDA_QUERY[name]
                res = fetch_usda(q)
                if res == "RATE_LIMITED":
                    print(f"  ⏳ {name}: DEMO_KEY agotado (necesita USDA key propia)"); miss += 1; continue
                if not res:
                    print(f"  ❌ {name}: sin match USDA ({q})"); miss += 1; continue
                p, c, fat, fiber, sodium = res["p"], res["c"], res["fat"], res["fiber"], res["sodium"]
                vit_d, calcium, iron, b12, sugars, potassium = (
                    res["vit_d"], res["calcium"], res["iron"], res["b12"], res["sugars"], res["potassium"])
                fdc, desc, dd, src = res["fdc"], res["desc"], name in _VIVERES_DD, "usda"
                time.sleep(0.8 if USDA_KEY != "DEMO_KEY" else 2.2)  # rate-limit cortesía
            else:
                print(f"  ⚠️  {name}: sin mapeo (revisar)"); miss += 1; continue

            kcal = round(4 * p + 4 * c + 9 * fat, 1)  # Atwater consistente con el solver
            tag = "📝manual" if src == "manual" else f"🌐usda#{fdc}"
            _mtag = "" if src == "manual" else f" Na{sodium:4.0f} Fib{fiber:4.1f} VitD{vit_d or 0:4.1f} Ca{calcium or 0:4.0f}"
            print(f"  ✅ {name:28} {tag:16} kcal={kcal:6} P{p:5.1f} C{c:5.1f} F{fat:5.1f}{_mtag}")
            if not args.dry_run:
                cur.execute("""UPDATE master_ingredients SET
                        kcal_per_100g=%s, protein_g_per_100g=%s, carbs_g_per_100g=%s,
                        fats_g_per_100g=%s, fiber_g_per_100g=%s, sodium_mg_per_100g=%s,
                        vitamin_d_mcg_per_100g=%s, calcium_mg_per_100g=%s, iron_mg_per_100g=%s,
                        vitamin_b12_mcg_per_100g=%s, sugars_g_per_100g=%s, potassium_mg_per_100g=%s,
                        nutrition_source=%s, nutrition_source_date=CURRENT_DATE,
                        fdc_id=%s, is_dominican_cultivar=%s
                       WHERE name=%s""",
                    (kcal, p, c, fat, fiber, sodium, vit_d, calcium, iron, b12, sugars, potassium,
                     src, fdc, dd, name))
                # [P1-CEILING-COVERAGE-AWARE · 2026-06-15] (G5) satfat/colesterol de embutidos MANUAL en un
                # UPDATE APARTE: NO están en el UPDATE compartido (que para filas USDA los escribiría NULL y
                # regresaría las 84 ya pobladas). Contenido a filas manuales con estimado → cero riesgo de regresión.
                if src == "manual" and name in MANUAL_MICROS:
                    _satf, _chol = MANUAL_MICROS[name][0], MANUAL_MICROS[name][1]
                    cur.execute(
                        "UPDATE master_ingredients SET saturated_fat_g_per_100g=%s, "
                        "cholesterol_mg_per_100g=%s WHERE name=%s",
                        (_satf, _chol, name))
            if src == "manual":
                manual += 1
            else:
                ok += 1
        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__}: {e}"); miss += 1

    print(f"\nRESUMEN: usda={ok} manual={manual} sin_data={miss} / {len(names)} total"
          f"{' (DRY-RUN, sin escribir)' if args.dry_run else ''}")
    cur.close(); conn.close()


if __name__ == "__main__":
    main()
