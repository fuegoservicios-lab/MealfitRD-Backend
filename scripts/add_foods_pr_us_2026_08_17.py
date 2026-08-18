#!/usr/bin/env python
"""[P1-COUNTRY-SYSTEM-F2 · 2026-08-17] Task 7 -- Catálogo Puerto Rico + Estados Unidos. Inserta
al catálogo (`master_ingredients`, Neon) los alimentos que `scripts/country_catalog_gap.py
--country PR/US` clasificó DROP contra las listas curadas de T1 (PR: 67 items, 13 DROP contra
el catálogo post-T6/284 filas; US: 78 items -- re-expresados a español por el ruling de Task 1,
ver STEP 0 del brief -- 27 DROP), MÁS 6 items que el harness reportaba RESUELVE-BIEN pero
resolvían a un alimento SEMÁNTICAMENTE DISTINTO vía un alias bare pre-existente demasiado amplio
(ej. 'Suero de mantequilla' colisionaba con 'Mantequilla' vía CONTAINS -- verificado en vivo con
`sc.normalize_name`/lectura de `_construir_indice_alias` ANTES de dar de alta, no descubierto en
review -- ver `task-7-report.md` §Colisiones encontradas y evitadas). Y añade UN sinónimo (alias
en fila existente: Parcha→Chinola, mismo patrón fila-vs-sinónimo de T6).

    19 ALTAS PR (fila nueva):
    Panapén              Pernil               Jamón de cocinar     Sofrito
    Recao                Adobo                Alcaparrado          Harina de yuca
    Pique                Pavochón             Bacalaítos            Ron de cocina
    Longaniza puertorriqueña   Chuleta ahumada      Sazón con culantro y achiote
    Aceite de achiote    Queso de papa        Especias para arroz con dulce
    Aceitunas rellenas

    43 ALTAS US (fila nueva -- 27 del DROP original + 16 reclasificadas desde una colisión
    verificada, ver arriba):
    Tocineta             Jamón de sándwich    Salchichas           Crema agria
    Crema mitad y mitad  Bagels               Panecillos ingleses  Pretzels
    Frijoles horneados   Jarabe de arce       Aderezo ranch        Salsa barbacoa
    Kétchup              Salsa inglesa        Malvaviscos          Coditos
    Masa para pie        Galletas Graham      Salsa de salchicha   Ensalada de macarrones
    Chile en polvo       Sazonador para tacos Pepperoni            Salchicha italiana
    Mezcla para panqueques  Wafles            Azúcar morena        Suero de mantequilla
    Pan de maíz          Sémola de maíz       Arándanos rojos      Duraznos
    Pan rallado          Panecillos de mantequilla  Huevos rellenos  Nuez de Castilla
    Nueces pecanas       Queso en hebras      Queso provolone      Carne molida mixta
    Bolitas de papa      Papas ralladas       Chili con carne

    1 SINÓNIMO (alias en fila existente, MISMO alimento nombre regional):
    Parcha→Chinola (P. edulis, misma especie -- CO/T6 ya tenía Curuba/Granadilla como especies
    DISTINTAS de Passiflora con fila propia; Parcha es la MISMA especie que Chinola, solo nombre
    regional puertorriqueño).

NUTRICIÓN: 100% USDA FoodData Central (SR Legacy 2018-04 + Foundation Foods, mismo volcado bulk
que T5/T6 -- sin USDA_API_KEY en este entorno, DEMO_KEY insuficiente). 54 filas con fdc_id real +
nutrition_source='usda' (incluye 2 sustituciones de fuente energía: Foundation reporta Energy
bajo 2047/2048 -- Atwater Factors -- en vez de 1008 para algunas filas, ej. 'Flour, cassava').
8 filas 'manual': 2 SIN fdc_id real (Recao, Adobo -- ausentes de ambos datasets tras múltiples
rondas de búsqueda) + 6 DERIVADAS como blend ponderado de 2 fdc_id reales (Alcaparrado, Pique,
Salsa de salchicha, Ensalada de macarrones, Huevos rellenos, Carne molida mixta -- mismo patrón
que Flor de Jamaica/T6 fix-round 1: 'manual' porque el valor persistido es una TRANSFORMACIÓN,
no lectura directa de un solo fdc; cada blend documentado en `_provenance` con los fdc_id + pesos
exactos usados).

REGLA FILA-vs-SINÓNIMO (contrato Task 7): mismo alimento nombre regional ⇒ alias sobre fila
EXISTENTE (Parcha→Chinola); alimento genuinamente distinto (aunque nutricionalmente cercano) ⇒
fila propia (ej. Aceitunas rellenas NO es sinónimo de Aceitunas: kcal +47%/grasa +88% verificado
con USDA real, cruza el umbral >15% en varias dimensiones -- vs Habichuelas rosadas, que SÍ se
aceptó como sinónimo de Habichuelas rojas porque su delta real es <7% en kcal/proteína/carbs).

COLISIONES ENCONTRADAS Y EVITADAS (proactivamente, ANTES de dar de alta -- no en un review
posterior): 16 de las 43 altas US y 5 de las 19 altas PR existían como "RESUELVE-BIEN" en el
harness ANTES de esta task pero resolvían -- vía un alias bare corto pre-existente sobre OTRA
fila (`_construir_indice_alias`, tier CONTAINS, orden descendente por longitud) -- a un alimento
semánticamente distinto: 'Suero de mantequilla'→'Mantequilla' (líquido bajo en grasa vs grasa
sólida pura), 'Pan de maíz'/'Sémola de maíz'→'Maíz dulce en granos' (bread horneado / grano seco
vs maíz crudo), 'Arándanos rojos'→'Arándanos' (cranberry≠blueberry), 'Nueces pecanas'/'Nuez de
Castilla'→'Almendras fileteadas' (alias genérico 'nueces' preexistente en esa fila), 'Chili con
carne'→'Carne de res' (plato preparado vs carne cruda), etc. -- ver tabla completa en
`task-7-report.md`. Dar de alta cada uno con su PROPIO nombre canónico (igual al string curado
del harness) resuelve el problema vía tier EXACTO (INTENTO 1 de `normalize_name`, que corre ANTES
que cualquier CONTAINS/fuzzy) sin tocar ningún alias de las filas pre-existentes -- MISMO
mecanismo, aplicado proactivamente, que T6 fix-round 1 aplicó reactivamente para 'Queso panela'.
'Nuez de Castilla' (no 'Nueces' bare) se eligió a propósito para Walnuts: sumar el token bare
'nueces' habría creado un empate de longitud NO determinista contra el alias 'nueces' ya
existente en 'Almendras fileteadas' en vez de resolver la colisión -- ver `_provenance` de esa
fila.

PRECIOS: NINGUNA de las 62 lleva precio RD (`price_per_lb=0, price_per_unit=0`) -- mismo patrón
T5/T6 (P1-BAKING-STAPLES generalizado, `shopping_calculator._COUNTRY_CATALOG_UNPRICED_TOKENS`,
knob `MEALFIT_COUNTRY_CATALOG_UNPRICED_KEEP`). PR/US son países BETA (`pricing_mode=
'beta_no_prices'`).

IDEMPOTENTE: mismo mecanismo any-column-diff que T5/T6 post fix-wave deploy-gate -- compara
`fdc_id` + TODAS las columnas nutricionales de `_COLMAP` contra la fila existente; si CUALQUIERA
difiere, UPDATE (nunca solo fdc_id/kcal). Salvaguarda: el UPDATE solo ESCRIBE una columna
nutricional cuando el JSON trae un valor NO-nulo para ella -- un dict parcial nunca sobreescribe
con NULL un dato real ya persistido.

USO:
    cd backend
    python scripts/add_foods_pr_us_2026_08_17.py              # DRY-RUN (altas + sinónimos)
    python scripts/add_foods_pr_us_2026_08_17.py --commit      # inserta/actualiza de verdad

[P2-LOGGER-EXEMPT: script CLI one-shot, la salida a stdout ES el producto]
"""
import datetime
import json
import os
import sys

try:
    from dotenv import load_dotenv
    for _p in (os.path.join(os.path.dirname(__file__), "..", ".env"),
               os.path.join(os.getcwd(), ".env"), "/opt/mealfit/backend/.env"):
        if os.path.exists(_p):
            load_dotenv(_p)
            break
except Exception:
    pass

import psycopg

_AQUI = os.path.dirname(os.path.abspath(__file__))
_NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
COMMIT = "--commit" in sys.argv

# record-key -> columna DB (idéntico a add_foods_es_2026_08_17.py / add_foods_mx_co_2026_08_17.py)
_COLMAP = {
    "kcal": "kcal_per_100g", "protein_g": "protein_g_per_100g", "carbs_g": "carbs_g_per_100g",
    "fats_g": "fats_g_per_100g", "fiber_g": "fiber_g_per_100g", "sugars_g": "sugars_g_per_100g",
    "satfat_g": "saturated_fat_g_per_100g", "sodium_mg": "sodium_mg_per_100g",
    "cholesterol_mg": "cholesterol_mg_per_100g", "calcium_mg": "calcium_mg_per_100g",
    "iron_mg": "iron_mg_per_100g", "potassium_mg": "potassium_mg_per_100g",
    "magnesium_mg": "magnesium_mg_per_100g", "phosphorus_mg": "phosphorus_mg_per_100g",
    "zinc_mg": "zinc_mg_per_100g", "vit_d_mcg": "vitamin_d_mcg_per_100g",
    "b12_mcg": "vitamin_b12_mcg_per_100g", "folate_mcg_dfe": "folate_mcg_dfe_per_100g",
    "vit_a_mcg_rae": "vitamin_a_mcg_rae_per_100g", "vit_c_mg": "vitamin_c_mg_per_100g",
    "vit_e_mg": "vitamin_e_mg_per_100g", "vit_k_mcg": "vitamin_k_mcg_per_100g",
    "selenium_mcg": "selenium_mcg_per_100g", "omega3_ala_g": "omega3_ala_g_per_100g",
}


def _load_json(filename):
    for p in (os.path.join(_AQUI, "data", filename),
              os.path.join(os.getcwd(), "scripts", "data", filename),
              f"/tmp/{filename}"):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    print(f"FATAL: no se encontró {filename}", file=sys.stderr)
    sys.exit(1)


def _apply_new_rows(conn, recs):
    hoy = datetime.date.today()
    puestos = ya = actualizados = 0
    # Mismo mecanismo any-column-diff que T5/T6 post fix-wave deploy-gate (ver docstring del
    # módulo): compara fdc_id + TODAS las columnas nutricionales, no solo fdc_id/kcal.
    # [P1-COUNTRY-SYSTEM-F2 · T7 fix-round 1 · 2026-08-17] `aliases` se sumó al set de comparación
    # tras un bug real: el script reportaba "sin diffs" para Recao/Duraznos aun con el JSON editado
    # (quitando "culantro"/"melocoton") porque _cmp_cols nunca miraba esa columna -- un --commit
    # habría dejado los alias viejos vivos en Neon en silencio. Ver _val_eq: las listas comparan
    # por SET (orden no es semántico en un bag de alias; evita false-diff por reordering).
    _nutri_cols = list(_COLMAP.values())
    _cmp_cols = ["fdc_id", "aliases"] + _nutri_cols
    existen = {
        row[0]: dict(zip(_cmp_cols, row[1:]))
        for row in conn.execute(
            f"SELECT name, {', '.join(_cmp_cols)} FROM public.master_ingredients").fetchall()
    }

    def _val_eq(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        if isinstance(a, list) or isinstance(b, list):
            # aliases: bag de sinónimos, no secuencia -- orden no es semántico.
            return set(a or []) == set(b or [])
        try:
            return abs(float(a) - float(b)) <= 0.05
        except (TypeError, ValueError):
            return a == b

    for r in recs:
        nm = r["name"]
        cols = {
            "slug": r["slug"], "name": nm, "category": r["category"],
            "aliases": r.get("aliases") or [], "default_unit": r["default_unit"],
            "is_dominican_cultivar": bool(r.get("is_dominican_cultivar")),
            "density_g_per_cup": r.get("density_g_per_cup"),
            "density_g_per_unit": r.get("density_g_per_unit"),
            "nutrition_source": r.get("nutrition_source", "usda"), "nutrition_source_date": hoy,
            "fdc_id": r.get("fdc_id"),
            # [T7 · SIN precio RD, a propósito -- ver docstring del módulo]
            "price_per_lb": 0, "price_per_unit": 0,
        }
        for k, dbcol in _COLMAP.items():
            cols[dbcol] = r.get(k)

        if nm in existen:
            db_row = existen[nm]
            diffs = [c for c in _cmp_cols if not _val_eq(db_row.get(c), cols.get(c))]
            if not diffs:
                print(f"  ~ EXISTE (sin diffs), salto: {nm}")
                ya += 1
                continue
            # Salvaguarda: columnas nutricionales solo entran al SET si el JSON trae un valor
            # no-nulo -- nunca clobber con NULL un dato real por un dict parcial.
            nombres_upd = [c for c in cols if c not in ("slug", "name")
                           and (c not in _nutri_cols or cols[c] is not None)]
            set_clause = ", ".join(f"{c} = %s" for c in nombres_upd)
            if COMMIT:
                conn.execute(
                    f"UPDATE public.master_ingredients SET {set_clause} WHERE name = %s",
                    [cols[c] for c in nombres_upd] + [nm])
            print(f"  {'~ ACTUALIZADO' if COMMIT else '~ (dry) actualizaria'}: {nm} [{r['category']}] "
                  f"diffs={diffs} fuente={r.get('_usda_description', '?')!r}")
            actualizados += 1
            continue

        nombres = list(cols.keys())
        if COMMIT:
            conn.execute(
                f"INSERT INTO public.master_ingredients ({', '.join(nombres)}) "
                f"VALUES ({', '.join(['%s'] * len(nombres))})",
                [cols[c] for c in nombres])
        _src = r.get("nutrition_source", "usda")
        print(f"  {'+ INSERTADO' if COMMIT else '+ (dry) insertaría'}: {nm} [{r['category']}] "
              f"{r.get('kcal','?')}kcal/{r.get('protein_g','?')}P "
              f"fdc={r.get('fdc_id')} src={_src} pais={r.get('_country')} SIN-PRECIO "
              f"fuente={r.get('_usda_description', '?')!r}")
        puestos += 1
    if COMMIT:
        conn.commit()
    return puestos, actualizados, ya


def _apply_synonyms(conn, syns):
    added_total = skipped_total = 0
    missing = []
    for s in syns:
        target = s["target"]
        row = conn.execute(
            "SELECT aliases FROM public.master_ingredients WHERE name = %s", (target,)
        ).fetchone()
        if row is None:
            missing.append(target)
            print(f"  ! FILA DESTINO NO ENCONTRADA: {target!r} (sinónimo {s['item']!r} no aplicado)")
            continue
        current = list(row[0] or [])
        current_lower = {str(a).strip().lower() for a in current}
        to_add = [a for a in s["new_aliases"] if a.strip().lower() not in current_lower]
        if not to_add:
            print(f"  ~ YA TIENE todos los alias, salto: {s['item']!r} -> {target!r}")
            skipped_total += 1
            continue
        new_aliases = current + to_add
        if COMMIT:
            conn.execute(
                "UPDATE public.master_ingredients SET aliases = %s WHERE name = %s",
                [new_aliases, target])
        print(f"  {'+ ALIAS AÑADIDO' if COMMIT else '+ (dry) añadiria alias'}: {s['item']!r} -> "
              f"{target!r} += {to_add}")
        added_total += 1
    if COMMIT:
        conn.commit()
    return added_total, skipped_total, missing


def main():
    new_rows = _load_json("new_foods_pr_us_2026_08_17.json")
    synonyms = _load_json("synonyms_pr_us_2026_08_17.json")
    if not _NEON:
        print("FATAL: NEON url ausente", file=sys.stderr)
        return 1

    with psycopg.connect(_NEON) as conn:
        before = conn.execute("SELECT count(*) FROM public.master_ingredients").fetchone()[0]

        print(f"=== Altas de fila nueva ({len(new_rows)}) ===")
        puestos, actualizados, ya = _apply_new_rows(conn, new_rows)

        print(f"\n=== Sinónimos sobre filas existentes ({len(synonyms)}) ===")
        added, skipped, missing = _apply_synonyms(conn, synonyms)

        after = conn.execute("SELECT count(*) FROM public.master_ingredients").fetchone()[0]

        if COMMIT:
            print(f"\nCOMMITTED. filas: insertadas={puestos}, actualizadas={actualizados}, "
                  f"ya-existían={ya}. sinónimos: añadidos={added}, ya-tenían-todo={skipped}, "
                  f"destino-no-encontrado={len(missing)}.")
        else:
            print(f"\nDRY-RUN. filas: insertaría={puestos}, actualizaría={actualizados}, "
                  f"ya-existen={ya}. sinónimos: añadiría={added}, ya-tienen-todo={skipped}, "
                  f"destino-no-encontrado={len(missing)}. Re-corre con --commit.")
        print(f"master_ingredients: {before} -> {after} filas "
              f"({'sin cambio, dry-run' if not COMMIT else f'+{after - before}'}).")
        if missing:
            print(f"\nCONCERN: sinónimos sin fila destino: {missing}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
