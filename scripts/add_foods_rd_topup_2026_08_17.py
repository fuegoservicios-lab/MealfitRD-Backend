#!/usr/bin/env python
"""[P1-COUNTRY-SYSTEM-F2 · 2026-08-17] Task 8 -- Top-up RD (rd_drops.json, `--rd-drops`, T1).

Cierra el drop mas alto por RD de los 7 medidos (338 corridas de `_creativity_kpi_job`/30d):

    1 ALTA (fila nueva, SIN precio RD a proposito -- ver ruling del brief, mismo mecanismo
    P1-BAKING-STAPLES/P1-COUNTRY-CATALOG-UNPRICED que T5-T7 pero por falta de precio verificado
    HOY, no por pais beta):
        Hummus (6 drops)

    2 SINONIMOS (alias sobre fila EXISTENTE, ya PRICED -- no hace falta darlas de alta):
        mereyes -> Merey        (62 drops, el MAS dropeado de los 7)
        rabanos -> Rábano       (refuerzo determinista; el drop real es "rábanos en láminas",
                                  4/30d, cerrado principalmente por el stop 'en láminas' en
                                  shopping_calculator._NORMALIZE_STOPS -- ver ese diff)

    4 ITEMS SIN cambio de catálogo (decisiones documentadas en task-8-report.md, NO en este
    script -- nada que insertar/actualizar):
        "2–3 ciruelas" (16 drops) -- contaminación de parsing (rango numérico lider), fix en
            shopping_calculator._preprocess_nlp_quantities. Ciruela YA existe con precio.
        tortilla (8 drops, bare) -- AMBIGUO (tortilla de huevos/omelette vs tortilla de
            trigo/integral/maíz -- gluten/huevo, riesgo clínico real). Se deja dropeando.
        requesón (6 drops) -- YA RESUELTO como efecto colateral de T5 (fila "Requesón" exacta,
            fdc 170851, registrada en _COUNTRY_CATALOG_UNPRICED_TOKENS desde T5). Verificado en
            vivo: normalize_name('requesón') -> 'Requesón'. Perseguir el "Queso ricotta" que
            citaba el brief original crearía un alias muerto (shadowed por el tier EXACT de la
            fila de T5) o exigiría un guard nuevo que retargetea territorio de T5 -- no vale la
            pena por una diferencia meramente cosmética (con precio vs CATÁLOGO SIN PRECIO).
        azúcar (4 drops) -- INTENCIONAL. `_DM2_SUGAR_SUBS` (condition_rules.py) trata
            azucar/azúcar/sugar como token OFENSOR a sustituir por Stevia en DM2/bariátrico;
            `IGNORED_TRACKING_TERMS` (constants.py) documenta explícitamente "no hay fila de
            catálogo ni la merece". Cero fila azúcar en todo el catálogo (346 filas, verificado).

NUTRICIÓN: Hummus vía USDA FoodData Central API pública (query puntual, DEMO_KEY -- 1 sola
llamada, no el bloqueo de volumen que T5-T7 documentaron para 32-62 altas), SR Legacy fdc_id
174289 "Hummus, commercial" (match directo, sin sustitución). Validación Atwater pre-commit:
4×7.78 + 4×15.00 + 9×17.82 = 251.5 vs kcal declarado 237 -> ratio 0.942, dentro de banda
[0.40, 1.40].

PRECIO: Hummus SIN precio RD a propósito (`price_per_lb=0, price_per_unit=0`) -- listado bajo
CATÁLOGO SIN PRECIO vía `shopping_calculator._COUNTRY_CATALOG_UNPRICED_TOKENS` (token "hummus"
añadido en el mismo commit de esta task) hasta que `supermarket_products` lo precifique.

IDEMPOTENTE: mismo mecanismo any-column-diff + ALIASES-AWARE que T7 post fix-round 1 (bug real:
el script original de T6/T7 solo comparaba `fdc_id` + columnas nutricionales, nunca `aliases` --
un --commit habría dejado alias viejos vivos en Neon en silencio aun con el JSON corregido). Este
script nace YA con `aliases` en `_cmp_cols` desde el primer commit (no hace falta un fix-round
propio para este bug conocido).

USO:
    cd backend
    python scripts/add_foods_rd_topup_2026_08_17.py              # DRY-RUN (alta + sinónimos)
    python scripts/add_foods_rd_topup_2026_08_17.py --commit      # inserta/actualiza de verdad

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

# record-key -> columna DB (idéntico a add_foods_es/mx_co/pr_us_2026_08_17.py)
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
    # Any-column-diff + ALIASES-AWARE desde el primer commit (lección T7 fix-round 1 — ver
    # docstring del módulo): compara fdc_id + aliases + TODAS las columnas nutricionales.
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
            # [T8 · SIN precio RD, a propósito -- ver docstring del módulo]
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
    new_rows = _load_json("new_foods_rd_topup_2026_08_17.json")
    synonyms = _load_json("synonyms_rd_topup_2026_08_17.json")
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
