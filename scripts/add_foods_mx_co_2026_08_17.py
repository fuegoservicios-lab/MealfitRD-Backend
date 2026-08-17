#!/usr/bin/env python
"""[P1-COUNTRY-SYSTEM-F2 · 2026-08-17] Task 6 — Catálogo México + Colombia. Inserta al catálogo
(`master_ingredients`, Neon) los alimentos que `scripts/country_catalog_gap.py --country MX/CO`
clasificó DROP contra las listas curadas de T1 (30 DROP de 76 para MX, 26 DROP de 74 para CO,
0 SUSTITUCION-SILENCIOSA en ambas), Y añade sinónimos (aliases) a filas YA EXISTENTES cuando el
alimento pedido es el MISMO alimento con nombre regional distinto (regla fila-vs-sinónimo del
contrato de la Task 6 — ver `task-6-report.md`).

    46 ALTAS (fila nueva):
    Tortilla de maíz    Chile jalapeño      Chile serrano       Chile poblano
    Chile chipotle      Chile guajillo      Chile ancho         Chile habanero
    Chile de árbol      Chile pasilla       Chile mulato        Nopal
    Jícama              Epazote             Chorizo mexicano    Chorizo verde
    Cecina              Frijoles refritos   Crema mexicana      Tuna de nopal
    Flor de Jamaica     Xoconostle          Achiote             Hoja santa
    Chocolate de mesa   Panela              Huitlacoche         Chicharrón
    Chorizo santarrosano Trucha             Chontaduro          Frijol cargamanto
    Suero costeño       Guascas             Arracacha           Lulo
    Curuba              Uchuva              Arequipe            Natilla
    Champús             Gallina criolla     Borojó              Feijoa
    Granadilla          Mora

    10 SINÓNIMOS (alias en fila existente, MISMO alimento nombre regional):
    Jitomate→Tomate · Chile morrón→Ají morrón (alias directo en la fila nueva Chile jalapeño para
    'chile cuaresmeño'; ambos consumidos vía scripts/data/synonyms_mx_co_2026_08_17.json salvo
    los que ya viven embebidos en los aliases de una fila NUEVA de este mismo lote: chile
    cuaresmeño→Chile jalapeño, piloncillo→Panela, color(bijol)→Achiote) · Mazorca→Maíz dulce en
    granos · Choclo→Maíz dulce en granos · Malanga→Yautía · Cuchuco de trigo→Bulgur.

NUTRICIÓN: 100% USDA FoodData Central (SR Legacy + Foundation, mismo volcado bulk que T5 — sin
USDA_API_KEY en este entorno, DEMO_KEY insuficiente, ver docstring de `add_foods_es_2026_08_17.py`).
DOS filas SIN fdc_id real (`nutrition_source='manual'`, NO 'usda' — el CHECK constraint de la
columna solo permite usda/off/faoinfoods/manual/NULL, 'manual' YA lo usan 16 filas pre-existentes):
Achiote y Hoja santa — CERO entrada USDA cubre semilla de achiote/hoja santa (confirmado tras 5
rondas de búsqueda con sinónimos: achiote/annatto/bixa/urucu/bija, hoja santa/acuyo — para achiote
la única real, 'Seasoning mix, dry, sazon, coriander & annatto', es un sazonador con
sodium_mg=17000mg/100g (~42.5% NaCl-equivalente aplicando el factor sodio→sal ×2.5; NO "74% sal
en peso" — corrección fix-round 1 · 2026-08-17, el número original confundía el mg de sodio
crudo con un porcentaje de sal sin aplicar el factor de conversión) — usarla representaría el
achiote puro como sal casi pura, una distorsión clínicamente relevante para un usuario HTA/renal.
Estimación conservadora documentada por fila en `_provenance` del JSON — ver Concern §11 del
reporte de la task para el detalle.
Flor de Jamaica SÍ tiene fdc_id real (168170, 'Roselle, raw') desde fix-round 1 · 2026-08-17 — la
búsqueda original de 5 rondas usó 'hibiscus'/'jamaica' y no dio con ella; una 6ª ronda con la
palabra 'roselle' (nombre en inglés de la especie Hibiscus sabdariffa) sí la encontró. La fila se
deriva por un factor de deshidratación desde el dato crudo/húmedo de esa entrada (ver
`_provenance` de esa fila en el JSON para el cálculo completo, con chequeo de autoconsistencia
Atwater); `nutrition_source` se mantiene 'manual' porque el valor persistido es un DERIVADO, no
una lectura directa del fdc — pero ahora cita una fuente USDA real en vez de ser una estimación
sin ancla.

REGLA FILA-vs-SINÓNIMO (contrato Task 6): mismo alimento nombre regional ⇒ alias sobre la fila
EXISTENTE (Jitomate→Tomate, Mazorca/Choclo→Maíz dulce en granos, Malanga→Yautía, Cuchuco de
trigo→Bulgur); alimento genuinamente distinto (aunque nutricionalmente cercano a otro) ⇒ fila
propia (ej. Frijol cargamanto NO es sinónimo de Frijoles pintos — mismo patrón que Judías
blancas/T5 vs Habichuelas blancas: variedades culturalmente distintas, filas separadas).

HOMÓGRAFOS (ver reporte §homógrafos): Tortilla de maíz (MX) vs Tortilla de trigo/integral (RD)
vs Tortilla española (ES, T5) — misma palabra, TRES alimentos, filas sin alias compartido. Tuna
de nopal (fruta, MX) vs Atún en agua (pescado) — 'Atún en agua' NUNCA lleva 'tuna' entre sus
aliases, cero colisión. Mora (CO, blackberry) vs 'mora azul' (alias PRE-EXISTENTE de Arándanos,
blueberry) — alias de Mora nunca incluyen 'azul'.

PRECIOS: NINGUNA de las 46 lleva precio RD (`price_per_lb=0, price_per_unit=0`) — mismo patrón
T5 (P1-BAKING-STAPLES generalizado, `shopping_calculator._COUNTRY_CATALOG_UNPRICED_TOKENS`, knob
`MEALFIT_COUNTRY_CATALOG_UNPRICED_KEEP`). MX/CO son países BETA (`pricing_mode='beta_no_prices'`).

IDEMPOTENTE: dos modos —
  1) Altas de fila nueva: salta por `name` existente CON el mismo `fdc_id` Y el mismo `kcal`
     (tolerancia 0.05); si CUALQUIERA de los dos difiere -- `fdc_id` (re-sourceo) O `kcal`
     (corrección de una fila `manual` sin fdc_id, ej. Atwater-consistencia, micro-round 2 T6) --
     UPDATE (mismo patrón fix-round-1 de T5). El chequeo de `kcal` es proxy de "algo en las
     macros cambió" sin listar cada columna: en este script kcal SIEMPRE es Atwater-derivada de
     las macros (nunca un valor independiente), así que un cambio real de macros implica cambio
     de kcal.
  2) Sinónimos: `UPDATE ... SET aliases = aliases || nuevos_no_presentes` -- solo añade los alias
     que la fila destino AÚN NO tiene (append idempotente, nunca duplica, nunca pisa un alias
     existente). Si la fila destino no existe (typo/orden de ejecución), lo reporta y NO falla el
     resto del batch.

USO:
    cd backend
    python scripts/add_foods_mx_co_2026_08_17.py              # DRY-RUN (altas + sinónimos)
    python scripts/add_foods_mx_co_2026_08_17.py --commit      # inserta/actualiza de verdad

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

# record-key → columna DB (idéntico a add_foods_es_2026_08_17.py / lotes previos)
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
    # [micro-round 2 T6 · 2026-08-17] antes solo comparaba `fdc_id` -- una fila `manual`
    # (fdc_id=None en DB Y en el JSON nuevo, sin re-sourceo) que necesita SOLO una corrección
    # de kcal/macros (ej. Atwater-consistencia) nunca disparaba el UPDATE: `None == None` la
    # saltaba en silencio como "EXISTE (fdc_id igual)". Ahora también compara `kcal` (tolerancia
    # 0.05 por redondeo float/Decimal) -- ver docstring del módulo.
    existen = {
        row[0]: (row[1], row[2])
        for row in conn.execute(
            "SELECT name, fdc_id, kcal_per_100g FROM public.master_ingredients").fetchall()
    }
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
            # [T6 · SIN precio RD, a propósito -- ver docstring del módulo]
            "price_per_lb": 0, "price_per_unit": 0,
        }
        for k, dbcol in _COLMAP.items():
            cols[dbcol] = r.get(k)

        if nm in existen:
            db_fdc, db_kcal = existen[nm]
            new_kcal = cols.get("kcal_per_100g")
            kcal_igual = (db_kcal is None and new_kcal is None) or (
                db_kcal is not None and new_kcal is not None
                and abs(float(db_kcal) - float(new_kcal)) <= 0.05
            )
            if db_fdc == cols["fdc_id"] and kcal_igual:
                print(f"  ~ EXISTE (fdc_id+kcal igual), salto: {nm}")
                ya += 1
                continue
            nombres_upd = [c for c in cols if c not in ("slug", "name")]
            set_clause = ", ".join(f"{c} = %s" for c in nombres_upd)
            if COMMIT:
                conn.execute(
                    f"UPDATE public.master_ingredients SET {set_clause} WHERE name = %s",
                    [cols[c] for c in nombres_upd] + [nm])
            print(f"  {'~ ACTUALIZADO' if COMMIT else '~ (dry) actualizaria'}: {nm} [{r['category']}] "
                  f"fdc_id {db_fdc} -> {cols['fdc_id']} kcal {db_kcal} -> {new_kcal} "
                  f"fuente={r.get('_usda_description', '?')!r}")
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
    new_rows = _load_json("new_foods_mx_co_2026_08_17.json")
    synonyms = _load_json("synonyms_mx_co_2026_08_17.json")
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
