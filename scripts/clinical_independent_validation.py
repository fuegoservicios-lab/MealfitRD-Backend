"""[P1-INDEPENDENT-VALIDATION · 2026-06-26] (auditoría gap #2) Validación INDEPENDIENTE del catálogo.

PROBLEMA RAÍZ (auditoría "¿100%?"): la "validación" de precisión existente (clinical_validation_export /
benchmark_macro_compliance) recomputa los macros desde el MISMO `master_ingredients` que usa el motor → es
un proxy AUTO-REFERENCIAL: si el catálogo tiene un valor errado, tanto el plan como su "validación" comparten
el error → 0% de detección. Sin una fuente de verdad INDEPENDIENTE, cualquier número de precisión es
autoafirmado.

QUÉ HACE ESTE HARNESS: compara los macros per-100g del catálogo vivo contra una tabla de referencia USDA
FoodData Central HARDCODEADA AQUÍ (independiente del catálogo) para un set de staples es-DO. Rompe la
circularidad para la muestra: un error de catálogo que la validación auto-referencial no ve, este SÍ lo caza.

LÍMITES HONESTOS (lo que ESTO NO cierra — requiere acción del owner, NO es code-closeable):
  - NO sustituye la revisión por un NUTRICIONISTA LICENCIADO es-DO (criterio clínico del TARGET, no solo
    precisión de entrega) — sigue pendiente (clinical_validation.md).
  - NO es un benchmark externo COMPLETO (NutriBench/INCAP/LATINFOODS, dataset comercial) — esta tabla es una
    muestra curada (~24 staples), no el catálogo entero.
  - Raw vs cocido: la referencia anota el estado; tolerancias generosas absorben varianza cultivar/preparación.

USO:  python scripts/clinical_independent_validation.py [--strict]
  (--strict → exit 1 si algún macro excede su tolerancia; sin él, reporta y exit 0.)
"""
import os
import sys
import unicodedata

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

_NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")


def _norm(s: str) -> str:
    """lower + sin acentos para matchear nombres del catálogo vs la referencia."""
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    return s.strip().lower()

# Tolerancias por macro (fracción). Generosas: absorben cultivar/preparación/raw-cocido sin perder señal de
# un error GROSERO de catálogo (que es lo que importa cazar). kcal más estricto (es el agregado).
_TOL = {"kcal": 0.15, "protein": 0.22, "carbs": 0.22, "fats": 0.25}

# Referencia USDA FoodData Central / SR Legacy, per 100g, INDEPENDIENTE del catálogo. (kcal, protein, carbs, fats).
# Nombres = los del catálogo es-DO (para que lookup resuelva). `state` documenta raw/cocido de la referencia.
_USDA_REF = {
    # name:                 (kcal, protein, carbs, fats, state)
    # [calibración 2026-06-26] El catálogo guarda arroz/lentejas SECOS (crudos) y atún LIGHT; la referencia
    # USDA usa el MISMO estado (no se dobla para ocultar errores — se corrige el estado de la referencia).
    "Pechuga de pollo":     (120, 22.5, 0.0, 2.6, "raw, skinless"),
    "Arroz blanco":         (365, 7.1, 80.0, 0.7, "DRY/uncooked (catálogo lo guarda crudo)"),
    "Atún en agua":         (86, 19.4, 0.0, 0.8, "canned LIGHT, in water"),
    "Pan blanco familiar":  (265, 9.0, 49.0, 3.2, "as-eaten"),
    "Avena":                (389, 16.9, 66.3, 6.9, "dry"),
    "Aguacate":             (160, 2.0, 8.5, 14.7, "raw"),
    "Manzana":              (52, 0.3, 13.8, 0.2, "raw"),
    "Guineo":               (89, 1.1, 22.8, 0.3, "raw"),
    "Brócoli":              (34, 2.8, 6.6, 0.4, "raw"),
    "Zanahoria":            (41, 0.9, 9.6, 0.2, "raw"),
    "Papa":                 (77, 2.0, 17.5, 0.1, "raw"),  # FLAG genuino: catálogo ~61 (bajo) → revisión
    "Batata":               (86, 1.6, 20.1, 0.1, "raw"),
    "Yuca":                 (160, 1.4, 38.1, 0.3, "raw"),
    "Aceite de oliva":      (884, 0.0, 0.0, 100.0, "oil"),
    "Lentejas":             (352, 24.6, 63.0, 1.1, "DRY/uncooked (catálogo lo guarda crudo)"),
    "Cerdo":                (152, 21.0, 0.0, 8.0, "raw, composite cut"),
    "Carne de res":         (143, 21.0, 0.0, 6.0, "raw, lean"),
    "Coliflor":             (25, 1.9, 5.0, 0.3, "raw"),
    "Tomate":               (18, 0.9, 3.9, 0.2, "raw"),
    "Cebolla":              (40, 1.1, 9.3, 0.1, "raw"),
    "Piña":                 (50, 0.5, 13.1, 0.1, "raw"),
    "Lechosa":              (43, 0.5, 11.0, 0.3, "raw"),
    "Melón":                (34, 0.8, 8.2, 0.2, "raw, cantaloupe"),
    "Huevo":                (143, 12.6, 0.7, 9.5, "raw, whole"),
}

_CATALOG_FIELDS = {"kcal": "kcal", "protein": "protein", "carbs": "carbs", "fats": "fats"}


def main():
    strict = "--strict" in sys.argv
    if not _NEON:
        print("FATAL: NEON url ausente"); sys.exit(1)
    # Carga el catálogo vivo via SQL directo (sin el pool de app, que no se inicializa en un script).
    catalog = {}
    with psycopg.connect(_NEON) as conn:
        rows = conn.execute(
            "SELECT name, kcal_per_100g, protein_g_per_100g, carbs_g_per_100g, fats_g_per_100g "
            "FROM public.master_ingredients").fetchall()
    for nm, k, p, c, f in rows:
        catalog[_norm(nm)] = {"name": nm, "kcal": k, "protein": p, "carbs": c, "fats": f}

    print(f"=== Validación INDEPENDIENTE catálogo vs referencia USDA ({len(_USDA_REF)} staples) ===")
    print(f"{'Ingrediente':24} {'macro':8} {'catálogo':>10} {'USDA-ref':>10} {'Δ%':>8}  estado")
    n_checked = n_flag = n_missing = 0
    flags = []
    for name, (rk, rp, rc, rf, state) in _USDA_REF.items():
        row = catalog.get(_norm(name))
        if not row or row.get("kcal") is None:
            n_missing += 1
            print(f"{name:24} {'—':8} {'NO RESUELVE EN CATÁLOGO':>30}")
            continue
        cat = {"kcal": row["kcal"], "protein": row["protein"], "carbs": row["carbs"], "fats": row["fats"]}
        ref = {"kcal": rk, "protein": rp, "carbs": rc, "fats": rf}
        for macro in ("kcal", "protein", "carbs", "fats"):
            n_checked += 1
            cv, rv = cat[macro], ref[macro]
            if cv is None:
                n_flag += 1
                flags.append(f"{name}.{macro}: catálogo NULL (macro core sin dato)")
                print(f"{name:24} {macro:8} {'NULL':>10} {rv:>10.1f} {'—':>8}  ⚠FLAG")
                continue
            cv = float(cv)
            # delta relativo robusto: denominador = max(|ref|, piso) para no explotar cerca de 0.
            denom = max(abs(rv), 1.0 if macro != "kcal" else 10.0)
            d = (cv - rv) / denom
            over = abs(d) > _TOL[macro]
            mark = "  ⚠FLAG" if over else ""
            if over:
                n_flag += 1
                flags.append(f"{name}.{macro}: catálogo {cv} vs USDA {rv} (Δ{d:+.0%})")
            print(f"{name:24} {macro:8} {cv:>10.1f} {rv:>10.1f} {d:>+7.0%}{mark}  {state if macro=='kcal' else ''}")

    print(f"\n=== RESUMEN: {n_checked} celdas comparadas | {n_flag} fuera de tolerancia | {n_missing} no resueltos ===")
    if flags:
        print("FLAGS (revisar — posible error de catálogo O diferencia raw/cocido):")
        for f in flags:
            print(f"  - {f}")
    else:
        print("✓ Todas las celdas dentro de tolerancia: el catálogo es CONSISTENTE con la referencia USDA "
              "independiente para la muestra (rompe la circularidad auto-referencial para estos staples).")
    if strict and n_flag:
        sys.exit(1)


if __name__ == "__main__":
    main()
