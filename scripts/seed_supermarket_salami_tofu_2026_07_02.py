"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de SALAMI + TOFU.

Familias 86-88 del Supermercado RD (capturas del owner, 2026-07-02).
CARNE DE RES: 0 filas — la familia YA estaba cargada completa (80/20 RD$209,
85/15 RD$235, Cadera 96/4 RD$309, Bola 96/4 RD$305, Mister Foods 75/25
RD$149, y "Carne de res Lb" RD$295 calza exacto con los cuadritos — criterio
cubanela: el único tipo ES el genérico, no duplicar). Excluido el alimento
para perros Mr Coco (no es comida humana).

SALAMI (33, La Sirena):
  * DOBLE CALCE con genéricos: "1.47 L" RD$194 = Induveca Estelar 1.47 Lb
    y "3.47 L" RD$459 = Induveca Estelar 3.47 Lb.
  * Induveca (5): Estelar ×2, Naranjal Super Especial 1.47, Naranjal
    Especial 2 Lb (FLAG: RD$93 por 2 Lb ≈ RD$46/lb, muy por debajo del
    resto — posible promo/error del listing, transcrito fiel) y Pork Rind.
  * Sosua (7): Super Especial 1.5/2.5/3.5, Genova Gold 1.5/3, Magistral
    Gold 1 Lb, Balance de Pavo 2.5 Lb (salami de pavo).
  * Chef (4): Cocido Premium 1.47/2.20/3.47 y Cuadrado 0.50 Lb (no disp).
  * Citterio (3): Genoa rebanado por libra, Soppressata 4 Oz, Palitos 2 Oz
    (palitos snack 100% salami — SÍ entra; los "Salami Chips con
    platanitos" Induveca NO: snack frito mixto, excluido).
  * Del Artesano ×2, Margherita ×2 (Genoa americano Lb + 8 Oz), Gustoso,
    Don Pedro Grano de Oro, Hugo Pork curado, El Cid genoa rebanado,
    Nutriciosa pollo, Pollo Cibao, Buen Día ahumado, Noel extra (venta
    por libra, código 11008), Dietz & Watson charcuterie italiana (no
    disp) y "Argentino 1.5 Lb": el TÍTULO dice Sosua pero el empaque
    muestra CHECO (empaque manda, criterio Greenland — FLAG al owner).
  * EXCLUIDOS: 11 combos promocionales (salami+jamoneta/mortadela/jamón y
    salami+salami — bundles multi-producto, no presentaciones de un food).

TOFU (4, **Supermercados Nacional** — notes por fila, criterio familia
mixta): CALCE del genérico "Lata 19 Oz" RD$250 = Roland Queso Tofu de Soya
19 Onz $249.95. House Foods extra firme 12 Oz, medium firm 14 Oz y grilled
super firme 10 Oz (agotado, referencia) + Roland lata.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_salami_tofu_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_salami_tofu_2026_07_02.py --commit   # inserta
"""
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

_NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
COMMIT = "--commit" in sys.argv

SIRENA = "Precio de referencia La Sirena · 2026-07"
NACIONAL = "Precio de referencia Supermercados Nacional · 2026-07"
CARNES = "Carnes, pescados y mariscos"
LEGUM = "Legumbres y proteína vegetal"

SAL = "Salami"

# (food_name, brand, presentation, price_rd, description, category, notes)
ROWS = [
    # ── Salami · Induveca ──
    (SAL, "Induveca", "Estelar 1.47 Lb", 194, "Salami Estelar", CARNES, SIRENA),
    (SAL, "Induveca", "Estelar 3.47 Lb", 459, "Salami Estelar, barra grande", CARNES, SIRENA),
    (SAL, "Induveca", "Naranjal Super Especial 1.47 Lb", 141, "Salami super especial línea Naranjal", CARNES, SIRENA),
    (SAL, "Induveca", "Naranjal Especial 2 Lb", 93, "Salami especial línea Naranjal (precio del listing, verificar)", CARNES, SIRENA),
    (SAL, "Induveca", "Pork Rind 1.5 Lb", 355, "Salami pork rind", CARNES, SIRENA),
    # ── Salami · Sosua ──
    (SAL, "Sosua", "Super Especial 1.5 Lb", 199, "Salami super especial", CARNES, SIRENA),
    (SAL, "Sosua", "Super Especial 2.5 Lb", 319, "Salami super especial", CARNES, SIRENA),
    (SAL, "Sosua", "Super Especial 3.5 Lb", 448, "Salami super especial, barra grande", CARNES, SIRENA),
    (SAL, "Sosua", "Genova Gold 1.5 Lb", 218, "Salami Genova línea Gold", CARNES, SIRENA),
    (SAL, "Sosua", "Genova Gold 3 Lb", 425, "Salami Genova línea Gold, barra grande", CARNES, SIRENA),
    (SAL, "Sosua", "Magistral Gold 1 Lb", 160, "Salami Magistral línea Gold", CARNES, SIRENA),
    (SAL, "Sosua", "Balance de Pavo 2.5 Lb", 152, "Salami de pavo línea Balance", CARNES, SIRENA),
    # ── Salami · Chef ──
    (SAL, "Chef", "Cocido Premium 1.47 Lb", 205, "Salami cocido premium", CARNES, SIRENA),
    (SAL, "Chef", "Cocido Premium 2.20 Lb", 315, "Salami cocido premium", CARNES, SIRENA),
    (SAL, "Chef", "Cocido Premium 3.47 Lb", 465, "Salami cocido premium, barra grande", CARNES, SIRENA),
    (SAL, "Chef", "Cuadrado 0.50 Lb", 88, "Salami cuadrado ideal para sándwich", CARNES, SIRENA),
    # ── Salami · Citterio (italiano) ──
    (SAL, "Citterio", "Genoa Rebanado Lb", 455, "Salami Genoa italiano rebanado, venta por libra", CARNES, SIRENA),
    (SAL, "Citterio", "Paquete Soppressata 4 Oz", 305, "Soppressata salame all natural", CARNES, SIRENA),
    (SAL, "Citterio", "Palitos 2 Oz", 250, "Palitos de salami estilo italiano (snack)", CARNES, SIRENA),
    # ── Salami · resto de marcas ──
    (SAL, "Del Artesano", "Extra Premium 1 Lb", 195, "Salami extra premium 100% carne", CARNES, SIRENA),
    (SAL, "Del Artesano", "Extra Premium 2 Lb", 380, "Salami extra premium 100% carne", CARNES, SIRENA),
    (SAL, "Margherita", "Genoa Americano Lb", 395, "Salami Genoa americano, venta por libra", CARNES, SIRENA),
    (SAL, "Margherita", "Paquete Genoa 8 Oz", 245, "Salami Genoa", CARNES, SIRENA),
    (SAL, "Gustoso", "Especial 1 Lb", 65, "Salami especial económico", CARNES, SIRENA),
    (SAL, "Don Pedro", "Grano de Oro 40 Oz", 225, "Salami Grano de Oro", CARNES, SIRENA),
    (SAL, "Hugo Pork", "Curado 1 Lb", 250, "Salami curado", CARNES, SIRENA),
    (SAL, "El Cid", "Paquete Genoa Rebanado", 205, "Salami Genoa rebanado", CARNES, SIRENA),
    (SAL, "Nutriciosa", "Pollo 2 Lb", 199, "Salami de pollo", CARNES, SIRENA),
    (SAL, "Pollo Cibao", "Super Especial 1.5 Lb", 135, "Salami de pollo super especial", CARNES, SIRENA),
    (SAL, "Buen Día", "Ahumado 2 Lb", 168, "Salami ahumado", CARNES, SIRENA),
    (SAL, "Noel", "Extra Lb", 495, "Salami extra, venta por libra (código 11008)", CARNES, SIRENA),
    (SAL, "Checo", "Argentino 1.5 Lb", 139, "Salami argentino (el listing lo titula Sosua; el empaque dice Checo)", CARNES, SIRENA),
    (SAL, "Dietz & Watson", "Paquete Charcuterie Italiana 8 Oz", 650, "Charcuterie estilo italiano: hot calabrese, peppered salami y hot capocolla", CARNES, SIRENA),
    # ── Tofu (Supermercados Nacional) ──
    ("Tofu", "House Foods", "Paquete Extra Firme 12 Oz", 174.95, "Tofu premium extra firme", LEGUM, NACIONAL),
    ("Tofu", "House Foods", "Paquete Medium Firm 14 Oz", 174.95, "Tofu premium de firmeza media", LEGUM, NACIONAL),
    ("Tofu", "House Foods", "Paquete Grilled Super Firme 10 Oz", 229.95, "Tofu super firme grillado, listo para usar", LEGUM, NACIONAL),
    ("Tofu", "Roland", "Lata 19 Oz", 249.95, "Queso tofu de soya (bean curd) en agua", LEGUM, NACIONAL),
]

_INSERT = """
INSERT INTO public.supermarket_products
    (food_name, brand, presentation, portion_label, duration_label,
     price_rd, notes, category, master_food_name, description, is_verified, active)
VALUES (%s, %s, %s, NULL, NULL, %s, %s, %s, %s, %s, true, true)
ON CONFLICT (lower(food_name), lower(coalesce(brand,'')), lower(coalesce(presentation,'')))
DO NOTHING
RETURNING id
"""


def main():
    if not _NEON:
        print("FATAL: NEON_DATABASE_URL no está definido (.env)")
        sys.exit(1)

    seen = set()
    for row in ROWS:
        if len(row) != 7:
            print(f"FATAL: fila con {len(row)} campos (esperados 7): {row[:3]}")
            sys.exit(1)
        (food, brand, pres, *_rest) = row
        key = (food.lower(), (brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    foods = {r[0] for r in ROWS}
    brands = {r[1] for r in ROWS if r[1]}
    print(f"Seed salami + tofu: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc, category, notes) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, notes, category, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
