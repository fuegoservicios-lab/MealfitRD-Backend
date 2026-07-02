"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de LECHE EVAPORADA (faltantes) y MANTEQUILLA.

Familias 51-52 del Supermercado RD: 31 SKUs transcritos del catálogo de La Sirena
(capturas del owner, 2026-07-02):

  * "Leche evaporada" (+4): la familia YA estaba cargada del seed de leches con
    precios idénticos a las capturas — solo entran las 4 referencias "No
    disponible": Wala cartón UHT 340 gr, Wala lata 170 gr, Wala cartón UHT 1 Lt
    y Carnation 4-pack 290 Ml. FLAGS para el owner: (a) la fila existente
    Carnation sabor queso 135 Ml está a RD$59 que hoy es el precio PROMO (-9%,
    lista RD$65); (b) la fila "Rica Pack 350 gr (2 uds)" RD$240 — la foto
    muestra 4 cartones y 4 × RD$60 (unidad) = RD$240 exacto, probable 4 uds.
  * "Mantequilla" (26): Sosúa (5 — el tarro pasteurizado 1 Lb RD$315 CALZA
    EXACTO con el genérico "Tarro L"; la barra con sal 113 gr ya estaba cargada
    calzando el genérico "Barrita"), Rica (1) + Rica La Vaquita (2),
    Président (6: barras/tarros con y sin sal), Lurpak (3 — barras a precio de
    LISTA RD$179, promos -10%/-12% ignoradas), Asturiana (4), Kerrygold (2),
    Elle & Vire (1), Crystal Farms para untar con canola (1, blend con aceite —
    descripción honesta) y Organic Valley Ghee (1, mantequilla clarificada —
    misma familia, descripción honesta).
  * "Margarina" (food NUEVO, 1): Dorina "con mantequilla" 1 Lb — el empaque
    manda: es MARGARINA refrigerada (con mantequilla añadida), no mantequilla.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_leche_evaporada_mantequilla_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_leche_evaporada_mantequilla_2026_07_02.py --commit   # inserta
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

NOTES = "Precio de referencia La Sirena · 2026-07"
LACTEO = "Lácteos y huevos"

EVAP = "Leche evaporada"
MANT = "Mantequilla"
MARG = "Margarina"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Leche evaporada (solo faltantes; el resto ya estaba del seed de leches) ──
    (EVAP, "Wala", "Cartón UHT 340 gr", 59, "Leche evaporada UHT, cartón", LACTEO),
    (EVAP, "Wala", "Lata 170 gr", 30, "Leche evaporada enriquecida con vitaminas A y D", LACTEO),
    (EVAP, "Wala", "Cartón UHT 1 Lt", 159, "Leche evaporada UHT, cartón familiar", LACTEO),
    (EVAP, "Nestlé Carnation", "Cartón UHT 290 Ml (4 pack)", 239, "Leche evaporada UHT, pack de 4 cartones", LACTEO),
    # ── Mantequilla · Sosúa ──
    (MANT, "Sosúa", "Tarro Pasteurizada 1 Lb", 315, "Mantequilla pasteurizada, tarro", LACTEO),
    (MANT, "Sosúa", "Tarro 0.5 Lb", 165, "Mantequilla pasteurizada, tarro", LACTEO),
    (MANT, "Sosúa", "Barra Sin Sal 0.25 Lb", 99, "Mantequilla sin sal, barra", LACTEO),
    (MANT, "Sosúa", "Tarro Con Sal 1.8 Lb", 580, "Mantequilla pasteurizada con sal, tarro familiar", LACTEO),
    (MANT, "Sosúa", "Gold Con Sal 200 gr", 275, "Mantequilla premium línea Gold, con sal", LACTEO),
    # ── Mantequilla · Rica / La Vaquita ──
    (MANT, "Rica", "Tarro Pasteurizada 1 Lb", 405, "Mantequilla pasteurizada con sal, tarro", LACTEO),
    (MANT, "Rica La Vaquita", "Barra 113 gr", 74, "Mantequilla con sal, barra", LACTEO),
    (MANT, "Rica La Vaquita", "Tarro 0.5 Lb", 159, "Mantequilla con sal, tarro", LACTEO),
    # ── Mantequilla · Président ──
    (MANT, "Président", "Tarro Con Sal 250 gr", 460, "Mantequilla francesa con sal (French butter), tarrina", LACTEO),
    (MANT, "Président", "Tarro Sin Sal 250 gr", 460, "Mantequilla francesa sin sal (French butter), tarrina", LACTEO),
    (MANT, "Président", "Barra Con Sal 100 gr", 179, "Mantequilla francesa con sal, barra", LACTEO),
    (MANT, "Président", "Barra Con Sal 200 gr", 350, "Mantequilla francesa con sal (demi-sel), barra", LACTEO),
    (MANT, "Président", "Barra Sin Sal 100 gr", 179, "Mantequilla francesa sin sal, barra", LACTEO),
    (MANT, "Président", "Barra Sin Sal 200 gr", 350, "Mantequilla francesa sin sal (doux), barra", LACTEO),
    # ── Mantequilla · Lurpak ──
    (MANT, "Lurpak", "Tarrina Spreadable Con Sal 250 gr", 460, "Mantequilla danesa untable (spreadable, slightly salted)", LACTEO),
    (MANT, "Lurpak", "Barra Con Sal 100 gr", 179, "Mantequilla danesa con sal, barra", LACTEO),
    (MANT, "Lurpak", "Barra Sin Sal 100 gr", 179, "Mantequilla danesa sin sal, barra", LACTEO),
    # ── Mantequilla · Asturiana ──
    (MANT, "Asturiana", "Pastilla Tradicional 125 gr", 145, "Mantequilla tradicional de leche española, pastilla", LACTEO),
    (MANT, "Asturiana", "Tarrina Tradicional 250 gr", 305, "Mantequilla tradicional de leche española, tarrina", LACTEO),
    (MANT, "Asturiana", "Barra Sin Lactosa 250 gr", 295, "Mantequilla tradicional sin lactosa, barra", LACTEO),
    (MANT, "Asturiana", "Rulo 250 gr", 290, "Mantequilla tradicional, rulo", LACTEO),
    # ── Mantequilla · importadas premium ──
    (MANT, "Kerrygold", "Barra Sin Sal 200 gr", 470, "Mantequilla pura de Irlanda (grass-fed), sin sal", LACTEO),
    (MANT, "Kerrygold", "Barra Con Sal 200 gr", 470, "Mantequilla pura de Irlanda (grass-fed), con sal", LACTEO),
    (MANT, "Elle & Vire", "Barra Con Sal 200 gr", 335, "Mantequilla gastronómica francesa (demi-sel), barra", LACTEO),
    (MANT, "Crystal Farms", "Tarrina Para Untar con Canola 8 Oz", 239, "Mantequilla para untar con aceite de canola (blend untable, no es 100% mantequilla)", LACTEO),
    (MANT, "Organic Valley", "Tarro Ghee 7.5 Oz", 765, "Ghee: mantequilla clarificada orgánica (pasture-raised), alto punto de humo", LACTEO),
    # ── Margarina (food nuevo — el empaque Dorina dice margarina) ──
    (MARG, "Dorina", "Tarro Con Mantequilla 1 Lb", 250, "Margarina refrigerada con mantequilla, tarrina", LACTEO),
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
        if len(row) != 6:
            print(f"FATAL: fila con {len(row)} campos (esperados 6): {row[:3]}")
            sys.exit(1)
        (food, brand, pres, *_rest) = row
        key = (food.lower(), (brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    foods = {r[0] for r in ROWS}
    brands = {r[1] for r in ROWS if r[1]}
    print(f"Seed leche evaporada + mantequilla: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc, category) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, NOTES, category, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
