"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de ACEITUNA.

Familia 70 del Supermercado RD: 82 SKUs transcritos del catálogo de La Sirena
(12 capturas del owner, 2026-07-02). La familia más grande hasta ahora junto a
yogurt regular y cerdo.

  * El frasco Goya "Para Ensalada con Pimiento 5 Oz" RD$145 CALZA EXACTO con el
    genérico "Frasco 5 Oz" ya cargado (categoría Otros — se mantiene por
    coherencia de familia).
  * 8 marcas: El Serpis (33 — normalizado desde títulos "Serpis"/"El Serpis";
    latas rellenas de anchoa en 5 tamaños + ligeras 35% menos sal, línea
    Sabores, 3-packs, doypacks Let's Go snack), Goya (24 — manzanilla con
    pimiento en 6 tamaños, gordales, stuffed line: salmón/jamón serrano/queso
    azul/manchego/chorizo/anchoa picada, 2 alcaparrados — mezcla de aceitunas
    con alcaparras y pimientos, en familia con descripción honesta),
    Maestranza (10), Wala (4), Borges (4), La Explanada (4, gourmet),
    El Corte Inglés (2) y Valenciana (1, precio de LISTA RD$115 — promo -7%
    RD$107 ignorada).
  * Los "No disponible" entran como referencia (patrón de siempre).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_aceituna_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_aceituna_2026_07_02.py --commit   # inserta
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
CAT = "Otros"

ACEIT = "Aceituna"

# (food_name, brand, presentation, price_rd, description)
ROWS = [
    # ── Wala ──
    (ACEIT, "Wala", "Lata Rellenas de Anchoa 300 gr", 115, "Aceitunas verdes rellenas de anchoa"),
    (ACEIT, "Wala", "Lata Manzanilla Sin Hueso 300 gr", 115, "Aceitunas manzanilla sin hueso"),
    (ACEIT, "Wala", "Lata Rellenas de Pimiento 300 gr", 115, "Aceitunas verdes rellenas de pimiento"),
    (ACEIT, "Wala", "Frasco 14 Oz", 101, "Aceitunas verdes, frasco 397 gr"),
    # ── El Serpis · rellenas de anchoa ──
    (ACEIT, "El Serpis", "Lata Rellenas de Anchoa 170 gr", 95, "Aceitunas verdes rellenas de anchoa"),
    (ACEIT, "El Serpis", "Lata Rellenas de Anchoa 200 gr", 145, "Aceitunas verdes rellenas de anchoa"),
    (ACEIT, "El Serpis", "Lata Rellenas de Anchoa 300 gr", 198, "Aceitunas verdes rellenas de anchoa"),
    (ACEIT, "El Serpis", "Lata Gran Selección Rellenas de Anchoa 350 gr", 199, "Aceitunas rellenas de anchoa, línea Gran Selección"),
    (ACEIT, "El Serpis", "Lata Rellenas de Anchoa 850 gr", 495, "Aceitunas verdes rellenas de anchoa, lata familiar"),
    (ACEIT, "El Serpis", "Lata Anchoa Ligeras 300 gr", 179, "Aceitunas rellenas de anchoa ligeras, 35% menos sal"),
    (ACEIT, "El Serpis", "Lata Rellenas de Anchoa (3 pack 120 gr)", 298, "Aceitunas rellenas de anchoa, 3 latitas de 120 gr"),
    (ACEIT, "El Serpis", "Lata Gran Selección Anchoa (3 pack 120 gr)", 309, "Aceitunas rellenas de anchoa Gran Selección, 3 latitas"),
    (ACEIT, "El Serpis", "Lata Anchoa Ligeras (3 pack)", 279, "Aceitunas rellenas de anchoa ligeras 35% menos sal, 3 latitas"),
    # ── El Serpis · rellenas de pimiento ──
    (ACEIT, "El Serpis", "Lata Rellenas de Pimiento 350 gr", 184, "Aceitunas verdes rellenas de pimiento"),
    (ACEIT, "El Serpis", "Frasco Rellenas de Pimiento 340 gr", 209, "Aceitunas rellenas de pimiento (red pepper), Las Tradicionales"),
    (ACEIT, "El Serpis", "Bolsa Rellenas de Pimiento 175 gr", 99, "Aceitunas rellenas de pimiento, doypack"),
    (ACEIT, "El Serpis", "Lata Rellenas de Pimiento (3 pack 120 gr)", 345, "Aceitunas rellenas de pimiento, 3 latitas de 120 gr"),
    # ── El Serpis · manzanilla / verdes ──
    (ACEIT, "El Serpis", "Bolsa Manzanilla Sin Hueso 175 gr", 94, "Aceitunas manzanilla sin hueso, doypack"),
    (ACEIT, "El Serpis", "Frasco Manzanilla Con Hueso 235 gr", 150, "Aceitunas manzanilla verde con hueso, Las Tradicionales"),
    (ACEIT, "El Serpis", "Frasco Sin Hueso 235 gr", 170, "Aceitunas verdes sin hueso, Tradicionales"),
    (ACEIT, "El Serpis", "Frasco Manzanilla Sin Hueso 340 gr", 205, "Aceitunas verdes sin hueso (pitted), Tradicionales"),
    (ACEIT, "El Serpis", "Bolsa Verdes Con Hueso 175 gr", 92, "Aceitunas verdes enteras con hueso, doypack"),
    (ACEIT, "El Serpis", "Frasco Verdes Con Hueso 340 gr", 185, "Aceitunas verdes enteras con hueso, Tradicionales"),
    (ACEIT, "El Serpis", "Frasco Verdes en Rodajas 260 gr", 159, "Aceitunas verdes en rodajas (sliced)"),
    # ── El Serpis · negras ──
    (ACEIT, "El Serpis", "Bolsa Negras en Rodajas 175 gr", 95, "Aceitunas negras en rodajas, doypack"),
    (ACEIT, "El Serpis", "Lata Negras Con Hueso 300 gr", 255, "Aceitunas negras con hueso"),
    (ACEIT, "El Serpis", "Lata Negras Sin Hueso 350 gr", 190, "Aceitunas negras sin hueso"),
    (ACEIT, "El Serpis", "Lata Negras Sin Hueso 200 gr", 145, "Aceitunas cacereñas negras sin hueso"),
    # ── El Serpis · línea Sabores ──
    (ACEIT, "El Serpis", "Lata Sabores Chili 300 gr", 190, "Aceitunas rellenas sabor chili"),
    (ACEIT, "El Serpis", "Lata Sabores Queso Azul 300 gr", 190, "Aceitunas rellenas sabor queso azul"),
    (ACEIT, "El Serpis", "Lata Sabores Queso Manchego 300 gr", 190, "Aceitunas rellenas sabor queso manchego"),
    (ACEIT, "El Serpis", "Lata Sabores Jamón Serrano 300 gr", 190, "Aceitunas rellenas sabor jamón serrano"),
    (ACEIT, "El Serpis", "Lata Sabores Atún 300 gr", 190, "Aceitunas rellenas sabor atún"),
    (ACEIT, "El Serpis", "Frasco Sabores Chorizo Picante 235 gr", 190, "Aceitunas rellenas sabor chorizo picante"),
    (ACEIT, "El Serpis", "Frasco Sabores Anchoa 235 gr", 135, "Aceitunas rellenas sabor anchoa"),
    # ── El Serpis · snacks Let's Go ──
    (ACEIT, "El Serpis", "Bolsa Let's Go Mix con Tomate Seco 70 gr", 117, "Mix de aceitunas verdes y negras sin hueso con tomate seco, snack"),
    (ACEIT, "El Serpis", "Bolsa Let's Go Verdes con Paprika 70 gr", 117, "Aceitunas verdes sin hueso con paprika, snack"),
    # ── Goya · manzanilla con pimiento ──
    (ACEIT, "Goya", "Frasco Manzanilla Rellenas de Pimiento 4.01 Oz", 140, "Aceitunas manzanilla españolas rellenas de pimiento"),
    (ACEIT, "Goya", "Frasco Manzanilla Rellenas de Pimiento 5.6 Oz", 160, "Aceitunas manzanilla rellenas de pimiento picado (minced)"),
    (ACEIT, "Goya", "Frasco Manzanilla Rellenas de Pimiento 5.75 Oz", 195, "Aceitunas manzanilla españolas rellenas de pimiento"),
    (ACEIT, "Goya", "Frasco Manzanilla Rellenas de Pimiento 7.52 Oz", 200, "Aceitunas manzanilla españolas rellenas de pimiento"),
    (ACEIT, "Goya", "Frasco Manzanilla Pimiento Bajo en Sal 355 gr", 249, "Aceitunas manzanilla rellenas de pimiento, bajo en sal"),
    (ACEIT, "Goya", "Frasco Manzanilla Enteras Bajo Sal 4 Oz", 200, "Aceitunas manzanilla españolas enteras, bajo en sal"),
    (ACEIT, "Goya", "Frasco Rellenas de Pimiento 269 gr", 379, "Aceitunas manzanilla españolas rellenas de pimiento"),
    # ── Goya · para ensalada ──
    (ACEIT, "Goya", "Frasco Para Ensalada con Pimiento 5 Oz", 145, "Aceitunas manzanilla troceadas para ensalada con pimiento natural"),
    (ACEIT, "Goya", "Frasco Para Ensalada con Pimiento 358 gr", 205, "Aceitunas manzanilla troceadas para ensalada con pimiento natural"),
    (ACEIT, "Goya", "Frasco Para Ensalada con Pimiento 567 gr", 465, "Aceitunas manzanilla troceadas para ensalada con pimiento natural"),
    # ── Goya · gordales ──
    (ACEIT, "Goya", "Frasco Gordales Enteras con Hueso 9.05 Oz", 355, "Aceitunas gordales españolas enteras con hueso"),
    (ACEIT, "Goya", "Frasco Gordales Rellenas de Pimiento 6.5 Oz", 360, "Aceitunas gordales rellenas de pimiento"),
    (ACEIT, "Goya", "Frasco Gordales Rellenas de Pimiento 8 Oz", 219, "Aceitunas gordales rellenas de pimiento"),
    (ACEIT, "Goya", "Frasco Gordales Rellenas de Pimiento 9 Oz", 465, "Aceitunas gordales rellenas de pimiento"),
    # ── Goya · stuffed line (latas) ──
    (ACEIT, "Goya", "Lata Rellenas de Salmón Ahumado 5.25 Oz", 299, "Aceitunas españolas rellenas de salmón ahumado picado"),
    (ACEIT, "Goya", "Lata Rellenas de Anchoa Picada (unidad)", 299, "Aceitunas selectas españolas rellenas de anchoa picada"),
    (ACEIT, "Goya", "Lata Rellenas de Jamón Serrano 5.25 Oz", 269, "Aceitunas españolas rellenas de jamón serrano"),
    (ACEIT, "Goya", "Lata Rellenas de Queso Azul 5.25 Oz", 289, "Aceitunas españolas rellenas de queso azul en pasta"),
    (ACEIT, "Goya", "Lata Rellenas de Queso Manchego 5.25 Oz", 295, "Aceitunas españolas rellenas de queso manchego en pasta"),
    (ACEIT, "Goya", "Lata Rellenas de Chorizo Picante 5.25 Oz", 260, "Aceitunas españolas rellenas de chorizo picante"),
    # ── Goya · negras / rodajas / alcaparrado ──
    (ACEIT, "Goya", "Frasco Negras en Rodajas 344 gr", 185, "Aceitunas negras en rodajas"),
    (ACEIT, "Goya", "Frasco Verdes en Rodajas 345 gr", 200, "Aceitunas manzanilla verdes en rodajas"),
    (ACEIT, "Goya", "Frasco Alcaparrado con Aceitunas 12.6 Oz", 200, "Alcaparrado: mezcla de aceitunas manzanilla, alcaparras y pimientos"),
    (ACEIT, "Goya", "Frasco Alcaparrado Bajo en Sal 198 gr", 165, "Alcaparrado bajo en sal: aceitunas, alcaparras y pimientos"),
    # ── Borges ──
    (ACEIT, "Borges", "Frasco Negras en Rodajas 230 gr", 125, "Aceitunas negras laminadas (black sliced)"),
    (ACEIT, "Borges", "Frasco Negras Enteras Deshuesadas 230 gr", 125, "Aceitunas negras enteras deshuesadas (black pitted)"),
    (ACEIT, "Borges", "Frasco Verdes Rellenas de Pimiento 230 gr", 128, "Aceitunas verdes rellenas de pimiento, sal reducida"),
    (ACEIT, "Borges", "Bolsa Gourmet Charm 350 gr", 215, "Mix de aceitunas gourmet, doypack"),
    # ── Maestranza ──
    (ACEIT, "Maestranza", "Bolsa Manzanilla Sin Hueso 180 gr", 99, "Aceitunas manzanilla verdes sin hueso, bolsa"),
    (ACEIT, "Maestranza", "Bolsa Manzanilla Pimienta 180 gr", 98, "Aceitunas manzanilla sin hueso a la pimienta, bolsa"),
    (ACEIT, "Maestranza", "Lata Rellenas de Salmón Ahumado 300 gr", 170, "Aceitunas rellenas de salmón ahumado, gran tamaño"),
    (ACEIT, "Maestranza", "Lata Rellenas de Pimiento Picante 300 gr", 175, "Aceitunas rellenas de pasta de pimiento picante"),
    (ACEIT, "Maestranza", "Lata Rellenas de Atún y Pimiento 300 gr", 175, "Aceitunas rellenas de atún y pimiento"),
    (ACEIT, "Maestranza", "Lata Rellenas de Limón 300 gr", 210, "Aceitunas rellenas de limón"),
    (ACEIT, "Maestranza", "Lata Negras Sin Hueso 350 gr", 200, "Aceitunas negras deshuesadas (pitted black)"),
    (ACEIT, "Maestranza", "Frasco Gordales Rellenas de Pimiento 340 gr", 285, "Aceitunas gordales rellenas de pasta de pimiento"),
    (ACEIT, "Maestranza", "Frasco Gordales Con Hueso 340 gr", 240, "Aceitunas gordales enteras con hueso (queen olives)"),
    (ACEIT, "Maestranza", "Frasco Gordales Aliñadas Sin Hueso 320 gr", 250, "Aceitunas gordales aliñadas sin hueso"),
    # ── La Explanada (gourmet) ──
    (ACEIT, "La Explanada", "Lata Rellenas de Anchoa 350 Ml", 220, "Aceitunas rellenas de anchoa, línea gourmet"),
    (ACEIT, "La Explanada", "Lata Rellenas de Queso Azul 350 Ml", 220, "Aceitunas rellenas de queso azul, línea gourmet"),
    (ACEIT, "La Explanada", "Lata Rellenas de Jalapeño 350 Ml", 229, "Aceitunas rellenas de jalapeño, línea gourmet"),
    (ACEIT, "La Explanada", "Lata Rellenas de Trufa 350 Ml", 260, "Aceitunas rellenas de trufa, línea gourmet"),
    # ── Valenciana ──
    (ACEIT, "Valenciana", "Frasco 14 Oz", 115, "Aceitunas verdes, frasco 397 gr"),
    # ── El Corte Inglés ──
    (ACEIT, "El Corte Inglés", "Frasco Manzanilla 500 gr", 1000, "Aceitunas manzanilla de Sevilla, frasco grande"),
    (ACEIT, "El Corte Inglés", "Lata Negras Deshuesadas 150 gr", 155, "Aceitunas negras deshuesadas extra"),
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
        if len(row) != 5:
            print(f"FATAL: fila con {len(row)} campos (esperados 5): {row[:3]}")
            sys.exit(1)
        (food, brand, pres, *_rest) = row
        key = (food.lower(), (brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    foods = {r[0] for r in ROWS}
    brands = {r[1] for r in ROWS if r[1]}
    print(f"Seed aceituna: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, NOTES, CAT, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
