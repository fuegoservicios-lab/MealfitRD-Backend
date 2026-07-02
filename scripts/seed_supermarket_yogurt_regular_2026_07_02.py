"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de YOGURT REGULAR.

Trigesimoprimera familia con variantes de MARCA del Supermercado RD: 133 SKUs
transcritos del catálogo de La Sirena (capturas del owner, 2026-07-02), repartidos
en 5 foods:

  * "Yogurt Regular" (114): Yoplait (batidos/Top/bebibles/Yopsi/deslactosado),
    Yopli y Yoki (líneas infantiles), Yoka (bebibles 8 Oz/32 Oz/medio galón/galón,
    vasos con fruta, Slender, Light, sin lactosa), Wala, Rica (3 tamaños × 3
    sabores), Deliciel (frascos K, bebibles, Kids), Del Artesano, La Yogurt,
    Activia, Danimals, Asturiana y Elle & Vire.
  * "Yogurt Griego" (+15): Chobani (core/Zero Sugar/Flip/20G Protein) y Oikos
    Triple Zero (Dannon) — son griegos, van a su food aunque salieron en el
    search de yogurt regular.
  * "Kefir" (food NUEVO, 2): Lifeway 32 Oz vainilla + 8 Oz fresa (leche cultivada,
    producto distinto del yogurt).
  * "Yogurt de cabra" (food NUEVO, 1): K by Deliciel cabra natural sin azúcar
    (leche de cabra ≠ vaca, mismo criterio que Queso de oveja).
  * "Galleta de soda" (BONUS, 1): Hatuey Regular 20 unid — se coló en el search
    pero es SKU legítimo del food ya existente.

Notas de curaduría:
  * El genérico del PDF calza exacto: Pote 1960gr RD$220 = Rica bebible ½ galón
    (los 3 sabores, a precio de LISTA; sus promos -10% se ignoran).
  * Promos → precio de lista: todos los Rica (8 Oz RD$44, 32 Oz RD$143, ½ galón
    RD$220) y Hatuey (RD$169, promo RD$152).
  * Dedupe: el vaso Yoka con fruta Fresa 6 Oz aparece DOS veces ("Prebiótico" y
    "Probiótico", mismo precio RD$50) — se carga UNA vez.
  * Conflicto de precio: Yoka Natural galón listado a RD$429 y a RD$365 — se
    carga RD$365 (consistente con fresa/vainilla/ciruela galón RD$365).
  * Yopli aparece como "150 Gr" y "5 Oz" (mismo envase) — normalizado a
    "5 Oz (150 gr)".

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_yogurt_regular_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_yogurt_regular_2026_07_02.py --commit   # inserta
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
PAN = "Panadería y harinas"

REG = "Yogurt Regular"
GRIEGO = "Yogurt Griego"
KEFIR = "Kefir"
CABRA = "Yogurt de cabra"
GALLETA = "Galleta de soda"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Yoplait: batidos ──
    (REG, "Yoplait", "Vaso Batido Natural 170 gr", 49, "Yogurt batido semidescremado natural", LACTEO),
    (REG, "Yoplait", "Vaso Batido Fresas 170 gr", 49, "Yogurt batido semidescremado con fresas", LACTEO),
    (REG, "Yoplait", "Vaso Batido Frutas Rojas 170 gr", 49, "Yogurt batido semidescremado con frutas rojas", LACTEO),
    (REG, "Yoplait", "Vaso Batido Frutas Tropicales 170 gr", 49, "Yogurt batido semidescremado con frutas tropicales", LACTEO),
    (REG, "Yoplait", "Vaso Batido Cereales y Ciruelas Pasas 170 gr", 49, "Yogurt batido semidescremado con cereales y ciruelas pasas", LACTEO),
    (REG, "Yoplait", "Vaso Batido Chinola 170 gr", 49, "Yogurt batido semidescremado con chinola", LACTEO),
    (REG, "Yoplait", "Vaso Batido Top Natural 172 gr", 90, "Yogurt batido natural con topping de granola, almendras y chocolate", LACTEO),
    (REG, "Yoplait", "Vaso Batido Top Fresa 172 gr", 90, "Yogurt batido de fresa con topping de granola, almendras y chocolate", LACTEO),
    (REG, "Yoplait", "Vaso Batido Top Vainilla 172 gr", 90, "Yogurt batido de vainilla con topping de granola y almendras", LACTEO),
    # ── Yoplait: bebibles y Yopsi ──
    (REG, "Yoplait", "Bebible Natural 250 gr", 45, "Yogurt bebible natural", LACTEO),
    (REG, "Yoplait", "Bebible Piña Colada 250 Ml", 45, "Yogurt bebible sabor piña colada", LACTEO),
    (REG, "Yoplait", "Botella Deslactosado Ciruela 1/2 Gl", 235, "Yogurt bebible deslactosado con ciruela pasa", LACTEO),
    (REG, "Yoplait", "Botella Yopsi Natural 1/2 Gl", 235, "Yogurt bebible Yopsi natural", LACTEO),
    (REG, "Yoplait", "Botella Yopsi Natural Sin Azúcar 1/2 Gl", 235, "Yogurt bebible Yopsi natural, 0% azúcar", LACTEO),
    (REG, "Yoplait", "Botella Yopsi Natural Sin Azúcar 1 Gl", 365, "Yogurt bebible Yopsi natural, 0% azúcar, galón", LACTEO),
    (REG, "Yoplait", "Botella Yopsi 0% Azúcar 1 Lt", 159, "Yogurt bebible Yopsi sin azúcar, litro", LACTEO),
    (REG, "Yoplait", "Botella Yopsi Fresa 1/2 Gl", 235, "Yogurt bebible Yopsi de fresa", LACTEO),
    (REG, "Yoplait", "Botella Yopsi Fresa 1 Gl", 365, "Yogurt bebible Yopsi de fresa, galón", LACTEO),
    (REG, "Yoplait", "Botella Yopsi Fresa 1 Lt (33.8 Oz)", 159, "Yogurt bebible Yopsi de fresa, litro", LACTEO),
    (REG, "Yoplait", "Botella Yopsi Guanábana 1 Lt (33.8 Oz)", 159, "Yogurt bebible Yopsi de guanábana, litro", LACTEO),
    (REG, "Yoplait", "Botella Yopsi Piña 1/2 Gl", 235, "Yogurt bebible Yopsi de piña", LACTEO),
    (REG, "Yoplait", "Botella Yopsi Piña 1 Gl", 365, "Yogurt bebible Yopsi de piña, galón", LACTEO),
    (REG, "Yoplait", "Botella Yopsi Vainilla 1/2 Gl", 235, "Yogurt bebible Yopsi de vainilla", LACTEO),
    (REG, "Yoplait", "Botella Yopsi Vainilla 1 Gl", 365, "Yogurt bebible Yopsi de vainilla, galón", LACTEO),
    # ── Yopli (línea infantil Yoplait) ──
    (REG, "Yopli", "Bebible Fresa 5 Oz (150 gr)", 42, "Yogurt bebible infantil de fresa, fuente de calcio", LACTEO),
    (REG, "Yopli", "Bebible Vainilla 5 Oz (150 gr)", 42, "Yogurt bebible infantil de vainilla", LACTEO),
    (REG, "Yopli", "Bebible Fresa y Guineo 5 Oz (150 gr)", 42, "Yogurt bebible infantil de fresa y guineo", LACTEO),
    (REG, "Yopli", "Bebible Cookies and Cream 5 Oz (150 gr)", 42, "Yogurt bebible infantil sabor cookies and cream", LACTEO),
    (REG, "Yopli", "Bebible Naranja y Zanahoria 5 Oz (150 gr)", 42, "Yogurt bebible infantil de naranja y zanahoria", LACTEO),
    (REG, "Yopli", "Bebible Uva 5 Oz (150 gr)", 42, "Yogurt bebible infantil de uva, con DHA", LACTEO),
    (REG, "Yopli", "Bebible Manzana 5 Oz (150 gr)", 42, "Yogurt bebible infantil de manzana, con DHA", LACTEO),
    (REG, "Yopli", "Bebible Deslactosado Fresa 5 Oz (150 gr)", 42, "Yogurt bebible infantil deslactosado de fresa", LACTEO),
    # ── Yoki ──
    (REG, "Yoki", "Bebible Fresa 5 Oz", 42, "Yogurt bebible infantil de fresa", LACTEO),
    (REG, "Yoki", "Bebible Uva 5 Oz", 42, "Yogurt bebible infantil de uva", LACTEO),
    (REG, "Yoki", "Bebible Natural 5 Oz", 42, "Yogurt bebible infantil natural", LACTEO),
    (REG, "Yoki", "Bebible Piña 5 Oz", 42, "Yogurt bebible infantil de piña", LACTEO),
    # ── Yoka: bebibles 8 Oz ──
    (REG, "Yoka", "Bebible Natural 8 Oz", 47, "Yogurt bebible natural, con azúcar añadida", LACTEO),
    (REG, "Yoka", "Bebible Vainilla 0% 8 Oz", 47, "Yogurt bebible de vainilla, 0% grasa", LACTEO),
    (REG, "Yoka", "Bebible Fresa 8 Oz", 47, "Yogurt bebible de fresa", LACTEO),
    (REG, "Yoka", "Bebible Fresa y Guineo 8 Oz", 47, "Yogurt bebible de fresa y guineo", LACTEO),
    (REG, "Yoka", "Bebible Ciruela Pasa 8 Oz", 47, "Yogurt bebible de ciruela pasa", LACTEO),
    (REG, "Yoka", "Bebible Chinola 0% 8 Oz", 47, "Yogurt bebible de chinola, 0% grasa", LACTEO),
    (REG, "Yoka", "Bebible Piña Colada 8 Oz", 47, "Yogurt bebible sabor piña colada", LACTEO),
    (REG, "Yoka", "Bebible Mango 8 Oz", 47, "Yogurt bebible de mango", LACTEO),
    (REG, "Yoka", "Bebible Sin Lactosa Natural 8 Oz", 57, "Yogurt bebible natural sin lactosa, fácil digestión", LACTEO),
    (REG, "Yoka", "Bebible Slender 0% Fresa 8 Oz", 59, "Yogurt bebible Slender de fresa, 0% grasa sin azúcar añadida", LACTEO),
    # ── Yoka: bebibles 32 Oz ──
    (REG, "Yoka", "Bebible Natural 32 Oz", 150, "Yogurt bebible natural", LACTEO),
    (REG, "Yoka", "Bebible Vainilla 32 Oz", 150, "Yogurt bebible de vainilla", LACTEO),
    (REG, "Yoka", "Bebible Fresa 32 Oz", 150, "Yogurt bebible de fresa", LACTEO),
    (REG, "Yoka", "Bebible Fresa y Guineo 32 Oz", 150, "Yogurt bebible de fresa y guineo", LACTEO),
    (REG, "Yoka", "Bebible Sin Lactosa Natural 32 Oz", 169, "Yogurt bebible natural sin lactosa, fácil digestión", LACTEO),
    # ── Yoka: bebibles 1/2 galón ──
    (REG, "Yoka", "Botella Natural Bebible 1/2 Gl", 239, "Yogurt bebible natural, medio galón", LACTEO),
    (REG, "Yoka", "Botella Fresa Bebible 1/2 Gl", 239, "Yogurt bebible de fresa, medio galón", LACTEO),
    (REG, "Yoka", "Botella Vainilla Bebible 1/2 Gl", 239, "Yogurt bebible de vainilla, medio galón", LACTEO),
    (REG, "Yoka", "Botella Piña Colada Bebible 1/2 Gl", 239, "Yogurt bebible piña colada, medio galón", LACTEO),
    (REG, "Yoka", "Botella Ciruela Pasa Bebible 1/2 Gl", 239, "Yogurt bebible de ciruela pasa, medio galón", LACTEO),
    (REG, "Yoka", "Botella Slender Natural 0% Sin Azúcar 1/2 Gl", 255, "Yogurt bebible Slender natural, 0% grasa sin azúcar, medio galón", LACTEO),
    # ── Yoka: galones ──
    (REG, "Yoka", "Galón Natural", 365, "Yogurt bebible natural, galón", LACTEO),
    (REG, "Yoka", "Galón Fresa", 365, "Yogurt bebible de fresa, galón", LACTEO),
    (REG, "Yoka", "Galón Vainilla", 365, "Yogurt bebible de vainilla, galón", LACTEO),
    (REG, "Yoka", "Galón Ciruela Pasa (3.4 Kg)", 365, "Yogurt bebible de ciruela pasa, galón", LACTEO),
    # ── Yoka: vasos con fruta 6 Oz ──
    (REG, "Yoka", "Vaso con Fruta Fresa 6 Oz", 50, "Yogurt con fruta fresa, con probióticos", LACTEO),
    (REG, "Yoka", "Vaso con Fruta Chinola 6 Oz", 50, "Yogurt con fruta chinola", LACTEO),
    (REG, "Yoka", "Vaso con Fruta Coco 6 Oz", 50, "Yogurt con fruta coco", LACTEO),
    (REG, "Yoka", "Vaso Prebiótico Tutti Frutti 6 Oz", 50, "Yogurt con fruta tutti frutti, prebiótico, 0% grasa", LACTEO),
    (REG, "Yoka", "Vaso Probiótico Natural 6 Oz", 50, "Yogurt natural con azúcar añadida, probiótico", LACTEO),
    (REG, "Yoka", "Vaso Light Mango 6 Oz", 50, "Yogurt light de mango, 0% grasa", LACTEO),
    (REG, "Yoka", "Vaso Light Vainilla 6 Oz", 50, "Yogurt light de vainilla, 0% grasa", LACTEO),
    # ── Yoka: Slender / Light tarros ──
    (REG, "Yoka", "Tarro Slender Natural 32 Oz (750 gr)", 169, "Yogurt Slender natural, 0% grasa sin azúcar añadida", LACTEO),
    (REG, "Yoka", "Vaso Slender Super Dietético Fresa 6 Oz", 55, "Yogurt Slender super dietético de fresa", LACTEO),
    (REG, "Yoka", "Vaso Slender Super Dietético Natural 6 Oz", 55, "Yogurt Slender super dietético natural", LACTEO),
    (REG, "Yoka", "Tarro Light Vainilla 32 Oz", 159, "Yogurt light de vainilla, 0% grasa", LACTEO),
    (REG, "Yoka", "Tarro Light Fresa 32 Oz", 159, "Yogurt light de fresa, 0% grasa", LACTEO),
    # ── Wala ──
    (REG, "Wala", "Bebible Vainilla 8 Oz", 39, "Yogurt bebible de vainilla", LACTEO),
    (REG, "Wala", "Bebible Fresa 8 Oz", 39, "Yogurt bebible de fresa", LACTEO),
    (REG, "Wala", "Bebible Natural 8 Oz", 39, "Yogurt bebible natural", LACTEO),
    (REG, "Wala", "Bebible Vainilla 32 Oz", 125, "Yogurt bebible de vainilla", LACTEO),
    (REG, "Wala", "Bebible Natural 32 Oz", 125, "Yogurt bebible natural", LACTEO),
    # ── Rica (precios de lista; promos -10% ignoradas) ──
    (REG, "Rica", "Bebible Fresa 8 Oz (250 gr)", 44, "Yogurt bebible de fresa, 0% grasa, 30% menos azúcar", LACTEO),
    (REG, "Rica", "Bebible Natural 8 Oz (250 gr)", 44, "Yogurt bebible natural, 0% grasa, 30% menos azúcar", LACTEO),
    (REG, "Rica", "Bebible Vainilla 8 Oz (250 gr)", 44, "Yogurt bebible de vainilla, 0% grasa, 30% menos azúcar", LACTEO),
    (REG, "Rica", "Bebible Fresa 32 Oz (980 gr)", 143, "Yogurt bebible de fresa, 0% grasa", LACTEO),
    (REG, "Rica", "Bebible Natural 32 Oz (980 gr)", 143, "Yogurt bebible natural, 0% grasa", LACTEO),
    (REG, "Rica", "Bebible Vainilla Rigurt 32 Oz", 143, "Yogurt bebible Rigurt de vainilla, 0% grasa", LACTEO),
    (REG, "Rica", "Bebible Fresa 1/2 Galón (1960 gr)", 220, "Yogurt bebible de fresa, medio galón", LACTEO),
    (REG, "Rica", "Bebible Natural 1/2 Galón (1960 gr)", 220, "Yogurt bebible natural, medio galón", LACTEO),
    (REG, "Rica", "Bebible Vainilla Rigurt 1/2 Galón (1960 gr)", 220, "Yogurt bebible Rigurt de vainilla, medio galón", LACTEO),
    # ── Deliciel (K by Deliciel) ──
    (REG, "Deliciel", "Frasco Semidescremado Natural 4 Oz", 80, "Yogurt natural semidescremado 2.5% grasa sin azúcar, frasco de vidrio K by Deliciel", LACTEO),
    (REG, "Deliciel", "Frasco Semidescremado Vainilla 4 Oz", 80, "Yogurt de vainilla semidescremado, frasco de vidrio K by Deliciel", LACTEO),
    (REG, "Deliciel", "Frasco Fresa 4 Oz", 80, "Yogurt de fresa, frasco de vidrio K by Deliciel", LACTEO),
    (REG, "Deliciel", "Bebible Vainilla 7 Oz", 80, "Yogurt bebible probiótico de vainilla de Madagascar, sin lactosa", LACTEO),
    (REG, "Deliciel", "Bebible Fresa 7 Oz", 80, "Yogurt bebible probiótico de fresa, con frutas naturales", LACTEO),
    (REG, "Deliciel", "Bebible Guayaba 7 Oz", 80, "Yogurt bebible probiótico de guayaba, con frutas naturales", LACTEO),
    (REG, "Deliciel", "Bebible Kids Vainilla 4 Oz", 54, "Yogurt bebible infantil de vainilla, línea Kids", LACTEO),
    (REG, "Deliciel", "Bebible Kids Fresa 4 Oz", 54, "Yogurt bebible infantil de fresa, línea Kids", LACTEO),
    (REG, "Deliciel", "Bebible Kids Natural 4 Oz", 54, "Yogurt bebible infantil natural sin azúcar, línea Kids", LACTEO),
    # ── Del Artesano ──
    (REG, "Del Artesano", "Botella Natural con Azúcar 32 Oz", 168, "Yogurt natural artesanal con azúcar", LACTEO),
    (REG, "Del Artesano", "Botella Natural Sin Azúcar 32 Oz", 168, "Yogurt natural artesanal sin azúcar", LACTEO),
    (REG, "Del Artesano", "Botella Ciruela 32 Oz", 168, "Yogurt artesanal sabor ciruela", LACTEO),
    # ── La Yogurt ──
    (REG, "La Yogurt", "Vaso Probiotic Mixed Berry 6 Oz", 70, "Yogurt probiótico low fat de mixed berry", LACTEO),
    (REG, "La Yogurt", "Vaso Probiotic Raspberry 6 Oz", 70, "Yogurt probiótico low fat de frambuesa", LACTEO),
    (REG, "La Yogurt", "Vaso Probiotic Strawberry 6 Oz", 70, "Yogurt probiótico low fat de fresa", LACTEO),
    (REG, "La Yogurt", "Vaso Probiotic Strawberry Banana 6 Oz", 70, "Yogurt probiótico low fat de fresa y guineo", LACTEO),
    (REG, "La Yogurt", "Vaso Probiotic Strawberry Light 6 Oz", 70, "Yogurt probiótico light de fresa (Lite & Sensible)", LACTEO),
    (REG, "La Yogurt", "Vaso Probiotic Piña Colada 6 Oz", 70, "Yogurt probiótico low fat sabor piña colada", LACTEO),
    (REG, "La Yogurt", "Vaso Probiotic Vainilla 6 Oz", 70, "Yogurt probiótico low fat de vainilla", LACTEO),
    # ── Activia / Danimals (Dannon) ──
    (REG, "Activia", "Pack 4 Vasos Vainilla", 368, "Pack de 4 yogurts Activia de vainilla, con probióticos (Dannon)", LACTEO),
    (REG, "Activia", "Pack 4 Vasos Ciruela", 368, "Pack de 4 yogurts Activia de ciruela, con probióticos (Dannon)", LACTEO),
    (REG, "Activia", "Pack 4 Vasos Fresa", 368, "Pack de 4 yogurts Activia de fresa, con probióticos (Dannon)", LACTEO),
    (REG, "Danimals", "Six Pack Smoothie Fresa 3.1 Oz", 428, "Smoothies de yogurt para niños, fresa (Dannon)", LACTEO),
    # ── Asturiana / Elle & Vire ──
    (REG, "Asturiana", "Vaso Mango 125 gr", 39, "Yogurt sabor mango (Central Lechera Asturiana, España)", LACTEO),
    (REG, "Asturiana", "Vaso Multifruta 125 gr", 39, "Yogurt sabor multifruta (España)", LACTEO),
    (REG, "Asturiana", "Vaso Fresa 125 gr", 39, "Yogurt sabor fresa (España)", LACTEO),
    (REG, "Elle & Vire", "Vaso Fruits Fresa 125 gr", 74, "Yogurt con fresas (Francia)", LACTEO),
    # ── Chobani / Oikos → Yogurt Griego ──
    (GRIEGO, "Chobani", "Vaso Non Fat Plain 5.3 Oz", 185, "Greek yogurt natural sin grasa, 14g de proteína", LACTEO),
    (GRIEGO, "Chobani", "Vaso Vanilla 5.3 Oz", 185, "Greek yogurt de vainilla", LACTEO),
    (GRIEGO, "Chobani", "Vaso Mixed Berry Blended 5.3 Oz", 185, "Greek yogurt blended de mixed berry", LACTEO),
    (GRIEGO, "Chobani", "Vaso Strawberry 5.3 Oz", 185, "Greek yogurt con fresa en el fondo (on the bottom)", LACTEO),
    (GRIEGO, "Chobani", "Vaso Mango 5.3 Oz", 185, "Greek yogurt con mango en el fondo (on the bottom)", LACTEO),
    (GRIEGO, "Chobani", "Vaso Coconut Blended 5.3 Oz", 185, "Greek yogurt blended de coco", LACTEO),
    (GRIEGO, "Chobani", "Vaso Zero Sugar Vainilla 5.3 Oz", 205, "Yogurt estilo griego cero azúcar, vainilla, sin lactosa", LACTEO),
    (GRIEGO, "Chobani", "Vaso Zero Sugar Strawberry Cheesecake 5.3 Oz", 205, "Yogurt estilo griego cero azúcar, strawberry cheesecake", LACTEO),
    (GRIEGO, "Chobani", "Vaso Flip Cookie & Cream 5.3 Oz", 200, "Greek yogurt Flip con cookies de chocolate", LACTEO),
    (GRIEGO, "Chobani", "Vaso Flip Key Lime Crumble 5.3 Oz", 200, "Greek yogurt Flip key lime con graham y chocolate blanco", LACTEO),
    (GRIEGO, "Chobani", "Vaso Flip S'more S'mores 5.3 Oz", 200, "Greek yogurt Flip s'mores con graham y chocolate", LACTEO),
    (GRIEGO, "Chobani", "Vaso Flip Almond Coco Loco 5.3 Oz", 200, "Greek yogurt Flip con almendras tostadas y coco", LACTEO),
    (GRIEGO, "Chobani", "Vaso 20G Protein Vainilla 6.7 Oz", 265, "Greek yogurt alto en proteína (20g), vainilla, cero azúcar añadida", LACTEO),
    (GRIEGO, "Chobani", "Vaso 20G Protein Cherry Berry 6.7 Oz", 265, "Greek yogurt alto en proteína (20g), cherry berry, cero azúcar añadida", LACTEO),
    (GRIEGO, "Oikos", "Vaso Triple Zero Strawberry 5.3 Oz", 209, "Greek yogurt Triple Zero de fresa: 0 azúcar añadida, 0 endulzantes artificiales, 0 grasa, 15g proteína (Dannon)", LACTEO),
    # ── Kefir (food nuevo) ──
    (KEFIR, "Lifeway", "Botella Low Fat Vainilla 32 Oz", 395, "Kefir de leche cultivada low fat, vainilla de Madagascar, con probióticos", LACTEO),
    (KEFIR, "Lifeway", "Botella Low Fat Fresa 8 Oz", 125, "Kefir de leche cultivada low fat, fresa, con probióticos", LACTEO),
    # ── Yogurt de cabra (food nuevo) ──
    (CABRA, "Deliciel", "Frasco Cabra Natural Sin Azúcar 4 Oz", 110, "Yogurt natural de leche de cabra sin azúcar, 3% grasa, frasco de vidrio K by Deliciel", LACTEO),
    # ── BONUS del search: galleta de soda ──
    (GALLETA, "Hatuey", "Caja Regular 20 unid", 169, "Galletas de soda regulares, 20 paquetes individuales, 0% colesterol", PAN),
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
    for (food, brand, pres, *_rest) in ROWS:
        key = (food.lower(), (brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    foods = {r[0] for r in ROWS}
    brands = {r[1] for r in ROWS}
    print(f"Seed yogurt regular: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
