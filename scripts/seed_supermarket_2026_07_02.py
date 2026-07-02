"""[P1-SUPERMARKET-DB · 2026-07-02] Seed inicial del Supermercado RD artificial.

Inserta en `supermarket_products` (Neon) las ~233 presentaciones del dataset de
alimentos verificados del owner (PDF "Dataset de alimentos verificados",
2026-07-01): alimento + presentación + porción + duración + precio RD$ + notas.
Todas las filas del seed son GENÉRICAS (`brand=NULL`); las variantes de marca se
agregan después desde la admin UI del landing (/supermercado).

Transcripción fiel al PDF con dos normalizaciones:
  * Presentación "L" a secas (produce/carnes por libra) → "Lb" (los lotes
    posteriores del mismo PDF ya usan "Lb"). Strings compuestos ("Paquete 2L",
    "Cartón L", "1.47 L") se preservan VERBATIM — se curan desde la admin UI.
  * Notas "Relativa" → "Relativo".

`master_food_name` arranca = `food_name` (best-guess del link suave al catálogo
`master_ingredients`); la futura integración con la lista de compras hará el
match case-insensitive/alias y la admin UI permite curarlo.

Idempotente: ON CONFLICT (unique index de variante) DO NOTHING — re-ejecutar NO
pisa ediciones hechas desde la admin UI.

USO:
  python scripts/seed_supermarket_2026_07_02.py            # DRY-RUN (no escribe)
  python scripts/seed_supermarket_2026_07_02.py --commit   # inserta
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

# Notas canónicas del dataset
EXC7 = "Uso exclusivo plan de 7 días"
R715 = "Rinde para planes de 7 y 15 días"
TODOS = "Rinde para todos los planes"
REL = "Relativo"

# Categorías (para navegación del landing; editables después)
COND = "Condimentos y especias"
SALSA = "Salsas y aderezos"
ACEITE = "Aceites y grasas"
GRANO = "Granos y cereales"
PAN = "Panadería y harinas"
LEGUM = "Legumbres y proteína vegetal"
CARNE = "Carnes, pescados y mariscos"
LACTEO = "Lácteos y huevos"
BEBVEG = "Bebidas y alternativas vegetales"
FRUTA = "Frutas"
VEG = "Vegetales y verduras"
VIVER = "Víveres y tubérculos"
SEMILLA = "Semillas y frutos secos"
OTRO = "Otros"

# (food_name, presentation, portion_label, duration_label, price_rd, notes, category)
ROWS = [
    # ── Página 1 ──
    ("Pimienta negra", "Sobre", "Mínima", "7 días", 59, EXC7, COND),
    ("Pimienta negra", "Frasco", "Mediana", "30 días", 89, TODOS, COND),
    ("Canela en polvo", "Sobre 0.5 Oz", "Mínima", "7 días", 55, EXC7, COND),
    ("Canela en polvo", "Pote 2 Oz", "Mediana", "30 días", 105, TODOS, COND),
    ("Sal", "Paquete 1L", "Mínima", "15 días", 17, R715, COND),
    ("Sal", "Paquete 2L", "Mayor", "30 días", 40, TODOS, COND),
    ("Mantequilla de maní", "Pote 16 Oz", "Mínima", "30 días", 117, TODOS, SEMILLA),
    ("Miel", "Pote 8 Oz", "Mínima", "30 días", 114, TODOS, OTRO),
    ("Avena", "Paquete 600gr", "Mínima", "7 días", 49, EXC7, GRANO),
    ("Avena", "Paquete 1200gr", "Mediana", "30 días", 189, TODOS, GRANO),
    ("Casabe", "Paquete 10 Oz", "Mínima", "7 días", 94, EXC7, PAN),
    ("Casabe", "Paquete 40 Oz", "Mayor", "30 días", 199, TODOS, PAN),
    ("Arroz integral", "Paquete 2L", "Única", "7 días", 145, EXC7, GRANO),
    ("Arroz blanco", "Paquete 2L", "Mínima", "7 días", 165, EXC7, GRANO),
    ("Arroz blanco", "Paquete 5L", "Mediana", "15 días", 235, R715, GRANO),
    ("Arroz blanco", "Paquete 10L", "Mayor", "30 días", 327, TODOS, GRANO),
    # ── Página 2 ──
    ("Habichuelas rojas", "Lata 15 Oz", "Mínima", "Relativo", 50, REL, LEGUM),
    ("Habichuelas rojas", "Paquete 800gr", "Mayor", "30 días", 129, TODOS, LEGUM),
    ("Habichuelas blancas", "Lata 15 Oz", "Mínima", "Relativo", 50, REL, LEGUM),
    ("Habichuelas blancas", "Paquete 800gr", "Mayor", "30 días", 115, TODOS, LEGUM),
    ("Orégano", "Sobre 45 gr", "Mínima", "7 días", 39, EXC7, COND),
    ("Orégano", "Pote 3.2 Oz", "Mayor", "30 días", 81, TODOS, COND),
    ("Vinagre de manzana", "Pote 16 Oz", "Mínima", "30 días", 51, TODOS, SALSA),
    ("Aceite de oliva", "Pote 250 ML", "Mínima", "30 días", 195, TODOS, ACEITE),
    ("Huevos", "Cartón 20 unid", "Mínima", "7 días", 200, EXC7, LACTEO),
    ("Huevos", "Cartón 30 unid", "Mayor", "15 días", 295, R715, LACTEO),
    ("Atún en agua", "Lata 170 gr", "Única", "Relativo", 60, REL, CARNE),
    ("Pan integral familiar", "Paquete", "Mayor", "7 días", 155, EXC7, PAN),
    ("Pan integral personal", "Paquete", "Mínima", "Relativo", 140, REL, PAN),
    ("Pan blanco familiar", "Paquete", "Mínima", "7 días", 140, EXC7, PAN),
    ("Pan blanco personal", "Paquete", "Mínima", "Relativo", 109, REL, PAN),
    ("Tortilla integral", "Paquete 6 unid", "Única", "7 días", 195, EXC7, PAN),
    ("Tortilla de Trigo", "Paquete 10 unid", "Mayor", "7 días", 74, EXC7, PAN),
    ("Perejil", "Paquete", "Única", "Relativo", 44, REL, VEG),
    ("Queso ricotta", "Tarro", "Única", "Relativo", 245, REL, LACTEO),
    ("Semillas de chía", "Paquete", "Única", "30 días", 380, TODOS, SEMILLA),
    # ── Página 3 ──
    ("Piña", "Unidad", "Única", "Relativo", 94, REL, FRUTA),
    ("Lenteja", "Lata 15.5 Oz", "Mínima", "7 días", 95, EXC7, LEGUM),
    ("Lenteja", "Paquete 800gr", "Mayor", "30 días", 114, TODOS, LEGUM),
    ("Mango", "Unidad", "Única", "Relativo", 34, REL, FRUTA),
    ("Habichuela negra", "Lata 15 Oz", "Mínima", "Relativo", 88, REL, LEGUM),
    ("Habichuela negra", "Paquete 800gr", "Mayor", "30 días", 105, TODOS, LEGUM),
    ("Vinagre Blanco", "Pote 16 Oz", "Mínima", "30 días", 25, TODOS, SALSA),
    ("Lechosa", "Lb", "Única", "Relativo", 20, REL, FRUTA),
    ("Fresa", "Paquete 1 L", "Mínima", "7 días", 165, EXC7, FRUTA),
    ("Fresa", "Paquete 1.5L", "Mediana", "15 días", 299, R715, FRUTA),
    ("Fresa", "Paquete 4L", "Mayor", "30 días", 699, TODOS, FRUTA),
    ("Aguacate", "Unidad", "Única", "Relativo", 59, REL, FRUTA),
    ("Limón", "Lb", "Única", "Relativo", 62, REL, FRUTA),
    ("Naranja", "Lb", "Única", "Relativo", 89, REL, FRUTA),
    ("Guineo verde", "Lb", "Única", "Relativo", 17, REL, FRUTA),
    ("Guineo maduro", "Lb", "Única", "Relativo", 20, REL, FRUTA),
    ("Queso mozzarella", "Lb", "Única", "Relativo", 214, REL, LACTEO),
    ("Yogurt Griego", "Pote 150gr", "Mínima", "Relativo", 100, REL, LACTEO),
    ("Yogurt Regular", "Pote 1960gr", "Mediana", "15 días", 220, R715, LACTEO),
    ("Filete pechuga de pollo", "Lb", "Única", "Relativo", 135, REL, CARNE),
    ("Camarón", "Paquete", "Única", "Relativo", 299, REL, CARNE),
    ("Lechuga", "Lb", "Única", "Relativo", 48, REL, VEG),
    ("Cilantro", "Paquete", "Única", "7 días", 48, EXC7, VEG),
    ("Tomate", "Lb", "Única", "Relativo", 47, REL, VEG),
    ("Berenjena", "Lb", "Única", "Relativo", 30, REL, VEG),
    # ── Página 4 ──
    ("Cebolla", "Lb", "Única", "Relativo", 47, REL, VEG),
    ("Brócoli", "Lb", "Única", "Relativo", 62, REL, VEG),
    ("Ajo", "Paquete 4 unid", "Mínima", "30 días", 60, TODOS, VEG),
    ("Ají morrón", "Lb", "Única", "Relativo", 83, REL, VEG),
    ("Batata", "Lb", "Única", "Relativo", 32, REL, VIVER),
    ("Yautía", "Lb", "Única", "Relativo", 78, REL, VIVER),
    ("Auyama", "Lb", "Única", "Relativo", 29, REL, VEG),
    ("Bacalao", "Paquete 16 Oz", "Mediano", "Relativo", 225, REL, CARNE),
    ("Bacalao", "Paquete 12 Oz", "Mínima", "Relativo", 170, REL, CARNE),
    ("Bacalao", "Lb", "Mayor", "Relativo", 255, REL, CARNE),
    ("Carne de Res molida", "Lb", "Única", "Relativo", 209, REL, CARNE),
    ("Cerdo", "Lb", "Única", "Relativo", 115, REL, CARNE),
    ("Filete de pescado blanco", "Paquete 32 Oz", "Única", "Relativo", 255, REL, CARNE),
    ("Jamón de pavo", "Lb", "Única", "Relativo", 255, REL, CARNE),
    ("Longaniza", "Lb", "Única", "Relativo", 136, REL, CARNE),
    ("Salami", "Lb", "Única", "Relativo", 132, REL, CARNE),
    ("Leche descremada", "Cartón L", "Única", "Relativo", 49, REL, LACTEO),
    ("Leche", "Cartón L", "Única", "Relativo", 55, REL, LACTEO),
    ("Leche evaporada", "Cartón 290 ml", "Única", "Relativo", 62, REL, LACTEO),
    ("Mantequilla", "Tarro L", "Mediana", "30 días", 315, TODOS, LACTEO),
    ("Mantequilla", "Barrita 113 gr", "Mínima", "15 días", 99, R715, LACTEO),
    ("Queso cottage", "Tarro 24 Oz", "Mínima", "30 días", 370, TODOS, LACTEO),
    ("Queso crema", "Lb", "Única", "Relativo", 289, REL, LACTEO),
    ("Queso parmesano", "5 Oz", "Única", "Relativo", 275, REL, LACTEO),
    ("Ají cubanela", "Lb", "Única", "Relativo", 68, REL, VEG),
    ("Coliflor", "Lb", "Única", "Relativo", 62, REL, VEG),
    # ── Página 5 ──
    ("Espinaca", "Paquete", "Mínima", "15 días", 42, R715, VEG),
    ("Espinaca", "Paquete 450 gr", "Mayor", "30 días", 140, TODOS, VEG),
    ("Jengibre", "Lb", "Única", "Relativo", 98, REL, VEG),
    ("Molondrones", "Lb", "Única", "Relativo", 54, REL, VEG),
    ("Pepino", "Lb", "Única", "Relativo", 24, REL, VEG),
    ("Repollo", "Unidad (mitad)", "Única", "Relativo", 56, REL, VEG),
    ("Tayota", "Lb", "Única", "Relativo", 24, REL, VEG),
    ("Vainitas", "Lb", "Única", "Relativo", 44, REL, VEG),
    ("Zanahoria", "Lb", "Única", "Relativo", 27, REL, VEG),
    ("Papa", "Lb", "Única", "Relativo", 34, REL, VIVER),
    ("Plátano maduro", "Unidad", "Única", "Relativo", 21, REL, VIVER),
    ("Plátano verde", "Unidad", "Única", "Relativo", 21, REL, VIVER),
    ("Yuca", "Lb", "Única", "Relativo", 39, REL, VIVER),
    ("Ñame", "Lb", "Única", "Relativo", 76, REL, VIVER),
    ("Chinola", "Lb", "Única", "Relativo", 79, REL, FRUTA),
    ("Manzana", "Lb", "Única", "Relativo", 78, REL, FRUTA),
    ("Melón", "Unidad", "Única", "Relativo", 72, REL, FRUTA),
    ("Sandia", "Lb", "Única", "Relativo", 30, REL, FRUTA),
    ("Aceite vegetal", "Botella 48 Oz", "Mínima", "30 días", 425, TODOS, ACEITE),
    ("Aceite de coco", "Botella 7.5 Oz", "Única", "30 días", 199, TODOS, ACEITE),
    ("Aceite de sésamo", "Botella 6.2 Oz", "Única", "30 días", 339, TODOS, ACEITE),
    ("Aceituna", "Frasco 5 Oz", "Única", "30 días", 145, TODOS, OTRO),
    ("Quinoa", "Paquete 12 Oz", "Única", "Relativo", 189, REL, GRANO),
    ("Guandules", "Lata 15 Oz", "Mínima", "Relativo", 84, REL, LEGUM),
    ("Guandules", "Paquete 800 gr", "Mayor", "30 días", 199, TODOS, LEGUM),
    ("Maíz dulce", "Lata 425 gr", "Única", "Relativo", 55, REL, VEG),
    ("Pasta integral", "Paquete 500 gr", "Única", "Relativo", 209, REL, GRANO),
    ("Maní", "Pote 300 gr", "Única", "Relativo", 185, REL, SEMILLA),
    # ── Página 6 ──
    ("Garbanzo", "Lb", "Única", "Relativo", 70, REL, LEGUM),
    ("Salsa de soya", "Frasco 10 Oz", "Única", "Relativo", 215, REL, SALSA),
    ("Harina de trigo", "Paquete 2 L", "Mínima", "Relativo", 49, REL, PAN),
    ("Mostaza", "Botella 8 Oz", "Única", "30 días", 75, TODOS, SALSA),
    ("Harina de maíz precocida", "Paquete 500 gr", "Única", "Relativo", 62, REL, PAN),
    ("Harina de trigo", "Paquete 5 L", "Mayor", "Relativo", 199, REL, PAN),
    ("Vainilla", "Botella 5 Oz", "Mínima", "30 días", 20, TODOS, COND),
    ("Ajo en polvo", "Botella 3 Oz", "Única", "30 días", 115, TODOS, COND),
    ("Salsa de tomate", "Sobre 200 gr", "Mínima", "7 días", 88, EXC7, SALSA),
    ("Salsa de tomate", "Pote 680 gr", "Mayor", "30 días", 159, TODOS, SALSA),
    ("Albahaca seca", "Sobre 25 gr", "Mínima", "15 días", 35, R715, COND),
    ("Albahaca seca", "Frasco 4 Oz", "Mayor", "30 días", 200, TODOS, COND),
    ("Galleta de soda", "Caja 20 unid", "Mínima", "15 días", 148, R715, PAN),
    ("Galleta de soda", "Caja 24 unid", "Mayor", "30 días", 265, TODOS, PAN),
    ("Granola", "Paquete L", "Única", "Relativo", 95, REL, GRANO),
    ("Almendras fileteadas", "Paquete 6 Oz", "Única", "Relativo", 289, REL, SEMILLA),
    ("Pimentón", "Frasco 85 gr", "Única", "30 días", 149, TODOS, COND),
    ("Vinagre balsámico", "Pote 0.25 Lt", "Única", "30 días", 179, TODOS, SALSA),
    ("Linaza", "Paquete 8 Oz", "Única", "Relativo", 70, REL, SEMILLA),
    ("Casabe albahaca", "Paquete 11 Oz", "Única", "Relativo", 99, REL, PAN),
    ("Queso de hoja", "Paquete 1L", "Única", "Relativo", 249, REL, LACTEO),
    # ── Página 7 ──
    ("Queso blanco", "Paquete 1L", "Única", "Relativo", 270, REL, LACTEO),
    ("Carne de res", "Lb", "Única", "Relativo", 295, REL, CARNE),
    ("Salami", "1.47 L", "Mínima", "Relativo", 194, REL, CARNE),
    ("Salami", "3.47 L", "Mayor", "Relativo", 459, REL, CARNE),
    ("Tofu", "Lata 19 Oz", "Mínima", "Relativo", 250, REL, LEGUM),
    ("Soya Texturizada", "Paquete 200 gr", "Mínima", "Relativo", 100, REL, LEGUM),
    ("Adámame", "Paquete 500 gr", "Única", "Relativo", 195, REL, LEGUM),
    ("Guisantes secos", "Lata 15 Oz", "Única", "Relativo", 125, REL, LEGUM),
    ("Frijoles pintos", "Paquete 800 gr", "Única", "Relativo", 127, REL, LEGUM),
    ("Habas", "Paquete 16 Oz", "Única", "Relativo", 205, REL, LEGUM),
    ("Semillas de cajuil", "Tarro 4 Oz", "Mínima", "Relativo", 255, REL, SEMILLA),
    ("Semillas de cajuil", "Tarro 7 Oz", "Mediana", "Relativo", 385, REL, SEMILLA),
    ("Semillas de cajuil", "Paquete 14.5 Oz", "Mayor", "30 días", 689, TODOS, SEMILLA),
    ("Semillas de girasol", "Paquete 400 gr", "Única", "Relativo", 145, REL, SEMILLA),
    ("Semillas de calabaza", "Tarro 8 Oz", "Única", "Relativo", 305, REL, SEMILLA),
    ("Nueces Mixtas", "Paquete 100gr", "Única", "Relativo", 95, REL, SEMILLA),
    ("Champiñones", "Paquete 8 Oz", "Única", "Relativo", 205, REL, VEG),
    ("Remolacha", "Lb", "Única", "Relativo", 45, REL, VEG),
    ("Apio", "Lb", "Única", "Relativo", 49, REL, VEG),
    ("Algas marinas", "Paquete 28g", "Única", "Relativo", 149, REL, VEG),
    ("Berro", "Paquete", "Única", "30 días", 44, TODOS, VEG),
    ("Rúcula", "Paquete 2 Oz", "Mínima", "Relativo", 43, REL, VEG),
    ("Rúcula", "Paquete 8 Oz", "Mayor", "30 días", 130, TODOS, VEG),
    ("Calabacín", "Lb", "Única", "Relativo", 49, REL, VEG),
    ("Kale Picado", "Paquete 6 Oz", "Única", "Relativo", 205, REL, VEG),
    ("Repollo morado", "Lb", "Única", "Relativo", 59, REL, VEG),
    ("Rábano", "Paquete 8 unid", "Única", "Relativo", 85, REL, VEG),
    ("Espárragos", "Paquete 450 gr", "Única", "Relativo", 405, REL, VEG),
    ("Coles de Bruselas", "Paquete 900 gr", "Única", "Relativo", 220, REL, VEG),
    # ── Página 8 ──
    ("Guayaba", "Lb", "Única", "Relativo", 48, REL, FRUTA),
    ("Guanábana", "Lb", "Única", "Relativo", 54, REL, FRUTA),
    ("Níspero", "Lb", "Única", "Relativo", 59, REL, FRUTA),
    ("Mandarina", "Lb", "Única", "Relativo", 109, REL, FRUTA),
    ("Toronja", "Lb", "Única", "Relativo", 140, REL, FRUTA),
    ("Uva", "Lb", "Única", "Relativo", 169, REL, FRUTA),
    ("Leche de almendras", "Cartón 32 Oz", "Única", "Relativo", 260, REL, BEBVEG),
    ("Coco", "Unidad", "Única", "Relativo", 69, REL, FRUTA),
    ("Yogur de coco regular", "Pote 6 Oz", "Única", "Relativo", 50, REL, BEBVEG),
    ("Yogur de coco griego", "Pote 8 Oz", "Única", "Relativo", 95, REL, BEBVEG),
    ("Sardinas en lata", "Lata 125 gr", "Mínima", "Relativo", 33, REL, CARNE),
    ("Sardinas en lata", "Lata 15 Oz", "Mayor", "Relativo", 57, REL, CARNE),
    ("Muslo de pollo", "Lb", "Única", "Relativo", 68, REL, CARNE),
    ("Hígado de res", "Lb", "Única", "Relativo", 119, REL, CARNE),
    ("Salmón", "Paquete 3 Oz", "Mínima", "Relativo", 490, REL, CARNE),
    ("Salmón", "Paquete 11.4 Oz", "Mayor", "Relativo", 1060, REL, CARNE),
    ("Tilapia", "Lb", "Única", "Relativo", 130, REL, CARNE),
    ("Pavo molido", "Paquete 16 Oz", "Única", "Relativo", 320, REL, CARNE),
    ("Mero", "Lb", "Única", "Relativo", 290, REL, CARNE),
    ("Leche de avena", "Cartón 1 Lt", "Única", "Relativo", 124, REL, BEBVEG),
    ("Leche de coco", "Cartón 32 Oz", "Única", "Relativo", 289, REL, BEBVEG),
    ("Leche de soya", "Cartón 32 Oz", "Única", "Relativo", 184, REL, BEBVEG),
    ("Puerro", "Paquete", "Única", "Relativo", 48, REL, VEG),
    ("Bok choy", "Paquete 250 gr", "Única", "Relativo", 90, REL, VEG),
    ("Lechuga romana", "Paquete 2 Lb", "Única", "Relativo", 158, REL, VEG),
    ("Cundeamor", "Paquete", "Mínima", "Relativo", 50, REL, VEG),
    ("Arenque", "Paquete 16 Oz", "Única", "Relativo", 175, REL, CARNE),
    ("Filete Arenque", "Paquete 8 Oz", "Única", "Relativo", 230, REL, CARNE),
    ("Leche de cabra en polvo", "Paquete 12 Oz", "Única", "Relativo", 1330, REL, LACTEO),
    ("Nabo", "Lb", "Única", "Relativo", 28, REL, VEG),
    ("Alcachofa", "Unidad", "Única", "Relativo", 345, REL, VEG),
    # ── Página 9 ──
    ("Palmito", "Lata 14.1 Oz", "Única", "Relativo", 275, REL, VEG),
    ("Cebollín", "Paquete 25 unid", "Única", "Relativo", 229, REL, VEG),
    ("Pera", "Lb", "Única", "Relativo", 99, REL, FRUTA),
    ("Kiwi", "Paquete 450 gr", "Única", "Relativo", 275, REL, FRUTA),
    ("Durazno", "Lata 15.2 Oz", "Única", "Relativo", 180, REL, FRUTA),
    ("Ciruela", "Paquete 16 Oz", "Mayor", "Relativo", 199, REL, FRUTA),
    ("Arándanos", "Paquete 450 gr", "Única", "Relativo", 198, REL, FRUTA),
    ("Tamarindo", "Paquete 1 Lb", "Única", "Relativo", 245, REL, FRUTA),
    ("Cereza", "Frasco 6 Oz", "Mínima", "Relativo", 175, REL, FRUTA),
    ("Cereza", "Frasco 10 Oz", "Mediana", "Relativo", 219, REL, FRUTA),
    ("Cereza", "Frasco 16 Oz", "Mayor", "Relativo", 298, REL, FRUTA),
    ("Granada", "Paquete 4 Oz", "Única", "Relativo", 290, REL, FRUTA),
    ("Bulgur", "Paquete 24 Oz", "Única", "Relativo", 255, REL, GRANO),
    ("Cebada", "Paquete 16 Oz", "Única", "Relativo", 118, REL, GRANO),
    ("Comino", "Pote 1 Oz", "Única", "Relativo", 55, REL, COND),
    ("Cúrcuma", "Lb", "Única", "Relativo", 99, REL, COND),
    ("Laurel", "Pote 100 gr", "Única", "Relativo", 150, REL, COND),
    ("Tomillo", "Sobre 0.5 Oz", "Mínima", "Relativo", 55, REL, COND),
    ("Tomillo", "Frasco 8 Oz", "Mayor", "Relativo", 325, REL, COND),
    ("Curry en polvo", "Frasco 2 Oz", "Única", "Relativo", 100, REL, COND),
    ("Cebolla en polvo", "Frasco 2.75 Oz", "Única", "Relativo", 105, REL, COND),
    ("Harina de Negrito", "Paquete 290 gr", "Única", "Relativo", 59, REL, PAN),
    ("Ajonjolí", "Lb", "Única", "Relativo", 235, REL, SEMILLA),
    ("Pistachos", "Tarro 8 Oz", "Única", "Relativo", 459, REL, SEMILLA),
    ("Mantequilla de almendras", "Tarro 200 gr", "Única", "Relativo", 525, REL, SEMILLA),
    ("Chuleta costillas", "Lb", "Única", "Relativo", 189, REL, CARNE),
    ("Conejo", "Lb", "Única", "Relativo", 225, REL, CARNE),
    ("Chivo", "Lb", "Única", "Relativo", 299, REL, CARNE),
    ("Pulpo", "Lb", "Única", "Relativo", 345, REL, CARNE),
    ("Calamar", "2 Lb", "Mínima", "Relativo", 460, REL, CARNE),
    ("Mejillones", "Paquete 32 Oz", "Única", "Relativo", 519, REL, CARNE),
    ("Mapuey", "Lb", "Única", "Relativo", 99, REL, VIVER),
    ("Queso cheddar", "Lb", "Única", "Relativo", 299, REL, LACTEO),
    ("Queso gouda", "Lb", "Única", "Relativo", 320, REL, LACTEO),
    # ── Página 10 ──
    ("Kéfir", "Pote 32 Oz", "Mayor", "Relativo", 420, REL, LACTEO),
    ("Kéfir", "Pote 6 Oz", "Mínima", "Relativo", 120, REL, LACTEO),
    ("Dátiles", "Lb", "Única", "Relativo", 340, REL, FRUTA),
    ("Cacao en polvo", "Paquete 200 gr", "Única", "Relativo", 130, REL, OTRO),
    ("Pasas", "Paquete 250 gr", "Única", "Relativo", 189, REL, FRUTA),
    ("Ciruela pasa", "Tarro 16 Oz", "Única", "Relativo", 199, REL, FRUTA),
    ("Cangrejo", "Paquete Lb", "Única", "Relativo", 479, REL, CARNE),
]

_INSERT = """
INSERT INTO public.supermarket_products
    (food_name, brand, presentation, portion_label, duration_label,
     price_rd, notes, category, master_food_name, is_verified, active)
VALUES (%s, NULL, %s, %s, %s, %s, %s, %s, %s, true, true)
ON CONFLICT (lower(food_name), lower(coalesce(brand,'')), lower(coalesce(presentation,'')))
DO NOTHING
RETURNING id
"""


def main():
    if not _NEON:
        print("FATAL: NEON_DATABASE_URL no está definido (.env)")
        sys.exit(1)

    # Sanity de duplicados internos del seed (fallaría el unique index a mitad).
    seen = set()
    for (food, pres, *_rest) in ROWS:
        key = (food.lower(), (pres or "").lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {pres}")
            sys.exit(1)
        seen.add(key)

    foods = {r[0] for r in ROWS}
    print(f"Seed: {len(ROWS)} presentaciones de {len(foods)} alimentos verificados.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, pres, portion, duration, price, notes, category) in ROWS:
                cur.execute(_INSERT, (food, pres, portion, duration, price, notes, category, food))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
