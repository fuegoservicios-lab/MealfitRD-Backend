#!/usr/bin/env python
"""[P1-COUNTRY-SYSTEM-F2 · 2026-08-17] Task 1 — el harness semántico + auditoría de drops.

LA MEDICIÓN que produce las listas de trabajo de T5-T8 (altas de catálogo por país). Spec:
`docs/superpowers/specs/2026-08-16-sistema-paises-design.md` §Fase 2 + Addendum del dueño
2026-08-17 punto 1 ("catálogo por completitud MEDIDA, no por cuota"). Plan:
`docs/superpowers/plans/2026-08-17-paises-fase-2.md` Task 1.

QUÉ HACE (sin una sola llamada LLM):
  --country CC   Pasa una lista curada de alimentos/platos típicos de CC (data inline, ver las
                 listas *_FOODS abajo) por el pipeline REAL de resolución de ingredientes
                 (`shopping_calculator.normalize_name`, el resolver SSOT recetas→catálogo:
                 exact/alias → fuzzy difflib ≥0.87 → embeddings Cohere ≥0.70 → pass-through)
                 contra el catálogo VIVO en Neon, y clasifica cada ítem en uno de tres veredictos:
                   RESUELVE-BIEN            — matchea un master_ingredients verificado, vía un
                                               tier léxico (exact/synonym) o fuzzy.
                   SUSTITUCION-SILENCIOSA   — matchea, pero SOLO vía el tier semántico (embeddings)
                                               — el alimento pedido no tiene alias/fuzzy en el
                                               catálogo; lo que devolvió es "parecido", no lo mismo.
                   DROP                     — no matchea nada verificado (lo que
                                               `_is_verified_for_shopping` también vería como False
                                               y `aggregate_and_deduct_shopping_list` dropearía de
                                               la lista de compras en producción).
                 Emite JSON a `backend/data/country_gaps/<cc>.json` + resumen a stdout.

  --rd-drops     Agrega la telemetría `record_verified_only_drop` (contador in-process que
                 `_creativity_kpi_job`, cron_tasks.py, snapshotea diario a `pipeline_metrics`,
                 node='_creativity_kpi_job', metadata->'top_verified_only_drops') de los últimos
                 30 días → el top de alimentos dropeados para el top-up de RD (Task 8). Mismo
                 shape de lectura que consume ese cron (mirror, no reimplementación).

CÓMO CLASIFICA SIN REIMPLEMENTAR SCORING: `normalize_name` es una función monolítica (6 INTENTOs
inline, sin hooks por-tier) que YA anuncia por logging.info() cuál tier resolvió cada match:

    INTENTO 5 (fuzzy, difflib, umbral 0.87):    "🔤 [Fuzzy Match] '<q>' -> '<m>' (ratio <r>)"
    INTENTO 6 (semántico, Cohere embed-v4, 0.70): "🧠 [Semantic Search] Resuelto: '<q>' -> '<m>' con score <s>"

Este script llama a `normalize_name` UNA vez por ítem (la función real, sin tocar su lógica) y
adjunta un `logging.Handler` temporal al root logger durante esa llamada para LEER cuál de esas
dos líneas apareció — no se recalcula ningún ratio ni se reimplementa difflib/cosine_similarity
aquí (`import difflib` está deliberadamente AUSENTE de este archivo; guard parser en
test_p1_country_system_f2.py). Si ninguna de las dos líneas aparece pero el resultado SÍ está
verificado, el match vino de un tier léxico (INTENTO 1-4: exact match o alias/synonym-contains).

⚠️ POOL: fuera de FastAPI el `ConnectionPool` de `db_core` nace `open=False` (lo abre el lifespan
de la app) — un script standalone que no lo abre mide el catálogo VACÍO, no el sistema (lección
del repo, ver CLAUDE.md "SQL forense antes de tocar código"). `_open_pool()` corre antes de
cualquier lectura de catálogo, en AMBOS modos.

⚠️ EMBEDDINGS ≠ LLM: el tier semántico usa Cohere embed-v4 (`COHERE_API_KEY`), permitido por el
brief de esta task ("embeddings ≠ LLM generation"). Si la key está ausente (o el init de Cohere
falla), el tier semántico simplemente no dispara nunca dentro de `normalize_name` — este script lo
detecta llamando `get_semantic_cache()` una vez al inicio y marca cada DROP resultante con
`semantic_tier_status: "unknown"` (en vez de afirmar un DROP definitivo que podría ser, en
realidad, una sustitución silenciosa nunca evaluada). Pre-calienta ese cache con un deadline
generoso (150s, ver `_SEMANTIC_PREWARM_DEADLINE_S`) — mismo propósito que el "calentador de
arranque" de producción (docstring de `get_semantic_cache`): sin esto, el primer ítem no resuelto
paga el build completo del cache (~200+ filas) contra el deadline de 30s de una request normal; si
no alcanza, activa un cooldown de 600s (`_SEMANTIC_INIT_FAIL_COOLDOWN_S`) que apagaría el tier
semántico para el RESTO de la corrida, falseando la medición hacia más DROPs de los reales.

⚠️ COBERTURA — SOLO EL TIER RESOLVER: este harness llama `sc.normalize_name` directo — el tier
léxico/fuzzy/semántico (INTENTO 1-6) que resuelve un nombre de alimento contra `master_ingredients`.
NO pasa por `sc.canonicalize_shopping_food_name` (la segunda cadena de canonicalización que
`aggregate_and_deduct_shopping_list` aplica DESPUÉS de esa resolución, para colapsar variantes de
preparación RD a un nombre de lista simplificado — ej. 'Plátano verde'→'Plátano'). Un RESUELVE-BIEN
aquí certifica que el resolver encuentra la fila correcta; NO certifica que esa fila sobreviva
intacta al agregador de compras — ese fue exactamente el CRITICAL #2 del fix-round 1 de T7
(2026-08-17): 6 altas T7 + 2 T5 resolvían bien en este harness y aun así llegaban renombradas (o,
en el caso de 'Nueces pecanas', DROPEADAS en silencio) a la lista de compras real. La cobertura de
esa capa vive en `tests/test_p1_country_system_f2.py` (sweep completo de
`canonicalize_shopping_food_name` + parametrize por fila sobre `aggregate_and_deduct_shopping_list`),
no en este script — añadir esa segunda llamada aquí duplicaría el gate para OPTIMIZAR una medición
que ya corre gratis en pytest, sin ganar nada (el harness es offline/manual, pytest es CI).

USO:
    cd backend
    python scripts/country_catalog_gap.py --country ES
    python scripts/country_catalog_gap.py --rd-drops

Read-only. No escribe nada en Neon. No hace ninguna llamada a un LLM de generación.

[P2-LOGGER-EXEMPT: script CLI one-shot, la salida a stdout ES el producto]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Reuso de las funciones de producción (NO reimplementadas): el resolver SSOT completo
# (`normalize_name`), el set de nombres verificados con precio (`_get_verified_shopping_name_set`),
# el umbral fuzzy nombrado (`_FUZZY_MATCH_THRESHOLD`) y el cache semántico (`get_semantic_cache`).
# Import por MÓDULO (no `from shopping_calculator import normalize_name`) a propósito: así un test
# puede monkeypatchear `shopping_calculator.normalize_name` y este script lo ve sin reimport, y en
# producción real el atributo se resuelve en el momento de la llamada, no al importar.
import shopping_calculator as sc  # noqa: E402
from constants import strip_accents  # noqa: E402

_AQUI = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_AQUI, "..", "data", "country_gaps")


# ════════════════════════════════════════════════════════════════════════════════════════════
# LISTAS CURADAS POR PAÍS — data inline, ~70-85 alimentos/ingredientes-clave-de-plato-típico
# cada una. Metodología (citada aquí, no por URL externa): mercado/supermercado categorías
# (proteínas, charcutería, lácteos, legumbres, verduras, frutas, granos/almidones, condimentos) +
# ingredientes CLAVE (no el nombre del plato) de los platos más representativos de cada país. En
# español donde el alimento tiene nombre español (ES/MX/CO/PR); nombre nativo donde no lo tiene
# (US). Ningún ítem es una llamada a un LLM — es conocimiento de dominio curado a mano.
# ════════════════════════════════════════════════════════════════════════════════════════════

# ES — España. Categorías: charcutería (jamón/chorizo/morcilla/embutidos), pescado/marisco
# (bacalao, pulpo, gambas, mejillones — costa atlántica y mediterránea), quesos (manchego, cabra,
# azul, idiazábal), legumbres (garbanzos, lentejas, judías), hortalizas de mercado semanal,
# condimentos (pimentón de la Vera, azafrán). Platos de referencia (solo para elegir el
# ingrediente clave, NUNCA el nombre del plato en sí): tortilla española (huevo/patata/cebolla),
# paella (arroz bomba/azafrán), fabada asturiana (judías blancas/morcilla/panceta), gazpacho
# (tomate/pimiento/ajo), cocido madrileño (garbanzos/chorizo), pulpo a la gallega (pulpo/pimentón).
ES_FOODS = [
    "Jamón serrano", "Jamón ibérico", "Chorizo español", "Morcilla", "Lomo embuchado",
    "Panceta ibérica", "Bacalao", "Pulpo", "Gambas", "Calamares", "Mejillones", "Almejas",
    "Boquerones", "Anchoas", "Conejo", "Cordero", "Chuletas de cerdo", "Huevos camperos",
    "Queso manchego", "Queso de cabra", "Queso azul", "Queso idiazábal", "Requesón", "Cuajada",
    "Nata", "Leche entera", "Garbanzos", "Lentejas", "Judías blancas", "Judías pintas", "Habas",
    "Pimiento del piquillo", "Pimiento verde italiano", "Alcachofa", "Espárragos trigueros",
    "Calabacín", "Berenjena", "Acelgas", "Espinacas", "Ajo", "Cebolla", "Tomate de rama", "Patata",
    "Puerro", "Coliflor", "Judías verdes", "Setas", "Champiñones", "Arroz bomba", "Pan de pueblo",
    "Pan candeal", "Fideos", "Harina de trigo", "Naranja de Valencia", "Mandarina", "Membrillo",
    "Higo", "Uva", "Melón", "Aceituna verde", "Aceituna negra", "Aceite de oliva virgen extra",
    "Pimentón de la Vera", "Azafrán", "Perejil", "Laurel", "Vinagre de Jerez", "Alioli",
    "Sal marina", "Turrón", "Mazapán", "Sobrasada", "Butifarra", "Percebes", "Vieira", "Sardinas",
    "Chistorra", "Piñones", "Almendra marcona", "Membrillo dulce",
]

# MX — México. Categorías: masa/nixtamal (tortillas, elote), chiles (frescos y secos — el eje
# entero de la cocina mexicana), hierbas/aromáticas (epazote, hoja santa, cilantro), proteínas de
# taquería (carnitas, cecina, chorizo), quesos frescos regionales. Platos de referencia (solo
# ingrediente clave): mole (chocolate de mesa, chile mulato/ancho), tamales (masa, hoja de maíz),
# pozole (maíz pozolero, chile guajillo), tacos al pastor (chile ancho, piña, achiote), pico de
# gallo (jitomate, cebolla morada, cilantro).
MX_FOODS = [
    "Masa harina", "Tortillas de maíz", "Tortillas de harina", "Elote", "Maíz cacahuazintle",
    "Maíz pozolero", "Chile jalapeño", "Chile serrano", "Chile poblano", "Chile chipotle",
    "Chile guajillo", "Chile ancho", "Chile habanero", "Chile de árbol", "Chile pasilla",
    "Chile mulato", "Chile cuaresmeño", "Chile morrón", "Tomatillo", "Nopal", "Calabacita",
    "Jícama", "Epazote", "Cilantro", "Cebolla morada", "Cebollín", "Aguacate", "Chayote",
    "Carne de res para asar", "Chorizo mexicano", "Chorizo verde", "Carnitas de cerdo", "Cecina",
    "Chicharrón de cerdo", "Camarones", "Pescado huachinango", "Longaniza", "Frijoles negros",
    "Frijoles pintos", "Frijoles refritos", "Queso oaxaca", "Queso fresco", "Queso cotija",
    "Queso panela", "Queso Chihuahua", "Crema mexicana", "Mango manila", "Guayaba", "Tamarindo",
    "Papaya", "Piña", "Limón", "Tuna (fruta de nopal)", "Jamaica (flor de)", "Xoconostle",
    "Comino", "Orégano mexicano", "Achiote", "Canela mexicana", "Hoja de plátano",
    "Hoja santa", "Chocolate de mesa", "Hoja de maíz seca", "Arroz", "Piloncillo", "Huitlacoche",
    "Flor de calabaza", "Ejotes", "Rábano", "Pepino", "Plátano macho", "Manteca de cerdo",
    "Pepitas de calabaza", "Cacahuate", "Nuez de Castilla", "Jitomate",
]

# CO — Colombia. Categorías: tubérculos andinos (papa criolla, papa pastusa, arracacha), maíz
# (arepa, mazorca, cuchuco), lácteos frescos costeños/andinos (queso costeño, suero costeño),
# frutas exóticas (lulo, uchuva, curuba, borojó). Platos de referencia (solo ingrediente clave):
# sancocho (yuca, mazorca, plátano), ajiaco santafereño (papa criolla, guascas, pollo), bandeja
# paisa (chicharrón, frijol cargamanto, arepa), arepa con hogao (harina de maíz, tomate chonto).
CO_FOODS = [
    "Arepa de maíz", "Harina de maíz precocida", "Plátano verde", "Plátano maduro", "Yuca",
    "Papa criolla", "Papa pastusa", "Papa sabanera", "Chicharrón", "Chorizo santarrosano",
    "Carne molida", "Costilla de res", "Pollo", "Pescado bagre", "Trucha", "Chontaduro",
    "Frijol cargamanto", "Garbanzo", "Lenteja", "Queso costeño", "Queso campesino",
    "Suero costeño", "Cuajada", "Guascas", "Cilantro cimarrón", "Ahuyama", "Mazorca", "Choclo",
    "Habichuela", "Arracacha", "Guanábana", "Lulo", "Maracuyá", "Mango biche", "Guayaba",
    "Curuba", "Uchuva", "Banano", "Panela", "Achiote", "Color (bijol)", "Comino", "Ají dulce",
    "Ají picante", "Cebolla larga", "Cebolla cabezona", "Tomate chonto", "Aguacate hass",
    "Arequipe", "Bocadillo de guayaba", "Natilla", "Champús", "Ají de maní",
    "Almidón de yuca agrio", "Envuelto de maíz", "Morcilla", "Longaniza santarrosana",
    "Cuchuco de trigo", "Gallina criolla", "Carne oreada", "Pescado capaz", "Palmito", "Borojó",
    "Feijoa", "Granadilla", "Tomate de árbol", "Mora", "Papaya paulina", "Ñame", "Batata",
    "Coco rallado", "Leche de coco", "Yuca brava", "Malanga",
]

# PR — Puerto Rico. La lista con más solape declarado contra RD (comparten mucho: arroz,
# habichuelas, viandas, sofrito) — el harness mide EXACTAMENTE cuán poco falta, per plan T7.
# Categorías: viandas (yautía, panapén, malanga), el sofrito/adobo/sazón como eje de sabor,
# platos de referencia (solo ingrediente clave): mofongo (plátano verde, chicharrón), pasteles
# (guineo, yautía, hoja de plátano), alcapurrias (yautía, guineo verde), arroz con dulce (arroz,
# coco, especias), bacalaítos (bacalao, harina).
PR_FOODS = [
    "Arroz", "Habichuelas rosadas", "Habichuelas blancas", "Habichuelas negras", "Gandules",
    "Plátano verde", "Plátano maduro", "Guineo verde", "Guineo maduro", "Yuca", "Batata", "Ñame",
    "Yautía blanca", "Yautía amarilla", "Panapén", "Malanga", "Pernil", "Pollo guisado",
    "Bacalao", "Jamón de cocinar", "Longaniza puertorriqueña", "Chuleta ahumada", "Sofrito",
    "Recao (culantro)", "Ajíes dulces", "Adobo", "Sazón con culantro y achiote", "Achiote",
    "Calabaza", "Chayote", "Repollo", "Apio (célery)", "Auyama", "Garbanzos", "Guayaba",
    "Parcha", "Tamarindo", "Mangó", "Limón verde criollo", "Coco", "Leche de coco",
    "Queso de papa", "Queso del país", "Aceite de achiote", "Alcaparrado",
    "Aceitunas rellenas", "Vinagre", "Papas", "Cebolla", "Ajo", "Pimiento verde", "Tomate",
    "Huevo", "Harina de maíz", "Hoja de plátano", "Chicharrón de cerdo", "Morcilla",
    "Harina de yuca", "Pique (vinagre picante)", "Surullitos (harina de maíz y queso)",
    "Carne de cerdo para pernil", "Pavochón", "Bacalaítos (masa de bacalao)", "Ron de cocina",
    "Especias para arroz con dulce", "Habichuelas pintas", "Yautía lila",
]

# US — Estados Unidos. [Task 1: Ruling (fold → T7) · 2026-08-17] Esta lista NO se dejó en inglés
# ("es su nombre nativo") como decía este comentario antes de T7 — el ruling lo corrigió: producción
# emite los ingredientes en ESPAÑOL incluso para planes de EE.UU. (el catálogo canónico en español es
# el SSOT; el LLM está instruido a usar esos nombres exactos) — medir contra nombres en inglés medía
# una distribución que producción NUNCA emite (drops inflados, artificiales). Cada ítem es el nombre
# español natural que diría un hispanohablante en EE.UU. comprando esto en un supermercado
# americano, Y el nombre que llevaría la fila del catálogo (ej. "Ground beef"→"Carne molida",
# "Bacon"→"Tocineta", "Turkey breast"→"Pechuga de pavo", "Maple syrup"→"Jarabe de arce", "Peanut
# butter"→"Mantequilla de maní", "Marshmallows"→"Malvaviscos"). Préstamos YA establecidos se
# conservan tal cual (Bagels, Pretzels, Granola, Pepperoni) — no tienen traducción de uso real,
# mismo criterio que loanwords aceptados en el resto del catálogo. Categorías: carnes de
# supermercado americano, lácteos/quesos estilo US, panadería (bagels, pan de maíz, sémola de
# maíz), enlatados/despensa (frijoles horneados, mantequilla de maní). Platos de referencia (solo
# ingrediente clave, en sus nombres de catálogo): mac and cheese (coditos, queso cheddar), meatloaf
# (carne molida, pan rallado), clam chowder (almejas, papa), apple pie (manzana, masa para pie),
# biscuits and gravy (panecillos de mantequilla, salsa de salchicha), s'mores (malvaviscos,
# galletas Graham).
US_FOODS = [
    "Carne molida", "Tocineta", "Pechuga de pavo", "Pechuga de pollo", "Chuletas de cerdo",
    "Filete de salmón", "Camarones", "Jamón de sándwich", "Salchichas", "Queso cheddar",
    "Queso crema", "Queso cottage", "Crema agria", "Suero de mantequilla", "Crema mitad y mitad",
    "Pan blanco", "Pan integral", "Bagels", "Panecillos ingleses", "Pan de maíz", "Sémola de maíz",
    "Avena", "Pretzels", "Galletas de soda", "Maíz dulce en granos", "Vainitas", "Papa",
    "Ají morrón", "Lechuga iceberg", "Apio", "Calabaza butternut", "Habichuelas negras",
    "Frijoles pintos", "Habichuelas rojas", "Frijoles horneados", "Arándanos", "Arándanos rojos",
    "Duraznos", "Manzana", "Calabaza", "Mantequilla de maní", "Jarabe de arce", "Aderezo ranch",
    "Salsa barbacoa", "Kétchup", "Mostaza", "Salsa inglesa", "Malvaviscos", "Coditos",
    "Pan rallado", "Almejas", "Masa para pie", "Galletas Graham", "Panecillos de mantequilla",
    "Salsa de salchicha", "Huevos rellenos", "Ensalada de repollo", "Ensalada de macarrones",
    "Chile en polvo", "Sazonador para tacos", "Nueces pecanas", "Nuez de Castilla", "Mantequilla de almendras",
    "Granola", "Yogur griego", "Queso en hebras", "Pepperoni", "Queso provolone",
    "Salchicha italiana", "Carne molida mixta", "Bolitas de papa", "Papas ralladas",
    "Mezcla para panqueques", "Wafles", "Miel", "Azúcar morena", "Extracto de vainilla",
    "Chili con carne",
]

CURATED_FOODS_BY_COUNTRY: dict[str, list[str]] = {
    "ES": ES_FOODS,
    "MX": MX_FOODS,
    "CO": CO_FOODS,
    "PR": PR_FOODS,
    "US": US_FOODS,
}


# ════════════════════════════════════════════════════════════════════════════════════════════
# CLASIFICADOR — lee, no reimplementa. Ver docstring del módulo.
# ════════════════════════════════════════════════════════════════════════════════════════════

# Umbral semántico: hardcoded INLINE en shopping_calculator.py (`normalize_name`, INTENTO 6,
# `if best_score >= 0.70:`) — no expuesto como constante nombrada como sí lo está
# `_FUZZY_MATCH_THRESHOLD`. Documentado aquí SOLO para el campo informativo del JSON de salida;
# nunca se usa para decidir un veredicto (eso lo decide production, leyendo su propio log).
_SEMANTIC_THRESHOLD_DOCUMENTED = 0.70

# Deadline del pre-warm del cache semántico (ver docstring del módulo). Producción usa 30s por
# default para no bloquear una request de usuario; este script es un batch offline sin usuario
# esperando, así que puede pagar el build completo del catálogo sin degradar la medición a un
# falso "tier inactivo" a mitad de corrida. Piso de emergencia si el cálculo dinámico falla.
_SEMANTIC_PREWARM_DEADLINE_FLOOR_S = 300.0


def _compute_semantic_prewarm_deadline_s() -> float:
    """Deadline DINÁMICO, no un número fijo adivinado: réplica de la aritmética que el propio
    docstring de `sc.get_semantic_cache` documenta para el calentador de arranque de
    producción (P1-EMBED-WARM-DEADLINE) — `batches = ceil(catálogo / EMBED_INIT_BATCH_SIZE)`,
    `mínimo = batches × EMBED_INIT_BATCH_DELAY_S`. Esto NO es "scoring" (no calcula ninguna
    similitud entre alimentos): es una cuenta de cuánto dura el delay deliberado entre batches
    para no saturar la API de Cohere. `EMBED_INIT_BATCH_SIZE`/`_DELAY_S` son knobs — sus
    valores reales en este entorno (medido: 3 ítems/batch, 3.0s de delay contra ~207 filas de
    catálogo ⇒ 69 batches ⇒ 207s de espera DORMIDA, antes de sumar la latencia de red de cada
    POST) difieren de sus defaults de código (10 / 0.5s), así que un número fijo adivinado a
    partir de los defaults se queda corto — es justo el bug que el propio docstring de
    producción documenta haber sufrido con un deadline de 30s. Margen ×1.6 sobre el mínimo
    dormido para cubrir la latencia de red real; si el cálculo falla por lo que sea (catálogo
    vacío, atributos ausentes), cae al piso fijo `_SEMANTIC_PREWARM_DEADLINE_FLOOR_S`."""
    try:
        n = len(sc.get_master_ingredients() or [])
        batch_size = max(1, int(sc.EMBED_INIT_BATCH_SIZE))
        delay_s = float(sc.EMBED_INIT_BATCH_DELAY_S)
        batches = -(-n // batch_size)  # ceil sin importar math
        needed_s = batches * delay_s
        return max(_SEMANTIC_PREWARM_DEADLINE_FLOOR_S, needed_s * 1.6)
    except Exception:
        return _SEMANTIC_PREWARM_DEADLINE_FLOOR_S

_FUZZY_LOG_RE = re.compile(
    r"\[Fuzzy Match\]\s*'(?P<q>[^']*)'\s*->\s*'(?P<m>[^']*)'\s*\(ratio\s*(?P<score>[0-9.]+)\)"
)
_SEMANTIC_LOG_RE = re.compile(
    r"\[Semantic Search\]\s*Resuelto:\s*'(?P<q>[^']*)'\s*->\s*'(?P<m>[^']*)'\s*con score\s*(?P<score>[0-9.]+)"
)


class _TierLogCapture(logging.Handler):
    """Handler temporal: captura los mensajes emitidos por UNA llamada a `normalize_name`
    para leer cuál INTENTO resolvió el match, sin recalcular ningún ratio."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(record.getMessage())
        except Exception:
            pass


def _resolve_with_tier(food_name: str) -> dict:
    """Llama UNA vez a `sc.normalize_name` (el resolver SSOT real) y determina qué tier
    resolvió el match leyendo las líneas de log que la propia función ya emite. Devuelve
    `{"canon": str, "tier": "fuzzy"|"semantic"|"lexical"|"error", "score": float|None}`.

    `tier="lexical"` cubre INTENTO 1-4 (exact match / alias-regex-contains) — no se distingue
    aquí exact de synonym (eso lo hace `classify_food`, comparando el texto normalizado, sin
    tocar ningún ratio)."""
    handler = _TierLogCapture()
    root = logging.getLogger()
    prev_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        canon = sc.normalize_name(food_name)
    except Exception:
        return {"canon": None, "tier": "error", "score": None}
    finally:
        root.removeHandler(handler)
        root.setLevel(prev_level)

    tier = "lexical"
    score: float | None = None
    for msg in handler.messages:
        m = _SEMANTIC_LOG_RE.search(msg)
        if m:
            tier, score = "semantic", float(m.group("score"))
            break
        m = _FUZZY_LOG_RE.search(msg)
        if m:
            tier, score = "fuzzy", float(m.group("score"))
            break
    return {"canon": canon, "tier": tier, "score": score}


def _catalog_name_set_including_unpriced() -> set:
    """[P1-COUNTRY-SYSTEM-F2 · T5 · 2026-08-17] Set de nombres canónicos (accent-stripped, lower)
    de TODO `master_ingredients` — a diferencia de `sc._get_verified_shopping_name_set()` (que
    exige `price_per_lb>0 OR price_per_unit>0`), esto incluye filas SIN precio a propósito
    (altas de catálogo-país como las de Task 5: España es país beta con
    `pricing_mode='beta_no_prices'` — el precio RD nunca aplica, así que exigirlo aquí mediría
    "¿tiene precio RD?" en vez de "¿tiene nutrición REAL verificada?", que es la pregunta que
    esta task-1 audita. NO reemplaza `_get_verified_shopping_name_set` en ningún call site de
    producción — vive solo en este script de medición offline."""
    rows = sc.get_master_ingredients() or []
    return {strip_accents(str(r.get("name") or "").lower().strip()) for r in rows if r.get("name")}


def classify_food(food_name: str, *, semantic_tier_active: bool = True, catalog_name_set: set | None = None) -> dict:
    """Clasifica UN alimento contra el catálogo vivo (o mockeado, en tests) en uno de los 3
    veredictos del contrato de Task 1:

      RESUELVE-BIEN          — matchea vía tier léxico (exact/synonym) o fuzzy (INTENTO 1-5).
      SUSTITUCION-SILENCIOSA — matchea, pero SOLO vía el tier semántico (INTENTO 6): el pedido
                                no tiene alias/fuzzy en catálogo; lo devuelto es "parecido".
      DROP                   — no matchea ningún master_ingredients verificado.

    `semantic_tier_active=False` (sin COHERE_API_KEY o falló el init) no cambia el veredicto —
    solo anota `semantic_tier_status="unknown"` en los DROP, porque esos son los únicos que
    PODRÍAN haber sido en realidad una sustitución silenciosa nunca evaluada.

    [P1-COUNTRY-SYSTEM-F2 · T5 · 2026-08-17] `catalog_name_set` (default `None` — preserva
    BYTE-IDÉNTICO el comportamiento pre-T5 y las 8 unit tests de la sección A que mockean
    `sc._get_verified_shopping_name_set` directamente) permite inyectar un set de "verificado"
    distinto al de precio-RD. `run_country_mode` lo puebla con
    `_catalog_name_set_including_unpriced()` para que un alta de catálogo-país SIN precio (T5)
    cuente como RESUELVE-BIEN si resuelve por tier léxico/fuzzy — la pregunta de Task 1 es "¿el
    catálogo tiene este alimento?", no "¿tiene precio RD?" (esa es una pregunta DISTINTA, la de
    `MEALFIT_VERIFIED_INGREDIENTS_ONLY`, con su propio mecanismo — nunca tocado aquí)."""
    resolved = _resolve_with_tier(food_name)
    canon = resolved["canon"]
    tier = resolved["tier"]
    score = resolved["score"]

    status = "checked" if semantic_tier_active else "unknown"

    if tier == "error" or canon is None:
        return {
            "food": food_name, "verdict": "DROP", "matched": None, "tier": None,
            "score": None, "semantic_tier_status": status,
        }

    _verified_set = catalog_name_set if catalog_name_set is not None else sc._get_verified_shopping_name_set()
    verified = strip_accents(str(canon).lower().strip()) in _verified_set

    if not verified:
        return {
            "food": food_name, "verdict": "DROP", "matched": None, "tier": None,
            "score": None, "semantic_tier_status": status,
        }

    if tier == "semantic":
        return {
            "food": food_name, "verdict": "SUSTITUCION-SILENCIOSA", "matched": canon,
            "tier": "semantic", "score": score, "semantic_tier_status": status,
        }

    if tier == "fuzzy":
        subtier = "fuzzy"
    else:
        q_norm = strip_accents(re.sub(r"\(.*?\)", "", food_name).strip().lower())
        c_norm = strip_accents(str(canon).strip().lower())
        subtier = "exact" if q_norm == c_norm else "synonym"

    return {
        "food": food_name, "verdict": "RESUELVE-BIEN", "matched": canon,
        "tier": subtier, "score": score, "semantic_tier_status": status,
    }


# ════════════════════════════════════════════════════════════════════════════════════════════
# POOL — fuera de FastAPI hay que abrirlo a mano (lección del repo). Ambos modos lo abren
# ANTES de tocar cualquier función que lea el catálogo.
# ════════════════════════════════════════════════════════════════════════════════════════════

def _open_pool():
    """Abre el ConnectionPool de Neon (nace `open=False`; el lifespan de FastAPI lo abre en la
    app real, y este es un script standalone). Sin esto, `get_master_ingredients()` devuelve
    `[]` y el harness mide el catálogo VACÍO, no el sistema."""
    from db_core import connection_pool
    if connection_pool is None:
        print(
            "FATAL: connection_pool es None (¿faltan NEON_DATABASE_URL/NEON_DATABASE_URL_POOLED en .env?)",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        connection_pool.open()
    except Exception as e:
        print(f"FATAL: no se pudo abrir el pool de Neon: {e}", file=sys.stderr)
        sys.exit(1)
    return connection_pool


def _write_json(filename: str, payload: dict) -> str:
    os.makedirs(_DATA_DIR, exist_ok=True)
    path = os.path.join(_DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


# ════════════════════════════════════════════════════════════════════════════════════════════
# MODO --country
# ════════════════════════════════════════════════════════════════════════════════════════════

def run_country_mode(cc: str) -> dict:
    cc = str(cc).strip().upper()
    if cc not in CURATED_FOODS_BY_COUNTRY:
        print(
            f"FATAL: país {cc!r} no tiene lista curada. Disponibles: "
            f"{sorted(CURATED_FOODS_BY_COUNTRY)}",
            file=sys.stderr,
        )
        sys.exit(1)

    _open_pool()  # ← pool abierto ANTES de tocar el catálogo (ver _open_pool docstring).

    # Pre-warm deliberado (ver docstring del módulo): reusa `get_semantic_cache`, la MISMA
    # función que el calentador de arranque de producción usa con el mismo propósito, con un
    # deadline calculado dinámicamente (ver `_compute_semantic_prewarm_deadline_s`) porque este
    # caller sí puede esperar y los defaults por-request (30s) existen para NO bloquear al
    # usuario, no para acotar un batch offline.
    deadline = _compute_semantic_prewarm_deadline_s()
    print(
        f"Pre-calentando cache semántico (deadline={deadline:.0f}s; puede tardar varios "
        "minutos la primera vez — batches de embeddings contra todo el catálogo)...",
        file=sys.stderr,
    )
    cache = sc.get_semantic_cache(deadline_s=deadline)
    semantic_tier_active = cache is not None
    if not semantic_tier_active:
        print(
            "AVISO: tier semántico NO disponible en esta corrida (sin COHERE_API_KEY, o "
            "falló el init) — los DROP quedan marcados semantic_tier_status='unknown': "
            "pueden ser falsos negativos (sustituciones silenciosas nunca evaluadas).",
            file=sys.stderr,
        )

    # [P1-COUNTRY-SYSTEM-F2 · T5 · 2026-08-17] Set de "verificado" propio de este modo — TODO
    # master_ingredients, sin exigir precio RD (ver docstring de `_catalog_name_set_including_unpriced`
    # y de `classify_food`). Altas de catálogo-país (Task 5, sin precio RD a propósito) cuentan
    # como catálogo REAL para esta medición.
    _catalog_names = _catalog_name_set_including_unpriced()
    items = CURATED_FOODS_BY_COUNTRY[cc]
    results = [
        classify_food(food, semantic_tier_active=semantic_tier_active, catalog_name_set=_catalog_names)
        for food in items
    ]

    counts = {"RESUELVE-BIEN": 0, "SUSTITUCION-SILENCIOSA": 0, "DROP": 0}
    for r in results:
        counts[r["verdict"]] += 1

    payload = {
        "mode": "country",
        "country": cc,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "semantic_tier_active": semantic_tier_active,
        "cohere_api_key_present": bool(
            os.environ.get("COHERE_API_KEY") or os.environ.get("EMBEDDINGS_API_KEY")
        ),
        "fuzzy_threshold": sc._FUZZY_MATCH_THRESHOLD,
        "semantic_threshold": _SEMANTIC_THRESHOLD_DOCUMENTED,
        "total_items": len(items),
        "counts": counts,
        "items": results,
    }
    path = _write_json(f"{cc.lower()}.json", payload)
    _print_country_summary(cc, payload, path)
    return payload


def _print_country_summary(cc: str, payload: dict, path: str) -> None:
    print(f"\n=== country_catalog_gap --country {cc} ===")
    tier_state = "ACTIVO" if payload["semantic_tier_active"] else "INACTIVO (DROPs marcados UNKNOWN)"
    print(f"{payload['total_items']} items curados · catálogo vivo · tier semántico: {tier_state}")
    c = payload["counts"]
    print(
        f"RESUELVE-BIEN: {c['RESUELVE-BIEN']}  |  "
        f"SUSTITUCION-SILENCIOSA: {c['SUSTITUCION-SILENCIOSA']}  |  DROP: {c['DROP']}"
    )

    silent = [r for r in payload["items"] if r["verdict"] == "SUSTITUCION-SILENCIOSA"]
    if silent:
        print(f"\n-- SUSTITUCION-SILENCIOSA ({len(silent)}) — bloquear ANTES de abrir el país --")
        for r in silent:
            print(f"  {r['food']!r} -> {r['matched']!r} (score={r['score']})")

    drops = [r for r in payload["items"] if r["verdict"] == "DROP"]
    if drops:
        print(f"\n-- DROP ({len(drops)}) --")
        for r in drops:
            tag = "" if r["semantic_tier_status"] == "checked" else "  [semántico UNKNOWN]"
            print(f"  {r['food']!r}{tag}")

    print(f"\nJSON: {path}")


# ════════════════════════════════════════════════════════════════════════════════════════════
# MODO --rd-drops
# ════════════════════════════════════════════════════════════════════════════════════════════

_RD_DROPS_LOOKBACK_DAYS = 30
# Mismo node que `_creativity_kpi_job` (cron_tasks.py) usa al INSERT en pipeline_metrics —
# mirror de su read shape, no reimplementación del agregador.
_RD_DROPS_NODE = "_creativity_kpi_job"

# [P1-COUNTRY-SYSTEM-F2 · Task 8 · 2026-08-17] `cron_runs_examined` NO es ≈ lookback_days. El
# cron está registrado a `MEALFIT_CREATIVITY_KPI_INTERVAL_MIN` (default 1440min = 24h, SIN
# override en `.env` de este entorno) — pero investigado en vivo (rd_drops.json del 2026-08-17:
# 338 filas / 30 días ≈ 11.3/día) el conteo real es ~11x el nominal. Causa RAÍZ (no un 2º cron ni
# drift de cadencia): `_add_job_jittered` (`cron_tasks.py`, P1-SCHEDULER-STAGGER · 2026-05-28) da
# a TODO cron `interval` sin `next_run_time`/`start_date` explícito un `next_run_time` inicial =
# `ahora + offset(job_id) ∈ [0, MEALFIT_SCHEDULER_STAGGER_MAX_S]` **en cada registro** — y sin
# jobstore persistente (default de APScheduler), cada arranque/redeploy del proceso RE-REGISTRA
# el job desde cero, así que dispara un run "bonus" dentro del primer minuto de CADA restart,
# independiente de cuándo corrió la última vez de verdad. Confirmado con los timestamps reales:
# 5 fechas tempranas (jul-19 a jul-23, baja actividad de deploy) muestran exactamente 1
# corrida/día ~24h00m aparte (el intervalo nominal puro); a partir de jul-24 (ritmo de desarrollo
# más intenso — múltiples "commit + deploy" por día, ver MEMORY.md del proyecto) el conteo diario
# salta a 6-47 corridas/día, correlacionando con la cadencia de redeploys, no con un cron más
# rápido. Es benigno por diseño (el propio comentario de P1-SCHEDULER-STAGGER: "todos los crons
# son idempotentes... un run extra al startup es benigno") — el agregado `top_verified_only_drops`
# solo se acumula MÁS densamente, no incorrectamente.


def _aggregate_rd_drops(rows: list[dict]) -> list[dict]:
    """Suma los pares `[nombre, count]` de `metadata->>'top_verified_only_drops'` a través de
    TODAS las filas (cada fila ya es un top-15 del cron, de un día distinto) → total por nombre,
    ordenado desc. Pura función de agregación — no toca la DB (testeable con filas mockeadas)."""
    totals: dict[str, int] = {}
    for row in rows:
        meta = row.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                continue
        if not isinstance(meta, dict):
            continue
        for pair in (meta.get("top_verified_only_drops") or []):
            try:
                name = str(pair[0]).strip()
                count = int(pair[1])
            except Exception:
                continue
            if not name:
                continue
            totals[name] = totals.get(name, 0) + count
    return [
        {"food": name, "count": count}
        for name, count in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def run_rd_drops_mode() -> dict:
    _open_pool()  # ← pool abierto ANTES de la query a pipeline_metrics.
    from db_core import execute_sql_query

    cutoff = datetime.now(timezone.utc) - timedelta(days=_RD_DROPS_LOOKBACK_DAYS)
    try:
        rows = execute_sql_query(
            "SELECT metadata, created_at FROM public.pipeline_metrics "
            "WHERE node = %s AND created_at >= %s ORDER BY created_at DESC",
            (_RD_DROPS_NODE, cutoff),
            fetch_all=True,
        ) or []
    except Exception as e:
        print(f"FATAL: query a pipeline_metrics falló: {e}", file=sys.stderr)
        sys.exit(1)

    top_drops = _aggregate_rd_drops(rows)
    note = None
    if not rows:
        note = (
            f"Sin corridas de '{_RD_DROPS_NODE}' en los últimos {_RD_DROPS_LOOKBACK_DAYS} días "
            "— artefacto vacío pero válido. El cron corre con "
            "MEALFIT_CREATIVITY_KPI_INTERVAL_MIN (default 1440min = 24h); si acaba de desplegarse "
            "o el knob está apagado, no hay historia todavía que agregar."
        )

    payload = {
        "mode": "rd-drops",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": _RD_DROPS_LOOKBACK_DAYS,
        "source_node": _RD_DROPS_NODE,
        "cron_runs_examined": len(rows),
        "distinct_foods": len(top_drops),
        "top_drops": top_drops,
        "note": note,
    }
    path = _write_json("rd_drops.json", payload)

    print(f"\n=== country_catalog_gap --rd-drops (lookback {_RD_DROPS_LOOKBACK_DAYS}d) ===")
    print(f"corridas de '{_RD_DROPS_NODE}' examinadas: {len(rows)}")
    if note:
        print(note)
    else:
        print(f"alimentos distintos dropeados: {len(top_drops)}")
        for entry in top_drops[:30]:
            print(f"  {entry['food']!r}: {entry['count']}")
    print(f"\nJSON: {path}")
    return payload


# ════════════════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="country_catalog_gap.py",
        description=(
            "Harness semántico + auditoría de drops del catálogo, por país "
            "(P1-COUNTRY-SYSTEM-F2 Task 1). Sin llamadas LLM. Embeddings Cohere permitidos si "
            "hay COHERE_API_KEY. Read-only sobre Neon."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--country", type=lambda s: str(s).strip().upper(),
        choices=sorted(CURATED_FOODS_BY_COUNTRY),
        help="País beta a auditar contra el catálogo vivo (ES/MX/CO/PR/US).",
    )
    group.add_argument(
        "--rd-drops", action="store_true",
        help="Agrega record_verified_only_drop de pipeline_metrics (30 días) para el top-up de RD.",
    )
    args = parser.parse_args()

    if args.rd_drops:
        run_rd_drops_mode()
    else:
        run_country_mode(args.country)


if __name__ == "__main__":
    main()
