"""[P1-CATALOG-VARIETY-OPENED · 2026-07-26] Auditoría de variedad: qué alimentos de la base no se
podían usar, y las tres cosas que se abrieron.

Números medidos el 2026-07-26 (y corregidos dos veces, porque mis propios instrumentos inflaron el
problema antes de resolver sinónimos y plurales):

    catálogo `master_ingredients`                204 alimentos
    pool del seeder (DOMINICAN_*)                142 → alcanza 157 con sinónimos
    huérfanos reales                              47 (23%), de los que ~43 son exclusiones CORRECTAS
                                                  (aceites, especias, vinagres, leches, guarniciones)
    `supermarket_products`                      1741 productos / 247 nombres de alimento
    nombres sin enlazar al catálogo               46 genuinos + 27 variantes de nombre

## Las tres piezas

1. **Tres alimentos al pool**: `Granada` (fruta) y `Granola`, `Galletas de soda` (bases de
   carbo). El cuarto aprobado, `Durazno en almíbar`, se REVIRTIÓ: ya había un test que lo excluye
   como treat-en-sirope de la rotación de fruta fresca. Tenían macros en el catálogo y el seeder no podía asignarlos
   nunca. Aprobados por el owner tras ver su perfil, así que **entran penalizados en el sorteo**:
   Galletas de soda trae **941 mg Na/100 g** — media cuota OMS del día en una sola base — y
   `P1-SODIUM-BOMB-POOL` sólo pesaba el pool de PROTEÍNAS, no el de carbos.

2. **11 alimentos nuevos al catálogo** con macros de USDA SR Legacy (`fdc_id` + provenance por fila,
   cruce Atwater). Comprables en `/supermercado` y hasta ahora imposibles de cocinar.

3. **301 productos re-enlazados** normalizando `master_food_name` (1400 de 1741 ya enlazan, antes
   ~1100).

## Lo que NO se hizo, a propósito

Cuatro alimentos se descartaron del lote de 14 y **siete renombres se dejaron fuera** porque son
alimentos distintos, no variantes: `Pasta de tomate`≠`Salsa de tomate`, `Margarina`≠`Mantequilla`,
`Cereza`≠`Cereza maraschino`… Un emparejamiento por similitud de texto los habría unido y el usuario
compraría otra cosa. Y `Yogurt de cabra` se quedó fuera porque USDA no lo tiene: rellenarlo con
yogurt de vaca sería inventar el dato con cara de fuente.
"""
import json
from pathlib import Path

import pytest

import constants as K
import graph_orchestrator as go


_RAIZ = Path(go.__file__).resolve().parent


# ───────────── 1. los cuatro alimentos, y su penalty ─────────────

def test_la_fruta_nueva_entra_al_pool():
    assert "Granada" in K.DOMINICAN_FRUITS


def test_durazno_en_almibar_se_queda_fuera_por_decision_con_test():
    """El owner aprobó los cuatro, pero `test_p1_variety_catalog_pools::
    test_treats_excluded_from_fruit_rotation` (2026-06-27) ya excluía los "treats en sirope" de la
    rotación de fruta FRESCA. Eso es información que la aprobación no tenía. Se revierte y se deja
    anclado: si alguien lo mete, ESTE test explica dónde está la otra decisión."""
    assert "Durazno en almíbar" not in K.DOMINICAN_FRUITS
    otros = ("Cereza maraschino", "Dátiles", "Pasas", "Ciruela pasa", "Coco", "Tamarindo")
    for x in otros:
        assert x not in K.DOMINICAN_FRUITS, x


@pytest.mark.parametrize("carbo", ["Granola", "Galletas de soda"])
def test_las_dos_bases_de_carbo_entran_al_pool(carbo):
    assert carbo in K.DOMINICAN_CARBS


@pytest.mark.parametrize("fruta", ["Granada"])
def test_el_gate_tambien_las_ve(fruta):
    """Si el seeder las asigna y el gate no las cuenta, una repetición se entrega invisible — el
    fallo que P1-FRUIT-SEEDER-GATE-CONTRACT cerró en la dirección contraria."""
    assert go._featured_fruits_in_name(fruta)


def test_el_pool_de_carbos_penaliza_sodio_y_azucar():
    """`P1-SODIUM-BOMB-POOL` pesa sólo proteínas. Sin este penalty, un día podía gastar media cuota
    OMS de sodio (941 mg/100 g de las galletas de soda) en su base de carbohidrato."""
    src = (_RAIZ / "ai_helpers.py").read_text(encoding="utf-8")
    assert "_SALTY_SWEET_CARB_TOKENS" in src
    assert "MEALFIT_SALTY_SWEET_CARB_PENALTY" in src
    i = src.index("_SALTY_SWEET_CARB_TOKENS")
    bloque = src[i:i + 420]
    for tok in ('"galleta"', '"granola"'):
        assert tok in bloque, tok
    assert "carb_weights" in bloque, "el penalty debe aplicarse al peso del sorteo de CARBOS"


def test_el_penalty_es_penalty_y_no_exclusion():
    """El owner pidió los cuatro: deben poder salir, sólo con menos frecuencia."""
    src = (_RAIZ / "ai_helpers.py").read_text(encoding="utf-8")
    i = src.index("_SALTY_SWEET_CARB_TOKENS")
    assert "*=" in src[i:i + 420], "multiplica el peso; no lo pone a 0 ni filtra el alimento"


# ───────────── 2. el lote de USDA: cero valores inventados ─────────────

def _lote():
    return json.loads((_RAIZ / "scripts" / "data" / "new_foods_variety_2026_07_26.json")
                      .read_text(encoding="utf-8"))


def test_el_lote_trae_los_once():
    assert len(_lote()) == 11


@pytest.mark.parametrize("campo", ["fdc_id", "provenance", "kcal", "protein_g", "carbs_g", "fats_g"])
def test_cada_alimento_es_auditable(campo):
    """Sin `fdc_id` + `provenance` un valor no se puede verificar, y este catálogo alimenta cálculos
    clínicos. La convención viene de los lotes de 2026-06-26."""
    for r in _lote():
        assert r.get(campo) not in (None, ""), f"{r['name']} sin {campo}"


def test_la_provenance_cita_usda_y_la_query():
    for r in _lote():
        assert "USDA" in r["provenance"]
        assert "query=" in r["provenance"]
        assert str(r["fdc_id"]) in r["provenance"]


def test_atwater_cruzado_y_lo_divergente_marcado():
    """4·prot + 4·carb + 9·grasa vs kcal declaradas. Lo que se pasa de 12% se MARCA en vez de
    aceptarse callado: en alimentos de ~16 kcal la fibra no cuenta para Atwater y la diferencia
    absoluta son 3 kcal (caso `Tomate enlatado`, +20,6%)."""
    for r in _lote():
        atw = 4 * r["protein_g"] + 4 * r["carbs_g"] + 9 * r["fats_g"]
        desv = abs(atw - r["kcal"]) / r["kcal"] * 100
        if desv > 12:
            assert "REVISAR" in r["provenance"], f"{r['name']} divergente y sin marca"


def test_ninguno_duplica_un_alimento_existente():
    """`Kale`, `Sardina fresca` y `Yogurt de cabra` se descartaron por esto mismo."""
    nombres = {r["name"] for r in _lote()}
    for ya in ("Kale", "Sardinas en lata", "Yogurt griego entero", "Chuleta", "Costilla de cerdo"):
        assert ya not in nombres


def test_los_descartes_quedan_explicados_en_el_script():
    """Un descarte sin razón escrita se "arregla" mal seis meses después."""
    src = (_RAIZ / "scripts" / "fetch_usda_foods_2026_07_26.py").read_text(encoding="utf-8")
    for nombre in ("Kale", "Sardina fresca", "Yogurt de cabra", "Chuleta costillas"):
        assert nombre in src, nombre
    assert "inventar el dato" in src


def test_no_se_inserta_sin_precio():
    """Gate anti-precio-0: un alimento a RD$0 sesga toda la lista de compras."""
    import re
    src = (_RAIZ / "scripts" / "add_foods_variety_2026_07_26.py").read_text(encoding="utf-8")
    assert "_is_priced" in src and "SIN PRECIO, salto" in src
    bloque = src[src.index("PRECIOS = {"):src.index("def _registros")]
    assert re.findall(r'"price_per_(?:lb|unit)":\s*None', bloque), "el bloque PRECIOS debe existir"
    assert not re.findall(r'"price_per_(?:lb|unit)":\s*[0-9]', bloque), \
        "los precios nacen en None: los llena el owner con valores reales del mercado RD"


def test_reusa_las_columnas_del_lote_previo():
    """Copiar `_COLMAP` daría dos verdades sobre el esquema del catálogo."""
    src = (_RAIZ / "scripts" / "add_foods_variety_2026_07_26.py").read_text(encoding="utf-8")
    assert "add_foods_batch1_2026_06_26.py" in src
    assert "L._COLMAP" in src and "L._derive_price_fields" in src


# ───────────── 3. la migración de nombres ─────────────

def _migracion() -> str:
    for base in (_RAIZ.parent / "migrations", _RAIZ / "migrations"):
        p = base / "p1_supermarket_master_food_name_normalize_2026_07_26.sql"
        if p.exists():
            return p.read_text(encoding="utf-8")
    raise AssertionError("falta la migración de normalización de nombres")


def test_la_migracion_vive_en_los_dos_directorios():
    """P3-MIGRATIONS-SSOT: cada repo necesita el archivo físico para que su push lo lleve."""
    nombre = "p1_supermarket_master_food_name_normalize_2026_07_26.sql"
    assert (_RAIZ.parent / "migrations" / nombre).exists()
    assert (_RAIZ / "migrations" / nombre).exists()


def test_es_idempotente_y_no_apunta_al_vacio():
    sql = _migracion()
    assert "EXISTS (SELECT 1 FROM public.master_ingredients" in sql, \
        "sólo debe renombrar si el destino EXISTE en el catálogo"
    assert "RAISE EXCEPTION" in sql, "sanity check obligatorio (P3-MIGRATION-IDEMPOTENCE-DOC)"


@pytest.mark.parametrize("par", [
    ("Aceituna", "Aceitunas"), ("Guandules", "Gandules"), ("Huevos", "Huevo"),
    ("Camarón", "Camarones"), ("Garbanzo", "Garbanzos"), ("Lenteja", "Lentejas"),
    ("Espinaca", "Espinacas"), ("Kale Picado", "Kale"),
])
def test_los_renombres_seguros_estan(par):
    sql = _migracion()
    assert f"'{par[0]}'" in sql and f"'{par[1]}'" in sql


@pytest.mark.parametrize("falso", ["Pasta de tomate", "Margarina", "Leche semidescremada",
                                  "Cereza", "Durazno", "Harina de trigo integral"])
def test_los_emparejamientos_falsos_quedan_documentados_fuera(falso):
    """Mi propio matcher difuso propuso estos siete y son alimentos DISTINTOS. Deben aparecer en la
    migración SÓLO dentro del bloque de excluidos, nunca como par a renombrar."""
    sql = _migracion()
    assert falso in sql, "el descarte debe estar escrito, no omitido"
    # La comprobación va contra el bloque de PARES, no contra el archivo entero: el encabezado
    # nombra los siete descartes para explicarlos, y un `index` sobre todo el texto encontraría
    # esa mención y daría el test por fallado (le pasó a la primera versión de este test).
    pares = sql[sql.index("_pares text[][] :="):sql.index("];", sql.index("_pares text[][] :="))]
    assert falso not in pares, f"{falso} aparece como renombre ACTIVO"
    assert falso in sql[sql.index("EXCLUIDOS A PROPÓSITO"):], f"{falso} sin razón en el bloque final"
