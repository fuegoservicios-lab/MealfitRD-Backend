"""[P1-COUNTRY-KEEP-RESPECT-QTY · 2026-08-21] Toda la comida nativa de los 5 países beta salía a
la lista de compras con una cantidad FALSA fija de 150 g.

La rama `keep` que P1-COUNTRY-SYSTEM-F2 T5 añadió al agregador —la que evita que un alimento de
catálogo-país sin precio se DROPEE— hace tres cosas seguidas:

    weight_in_lbs = _COUNTRY_CATALOG_UNPRICED_DEFAULT_G / 453.592   # 150 g
    has_weight = True
    units = {}                                                     # ← borra la demanda real

Ese `units = {}` tira al suelo el dict que el bucle acababa de traer con lo que las recetas
pidieron de verdad. Medido sobre los DOS planes beta vivos en producción, los 7 ítems de
catálogo-país tienen `base_qty` **exactamente 150.0**:

    Almejas (ES)          recetas piden 653 g   ->  «¼ lb»  («alcanza ~2 de 7 días»)
    Membrillo (ES)                        443 g ->  «¼ lb»  («alcanza ~2 de 7 días»)
    Queso provolone (US)                  386 g ->  «¼ lb»  («alcanza ~3 de 7 días»)
    Acelgas (ES)                          504 g ->  «¼ lb»  SIN NOTA
    Judías pintas (ES) · Aderezo ranch (US) · Salsa inglesa (US)  ->  «¼ lb»  SIN NOTA

En 4 de los 7 la receta pide en tazas o cucharadas, el déficit no se puede calcular en gramos y la
maquinaria de honestidad (P1-VEG-BACKFILL-HONESTY) no llega: **sub-compra muda**. Y «¼ lb de salsa
Worcestershire» es una unidad absurda para un líquido embotellado.

LA INVERSIÓN. El default de 150 g se diseñó para el caso en que el agregador NO supo extraer peso
—«un puñado», «al gusto», sin cantidad—, y acabó ganando SIEMPRE. El fix invierte la precedencia:
si la receta dio peso o volumen convertible, mandan las recetas; el default queda como último
recurso. La rama hermana de horneado NO se toca: 100 g de polvo de hornear *es* la respuesta
correcta a «1 cdta», porque ahí lo que se compra es el envase, no la cantidad.

LO QUE ESTE FIX DESTAPA, Y POR QUÉ VA EN EL MISMO CAMBIO. 136 de las 141 filas beta no tienen
densidad. Mientras todo salía a 150 g fijos ese hueco estaba tapado; al respetar la receta, «1
taza de Nata» empieza a convertirse por el fallback genérico de 150 g/taza (una nata real son
~240). Por eso las 13 filas cremosas/líquidas del lote beta reciben su densidad aquí mismo — un
arreglo bueno que destapa uno malo no está terminado.

ORDEN. Este P-fix va ANTES de P1-COHERENCE-MIRROR-KEEP. Espejar el guard primero convertiría 4
avisos inocuos en 3 bloqueos con retry garantizado-fútil: al conservar estas filas en el lado
esperado aparecen divergencias de MAGNITUD contra los 150 g inventados, y ningún reintento las
elimina porque la divergencia es estructural.

Cubre:
  A. La receta manda cuando dio peso, en g / kg / oz / lb / ml.
  B. El default sobrevive cuando la receta no dio cantidad convertible.
  C. La rama de horneado no se movió.
  D. El contrato de `beta_no_prices` intacto: la fila sigue sin precio y sin costo.
  E. El knob de rollback.
  F. Parser-based.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# [P2-CI-BACKEND-SIBLINGS · 2026-09-04] Este módulo necesita el catálogo/la base de datos o el
# .env local (pasa en el checkout del dueño; en el CI sin NEON_DATABASE_URL se salta con motivo).
pytestmark = pytest.mark.needs_local_data

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SC_PATH = _BACKEND_ROOT / "shopping_calculator.py"


@pytest.fixture(scope="module")
def sc():
    import shopping_calculator as _sc
    return _sc


@pytest.fixture(autouse=True)
def verified_only(monkeypatch):
    """El keep sólo existe dentro de la rama VERIFIED-ONLY."""
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")


def _item(sc, linea, nombre):
    res = sc.aggregate_and_deduct_shopping_list([linea], structured=True)
    items = res.get("items") if isinstance(res, dict) else res
    return next((i for i in items if i.get("name") == nombre), None)


# ── A. La receta manda ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("linea,nombre,gramos_esperados", [
    ("650 g de Almejas", "Almejas", 650),
    ("500 g de Acelgas", "Acelgas", 500),
    ("1 kg de Almejas", "Almejas", 1000),
    ("16 oz de Acelgas", "Acelgas", 453.6),
    ("2 lb de Almejas", "Almejas", 907.2),
])
def test_la_cantidad_de_la_receta_gana_al_default_de_150g(sc, linea, nombre, gramos_esperados):
    """RED pre-fix: los 5 salían a 150,0 g. Un español que compra ¼ lb de almejas para un plan
    que pide 650 g cocina el primer día y el resto de la semana no sale."""
    it = _item(sc, linea, nombre)
    assert it is not None, f"'{nombre}' se dropeó de la lista"
    base = float(it.get("base_qty") or 0)
    assert abs(base - gramos_esperados) < gramos_esperados * 0.02, (
        f"'{nombre}': la lista pide {base:.1f} g para una receta de {gramos_esperados} g"
    )


def test_el_volumen_de_la_receta_tambien_gana(sc):
    """Los líquidos beta (Salsa inglesa, Aderezo ranch, Nata) son los que peor se leían: «¼ lb de
    salsa Worcestershire» no es una cantidad que nadie pueda comprar."""
    it = _item(sc, "300 ml de Nata", "Nata")
    assert it is not None
    assert float(it.get("base_qty") or 0) > 200, (
        "un volumen explícito sigue perdiendo contra el default de 150 g"
    )


def test_varias_lineas_del_mismo_alimento_se_suman(sc):
    """La demanda real de un plan llega repartida en días. Antes, N líneas daban 150 g igual."""
    res = sc.aggregate_and_deduct_shopping_list(
        ["200 g de Acelgas", "300 g de Acelgas", "150 g de Acelgas"], structured=True)
    items = res.get("items") if isinstance(res, dict) else res
    it = next((i for i in items if i.get("name") == "Acelgas"), None)
    assert it is not None
    assert float(it.get("base_qty") or 0) > 600, "las líneas no se sumaron: sigue el valor fijo"


# ── B. El default sobrevive donde se diseñó ─────────────────────────────────────────────────────

@pytest.mark.parametrize("linea", ["Acelgas al gusto", "una pizca de Acelgas"])
def test_sin_cantidad_convertible_el_default_sigue_aplicando(sc, linea):
    """El default de 150 g NO se borra: es la respuesta correcta a «al gusto». Lo que cambia es
    que deja de ganarle a una cantidad explícita. Sin este control, el fix sería un swap de un
    defecto por otro (el alimento volvería a dropearse por no tener peso)."""
    it = _item(sc, linea, "Acelgas")
    assert it is not None, "sin cantidad convertible el alimento volvió a dropearse"


# ── C. La rama hermana de horneado no se movió ──────────────────────────────────────────────────

def test_el_staple_de_horneado_conserva_su_empaque(sc):
    """100 g de levadura para «1 cdta» es CORRECTO: ahí lo que se compra es el ENVASE, no la
    cantidad. Este control impide arreglar la rama de al lado por simetría equivocada.

    Se usa Levadura y no «Polvo de hornear» porque este último SÍ tiene precio en el catálogo
    vivo: nunca entra en la rama de staples, así que como control no probaba nada — la primera
    versión de este test medía otra rama y lo delató al fallar contra el código correcto."""
    it = _item(sc, "1 cdta de Levadura", "Levadura")
    assert it is not None, "el staple de horneado se dropeó"
    assert abs(float(it.get("base_qty") or 0) - sc._BAKING_STAPLE_DEFAULT_G) < 1.0, (
        "el default de horneado cambió: era el correcto y no entraba en este P-fix"
    )


def test_el_staple_de_horneado_ignora_la_cantidad_de_la_receta_a_proposito(sc):
    """El otro lado del control, y la razón de que las dos ramas diverjan: aunque la receta pida
    500 g de levadura, lo que se compra sigue siendo un envase. La inversión de precedencia es
    SÓLO para la rama de catálogo-país, donde la cantidad sí es lo que se compra."""
    it = _item(sc, "500 g de Levadura", "Levadura")
    assert it is not None
    assert abs(float(it.get("base_qty") or 0) - sc._BAKING_STAPLE_DEFAULT_G) < 1.0


# ── D. El contrato beta_no_prices sigue intacto ─────────────────────────────────────────────────

def test_la_fila_sigue_sin_precio_y_sin_costo(sc):
    """Respetar la cantidad no le inventa un precio: el alimento sigue siendo `beta_no_prices` y
    su costo estimado sigue siendo None. Si esto fallara, el fix estaría metiendo montos en RD$ en
    la lista de un usuario al que se le prometió que no los vería."""
    it = _item(sc, "650 g de Almejas", "Almejas")
    assert it is not None
    assert it.get("estimated_cost_rd") is None


# ── E. Knob de rollback ─────────────────────────────────────────────────────────────────────────

def test_el_knob_apagado_devuelve_la_conducta_anterior(sc, monkeypatch):
    """Cambio en el camino caliente del agregador (categoría/peso/SKU/costo) ⇒ knob propio, según
    la convención del repo: «cambios de comportamiento que pueden necesitar revertirse sin
    redeploy van como knob, no como hardcode»."""
    monkeypatch.setenv("MEALFIT_COUNTRY_KEEP_RESPECT_RECIPE_QTY", "false")
    it = _item(sc, "650 g de Almejas", "Almejas")
    assert it is not None
    assert abs(float(it.get("base_qty") or 0) - 150.0) < 1.0, (
        "con el knob apagado debe volver el 150 g fijo"
    )


# ── F. Parser-based ─────────────────────────────────────────────────────────────────────────────

def test_la_migracion_de_densidad_existe_en_los_dos_directorios_y_es_idempotente():
    """El backfill que este P-fix destapa viaja como migración, y el repo exige que TODA migración
    viva en `migrations/` Y en `backend/migrations/` (P3-MIGRATIONS-SSOT): el workspace-root
    excluye backend/ de su .gitignore, así que sin las dos copias el push de uno de los repos no
    la lleva. El `WHERE ... IS NULL` es lo que la hace re-ejecutable sin pisar un valor curado a
    mano después."""
    nombre = "p1_country_keep_density_beta_2026_08_21.sql"
    root = _BACKEND_ROOT.parent
    a, b = root / "migrations" / nombre, _BACKEND_ROOT / "migrations" / nombre
    assert a.exists() and b.exists(), "la migración no está en los DOS directorios"
    sql = b.read_text(encoding="utf-8")
    assert a.read_text(encoding="utf-8") == sql, "las dos copias divergieron"
    assert "density_g_per_cup IS NULL" in sql, "la migración no es idempotente"
    assert "RAISE EXCEPTION" in sql, "sin sanity check: una migración muda no se entera de fallar"

    # La primera versión de este assert escaneaba el SQL entero y se chocó con la PROSA de la
    # propia migración, que nombra esos tres sólidos justamente para explicar por qué NO se tocan
    # — el patrón «un comentario derrota al guard» que este repo ya ha pagado seis veces. El
    # filtro tiene que ser conservador en la dirección correcta: en un check de «esto no debe
    # aparecer», comerse CÓDIGO sería un falso VERDE, así que se quitan sólo los comentarios de
    # línea de SQL, que no pueden contener una fila de datos.
    codigo = "\n".join(l.split("--", 1)[0] for l in sql.splitlines())
    for solido in ("Chocolate de mesa", "Masa para pie", "Especias para arroz con dulce"):
        assert solido not in codigo, (
            f"'{solido}' es un sólido que se compra por envase: inventarle una densidad sería peor "
            f"que no tenerla"
        )
    assert "Jarabe de arce" in codigo, "el filtro de comentarios se comió también las filas reales"


def test_el_fuente_declara_el_marker_y_no_toca_la_rama_de_horneado():
    src = _SC_PATH.read_text(encoding="utf-8", errors="replace")
    assert "P1-COUNTRY-KEEP-RESPECT-QTY" in src
    assert "MEALFIT_COUNTRY_KEEP_RESPECT_RECIPE_QTY" in src
    # La rama de horneado conserva su forma: preset del peso + units vaciado.
    # OJO al buscarla: desde P1-COHERENCE-MIRROR-KEEP hay DOS sitios con esa condición — el SSOT
    # `_survives_shopping_list` (que sólo responde True/False) y la rama real del agregador. La
    # primera versión de este assert cogía la primera ocurrencia y medía el helper equivocado.
    i = src.find("if _baking_staples_keep_enabled() and is_baking_pantry_staple(name):\n                weight_in_lbs")
    assert i > 0, "la rama de horneado del agregador desapareció o cambió de forma"
    rama = src[i:i + 400]
    assert "_BAKING_STAPLE_DEFAULT_G / 453.592" in rama and "units = {}" in rama, (
        "la rama de horneado cambió: su default es correcto y no entraba en este P-fix"
    )
