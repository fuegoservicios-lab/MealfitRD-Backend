"""[P1-RECONCILE-CDA-DENSITY · 2026-08-02] La inflación del solver en líneas cda/taza/conteo
llegaba ENTREGADA al usuario porque el reconciliador display↔raw era ciego a esas unidades.

## El defecto, con evidencia de producción (plan `e2bbb280`, entregado 2026-08-01)

`_dup_merge_line_to_grams` devolvía `None` para `cda`/`cdta` ("no se fusiona"). Ese `None` viaja
por `_resolve_line_food_grams` hasta `_reconcile_display_raw_lines`, donde marca la entrada como
`known=False` y hace que la reparación `qty_divergence` se SALTE la línea entera. O sea: el
reconciliador no es que decidiera dejarla, es que no podía verla.

Consecuencia medida en el plan entregado:

    display  "½ cda de cebolla picada"      raw  "30 cdas de cebolla picada"      60×
    display  "½ taza de rábanos"            raw  "30 tazas de rábanos"            60×
    display  (—)                            raw  "34.75 cdas de cebollín picado"
    display  "4.54 calabacín mediano en cubos"  (sin techo, ninguna rama lo veía)

La lista de compras se construye del lado raw (`meal.get("ingredients_raw") or ...`), así que el
usuario recibió "Calabacín 2270 g" para una persona y notas de recompra fabricadas por la basura
del solver ("Cebollín: alcanza ~3 de 30 días — recompra", ~10 mazos al mes para un adorno de dos
cucharadas). Los macros estampados del plato seguían al lado inflado (689 kcal para un plato cuyo
display son ~350).

## Las tres piezas del fix (todas bajo el MISMO knob `MEALFIT_RECONCILE_CDA_DENSITY`)

(a) `_dup_merge_line_to_grams(..., allow_spoon=True)` convierte cucharadas por DENSIDAD del
    catálogo: `cda = density_g_per_cup / 16`, `cdta = density_g_per_cup / 48`. No es una constante
    inventada — es exactamente lo que ya hace `nutrition_db.to_grams` por la vía
    `to_base_amount(q,'cda') → ml → × density_g_per_cup/240` (240/15 = 16, 240/5 = 48), verificado
    en este mismo test. Sin densidad en el catálogo sigue devolviendo `None`: este repo ya tuvo un
    incidente por INVENTAR una densidad (P1-VOLUME-FALLBACK-DENSITY, ml×5 fantasma).

    `allow_spoon` es keyword-only y default `False` a propósito: el otro consumidor de este helper
    es el dedupe (`_merge_duplicate_food_lines` → `_dup_merge_format`), y `_dup_merge_format` NO
    sabe escribir cucharadas — para una unidad que no es g/ml/taza escribe `f"{qty} {canon}"`, o
    sea que un grupo cda-dominante saldría como "3 Cebolla", una línea SIN UNIDAD. Habilitar la
    conversión ahí exigiría además un escritor de cucharadas y una medición del impacto del
    dedupe, que no es lo que este fix cierra. El default protege ese camino byte a byte.

(b) Las ramas de taza/cdta/conteo de `_cap_unrealistic_portions` usaban tuplas de tokens A MANO
    (`_REALISM_CUP_CAPS`, `_REALISM_COUNT_CAPS`) mientras la rama de gramos ya consume el set
    DERIVADO del catálogo (`_watery_veg_tokens()`, P2-VEG-VOLUME-TOKENS-2). Los propios comentarios
    del archivo lo admiten: "CADA unidad necesita su rama… tercera vez". Por eso ni el calabacín
    (conteo) ni el rábano (tazas) tenían techo. El fix NO añade un token más a mano: añade una rama
    de MASA IMPLÍCITA que convierte taza/cda/cdta/conteo a gramos con `db.grams_from_ingredient_
    string` y aplica el techo que ya existe (`REALISM_VEG_VOLUME_CAP_G`) al set derivado.

    Capear por MASA y no por un conteo inventado importa: 20 rabanitos son 240 g (una ración
    normal) y 4½ calabacines son 908 g. Un "cap de 2 unidades" para todo vegetal acuoso habría
    destrozado el primero para arreglar el segundo.

(c) Tras una reparación, `_reconcile_display_raw_lines` re-sincroniza los macros del plato con el
    display entregado (`_truth_up_meal_macros_from_strings`) — hoy el crédito nutricional seguía
    al lado inflado.

## Modo de fallo de subcadena (la trampa recurrente de este repo)

`"sal"` ⊂ `"Salami"`, `"pollo"` ⊂ `"repollo"`, `"pina"` ⊂ `"Espinacas"`, `"cos"` ⊂ `"Costilla de
cerdo"` (esta última documentada en el propio `_WATERY_VEG_TOKEN_EXCLUDE`). La rama nueva NO usa
el matcher histórico `\\b + token` (prefijo sin límite final): usa `_watery_token_hits`, que exige
límite de palabra AL FINAL tolerando plural español (`-s`/`-es`). Así "molondron" sigue matcheando
"molondrones" pero "cos" ya no matchea "costillas".

La rama de GRAMOS conserva su matcher histórico a propósito: apretarlo estrecharía un cap ya
desplegado sin medición previa, y este fix no lo cierra.
"""
from __future__ import annotations

import os

import pytest

import graph_orchestrator as g
import shopping_calculator as sc
from nutrition_db import IngredientNutritionDB

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as _f:
    _GO = _f.read()
with open(os.path.join(_BACKEND, "app.py"), encoding="utf-8") as _f:
    _APP = _f.read()


# ═════════════════════════ catálogo sintético (100% OFFLINE) ═════════════════════════
#
# El `.env` del repo apunta a PRODUCCIÓN y el worktree no lo tiene, así que `master_ingredients`
# sale VACÍO si no se stubea. Todo lo que este fix hace depende de densidades del catálogo, o sea
# que un test que leyera el catálogo vivo mediría el vacío (y encima haría red).

ROWS = [
    # Cebolla: el caso real. Excluida del set watery a propósito (`_WATERY_VEG_ROW_EXCLUDE`:
    # 'cebolla' ⊂ 'Cebolla en polvo'), así que aquí SOLO ejercita el reconciliador, no el cap.
    {"name": "Cebolla", "category": "Vegetales", "kcal_per_100g": 42.7,
     "aliases": ["cebolla roja", "cebolla picada"],
     "density_g_per_cup": 160.0, "density_g_per_unit": 150.0,
     "protein_per_100g": 1.1, "carbs_per_100g": 9.3, "fat_per_100g": 0.1},
    # Calabacín: el conteo sin techo ("4.54 calabacín mediano en cubos").
    {"name": "Calabacin", "category": "Vegetales", "kcal_per_100g": 17.0,
     "aliases": ["calabacines", "calabacin mediano"],
     "density_g_per_cup": 124.0, "density_g_per_unit": 200.0,
     "protein_per_100g": 1.2, "carbs_per_100g": 3.1, "fat_per_100g": 0.3},
    # Rábano: las tazas sin techo ("30 tazas de rábanos").
    {"name": "Rabano", "category": "Vegetales", "kcal_per_100g": 16.0,
     "aliases": ["rabanos", "rabanito"],
     "density_g_per_cup": 116.0, "density_g_per_unit": 12.0,
     "protein_per_100g": 0.7, "carbs_per_100g": 3.4, "fat_per_100g": 0.1},
    # Lechuga romana con el alias suelto 'cos' — la colisión documentada contra 'Costilla de
    # cerdo'. Vive aquí para probar que la rama nueva NO cae en el fallo de subcadena.
    {"name": "Lechuga romana", "category": "Vegetales", "kcal_per_100g": 17.0,
     "aliases": ["cos", "romana"],
     "density_g_per_cup": 47.0, "density_g_per_unit": 300.0,
     "protein_per_100g": 1.2, "carbs_per_100g": 3.3, "fat_per_100g": 0.3},
    {"name": "Costilla de cerdo", "category": "Proteínas", "kcal_per_100g": 277.0,
     "aliases": ["costilla", "costillas"],
     "density_g_per_cup": 0.0, "density_g_per_unit": 90.0,
     "protein_per_100g": 18.0, "carbs_per_100g": 0.0, "fat_per_100g": 22.0},
    # Alimento SIN densidad volumétrica: la rama cda debe seguir devolviendo None (no adivinar).
    {"name": "Casabe", "category": "Víveres", "kcal_per_100g": 340.0,
     "aliases": ["casabes"],
     "density_g_per_cup": 0.0, "density_g_per_unit": 0.0,
     "protein_per_100g": 1.0, "carbs_per_100g": 80.0, "fat_per_100g": 0.5},
    # Los 3 casos que la ronda 1 midió como REGRESIÓN de la rama de masa: líneas de 1 unidad
    # (o media) cuya masa supera el techo. Antes salían "0.83 pepino" / "0.43 coliflor" /
    # "0.28 repollo" — conteos que nadie cocina ni compra.
    {"name": "Pepino", "category": "Vegetales", "kcal_per_100g": 13.0,
     "aliases": ["pepinos"],
     "density_g_per_cup": 119.0, "density_g_per_unit": 300.0,
     "protein_per_100g": 0.6, "carbs_per_100g": 3.6, "fat_per_100g": 0.1},
    {"name": "Coliflor", "category": "Vegetales", "kcal_per_100g": 25.0,
     "aliases": ["coliflores"],
     "density_g_per_cup": 107.0, "density_g_per_unit": 580.0,
     "protein_per_100g": 1.9, "carbs_per_100g": 5.0, "fat_per_100g": 0.3},
    {"name": "Repollo", "category": "Vegetales", "kcal_per_100g": 25.0,
     "aliases": ["repollos"],
     "density_g_per_cup": 89.0, "density_g_per_unit": 900.0,
     "protein_per_100g": 1.3, "carbs_per_100g": 5.8, "fat_per_100g": 0.1},
    # Vegetal acuoso (entra al set derivado) pero SIN densidad de unidad: la rama de masa no
    # puede medirlo, así que debe abstenerse. Y no tiene entrada en `_REALISM_COUNT_CAPS`, así
    # que ninguna otra rama lo tapa — el test mide exactamente lo que dice medir.
    {"name": "Molondron", "category": "Vegetales", "kcal_per_100g": 33.0,
     "aliases": ["molondrones"],
     "density_g_per_cup": 0.0, "density_g_per_unit": 0.0,
     "protein_per_100g": 1.9, "carbs_per_100g": 7.5, "fat_per_100g": 0.2},
]


def _catalogo():
    return [dict(r) for r in ROWS]


def _db():
    """`IngredientNutritionDB` con filas INYECTADAS — `_ensure_loaded()` nunca corre, así que no
    hay ni una llamada a `get_master_ingredients()` desde el motor de macros."""
    return IngredientNutritionDB(rows=_catalogo())


@pytest.fixture(autouse=True)
def _catalogo_sintetico_y_caches_limpias(monkeypatch):
    """Los índices del catálogo y el resolvedor de líneas se cachean a nivel de MÓDULO. Sin
    resetear entre tests, el primero que corra decide el catálogo (y el valor del knob) para todos
    los demás — y con el knob monkeypatcheado a False el caché serviría un `None` obsoleto."""
    monkeypatch.setattr(sc, "get_master_ingredients", _catalogo)
    _limpia()
    yield
    _limpia()


def _limpia():
    g._CATALOG_DENSITY_INDEX_CACHE = None
    g._PHANTOM_CATALOG_INDEX_CACHE = None
    g._WATERY_VEG_TOKENS_CACHE = None
    g._LINE_FOOD_GRAMS_CACHE.clear()
    for who in ("_catalog_density_index", "_phantom_catalog_index", "_watery_veg_tokens"):
        g._CATALOG_INDEX_NEG_AT.pop(who, None)


# ═══════════════ Sección 1 — cda/cdta convierten por densidad del catálogo ═══════════════

def test_cda_convierte_via_densidad_del_catalogo():
    """Caso real: "30 cdas de cebolla picada". 160 g/taza ÷ 16 = 10 g/cda → 300 g."""
    grams = g._dup_merge_line_to_grams(30.0, "cda", "Cebolla", allow_spoon=True)
    assert grams is not None, "cda seguía siendo invisible para el resolvedor"
    assert abs(grams - 300.0) < 1.0, grams


def test_cdta_convierte_via_densidad_del_catalogo():
    """48 cdtas = 1 taza. 160 g/taza ÷ 48 = 3.33 g/cdta → 48 cdtas = 160 g."""
    grams = g._dup_merge_line_to_grams(48.0, "cdta", "Cebolla", allow_spoon=True)
    assert grams is not None
    assert abs(grams - 160.0) < 1.0, grams


@pytest.mark.parametrize("alias", ["cda", "cdas", "cucharada", "cucharadas"])
def test_todos_los_alias_de_cucharada(alias):
    assert abs(g._dup_merge_line_to_grams(1.0, alias, "Cebolla", allow_spoon=True) - 10.0) < 0.01


@pytest.mark.parametrize("alias", ["cdta", "cdtas", "cdita", "cditas",
                                   "cucharadita", "cucharaditas"])
def test_todos_los_alias_de_cucharadita(alias):
    assert abs(g._dup_merge_line_to_grams(1.0, alias, "Cebolla", allow_spoon=True) - 160.0 / 48.0) < 0.01


def test_la_division_por_16_coincide_con_nutrition_db():
    """No es una constante inventada: `nutrition_db.to_grams` ya convierte cda vía
    `to_base_amount → 15 ml → × density_g_per_cup/240`. 240/15 = 16 exacto. Si alguien cambia una
    de las dos conversiones, el reconciliador y el motor de macros dejarían de decir lo mismo
    sobre la MISMA línea — que es justo la clase de bug que este fix cierra."""
    assert abs(_db().grams_from_ingredient_string("30 cdas de cebolla picada")
               - g._dup_merge_line_to_grams(30.0, "cda", "Cebolla", allow_spoon=True)) < 0.01


def test_sin_densidad_en_catalogo_sigue_none():
    """El repo ya tuvo un incidente por INVENTAR una densidad (P1-VOLUME-FALLBACK-DENSITY, ml×5).
    Sin `density_g_per_cup` la respuesta correcta es 'no sé', no un número."""
    assert g._dup_merge_line_to_grams(3.0, "cda", "Casabe", allow_spoon=True) is None


def test_alimento_fuera_de_catalogo_sigue_none():
    assert g._dup_merge_line_to_grams(3.0, "cda", "Bacalao noruego imaginario",
                                      allow_spoon=True) is None


@pytest.mark.parametrize("unit", ["lonja", "lonjas", "pote", "lata", "al gusto", "pizca"])
def test_lonja_pote_lata_al_gusto_siguen_none(unit):
    """Fuera de alcance por decisión: no son convertibles por densidad."""
    assert g._dup_merge_line_to_grams(2.0, unit, "Cebolla", allow_spoon=True) is None


def test_dedupe_no_ve_cucharadas_por_default():
    """`allow_spoon` default False: el camino del dedupe queda EXACTAMENTE como hoy."""
    assert g._dup_merge_line_to_grams(30.0, "cda", "Cebolla") is None


def test_dedupe_no_escribe_una_linea_sin_unidad():
    """Guard funcional del default: `_dup_merge_format` no sabe escribir cucharadas — para una
    unidad que no es g/ml/taza escribe `f"{qty} {canon}"`. Si el dedupe empezara a ver cda, un
    grupo cda-dominante saldría como "3 Cebolla" (sin unidad), que es peor que no fusionar."""
    meal = {"name": "Sofrito", "ingredients": ["2 cdas de cebolla picada",
                                               "1 cda de cebolla picada"]}
    g._merge_duplicate_food_lines([{"day": 1, "meals": [meal]}])
    assert meal["ingredients"] == ["2 cdas de cebolla picada", "1 cda de cebolla picada"], (
        "el dedupe fusionó un grupo de cucharadas y `_dup_merge_format` no sabe escribirlas")


def test_resolve_line_food_grams_ya_ve_las_cucharadas():
    """El resolvedor que alimenta al reconciliador (modo completo) y al tracer (modo cheap)."""
    assert g._resolve_line_food_grams("30 cdas de cebolla picada")[1] == pytest.approx(300.0, abs=1)
    assert g._resolve_line_food_grams("30 cdas de cebolla picada",
                                      cheap=True)[1] == pytest.approx(300.0, abs=1)


def test_knob_off_restaura_el_none_historico(monkeypatch):
    monkeypatch.setattr(g, "RECONCILE_CDA_DENSITY", False)
    _limpia()
    assert g._dup_merge_line_to_grams(30.0, "cda", "Cebolla", allow_spoon=True) is None
    assert g._resolve_line_food_grams("30 cdas de cebolla picada")[1] is None


# ═══════════════ Sección 2 — el reconciliador ya repara la divergencia ═══════════════

def _meal_caso_real():
    return {"name": "Ensalada Tibia de Rábanos", "meal": "Almuerzo",
            "ingredients": ["½ cda de cebolla picada"],
            "ingredients_raw": ["30 cdas de cebolla picada"],
            "protein": 1, "carbs": 4, "fats": 0, "cals": 689}


def test_reconcile_repara_el_caso_real_e2bbb280():
    meal = _meal_caso_real()
    out = g._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}])
    assert out, "el reconciliador seguía sin ver la línea (known=False por la cda)"
    assert meal["ingredients_raw"] == ["½ cda de cebolla picada"], meal["ingredients_raw"]
    assert out[0]["kind"] == "qty_divergence"


def test_la_reparacion_solo_baja_el_raw_al_display_nunca_al_reves():
    """Contrato vigente del reconciliador: el DISPLAY manda. Este fix amplía QUÉ líneas se ven,
    no invierte la autoridad."""
    meal = _meal_caso_real()
    display_antes = list(meal["ingredients"])
    g._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}])
    assert meal["ingredients"] == display_antes


def test_la_direccion_es_raw_hacia_display_no_siempre_hacia_abajo():
    """Precisión sobre el contrato, porque es fácil describirlo mal: la reparación mueve el RAW
    hacia el DISPLAY — no "baja el raw". En los casos medidos en prod el raw venía inflado, así
    que el efecto observado es una bajada; pero P1-DISPLAY-RAW-QTY-RECONCILE midió divergencia en
    AMBAS direcciones (0.16× a 4.9×) y con el display por encima el raw SUBE. Invertir la
    autoridad sería otro fix, no éste."""
    meal = {"name": "Sofrito", "ingredients": ["30 cdas de cebolla picada"],
            "ingredients_raw": ["½ cda de cebolla picada"]}
    g._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}])
    assert meal["ingredients_raw"] == ["30 cdas de cebolla picada"]
    assert meal["ingredients"] == ["30 cdas de cebolla picada"], "el display nunca se toca"


def test_un_plato_ya_coherente_no_se_toca():
    meal = {"name": "Sofrito", "ingredients": ["2 cdas de cebolla picada"],
            "ingredients_raw": ["2 cdas de cebolla picada"],
            "protein": 0, "carbs": 2, "fats": 0, "cals": 8}
    assert g._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}]) == []
    assert meal["ingredients_raw"] == ["2 cdas de cebolla picada"]


def test_divergencia_bajo_tolerancia_no_se_toca():
    """`2 cdas` vs `2.1 cdas` es 5% — por debajo de RECONCILE_DISPLAY_RAW_TOL (10%). Se respeta
    la precisión del raw, igual que antes del fix."""
    meal = {"name": "Sofrito", "ingredients": ["2 cdas de cebolla picada"],
            "ingredients_raw": ["2.1 cdas de cebolla picada"]}
    assert g._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}]) == []
    assert meal["ingredients_raw"] == ["2.1 cdas de cebolla picada"]


def test_alimento_fuera_del_catalogo_queda_exactamente_como_hoy():
    """Sin fila en `master_ingredients` no hay densidad → `known=False` → el reconciliador se
    salta la línea, igual que antes. No adivinar es el comportamiento correcto."""
    meal = {"name": "X", "ingredients": ["½ cda de gundundun"],
            "ingredients_raw": ["30 cdas de gundundun"]}
    g._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}])
    assert meal["ingredients_raw"] == ["30 cdas de gundundun"]


def test_reparacion_idempotente():
    days = [{"day": 1, "meals": [_meal_caso_real()]}]
    assert g._reconcile_display_raw_lines(days)
    assert g._reconcile_display_raw_lines(days) == [], "tras alinear, el ratio es 1.0"


def test_knob_off_devuelve_al_reconciliador_su_ceguera(monkeypatch):
    monkeypatch.setattr(g, "RECONCILE_CDA_DENSITY", False)
    _limpia()
    meal = _meal_caso_real()
    g._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}])
    assert meal["ingredients_raw"] == ["30 cdas de cebolla picada"], (
        "con el knob apagado el rollback debe ser total (sin redeploy)")


# ═══════════════ Sección 3 — las ramas taza/conteo consumen el set DERIVADO ═══════════════

def _cap(line, *, meal_field="Almuerzo"):
    meal = {"name": "Ensalada", "meal": meal_field, "ingredients": [line],
            "ingredients_raw": [line], "protein": 2, "carbs": 12, "fats": 1, "cals": 60}
    n = g._cap_unrealistic_portions([{"meals": [meal]}], db=_db())
    return meal["ingredients"][0], n


def test_conteo_de_calabacin_capeado_via_set_derivado():
    """Caso real: "4.54 calabacín mediano en cubos" = 908 g para una persona, y NINGUNA rama lo
    veía (no es taza, no es cdta, y 'calabacin' no estaba en `_REALISM_COUNT_CAPS`).

    [ronda 1] La salida es un ENTERO: 4.54 → 1.25 por masa → `1` por el cuantizador SSOT."""
    out, n = _cap("4.54 calabacin mediano en cubos")
    assert n == 1, "el conteo de vegetal acuoso seguía sin techo"
    assert out == "1 calabacin mediano en cubos", out


def test_tazas_de_rabano_capeadas_via_set_derivado():
    """"30 tazas de rábanos" = 3480 g. `_REALISM_CUP_CAPS` solo cubre hierbas/aromáticos/frutas
    de volumen/lácteos líquidos — el vegetal acuoso no tenía techo en tazas.

    [ronda 1] En tazas SÍ se conservan fracciones (son medidas legítimas), pero pasan por el
    cuantizador: 2.25 es un cuarto de taza exacto, no el "2.16" que salía antes. El snap va a la
    fracción MÁS CERCANA, así que puede quedar un incremento por encima del techo (2.25 × 116 =
    261 g vs 250) — es el trade-off que el propio `P3-PORTION-QUANTIZE` ya acepta en todo el
    repo: una cantidad medible un pelo por encima vale más que una exacta e inservible."""
    out, n = _cap("30 tazas de rabanos")
    assert n == 1, "las tazas de vegetal acuoso seguían sin techo"
    assert out == "2.25 tazas de rabanos", out
    assert (_db().grams_from_ingredient_string(out)
            <= g.REALISM_VEG_VOLUME_CAP_G * 1.10), out


# ───────────── [ronda 1] los 4 conteos fraccionarios medidos como regresión ─────────────
#
# La rama de masa aplicaba `CAP/_wg` a pelo y no snapeaba a una cantidad medible, así que
# fabricaba conteos que ningún humano escribe en líneas que HOY salen limpias. El propio archivo
# ya denunciaba esta clase ("y encima fraccionario, que ningún humano escribe",
# P1-CAP-STRICTEST-WINS) y `P3-PORTION-QUANTIZE` existe justo por esto ("0.66 huevos matan la
# adherencia"). La rama de conteo histórica nunca tuvo el defecto: su factor es `_cap_n/cur_n`,
# o sea cae en el entero del techo por construcción.

@pytest.mark.parametrize("linea,antes_medido", [
    ("1 pepino", "0.83 pepino"),
    ("1 coliflor", "0.43 coliflor"),
    ("½ repollo", "0.28 repollo"),
])
def test_una_unidad_entera_es_el_piso_no_se_capea_por_debajo(linea, antes_medido):
    """Una verdura ENTERA es el mínimo cocinable y comprable. Si el techo de 250 g pediría menos
    de una unidad, la línea NO se capea — no existe "0.43 coliflor" ni en la cocina ni en el
    colmado."""
    out, n = _cap(linea)
    assert out == linea, f"salió {out!r} (la ronda 1 midió {antes_medido!r})"
    assert n == 0


def test_dos_calabacines_se_capean_a_un_entero():
    out, n = _cap("2 calabacines")
    assert out == "1 calabacines", out
    assert n == 1


@pytest.mark.parametrize("linea", ["1 pepino", "1 coliflor", "½ repollo", "2 calabacines",
                                   "4.54 calabacin mediano en cubos", "30 tazas de rabanos"])
def test_tras_el_cap_display_y_raw_siguen_coincidiendo(linea):
    """El cuantizador se compone en el FACTOR, antes del rescale compartido, así que las dos
    listas reciben la MISMA cantidad. Cuantizar después (o dejar que el pulido final redondee
    solo el display) habría fabricado divergencia display↔raw NUEVA — con macros stale encima —
    en líneas que hoy salen limpias."""
    meal = {"name": "Ensalada", "meal": "Almuerzo", "ingredients": [linea],
            "ingredients_raw": [linea], "protein": 2, "carbs": 12, "fats": 1, "cals": 60}
    g._cap_unrealistic_portions([{"meals": [meal]}], db=_db())
    assert meal["ingredients"][0] == meal["ingredients_raw"][0], (
        f"display {meal['ingredients'][0]!r} != raw {meal['ingredients_raw'][0]!r}")


@pytest.mark.parametrize("linea", ["1 pepino", "½ repollo", "2 calabacines",
                                   "4.54 calabacin mediano en cubos", "30 tazas de rabanos"])
def test_tras_el_pulido_de_display_las_dos_listas_siguen_diciendo_lo_mismo(linea):
    """El cierre honesto de "display y raw coinciden": el boundary polish
    (`_prettify_quantity_display`) reescribe SOLO el display, así que la pregunta real es si
    puede cambiar la CANTIDAD.

    Antes de la ronda 1 sí podía: el cap dejaba "1.25 calabacines" y el pulido lo volvía
    "1¼ calabacines" mientras raw se quedaba en 1.25 — y "0.83 pepino" ni lo tocaba, llegando así
    al PDF. Ahora los conteos salen enteros (el pulido es no-op) y las tazas salen en cuartos
    exactos, donde el pulido solo cambia el GLIFO (2.25 → 2¼): mismo número, que es justo lo que
    esta aserción comprueba — se comparan gramos parseados, no cadenas."""
    from humanize_ingredients import _prettify_quantity_display as _pretty
    meal = {"name": "Ensalada", "meal": "Almuerzo", "ingredients": [linea],
            "ingredients_raw": [linea], "protein": 2, "carbs": 12, "fats": 1, "cals": 60}
    g._cap_unrealistic_portions([{"meals": [meal]}], db=_db())
    disp_pulido = _pretty(meal["ingredients"][0])
    raw_final = meal["ingredients_raw"][0]
    gd = _db().grams_from_ingredient_string(disp_pulido)
    gr = _db().grams_from_ingredient_string(raw_final)
    assert gd is not None and gr is not None, (disp_pulido, raw_final)
    assert abs(gd - gr) < 0.5, (
        f"tras el pulido el display dice {gd:.1f} g ({disp_pulido!r}) y la lista compra "
        f"{gr:.1f} g ({raw_final!r})")


def test_la_salida_del_cap_no_deja_conteos_fraccionarios_raros():
    """Barrido: ninguna línea count-led capeada puede salir con una cantidad que el cuantizador
    SSOT no aprobaría (enteros o ½ para discretos)."""
    for linea in ("1 pepino", "1 coliflor", "½ repollo", "2 calabacines", "9 pepinos",
                  "4.54 calabacin mediano en cubos", "3 coliflores"):
        out, _ = _cap(linea)
        lead = g._realism_lead_qty(out.lower())
        if lead is None:
            continue
        assert abs(lead * 2 - round(lead * 2)) < 1e-6, f"{linea!r} → {out!r} (lead {lead})"


def test_porcion_razonable_en_tazas_no_se_toca():
    out, n = _cap("½ taza de rabanos")      # 58 g
    assert out == "½ taza de rabanos", out
    assert n == 0


def test_conteo_alto_pero_masa_razonable_no_se_toca():
    """20 rabanitos son 240 g — una ración normal. Capear por CONTEO (un "máx. 2 unidades" para
    todo vegetal acuoso) habría destrozado este caso para arreglar el calabacín; por eso la rama
    nueva capea por MASA."""
    out, n = _cap("20 rabanos")
    assert out == "20 rabanos", out
    assert n == 0


def test_linea_en_gramos_la_sigue_gobernando_la_rama_de_gramos():
    """La rama nueva se abstiene cuando la línea DECLARA gramos (líder o entre paréntesis): esa
    dimensión ya tiene dueño y duplicar el gobierno invita a recortes compuestos."""
    out, _ = _cap("400 g de calabacin")
    assert out.startswith("250"), out


def test_vegetal_acuoso_sin_densidad_de_unidad_queda_como_hoy():
    """El molondrón está en el set derivado (33 kcal, Vegetales) pero el catálogo no le da
    `density_g_per_unit`: sin masa que medir, la rama nueva se abstiene en vez de inventar."""
    out, n = _cap("9 molondrones")
    assert out == "9 molondrones", out
    assert n == 0


def test_alimento_fuera_del_catalogo_en_el_cap_queda_como_hoy():
    """(c) del riesgo declarado: una línea cuyo alimento no resuelve debe salir EXACTAMENTE
    igual que antes del fix."""
    out, n = _cap("9 gundunduns frescos")
    assert out == "9 gundunduns frescos", out
    assert n == 0


# ───────────── el modo de fallo de subcadena ─────────────

@pytest.mark.parametrize("food,token", [
    ("costillas", "cos"),        # documentado en _WATERY_VEG_TOKEN_EXCLUDE
    ("repollo", "pollo"),
    ("salami", "sal"),
    ("espinacas", "pina"),
    ("guisantes", "guisa"),
    ("fresco", "res"),
])
def test_el_matcher_exige_limite_de_palabra(food, token):
    assert not g._watery_token_hits(food, frozenset({token})), (
        f"{token!r} no debe matchear dentro de {food!r} — el modo de fallo de subcadena "
        f"que este repo ya sufrió media docena de veces")


@pytest.mark.parametrize("food,token", [
    ("rabanos", "rabano"),                     # plural -s
    ("molondrones", "molondron"),              # plural -es
    ("calabacin mediano en cubos", "calabacin"),
    ("coles de bruselas", "coles de bruselas"),  # token multi-palabra
    ("pepino", "pepino"),
])
def test_el_matcher_sigue_cubriendo_las_formas_reales(food, token):
    assert g._watery_token_hits(food, frozenset({token}))


def test_knob_off_devuelve_el_conteo_a_su_estado_sin_techo(monkeypatch):
    """Rollback total: el mismo knob apaga las tres piezas del fix."""
    monkeypatch.setattr(g, "RECONCILE_CDA_DENSITY", False)
    _limpia()
    out, n = _cap("4.54 calabacin mediano en cubos")
    assert out == "4.54 calabacin mediano en cubos", out
    assert n == 0


def test_el_alias_cos_no_entra_al_set_derivado():
    """[ronda 1] Reemplaza a un test que era VACUO: comprobaba que "4 costillas de cerdo" no se
    capeaba por el alias 'cos', pero 'cos' está en `_WATERY_VEG_TOKEN_EXCLUDE` y por tanto nunca
    entra al set — el test no ejercitaba nada. Lo que SÍ es un invariante real es la exclusión
    misma; eso es lo que se ancla aquí. La segunda línea de defensa (que el matcher protegería a
    'costillas' aunque la exclusión desapareciera) la cubre el parametrizado de arriba."""
    toks = g._watery_veg_tokens()
    assert "cos" not in toks, (
        "'cos' (alias de Lechuga romana) volvió al set derivado — colisiona con "
        "'Costilla de cerdo'; es la clase 'sal' ⊂ 'Salami' documentada en el repo")
    assert "romana" in toks, "el resto de aliases de la fila NO debe excluirse"


def test_mencionar_un_vegetal_acuoso_no_recorta_el_alimento_principal():
    """MENCIÓN ≠ ATRIBUCIÓN. Las ramas históricas buscan el token en la línea entera, así que
    "4 costillas de cerdo con rábanos" recortaría el CERDO porque el rábano aparece en el texto.
    La rama nueva pregunta al motor de macros de quién son esos gramos ('Costilla de cerdo') y
    se abstiene — la lección de 'atribución por CLÁUSULA' de P1-CULINARY-CONTRACT."""
    out, n = _cap("4 costillas de cerdo con rabanos")
    assert out == "4 costillas de cerdo con rabanos", out
    assert n == 0


# ═══════════════ Sección 4 — los macros vuelven a seguir al display ═══════════════

def test_tras_reparar_los_macros_siguen_al_display():
    """El plato entregado declaraba 689 kcal con un display de ~350: los macros estampados
    seguían al lado inflado del solver."""
    meal = _meal_caso_real()
    g._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}], db=_db())
    esperado = _db().macros_from_ingredient_string("½ cda de cebolla picada")["kcal"]
    assert abs(float(meal["cals"]) - esperado) <= 2.0, (
        f"macros estampados {meal['cals']} vs display {esperado:.1f} kcal")


def test_sin_db_explicito_el_truth_up_igual_corre():
    """NO-INERCIA. Los 7 callsites reales (assemble, finalize, db_plans, tools y tres en routers)
    llaman `_reconcile_display_raw_lines(days)` SIN db. Si la corrección de macros dependiera de
    un parámetro que nadie llena, la feature estaría muerta pareciendo sana — el modo de fallo
    que ya cazamos en P1-PLAN-QUALITY-INDEX. La instancia se construye perezosamente."""
    meal = _meal_caso_real()
    g._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}])
    assert meal["ingredients_raw"] == ["½ cda de cebolla picada"]
    assert meal["cals"] != 689, "los macros seguían al lado inflado del solver"


def test_los_callsites_reales_no_pasan_db():
    """Ancla de la razón anterior: si algún día alguien empieza a pasar `db` explícito, este test
    no falla — pero si el lazy desaparece Y los callsites siguen sin pasarlo, el de arriba sí."""
    assert "_reconcile_display_raw_lines(days)" in _GO
    assert '_reconcile_display_raw_lines(result.get("days") or [])' in _GO


def test_sin_reparacion_no_se_tocan_los_macros():
    """Un plato coherente no debe ver sus macros reescritos por este pase."""
    meal = {"name": "Sofrito", "ingredients": ["2 cdas de cebolla picada"],
            "ingredients_raw": ["2 cdas de cebolla picada"],
            "protein": 0, "carbs": 2, "fats": 0, "cals": 777}
    g._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}], db=_db())
    assert meal["cals"] == 777


def test_missing_in_raw_no_dispara_el_truth_up():
    """[ronda 1] El truth-up solo se justifica cuando se CONSTATÓ que display y raw declaraban
    cantidades distintas (`qty_divergence`) — esa es la evidencia de que los macros estampados
    pueden venir del lado inflado. En `missing_in_raw` el display nunca discrepó de nada: solo
    faltaba una línea en la lista. Reescribir ahí el bloque entero de macros excede la razón del
    fix, y se midió caro: `cals=999` salía como `cals=135`."""
    meal = {"name": "Ensalada", "ingredients": ["2 cdas de cebolla picada", "1 taza de rabanos"],
            "ingredients_raw": ["2 cdas de cebolla picada"],
            "protein": 9, "carbs": 90, "fats": 9, "cals": 999}
    out = g._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}], db=_db())
    assert [r["kind"] for r in out] == ["missing_in_raw"]
    assert meal["ingredients_raw"] == ["2 cdas de cebolla picada", "1 taza de rabanos"], (
        "la reparación de raw sí debe ocurrir")
    assert meal["cals"] == 999, "los macros NO deben reescribirse en missing_in_raw"


def test_la_telemetria_declara_si_los_macros_se_movieron():
    """[ronda 1] Descartar el `changed` de `_truth_up_meal_macros_from_strings` dejaba el cambio
    de macros sin rastro en `_display_raw_reconciled`: nadie podía auditar después de quién fue."""
    meal = _meal_caso_real()
    out = g._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}], db=_db())
    assert out and out[0]["kind"] == "qty_divergence"
    assert out[0].get("macros_truthed_up") is True, out[0]


def test_missing_in_raw_no_lleva_la_clave_de_truth_up():
    meal = {"name": "Ensalada", "ingredients": ["2 cdas de cebolla picada", "1 taza de rabanos"],
            "ingredients_raw": ["2 cdas de cebolla picada"]}
    out = g._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}], db=_db())
    assert "macros_truthed_up" not in out[0]


# ═══════ Sección 4b — [ronda 1] pureza del mutator: la llamada cara es la excepción ═══════

class _DBEspia:
    """Cuenta las resoluciones de macros. `macros_from_ingredient_string` puede caer al Tier 3
    (`normalize_name` → `get_semantic_cache`/`embed_query`: RED con reintentos) cuando la
    `IngredientNutritionDB` no viene inyectada, y esta rama corre por-línea DENTRO de
    `_swap_mutator`, o sea bajo el `SELECT … FOR UPDATE`. P2-MUTATOR-PURITY exige que el camino
    base bajo el lock sea CPU puro."""

    def __init__(self):
        self.n = 0
        self._real = _db()

    def macros_from_ingredient_string(self, s):
        self.n += 1
        return self._real.macros_from_ingredient_string(s)

    def grams_from_ingredient_string(self, s):
        return self._real.grams_from_ingredient_string(s)

    def lookup(self, s):
        return self._real.lookup(s)


def test_linea_sin_token_acuoso_no_resuelve_macros():
    """El match de tokens (CPU puro) va PRIMERO; la resolución de macros solo si pasó el filtro."""
    espia = _DBEspia()
    meal = {"name": "X", "meal": "Cena", "ingredients": ["3 costillas de cerdo"],
            "ingredients_raw": ["3 costillas de cerdo"]}
    g._cap_unrealistic_portions([{"meals": [meal]}], db=espia)
    assert espia.n == 0, (
        f"se resolvieron macros {espia.n} vez/veces para una línea sin ningún vegetal acuoso — "
        f"la llamada cara debe ser la excepción, no el caso base")


def test_linea_con_token_acuoso_si_resuelve_macros():
    """Control positivo: sin esto, el test de arriba pasaría también con la rama muerta."""
    espia = _DBEspia()
    meal = {"name": "X", "meal": "Cena", "ingredients": ["4.54 calabacin mediano en cubos"],
            "ingredients_raw": ["4.54 calabacin mediano en cubos"]}
    g._cap_unrealistic_portions([{"meals": [meal]}], db=espia)
    assert espia.n >= 1
    assert meal["ingredients"][0] == "1 calabacin mediano en cubos"


# ═══════ Sección 4c — [ronda 1] el reconcile corre ANTES de medir la banda ═══════
#
# `apply_update_band_parity` computa el band score y refresca `delivered_macros`. Desde este fix
# el reconciliador REESCRIBE los macros del plato reparado, así que corriendo después dejaba el
# banner `_quality_degraded`, los chips `_macro_band_low` y `delivered_macros` calculados sobre
# números que el pase siguiente cambiaba — la clase que el repo ya cerró en
# P2-REGEN-DAY-CHIP-STALE-CLEAR y P1-MICRO-REPORT-REFRESH. `tools.py` (chat-modify) ya tenía el
# orden correcto; las 3 superficies de `routers/plans.py` eran la excepción.

with open(os.path.join(_BACKEND, "routers", "plans.py"), encoding="utf-8") as _f:
    _PLANS = _f.read()
with open(os.path.join(_BACKEND, "tools.py"), encoding="utf-8") as _f:
    _TOOLS = _f.read()


@pytest.mark.parametrize("alias_reconcile,alias_banda", [
    ("_rdrl_sw(", "apply_update_band_parity as _ubp_sw"),
    ("_rdrl_rd(", "apply_update_band_parity as _ubp_rd"),
    ("_rdrl_ex(", "apply_update_band_parity as _ubp_exp"),
])
def test_reconcile_antes_de_band_parity_en_las_3_superficies(alias_reconcile, alias_banda):
    i_rec, i_band = _PLANS.find(alias_reconcile), _PLANS.find(alias_banda)
    assert i_rec != -1 and i_band != -1, (alias_reconcile, alias_banda)
    assert i_rec < i_band, (
        f"{alias_reconcile} (offset {i_rec}) debe correr ANTES de {alias_banda} (offset "
        f"{i_band}): el reconcile reescribe macros y la banda los mide")


def test_el_reconcile_sigue_antes_del_rebuild_de_listas():
    """No-regresión del contrato original de P1-UPDATE-RAW-BY-FOOD: mover el bloque arriba no
    puede dejarlo después del rebuild, que lee `ingredients_raw` primero."""
    assert _PLANS.find("_rdrl_sw(") < _PLANS.find(
        '_rebuild_plan_shopping_lists_inline(\n                plan_data, verified_user_id')


def test_chat_modify_ya_tenia_el_orden_correcto():
    """Ancla del precedente citado en los comentarios: si alguien invierte tools.py, este test
    avisa de que la excepción cambió de bando."""
    assert _TOOLS.find("_rdrl_cm(") < _TOOLS.find("apply_update_band_parity as _ubp_cm")


# ═══════════════ Sección 5 — knob, marker y bump anclados ═══════════════

def test_knob_default_y_registro():
    from knobs import _KNOBS_REGISTRY
    assert g.RECONCILE_CDA_DENSITY is True
    assert "MEALFIT_RECONCILE_CDA_DENSITY" in _KNOBS_REGISTRY, (
        "el knob debe auto-registrarse vía `_env_bool` (nunca `os.environ` crudo)")
    assert '_env_bool("MEALFIT_RECONCILE_CDA_DENSITY", True)' in _GO


def test_marker_anclado_en_fuente():
    assert "P1-RECONCILE-CDA-DENSITY" in _GO
    assert "e2bbb280" in _GO, "el plan vivo que motivó el fix debe quedar anclado en el código"
    assert "_watery_token_hits" in _GO


def test_last_known_pfix_bumpeado():
    assert '_LAST_KNOWN_PFIX = "P1-RECONCILE-CDA-DENSITY · 2026-08-02"' in _APP
