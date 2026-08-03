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
    veía (no es taza, no es cdta, y 'calabacin' no estaba en `_REALISM_COUNT_CAPS`)."""
    out, n = _cap("4.54 calabacin mediano en cubos")
    assert n >= 1, "el conteo de vegetal acuoso seguía sin techo"
    assert not out.startswith("4"), out
    assert _db().grams_from_ingredient_string(out) <= g.REALISM_VEG_VOLUME_CAP_G + 1, out


def test_tazas_de_rabano_capeadas_via_set_derivado():
    """"30 tazas de rábanos" = 3480 g. `_REALISM_CUP_CAPS` solo cubre hierbas/aromáticos/frutas
    de volumen/lácteos líquidos — el vegetal acuoso no tenía techo en tazas."""
    out, n = _cap("30 tazas de rabanos")
    assert n >= 1, "las tazas de vegetal acuoso seguían sin techo"
    assert _db().grams_from_ingredient_string(out) <= g.REALISM_VEG_VOLUME_CAP_G + 1, out


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


def test_costilla_de_cerdo_no_se_capea_pese_al_alias_cos():
    """Funcional de punta a punta del fallo de subcadena: 'cos' es alias de 'Lechuga romana' y es
    subcadena de 'costillas'. Si la rama nueva usara el matcher histórico (`\\b` + token, sin
    límite final), 4 costillas de cerdo (360 g) quedarían recortadas a 250 g de proteína."""
    out, n = _cap("4 costillas de cerdo")
    assert out == "4 costillas de cerdo", out
    assert n == 0


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
