"""[P2-VEG-VOLUME-TOKENS-2 · 2026-08-01] Ola de fixes derivada de 2 diagnósticos SOLO-LECTURA
sobre los 17 planes de prod (`.superpowers/esparragos-600g-diagnostico.md`).

## Sección 1 — 4 vegetales acuosos faltantes en `_REALISM_VOLUME_VEG_TOKENS`

`_REALISM_VOLUME_VEG_TOKENS` (cap de 250 g/línea, `P1-VEG-VOLUME-TOKENS`) ya había tenido que
ampliarse una vez (2026-07-26) y el forense midió que seguía incompleta: 3/17 planes de prod
(17.6%) tenían ≥1 línea de vegetal ACUOSO en 300-600 g para una sola porción, con 4 vegetales
DISTINTOS que el allowlist no cubría — ninguno repetido, la firma clásica de "lista a mano que
drifea" que el propio diagnóstico documenta:

    600 g de espárragos           (plan 5f4bb17e — el solver LSQ per-meal infló el único "carbs
                                    driver" barato en gramos del plato; 100g→343g reproducido en
                                    vivo, factor 3.43×. El 600 exacto además evade
                                    LINE_GRAM_HARD_CAP, que exige `>600` estricto)
    400 g de vainitas cortadas    (plan cf3a81fb)
    2.75 tazas de coles de Bruselas (413g)   (plan 7c545d59)
    21.5 molondrones medianos (≈322 g)       (plan 7c545d59)

Fix: añadir `esparrago`, `vainita`, `coles de bruselas`, `molondron` a la tuple
(`graph_orchestrator.py`). `LINE_GRAM_HARD_CAP` (el techo genérico de 600g) NO se toca — el
diagnóstico lo desaconseja explícitamente (bajarlo rompería porciones legítimas de arroz/pollo en
household grande; el backstop correcto para vegetales acuosos es el cap específico, no el
genérico). El refactor estructural que el diagnóstico propone (derivar el set desde
`master_ingredients WHERE category='Vegetales'` en vez de una tuple a mano, mismo principio que
P1-DIET-CANON-SSOT) quedó documentado como follow-up — **implementado en la misma sesión, ver
Sección 3**: UNA HORA después del parche de los 4 tokens de arriba, el siguiente plan real
(8d3f246a) trajo un 5º vegetal fuera de lista ("470 g de tayota" ×2, 19-22.5 kcal/100g) —
confirmando la lección ya anotada en la memoria del repo: "una lista de tokens que crece
por-incidente garantiza el próximo incidente" (mismo diagnóstico que motivó P1-DIET-CANON-SSOT
sobre las 3 tablas de dietType a mano).

## Sección 2 — qty-sync ciego a "lonjas/pedazos" (`_STEP_QTY_UNITS` + `_STEP_QTY_MENTION_RE`)

Causa del caso queso 30↔45 del mismo diagnóstico (§3, plan 5f4bb17e): `ingredients` decía
"30 g de queso" pero el paso de Mise en place decía "desmenuza 1¾ lonjas/pedazos de queso (45 g)"
— `_sync_recipe_step_quantities` (P1-RECIPE-QTY-SYNC) EXISTE precisamente para reconciliar esto,
pero (a) `pedazo(?:s)?` no estaba en `_STEP_QTY_UNITS`, y (b) aunque "lonja" sí estaba, el slash
PEGADO ("lonjas/pedazos", sin espacio) rompía el match: tras la unidad el patrón exige
`\\.?\\s+de\\s+` inmediatamente, y lo que sigue a "lonjas" es "/pedazos", no espacio+"de". Fix:
`pedazo(?:s)?` añadido a `_STEP_QTY_UNITS` + `_STEP_QTY_MENTION_RE` tolera un 2º token de unidad
opcional pegado por slash (`(?:/(?:...))?`) sin cambiar qué unidad queda en el grupo `unit`
(sigue siendo la primera) — la reescritura reemplaza la mención COMPLETA por
"<qty_actual> <unit_actual> de <food>", así que la forma compuesta del ORIGEN no necesita
sobrevivir. Solo se tocó `_STEP_QTY_MENTION_RE` (lado pasos/receta) — `_ING_LEAD_QTY_RE`/
`_ING_LEAD_QTY_MIXED_RE` (lado `ingredients[]`) no reciben la forma "unidad/unidad" en la
práctica, así que se dejaron sin cambios.
"""
from __future__ import annotations

import os
import re

import pytest

import graph_orchestrator as g
import shopping_calculator as sc

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()
with open(os.path.join(_BACKEND, "routers", "plans.py"), encoding="utf-8") as f:
    _PLANS = f.read()
with open(os.path.join(_BACKEND, "app.py"), encoding="utf-8") as f:
    _APP = f.read()


@pytest.fixture(autouse=True)
def _limpio_watery_veg_cache():
    """El set derivado se cachea módulo-level (mismo gate negative-TTL que `_phantom_catalog_
    index`/`_catalog_kcal_by_name`, ver `test_p1_catalog_index_no_sticky.py`) — sin resetear entre
    tests, el primero que corre decide el catálogo (real o monkeypatcheado) para TODOS los demás."""
    g._WATERY_VEG_TOKENS_CACHE = None
    g._CATALOG_INDEX_NEG_AT.pop("_watery_veg_tokens", None)
    yield
    g._WATERY_VEG_TOKENS_CACHE = None
    g._CATALOG_INDEX_NEG_AT.pop("_watery_veg_tokens", None)


# ═══════════════════════ Sección 1 — veg volume tokens ═══════════════════════

_NUEVOS_TOKENS = ("esparrago", "vainita", "coles de bruselas", "molondron")


def test_parser_los_4_tokens_nuevos_estan_en_la_tuple():
    for tok in _NUEVOS_TOKENS:
        assert tok in g._REALISM_VOLUME_VEG_TOKENS, (
            f"falta {tok!r} en _REALISM_VOLUME_VEG_TOKENS — regresión del fix P2-VEG-VOLUME-TOKENS-2"
        )


def test_parser_marker_y_evidencia_anclados_en_fuente():
    assert "P2-VEG-VOLUME-TOKENS-2" in _GO
    assert "esparragos-600g-diagnostico.md" in _GO or "esparragos-600g" in _GO


class _DB:
    """Espejo del dummy de `test_p1_apio_stalk_cap.py`/`test_p1_recipe_blockers_2.py`: solo
    necesita responder a `_ingredient_macro_group` (via `macros_from_ingredient_string`) para que
    la clasificación de grupo (protein/carbs/fats) del cap de LINE_GRAM_HARD_CAP no interfiera
    con la rama del cap de volumen vegetal, que es la que este test ejercita."""

    def macros_from_ingredient_string(self, s):
        return {"protein": 0.0, "carbs": 0.0, "fats": 0.0, "kcal": 0.0}

    def lookup(self, s):
        return object()

    def _ingredient_macro_group(self, *a, **k):
        return None


def _cap(line):
    meal = {"name": "Batata Majada Gratinada con Queso de Hoja, Tomate y Espárragos",
            "meal": "Cena", "ingredients": [line], "ingredients_raw": [line],
            "protein": 24, "carbs": 62, "fats": 13, "cals": 464}
    g._cap_unrealistic_portions([{"meals": [meal]}], db=_DB())
    return meal["ingredients"][0]


def test_funcional_esparragos_600g_capeado_a_250():
    """Caso real literal del diagnóstico (plan 5f4bb17e): "600 g de espárragos" para 1 persona
    debe quedar recortado al techo de REALISM_VEG_VOLUME_CAP_G (250g, default)."""
    out = _cap("600 g de esparragos")
    assert out.startswith("250"), out
    assert g.REALISM_VEG_VOLUME_CAP_G == 250


def test_funcional_vainitas_400g_capeado():
    assert _cap("400 g de vainitas cortadas").startswith("250")


def test_funcional_coles_de_bruselas_capeado_pese_a_ser_multi_palabra():
    """'coles de bruselas' es un token de 3 palabras en la tuple — el matcher del consumidor
    (`_re.search(r"\\b" + t, il)`) hace substring sobre texto lowercased+accent-stripped, así
    que un token con espacios matchea igual que uno de una sola palabra. Verificado en runtime,
    no solo como afirmación de parser."""
    assert _cap("280 g de coles de bruselas").startswith("250")


def test_funcional_molondron_singular_matchea_forma_plural():
    """El token es singular ('molondron'); la línea real usa plural ('molondrones') — substring
    search sin límite de fin de palabra, así que el singular matchea dentro del plural."""
    assert _cap("350 g de molondrones").startswith("250")


def test_funcional_porcion_razonable_no_se_toca():
    """Control negativo: una porción realista (por debajo del cap) queda intacta."""
    out = _cap("150 g de esparragos")
    assert out.startswith("150"), out


def test_la_auyama_sigue_fuera_decision_explicita():
    """No-regresión de la decisión P1-VEG-VOLUME-TOKENS: la auyama es base de carbohidrato, no
    relleno de volumen — no debe entrar nunca sin su propia medición de impacto en macros."""
    assert not any(re.search(r"\b" + t, "auyama") for t in g._REALISM_VOLUME_VEG_TOKENS)


def test_line_gram_hard_cap_no_tocado():
    """El diagnóstico desaconseja bajar el techo genérico de 600g — este fix es SOLO el
    allowlist de vegetales, LINE_GRAM_HARD_CAP debe seguir en su default."""
    assert g.LINE_GRAM_HARD_CAP == 600


# ═══════════════════════ Sección 2 — qty-sync "lonjas/pedazos" ═══════════════════════

def test_parser_pedazo_en_step_qty_units():
    assert "pedazo(?:s)?" in _GO, "falta pedazo(?:s)? en _STEP_QTY_UNITS"


def test_caso_real_lonjas_slash_pedazos_se_sincroniza(monkeypatch):
    """Texto REAL del diagnóstico (plan 5f4bb17e, §3): ingrediente re-escalado a "30 g de queso"
    (post-solver) pero el paso de Mise en place seguía congelado en "1¾ lonjas/pedazos de queso
    (45 g)" — el slash pegado rompía el match del qty-sync. Debe sincronizarse a "30 g de queso"."""
    monkeypatch.setattr(g, "RECIPE_STEP_QTY_SYNC_ENABLED", True)
    meal = {
        "name": "Batata Majada Gratinada con Queso de Hoja, Tomate y Espárragos",
        "ingredients": ["30 g de queso", "600 g de esparragos"],
        "recipe": [
            "Mise en place: Pela y aplasta ½ batata mediana (145 g); desmenuza 1¾ "
            "lonjas/pedazos de queso (45 g); corta los espárragos en trozos de 4 cm.",
            "El Toque de Fuego: saltea los espárragos con el aceite de oliva a fuego medio "
            "durante 5-6 minutos.",
        ],
    }
    n = g._sync_recipe_step_quantities(meal)
    assert n >= 1, "esperada ≥1 mención reescrita"
    assert "30 g de queso" in meal["recipe"][0], meal["recipe"][0]
    assert "1¾ lonjas/pedazos" not in meal["recipe"][0], meal["recipe"][0]


def test_lonjas_slash_pedazos_aislado_via_step_qty_mention_re():
    """El regex compilado matchea la mención compuesta directamente (sin pasar por el pipeline
    completo de _sync_recipe_step_quantities) — aísla la causa exacta que el diagnóstico
    documentó: unit='lonjas', el '/pedazos' pegado ya no rompe el `\\.?\\s+de\\s+` posterior."""
    texto = "desmenuza 1¾ lonjas/pedazos de queso (45 g)"
    m = g._STEP_QTY_MENTION_RE.search(texto)
    assert m is not None, "el regex debe matchear la forma compuesta con slash"
    assert m.group("unit") == "lonjas"
    assert m.group("food") == "queso"


def test_pedazos_solo_sin_slash_tambien_matchea():
    """Control: 'pedazos' solo (sin compuesto con slash) ahora es una unidad reconocida por sí
    misma, no solo como 2º término de un compuesto."""
    texto = "corta 2 pedazos de queso"
    m = g._STEP_QTY_MENTION_RE.search(texto)
    assert m is not None
    assert m.group("unit") == "pedazos"
    assert m.group("food") == "queso"


# ═══════════════ Sección 3 — P2-VEG-VOLUME-TOKENS-2 estructural: derivar del catálogo ═══════════
#
# UNA HORA después de parchear los 4 tokens de la Sección 1, el siguiente plan real (8d3f246a)
# trajo un 5º vegetal fuera de lista: "470 g de tayota" ×2 (19-22.5 kcal/100g). La lista a mano
# había vuelto a quedarse corta antes de que se secara la tinta del fix anterior — la firma exacta
# de "una lista de tokens que crece por-incidente garantiza el próximo incidente" (misma clase que
# P1-DIET-CANON-SSOT, que colapsó 3 tablas de dietType a mano en una función SSOT).
#
# Fix: `_watery_veg_tokens()` (graph_orchestrator.py) deriva el set de `master_ingredients WHERE
# category='Vegetales' AND kcal_per_100g <= MEALFIT_WATERY_VEG_KCAL_MAX` (default 45.0) — un
# vegetal acuoso nuevo que el catálogo adquiera entra SOLO, sin PR. La tuple estática original
# (renombrada `_REALISM_VOLUME_VEG_TOKENS_FALLBACK`) NO se borra: el consumidor usa la UNIÓN
# derivado∪fallback SIEMPRE (no "derivado O fallback" condicionado a catálogo-vacío) — así los 4
# tokens del parche manual (esparrago/vainita/coles de bruselas/molondron) y los 5 originales
# (pepino/berro/rabano/rabanito/apio) están cubiertos pase lo que pase con la DB, incluso cuando
# alguno de ellos (p.ej. "coles de bruselas", 52 kcal/100g) queda por ENCIMA del umbral de
# derivación y por ende ausente de la parte derivada.
#
# Verificado contra el catálogo REAL de prod (Neon, SOLO-LECTURA, 2026-08-01, 204 filas / 39
# 'Vegetales'): 30 filas Vegetales califican bajo el umbral 45.0, expandidas a 149 tokens vía
# nombre+aliases (singular/plural/sinónimos EN incluidos gratis — el mismo mecanismo con el que
# `_phantom_catalog_index` ya resuelve "esparrago" como alias de "Espárragos"). Colisiones
# encontradas y excluidas (verificación exhaustiva: cada token candidato vs los 204 nombres+alias
# del catálogo completo, con el MISMO matcher `\b`+substring que usa el consumidor):
#   - fila 'Tomate' (fila completa excluida): su forma base 'tomate' matchea 'Salsa de tomate'
#     (Despensa). Cero regresión — 'tomate' nunca estuvo en el fallback.
#   - fila 'Cebolla' (fila completa excluida): su forma base 'cebolla' matchea 'Cebolla en polvo'
#     (Despensa, 4 formas de alias). Cero regresión — 'cebolla' nunca estuvo en el fallback.
#   - alias suelto 'cos' (de 'Lechuga romana', ing. "cos lettuce"): matchea 'Costilla de cerdo'
#     (Proteínas) — la clase EXACTA ya documentada en la memoria del repo ('sal' ⊂ 'Salami'/
#     'Salmón'). Se excluye solo el alias, no la fila (sus otros aliases no colisionan).
# Union final medida contra prod: 151 tokens (149 derivados + 2 del fallback que no derivan por
# umbral: 'coles de bruselas' 52 kcal, 'vainita' singular sin alias en el catálogo).


class _SyntheticCatalog:
    """Catálogo mínimo controlado — no depende de la DB real, hermético para CI."""

    ROWS = [
        {"name": "Tayota", "category": "Vegetales", "kcal_per_100g": 19.0,
         "aliases": ["tayotas", "chayote"]},
        {"name": "Espárragos", "category": "Vegetales", "kcal_per_100g": 25.4,
         "aliases": ["esparrago", "esparragos verdes"]},
        {"name": "Batata", "category": "Víveres", "kcal_per_100g": 90.0, "aliases": ["batatas"]},
        {"name": "Ajo", "category": "Vegetales", "kcal_per_100g": 142.7,
         "aliases": ["diente de ajo"]},  # denso, no acuoso: por encima del umbral
        {"name": "Tomate", "category": "Vegetales", "kcal_per_100g": 20.9,
         "aliases": ["tomates"]},  # fila colisionante — debe quedar excluida
        {"name": "Salsa de tomate", "category": "Despensa", "kcal_per_100g": 28.7, "aliases": []},
    ]

    @classmethod
    def get(cls):
        return [dict(r) for r in cls.ROWS]


# ───────────── (a) funcional: catálogo sintético inyectado ─────────────

def test_a_tayota_bajo_kcal_entra_en_el_set(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", _SyntheticCatalog.get)
    toks = g._watery_veg_tokens()
    assert "tayota" in toks, f"tayota (19 kcal/100g, Vegetales) debe derivarse — set: {sorted(toks)}"


def test_a_batata_densa_no_entra_por_categoria_ni_kcal(monkeypatch):
    """Batata es 'Víveres' (no 'Vegetales') Y 90 kcal/100g > 45 — doble motivo de exclusión."""
    monkeypatch.setattr(sc, "get_master_ingredients", _SyntheticCatalog.get)
    toks = g._watery_veg_tokens()
    assert "batata" not in toks


def test_a_ajo_denso_no_entra_pese_a_ser_vegetales(monkeypatch):
    """Control de umbral: categoría correcta pero kcal muy por encima — no debe colarse."""
    monkeypatch.setattr(sc, "get_master_ingredients", _SyntheticCatalog.get)
    toks = g._watery_veg_tokens()
    assert "ajo" not in toks


def test_a_tomate_excluido_por_colision_documentada(monkeypatch):
    """'tomate' (20.9 kcal, Vegetales) colisiona con 'Salsa de tomate' (Despensa) — fila entera
    excluida a propósito. No es una regresión: 'tomate' nunca estuvo en el fallback estático."""
    monkeypatch.setattr(sc, "get_master_ingredients", _SyntheticCatalog.get)
    toks = g._watery_veg_tokens()
    assert "tomate" not in toks


# ───────────── (b) fail-open: catálogo vacío → fallback estático ─────────────

def test_b_catalogo_vacio_cae_al_fallback_estatico(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: [])
    toks = g._watery_veg_tokens()
    assert toks == g._REALISM_VOLUME_VEG_TOKENS_FALLBACK_SET, (
        "con catálogo vacío el set debe ser EXACTAMENTE el piso estático — el cap jamás se queda "
        "sin lista por un blip de DB")


def test_b_catalogo_que_lanza_excepcion_tambien_cae_al_fallback(monkeypatch):
    def _boom():
        raise RuntimeError("Neon caído (blip)")
    monkeypatch.setattr(sc, "get_master_ingredients", _boom)
    toks = g._watery_veg_tokens()
    assert toks == g._REALISM_VOLUME_VEG_TOKENS_FALLBACK_SET


def test_b_fallo_no_se_pega_para_siempre(monkeypatch):
    """Mismo contrato que `_phantom_catalog_index` (P1-CATALOG-INDEX-NO-STICKY): tras el TTL
    negativo, la siguiente llamada reintenta contra el catálogo real."""
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: (_ for _ in ()).throw(
        RuntimeError("blip")))
    assert g._watery_veg_tokens() == g._REALISM_VOLUME_VEG_TOKENS_FALLBACK_SET
    monkeypatch.setattr(g, "_CATALOG_INDEX_NEG_TTL_S", 0.0)  # como si el TTL ya pasara
    monkeypatch.setattr(sc, "get_master_ingredients", _SyntheticCatalog.get)
    toks = g._watery_veg_tokens()
    assert "tayota" in toks, "tras el TTL debe reintentar y reconstruir con el catálogo bueno"


# ───────────── (c) caso real: 470 g de tayota → capeado a 250 ─────────────

class _DB3:
    def macros_from_ingredient_string(self, s):
        return {"protein": 0.0, "carbs": 0.0, "fats": 0.0, "kcal": 0.0}

    def lookup(self, s):
        return object()

    def _ingredient_macro_group(self, *a, **k):
        return None


def test_c_470g_de_tayota_capeado_a_250_caso_real_8d3f246a(monkeypatch):
    """Caso literal del plan vivo 8d3f246a: "470 g de tayota" ×2 — el 5º vegetal fuera de lista,
    UNA HORA después del parche manual de los 4 anteriores. Con el set derivado del catálogo,
    tayota entra sin necesitar un 5º token a mano."""
    monkeypatch.setattr(sc, "get_master_ingredients", _SyntheticCatalog.get)
    meal = {"name": "Test", "meal": "Cena", "ingredients": ["470 g de tayota"],
            "ingredients_raw": ["470 g de tayota"],
            "protein": 24, "carbs": 62, "fats": 13, "cals": 464}
    n = g._cap_unrealistic_portions([{"meals": [meal]}], db=_DB3())
    assert n == 1
    assert meal["ingredients"][0].startswith("250"), meal["ingredients"][0]
    assert g.REALISM_VEG_VOLUME_CAP_G == 250


# ───────────── (d) los 4 tokens del parche manual siguen cubiertos ─────────────

@pytest.mark.parametrize("tok", ["esparrago", "vainita", "coles de bruselas", "molondron"])
def test_d_los_4_tokens_del_parche_manual_siguen_cubiertos(tok, monkeypatch):
    """Vía derivación O fallback — no importa cuál de las dos rutas los cubra, lo que no puede
    pasar es que el refactor estructural haga desaparecer una cobertura ya probada en producción.
    Ejercitado con un catálogo sintético que NO los contiene (fuerza la cobertura por fallback)."""
    monkeypatch.setattr(sc, "get_master_ingredients", _SyntheticCatalog.get)
    toks = g._watery_veg_tokens()
    assert tok in toks, (
        f"{tok!r} debe seguir cubierto (vía derivación o vía fallback) tras el refactor "
        f"estructural — regresión del incidente P2-VEG-VOLUME-TOKENS-2")


def test_d_los_5_originales_tambien_siguen_cubiertos(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", _SyntheticCatalog.get)
    toks = g._watery_veg_tokens()
    for orig in ("pepino", "berro", "rabano", "rabanito", "apio"):
        assert orig in toks, orig


def test_d_fallback_es_siempre_subconjunto_del_resultado(monkeypatch):
    """Invariante de diseño: la unión NUNCA pierde el piso estático, catálogo presente o no."""
    monkeypatch.setattr(sc, "get_master_ingredients", _SyntheticCatalog.get)
    toks = g._watery_veg_tokens()
    assert g._REALISM_VOLUME_VEG_TOKENS_FALLBACK_SET <= toks


# ───────────── knob + clamp ─────────────

def test_knob_watery_veg_kcal_max_default_y_clamp():
    assert g.MEALFIT_WATERY_VEG_KCAL_MAX == 45.0
    assert "10.0 <= v <= 100.0" in _GO, "el clamp del knob MEALFIT_WATERY_VEG_KCAL_MAX debe seguir vivo"
    assert "MEALFIT_WATERY_VEG_KCAL_MAX" in _GO


def test_marker_estructural_anclado_en_fuente():
    assert "P2-VEG-VOLUME-TOKENS-2" in _GO
    assert "_watery_veg_tokens" in _GO
    assert "8d3f246a" in _GO, "el plan vivo que motivó el refactor debe quedar anclado en el código"


# ═══════════════ Sección 4 — P2-MUTATOR-PURITY: warm-up fuera de locks ═══════════════
#
# Review CRÍTICO post-Sección 3: `_watery_veg_tokens()` en cache-miss llama
# `get_master_ingredients()` → `execute_sql_query` → re-entrada al pool. Ese código corre DENTRO
# de `_swap_mutator` (routers/plans.py, vía `finalize_single_meal_recipe_coherence` →
# `_cap_unrealistic_portions`), que `update_plan_data_atomic` (db_plans.py) ejecuta reteniendo
# row-lock + una conexión del pool sync — violación del contrato P2-MUTATOR-PURITY
# (db_plans.py:564: el mutator DEBE ser CPU-only, "nada de ... re-entrada al pool"). Ventana
# real: solo el PRIMER swap tras cada restart (luego siempre cache-hit) — pero el contrato es
# absoluto, no "casi siempre".
#
# Fix en 2 capas: (1) warm-up SÍNCRONO en `app.py::lifespan`, fuera de CUALQUIER lock y antes
# de servir tráfico — con esto ningún mutator ve cache-miss en el camino feliz; (2) hoist
# defensivo en `routers/plans.py`, justo antes de `update_plan_data_atomic(plan_id,
# _swap_mutator, ...)` — cinturón por si el warm-up de arranque falló (DB caída al boot, etc).
# `_watery_veg_tokens()` sigue siendo fail-open por su propio diseño (Sección 3) — esto es una
# capa ADICIONAL que evita pagar el costo del miss DENTRO de un lock, no un reemplazo.


# ───────────── (a) parser: el hoist vive ANTES de la llamada atómica del swap ─────────────

def test_a_parser_warmup_hoist_antes_del_update_plan_data_atomic_swap():
    """En routers/plans.py, la llamada de warm-up debe aparecer ANTES (offset menor) que
    `update_plan_data_atomic(plan_id, _swap_mutator, ...)` en el mismo handler — si alguien
    reordena el hoist DESPUÉS de la llamada atómica, deja de proteger nada."""
    warmup_idx = _PLANS.find("_warm_watery_veg_tokens_sw()")
    assert warmup_idx != -1, "falta la llamada de warm-up hoisteada en routers/plans.py"

    # OJO: hay una mención de UNA LÍNEA en el docstring del handler (~6042, entre backticks,
    # "update_plan_data_atomic(plan_id, _swap_mutator, user_id=...)") que NO es la llamada real
    # — el regex exige salto de línea inmediatamente tras el "(" (la forma multi-línea real,
    # `update_plan_data_atomic(\n    plan_id,\n    _swap_mutator,`), así que no matchea el
    # docstring de una sola línea.
    m = re.search(r"update_plan_data_atomic\(\n\s*plan_id,\n\s*_swap_mutator,", _PLANS)
    assert m is not None, (
        "no encontré la llamada real (multi-línea) a update_plan_data_atomic(plan_id, "
        "_swap_mutator, ...) en routers/plans.py (¿cambió de forma?)")
    mutator_call_idx = m.start()

    assert warmup_idx < mutator_call_idx, (
        f"el warm-up (offset {warmup_idx}) debe estar ANTES de "
        f"update_plan_data_atomic(plan_id, _swap_mutator, ...) (offset {mutator_call_idx}) — "
        f"si no, el mutator puede seguir viendo cache-miss dentro del lock")


def test_a_parser_warmup_hoist_tiene_el_marker_y_la_razon():
    assert "P2-VEG-VOLUME-TOKENS-2" in _PLANS
    assert "P2-MUTATOR-PURITY" in _PLANS
    assert "_watery_veg_tokens" in _PLANS


# ───────────── (b) parser: el startup hook calienta el cache con el marker ─────────────

def test_b_parser_startup_hook_calienta_watery_veg_tokens():
    assert "async def lifespan" in _APP, "no encontré lifespan() en app.py — ¿se movió?"
    lifespan_idx = _APP.find("async def lifespan")
    warm_idx = _APP.find("_warm_watery_veg_tokens")
    assert warm_idx != -1, "falta el warm-up de _watery_veg_tokens en app.py"
    assert warm_idx > lifespan_idx, "el warm-up debe vivir DENTRO de lifespan(), no antes"
    assert "P2-VEG-VOLUME-TOKENS-2" in _APP
    assert "P2-MUTATOR-PURITY" in _APP


def test_b_parser_startup_hook_es_best_effort_try_except():
    """El warm-up de arranque debe ser fail-open — un fallo NO puede abortar el startup."""
    warm_block_start = _APP.find("_watery_veg_tokens as _warm_watery_veg_tokens")
    assert warm_block_start != -1
    preceding = _APP[max(0, warm_block_start - 200):warm_block_start]
    assert "try:" in preceding, (
        "el warm-up de arranque debe estar envuelto en try/except (best-effort) — un fallo de "
        "DB al boot no puede tumbar el startup completo")


# ───────────── (c) funcional: con cache caliente, el mutator NO toca DB ─────────────

def test_c_mutator_con_cache_caliente_no_toca_db(monkeypatch):
    """Con el cache YA poblado (primera llamada exitosa, lo que hace el warm-up de arranque o
    el hoist), una 2ª llamada NO debe invocar `get_master_ingredients()` — prueba que cualquier
    código que corra DENTRO del mutator (bajo FOR UPDATE) sirve cache-hit puro."""
    _llamadas = {"n": 0}

    def _catalogo_ok():
        _llamadas["n"] += 1
        return _SyntheticCatalog.get()

    monkeypatch.setattr(sc, "get_master_ingredients", _catalogo_ok)

    # Warm-up: SÍ toca DB (una vez).
    toks1 = g._watery_veg_tokens()
    assert _llamadas["n"] == 1
    assert "tayota" in toks1

    # Simula la DB cayéndose — si el mutator re-entrara al pool pese al warm-up, esto explota.
    def _catalogo_explota():
        raise AssertionError(
            "el mutator re-entró al pool pese al warm-up — violación de P2-MUTATOR-PURITY")

    monkeypatch.setattr(sc, "get_master_ingredients", _catalogo_explota)

    # Exactamente lo que hace `_cap_unrealistic_portions` dentro de `_swap_mutator`.
    for _ in range(3):
        assert g._watery_veg_tokens() == toks1

    assert _llamadas["n"] == 1, (
        "get_master_ingredients() se llamó de nuevo pese al cache caliente — el mutator NO "
        "es puro bajo esta condición")


def test_c_funcional_cap_unrealistic_portions_con_cache_caliente_no_toca_db(monkeypatch):
    """Mismo contrato, pero ejercitando la función REAL que corre dentro del mutator
    (`_cap_unrealistic_portions`) en vez de llamar `_watery_veg_tokens()` directo — cierra la
    brecha entre 'el helper es puro' y 'el call site real también lo es'."""
    monkeypatch.setattr(sc, "get_master_ingredients", _SyntheticCatalog.get)
    g._watery_veg_tokens()  # warm-up simulado (lo que haría app.py lifespan / el hoist)

    def _catalogo_explota():
        raise AssertionError("_cap_unrealistic_portions re-entró al pool con cache caliente")

    monkeypatch.setattr(sc, "get_master_ingredients", _catalogo_explota)

    meal = {"name": "Test", "meal": "Cena", "ingredients": ["470 g de tayota"],
            "ingredients_raw": ["470 g de tayota"],
            "protein": 24, "carbs": 62, "fats": 13, "cals": 464}
    n = g._cap_unrealistic_portions([{"meals": [meal]}], db=_DB3())
    assert n == 1
    assert meal["ingredients"][0].startswith("250"), meal["ingredients"][0]
