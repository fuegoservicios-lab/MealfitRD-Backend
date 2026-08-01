"""[P1-CULINARY-CONTRACT · 2026-07-31] Capa determinista de coherencia culinaria
(F1 del spec): migración de metadata + scan V1/V2/V3 + 3 superficies en warn."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_MIG = _BACKEND / "migrations" / "p1_culinary_metadata_master_ingredients_2026_07_31.sql"
_MIG_ROOT = _BACKEND.parent / "migrations" / _MIG.name


def test_migracion_existe_en_ambos_dirs_identica():
    assert _MIG.exists(), "falta la migración en backend/migrations/"
    assert _MIG_ROOT.exists(), "falta la copia en migrations/ (P3-MIGRATIONS-SSOT)"
    assert _MIG.read_bytes() == _MIG_ROOT.read_bytes(), "las dos copias divergen"


def test_migracion_idempotente_y_con_sanity():
    sql = _MIG.read_text(encoding="utf-8")
    assert sql.count("ADD COLUMN IF NOT EXISTS") >= 2
    assert "prep_methods text[]" in sql and "ready_to_eat boolean" in sql
    assert "DO $$" in sql, "falta el bloque sanity DO $$"
    # Regex (no substring literal): las filas de Backfill 1 combinan la categoría con
    # el filtro IS NULL en la misma cláusula WHERE ("WHERE category = 'X' AND
    # prep_methods IS NULL"), así que la subcadena "WHERE prep_methods IS NULL" sola
    # nunca aparece verbatim aunque el filtro SÍ esté aplicado.
    assert re.search(r"WHERE\b.*prep_methods IS NULL", sql), (
        "el backfill debe filtrar IS NULL para ser re-ejecutable sin pisar "
        "overrides manuales posteriores")


# ---------------------------------------------------------------------------
# [P1-CULINARY-CONTRACT · Task 4] Matching + V1 (verbo↔alimento). Módulo puro
# `culinary_coherence.py` — sin env vars, sin LLM, sin DB.
# ---------------------------------------------------------------------------
import culinary_coherence as cc

_CAT = [
    {"name": "Casabe", "prep_methods": ["tostar", "ninguno"], "ready_to_eat": True},
    {"name": "Pechuga de pollo", "prep_methods": ["hervir", "plancha", "freir", "hornear", "guisar", "saltear"], "ready_to_eat": False},
    {"name": "Repollo", "prep_methods": ["hervir", "saltear", "crudo"], "ready_to_eat": None},
    {"name": "Pollo guisado", "prep_methods": ["guisar"], "ready_to_eat": False},
    {"name": "Tomate", "prep_methods": ["crudo", "saltear"], "ready_to_eat": True},
    {"name": "Bistec de res", "prep_methods": ["plancha", "guisar"], "ready_to_eat": False},
    {"name": "Misterio sin metadata", "prep_methods": None, "ready_to_eat": None},
]


def _plan(pasos, ingredientes=None, nombre="Plato de prueba"):
    return {"days": [{"day": 1, "meals": [{
        "meal": "Almuerzo", "name": nombre,
        "ingredients": ingredientes or ["100 g Pechuga de pollo"],
        "recipe": pasos}]}]}


def test_v1_verbo_imposible_sobre_ready_to_eat():
    v = cc.culinary_contract_scan(_plan(["Cuece el Casabe según el paquete."],
                                        ["30 g Casabe"]), _CAT)
    assert any(x["check"] == "V1" and x["food"] == "Casabe" for x in v), v


def test_v1_metodo_fuera_de_prep_methods():
    v = cc.culinary_contract_scan(_plan(["Licúa el Bistec de res hasta obtener crema."],
                                        ["120 g Bistec de res"]), _CAT)
    assert any(x["check"] == "V1" and x["food"] == "Bistec de res" for x in v), v


def test_v1_no_dispara_sobre_metodo_valido():
    v = cc.culinary_contract_scan(_plan(["Cocina la Pechuga de pollo a la plancha."]), _CAT)
    assert not [x for x in v if x["check"] == "V1"], v


def test_v1_word_boundary_pollo_no_es_repollo():
    """pollo⊂repollo: 'Saltea el Repollo' NO debe leerse como pollo."""
    v = cc.culinary_contract_scan(_plan(["Saltea el Repollo con ajo."], ["80 g Repollo"]), _CAT)
    assert not [x for x in v if x["check"] == "V1"], v


def test_v1_alias_mas_largo_gana():
    """'Pollo guisado' (plato del catálogo) menciona 'guisa' — el match debe ser
    el alias LARGO, no 'pollo' suelto con verbo guisar."""
    v = cc.culinary_contract_scan(_plan(["Guisa el Pollo guisado a fuego lento."],
                                        ["150 g Pollo guisado"]), _CAT)
    assert not [x for x in v if x["check"] == "V1"], v


def test_v1_fail_open_sin_metadata():
    v = cc.culinary_contract_scan(_plan(["Hornea el Misterio sin metadata 20 min."],
                                        ["50 g Misterio sin metadata"]), _CAT)
    assert not v, "sin metadata el scan debe callar (fail-open), no inventar"


def test_plural_bidireccional():
    """FP real del dry-run: '2 tomates' (ingrediente) vs 'el tomate' (paso)."""
    assert cc.find_catalog_foods("Ralla el tomate encima.", cc.build_culinary_index(_CAT)) == ["Tomate"]
    assert cc.find_catalog_foods("2½ tomates maduros", cc.build_culinary_index(_CAT)) == ["Tomate"]


# --- Resoluciones de ambigüedad del controller (task-4-brief.md), ganan sobre
# la letra del brief. Ver comentarios inline en culinary_coherence.py. ---

def test_v1_sofreir_no_es_freir_permite_saltear():
    """RESOLUCIÓN 1: 'sofr[ií]\\w*' vive bajo 'saltear', NO 'freir'. Tomate
    acepta 'saltear' pero no 'freir' — si sofreír colgara de freir, este paso
    dispararía un V1 falso-positivo sobre una técnica legítima RD."""
    v = cc.culinary_contract_scan(_plan(["Sofríe el Tomate picado."],
                                        ["50 g Tomate"]), _CAT)
    assert not [x for x in v if x["check"] == "V1"], v


def test_v1_multi_alimento_no_acusa_acompanantes():
    """RESOLUCIÓN 2: 'Hierve el Repollo y sirve con Casabe' — Repollo SÍ acepta
    hervir; Casabe (ready-to-eat) NO. Con ≥2 alimentos y ≥1 destinatario válido,
    el check no acusa al acompañante Casabe."""
    v = cc.culinary_contract_scan(_plan(["Hierve el Repollo y sirve con Casabe."],
                                        ["80 g Repollo", "30 g Casabe"]), _CAT)
    assert not [x for x in v if x["check"] == "V1"], v


def test_v1_multi_alimento_sin_destinatario_valido_acusa_a_todos():
    """RESOLUCIÓN 2, contraste: si NINGÚN alimento del paso acepta el método,
    la salvaguarda no aplica — ambos deben acusarse (no hay 'destinatario
    válido' que la lea como un paso legítimo con acompañante inocente)."""
    v = cc.culinary_contract_scan(_plan(["Licúa el Casabe y el Bistec de res."],
                                        ["30 g Casabe", "120 g Bistec de res"]), _CAT)
    v1_foods = {x["food"] for x in v if x["check"] == "V1"}
    assert v1_foods == {"Casabe", "Bistec de res"}, v


def test_v1_sofr_y_saltea_en_el_mismo_paso_no_duplica_violacion():
    """[Fix post-review] 'sofr[ií]\\w*' y 'saltea\\w*' resuelven al MISMO
    método ('saltear'). Un paso que dispara ambos fragmentos regex
    ('Sofríe y saltea...') debe producir UNA sola violación V1 por alimento,
    no dos idénticas — la dedup vive en la alternancia fusionada de
    VERB_TO_METHOD Y en `dict.fromkeys` al construir `metodos`."""
    v = cc.culinary_contract_scan(_plan(["Sofríe y saltea el Bistec de res."],
                                        ["120 g Bistec de res"]), _CAT)
    v1 = [x for x in v if x["check"] == "V1"]
    assert len(v1) == 1, v1


# ---------------------------------------------------------------------------
# [P1-CULINARY-CONTRACT · Task 5] V2 (estado imposible) + V3 (huérfanos).
# ---------------------------------------------------------------------------

def test_v2_ya_viene_cocido_sobre_fresco():
    v = cc.culinary_contract_scan(_plan(
        ["Sirve la Pechuga de pollo (ya viene cocida) sobre la ensalada."]), _CAT)
    hit = [x for x in v if x["check"] == "V2"]
    assert hit and hit[0]["severity"] == "high", v


def test_v2_legal_sobre_ready_to_eat():
    v = cc.culinary_contract_scan(_plan(["Sirve el Casabe (ya viene cocido)."],
                                        ["30 g Casabe"]), _CAT)
    assert not [x for x in v if x["check"] == "V2"], v


def test_v3_huerfano_resoluble_al_catalogo():
    v = cc.culinary_contract_scan(_plan(
        ["Cocina la Pechuga de pollo a la plancha y sirve."],
        ["120 g Pechuga de pollo", "80 g Repollo"]), _CAT)   # Repollo sin paso
    hit = [x for x in v if x["check"] == "V3"]
    assert hit and hit[0]["food"] == "Repollo" and hit[0]["repairable"] is True, v


def test_v3_ignora_condimentos_y_no_resolubles():
    """'picados' (participio) no matchea catálogo ⇒ jamás huérfano (el FP del
    plan real 43ec0576); condimentos exentos por lista canónica."""
    v = cc.culinary_contract_scan(_plan(
        ["Cocina la Pechuga de pollo a la plancha."],
        ["120 g Pechuga de pollo", "tomates picados", "1 pizca de sal", "aceite de oliva"]), _CAT)
    hit = [x for x in v if x["check"] == "V3"]
    assert [x["food"] for x in hit] == ["Tomate"], (
        f"solo Tomate (resoluble, sin paso) debe ser huérfano — nunca 'picados' "
        f"ni condimentos: {hit}")


def test_v3_plural_singular_no_da_fp():
    v = cc.culinary_contract_scan(_plan(
        ["Ralla el tomate encima."], ["2 tomates", "120 g Pechuga de pollo",
                                      "Cocina la pechuga"]), _CAT)
    assert not [x for x in v if x["check"] == "V3" and x["food"] == "Tomate"], v


def test_scan_coverage_fraccion_con_y_sin_metadata():
    """[Task-5, gap señalado en el review de T4] scan_coverage con _CAT: un
    plan que menciona 2 alimentos CON metadata (Pechuga de pollo, Tomate) y 1
    SIN metadata (Misterio sin metadata) ⇒ coverage == 2/3."""
    plan = _plan(["Cocina la Pechuga de pollo y sirve con Tomate."],
                 ["120 g Pechuga de pollo", "50 g Tomate", "10 g Misterio sin metadata"])
    cov = cc.scan_coverage(plan, _CAT)
    assert cov == pytest.approx(2 / 3), cov


# [Fix post-review T5] red DB-independiente: los tests golden skipean sin pool
# (CI no tiene NEON_DATABASE_URL). Los dos fixes que el examen golden encontró
# (dora≠tostar, guard de ambigüedad de _mencionado_por_prefijo) solo estaban
# anclados por esos tests golden-with-DB — si alguien los rompe, CI queda
# verde. Estos dos tests anclan el MISMO comportamiento con catálogo sintético
# local, sin DB.

def test_v1_dora_no_es_tostar():
    """[Fix post-review T5] red DB-independiente: los tests golden skipean sin
    pool (CI no tiene NEON_DATABASE_URL). Ancla el fix del examen golden (FP
    real Carne de res/Pechuga de pollo con "...y dora."): 'dora\\w*' vive bajo
    'saltear', NO bajo 'tostar' — Pechuga de pollo acepta saltear pero no
    tostar en _CAT."""
    v = cc.culinary_contract_scan(_plan(["Dora la Pechuga de pollo en la sartén."]), _CAT)
    assert not [x for x in v if x["check"] == "V1"], v

    # Protección inversa: 'tostar' sigue detectándose como método propio
    # (dora no absorbió/anuló la rama tostar\w*|tuesta\w*).
    v2 = cc.culinary_contract_scan(_plan(["Tuesta la Pechuga de pollo."]), _CAT)
    v1_hits = [x for x in v2 if x["check"] == "V1"]
    assert v1_hits and v1_hits[0]["food"] == "Pechuga de pollo", v2


def test_v3_prefijo_ambiguo_no_enmascara():
    """[Fix post-review T5] red DB-independiente: los tests golden skipean sin
    pool (CI no tiene NEON_DATABASE_URL). Ancla el mecanismo
    `_mencionado_por_prefijo`: 'Ají morrón' y 'Ají cubanela' comparten la
    cabeza 'ají'. Si los pasos solo mencionan 'ají cubanela', el guard de
    ambigüedad NO debe dejar que ese match enmascare a 'Ají morrón' como
    huérfano (el bug real que el examen golden habría reproducido con
    'Ají morrón' inyectado junto a 'Ají cubanela' en la misma comida)."""
    cat_aji = _CAT + [
        {"name": "Ají morrón", "prep_methods": ["hervir", "saltear", "guisar"], "ready_to_eat": None},
        {"name": "Ají cubanela", "prep_methods": ["hervir", "saltear", "guisar"], "ready_to_eat": None},
    ]
    v = cc.culinary_contract_scan(_plan(
        ["Sofríe el ají cubanela."],
        ["80 g Ají morrón", "50 g Ají cubanela"]), cat_aji)
    hit = [x for x in v if x["check"] == "V3"]
    assert hit and hit[0]["food"] == "Ají morrón", (
        f"'Ají morrón' debe reportarse huérfano — el guard de ambigüedad no "
        f"debe dejar que 'ají cubanela' en pasos lo enmascare: {hit}")


# ---------------------------------------------------------------------------
# [P1-CULINARY-CONTRACT · Task 6] Superficie 1 — gate en `review_plan_node`
# (warn default). Parser-based sobre el SOURCE de graph_orchestrator.py: el
# módulo real no se importa aquí para no arrastrar sus dependencias de
# import-time al test suite del culinary contract.
# ---------------------------------------------------------------------------

_GO_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def test_knob_guard_nace_en_warn():
    m = re.search(r'CULINARY_CONTRACT_GUARD = \(_env_str\("MEALFIT_CULINARY_CONTRACT_GUARD", "(\w+)"\)', _GO_SRC)
    assert m, "falta el knob MEALFIT_CULINARY_CONTRACT_GUARD registrado vía _env_str"
    assert m.group(1) == "warn", (
        "F1 nace en warn (mismo camino warn→block que MEALFIT_SHOPPING_COHERENCE_GUARD); "
        "la escalada a block es P1-CULINARY-CONTRACT-BLOCK (F2) con su propio test")


def test_review_invoca_el_scan_y_traduce_a_issues():
    i = _GO_SRC.index("async def review_plan_node")
    # [Task-6, fix sobre el brief] `review_plan_node` mide ~109k chars (medido
    # 2026-07-31) — muy por encima de una ventana fija de 60k (la del brief
    # original, calibrada sin medir la función real). Se acota al límite REAL
    # de la función — el próximo `def`/`async def` a nivel de módulo (columna
    # 0) — en vez de un número inventado que quedaría corto si la función
    # sigue creciendo.
    _next_def = re.search(r"\n(?:async )?def ", _GO_SRC[i + 30:])
    end = (i + 30 + _next_def.start()) if _next_def else len(_GO_SRC)
    win = _GO_SRC[i:end]
    assert "culinary_contract_scan(" in win, "review_plan_node no invoca el scan"
    j = win.index("culinary_contract_scan(")
    gate = win[j:j + 3000]
    assert 'CULINARY_CONTRACT_GUARD == "block"' in gate
    assert "_severity_max(" in gate, "en block las violaciones deben escalar severity"
    assert "issues.append" in gate
    assert "scan_coverage(" in win, "falta la telemetría de cobertura"


def test_scan_corre_despues_de_los_reparadores():
    """Orden reparar→medir (dueño único): el scan corre DESPUÉS del AUTO-PATCH
    de huérfanos en review. Ancla: el callsite del scan aparece en el archivo
    DESPUÉS del marker del auto-patch."""
    autopatch = _GO_SRC.index("huérfanos eliminados") if "huérfanos eliminados" in _GO_SRC \
        else _GO_SRC.index("AUTO-PATCH")
    assert _GO_SRC.index("culinary_contract_scan(") > autopatch


# ---------------------------------------------------------------------------
# [P1-CULINARY-CONTRACT · Task 7] Superficie 2 — paridad de
# `_fix_refill_step_verb` en `finalize_plan_data_coherence`. Parser-based
# sobre el SOURCE (mismo motivo que Superficie 1: no importar el módulo aquí).
# ---------------------------------------------------------------------------

def test_fix_refill_step_verb_corre_en_finalize():
    """Gap de paridad (spec §4c): _fix_refill_step_verb hoy solo corría en
    assemble; los paths que saltan assemble (partial/degradado/chunk) persistían
    sin él. Ancla: su llamada debe existir DENTRO de finalize_plan_data_coherence.

    [Task-7, fix sobre el brief] La ventana fija de 20000 chars del brief original
    no alcanza: `finalize_plan_data_coherence` mide ~42000 chars (medido 2026-07-31,
    ver el bloque P1-FINALIZE-TAIL-PARITY/gainmuscle-refill a mitad de función).
    Mismo patrón ventana-dinámica que `test_review_invoca_el_scan_y_traduce_a_issues`
    (Superficie 1, arriba): se acota al límite REAL de la función — el próximo
    `def`/`async def` a nivel de módulo — en vez de un número inventado que
    quedaría corto si la función sigue creciendo."""
    i = _GO_SRC.index("def finalize_plan_data_coherence")
    _next_def = re.search(r"\n(?:async )?def ", _GO_SRC[i + 30:])
    end = (i + 30 + _next_def.start()) if _next_def else len(_GO_SRC)
    fin = _GO_SRC[i:end]
    assert "_fix_refill_step_verb(" in fin, (
        "finalize_plan_data_coherence no invoca _fix_refill_step_verb — los "
        "planes del path degradado/chunk siguen con '🍚 Cuece el Casabe'")


# ---------------------------------------------------------------------------
# [P1-CULINARY-CONTRACT · Task 8] Superficie 3 — path degradado
# (`cron_tasks._build_filtered_edge_recipe_day`). Es la ÚNICA capa posible
# ahí: por construcción no hay LLM en este path (patrón
# P0-DEGRADED-SAFETY-SCAN), así que review_plan_node/assemble_plan_node nunca
# corren sobre lo que sale de aquí. Parser-based sobre el SOURCE de
# cron_tasks.py (mismo motivo que Superficies 1/2: no importar el módulo aquí
# para no arrastrar sus dependencias de import-time — DB pool, scheduler,
# etc. — al test suite del culinary contract).
# ---------------------------------------------------------------------------

_CRON_SRC = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")


def test_degradado_pasa_por_el_scan():
    """Única capa posible en el path degradado (no hay LLM ahí por construcción,
    patrón P0-DEGRADED-SAFETY-SCAN). Cada edge_day se escanea; violación ⇒
    degradar el paso a 'Sirve {food}' (jamás entregar el verbo imposible)."""
    i = _CRON_SRC.index("def _build_filtered_edge_recipe_day")
    win = _CRON_SRC[i:i + 15000]
    assert "culinary_contract_scan(" in win, "el edge day no se escanea"


def test_placeholder_generico_reemplazado():
    """El placeholder genérico ("según método tradicional") no debe quedar
    vivo en ninguna forma — ni en código ni en comentarios — porque invita a
    reañadirlo citando el propio comentario. Verificado por grep manual antes
    de escribir este assert: la única ocurrencia histórica era la línea que
    esta tarea reemplaza; no hay otra legítima que excluir."""
    assert "según método tradicional" not in _CRON_SRC, (
        "el placeholder genérico sigue vivo — debe generarse el verbo real "
        "desde prep_methods (spec §4c bonus)")


def test_step_has_cooking_verb_exportado_y_correcto():
    """[Fix post-review, controller Resolución 1] `culinary_coherence.py`
    exporta `step_has_cooking_verb` en vez de forzar al path degradado a
    alcanzar `_VERB_RES` (privado del módulo) vía `__import__`. Ancla
    funcional DB-independiente: True sobre un paso con verbo de cocción
    reconocido, False sobre un paso que ya es 'Sirve' (el destino de la
    degradación, para que no vuelva a degradarse una segunda vez)."""
    assert cc.step_has_cooking_verb("Hierve el arroz.") is True
    assert cc.step_has_cooking_verb("Sirve el casabe.") is False


# ---------------------------------------------------------------------------
# [P1-CULINARY-CONTRACT · Task 8, fix post-review] `_degrade_offending_steps`
# — helper privado extraído del bloque de degradación tras 2 hallazgos
# Important del reviewer sobre el pseudocódigo del brief (que contradecía las
# Global Constraints del plan; la constraint gana). Import LOCAL de
# `cron_tasks` dentro de cada test (no a nivel de módulo, mismo patrón que
# `tests/test_p0_degraded_safety_scan.py`): el resto de este archivo es
# deliberadamente parser-based/puro para no arrastrar las dependencias de
# import-time de `cron_tasks` (scheduler, DB pool) a la colección completa.
# ---------------------------------------------------------------------------

def test_degrade_offending_steps_matcher_canonico_no_substring():
    """[Fix post-review, Important #1] El bloque original usaba
    `food.lower() in paso.lower()` — substring plano, la MISMA clase de bug
    que P3-PANTRY-INTERSECT-WB ya cerró 150 líneas arriba en este mismo
    archivo (`"sal"⊂"salsa"`). Aquí el caso concreto: `"Sal"⊂"Saltea"`. Una
    violación V1 sobre el alimento "Sal" NO debe degradar un paso "Saltea los
    vegetales." que no lo menciona según el matcher canónico
    (`find_catalog_foods`, word-boundary) — aunque `step_has_cooking_verb`
    para ese paso sea True. Contraste positivo: una violación real (Casabe
    horneado, imposible — Casabe solo acepta tostar/ninguno) SÍ degrada, y
    preserva el prefijo de sección ("El Toque de Fuego: ") que RecipesView
    usa para agrupar pasos."""
    import cron_tasks as ct

    catalog = [
        {"name": "Casabe", "prep_methods": ["tostar", "ninguno"], "ready_to_eat": True},
        {"name": "Sal", "prep_methods": ["ninguno"], "ready_to_eat": True},
    ]
    index = cc.build_culinary_index(catalog)

    edge_day = {"meals": [
        {"meal": "Desayuno", "recipe": [
            "Mise en place: Prepara y mide los ingredientes.",
            "El Toque de Fuego: Hornea el Casabe.",
            "Montaje: Sirve junto al acompañante y disfruta.",
        ]},
        {"meal": "Almuerzo", "recipe": [
            "El Toque de Fuego: Saltea los vegetales.",
        ]},
    ]}
    violaciones = [
        {"check": "V1", "food": "Casabe", "meal": "Desayuno"},
        {"check": "V1", "food": "Sal", "meal": "Almuerzo"},
    ]

    n = ct._degrade_offending_steps(edge_day, violaciones, index)

    assert edge_day["meals"][0]["recipe"] == [
        "Mise en place: Prepara y mide los ingredientes.",
        "El Toque de Fuego: Sirve el Casabe.",
        "Montaje: Sirve junto al acompañante y disfruta.",
    ], edge_day["meals"][0]["recipe"]
    assert edge_day["meals"][1]["recipe"] == ["El Toque de Fuego: Saltea los vegetales."], (
        "'Sal' NO debe emparejar con 'Saltea' vía substring — el matcher "
        f"canónico no lo encuentra ahí: {edge_day['meals'][1]['recipe']}")
    assert n == 1, f"debe degradar exactamente 1 paso (Casabe), no {n}"


def test_degrade_offending_steps_acota_por_meal():
    """[Fix post-review, Important #2] El bloque original ignoraba
    `_v["meal"]` y barría TODAS las comidas del edge_day por cada violación
    — una violación en Desayuno degradaba pasos de Cena que ni siquiera
    generaron esa violación. Dos comidas con el MISMO paso ofensor
    literal; la violación solo lista una — solo esa debe degradarse."""
    import cron_tasks as ct

    catalog = [{"name": "Casabe", "prep_methods": ["tostar", "ninguno"], "ready_to_eat": True}]
    index = cc.build_culinary_index(catalog)

    edge_day = {"meals": [
        {"meal": "Desayuno", "recipe": ["El Toque de Fuego: Hornea el Casabe."]},
        {"meal": "Cena", "recipe": ["El Toque de Fuego: Hornea el Casabe."]},
    ]}
    violaciones = [{"check": "V1", "food": "Casabe", "meal": "Desayuno"}]

    n = ct._degrade_offending_steps(edge_day, violaciones, index)

    assert edge_day["meals"][0]["recipe"][0] == "El Toque de Fuego: Sirve el Casabe."
    assert edge_day["meals"][1]["recipe"][0] == "El Toque de Fuego: Hornea el Casabe.", (
        "la violación de Desayuno NO debe degradar el paso idéntico de Cena")
    assert n == 1
