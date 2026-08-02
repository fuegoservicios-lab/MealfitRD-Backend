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


# ---------------------------------------------------------------------------
# [IMPORTANT-5 · post-review-final] La exención V3 de condimentos matchea por TOKEN
# con word-boundary, no substring plano. Antes del fix, `"sal" in n` exentaba
# CUALQUIER ingrediente que contuviera "sal"/"agua" como substring —
# "Salami"/"Salmón"/"EnSALada"/"Aguacate" quedaban falsamente exentos de V3 (14ª
# aparición documentada de esta clase de bug en el repo).
# ---------------------------------------------------------------------------
def test_v3_condimento_exemption_no_es_substring_plano():
    """Salami/Aguacate/Ensalada verde SON huérfanos resolubles al catálogo — la
    exención de condimentos NO debe enmascararlos por contener 'sal'/'agua' como
    substring dentro de otra palabra."""
    cat = _CAT + [
        {"name": "Salami", "prep_methods": ["freir"], "ready_to_eat": True},
        {"name": "Aguacate", "prep_methods": ["crudo"], "ready_to_eat": True},
        {"name": "Ensalada verde", "prep_methods": ["crudo"], "ready_to_eat": True},
    ]
    v = cc.culinary_contract_scan(_plan(
        ["Cocina la Pechuga de pollo a la plancha y sirve."],
        ["120 g Pechuga de pollo", "80 g Salami", "1 Aguacate", "1 Ensalada verde"]), cat)
    hit_foods = {x["food"] for x in v if x["check"] == "V3"}
    assert hit_foods == {"Salami", "Aguacate", "Ensalada verde"}, (
        f"Salami/Aguacate/Ensalada verde son huérfanos resolubles al catálogo — NO deben "
        f"quedar exentos por contener 'sal'/'agua' como substring: {v}")


def test_v3_condimento_exemption_legitima_multipalabra_y_plural():
    """Las exenciones legítimas (incluida la multi-palabra 'ajo en polvo' y el
    plural 'sales'/'especias') SIGUEN exentas bajo el matching por token."""
    v = cc.culinary_contract_scan(_plan(
        ["Cocina la Pechuga de pollo a la plancha."],
        ["120 g Pechuga de pollo", "1 pizca de sal", "aceite de oliva",
         "1 cdta ajo en polvo", "sales de mar", "especias variadas"]), _CAT)
    assert not [x for x in v if x["check"] == "V3"], v


def test_v3_condimento_exemption_comino():
    """[P1-CULINARY-CONTRACT] Comino listado como condimento exento — no debe
    disparar V3 huérfano aunque ningún paso lo mencione (plan real 9bb38a86)."""
    v = cc.culinary_contract_scan(_plan(
        ["Cocina la Pechuga de pollo a la plancha."],
        ["120 g Pechuga de pollo", "1 cdta comino"]), _CAT)
    assert not [x for x in v if x["check"] == "V3"], v


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


# ---------------------------------------------------------------------------
# [P1-CULINARY-CONTRACT-FP1 · 2026-08-01] FP reales 2026-08-01 plan 165dd761 —
# primer plan real en producción (fase warn), 9/9 violaciones V1 eran falsos
# positivos de 2 mecanismos: Clase A — participio/adjetival capturado por el
# `\w*` genérico tras la raíz del verbo (6/9, TODAS en pasos de "Montaje:"):
# «el yaniqueque HORNEADO», «pollo desmechado SALTEADO», «las TOSTADAS» ×2.
# Clase B — atribución cruzada (3/9): el verbo apuntaba a un alimento
# no-catalogado ("almendras") y, sin destinatario válido, la salvaguarda
# multi-alimento acusaba a los acompañantes catalogados (avena/leche/clara)
# que sí estaban en el paso pero NO eran el objeto real del verbo.
# Fix: negative-lookahead de participio en VERB_TO_METHOD (hornear/guisar/
# saltear-sofreír-dorar/licuar/tostar) + skip de pasos "Montaje:" en V1
# (nunca cocina) + veto de ventana post-verbo cuando NINGÚN alimento del
# paso acepta el método. [P1-CULINARY-CONTRACT-FP2 · 2026-08-01] La ventana
# post-verbo, originalmente un conteo fijo de 4 palabras, se reemplazó por
# escaneo hasta la FRONTERA DE ORACIÓN (siguiente '.'/';' o fin del paso) —
# el conteo fijo perdía recall sobre el patrón "verbo + relleno de tiempo/
# fuego mandatado por el prompt SSOT + objeto" (ver sección FP reales round
# 2 más abajo). Traza completa por fragmento de verbo + ronda 2:
# `.superpowers/culinary-fp-round1-report.md`.
# ---------------------------------------------------------------------------

_CAT_FP1 = _CAT + [
    {"name": "Yaniqueque", "prep_methods": ["hornear", "freir"], "ready_to_eat": True},
    {"name": "Avena", "prep_methods": ["hervir", "ninguno"], "ready_to_eat": None},
    {"name": "Leche", "prep_methods": ["ninguno", "crudo", "licuar", "hervir"], "ready_to_eat": True},
]


def test_fp1_step_has_cooking_verb_no_lee_participios():
    """Aísla el mecanismo del participio (Clase A) vía la función pública
    `step_has_cooking_verb` — independiente del skip de Montaje y de la
    ventana post-verbo (que viven en `_v1_verbo_alimento`, no en el regex).
    Las 4 formas participiales de los 9 FPs reales + el resto del
    vocabulario con participio en español NO deben leerse como verbo de
    cocción; los imperativos/gerundios reales SIGUEN detectándose."""
    participios_no_deben_matchear = [
        "el yaniqueque horneado", "pollo desmechado salteado con los vegetales",
        "coloca el huevo sobre las tostadas", "almendras tostadas",
        "pollo guisado listo para servir", "el sofrito de la olla",
        "pollo dorado", "un licuado de frutas",
    ]
    for texto in participios_no_deben_matchear:
        assert cc.step_has_cooking_verb(texto) is False, texto

    imperativos_deben_matchear = [
        "Hornea el pollo", "Saltea los vegetales", "Sofríe la cebolla",
        "Dora la carne", "Tuesta el pan", "Licua las frutas",
        "Guisa el pollo", "Hierve el agua", "Cuece el arroz",
    ]
    for texto in imperativos_deben_matchear:
        assert cc.step_has_cooking_verb(texto) is True, texto


def test_fp1_clase_a_montaje_con_participio_no_dispara_v1():
    """(a) FP real literal: 'Montaje: Sirve el revoltillo sobre el
    yaniqueque horneado y el Casabe.' NO produce V1 — doble defensa
    (participio 'horneado' ya excluido del regex de hornear, Y el paso
    entero es 'Montaje:' así que V1 ni siquiera lo evalúa)."""
    v = cc.culinary_contract_scan(_plan(
        ["Montaje: Sirve el revoltillo sobre el yaniqueque horneado y el Casabe."],
        ["1 Yaniqueque", "30 g Casabe"]), _CAT_FP1)
    assert not [x for x in v if x["check"] == "V1"], v


def test_fp1_clase_b_ventana_post_verbo_no_resoluble_no_acusa():
    """(b) FP real literal: 'El Toque de Fuego: Coloca la Avena y la Leche
    en la olla; tuesta las almendras aparte.' con Avena/Leche en catálogo
    (ninguna acepta 'tostar') y 'almendras' FUERA del catálogo → 0 V1. El
    verbo 'tuesta' apunta a 'almendras' (ventana post-verbo no resuelve a
    ningún alimento del catálogo), así que la salvaguarda no debe extender
    la acusación a los acompañantes Avena/Leche."""
    v = cc.culinary_contract_scan(_plan(
        ["El Toque de Fuego: Coloca la Avena y la Leche en la olla; "
         "tuesta las almendras aparte."],
        ["100 g Avena", "200 ml Leche"]), _CAT_FP1)
    assert not [x for x in v if x["check"] == "V1"], v


def test_fp1_positivos_reales_siguen_disparando():
    """(c) Los positivos de siempre SIGUEN disparando tras el fix — ni el
    negative-lookahead de participio ni el veto de ventana post-verbo deben
    apagar un imperativo real con destinatario catalogado."""
    v1 = cc.culinary_contract_scan(_plan(
        ["Cuece el Casabe según el paquete."], ["30 g Casabe"]), _CAT)
    assert any(x["check"] == "V1" and x["food"] == "Casabe" for x in v1), v1

    v2 = cc.culinary_contract_scan(_plan(["Tuesta la Pechuga de pollo."]), _CAT)
    hits = [x for x in v2 if x["check"] == "V1"]
    assert hits and hits[0]["food"] == "Pechuga de pollo", v2


def test_fp1_montaje_skip_es_especifico_de_v1_no_afecta_v3():
    """El skip de 'Montaje:' vive SOLO en `_v1_verbo_alimento` — V3 sigue
    leyendo el blob completo de pasos (incluido Montaje) para resolver
    menciones. Si el skip se filtrara a V3, un ingrediente solo nombrado en
    el paso de Montaje se marcaría huérfano por error; aquí Repollo SÍ se
    menciona (en Montaje) y NO debe reportarse huérfano."""
    v = cc.culinary_contract_scan(_plan(
        ["Montaje: Sirve el Repollo sobre el plato."],
        ["80 g Repollo"]), _CAT)
    assert not [x for x in v if x["check"] == "V3" and x["food"] == "Repollo"], v


# ---------------------------------------------------------------------------
# [P1-CULINARY-CONTRACT-FP2 · 2026-08-01] Regresión de recall encontrada por
# el reviewer sobre la ronda 1 (ejecutando antes/después contra
# `git show f788460:...`): la ventana post-verbo de 4 palabras SILENCIABA
# «Cuece a fuego medio por 15 minutos hasta ablandar el Casabe.» — el prompt
# SSOT de `day_generator` EXIGE "cocción con AL MENOS un TIEMPO concreto",
# así que verbo + relleno-mandatado-de-tiempo/fuego + objeto es la NORMA de
# los planes reales, no la excepción; 4 palabras se comen el relleno antes
# de llegar al objeto. Fix: la ventana ahora corre hasta la FRONTERA DE
# ORACIÓN (siguiente '.'/';' o fin del paso) — una frase adverbial de
# tiempo/fuego no es otra cláusula, pero una cláusula real después de ';'
# SÍ lo es (ver el caso (b) abajo, trade-off documentado y elegido a propósito).
# ---------------------------------------------------------------------------

def test_fp2_ventana_hasta_frontera_de_oracion_recall_tiempo_mandatado():
    """(a) El patrón real de los planes: verbo + relleno de tiempo/fuego
    MANDATADO por el prompt SSOT de day_generator + objeto, todo en la
    MISMA oración. La ventana de 4 palabras fijas se comía el relleno y
    silenciaba esto — la ventana hasta frontera de oración SÍ alcanza el
    objeto real (Casabe) y el check debe seguir disparando."""
    v = cc.culinary_contract_scan(_plan(
        ["El Toque de Fuego: Cuece a fuego medio por 15 minutos hasta "
         "ablandar el Casabe."],
        ["30 g Casabe"]), _CAT)
    v1 = [x for x in v if x["check"] == "V1"]
    assert v1 and v1[0]["food"] == "Casabe", v1


def test_fp2_frontera_de_oracion_no_cruza_hacia_la_clausula_siguiente():
    """(b) Trade-off honesto y elegido a propósito: si el objeto real vive
    en la cláusula DESPUÉS de un '.'/';' que cierra la oración del verbo
    ("Cuece a fuego lento por 10 minutos; luego agrega el Casabe y sirve." —
    el Casabe está en la cláusula posterior al ';', que es un 'agrega'
    nuevo, no parte de la cocción), el veto de ventana SIGUE aplicando y el
    check calla. Es el mismo criterio fail-open del resto del módulo:
    mejor no acusar a un alimento que aparece en una cláusula distinta a la
    del verbo que se está evaluando, aunque eso cueste algo de recall en
    este caso límite. Si esto empieza a colar defectos reales en producción,
    revisar — por ahora es la decisión documentada, no un bug."""
    v = cc.culinary_contract_scan(_plan(
        ["Cuece a fuego lento por 10 minutos; luego agrega el Casabe y sirve."],
        ["30 g Casabe"]), _CAT)
    assert not [x for x in v if x["check"] == "V1"], v


def test_fp2_caso_almendras_sigue_vetado_tras_el_fix_de_frontera():
    """(c) El caso real de la ronda 1 (almendras no catalogadas) SIGUE
    vetado tras cambiar a frontera de oración: la oración del verbo
    ('Tuesta las almendras aparte.') es una oración PROPIA separada por '.'
    de la oración de Avena/Leche — nada resoluble en ESA oración → veto
    sigue aplicando, avena/leche no se acusan."""
    v = cc.culinary_contract_scan(_plan(
        ["El Toque de Fuego: Coloca la Avena y la Leche en la olla. "
         "Tuesta las almendras aparte."],
        ["100 g Avena", "200 ml Leche"]), _CAT_FP1)
    assert not [x for x in v if x["check"] == "V1"], v


# ---------------------------------------------------------------------------
# [P1-CULINARY-CONTRACT-FP round 2 · 2026-08-01] FP reales del plan bcb8bd98
# (2 residuales tras round 1, dry-run):
#   FP-C: «El Toque de Fuego: Tuesta la tortilla integral en una sartén
#   antiadherente a fuego medio... y coloca el huevo...» acusaba a Huevo de
#   'tostar' — la ventana hasta frontera de oración (round 1) cruzaba la
#   conjunción "y coloca..." y leía el destinatario del verbo SIGUIENTE
#   como si fuera de "tuesta". Fix: heurística de objeto inmediato — el
#   primer token de contenido tras "tuesta" es "tortilla" (no catalogado),
#   así que el destinatario real fue nombrado y NO es Huevo: veto, aunque
#   Huevo aparezca más adelante en la misma oración.
#   FP-D: «Coloca la pasta integral cocida en un recipiente apto para
#   horno, distribuye el queso de hoja...» acusaba a Queso de hoja de
#   'hornear' — el disparador era el SUSTANTIVO "horno" en "apto para
#   horno" (describe el envase), no una instrucción de cocción. Fix:
#   lookbehind `(?<!para )` en el fragmento de horno de VERB_TO_METHOD.
#   Caso (c) [construcción "objeto ANTES del disparador"]: «Lleva el Queso
#   de hoja al horno por 10 minutos.» SÍ debe seguir disparando — "al
#   horno" es cocción real y el objeto (Queso de hoja) fue nombrado ANTES
#   del disparador, con solo relleno de tiempo detrás ("por 10 minutos").
#   El fallback de objeto inmediato (c) mira hacia atrás, acotado a la
#   MISMA cláusula, cuando la cola hacia adelante es 100% relleno.
# Traza completa: `.superpowers/culinary-fp-round1-report.md`, sección
# "Ronda 2".
# ---------------------------------------------------------------------------

_CAT_FP2 = _CAT + [
    {"name": "Huevo", "prep_methods": ["hervir", "plancha", "freir", "hornear", "guisar", "saltear"], "ready_to_eat": False},
    {"name": "Queso de hoja", "prep_methods": ["ninguno", "crudo"], "ready_to_eat": True},
]


def test_fp_round2_caso_a_objeto_inmediato_no_catalogado_veta_huevo():
    """(a) FP-C real: el objeto inmediato de 'tuesta' es 'tortilla' (no
    catalogado) — no debe acusar a Huevo, mencionado más adelante en la
    misma oración detrás de un 'y coloca' (verbo distinto)."""
    v = cc.culinary_contract_scan(_plan(
        ["Tuesta la tortilla integral en una sartén y coloca el Huevo encima."],
        ["1 Huevo"]), _CAT_FP2)
    assert not [x for x in v if x["check"] == "V1"], v


def test_fp_round2_caso_b_horno_sustantivo_del_envase_no_dispara():
    """(b) FP-D real: 'apto para horno' describe el recipiente, no cocina —
    0 V1 sobre Queso de hoja."""
    v = cc.culinary_contract_scan(_plan(
        ["Coloca la pasta en un recipiente apto para horno, distribuye el "
         "Queso de hoja."],
        ["30 g Queso de hoja"]), _CAT_FP2)
    assert not [x for x in v if x["check"] == "V1"], v


def test_fp_round2_caso_c_al_horno_con_objeto_antes_del_disparador_dispara():
    """(c) 'al horno' SIGUE siendo instrucción real de cocción — y el
    objeto (Queso de hoja, ready_to_eat sin 'hornear' en prep) vive ANTES
    del disparador 'horno', con solo relleno de tiempo detrás. El fallback
    de objeto inmediato hacia atrás debe seguir disparando V1."""
    v = cc.culinary_contract_scan(_plan(
        ["Lleva el Queso de hoja al horno por 10 minutos."],
        ["30 g Queso de hoja"]), _CAT_FP2)
    v1 = [x for x in v if x["check"] == "V1"]
    assert v1 and v1[0]["food"] == "Queso de hoja", v1


def test_fp_round2_anclas_existentes_siguen_vivas():
    """(d) Los 4 anclas de round 1/FP2 siguen disparando/callando igual
    tras el fix de round 2 (regresión explícita, no solo confiar en que el
    resto de la suite los cubre)."""
    dispara_1 = cc.culinary_contract_scan(_plan(
        ["El Toque de Fuego: Cuece a fuego medio por 15 minutos hasta "
         "ablandar el Casabe."],
        ["30 g Casabe"]), _CAT)
    v1_1 = [x for x in dispara_1 if x["check"] == "V1"]
    assert v1_1 and v1_1[0]["food"] == "Casabe", v1_1

    calla_1 = cc.culinary_contract_scan(_plan(
        ["El Toque de Fuego: Coloca la Avena y la Leche en la olla; "
         "tuesta las almendras aparte."],
        ["100 g Avena", "200 ml Leche"]), _CAT_FP1)
    assert not [x for x in calla_1 if x["check"] == "V1"], calla_1

    calla_2 = cc.culinary_contract_scan(_plan(
        ["Montaje: Sirve el revoltillo sobre el yaniqueque horneado y el Casabe."],
        ["1 Yaniqueque", "30 g Casabe"]), _CAT_FP1)
    assert not [x for x in calla_2 if x["check"] == "V1"], calla_2

    dispara_2 = cc.culinary_contract_scan(_plan(
        ["Cuece el Casabe según el paquete."], ["30 g Casabe"]), _CAT)
    assert any(x["check"] == "V1" and x["food"] == "Casabe" for x in dispara_2), dispara_2


# ---------------------------------------------------------------------------
# [P1-CULINARY-CONTRACT-FP round 3 · 2026-08-01] FPs reales del plan 5fba379a
# (score 96.5, corr a046005f) — 7/7 violaciones V1 eran falsos positivos de
# UN SOLO mecanismo nuevo, distinto de rounds 1/2: `_v1_verbo_alimento`
# atribuía alimentos por PASO COMPLETO en vez de por ORACIÓN (cláusula) del
# verbo — si CUALQUIER ocurrencia de un método en el paso resolvía a un
# alimento legítimo, la acusación se desbloqueaba para TODOS los alimentos
# del paso, incluidos los de oraciones SIN relación con esa ocurrencia:
#   - Huevo (Desayuno real): «Tuesta el pan integral... [oración 2, sin
#     'tostar': agrega el huevo... cocina 3-4 min...]; tuesta las semillas
#     de calabaza...» — la 2ª ocurrencia de 'tostar' (semillas de calabaza,
#     legítima, sin metadata ⇒ fail-open) "rescataba" el veto para el paso
#     ENTERO y acusaba a Huevo, que vive en la oración del MEDIO y no es
#     objeto de NINGUNA ocurrencia de 'tostar'. "pan integral" (2 tokens) en
#     el paso NO resuelve al catálogo real, que solo tiene "Pan integral
#     personal"/"Pan integral familiar" (3 tokens) — la heurística de objeto
#     inmediato de round 2 SÍ vetaba correctamente la 1ª ocurrencia
#     ('vetoes', pan integral no catalogado), pero el veto viejo era
#     agregado por MÉTODO (`_post_verb_resolves` con TODOS los spans), así
#     que la 2ª ocurrencia (otra oración) igual desbloqueaba el paso entero.
#   - Cebolla/Ají cubanela/Ajo/Tomate/Berenjena/Limón (Almuerzo real): la
#     ÚNICA ocurrencia de 'tostar' del paso («Tuesta las semillas de
#     calabaza...») vive en la ÚLTIMA oración; el resto son sofrito/guisado
#     de oraciones ANTERIORES sin relación con 'tostar' — mismo mecanismo.
# Fix: la atribución de alimentos pasa de PASO a CLÁUSULA (oración,
# `_clause_bounds`) POR OCURRENCIA — cada ocurrencia del verbo solo "ve" los
# alimentos de SU PROPIA oración. La heurística de objeto inmediato de
# round 2 (`_object_inmediato_status`/`_occurrence_resolves`) se conserva
# sin cambios — sigue resolviendo el caso "dos verbos, misma oración, unidos
# por 'y'" (FP-C) que el acotado por cláusula por sí solo NO cubre.
# Traza completa (incluida la reproducción SQL contra el plan real vía
# `culinary_coherence.py` puro, sin tocar prod): `.superpowers/culinary-fp-
# round1-report.md`, sección "Ronda 3".
# ---------------------------------------------------------------------------

_CAT_FP3 = _CAT + [
    {"name": "Huevo", "prep_methods": ["hervir", "plancha", "freir", "hornear", "guisar", "saltear"], "ready_to_eat": False},
    {"name": "Cebolla", "prep_methods": ["hervir", "saltear", "plancha", "hornear", "guisar", "crudo"], "ready_to_eat": None},
    {"name": "Ají cubanela", "prep_methods": ["hervir", "saltear", "plancha", "hornear", "guisar", "crudo"], "ready_to_eat": None},
    {"name": "Ajo", "prep_methods": ["hervir", "saltear", "plancha", "hornear", "guisar", "crudo"], "ready_to_eat": None},
    {"name": "Tomate", "prep_methods": ["hervir", "saltear", "plancha", "hornear", "guisar", "crudo"], "ready_to_eat": None},
    {"name": "Berenjena", "prep_methods": ["hervir", "saltear", "plancha", "hornear", "guisar", "crudo"], "ready_to_eat": None},
    {"name": "Limón", "prep_methods": ["crudo", "licuar", "ninguno"], "ready_to_eat": True},
    {"name": "Pan integral familiar", "prep_methods": ["tostar", "ninguno"], "ready_to_eat": True},
    # Sin metadata en el catálogo real (fail-open) — incluidos para fidelidad
    # con el plan real, no participan en ninguna aserción por sí mismos.
    {"name": "Semillas de calabaza", "prep_methods": None, "ready_to_eat": None},
    {"name": "Aceite de oliva", "prep_methods": None, "ready_to_eat": None},
    {"name": "Pimienta negra", "prep_methods": None, "ready_to_eat": None},
    {"name": "Orégano dominicano", "prep_methods": None, "ready_to_eat": None},
    {"name": "Laurel", "prep_methods": None, "ready_to_eat": None},
    {"name": "Arroz blanco", "prep_methods": None, "ready_to_eat": None},
]

# Texto REAL del plan 5fba379a, día 2, Desayuno paso "El Toque de Fuego"
# (única fuente: SELECT plan_data->'days' contra Neon, solo lectura).
_FP3_PASO_DESAYUNO = (
    "El Toque de Fuego: Tuesta el pan integral en sartén a fuego medio "
    "durante 2-3 minutos por lado. En la misma sartén, calienta el aceite "
    "de oliva a fuego medio, agrega el huevo con la pimienta negra y el "
    "orégano dominicano, y cocina 3-4 minutos hasta que estén completamente "
    "cuajados; tuesta las semillas de calabaza a fuego bajo durante 2 "
    "minutos."
)

# Texto REAL del plan 5fba379a, día 2, Almuerzo paso "El Toque de Fuego".
_FP3_PASO_ALMUERZO = (
    "El Toque de Fuego: Precalienta el horno a 200°C. Coloca los guineítos "
    "verdes en una bandeja y hornéalos 18-20 minutos hasta que estén "
    "tiernos. Mientras tanto, calienta el aceite de oliva en una sartén a "
    "fuego medio; sofríe la cebolla, el ají cubanela y el ajo durante 3 "
    "minutos. Añade el pollo, el tomate, la hoja de laurel, el orégano "
    "dominicano y la pimienta negra, y cocina 10-12 minutos hasta que el "
    "pollo alcance cocción completa. Incorpora la berenjena y cocina 5 "
    "minutos más; retira la hoja de laurel y termina con el jugo de limón. "
    "Tuesta las semillas de calabaza en una sartén seca a fuego bajo "
    "durante 2 minutos."
)


def test_fp3_huevo_oracion_intermedia_no_contaminada_por_tostar_de_otra_oracion():
    """Huevo real: el FP acusaba a Huevo de 'tostar' porque el paso tiene
    DOS ocurrencias de 'tostar' en DOS oraciones distintas (pan integral —
    oración 1 — y semillas de calabaza — oración 3) y Huevo vive en la
    oración del MEDIO, sin relación con ninguna. 0 V1 tras el fix."""
    v = cc.culinary_contract_scan(_plan(
        [_FP3_PASO_DESAYUNO],
        ["2 rebanadas de pan integral familiar", "2 huevos",
         "1½ cdtas de aceite de oliva", "Pimienta negra al gusto",
         "Orégano dominicano al gusto"]), _CAT_FP3)
    v1 = [x for x in v if x["check"] == "V1"]
    assert not v1, v1


def test_fp3_pan_integral_no_resuelve_al_catalogo_de_dos_tres_tokens():
    """Ancla el porqué exacto de la 1ª oración: 'pan integral' (2 tokens en
    el paso) NO matchea 'Pan integral familiar' (3 tokens en el catálogo
    real) — el alias exige la frase completa. Confirma que la 1ª ocurrencia
    de 'tostar' no tiene NINGÚN alimento catalogado en su propia cláusula
    (no por casualidad del fix, sino porque el objeto real es no-catalogado)."""
    clause_start, clause_end = cc._clause_bounds(cc._norm(_FP3_PASO_DESAYUNO), 19)
    index = cc.build_culinary_index(_CAT_FP3)
    clausula = cc._norm(_FP3_PASO_DESAYUNO)[clause_start:clause_end]
    assert cc.find_catalog_foods(clausula, index) == [], (
        f"'pan integral' no debe resolver al catálogo dentro de su propia "
        f"oración: {clausula!r}")


def test_fp3_sofrito_y_guisado_no_contaminados_por_tostar_de_otra_oracion():
    """Almuerzo real: Cebolla/Ají cubanela/Ajo (sofrito) y Tomate/Berenjena
    (guisado) y Limón (exprimido crudo) viven en oraciones ANTERIORES a la
    única ocurrencia de 'tostar' del paso ('Tuesta las semillas de
    calabaza...', última oración) — ninguno debe acusarse de 'tostar'.
    0 V1 tras el fix (antes: 6 FPs, uno por cada alimento de esas
    oraciones)."""
    v = cc.culinary_contract_scan(_plan(
        [_FP3_PASO_ALMUERZO],
        ["1 cebolla", "1 ají cubanela", "1½ dientes de ajo",
         "½ tomate mediano", "1½ berenjena pequeña", "1 limón",
         "10 g de semillas de calabaza"]), _CAT_FP3)
    v1 = [x for x in v if x["check"] == "V1"]
    assert not v1, v1


def test_fp3_ocurrencia_ilegitima_en_su_propia_oracion_sigue_disparando():
    """Control positivo: el fix acota por CLÁUSULA, no apaga el check. Un
    'tostar' cuyo objeto real (en SU PROPIA oración) SÍ está catalogado y NO
    acepta el método sigue disparando — y un alimento de OTRA oración con un
    método que SÍ acepta sigue sin acusarse (misma separación que protege
    los FPs reales, ahora probada en sentido positivo)."""
    v = cc.culinary_contract_scan(_plan(
        ["Sofríe el Repollo. Tuesta el Bistec de res en la sartén."],
        ["80 g Repollo", "120 g Bistec de res"]), _CAT)
    v1 = [x for x in v if x["check"] == "V1"]
    assert [x["food"] for x in v1] == ["Bistec de res"], v1


def test_fp3_dedup_ocurrencias_repetidas_mismo_metodo_en_oraciones_distintas():
    """[P1-CULINARY-CONTRACT-FP round 3] El dedup `(food, método)` vive a
    nivel de PASO (set `accused` fuera del loop de ocurrencias) — dos
    ocurrencias del MISMO método en DOS oraciones distintas que ambas
    apuntan al MISMO alimento no-catalogado-para-ese-método deben producir
    UNA sola violación, no dos (a diferencia de round 1, donde 'Sofríe y
    saltea...' probaba el dedup dentro de una ÚNICA cláusula; aquí el
    dedup debe sobrevivir CRUZANDO cláusulas)."""
    v = cc.culinary_contract_scan(_plan(
        ["Tuesta el Bistec de res. Tuesta el Bistec de res otra vez."],
        ["120 g Bistec de res"]), _CAT)
    v1 = [x for x in v if x["check"] == "V1"]
    assert len(v1) == 1 and v1[0]["food"] == "Bistec de res", v1


def test_fp_round3_anclas_existentes_siguen_vivas():
    """Regresión explícita: los anclas de round 1/2 (participio+montaje,
    ventana post-verbo, objeto inmediato, horno-sustantivo, multi-alimento)
    siguen disparando/callando igual tras el refactor de atribución por
    cláusula de round 3."""
    # round 1/2: Cuece a fuego medio... el Casabe -> dispara.
    v1 = cc.culinary_contract_scan(_plan(
        ["El Toque de Fuego: Cuece a fuego medio por 15 minutos hasta "
         "ablandar el Casabe."],
        ["30 g Casabe"]), _CAT)
    hits = [x for x in v1 if x["check"] == "V1"]
    assert hits and hits[0]["food"] == "Casabe", hits

    # round 1: almendras no catalogadas -> avena/leche NO se acusan.
    v2 = cc.culinary_contract_scan(_plan(
        ["El Toque de Fuego: Coloca la Avena y la Leche en la olla; "
         "tuesta las almendras aparte."],
        ["100 g Avena", "200 ml Leche"]), _CAT_FP1)
    assert not [x for x in v2 if x["check"] == "V1"], v2

    # round 1: participio en Montaje -> no dispara.
    v3 = cc.culinary_contract_scan(_plan(
        ["Montaje: Sirve el revoltillo sobre el yaniqueque horneado y el Casabe."],
        ["1 Yaniqueque", "30 g Casabe"]), _CAT_FP1)
    assert not [x for x in v3 if x["check"] == "V1"], v3

    # round 2 FP-C: objeto inmediato no catalogado ('tortilla') veta Huevo,
    # aunque Huevo esté en la MISMA oración tras un 'y coloca'.
    v4 = cc.culinary_contract_scan(_plan(
        ["Tuesta la tortilla integral en una sartén y coloca el Huevo encima."],
        ["1 Huevo"]), _CAT_FP2)
    assert not [x for x in v4 if x["check"] == "V1"], v4

    # round 2 FP-D: 'apto para horno' es el envase, no cocina.
    v5 = cc.culinary_contract_scan(_plan(
        ["Coloca la pasta en un recipiente apto para horno, distribuye el "
         "Queso de hoja."],
        ["30 g Queso de hoja"]), _CAT_FP2)
    assert not [x for x in v5 if x["check"] == "V1"], v5

    # multi-alimento con destinatario válido -> no acusa al acompañante.
    v6 = cc.culinary_contract_scan(_plan(
        ["Hierve el Repollo y sirve con Casabe."],
        ["80 g Repollo", "30 g Casabe"]), _CAT)
    assert not [x for x in v6 if x["check"] == "V1"], v6

    # positivo de siempre: Tuesta la Pechuga de pollo -> sigue disparando.
    v7 = cc.culinary_contract_scan(_plan(["Tuesta la Pechuga de pollo."]), _CAT)
    hits7 = [x for x in v7 if x["check"] == "V1"]
    assert hits7 and hits7[0]["food"] == "Pechuga de pollo", hits7


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


# ---------------------------------------------------------------------------
# [P1-CULINARY-CONTRACT-YOGUR · 2026-08-01] Defecto REAL del plan de producción
# 97.2 (5f4bb17e-14cb-4db3-8d97-79933af690cf, día 2 Desayuno "Batido Caribeño
# de Mango, Avena y Chía", encontrado en auditoría manual — SELECT de solo
# lectura contra Neon prod). El pase de sustitución de seguridad alimentaria
# (huevo crudo → yogur griego, `graph_orchestrator._substitute_blended_raw_egg`)
# renombra la MENCIÓN pero no la instrucción de cocción, dejando: «El Toque de
# Fuego: Hierve el yogur griego en agua durante 8 minutos hasta que estén
# firmes; pélalos y córtalos en trozos.» — huevo duro aplicado a un lácteo
# listo-para-comer. V1 debía detectar esto (hervir ∉ prep_methods de un
# ready_to_eat) pero quedó CIEGO: el catálogo real (`master_ingredients`,
# verificado por SELECT) declara "Yogurt griego sin azúcar"/"Yogurt" — CON 't'
# — mientras la prosa generada (y el propio swap huevo→yogur) siempre usa "el
# yogur griego" SIN 't'. `find_catalog_foods` no encontraba NINGÚN alimento en
# el paso ofensor y el check ni evaluaba el método; el scan solo disparaba V3
# huérfano (el ingrediente "Yogurt griego sin azúcar" nunca se mencionaba
# textualmente en ningún paso), ciego a la violación real de cocción.
# ---------------------------------------------------------------------------

# Catálogo fiel a la fila REAL de `master_ingredients` (verificado por SELECT
# de solo lectura, 2026-08-01): "Yogurt" (bare, 1 token) Y "Yogurt griego sin
# azúcar" (4 tokens) coexisten como filas separadas — el alias corto es el que
# resuelve contra "el yogur griego" en prosa (V1 no hace prefix-matching de
# nombres compuestos, a diferencia de V3/`_mencionado_por_prefijo`; fuera de
# alcance de este fix, que es SOLO la normalización yogur↔yogurt).
_CAT_YOGUR = [
    {"name": "Yogurt", "prep_methods": ["ninguno", "crudo", "licuar"], "ready_to_eat": True},
    {"name": "Yogurt griego sin azúcar", "prep_methods": ["ninguno", "crudo", "licuar"],
     "ready_to_eat": True},
]

# Paso REAL del plan 97.2 (día 2, Desayuno, "El Toque de Fuego") — huevo duro
# renombrado a yogur por el swap de seguridad alimentaria, cocción intacta.
_PASO_REAL_HIERVE_YOGUR = (
    "El Toque de Fuego: Hierve el yogur griego en agua durante 8 minutos "
    "hasta que estén firmes; pélalos y córtalos en trozos."
)


def test_yogur_sin_t_resuelve_contra_catalogo_con_t():
    """Ancla mínima del alias: 'yogur' (prosa) y 'Yogurt' (catálogo) deben
    resolver al MISMO alimento — en cualquiera de las dos direcciones."""
    index = cc.build_culinary_index(_CAT_YOGUR)
    assert cc.find_catalog_foods("Licúa el yogur con hielo.", index) == ["Yogurt"]
    assert cc.find_catalog_foods("Licúa el yogurt con hielo.", index) == ["Yogurt"]


def test_yogur_variantes_plural_bidireccional():
    """Las 4 variantes (yogur/yogurt/yogures/yogurts) convergen al mismo
    match, en cualquier combinación prosa↔catálogo."""
    index_sin_t = cc.build_culinary_index(
        [{"name": "Yogur de coco", "prep_methods": ["crudo"], "ready_to_eat": True}])
    assert cc.find_catalog_foods("2 yogures de coco fríos", index_sin_t) == ["Yogur de coco"]
    assert cc.find_catalog_foods("2 yogurts de coco fríos", index_sin_t) == ["Yogur de coco"]

    index_con_t = cc.build_culinary_index(
        [{"name": "Yogurt de coco", "prep_methods": ["crudo"], "ready_to_eat": True}])
    assert cc.find_catalog_foods("2 yogures de coco fríos", index_con_t) == ["Yogurt de coco"]


def test_v1_defecto_real_plan_97_2_hervir_sobre_yogur_ready_to_eat():
    """El paso REAL del plan de producción 97.2 — antes de este fix, V1 no
    encontraba NINGÚN alimento en este paso (yogur↔Yogurt no resolvía) y
    callaba; solo V3 (huérfano) disparaba, sin decir NADA sobre la cocción
    imposible. Con el alias, V1 debe producir la violación real: 'hervir'
    aplicado a un ready_to_eat."""
    plan = _plan([_PASO_REAL_HIERVE_YOGUR],
                 ["⅔ taza de Yogurt griego sin azúcar"],
                 nombre="Batido Caribeño de Mango, Avena y Chía")
    v = cc.culinary_contract_scan(plan, _CAT_YOGUR)
    v1 = [x for x in v if x["check"] == "V1"]
    assert v1 and v1[0]["food"] == "Yogurt" and v1[0]["severity"] == "minor", v1
    assert "hervir" in v1[0]["detail"] and "listo-para-comer" in v1[0]["detail"], v1


def test_v1_legitimo_licuar_yogur_no_dispara():
    """Control positivo: 'Licúa el yogur griego con el mango.' — licuar SÍ
    está en prep_methods del catálogo real — 0 violaciones V1. Sin este
    control, un fix demasiado agresivo del alias podría convertir CUALQUIER
    mención de yogur en una acusación."""
    plan = _plan(["Licúa el yogur griego con el mango."],
                 ["⅔ taza de Yogurt griego sin azúcar", "1 mango"])
    v = cc.culinary_contract_scan(plan, _CAT_YOGUR)
    assert not [x for x in v if x["check"] == "V1"], v


# ---------------------------------------------------------------------------
# [P1-CULINARY-CONTRACT · V4 · 2026-08-01] Consistencia de cantidades
# ingredientes↔Mise en place. Caso real que motiva el check (plan 5f4bb17e,
# capturas del owner): `ingredients` dice "30 g de queso" pero el paso de
# Mise en place dice "desmenuza 1¾ lonjas/pedazos de queso de hoja (45 g)" —
# 30≠45, ambos EN GRAMOS del MISMO alimento, y el usuario no sabe a cuál
# creer. V4 SOLO compara gramos con gramos (nunca inventa una conversión
# taza/cdta/unidad → gramos) y resuelve el alimento con el matcher canónico
# del módulo (`find_catalog_foods`, word-boundary), jamás substring.
# ---------------------------------------------------------------------------

_CAT_V4 = _CAT + [
    {"name": "Queso de hoja", "prep_methods": ["ninguno", "crudo"], "ready_to_eat": True},
    {"name": "Merey", "prep_methods": ["crudo", "tostar"], "ready_to_eat": True},
    {"name": "Granola", "prep_methods": ["ninguno", "crudo"], "ready_to_eat": True},
    {"name": "Yogurt", "prep_methods": ["ninguno", "crudo", "licuar"], "ready_to_eat": True},
]


def test_v4_caso_real_queso_30_vs_45():
    """(a) Caso real literal del plan 5f4bb17e: ingrediente "30 g de queso de
    hoja" vs Mise en place "... (45 g)" — 30≠45 (33% de divergencia, por
    encima del 25% de tolerancia) debe disparar V4 con ambos números en el
    detail."""
    v = cc.culinary_contract_scan(_plan(
        ["Mise en place: desmenuza 1¾ lonjas/pedazos de queso de hoja (45 g)."],
        ["30 g de queso de hoja"]), _CAT_V4)
    v4 = [x for x in v if x["check"] == "V4"]
    assert v4 and v4[0]["food"] == "Queso de hoja" and v4[0]["severity"] == "minor" \
        and v4[0]["repairable"] is False, v4
    assert "30" in v4[0]["detail"] and "45" in v4[0]["detail"], v4


def test_v4_dentro_de_tolerancia_no_dispara():
    """(b) 100 g vs 110 g = 9% de divergencia, por debajo del 25% de
    tolerancia (generosa a propósito: redondeos de lonjas/tazas legítimos) —
    0 violaciones."""
    v = cc.culinary_contract_scan(_plan(
        ["Mise en place: pesa 110 g de queso de hoja."],
        ["100 g de queso de hoja"]), _CAT_V4)
    assert not [x for x in v if x["check"] == "V4"], v


def test_v4_unidades_no_comparables_skip():
    """(c) Ingrediente en taza (SIN 'N g' explícito) vs paso con paréntesis
    en gramos — regla dura (a): jamás inventar una conversión taza↔gramo, se
    salta la comparación en silencio."""
    v = cc.culinary_contract_scan(_plan(
        ["Mise en place: sirve el yogurt (120 g)."],
        ["½ taza de yogurt"]), _CAT_V4)
    assert not [x for x in v if x["check"] == "V4"], v


def test_v4_multi_alimento_misma_clausula_empareja_por_proximidad():
    """(d) "mide 15 g de merey y 10 g de granola" con ingredientes "15 g de
    merey"/"10 g de granola" separados — cada gramaje debe emparejarse con SU
    alimento por proximidad dentro de la cláusula, no con el primero que
    matchee (si emparejara mal, 15↔granola/10↔merey dispararía V4 falso)."""
    v = cc.culinary_contract_scan(_plan(
        ["Mise en place: mide 15 g de merey y 10 g de granola."],
        ["15 g de merey", "10 g de granola"]), _CAT_V4)
    assert not [x for x in v if x["check"] == "V4"], v


def test_v4_fail_open_catalogo_vacio():
    """(e) Catálogo vacío ⇒ `build_culinary_index` produce un índice vacío y
    `culinary_contract_scan` retorna [] antes de evaluar ningún check
    (incluido V4) — fail-open total, nunca una excepción."""
    v = cc.culinary_contract_scan(_plan(
        ["Mise en place: desmenuza 1¾ lonjas de queso de hoja (45 g)."],
        ["30 g de queso de hoja"]), [])
    assert v == []


def test_v4_prioriza_mise_en_place_sobre_otros_pasos():
    """(c del diseño) Con gramaje en DOS pasos (Mise en place Y El Toque de
    Fuego) que DIVERGEN entre sí, V4 debe comparar contra la primera mención
    de Mise en place, no contra la de El Toque de Fuego — aunque esta última
    coincida exactamente con el ingrediente (lo que enmascararía el bug real
    si el check mirara el paso equivocado)."""
    v = cc.culinary_contract_scan(_plan(
        ["Mise en place: desmenuza 1¾ lonjas de queso de hoja (45 g).",
         "El Toque de Fuego: calienta el queso de hoja (30 g) en la sartén."],
        ["30 g de queso de hoja"]), _CAT_V4)
    v4 = [x for x in v if x["check"] == "V4"]
    assert v4 and v4[0]["food"] == "Queso de hoja", (
        f"debe comparar contra Mise en place (45 g), no contra El Toque de "
        f"Fuego (30 g, que coincidiría y enmascararía el bug): {v4}")


def test_v4_sin_mise_en_place_cae_al_primer_paso():
    """Sin paso de Mise en place, V4 cae al primer paso (en orden) que
    declare gramaje explícito para el alimento."""
    v = cc.culinary_contract_scan(_plan(
        ["El Toque de Fuego: calienta 60 g de queso de hoja en la sartén."],
        ["30 g de queso de hoja"]), _CAT_V4)
    v4 = [x for x in v if x["check"] == "V4"]
    assert v4 and v4[0]["food"] == "Queso de hoja", v4
    assert "30" in v4[0]["detail"] and "60" in v4[0]["detail"], v4


def test_v4_ingrediente_sin_gramaje_no_dispara():
    """Ingrediente sin gramaje explícito ('1 lonja de queso de hoja', sin
    'N g') no tiene con qué comparar aunque el paso sí declare gramos —
    skip silencioso, no falla."""
    v = cc.culinary_contract_scan(_plan(
        ["Mise en place: desmenuza el queso de hoja (45 g)."],
        ["1 lonja de queso de hoja"]), _CAT_V4)
    assert not [x for x in v if x["check"] == "V4"], v


# ---------------------------------------------------------------------------
# [V4-FIX3 · 2026-08-01] Aproximación declarada (≈/~) NO es un contrato exacto.
# Finding "Important" del review de V4: "(≈200 g)" se leía como 200 EXACTO y
# disparaba falso positivo contra hints que el propio sistema genera vía
# `append_gram_hint` (`humanize_ingredients.py`) sobre unidades vagas
# (lonja/pedazo/porción). Ver evidencia real: "21.5 molondrones medianos
# (≈322 g)" (diagnóstico `.superpowers/esparragos-600g-diagnostico.md`).
# ---------------------------------------------------------------------------

_CAT_V4B = _CAT_V4 + [
    {"name": "Jamón", "prep_methods": ["ninguno", "crudo"], "ready_to_eat": True},
]


def test_v4_aproximacion_en_ingrediente_no_dispara():
    """(a) Ingrediente con gramaje APROXIMADO ('≈20 g') vs paso con gramaje
    EXACTO ('30 g') — el lado aproximado se descarta silenciosamente, 0 V4.
    Sin el fix, esto sería un falso positivo (33% de divergencia, por encima
    de la tolerancia)."""
    v = cc.culinary_contract_scan(_plan(
        ["Mise en place: sirve el jamón (30 g)."],
        ["1 lonja de jamón (≈20 g)"]), _CAT_V4B)
    assert not [x for x in v if x["check"] == "V4"], v


def test_v4_aproximacion_en_paso_no_dispara():
    """(a espejo) La aproximación puede vivir en CUALQUIER lado de la
    comparación — ingrediente exacto vs paso aproximado también se salta."""
    v = cc.culinary_contract_scan(_plan(
        ["Mise en place: sirve el jamón (≈30 g)."],
        ["20 g de jamón"]), _CAT_V4B)
    assert not [x for x in v if x["check"] == "V4"], v


def test_v4_aproximacion_con_virgulilla_tambien_se_salta():
    """El marcador '~' (además de '≈') también cuenta como aproximación
    declarada."""
    v = cc.culinary_contract_scan(_plan(
        ["Mise en place: sirve el jamón (~30 g)."],
        ["20 g de jamón"]), _CAT_V4B)
    assert not [x for x in v if x["check"] == "V4"], v


def test_v4_caso_real_30_vs_45_exactos_sigue_disparando_tras_el_fix():
    """El fix de aproximación NO debe convertirse en una vía de escape para
    divergencias reales — el caso real del plan 5f4bb17e (30 g vs 45 g,
    AMBOS exactos, sin ≈/~) debe seguir disparando V4 exactamente igual que
    antes del fix (mismo escenario que `test_v4_caso_real_queso_30_vs_45`,
    repetido aquí como regresión explícita del fix de aproximación)."""
    v = cc.culinary_contract_scan(_plan(
        ["Mise en place: desmenuza 1¾ lonjas/pedazos de queso de hoja (45 g)."],
        ["30 g de queso de hoja"]), _CAT_V4B)
    v4 = [x for x in v if x["check"] == "V4"]
    assert v4 and v4[0]["food"] == "Queso de hoja", v4
    assert "30" in v4[0]["detail"] and "45" in v4[0]["detail"], v4


# ---------------------------------------------------------------------------
# [P1-CULINARY-CONTRACT-BATTER · 2026-08-01] Clase FP "mezclas horneadas" —
# la única que aún bloqueaba el reloj de F2 (necesita ≥7 días de warn limpio;
# una torta de avena horneada dispara en CADA plan con desayuno horneado).
# Caso real (plan 054a43c2, día 3 Desayuno — torta de avena horneada / baked
# oats; el plan ya NO vive en Neon al momento de este fix — verificado por
# SELECT de solo lectura, 0 filas — así que el texto abajo es fiel a la cita
# TEXTUAL del reporte, con relleno plausible donde el reporte truncaba con
# "..."). V1 acusaba a los 5 alimentos integrados (Avena, Leche descremada,
# Polvo de hornear, Canela, Vainilla) porque ninguno lista 'hornear'
# individualmente en `prep_methods` — cierto pero irrelevante: el nombre del
# alimento "Polvo de hornear" contiene LITERALMENTE la palabra "hornear", así
# que el regex de `VERB_TO_METHOD` lo lee como una ocurrencia real del verbo
# de cocción dentro de la MISMA cláusula que lo integra ("Integra la avena,
# la leche descremada, el polvo de hornear, la canela y la vainilla..."). El
# plato es culinariamente NORMAL: lo que se hornea es LA MEZCLA resultante de
# integrar los ingredientes, no cada componente por separado. Ver
# `MEZCLA_VERBOS`/`_es_clausula_mezcla` en `culinary_coherence.py` para el
# razonamiento completo (incluida la colisión "integral" ⊃ "integra" ya
# protagonista de FP round 2, y por qué 'bate'/'une' usan alternancia exacta
# en vez de raíz+comodín).
# ---------------------------------------------------------------------------

_CAT_MEZCLA = _CAT_FP1 + [
    {"name": "Polvo de hornear", "prep_methods": ["ninguno"], "ready_to_eat": True},
    {"name": "Canela", "prep_methods": ["ninguno"], "ready_to_eat": True},
    {"name": "Vainilla", "prep_methods": ["ninguno"], "ready_to_eat": True},
]

_PASO_REAL_BAKED_OATS = (
    "El Toque de Fuego: Precalienta el horno a 180°C. Integra la avena, la "
    "leche descremada, el polvo de hornear, la canela y la vainilla en un "
    "tazón hasta formar una mezcla homogénea. Vierte la mezcla en un molde "
    "apto para horno y hornea durante 25 minutos hasta que esté firme."
)


def test_mezcla_caso_real_baked_oats_hornear_la_mezcla_no_los_componentes():
    """Caso real completo (plan 054a43c2): la ocurrencia de 'hornear' que
    antes disparaba las 5 acusaciones vive DENTRO de la cláusula de
    integración ("...el polvo de hornear...", el propio nombre del alimento
    contiene el verbo). La cláusula integra 5 alimentos (≥3) bajo 'integra'
    → salvaguarda de mezcla activa → 0 V1. La ocurrencia de 'horno' en la
    primera oración ("Precalienta el horno") y la de 'hornea' en la tercera
    ("...y hornea durante 25 minutos") no acusan a nadie de todos modos —
    ninguna de esas dos cláusulas menciona alimento alguno (el molde/la
    mezcla no son alimentos catalogados) — la salvaguarda de mezcla ni
    siquiera hace falta ahí (`if not foods: continue` ya las cubre)."""
    v = cc.culinary_contract_scan(_plan(
        [_PASO_REAL_BAKED_OATS],
        ["50 g Avena", "100 ml Leche descremada", "5 g Polvo de hornear",
         "2 g Canela", "5 ml Vainilla"],
        nombre="Torta de avena horneada"), _CAT_MEZCLA)
    v1 = [x for x in v if x["check"] == "V1"]
    assert v1 == [], v1


def test_mezcla_control_a_un_solo_alimento_sin_integracion_sigue_disparando():
    """(a) 'Cuece el Casabe según el paquete.' — 1 alimento, sin verbo de
    integración: la salvaguarda de mezcla no debe apagar el positivo de
    siempre (mismo caso que ancla `test_v1_verbo_imposible_sobre_ready_to_eat`,
    repetido aquí como regresión explícita del fix de mezcla)."""
    v = cc.culinary_contract_scan(_plan(
        ["Cuece el Casabe según el paquete."], ["30 g Casabe"]), _CAT)
    assert any(x["check"] == "V1" and x["food"] == "Casabe" for x in v), v


def test_mezcla_control_b_fosil_yogur_sin_verbo_integracion_sigue_disparando():
    """(b) El fósil real (P1-CULINARY-CONTRACT-YOGUR): 'Hierve el yogur
    griego en agua durante 8 minutos hasta que estén firmes; pélalos...' NO
    tiene verbo de integración (`MEZCLA_VERBOS`) en su cláusula — la nueva
    salvaguarda no debe apagarlo. Reusa el fixture REAL exacto de esa
    regresión (`_CAT_YOGUR`/`_PASO_REAL_HIERVE_YOGUR`) en vez de reinventar
    el texto."""
    plan = _plan([_PASO_REAL_HIERVE_YOGUR],
                 ["⅔ taza de Yogurt griego sin azúcar"],
                 nombre="Batido Caribeño de Mango, Avena y Chía")
    v = cc.culinary_contract_scan(plan, _CAT_YOGUR)
    v1 = [x for x in v if x["check"] == "V1"]
    assert v1 and v1[0]["food"] == "Yogurt", v1


def test_mezcla_control_c_tuesta_pechuga_de_pollo_sigue_disparando():
    """(c) 'Tuesta la Pechuga de pollo.' — 1 alimento, sin integración,
    'tostar' fuera de sus `prep_methods` reales: sigue disparando (mismo
    caso que `test_fp1_positivos_reales_siguen_disparando`, repetido aquí
    como regresión explícita del fix de mezcla)."""
    v = cc.culinary_contract_scan(_plan(["Tuesta la Pechuga de pollo."]), _CAT)
    v1 = [x for x in v if x["check"] == "V1"]
    assert v1 and v1[0]["food"] == "Pechuga de pollo", v1


def test_mezcla_control_d_integracion_bajo_el_umbral_no_exime_al_licuado_aparte():
    """(d) Un paso que integra SOLO 2 alimentos (Avena+Leche, bajo el umbral
    de 3) en una cláusula, y LICÚA un tercero (Bistec de res) no-integrado
    en una cláusula APARTE: la salvaguarda de mezcla no debe activarse en
    NINGUNA de las dos cláusulas (la de integración no alcanza el umbral; la
    del licuado no contiene verbo de integración) — Bistec de res sigue
    acusado con normalidad. Prueba que la exención nunca escapa de su propia
    cláusula de integración, ni se dispara por un umbral no alcanzado."""
    v = cc.culinary_contract_scan(_plan(
        ["Integra la Avena y la Leche en un tazón. Licúa el Bistec de res "
         "con hielo."],
        ["50 g Avena", "100 ml Leche", "120 g Bistec de res"]), _CAT_FP1)
    v1 = [x for x in v if x["check"] == "V1"]
    assert v1 and v1[0]["food"] == "Bistec de res", v1


def test_mezcla_no_colisiona_con_integral_pan_arroz_tortilla():
    """Guard adicional (no listado en los 4 controles, encontrado por
    inspección propia): 'integral' (pan/arroz/tortilla integral, calificador
    COMÚN en la prosa dominicana, protagonista real de FP round 2 caso FP-C)
    NO debe leerse como una conjugación de 'integrar' — sin el lookahead
    `al(?:es)?\\b` en `_MEZCLA_RE`, el mero hecho de decir 'tortilla
    integral' en un paso con ≥3 alimentos catalogados activaría la
    salvaguarda de mezcla de forma espuria y apagaría acusaciones legítimas
    que nada tienen que ver con integrar/mezclar nada."""
    assert not cc._MEZCLA_RE.search("tortilla integral")
    assert not cc._MEZCLA_RE.search("arroz integral")
    assert not cc._MEZCLA_RE.search("pan integrales")
    assert cc._MEZCLA_RE.search("integra los ingredientes")
