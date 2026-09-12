# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-13 · 2026-09-12] Decimotercer lote del plan de pendientes: E9 y los residuos técnicos.

E9 · «vegetariano · día 25 sin congelador» en ES dejaba UN almuerzo. Medido con `scripts/coverage_funnel.py`
antes de escribir un plato: de los 10 almuerzos vegetarianos españoles, 9 llevaban calabacín, berenjena, tomate,
pimiento, espinacas o acelgas (7-10 días) y sólo la tortilla aguantaba. La cura no es tocar la regla de
conservación —dice la verdad— sino curar almuerzos de DESPENSA: 9 plantillas cuyos constituyentes aguantan ≥ 35
días (legumbre seca, arroz, pasta, huevo, patata, zanahoria, col, cebolla, ajo, conserva). Resultado medido:
almuerzo 1 → 10, cena 3 → 8 en ese cruce; la ⚠ del embudo desaparece.

Residuos, con su causa medida:
  · Los 3 tests «dependientes del orden» tenían UNA causa: `test_p1_form_efecto` escribía
    `os.environ["MEALFIT_COUNTRY_SYSTEM"]="true"` sin restaurar y hacía `importlib.reload(constants)`. Con el knob
    filtrado, la ruta USD del piso de presupuesto cambia de fórmula, «tortilla de maíz» resuelve a su fila propia y
    la identidad `go._RENAL_CONDITION_TERMS is constants.RENAL_CONDITION_TERMS` muere con el reload. Reproducido
    en un solo proceso (3 failed) y verde tras el arreglo (0 failed). *Tres síntomas en tres ficheros pueden ser
    un solo defecto en un cuarto.*
  · Las 5 bibliotecas no dominicanas llevaban DRIFT de catálogo desde el 09-09. Recompiladas y re-ancladas con lo
    MEDIDO plato a plato contra el árbol firmado: MX/PR/US cero cambios; CO un dato derivado (el fósforo del
    Borojó dejó de ser desconocido); ES +9.
  · El generador `build_dish_constituents_do.py` ya no reproducía el SSOT (75 entradas divergían): retirado; en su
    lugar `scripts/check_dish_constituents_do.py` valida la coherencia tabla↔plantillas sin catálogo.
  · El refresh nocturno resolvía el huso ausente a 0 (UTC) en dos `COALESCE(...,0)` del SQL y en su helper: ahora
    cae al SSOT `DEFAULT_TZ_OFFSET_MIN`, como los 8 sitios de LOTE-9.
  · Los 7 pasos del wizard sin `fields` salían como `step_<índice>` en el embudo: llevan `id` propio (frontend).

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_ROOT = _BACKEND.parent


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _json(rel: str):
    return json.loads(_src(rel))


def _script(name: str):
    """Carga `scripts/<name>.py` por RUTA, sin tocar sys.path. Medido en el gate de este mismo lote: con `scripts/` en
    CABEZA de sys.path, `import plan_gym` cargaba `scripts/plan_gym.py` (el CLI), que se importa a sí mismo — ImportError
    circular en `test_p1_next_level_batch`, un fichero que no había tocado nadie."""
    import importlib.util
    p = _BACKEND / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_lote13_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NUEVAS_ES = {
    "Lentejas estofadas con patata y zanahoria", "Garbanzos guisados con pimentón y huevo duro",
    "Arroz a la cubana con huevo y salsa de tomate", "Coditos con salsa de tomate y queso gouda",  # [P1-PLAN-LOTE-16 · A9] renombrado por el dueño
    "Judías blancas estofadas con patata, laurel y pimentón", "Ensalada de garbanzos con huevo duro, aceitunas y cebolla",
    "Patatas guisadas con huevo escalfado y pimentón", "Arroz con garbanzos, zanahoria y pimentón",
    "Trinxat de col y patata con huevo",
}


# ─────────────────────────── E9 · los nueve almuerzos de despensa ───────────────────────────

def test_e9_las_nueve_plantillas_estan_en_la_biblioteca_y_compilan_integras():
    fuentes = {t["name"]: t for t in _json("data/dish_templates_es.json")["templates"]}
    assert NUEVAS_ES <= set(fuentes), NUEVAS_ES - set(fuentes)
    reg = {t["name"]: t for t in _json("data/registry/dish_registry_es_v1.json")["templates"]}
    assert NUEVAS_ES <= set(reg)
    for n in NUEVAS_ES:
        t = reg[n]
        assert t["status"] == "ok" and t["excluded"] == [], (n, t["excluded"])
        assert "almuerzo" in t["slots"], n
        assert t["template_id"].startswith("tpl_")


def test_e9_todo_constituyente_aguanta_el_dia_25_sin_congelador():
    """La regla de conservación no se tocó: los platos la cumplen. Día 25 ⇒ hacen falta ≥ 26 días en fresco."""
    import pantry_durability as pdur
    reg = {t["name"]: t for t in _json("data/registry/dish_registry_es_v1.json")["templates"]}
    for n in NUEVAS_ES:
        lg = reg[n]["logistics"]
        assert lg["days_fresh_min"] >= 26 and lg["pantry_only"] is True, (n, lg)
        assert pdur.template_fits(lg["days_fresh_min"], lg["days_with_freezer_min"], 26, False), n
        for c in reg[n]["constituents"]:
            cl = pdur.classify(c["canonical"])
            assert cl["days_fresh"] >= 26, (n, c["name"], cl)


def test_e9_son_vegetarianos_y_ninguno_es_un_fantasma_de_nombre():
    """Vegetarianos por la GUARDA real sobre constituyentes (la del selector), no por la etiqueta `protein`."""
    from graph_orchestrator import _diet_pool_item_banned
    reg = {t["name"]: t for t in _json("data/registry/dish_registry_es_v1.json")["templates"]}
    for n in NUEVAS_ES:
        for c in reg[n]["constituents"]:
            assert not _diet_pool_item_banned(c["name"], "vegetariana"), (n, c["name"])
        np_ = reg[n]["nutrition_per_serving"]
        assert 300 <= np_["kcal"] <= 700 and np_["protein_g"] >= 15, (n, np_)
        assert np_["sodium_mg"] < 600, (n, np_["sodium_mg"])


def test_e9_el_cruce_deja_de_ser_el_unico_hueco_del_embudo():
    """El embudo del producto (mismo guard de dieta y misma regla de conservación que el selector), sin mercado."""
    cf = _script("coverage_funnel")
    alm = cf.embudo("ES", "almuerzo", "vegetariana", (), 25, "none")
    cena = cf.embudo("ES", "cena", "vegetariana", (), 25, "none")
    assert alm["elegibles"] >= 8, alm["etapas"]
    assert cena["elegibles"] >= 6, cena["etapas"]
    ids = {t["template_id"] for t in _json("data/registry/dish_registry_es_v1.json")["templates"] if t["name"] in NUEVAS_ES}
    assert len(ids & set(alm["supervivientes"])) == 9, "las nueve sobreviven a la conservación del día 25"
    # y sin restricción de conservación el almuerzo vegetariano también creció (10 → ≥ 18)
    assert cf.embudo("ES", "almuerzo", "vegetariana", (), None, "limited")["elegibles"] >= 18


def test_e9_la_biblioteca_es_sigue_pasando_su_propia_vara():
    """Las 124 anteriores no cambian; las nuevas no rompen las varas de F7 (franjas, técnicas, sin base de almidón en cena)."""
    ts = _json("data/dish_templates_es.json")["templates"]
    assert len(ts) == 133
    for t in ts:
        if t["name"] in NUEVAS_ES and str(t["base"]).lower() in ("arroz", "pasta"):
            assert set(t["slots"]) == {"almuerzo"}, (t["name"], "arroz/pasta como BASE nunca en desayuno ni cena")
    assert len({t["name"] for t in ts}) == 133, "nombre duplicado"


# ─────────────────────────── residuo: las cinco firmas, re-ancladas con lo medido ───────────────────────────

@pytest.mark.parametrize("pid,lib", [("spain_mediterranea", "es"), ("mexico_casera", "mx"), ("colombia_casera", "co"),
                                     ("puertorico_criolla", "pr"), ("us_everyday", "us"), ("dominican_criolla", "do")])
def test_las_seis_firmas_apuntan_a_su_snapshot_y_el_informe_tambien(pid, lib):
    reg = _json(f"data/registry/dish_registry_{lib}_v1.json")
    rev = _json("data/registry/cultural_curation_review_v1.json")
    bench = _json("data/registry/cultural_benchmark_v1.json")
    assert rev["profiles"][pid]["snapshot_hash"] == reg["snapshot_hash"], f"{pid}: firma caducada — recompilaste sin re-anclar"
    assert bench["profiles"][pid]["snapshot_hash"] == reg["snapshot_hash"], f"{pid}: informe desfasado"
    assert reg["stats"]["partial"] == 0 and reg["stats"]["excluded"] == 0


def test_el_reanclaje_dice_lo_que_midio_y_deja_las_altas_es_pendientes_del_dueno():
    rev = _json("data/registry/cultural_curation_review_v1.json")
    assert "reanchor_note_2026_09_12" in rev and "P1-PLAN-LOTE-13" in rev["reanchor_note_2026_09_12"]
    for pid in ("mexico_casera", "puertorico_criolla", "us_everyday"):
        d = rev["profiles"][pid]["decisions"][-1]
        assert "P1-PLAN-LOTE-13" in d and "CERO plantillas cambiadas" in d, pid
    co = rev["profiles"]["colombia_casera"]["decisions"][-1]
    assert "P1-PLAN-LOTE-13" in co and "Borojó" in co and "fósforo" in co, "el único cambio de CO es un dato derivado y se dice"
    es = rev["profiles"]["spain_mediterranea"]
    assert any("E9" in d and "1 → 10" in d for d in es["decisions"]), "la decisión de E9 sigue en el acta aunque ya no sea la última"
    # [P1-PLAN-LOTE-16 · A9] Pendientes ANTES del juicio del dueño, juzgados DESPUÉS: el acta guarda los nombres de entonces
    # («Macarrones…»), así que se comparan a través del alias del renombre.
    import plan_policy
    viejo_a_nuevo = {v: k for k, v in plan_policy.TEMPLATE_ALIASES.items()}
    bloque = es.get("pendiente_de_juicio_humano") or es["juicio_humano_a9"]
    assert {viejo_a_nuevo.get(x, x) for x in bloque["platos"]} == NUEVAS_ES and bloque["fecha"] == "2026-09-12"
    assert rev["date"] == "2026-09-12"


# ─────────────────────────── residuo: el generador se retiró, el validador manda ───────────────────────────

def test_el_generador_que_no_reproducia_el_ssot_se_retiro():
    assert not (_BACKEND / "scripts" / "build_dish_constituents_do.py").exists()
    for rel in ("dish_registry.py", "docs/dish_registry_f6.md", "tests/test_p1_ciclo_30d_duradero.py"):
        assert "check_dish_constituents_do.py" in _src(rel), rel
    note = _json("data/dish_constituents_do.json")["_note"]
    assert "ES EL SSOT" in note and "check_dish_constituents_do.py" in note and "no lo regeneres" not in note


def test_el_validador_da_coherente_la_tabla_committed_y_caza_una_huerfana(tmp_path):
    chk = _script("check_dish_constituents_do")
    res = chk.check()
    assert res["ok"], res["fallos"]
    assert res["sin_inline"] == res["entradas"] == 138 and res["plantillas"] == 193
    # una plantilla renombrada sin mover la clave (lo que C8 hizo tres veces) tiene que caer
    tabla = _json("data/dish_constituents_do.json")
    k = next(iter(tabla["templates"]))
    tabla["templates"]["Plato que ya no existe"] = tabla["templates"].pop(k)
    p = tmp_path / "t.json"
    p.write_text(json.dumps(tabla, ensure_ascii=False), encoding="utf-8")
    res2 = chk.check(table_path=str(p))
    assert not res2["ok"] and any("huérfana" in f for f in res2["fallos"]) and any(k in f for f in res2["fallos"])


# ─────────────────────────── residuo: el refresh nocturno y el huso ausente ───────────────────────────

def test_el_refresh_nocturno_sin_huso_cae_al_ssot_no_a_utc():
    import cron_tasks as ct
    from constants import DEFAULT_TZ_OFFSET_MIN
    # 07:00 UTC = 03:00 en RD (UTC-4, offset +240): un snapshot SIN huso debe verse como RD, no como UTC
    now = datetime(2026, 9, 12, 7, 0, tzinfo=timezone.utc)
    assert DEFAULT_TZ_OFFSET_MIN == 240
    assert ct._is_user_local_refresh_hour(now, None) is True
    assert ct._is_user_local_refresh_hour(now, "") is True
    assert ct._is_user_local_refresh_hour(now, "basura") is True, "basura = «no sé» ⇒ SSOT"
    assert ct._is_user_local_refresh_hour(now, 0) is False, "un 0 explícito sigue siendo UTC (07:00 ≠ 03:00)"
    assert ct._is_user_local_refresh_hour(datetime(2026, 9, 12, 3, 0, tzinfo=timezone.utc), 0) is True
    assert ct._is_user_local_refresh_hour(now, 240) is True and ct._is_user_local_refresh_hour(now, "240") is True


def test_el_sql_del_refresh_nocturno_no_resuelve_el_huso_a_cero():
    src = _src("cron_tasks.py")
    i = src.find("def _nightly_refresh_all_pending_snapshots(")
    cuerpo = src[i:i + 9000]
    assert "tooltip-anchor: _is_user_local_refresh_hour (test_p1_plan_lote_13.py)" in src
    assert not re.search(r"'tz_offset_minutes'\)::int,\s*0\s*\)", cuerpo), "vuelve COALESCE(..., 0) para el huso"
    assert cuerpo.count("%s::int\n") >= 2 or cuerpo.count("%s::int\r\n") >= 2
    assert "(int(_default_tz), now_utc, int(_default_tz), CHUNK_PANTRY_PROACTIVE_REFRESH_MAX_USERS * 8,)" in cuerpo
    assert "from constants import DEFAULT_TZ_OFFSET_MIN as _default_tz" in cuerpo


# ─────────────────────────── residuo: los tres «flaky» y su única causa ───────────────────────────

def test_form_efecto_ya_no_filtra_el_knob_maestro_al_worker():
    src = _src("tests/test_p1_form_efecto.py")
    assert 'monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")' in src
    assert 'os.environ["MEALFIT_COUNTRY_SYSTEM"] = "true"' not in src
    assert not _RELOAD_CONSTANTS.search(src), "el docstring puede nombrar el reload; el código no puede ejecutarlo"


_RELOAD_CONSTANTS = re.compile(r"^\s*importlib\.reload\(\s*constants\s*\)", re.M)
_ESCRIBE_KNOB_MAESTRO = re.compile(r"""^\s*os\.environ\[\s*["']MEALFIT_COUNTRY_SYSTEM["']\s*\]\s*=\s*(?P<v>[^\n]+)$""", re.M)


def test_ratchet_ningun_test_recarga_constants():
    """`importlib.reload(constants)` re-ejecuta el módulo REAL: toda identidad `is` del worker muere. Contaminación
    irreversible (P1-SUITE-SWEEP la midió en julio; LOTE-13 quitó los 3 usos que quedaban)."""
    malos = []
    for p in sorted((_BACKEND / "tests").glob("test_*.py")):
        s = p.read_text(encoding="utf-8")
        for m in _RELOAD_CONSTANTS.finditer(s):
            linea = s[:m.start()].count("\n") + 1
            malos.append(f"{p.name}:{linea}")
    assert not malos, f"vuelve `importlib.reload(constants)`: {malos} — usa monkeypatch.setenv/setattr; el knob se lee por llamada"


def test_ratchet_el_knob_maestro_de_paises_no_se_escribe_sin_restaurar():
    """El knob que cambia de producto a TRES subsistemas (mercado, piso de presupuesto, guard de preparaciones): si
    un test lo escribe a mano en os.environ, tiene que restaurarlo en un `finally` de las 25 líneas siguientes."""
    malos = []
    for p in sorted((_BACKEND / "tests").glob("test_*.py")):
        s = p.read_text(encoding="utf-8")
        lineas = s.split("\n")
        for m in _ESCRIBE_KNOB_MAESTRO.finditer(s):
            i = s[:m.start()].count("\n")
            valor = m.group("v").strip()
            if valor in ("prev", "old", "_prev", "previo", "anterior") or valor.startswith(("prev", "old", "_prev")):
                continue   # esto ES la restauración
            ventana = "\n".join(lineas[i:i + 25])
            if "finally:" not in ventana:
                malos.append(f"{p.name}:{i + 1} = {valor}")
    assert not malos, f"escritura del knob maestro sin restaurar (contamina al worker entero): {malos} — usa monkeypatch.setenv"


_SCRIPTS_EN_CABEZA = re.compile(r"sys\.path\.insert\(\s*0\s*,[^\n]*scripts", re.M)


def test_ratchet_ningun_test_pone_scripts_en_cabeza_del_sys_path():
    """`scripts/plan_gym.py` y `plan_gym.py` comparten nombre (el único par que colisiona): con scripts/ en cabeza,
    `import plan_gym` carga el CLI, que se importa a sí mismo — ImportError circular en `test_p1_next_level_batch`. Cayó
    en el gate de este lote por la primera versión de este mismo fichero. Al final del path (`append`) o por ruta."""
    malos = [p.name for p in sorted((_BACKEND / "tests").glob("test_*.py"))
             if _SCRIPTS_EN_CABEZA.search(p.read_text(encoding="utf-8"))]
    assert not malos, f"scripts/ en cabeza de sys.path (sombrea plan_gym): {malos} — usa sys.path.append o importlib por ruta"


def test_los_tres_flaky_pasan_en_el_orden_que_los_hacia_caer():
    """Lo medido antes del arreglo: con form_efecto delante en el mismo proceso, exactamente estos tres caían."""
    from constants import RENAL_CONDITION_TERMS
    import graph_orchestrator as go
    import shopping_calculator as sc
    assert go._RENAL_CONDITION_TERMS is RENAL_CONDITION_TERMS
    assert sc.resolve_preparation_distinct("tortilla de maíz") == (True, None)


# ─────────────────────────── residuo: los pasos del wizard sin campo ───────────────────────────

def test_los_siete_pasos_sin_campo_llevan_id_propio():
    flow = (_ROOT / "frontend" / "src" / "components" / "assessment" / "InteractiveAssessmentFlow.jsx")
    if not flow.exists():
        pytest.skip("frontend no presente (repo hermano)")
    s = flow.read_text(encoding="utf-8")
    assert "step_id: step?.id || field || `step_${currentStep}`," in s
    ids = re.findall(r"^\s+id: '([A-Za-z]+)',$", s, re.M)
    assert sorted(ids) == sorted(["habits", "shoppingHabits", "stapleFoods", "goalTarget", "supplements", "pantryBuilder", "trackingFinish"]), ids
    assert "P1-PLAN-LOTE-13" in _src("docs/plan_policy_f4.md")


# ─────────────────────────── docs y marker ───────────────────────────

def test_los_docs_cuentan_el_lote():
    plan = _src("docs/plan_pendientes_2026_09_11.md")
    assert re.search(r"^\| E9 \| ✅ 2026-09-12", plan, re.M)
    assert "P1-PLAN-LOTE-13" in plan
    cf = _src("docs/coverage_funnel.md")
    assert "P1-PLAN-LOTE-13" in cf and "veg · día 25 sin congelador" in cf


def test_marker_bumpeado():
    import app
    assert "[P1-PLAN-LOTE-13 · 2026-09-12]" in _src("app.py")
    # [P1-PLAN-LOTE-13 · 2026-09-12] «no anterior a este lote», no «igual a hoy»: el pin de la fecha y del prefijo
    # `P1-PLAN-` rompía 12 tests el primer día en que otro P-fix bumpeaba el marker.
    assert app._LAST_KNOWN_PFIX.split("·")[-1].strip() >= "2026-09-12", app._LAST_KNOWN_PFIX
