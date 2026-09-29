# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-850 · 2026-09-29] La asignación previa y el prompt ya no empujan cocina dominicana a los países beta.

Batería real G24 (29-sep, 6 planes con el código desplegado del 815):
  · España recibió «Harina de trigo + Batata» como carbohidratos los 3 días y las técnicas «Estilo Fusión Criolla» y
    «Desmenuzado (Ropa Vieja)» — el planificador escribió «Día mediterráneo-criollo»; México, «Yuca + Harina de maíz
    precocida» (harina de AREPA) y «Relleno (Ej. Canoas…)»; Estados Unidos, «Ropa Vieja» aplicada a huevos.
  · Los 18 prompts beta del generador de días decían «intégralos en comidas dominicanas variadas y sabrosas» y
    «Potasio → guineo, plátano, batata»; el de sistema seguía con «Mangú/tubérculos», habichuelas, gandules y guineo;
    el del planificador ponía de ejemplo «Mangú de plátano / ñame / batata».

Contrato:
  · Los carbohidratos que el sembrador ASIGNA salen de la biblioteca de platos de la cocina del perfil (registry
    compilado) menos lo que `cultural_profiles.PROFILE_KITCHEN` declara ajeno como base; las técnicas, de las etiquetas
    del perfil. El catálogo no cambia (el modelo puede seguir usando cualquier alimento que su mercado venda).
  · Los NOMBRES del catálogo (identificadores del motor) no se tocan: sólo se elige cuáles se asignan.
  · DO no cambia NADA (ni la cocina DO en un mercado beta). Knob `MEALFIT_BETA_CULTURAL_ASSIGNMENT` (True).
tooltip-anchor: P1-PLAN-LOTE-850
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_BETA = ("ES", "US", "MX", "PR", "CO")
_CULTURA = {"DO": "dominican_criolla", "ES": "spain_mediterranea", "US": "us_everyday", "MX": "mexico_casera",
            "PR": "puertorico_criolla", "CO": "colombia_casera"}
_KNOB = "MEALFIT_BETA_CULTURAL_ASSIGNMENT"


def _src(rel):
    return (_BACKEND / rel).read_text(encoding="utf-8")


@pytest.fixture
def paises(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.delenv(_KNOB, raising=False)
    monkeypatch.delenv("MEALFIT_MARKET_POOL_UNIVERSAL", raising=False)
    monkeypatch.delenv("MEALFIT_CULTURAL_PROFILES", raising=False)
    yield monkeypatch


def _form(cc, uid="749a56da-5817-4921-9df2-08b6ae28f7f6"):
    return {"country": cc, "cultureProfiles": {"main": _CULTURA[cc], "secondary": []}, "dietType": "balanced",
            "allergies": ["Ninguna"], "dislikes": ["Ninguno"], "cookingTime": "30min", "mainGoal": "maintenance",
            "budget": "medium", "gender": "female", "age": "30", "user_id": uid}


def _carbs(cc, cultura):
    import constants
    return constants._get_fast_filtered_catalogs((), (), "", country=cc, market_extras=True, culture_country=cultura)[1]


# ─── 1. los carbohidratos asignables salen de la cocina del perfil ──────────────────────────────────────────────────
@pytest.mark.parametrize("cc,fuera,dentro", [
    ("ES", ("Batata", "Yuca", "Plátano verde", "Plátano maduro", "Harina de trigo", "Harina de maíz precocida",
            "Tortilla de maíz", "Habichuelas rojas"), ("Arroz blanco", "Papa", "Lentejas", "Garbanzos")),
    ("MX", ("Yuca", "Harina de maíz precocida", "Plátano verde"), ("Tortilla de maíz", "Frijoles pintos", "Arroz blanco")),
    ("US", ("Yuca", "Plátano verde", "Plátano maduro", "Harina de maíz precocida"), ("Bagels", "Papa", "Batata")),
    ("CO", ("Tortilla de maíz",), ("Harina de maíz precocida", "Plátano verde", "Yuca", "Papa")),
    ("PR", ("Tortilla de maíz",), ("Yuca", "Batata", "Plátano verde", "Gandules")),
])
def test_carbos_del_pool_beta_salen_de_la_cocina_del_perfil(paises, cc, fuera, dentro):
    c = _carbs(cc, cc)
    assert not [x for x in fuera if x in c], (cc, c)
    for x in dentro:
        assert x in c, (cc, x, c)
    assert len(c) >= 4


def test_carbos_sin_cereales_de_desayuno(paises):
    for cc in _BETA:
        c = _carbs(cc, cc)
        assert not [x for x in c if re.search(r"avena|granola|cereal", x, re.I)], (cc, c)


def test_carbos_knob_apagado_y_cocina_do_como_antes(paises):
    import constants
    paises.setenv(_KNOB, "false")
    assert "Batata" in _carbs("ES", "ES") and "Yuca" in _carbs("MX", "MX")
    paises.delenv(_KNOB)
    # cocina DO en mercado beta: el sesgo criollo de F7-H intacto (yuca, plátano, habichuelas)
    for must in ("Plátano verde", "Yuca", "Habichuelas rojas"):
        assert must in _carbs("US", "DO"), must
    # sin cocina (camino degradado del cron): el pool de antes
    assert "Batata" in constants._get_fast_filtered_catalogs((), (), "", country="ES", market_extras=True)[1]


def test_datos_de_la_cocina_por_perfil(paises):
    import constants
    from cultural_profiles import PROFILE_KITCHEN, PROFILES
    assert "dominican_criolla" not in PROFILE_KITCHEN, "DO no se localiza"
    assert set(PROFILE_KITCHEN) == {p for p in PROFILES if p not in ("dominican_criolla", "neutral")}
    pools = {x for k in ("carbs",) for cc in constants.COUNTRY_POOLS for x in constants.COUNTRY_POOLS[cc][k]}
    pools |= set(constants.UNIVERSAL_MARKET_STAPLES["carbs"])
    for pid, datos in PROFILE_KITCHEN.items():
        for n in datos.get("carb_bases_excluded") or ():
            assert n in pools, (pid, n, "nombre fuera de los pools: el filtro no casaría nada")
        for canon, local in (datos.get("technique_labels") or {}).items():
            assert canon in constants.ALL_TECHNIQUES, (pid, canon)
            assert local is None or (isinstance(local, str) and local.strip()), (pid, canon)
        voc = datos.get("vocab") or {}
        for k in ("legumbres", "legumbre", "legumbre_1", "banana", "banana_maduro", "potasio"):
            assert str(voc.get(k) or "").strip(), (pid, k)


# ─── 2. el sembrador (la asignación que ve el planificador) ─────────────────────────────────────────────────────────
def _opciones(prompt):
    return "\n".join(l for l in prompt.splitlines() if "OPCIÓN" in l and "obligatoriamente" in l)


# La avena no es base de almuerzo/cena (P1-OATS-NOT-A-DINNER) y con ella en el par salía «Avena + otra base distinta
# del catálogo» (G24, PR: «Atún en agua + Avena»); el relleno «otra base…» tampoco debe aparecer con pools de su cocina.
@pytest.mark.parametrize("cc,prohibidos", [
    ("ES", ("Batata", "Yuca", "Harina de trigo", "Harina de maíz precocida", "Plátano", "Avena", "otra base")),
    ("MX", ("Yuca", "Harina de maíz precocida", "Avena", "otra base")),
    ("US", ("Yuca", "Plátano", "Harina de maíz precocida", "Avena", "otra base")),
    ("PR", ("Tortilla de maíz", "Avena", "otra base")),
    ("CO", ("Tortilla de maíz", "Avena", "otra base")),
])
def test_sembrador_no_asigna_bases_ajenas(paises, cc, prohibidos):
    from ai_helpers import get_deterministic_variety_prompt
    for i in range(12):
        uid = f"g24-850-{cc}-{i}"
        p = get_deterministic_variety_prompt("", _form(cc, uid), user_id=None, days_count=3)
        ops = _opciones(p)
        assert ops, p[:400]
        m = re.findall(r"\+ ([^\n]+?) y como acompañante|usa ([^—\n]+?) — NUNCA", ops)
        bases = [x for par in m for x in par if x]
        assert not [b for b in bases if any(pr in b for pr in prohibidos)], (cc, uid, bases)


# ─── 3. técnicas ────────────────────────────────────────────────────────────────────────────────────────────────────
_DO_TECNICA = re.compile(r"criolla|ropa vieja|canoas|dominican|tropical", re.I)


@pytest.mark.parametrize("cc", ("ES", "US", "MX", "CO"))
def test_tecnicas_sin_etiquetas_dominicanas(paises, cc):
    import graph_orchestrator as go
    vistas = set()
    for seed in range(40):
        random.seed(seed)
        for tiempo in ("30min", "none", None):
            sel = go._select_techniques(None, cooking_time=tiempo, cultura=cc)
            assert len(sel) == 3 and len(set(sel)) == 3, sel
            vistas |= set(sel)
    assert not [t for t in vistas if _DO_TECNICA.search(t)], (cc, sorted(vistas))


def test_tecnicas_pr_conserva_lo_puertorriqueno_sin_gentilicio(paises):
    import cocina_del_perfil as cdp
    tec, fam = cdp.tecnicas_de_la_cocina("30min", "PR")
    assert "Wrap o Burrito Dominicano" not in tec and "Wrap o Burrito" in tec
    assert "Relleno (Ej. Canoas, Vegetales rellenos)" in tec     # las canoas de plátano también son de Puerto Rico
    assert all(fam.get(t) for t in tec)


def test_tecnicas_do_identicas_byte_a_byte(paises):
    import constants
    import graph_orchestrator as go
    import cocina_del_perfil as cdp
    for tiempo in ("30min", "none", None, "plenty"):
        tec, fam = cdp.tecnicas_de_la_cocina(tiempo, "DO")
        assert tec == constants.techniques_for_cooking_time(tiempo) and fam is constants.TECH_TO_FAMILY
        assert cdp.tecnicas_de_la_cocina(tiempo, None)[1] is constants.TECH_TO_FAMILY
    for seed in range(25):
        random.seed(seed)
        a = go._select_techniques(None, cooking_time="30min")
        random.seed(seed)
        b = go._select_techniques(None, cooking_time="30min", cultura="DO")
        assert a == b
    paises.setenv(_KNOB, "false")
    for seed in range(10):
        random.seed(seed)
        a = go._select_techniques(None, cooking_time="30min")
        random.seed(seed)
        assert go._select_techniques(None, cooking_time="30min", cultura="ES") == a


def test_el_planificador_pasa_la_cocina_al_selector():
    src = _src("graph_orchestrator.py")
    assert re.search(r"_select_techniques\(_uid, succ_techs, aban_techs, cooking_time=form_data\.get\(\"cookingTime\"\),\s*"
                     r"cultura=cultural_country_for_form_data\(form_data\)\)", src)


# ─── 4. el bloque de micronutrientes del día ────────────────────────────────────────────────────────────────────────
def test_micros_por_cocina(paises):
    from micronutrients import build_micronutrient_targets_directive as b
    base = b(sex="female", age=30)
    assert "comidas dominicanas variadas y sabrosas" in base and "guineo, plátano, batata" in base
    assert b(sex="female", age=30, cocina="DO") == base and b(sex="female", age=30, cocina=None) == base
    es = b(sex="female", age=30, cocina="ES")
    assert "dominican" not in es.lower() and "guineo" not in es and "habichuela" not in es
    assert "cocina española" in es and "patata" in es
    pr = b(sex="female", age=30, cocina="PR")
    assert "cocina puertorriqueña" in pr and "guineo" in pr and "dominican" not in pr.lower()
    for cc in ("US", "MX", "CO"):
        t = b(sex="female", age=30, cocina=cc, goal="gain_muscle")
        assert "dominican" not in t.lower() and "guineo" not in t and "habichuela" not in t, cc
    paises.setenv(_KNOB, "false")
    assert b(sex="female", age=30, cocina="ES") == base


def test_micros_los_dos_llamadores_pasan_la_cocina():
    src = _src("graph_orchestrator.py")
    assert src.count("cocina=cultural_country_for_form_data(form_data))") >= 2


# ─── 5. prompt de sistema del generador de días ─────────────────────────────────────────────────────────────────────
_SURVIVOR_RE = re.compile(r"TÉCNICA CORRECTA POR ALIMENTO \[P1-CASABE-NO-BOIL[^\n]*")


def _sistema(cc, dieta="balanced"):
    import graph_orchestrator as go
    f = _form(cc)
    f["dietType"] = dieta
    return _SURVIVOR_RE.sub("", go._day_system_instruction_for_diet(f))


@pytest.mark.parametrize("dieta", ("balanced", "vegetarian", "vegan"))
@pytest.mark.parametrize("cc", ("ES", "US", "MX", "CO"))
def test_sistema_beta_sin_lexico_dominicano(paises, cc, dieta):
    t = _sistema(cc, dieta).lower()
    # la tabla de macros (clave «habichuela» del MOCK) es un dato, no un ejemplo
    t = re.sub(r"\n  - habichuela: [^\n]*", "", t)
    for w in ("mangú", "habichuela", "gandul", "guineo", "casabe"):
        assert w not in t, (cc, dieta, w, t[max(0, t.find(w) - 120): t.find(w) + 80])


def test_sistema_pr_sin_mangu_ni_casabe_con_su_lexico(paises):
    t = _sistema("PR").lower()
    assert "mangú" not in t and "casabe" not in t
    assert "habichuelas" in t and "guineo" in t          # palabras de Puerto Rico


def test_sistema_do_identico_y_knob_apagado(paises):
    import graph_orchestrator as go
    from prompts.day_generator import build_day_generator_system_prompt as bd
    base_do = go._day_system_instruction_for_diet(_form("DO"))
    assert base_do == go._DAY_SYSTEM_INSTRUCTION_CACHED
    assert "Mangú/tubérculos" in base_do
    for d in ("vegetarian", "vegan"):
        f = _form("DO"); f["dietType"] = d
        assert go._day_system_instruction_for_diet(f) == bd(d, "DO") + go._DAY_SCHEMA_INSTRUCTION + go._NUTRITION_LOOKUP_INSTRUCTION
    paises.setenv(_KNOB, "false")
    assert "Mangú/tubérculos" in go._day_system_instruction_for_diet(_form("ES"))
    assert "NO elijas mangú" in bd("balanced", "ES")


# ─── 6. planificador y prompt de variedad ───────────────────────────────────────────────────────────────────────────
def test_planificador_beta_sin_mangu_do_intacto(paises):
    from prompts.planner import build_planner_system_prompt as bp, PLANNER_SYSTEM_PROMPT
    assert bp("DO") is PLANNER_SYSTEM_PROMPT
    for cc in _BETA:
        out = bp(cc)
        assert "mangú" not in out.lower(), cc
        assert "Ejemplo INCORRECTO: Día 1=Avena con fresas (B)" in out
    paises.setenv(_KNOB, "false")
    assert "Mangú de plátano (A)" in bp("ES")


def test_variedad_beta_sin_mangu_ni_viver(paises):
    from prompts.preferences import build_deterministic_variety_prompt as bv, DETERMINISTIC_VARIETY_PROMPT
    assert bv(3, "DO") == DETERMINISTIC_VARIETY_PROMPT
    for cc in _BETA:
        out = bv(3, cc).lower()
        assert "mangú" not in out and "víver" not in out, cc
    paises.setenv(_KNOB, "false")
    assert "mangú" in bv(3, "ES")


# ─── 7. convenciones ────────────────────────────────────────────────────────────────────────────────────────────────
def test_knob_documentado_y_marker():
    assert _KNOB in _src("docs/knobs_reference.md")
    for rel in ("cocina_del_perfil.py", "cultural_profiles.py", "constants.py", "graph_orchestrator.py",
                "micronutrients.py", "prompts/day_generator.py", "prompts/planner.py", "prompts/preferences.py"):
        assert "P1-PLAN-LOTE-850" in _src(rel), rel
