# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-748 · 2026-09-28] G24, parte determinista: el bloque DINÁMICO del día ya no habla en dominicano a
los cinco países beta.

Auditoría del 28-sep sobre `build_day_assignment_context` (el tramo del prompt que el generador lee al final, el más
concreto): con el país en ES/US/MX/PR/CO seguía diciendo
  · «en la mesa dominicana eso no es un almuerzo ni una cena» (sale en cuanto hay avena en los carbos);
  · «casabe» entre las alternativas de merienda, en el plan B de «estas son OK siempre» y en la línea de alergias —
    el catálogo beta EXCLUYE el Casabe (`graph_orchestrator._BETA_CATALOG_DO_EXCLUSIVE_NAMES`);
  · «Salami dominicano» en la lista de proteínas prohibidas;
  · «Tubérculos/plátano» como etiqueta de la categoría A del desayuno y «NO uses tubérculo/plátano» en las otras
    cuatro: a una española «plátano» es la BANANA, así que empujaba «tortilla… con guineo» y le vetaba el guineo del
    desayuno los otros días. Y el brief de los otros días le pasaba el enum crudo «Mangú/Tubérculos».
Y el system prompt justificaba el «arroz de noche» con «no se acostumbra en la cena dominicana y el gate lo rechaza»,
cuando en beta la regla es blanda (`constants.slot_rules_for_country`) y sólo avisa.

Contrato: la rama DO queda BYTE-IDÉNTICA (huellas sha256 capturadas con el código de ANTES de este lote), el gate NO
cambia, y la etiqueta A de beta sale del desayuno típico de `cultural_profiles.PROFILES` (la categoría del esquema,
`schemas.py`, no se toca). tooltip-anchor: P1-PLAN-LOTE-748
"""
from __future__ import annotations

import copy
import hashlib
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_BETA = ("ES", "US", "MX", "PR", "CO")
_CATEGORIAS = ("Mangú/Tubérculos", "Avena/Cereales", "Pan/Tostadas", "Batido/Bowl", "Revoltillo/Tortilla", "Libre")


def _esqueleto(categoria="Mangú/Tubérculos", carbs=("Avena", "Yuca")):
    return {"brief_concept": "Día casero", "assigned_technique": "Guiso",
            "protein_pool": ["Pollo", "Huevos"], "carb_pool": list(carbs), "fruit_pool": ["Guineo"],
            "meal_types": ["Desayuno", "Almuerzo", "Merienda", "Cena"], "breakfast_category": categoria,
            "_other_days_brief": [{"technique": "salteado", "breakfast": "Mangú/Tubérculos"},
                                  {"technique": "horno", "breakfast": "Avena/Cereales"}]}


def _escenarios():
    """(nombre, esqueleto, kwargs). Cubre las cinco superficies del lote en la rama DO."""
    out = [("avena_mangu", _esqueleto(), dict(day_name="Lunes"))]
    out.append(("vetos_todo", _esqueleto("Pan/Tostadas", ("Papa", "Arroz")),
                dict(day_name="Martes", allergies=["lácteos", "huevo", "maní", "frutos secos"], dislikes=["yogurt"])))
    for i, c in enumerate(_CATEGORIAS):
        out.append((f"cat_{i}", _esqueleto(c, ("Papa", "Arroz")), dict(day_name="Lunes")))
    return out


@pytest.fixture
def entorno_fijo(monkeypatch):
    """Todo lo que el render lee de FUERA de este lote, congelado: knobs por defecto, biblioteca de inspiración apagada
    (su contenido depende de ficheros de datos) y el vocabulario de alergias sustituido por uno fijo — la huella mide
    SÓLO las ramas del prompt, no el diccionario de alérgenos de otro lote."""
    for k in ("MEALFIT_DAYGEN_DINNER_IDENTITY", "MEALFIT_DAYGEN_PROTEIN_DIVERSITY",
              "MEALFIT_DAYGEN_SLOT_TARGETS_IN_PROMPT"):
        monkeypatch.delenv(k, raising=False)
    import dish_library
    monkeypatch.setattr(dish_library, "DISH_LIBRARY_ENABLED", False)
    import graph_orchestrator as go
    _vocab = {"lácteos": {"leche", "queso", "yogur", "yogurt"}, "huevo": {"huevo", "huevos", "clara", "claras"},
              "maní": {"mani", "maní", "mantequilla de maní"}, "frutos secos": {"frutos secos", "nueces", "almendras"},
              "yogurt": {"yogur", "yogurt"}}

    def _expand(alg):
        out = set()
        for a in alg or []:
            out |= _vocab.get(str(a).lower(), {str(a).lower()})
        return out

    def _banned(item, alg):
        it = str(item).lower()
        return any(t in it for t in _expand(alg))

    monkeypatch.setattr(go, "_expand_allergy_declarations", _expand)
    monkeypatch.setattr(go, "_allergen_pool_item_banned", _banned)
    return go


def _render(country, nombre_filtro=None):
    from prompts.day_generator import build_day_assignment_context as bdac
    res = {}
    for nombre, sk, kw in _escenarios():
        if nombre_filtro and nombre != nombre_filtro:
            continue
        kw = dict(kw)
        if country is not None:
            kw["country"] = country
        res[nombre] = bdac(copy.deepcopy(sk), 1, **kw)
    return res


# Huellas de la rama DO capturadas con el código de origin/main ANTES de este lote (3ce2c235), con `entorno_fijo`.
# Si un lote futuro cambia A PROPÓSITO el texto dominicano de la asignación del día, se actualizan aquí — y ese cambio
# se ve en el diff. Si cambian sin querer, este test es el que lo dice.
_HUELLAS_DO_ANTES = {
    "avena_mangu": "5e75ce694a335dc2b62717313b73208722fcae3f4d76b24024e16e4d28cf3801",
    "vetos_todo": "43d937e635e896d510eb5e8385944bf6194ff58671bc69e8d4ffb0905ba07bbd",
    "cat_0": "837515930e91868e70f6fd6e5fe0eca2555394c05b2a829487e4a1178c367437",
    "cat_1": "63a6c8cc70367af15a7b6589d655ad70c5383256302fd024dd0e89400e66b8e3",
    "cat_2": "1fa257a86e52b28ffbd3b20cce45e219e99b2a3d35475b6f108e1ef4d98e6f1c",
    "cat_3": "33e5f672c380ccb96ce05d65ebac1293ff02e868b77098daa228164cdb259279",
    "cat_4": "de6b2dacae6768a7e58d70ff5f266e0cab31312aef584c2d5c4cf99540e28f0a",
    "cat_5": "4a441e83f51f38a75839ec946e7fdd5ba1852271dbcb5ef8898db6d2779ebcdc",
}


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


# ─────────────── A. la rama dominicana, byte a byte ───────────────

def test_rama_do_byte_identica_a_la_de_antes(entorno_fijo):
    actuales = {k: _sha(v) for k, v in _render("DO").items()}
    assert actuales == _HUELLAS_DO_ANTES, actuales


def test_sin_pais_es_la_rama_do(entorno_fijo):
    assert _render(None) == _render("DO")


def test_la_rama_do_conserva_sus_literales(entorno_fijo):
    r = _render("DO")
    assert "en la mesa dominicana eso no es un almuerzo ni una cena, y el plan se rechaza." in r["avena_mangu"]
    assert "Nada de tortitas, arepitas, bowls salados" in r["avena_mangu"]
    assert "pan integral, casabe, tostada de maíz" in r["avena_mangu"]
    assert "Salami dominicano" in r["avena_mangu"]
    assert "CATEGORÍA DE DESAYUNO ASIGNADA: Mangú/Tubérculos" in r["avena_mangu"]
    assert "NO uses mangú/tubérculos si la categoría asignada es otra" in r["cat_1"]
    assert "(desayuno: Mangú/Tubérculos)" in r["avena_mangu"]
    assert "casabe con aguacate" in r["vetos_todo"]


def test_plan_b_do_intacto_con_todo_vetado(entorno_fijo, monkeypatch):
    """Todo lo de «OK siempre» vetado ⇒ plan B. En DO sigue siendo fruta, casabe, aguacate."""
    import prompts.day_generator as dg
    monkeypatch.setattr(dg, "_vetado", lambda item, vetos: True)
    sk = _esqueleto()
    sk["protein_pool"] = ["Lentejas"]
    do = dg.build_day_assignment_context(copy.deepcopy(sk), 1, allergies=["x"], country="DO")
    assert "Para diversificar desayuno/merienda usa: fruta, casabe, aguacate (estas son OK siempre" in do
    for cc in _BETA:
        beta = dg.build_day_assignment_context(copy.deepcopy(sk), 1, allergies=["x"], country=cc)
        assert "casabe" not in beta.casefold(), cc
        assert "Para diversificar desayuno/merienda usa: fruta, aguacate (estas son OK siempre" in beta, cc


# ─────────────── B. los cinco países beta ───────────────

def _sin_inspiracion(t: str) -> str:
    return re.sub(r"🍽️ INSPIRACI.*?(?=\n⛔|\nDEBES)", "", t, flags=re.S)


@pytest.mark.parametrize("cc", _BETA)
def test_beta_sin_gentilicio_ni_casabe_ni_mangu(entorno_fijo, cc):
    for nombre, txt in _render(cc).items():
        t = _sin_inspiracion(txt).casefold()
        for tok in ("dominican", "casabe", "arepitas", "mangú", "mesa dominicana"):
            assert tok not in t, (cc, nombre, tok)


@pytest.mark.parametrize("cc", _BETA)
def test_beta_avena_con_motivo_neutro(entorno_fijo, cc):
    from prompts import asignacion_pais as ap
    t = _render(cc, "avena_mangu")["avena_mangu"]
    assert ap.AVENA_CIERRE_BETA in t
    assert "nunca el plato principal del almuerzo ni de la cena" in t, "la prohibición se queda"


@pytest.mark.parametrize("cc", _BETA)
def test_beta_alternativas_de_merienda_sin_casabe(entorno_fijo, cc):
    t = _render(cc, "avena_mangu")["avena_mangu"]
    assert "(usa fruta con lácteo, pan integral, tostada de maíz, frutos secos o yogur)" in t


@pytest.mark.parametrize("cc", _BETA)
def test_beta_salami_sin_gentilicio(entorno_fijo, cc):
    t = _render(cc, "avena_mangu")["avena_mangu"]
    linea = re.search(r"→ (.+)", t).group(1)
    assert "Salami" in linea.split(", "), linea
    assert "Salami dominicano" not in linea


@pytest.mark.parametrize("cc", _BETA)
def test_beta_linea_de_alergias_sin_casabe(entorno_fijo, cc):
    from prompts.day_generator import allergy_hard_line
    do = allergy_hard_line(["lácteos"])
    beta = allergy_hard_line(["lácteos"], country=cc)
    assert "casabe con aguacate" in do
    assert "casabe" not in beta.casefold()
    assert "tostada integral con aguacate" in beta
    assert allergy_hard_line(["lácteos"], country="DO") == do


# ─────────────── C. la etiqueta del desayuno sale de PROFILES ───────────────

def _desayuno_tipico(cc):
    from cultural_profiles import PROFILES, profile_for_market
    return PROFILES[profile_for_market(cc)]["slot_affinity"]["desayuno"]


@pytest.mark.parametrize("cc", _BETA)
def test_beta_etiqueta_a_es_el_desayuno_tipico_del_pais(entorno_fijo, cc):
    t = _render(cc, "cat_0")["cat_0"]
    esperado = f"CATEGORÍA DE DESAYUNO ASIGNADA: Desayuno típico local ({', '.join(_desayuno_tipico(cc))})"
    assert esperado in t, t
    assert "Tubérculos/plátano" not in t and "tubérculo/plátano" not in t


def test_espana_ya_no_lee_platano_en_el_desayuno(entorno_fijo):
    for nombre, t in _render("ES").items():
        assert "plátano" not in _sin_inspiracion(t).casefold(), nombre


@pytest.mark.parametrize("cc", _BETA)
def test_beta_las_otras_categorias_no_cambian_de_nombre(entorno_fijo, cc):
    r = _render(cc)
    for i, c in enumerate(_CATEGORIAS[1:], start=1):
        assert f"CATEGORÍA DE DESAYUNO ASIGNADA: {c}\n" in r[f"cat_{i}"], (cc, c)


@pytest.mark.parametrize("cc", _BETA)
def test_beta_brief_de_otros_dias_con_la_etiqueta_corta(entorno_fijo, cc):
    t = _render(cc, "avena_mangu")["avena_mangu"]
    assert "salteado (desayuno: Desayuno típico local); horno (desayuno: Avena/Cereales)" in t


def test_el_enum_del_esquema_no_se_muta(entorno_fijo):
    from prompts.day_generator import build_day_assignment_context as bdac
    sk = _esqueleto()
    antes = copy.deepcopy(sk)
    for cc in _BETA:
        bdac(sk, 1, country=cc)
    assert sk == antes


def test_perfil_sin_desayuno_cae_a_la_etiqueta_neutra(monkeypatch):
    import cultural_profiles
    from prompts import asignacion_pais as ap
    monkeypatch.setitem(cultural_profiles.PROFILES["spain_mediterranea"], "slot_affinity", {})
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "ES") == ap.ETIQUETA_A_BETA_RESPALDO
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "DO") == "Mangú/Tubérculos"
    assert ap.etiqueta_desayuno("Pan/Tostadas", "ES") == "Pan/Tostadas"


# ─────────────── D. el system prompt: el arroz de noche con motivo neutro, el gate igual ───────────────

@pytest.mark.parametrize("cc", _BETA)
@pytest.mark.parametrize("dieta", ["balanced", "vegetarian", "vegan", "pescatarian"])
def test_system_prompt_beta_arroz_de_noche_motivo_neutro(cc, dieta):
    from prompts.day_generator import build_day_generator_system_prompt as build
    from prompts import asignacion_pais as ap
    out = build(dieta, cc)
    assert "no se acostumbra en la cena dominicana" not in out
    assert ap.ARROZ_NOCHE_MOTIVO_BETA in out
    assert 'PROHIBIDO el "ARROZ DE NOCHE"' in out, "la prohibición se queda"


def test_system_prompt_do_intacto():
    from prompts.day_generator import build_day_generator_system_prompt as build, DAY_GENERATOR_SYSTEM_PROMPT
    from prompts import asignacion_pais as ap
    assert build("balanced", "DO") is DAY_GENERATOR_SYSTEM_PROMPT
    assert ap.ARROZ_NOCHE_MOTIVO_DO in DAY_GENERATOR_SYSTEM_PROMPT
    assert ap.ARROZ_NOCHE_MOTIVO_DO == "(no se acostumbra en la cena dominicana y el gate lo rechaza)"


def test_el_gate_del_arroz_de_noche_no_cambia():
    import constants
    regla = next(r for r in constants.SLOT_INAPPROPRIATE_FOODS["cena"] if "arroz de noche" in r["label"])
    assert regla["hardness"] == "soft"
    assert constants.slot_rules_for_country("DO") is constants.SLOT_INAPPROPRIATE_FOODS
    for cc in _BETA:
        assert all(r["hardness"] == "soft" for r in constants.slot_rules_for_country(cc)["cena"])


# ─────────────── E. anclas ───────────────

def test_marker_en_el_codigo():
    src_dg = (_BACKEND / "prompts" / "day_generator.py").read_text(encoding="utf-8")
    src_ap = (_BACKEND / "prompts" / "asignacion_pais.py").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-748" in src_dg and "tooltip-anchor: P1-PLAN-LOTE-748" in src_ap
