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


# [P1-PLAN-LOTE-748 · ronda 1] La lista del perfil YA NO va entera: pierde lo que es base de otra categoría (avena,
# pan/tostada, huevo, yogur), las sopas (§15a beta las prohíbe en el desayuno) y lo que la alergia o la dieta vetan. En
# US y PR no queda nada propio ⇒ el respaldo. [ronda 2] La fruta tampoco es base propia (va en todo desayuno): España
# ⇒ el respaldo; y el maíz va nombrado (la puerta de alergias no lo ve en «arepa»). El contrato por país, a la vista:
_ETIQUETA_A_ESPERADA = {
    "ES": "Base de tubérculo local",
    "MX": "Desayuno típico local (frijoles, tortilla de maíz)",
    "CO": "Desayuno típico local (arepa de maíz)",
    "US": "Base de tubérculo local",
    "PR": "Base de tubérculo local",
}


@pytest.mark.parametrize("cc", _BETA)
def test_beta_etiqueta_a_es_el_desayuno_tipico_del_pais(entorno_fijo, cc):
    t = _render(cc, "cat_0")["cat_0"]
    assert f"CATEGORÍA DE DESAYUNO ASIGNADA: {_ETIQUETA_A_ESPERADA[cc]}\n" in t, t
    assert "Tubérculos/plátano" not in t and "tubérculo/plátano" not in t
    # y lo que queda sale del perfil de su cocina, no de un texto por país
    tipico = [x.casefold() for x in _desayuno_tipico(cc)]
    dentro = re.search(r"\((.*)\)", _ETIQUETA_A_ESPERADA[cc])
    for item in (dentro.group(1).split(", ") if dentro else []):
        assert item in tipico, (cc, item)


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
    corta = _ETIQUETA_A_ESPERADA[cc].split(" (")[0]
    assert f"salteado (desayuno: {corta}); horno (desayuno: Avena/Cereales)" in t


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


# ─────────────── F. ronda 1 de la revisión adversaria (P1-PLAN-LOTE-748) ───────────────
# 1 CRÍTICO: la etiqueta A de beta imponía alérgenos y productos animales (PR gluten+huevo ⇒ «avena, huevo, pan»; un
#   vegano ⇒ huevo/yogur/caldo). `desayuno_por_alergia.reasignar` usa la A como REFUGIO de esos usuarios.
# 2/3 IMPORTANTE: la A repetía las bases de las otras cuatro (en US y PR, todas) y en CO pedía «caldo», que §15a beta
#   prohíbe en el desayuno.
# 4 IMPORTANTE: «el validador de horario lo señala» — la avena en la cena la RECHAZA el revisor cultural también en beta.
# 5 IMPORTANTE (I16): el catálogo es del MERCADO y la dureza del rechazo es del GATE, no de la cocina del día.
# 6/7/8 MENOR: una derivación de país por render; el brief de otros días con la cocina de CADA día; «habichuelas» en
#   el bloque de dieta beta.

def _linea_a(t: str) -> str:
    return next(l for l in t.splitlines() if "CATEGORÍA DE DESAYUNO ASIGNADA" in l)


def _tiene(tok: str, texto: str) -> bool:
    from constants import strip_accents
    return re.search(rf"\b{tok}(?:s|es)?\b", strip_accents(texto.casefold())) is not None


@pytest.fixture
def vocab_real(monkeypatch):
    """El vocabulario de alergias REAL (el del revisor), sólo con la inspiración apagada."""
    for k in ("MEALFIT_DAYGEN_DINNER_IDENTITY", "MEALFIT_DAYGEN_PROTEIN_DIVERSITY",
              "MEALFIT_DAYGEN_SLOT_TARGETS_IN_PROMPT"):
        monkeypatch.delenv(k, raising=False)
    import dish_library
    monkeypatch.setattr(dish_library, "DISH_LIBRARY_ENABLED", False)


def _bdac(*a, **kw):
    from prompts.day_generator import build_day_assignment_context as bdac
    return bdac(*a, **kw)


# ── 1. alergias y dieta ──

def test_pr_gluten_y_huevo_la_etiqueta_a_no_impone_sus_alergenos(vocab_real):
    import desayuno_por_alergia as dpa
    dias = [{"day": 1, "breakfast_category": "Avena/Cereales"}, {"day": 2, "breakfast_category": "Revoltillo/Tortilla"}]
    dpa.reasignar(dias, {"allergies": ["gluten", "huevo"]})
    assert "Mangú/Tubérculos" in [d["breakfast_category"] for d in dias], "la A es el refugio de este usuario"
    linea = _linea_a(_bdac(_esqueleto(), 1, day_name="Lunes", allergies=["gluten", "huevo"], country="PR"))
    for tok in ("avena", "huevo", "pan", "tostada", "cereal"):
        assert not _tiene(tok, linea), (tok, linea)


@pytest.mark.parametrize("cc", _BETA)
def test_vegano_la_etiqueta_a_no_impone_huevo_yogur_ni_caldo(vocab_real, cc):
    sk = _esqueleto()
    sk["protein_pool"] = ["Lentejas", "Garbanzos"]
    linea = _linea_a(_bdac(sk, 1, day_name="Lunes", diet_type="vegan", country=cc))
    for tok in ("huevo", "yogur", "caldo", "queso", "leche"):
        assert not _tiene(tok, linea), (cc, tok, linea)


@pytest.mark.parametrize("cc", _BETA)
def test_vegetariano_la_etiqueta_a_sin_caldo(vocab_real, cc):
    linea = _linea_a(_bdac(_esqueleto(), 1, day_name="Lunes", diet_type="vegetarian", country=cc))
    assert not _tiene("caldo", linea), (cc, linea)


def test_etiqueta_a_filtra_por_la_puerta_de_vetos_y_cae_al_respaldo():
    from prompts import asignacion_pais as ap
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "MX", vetado=lambda i: "frijol" in i) == \
        "Desayuno típico local (tortilla de maíz)"
    # [ronda 3] el respaldo pasa por la MISMA puerta: una que lo veta todo veta también el tubérculo ⇒ la neutra
    solo_mx = lambda i: bool(re.search(r"ma[ií]z|elote|choclo|mazorca|maicena|frijol|habichuela|legum", i))  # noqa: E731
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "MX", vetado=solo_mx) == ap.ETIQUETA_A_BETA_RESPALDO
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "MX", vetado=solo_mx, detalle=False) == \
        ap.ETIQUETA_A_BETA_RESPALDO
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "MX", vetado=lambda i: True) == ap.ETIQUETA_A_BETA_NEUTRA
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "MX", vetado=lambda i: True, detalle=False) == \
        ap.ETIQUETA_A_BETA_NEUTRA
    # DO: el enum tal cual, pase lo que pase (su huella está arriba)
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "DO", vetado=lambda i: True, dieta="vegan") == "Mangú/Tubérculos"


def test_etiqueta_a_filtra_por_la_dieta_con_el_vocabulario_del_escaner(monkeypatch):
    import cultural_profiles
    from prompts import asignacion_pais as ap
    monkeypatch.setitem(cultural_profiles.PROFILES["mexico_casera"], "slot_affinity",
                        {"desayuno": ["chilaquiles", "queso fresco", "jamón", "frijoles"]})
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "MX", dieta="vegan") == \
        "Desayuno típico local (chilaquiles, frijoles)"
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "MX", dieta="vegetarian") == \
        "Desayuno típico local (chilaquiles, queso fresco, frijoles)"


# ── 2/3. la A no repite otra categoría ni pide sopa ──

_BASES_OTRAS = ("avena", "cereal", "granola", "pan", "tostada", "huevo", "yogur", "yogurt", "batido", "bowl", "revoltillo",
                "fruta")   # [ronda 2] la fruta va en todo desayuno: no es la base de la A


@pytest.mark.parametrize("cc", _BETA)
def test_la_etiqueta_a_no_repite_la_base_de_otra_categoria(entorno_fijo, cc):
    linea = _linea_a(_render(cc, "cat_0")["cat_0"])
    for tok in _BASES_OTRAS:
        assert not _tiene(tok, linea), (cc, tok, linea)


@pytest.mark.parametrize("cc", _BETA)
def test_la_etiqueta_a_no_pide_sopa_en_el_desayuno(entorno_fijo, cc):
    linea = _linea_a(_render(cc, "cat_0")["cat_0"])
    assert not _tiene("caldo", linea) and not _tiene("sopa", linea), (cc, linea)


def test_el_respaldo_no_ofrece_cereal_ni_platano():
    """El respaldo es el refugio del alérgico al gluten (`reasignar`): «cereal» no lo veta el vocabulario del gluten
    (`_allergen_pool_item_banned('cereal', ['gluten'])` es False) y además es la base de la B."""
    from prompts import asignacion_pais as ap
    r = ap.ETIQUETA_A_BETA_RESPALDO.casefold()
    assert "cereal" not in r and "plátano" not in r and "tubérculo" in r


# ── 4. la avena en la cena: el revisor la rechaza también en beta ──

@pytest.mark.parametrize("cc", _BETA)
def test_beta_avena_no_promete_solo_un_aviso(entorno_fijo, cc):
    t = _render(cc, "avena_mangu")["avena_mangu"]
    frase = re.search(r"es base de DESAYUNO o MERIENDA.*", t).group(0)
    assert "señala" not in frase, frase
    assert frase.endswith("eso no es un almuerzo ni una cena, y el revisor lo rechaza."), frase
    assert "arepitas" not in frase and "dominican" not in frase


# ── 5. I16: mercado ≠ cocina del día ≠ gate ──

def test_mercado_do_en_un_dia_de_cocina_espanola(entorno_fijo):
    """Usuario DO con cocina secundaria ES: su catálogo DO SÍ vende casabe y su gate DO reintenta hasta el final."""
    t = _bdac(_esqueleto(), 1, day_name="Lunes", country="ES", mercado="DO", pais_gate="DO")
    assert "pan integral, casabe, tostada de maíz" in t
    assert "Salami dominicano" in t
    frase = re.search(r"es base de DESAYUNO o MERIENDA.*", t).group(0)
    assert frase.endswith("eso no es un almuerzo ni una cena, y el plan se rechaza."), frase
    assert "mesa dominicana" not in frase and "arepitas" not in frase, "la cocina de ESTE día es española"


def test_mercado_beta_en_un_dia_de_cocina_dominicana(entorno_fijo):
    t = _bdac(_esqueleto(), 1, day_name="Lunes", country="DO", mercado="US", pais_gate="US")
    assert "casabe" not in _sin_inspiracion(t).casefold(), "el catálogo US no vende casabe"
    assert "Salami dominicano" not in t
    assert "en la mesa dominicana eso no es un almuerzo ni una cena, y el revisor lo rechaza." in t
    assert "CATEGORÍA DE DESAYUNO ASIGNADA: Mangú/Tubérculos\n" in t, "la cocina del día sigue siendo la dominicana"


def test_la_linea_de_alergias_sigue_al_mercado(entorno_fijo):
    assert "casabe con aguacate" in _bdac(_esqueleto(), 1, allergies=["lácteos"], country="ES", mercado="DO")
    linea = _bdac(_esqueleto(), 1, allergies=["lácteos"], country="DO", mercado="ES").split("• Concepto Temático")[0]
    assert "Sin lácteos" in linea and "casabe" not in linea.casefold()


@pytest.mark.parametrize("cc", ("DO", None) + _BETA)
def test_sin_mercado_ni_gate_es_la_conducta_de_antes(entorno_fijo, cc):
    for nombre, sk, kw in _escenarios():
        a = _bdac(copy.deepcopy(sk), 1, country=cc, **kw)
        b = _bdac(copy.deepcopy(sk), 1, country=cc, mercado=cc, pais_gate=cc, **kw)
        assert a == b, (cc, nombre)


def _llamadas(src: str, nombre: str) -> list:
    out, i = [], 0
    while True:
        i = src.find(nombre + "(", i)
        if i < 0:
            return out
        prof, j = 0, i + len(nombre)
        for j in range(i + len(nombre), len(src)):
            prof += {"(": 1, ")": -1}.get(src[j], 0)
            if prof == 0:
                break
        out.append(src[i:j + 1])
        i = j


def test_los_tres_llamadores_pasan_el_mercado_y_el_del_dia_el_gate():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    llamadas = [c for c in _llamadas(src, "build_day_assignment_context") if "skeleton_day" in c]
    assert len(llamadas) == 3, len(llamadas)
    for c in llamadas:
        assert "mercado=country_for_form_data(form_data)" in c, c[:200]
    del_dia = [c for c in llamadas if "day_index=" in c]
    assert len(del_dia) == 1 and "pais_gate=cultural_country_for_form_data(form_data)" in del_dia[0]


# ── 6. una derivación de país por render ──

def test_pais_corrupto_avisa_una_vez_por_render(entorno_fijo, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    _bdac(_esqueleto(), 1, day_name="Lunes", allergies=["lácteos"], country="Marte")
    n = sum("P2-COUNTRY-HOUSEKEEPING" in r.getMessage() for r in caplog.records)
    assert n == 1, n


# ── 7. el brief de los otros días, con la cocina de CADA día ──

def test_brief_de_otros_dias_con_la_cocina_de_cada_dia(entorno_fijo):
    sk = _esqueleto()
    # [ronda 2] CO y no ES: la A española ya es el respaldo (la fruta no es base propia); ES va en el tercer día
    sk["_other_days_brief"] = [{"technique": "salteado", "breakfast": "Mangú/Tubérculos", "country": "DO"},
                               {"technique": "horno", "breakfast": "Mangú/Tubérculos", "country": "CO"},
                               {"technique": "plancha", "breakfast": "Mangú/Tubérculos", "country": "ES"}]
    esperado = ("salteado (desayuno: Mangú/Tubérculos); horno (desayuno: Desayuno típico local); "
                "plancha (desayuno: Base de tubérculo local)")
    assert esperado in _bdac(copy.deepcopy(sk), 1, country="ES")
    assert esperado in _bdac(copy.deepcopy(sk), 1, country="DO")


def test_el_brief_lleva_la_cocina_de_cada_dia():
    # [ronda 2] derivada UNA vez por día antes de la comprensión (`_abd_paises`); el contrato completo en
    # test_r2_el_brief_deriva_la_cocina_una_vez_por_dia
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index('_abd_sd["_other_days_brief"] = [')
    bloque = src[i:src.index("for _abd_j", i)]
    assert '"country": _abd_paises[_abd_j]' in bloque, bloque


# ── 8. «habichuelas» en el bloque de dieta beta ──

@pytest.mark.parametrize("cc", ("ES", "US", "MX", "CO"))
@pytest.mark.parametrize("dieta", ("vegan", "vegetarian"))
def test_beta_dieta_veg_sin_habichuelas(entorno_fijo, cc, dieta):
    t = _sin_inspiracion(_bdac(_esqueleto(), 1, day_name="Lunes", diet_type=dieta, country=cc))
    assert "habichuela" not in t.casefold(), cc
    assert "frijoles/lentejas/garbanzos" in t


@pytest.mark.parametrize("cc", ("DO", "PR"))
def test_do_y_pr_conservan_habichuelas(entorno_fijo, cc):
    """En Puerto Rico «habichuelas» ES la palabra (está en los staples de su perfil)."""
    t = _bdac(_esqueleto(), 1, day_name="Lunes", diet_type="vegan", country=cc)
    assert "habichuelas/lentejas/garbanzos" in t


# ─────────────── G. ronda 2 de la revisión adversaria (P1-PLAN-LOTE-748) ───────────────
# 1 IMPORTANTE (seguridad clínica, regresión frente a main): la A imponía MAÍZ al alérgico al maíz. El vocabulario de
#   alergias no expande maíz → arepa/tortilla ni legumbres → frijoles (`_allergen_pool_item_banned('arepa', ['maíz'])`
#   es False, y el escáner tampoco ve «1 arepa mediana»): la etiqueta tiene que preguntar con el nombre CALIFICADO
#   («arepa de maíz», la base del catálogo, la clase «legumbres»). La causa de fondo —el vocabulario— es otro lote.
# 2 MENOR: el brief derivaba la cocina por cada par (día, otro día): 42 derivaciones por bloque de 7 días.
# 3 MENOR: la A de España quedaba en «(fruta)» — la fruta va en TODO desayuno («base sólida + proteína + fruta») y es
#   la base de la D (Batido/Bowl): no es una base propia ⇒ el respaldo.

def _linea_a_con(cc, alergias, **kw):
    return _linea_a(_bdac(_esqueleto(), 1, day_name="Lunes", allergies=alergias, country=cc, **kw))


@pytest.mark.parametrize("cc", ("MX", "CO"))
@pytest.mark.parametrize("alergias", (["maíz"], ["maiz"], ["Maíz"], ["gluten", "maíz"]))
def test_r2_maiz_la_etiqueta_a_no_impone_arepa_ni_tortilla(vocab_real, cc, alergias):
    linea = _linea_a_con(cc, alergias)
    for tok in ("arepa", "tortilla", "maiz"):
        assert not _tiene(tok, linea), (cc, alergias, linea)


def test_r2_gluten_y_maiz_el_refugio_no_trae_maiz(vocab_real):
    import desayuno_por_alergia as dpa
    dias = [{"day": 1, "breakfast_category": "Avena/Cereales"}, {"day": 2, "breakfast_category": "Pan/Tostadas"}]
    dpa.reasignar(dias, {"allergies": ["gluten", "maíz"]})
    assert "Mangú/Tubérculos" in [d["breakfast_category"] for d in dias], "la A es el refugio de este usuario"
    for cc in ("CO", "MX"):
        linea = _linea_a_con(cc, ["gluten", "maíz"])
        assert not _tiene("arepa", linea) and not _tiene("tortilla", linea), (cc, linea)


@pytest.mark.parametrize("alergias", (["legumbres"], ["legumbre"], ["leguminosas"], ["frijoles"], ["Legumbres"]))
def test_r2_legumbres_la_etiqueta_a_no_impone_frijoles(vocab_real, alergias):
    linea = _linea_a_con("MX", alergias)
    assert not _tiene("frijol", linea), (alergias, linea)


def test_r2_el_celiaco_conserva_la_arepa_de_maiz(vocab_real):
    """Sin sobre-filtrar el refugio del celíaco: la arepa de maíz no lleva gluten, ni la tortilla de maíz."""
    assert "arepa de maíz" in _linea_a_con("CO", ["gluten"])
    assert "tortilla de maíz" in _linea_a_con("MX", ["gluten"])
    # y a quien no declara nada, la A no se le recorta por la clase de legumbre
    assert "frijoles" in _linea_a_con("MX", [])


def test_r2_la_etiqueta_mexicana_no_se_confunde_con_la_tortilla_de_huevo(vocab_real):
    """«tortilla» a secas es la de HUEVO en la E (Revoltillo/Tortilla) y en el catálogo (`GLOBAL_REVERSE_MAP`): la A
    de México nombra la de maíz."""
    linea = _linea_a_con("MX", [])
    assert "tortilla de maíz" in linea and not re.search(r"tortilla(?! de maíz)", linea), linea


def test_r2_la_puerta_pregunta_tambien_por_la_base_y_la_clase(monkeypatch):
    """Defensa aunque el dato venga sin calificar: la base del catálogo (`constants.GLOBAL_REVERSE_MAP`: arepa ⇒
    harina de maíz precocida) y la clase (frijoles/habichuelas/lentejas ⇒ legumbres) también pasan por la puerta."""
    import cultural_profiles
    import graph_orchestrator as go
    from prompts import asignacion_pais as ap
    monkeypatch.setitem(cultural_profiles.PROFILES["colombia_casera"], "slot_affinity",
                        {"desayuno": ["arepa", "habichuelas", "lentejas", "chocolate"]})

    def puerta(alergias):
        return lambda i: go._allergen_pool_item_banned(i, alergias)
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "CO", vetado=puerta(["maíz"])) == \
        "Desayuno típico local (habichuelas, lentejas, chocolate)"
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "CO", vetado=puerta(["legumbres"])) == \
        "Desayuno típico local (arepa, chocolate)"
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "CO", vetado=puerta(["gluten"])) == \
        "Desayuno típico local (arepa, habichuelas, lentejas, chocolate)"
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "CO", vetado=puerta([])) == \
        "Desayuno típico local (arepa, habichuelas, lentejas, chocolate)"


def test_r2_espana_no_queda_en_solo_fruta(entorno_fijo):
    linea = _linea_a(_render("ES", "cat_0")["cat_0"])
    assert "(fruta)" not in linea, linea
    assert linea.endswith(f"ASIGNADA: {_ETIQUETA_A_ESPERADA['ES']}"), linea


@pytest.mark.parametrize("cc", _BETA)
def test_r2_la_fruta_no_es_la_base_de_la_a(entorno_fijo, cc):
    assert not _tiene("fruta", _linea_a(_render(cc, "cat_0")["cat_0"])), cc


def test_r2_el_brief_deriva_la_cocina_una_vez_por_dia():
    """La cocina de cada día se deriva UNA vez por día (7 derivaciones), no por cada par (día, otro día) (42): con un
    país corrupto eran 42 avisos P2-COUNTRY-HOUSEKEEPING por bloque."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index('_abd_sd["_other_days_brief"] = [')
    fin = src.index("for _abd_j", i)
    comprension = src[i:fin]
    assert "cultural_country_for_form_data" not in comprension, comprension
    assert '"country": _abd_paises[_abd_j]' in comprension, comprension
    ini = src.rindex("_abd_days = skeleton_days[:days_in_chunk]", 0, i)
    previo = src[ini:i]
    assert "_abd_paises = [cultural_country_for_form_data(form_data, day_index=" in previo, previo
    assert previo.count("cultural_country_for_form_data(") == 1, previo



# ─────────────── H. ronda 3 de la revisión adversaria (P1-PLAN-LOTE-748) ───────────────
# 1 IMPORTANTE (seguridad clínica, regresión frente a main): la defensa del «ingrediente base» no actuaba con los DATOS
#   REALES. `GLOBAL_REVERSE_MAP.get("arepa de maíz")` y `.get("tortilla de maíz")` son None — sólo «arepa» a secas
#   lleva a «harina de maíz precocida» —, y el test de la ronda 2 lo probaba con el dato sustituido por «arepa». CO con
#   alergia «harina de maíz» recibía «⚠️ OBLIGATORIO … (arepa de maíz)» y el escáner marcaba después «Harina de maíz
#   precocida»: un intento quemado, el fallo del lote 227. Estos tests NO sustituyen datos ni vocabulario.
# 2 MENOR: sinónimos que la puerta no veía — maíz (elote, choclo, mazorca, maicena) y frijoles (habichuelas, porotos,
#   judías, alubias).
# 3 MENOR: el respaldo «Base de tubérculo local» salía sin pasar por la puerta, y `reasignar` no apartaba la A del
#   alérgico a los tubérculos.

def _puerta_real(alergias):
    import graph_orchestrator as go
    return lambda i: go._allergen_pool_item_banned(i, alergias)


@pytest.mark.parametrize("alergias", (["harina de maíz"], ["Harina de Maíz"], ["harina de maiz"], ["maíz precocido"]))
def test_r3_co_harina_de_maiz_no_muestra_la_arepa(vocab_real, alergias):
    linea = _linea_a_con("CO", alergias)
    assert not _tiene("arepa", linea) and not _tiene("maiz", linea), (alergias, linea)
    from prompts import asignacion_pais as ap
    et = ap.etiqueta_desayuno("Mangú/Tubérculos", "CO", vetado=_puerta_real(alergias))
    assert "arepa" not in et.casefold(), (alergias, et)


@pytest.mark.parametrize("alergias", (["harina de maíz"], ["elote"], ["maíz precocido"]))
def test_r3_mx_harina_de_maiz_o_elote_no_muestra_la_tortilla_de_maiz(vocab_real, alergias):
    linea = _linea_a_con("MX", alergias)
    assert not _tiene("tortilla", linea) and not _tiene("maiz", linea), (alergias, linea)
    assert "frijoles" in linea, ("los frijoles no llevan maíz", linea)


@pytest.mark.parametrize("alergias", (["huevo"], ["Huevos"], ["huevo", "gluten"]))
def test_r3_el_alergico_al_huevo_conserva_la_tortilla_y_la_arepa_de_maiz(vocab_real, alergias):
    """«tortilla» a secas lleva a «huevos» en el catálogo (`GLOBAL_REVERSE_MAP`): la base del nombre sin calificar NO
    puede arrastrarla — la tortilla de maíz no lleva huevo."""
    assert "tortilla de maíz" in _linea_a_con("MX", alergias), alergias
    assert "arepa de maíz" in _linea_a_con("CO", alergias), alergias


def test_r3_la_puerta_pregunta_por_la_base_con_los_datos_reales():
    """Sin monkeypatch: el dato de producción es «arepa de maíz» / «tortilla de maíz»."""
    import cultural_profiles
    from prompts import asignacion_pais as ap
    tipicos = {pid: (p.get("slot_affinity") or {}).get("desayuno") or [] for pid, p in cultural_profiles.PROFILES.items()}
    assert "arepa de maíz" in tipicos["colombia_casera"] and "tortilla de maíz" in tipicos["mexico_casera"]
    arepa = [n.casefold() for n in ap._nombres_para_la_puerta("arepa de maíz")]
    tortilla = [n.casefold() for n in ap._nombres_para_la_puerta("tortilla de maíz")]
    assert "harina de maíz precocida" in arepa, arepa
    assert "harina de maíz precocida" in tortilla, tortilla
    assert not any("huevo" in n for n in tortilla), ("la tortilla de maíz no es la de huevo", tortilla)


@pytest.mark.parametrize("cc,alergia", [("CO", a) for a in ("elote", "elotes", "choclo", "mazorca", "maicena")] +
                                       [("MX", a) for a in ("elote", "elotes", "choclo", "mazorca", "maicena")])
def test_r3_sinonimos_del_maiz(vocab_real, cc, alergia):
    linea = _linea_a_con(cc, [alergia])
    for tok in ("arepa", "tortilla", "maiz"):
        assert not _tiene(tok, linea), (cc, alergia, linea)


@pytest.mark.parametrize("alergia", ("habichuelas", "habichuela", "porotos", "poroto", "judías", "judias", "alubias",
                                     "alubia"))
def test_r3_sinonimos_de_los_frijoles(vocab_real, alergia):
    linea = _linea_a_con("MX", [alergia])
    assert not _tiene("frijol", linea), (alergia, linea)
    assert "tortilla de maíz" in linea, ("la tortilla de maíz no es una legumbre", alergia, linea)


def test_r3_los_sinonimos_no_recortan_a_quien_no_los_declara(vocab_real):
    for alergias in ([], ["gluten"], ["lácteos"], ["maní"], ["soya"], ["frutos secos"], ["mariscos"]):
        assert "frijoles, tortilla de maíz" in _linea_a_con("MX", alergias), alergias
        assert "arepa de maíz" in _linea_a_con("CO", alergias), alergias


_TUBERCULOS_DECLARADOS = (["papa"], ["Papas"], ["patata"], ["yuca"], ["tubérculos"], ["tubérculo"], ["Tuberculos"],
                          ["batata"])


@pytest.mark.parametrize("cc", ("ES", "US", "PR"))
@pytest.mark.parametrize("alergias", _TUBERCULOS_DECLARADOS)
def test_r3_el_respaldo_pasa_por_la_puerta(vocab_real, cc, alergias):
    from prompts import asignacion_pais as ap
    linea = _linea_a_con(cc, alergias)
    assert linea.endswith(f"ASIGNADA: {ap.ETIQUETA_A_BETA_NEUTRA}"), (cc, alergias, linea)
    for tok in ("tuberculo", "papa", "patata", "yuca", "batata"):
        assert not _tiene(tok, linea), (cc, alergias, tok, linea)


def test_r3_co_maiz_y_papa_no_cae_en_el_tuberculo(vocab_real):
    from prompts import asignacion_pais as ap
    assert _linea_a_con("CO", ["maíz"]).endswith(f"ASIGNADA: {ap.ETIQUETA_A_BETA_RESPALDO}")
    assert _linea_a_con("CO", ["maíz", "papa"]).endswith(f"ASIGNADA: {ap.ETIQUETA_A_BETA_NEUTRA}")


def test_r3_el_brief_de_otros_dias_tambien_pasa_el_respaldo_por_la_puerta(vocab_real):
    from prompts import asignacion_pais as ap
    brief = [{"technique": "horno", "breakfast": "Mangú/Tubérculos", "country": "ES"}]
    sk = _esqueleto("Batido/Bowl")
    sk["_other_days_brief"] = copy.deepcopy(brief)
    t = _bdac(sk, 1, day_name="Lunes", allergies=["papa"], country="ES")
    assert f"horno (desayuno: {ap.ETIQUETA_A_BETA_NEUTRA})" in t, t
    sk = _esqueleto("Batido/Bowl")
    sk["_other_days_brief"] = copy.deepcopy(brief)
    t = _bdac(sk, 1, day_name="Lunes", allergies=["gluten"], country="ES")
    assert f"horno (desayuno: {ap.ETIQUETA_A_BETA_RESPALDO})" in t, "sin alergia a tubérculos, el respaldo de siempre"


def test_r3_sin_alergias_el_respaldo_no_cambia(vocab_real):
    from prompts import asignacion_pais as ap
    for cc in ("ES", "US", "PR"):
        assert _linea_a_con(cc, []).endswith(f"ASIGNADA: {ap.ETIQUETA_A_BETA_RESPALDO}"), cc
        assert _linea_a_con(cc, ["gluten", "huevo"]).endswith(f"ASIGNADA: {ap.ETIQUETA_A_BETA_RESPALDO}"), cc
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "ES") == ap.ETIQUETA_A_BETA_RESPALDO
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "DO", vetado=lambda i: True) == "Mangú/Tubérculos"


def test_r3_la_etiqueta_neutra_no_impone_un_alergeno():
    """La neutra no nombra un alimento: ni un tubérculo, ni el maíz, ni la base de otra categoría, ni «lo típico
    local» (en Colombia, la arepa)."""
    import graph_orchestrator as go
    from prompts import asignacion_pais as ap
    n = ap.ETIQUETA_A_BETA_NEUTRA
    for tok in ("tuberculo", "papa", "patata", "yuca", "batata", "maiz", "arepa", "tortilla", "huevo", "avena", "pan",
                "cereal", "trigo", "leche", "queso", "yogur", "frijol", "platano", "fruta", "local", "tipico"):
        assert not _tiene(tok, n), (tok, n)
    todas = ["gluten", "huevo", "lácteos", "maní", "frutos secos", "soya", "mariscos", "pescado", "sésamo", "maíz",
             "legumbres", "papa", "yuca", "tubérculos"]
    assert not go._allergen_pool_item_banned(n, todas), n
    assert n != ap.ETIQUETA_A_BETA_RESPALDO


def test_r3_reasignar_aparta_la_a_del_alergico_a_los_tuberculos():
    import desayuno_por_alergia as dpa
    for alergias in (["tubérculos"], ["tubérculo"], ["Tuberculos"]):
        dias = [{"day": 1, "breakfast_category": "Mangú/Tubérculos"}, {"day": 2, "breakfast_category": "Avena/Cereales"}]
        assert dpa.reasignar(dias, {"allergies": alergias}) == 1, alergias
        assert "Mangú/Tubérculos" not in [d["breakfast_category"] for d in dias], (alergias, dias)
    # gluten + huevo + tubérculos: queda el Batido/Bowl
    dias = [{"day": i, "breakfast_category": c} for i, c in enumerate(
        ("Mangú/Tubérculos", "Avena/Cereales", "Pan/Tostadas", "Revoltillo/Tortilla", "Batido/Bowl"), 1)]
    dpa.reasignar(dias, {"allergies": ["gluten", "huevo", "tubérculos"]})
    assert {d["breakfast_category"] for d in dias} == {"Batido/Bowl"}, dias


def test_r3_reasignar_no_aparta_la_a_por_un_solo_tuberculo():
    """La A dominicana es el mangú (plátano): una alergia a la papa no la aparta. En beta la etiqueta del respaldo pasa
    por la puerta (test_r3_el_respaldo_pasa_por_la_puerta)."""
    import desayuno_por_alergia as dpa
    dias = [{"day": 1, "breakfast_category": "Mangú/Tubérculos"}]
    assert dpa.reasignar(dias, {"allergies": ["papa"]}) == 0
    assert dias[0]["breakfast_category"] == "Mangú/Tubérculos"


# ─────────────── I. ronda 4 de la revisión adversaria (P1-PLAN-LOTE-748) ───────────────
# 1 BLOQUEANTE (seguridad clínica, regresión frente a main: main nunca imponía maíz ni frijoles en la A beta): la puerta
#   sólo veía la alergia cuando lo DECLARADO cabía DENTRO de los nombres del ítem. Una declaración MÁS específica que el
#   ítem (calificativo, plural, «y derivados») se escapaba: «maíz amarillo», «almidón de maíz», «maíz y derivados»
#   ⇒ «⚠️ OBLIGATORIO … (arepa de maíz)» contra la línea de alergias del MISMO prompt; el rechazo «arepas» (la forma
#   natural) ⇒ arepa y reintento; «tortilla» ⇒ la puerta la expandía al gluten; «frijoles negros» ⇒ frijoles. Ahora la
#   puerta lee también las declaraciones CRUDAS: si una nombra la cabeza del ítem o una raíz de su familia, lo veta.
# 2 MENOR (rendimiento): ~150 llamadas al escáner por render (7 etiquetas × ~12 nombres + el respaldo), síncronas antes
#   de cada llamada al modelo. La puerta se memoriza dentro del render.
# Estos tests NO sustituyen datos ni vocabulario.

_MAIZ_MAS_ESPECIFICO = ("maíz amarillo", "maíz blanco", "almidón de maíz", "fécula de maíz", "masa de maíz",
                        "sémola de maíz", "pan de maíz", "maíz y derivados", "derivados del maíz",
                        "harina de maíz nixtamalizada", "Maíz Amarillo", "maiz blanco")


@pytest.mark.parametrize("cc", ("CO", "MX"))
@pytest.mark.parametrize("alergia", _MAIZ_MAS_ESPECIFICO)
def test_r4_la_declaracion_mas_especifica_que_el_item_veta_el_maiz(vocab_real, cc, alergia):
    linea = _linea_a_con(cc, [alergia])
    for tok in ("arepa", "tortilla", "maiz"):
        assert not _tiene(tok, linea), (cc, alergia, linea)
    if cc == "MX":
        assert "frijoles" in linea, ("los frijoles no llevan maíz", alergia, linea)


@pytest.mark.parametrize("rechazo", (["arepas"], ["Arepas"], ["arepa"], ["masa de arepa"], ["harina para arepas"]))
def test_r4_el_rechazo_arepas_no_impone_la_arepa(vocab_real, rechazo):
    linea = _linea_a_con("CO", [], dislikes=rechazo)
    assert not _tiene("arepa", linea), (rechazo, linea)


def test_r4_el_rechazo_arepas_quemaba_un_intento():
    """La razón del test de arriba: el guard de rechazos SÍ marca la arepa que la A pedía (el fallo del lote 227)."""
    from rechazos import _scan_dislike_violations
    plan = {"days": [{"meals": [{"name": "Desayuno", "ingredients": ["2 arepas de harina de maíz precocida"]}]}]}
    assert _scan_dislike_violations(plan, {"dislikes": ["arepas"]})


@pytest.mark.parametrize("vetos", (dict(dislikes=["tortilla"]), dict(dislikes=["tortillas"]),
                                   dict(dislikes=["Tortillas de maíz"]), dict(alergias=["tortilla"])))
def test_r4_la_tortilla_declarada_no_impone_la_tortilla_de_maiz(vocab_real, vetos):
    linea = _linea_a_con("MX", vetos.get("alergias", []), dislikes=vetos.get("dislikes"))
    assert not _tiene("tortilla", linea), (vetos, linea)
    assert "frijoles" in linea, (vetos, linea)


@pytest.mark.parametrize("campo", ("allergies", "dislikes"))
@pytest.mark.parametrize("decl", ("frijoles negros", "Frijoles refritos", "frijol negro", "frijol pinto",
                                  "Habichuelas rojas", "porotos negros", "judías blancas", "beans"))
def test_r4_los_frijoles_calificados_no_imponen_frijoles(vocab_real, campo, decl):
    kw = {campo: [decl]}
    linea = _linea_a(_bdac(_esqueleto(), 1, day_name="Lunes", country="MX", **kw))
    assert not _tiene("frijol", linea), (campo, decl, linea)
    assert "tortilla de maíz" in linea, ("la tortilla de maíz no es una legumbre", campo, decl, linea)


@pytest.mark.parametrize("alergia", ("Maseca", "masarepa", "polenta", "nixtamal", "cornmeal", "corn starch",
                                     "cornstarch", "maize"))
def test_r4_el_maiz_con_otro_nombre_tambien_cierra_la_puerta(vocab_real, alergia):
    """El vocabulario del escáner no los conoce (lote aparte); la puerta de la A, sí: es la que IMPONE el maíz."""
    for cc in ("CO", "MX"):
        linea = _linea_a_con(cc, [alergia])
        for tok in ("arepa", "tortilla", "maiz"):
            assert not _tiene(tok, linea), (cc, alergia, linea)


@pytest.mark.parametrize("vetos", (dict(alergias=["huevo"]), dict(alergias=["Huevos"]), dict(alergias=["clara de huevo"]),
                                   dict(alergias=["yema"]), dict(alergias=["egg"]), dict(alergias=["gluten"]),
                                   dict(alergias=["trigo"]), dict(alergias=["celíaco"]),
                                   dict(alergias=["harina de trigo"]), dict(alergias=["soya", "maní"]),
                                   dict(alergias=["tortillas de harina"]), dict(dislikes=["tortillas de harina"]),
                                   dict(dislikes=["tortilla de huevo"]), dict(dislikes=["arepa de huevo"]),
                                   dict(dislikes=["arepas de queso"]), dict(alergias=["lácteos"], dislikes=["cebolla"])))
def test_r4_sin_falsos_bloqueos(vocab_real, vetos):
    """Una tortilla de HARINA o de HUEVO no es la de maíz, y una arepa de huevo o de queso sigue siendo de maíz pero el
    rechazo nombra el PLATO, no la arepa: ni la puerta ni el escáner la ven en «2 arepas de harina de maíz precocida»."""
    alg, dis = vetos.get("alergias", []), vetos.get("dislikes")
    assert "frijoles, tortilla de maíz" in _linea_a_con("MX", alg, dislikes=dis), vetos
    assert "arepa de maíz" in _linea_a_con("CO", alg, dislikes=dis), vetos


@pytest.mark.parametrize("vetos", (dict(alergias=["tubérculos y derivados"]), dict(dislikes=["tubérculos"]),
                                   dict(dislikes=["tubérculos y derivados"]), dict(alergias=["raíces y tubérculos"])))
def test_r4_el_respaldo_con_la_declaracion_mas_especifica(vocab_real, vetos):
    from prompts import asignacion_pais as ap
    for cc in ("ES", "US", "PR"):
        linea = _linea_a_con(cc, vetos.get("alergias", []), dislikes=vetos.get("dislikes"))
        assert linea.endswith(f"ASIGNADA: {ap.ETIQUETA_A_BETA_NEUTRA}"), (cc, vetos, linea)


def test_r4_la_puerta_lee_las_declaraciones_crudas():
    """Contrato de la función: sin `declarados` se comporta como antes; con ellos, veta el ítem que la declaración
    nombra por su cabeza o su familia. DO no cambia nunca."""
    from prompts import asignacion_pais as ap
    nunca = lambda i: False                                                    # noqa: E731
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "MX", vetado=nunca) == \
        "Desayuno típico local (frijoles, tortilla de maíz)"
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "MX", vetado=nunca, declarados=["maíz amarillo"]) == \
        "Desayuno típico local (frijoles)"
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "MX", vetado=nunca, declarados=["frijoles negros"]) == \
        "Desayuno típico local (tortilla de maíz)"
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "CO", vetado=nunca, declarados=["arepas"]) == \
        ap.ETIQUETA_A_BETA_RESPALDO
    assert ap.etiqueta_desayuno("Mangú/Tubérculos", "CO", vetado=nunca, declarados=["arepas", "patatas"]) == \
        ap.ETIQUETA_A_BETA_NEUTRA
    for decl in (["maíz amarillo"], ["arepas"], ["tortilla"], ["frijoles negros"], ["tubérculos"]):
        assert ap.etiqueta_desayuno("Mangú/Tubérculos", "DO", vetado=nunca, declarados=decl) == "Mangú/Tubérculos"


def test_r4_day_generator_pasa_las_declaraciones_a_la_puerta():
    src = (_BACKEND / "prompts" / "day_generator.py").read_text(encoding="utf-8")
    llamadas = _llamadas(src, "_ap.etiqueta_desayuno")
    assert len(llamadas) == 2, llamadas
    for c in llamadas:
        assert "declarados=_vetos" in c, c


def _brief_a(cc, n=6):
    sk = _esqueleto()
    sk["_other_days_brief"] = [{"technique": f"t{i}", "breakfast": "Mangú/Tubérculos", "country": cc} for i in range(n)]
    return sk


@pytest.mark.parametrize("cc", ("MX", "CO", "ES"))
def test_r4_la_puerta_se_memoriza_dentro_del_render(vocab_real, monkeypatch, cc):
    """Cada nombre de la puerta de la A pasa por el escáner UNA vez por render (antes: una vez por cada una de las 7
    etiquetas). Sólo los nombres de la puerta: «leche/queso/yogur» los preguntan otras líneas del render (también en
    DO), fuera de este lote."""
    import graph_orchestrator as go
    from prompts import asignacion_pais as ap
    puerta = set(ap._nombres_del_respaldo())
    for i in ap._desayuno_tipico(cc):
        if not ap._contiene(i, ap._BASES_DE_OTRAS_CATEGORIAS + ap._NO_ES_DESAYUNO + ap._NO_ES_BASE_PROPIA):
            puerta |= set(ap._nombres_para_la_puerta(i))
    original = go._allergen_pool_item_banned
    llamadas = []

    def contar(item, alergias):
        llamadas.append(str(item))
        return original(item, alergias)
    monkeypatch.setattr(go, "_allergen_pool_item_banned", contar)
    t = _bdac(_brief_a(cc), 1, day_name="Lunes", country=cc, allergies=["gluten", "huevo"], dislikes=["cebolla"])
    de_la_puerta = [n for n in llamadas if n in puerta]
    assert de_la_puerta, "la puerta pregunta al escáner"
    repetidas = sorted({n for n in de_la_puerta if de_la_puerta.count(n) > 1})
    assert not repetidas, (cc, len(de_la_puerta), repetidas)
    monkeypatch.setattr(go, "_allergen_pool_item_banned", original)
    assert _bdac(_brief_a(cc), 1, day_name="Lunes", country=cc, allergies=["gluten", "huevo"],
                 dislikes=["cebolla"]) == t, "memorizar no cambia el texto"


# [ronda 5] La declaración de la CLASE, más específica que el ítem: «legumbres secas», «alergia a legumbres»,
# «leguminosas y derivados» no caben en «frijoles», y la A de México imponía frijoles contra la línea de alergias del
# MISMO prompt. La raíz «legumbre»/«leguminosa» cierra la puerta del frijol desde la declaración cruda.
@pytest.mark.parametrize("campo", ("allergies", "dislikes"))
@pytest.mark.parametrize("decl", ("legumbres secas", "todas las legumbres", "alergia a legumbres",
                                  "leguminosas y derivados", "no me gustan las legumbres", "Legumbres",
                                  "leguminosa"))
def test_r5_la_clase_legumbre_no_impone_frijoles(vocab_real, campo, decl):
    kw = {campo: [decl]}
    linea = _linea_a(_bdac(_esqueleto(), 1, day_name="Lunes", country="MX", **kw))
    assert not _tiene("frijol", linea), (campo, decl, linea)
    assert "tortilla de maíz" in linea, ("la tortilla de maíz no es una legumbre", campo, decl, linea)


@pytest.mark.parametrize("campo", ("allergies", "dislikes"))
@pytest.mark.parametrize("decl", ("lentejas", "garbanzos", "soya", "maní", "cacahuate", "arvejas"))
def test_r5_otra_legumbre_concreta_no_veta_los_frijoles(vocab_real, campo, decl):
    """Nombrar UNA legumbre concreta no es declarar la clase: la A sigue con sus frijoles."""
    kw = {campo: [decl]}
    linea = _linea_a(_bdac(_esqueleto(), 1, day_name="Lunes", country="MX", **kw))
    assert "frijoles, tortilla de maíz" in linea, (campo, decl, linea)
