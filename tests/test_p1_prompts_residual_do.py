"""[P1-PROMPTS-RESIDUAL-DO · 2026-08-21] El planner y el prompt de variedad no nombraban el país
del usuario y conservaban léxico dominicano.

Fase 1 neutralizó la Categoría A del planner y el bullet «FIDELIDAD CULTURAL» del prompt de
variedad. Lo que quedó, medido con los renders reales:

    planner    ES == MX -> True   ·  ES menciona «España» -> False
               tokens DO en el render ES: arepitas, casabe, dominicano, queso de hoja, salami
    variedad   ES == MX -> True   ·  ES menciona «España» -> False
               tokens DO en el render ES: arepitas, casabe, dominicana, dominicano,
                                          queso de hoja, sancocho

Que ES y MX salgan **byte-idénticos** es el dato que ordena el resto: no hay adaptación por país,
sólo una supresión parcial de lo dominicano. Y los tokens que sobreviven no son adorno — el
planner propone «Día 2 merienda = Casabe+queso» como EJEMPLO CORRECTO, y el de variedad reparte
«arepitas» como transformación de base.

DOS PIEZAS, PORQUE UNA SOLA NO BASTA. Este repo ya midió en P1-DIET-BLIND-DIRECTIVES que «una
directiva de cabecera SOLA pierde contra órdenes específicas»: el retry informado de un plan
VEGANO llevaba inyectada la orden «fuente animal de alta densidad (pollo, pescado…)» y el modelo
obedecía. Así que:

  1. una CABECERA que nombra el país (necesaria: hoy el modelo no sabe para quién cocina), y
  2. la neutralización de los EJEMPLOS concretos que la contradicen.

LO QUE ESTE P-FIX **NO** HACE. No convierte el planner en un planner español. Sustituir «casabe»
por «pan tostado» quita la contradicción pero no añade cultura: un plan MX no sale mexicano por
esto. Esa es una tarea de CONTENIDO —curar guía por país con la densidad que hoy tiene la
dominicana— y queda registrada aparte (P1-BETA-FRAGMENT-DEPTH); mezclarla aquí sería esconder una
decisión de producto dentro de un arreglo de plomería.

Cubre:
  A. Byte-identidad dominicana y con el knob apagado.
  B. La cabecera nombra el país, y sale del mismo SSOT que el resto del sistema.
  C. Los tokens dominicanos medidos desaparecen del render beta.
  D. El prompt beta sigue siendo un prompt (no se vació de reglas).
  E. Los países beta dejan de ser byte-idénticos entre sí.
  F. Parser-based.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Los tokens medidos en los renders ES antes del fix.
_TOKENS_DO = ("casabe", "arepitas", "queso de hoja", "sancocho", "salami dominicano")


@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


def _planner(cc):
    from prompts.planner import build_planner_system_prompt
    return build_planner_system_prompt(cc)


def _variedad(cc):
    from prompts.preferences import build_deterministic_variety_prompt
    return build_deterministic_variety_prompt(3, cc)


_RENDERS = {"planner": _planner, "variedad": _variedad}


# ── A. Byte-identidad dominicana ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("nombre", sorted(_RENDERS))
def test_el_dominicano_no_cambia(knob_on, nombre):
    """Control primero: en RD estos ejemplos son los correctos y no se mueven."""
    do = _RENDERS[nombre]("DO")
    assert "casabe" in do.lower()
    assert "España" not in do


def test_el_knob_se_aplica_en_el_CALLER_no_en_estos_builders():
    """El rollback existe, pero una capa más arriba — y este test lo fija ahí en vez de fingir
    que vive aquí.

    Estos dos builders reciben un país YA canónico: `graph_orchestrator` los llama con
    `ctx['country']`, que es `_shared_ctx_country = country_for_form_data(form_data)`, y ESA es la
    función que aplica `MEALFIT_COUNTRY_SYSTEM`. Mi primera versión de este test les pasaba 'ES'
    a pelo con el knob apagado y esperaba conducta dominicana — estaba midiendo la capa
    equivocada: con el knob apagado el caller nunca les pasa 'ES', les pasa 'DO'.

    Lo que sí hay que anclar es que el caller siga pasando el valor GATEADO."""
    go = (_BACKEND_ROOT / "graph_orchestrator.py").read_text(encoding="utf-8", errors="replace")
    assert "build_planner_system_prompt(ctx['country'])" in go, (
        "el planner dejó de recibir el país gateado del contexto compartido"
    )
    assert "_shared_ctx_country = country_for_form_data(" in go, (
        "el contexto compartido dejó de derivar el país por la puerta que aplica el knob"
    )


@pytest.mark.parametrize("nombre", sorted(_RENDERS))
def test_pasar_DO_devuelve_el_prompt_dominicano(nombre):
    """La otra mitad del rollback: cuando el caller manda 'DO' —que es lo que manda con el knob
    apagado— el render es el dominicano de siempre."""
    do = _RENDERS[nombre]("DO")
    assert "casabe" in do.lower() and "España" not in do


# ── B. La cabecera nombra el país ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("nombre", sorted(_RENDERS))
@pytest.mark.parametrize("cc,pais", [("ES", "España"), ("MX", "México"), ("CO", "Colombia")])
def test_el_prompt_nombra_el_pais_del_usuario(knob_on, nombre, cc, pais):
    """RED pre-fix: ninguno de los dos mencionaba el país. El modelo no sabía para quién cocina —
    sólo que no debía cocinar dominicano, que es una instrucción en negativo."""
    assert pais in _RENDERS[nombre](cc), f"{nombre}: el render de {cc} no nombra a {pais}"


def test_el_nombre_del_pais_sale_del_ssot(knob_on):
    """De `COUNTRY_PROFILES[cc]['name_es']`, el mismo que usan el juez culinario y la biblioteca
    de platos — no una segunda tabla de gentilicios."""
    from constants import COUNTRY_PROFILES
    for cc, perfil in COUNTRY_PROFILES.items():
        if cc == "DO":
            continue
        assert perfil["name_es"] in _planner(cc)


# ── C. Los tokens medidos desaparecen ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("nombre", sorted(_RENDERS))
@pytest.mark.parametrize("token", _TOKENS_DO)
def test_los_tokens_dominicanos_no_viajan_a_beta(knob_on, nombre, token):
    """RED pre-fix: los cinco vivían en los dos renders. No son adorno — el planner propone
    «Día 2 merienda = Casabe+queso» como EJEMPLO CORRECTO."""
    render = _RENDERS[nombre]("ES").lower()
    assert token not in render, f"{nombre}: sigue nombrando «{token}» a un usuario español"


@pytest.mark.parametrize("nombre", sorted(_RENDERS))
def test_no_queda_ningun_gentilicio_dominicano(knob_on, nombre):
    render = _RENDERS[nombre]("ES")
    assert not re.search(r"dominican", render, re.I), (
        f"{nombre}: el render beta sigue diciendo «dominicano»"
    )


# ── D. El prompt sigue siendo un prompt ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("nombre", sorted(_RENDERS))
def test_el_prompt_beta_no_se_vacio(knob_on, nombre):
    """Vaciar los bloques habría sido el otro error: las REGLAS (variedad, techos de embutidos,
    piso de proteína, no repetir base) son nutricionales y valen en los 6 países. Lo que cambia
    son los EJEMPLOS."""
    do, es = _RENDERS[nombre]("DO"), _RENDERS[nombre]("ES")
    assert len(es) > len(do) * 0.85, (
        f"{nombre}: el prompt beta perdió más del 15% de su contenido — se están tirando reglas, "
        f"no sólo ejemplos"
    )


# ── E. Los países beta dejan de ser idénticos ───────────────────────────────────────────────────

@pytest.mark.parametrize("nombre", sorted(_RENDERS))
def test_los_paises_beta_ya_no_son_byte_identicos(knob_on, nombre):
    """RED pre-fix: `ES == MX` era True en los dos. Era la prueba de que no había adaptación por
    país, sólo supresión de lo dominicano. La cabecera es el mínimo que los distingue — la
    adaptación de CONTENIDO es P1-BETA-FRAGMENT-DEPTH, no este P-fix."""
    assert _RENDERS[nombre]("ES") != _RENDERS[nombre]("MX")


# ── F. Parser-based ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("modulo", ["planner.py", "preferences.py"])
def test_el_fuente_declara_el_marker(modulo):
    src = (_BACKEND_ROOT / "prompts" / modulo).read_text(encoding="utf-8", errors="replace")
    assert "P1-PROMPTS-RESIDUAL-DO" in src
