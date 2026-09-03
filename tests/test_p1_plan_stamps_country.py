"""[P1-PLAN-STAMPS-COUNTRY + P1-PRICING-MODE-REDERIVE · 2026-08-21] El plan no llevaba su propio
país, y el régimen de precios se estampaba una vez y no lo re-derivaba nadie.

Van juntos porque son las dos mitades de la misma pregunta —«¿de qué país es ESTE plan?»— y hoy
se responden con fuentes distintas que ya divergen en producción:

    plan_data guarda    `_pricing_mode`  →  congelado desde `assemble_plan_node`, nunca re-derivado
    plan_data NO guarda `country`        →  cada superficie post-generación lo saca del PERFIL ACTUAL

Medido: los 2 planes beta vivos (6a4321f5 ES, 2245eb45 US) no tienen `country` en `plan_data`, y
el perfil de su dueño dice **'DO'**. Es decir, el sistema ya está en el estado inconsistente.

QUÉ ROMPE, EN LAS DOS DIRECCIONES:

  DO → beta (lo que el producto ofrece activamente en Configuración): un dominicano con un plan
  de 30 días en curso decide que ahora compra en España. Su plan no tiene `_pricing_mode`, así
  que la lista, el PDF y el banner de presupuesto siguen mostrando importes en RD$ de colmado
  dominicano PARA SIEMPRE — incluso tras recalcular, porque `/recalculate-shopping-list` LEE el
  valor viejo del jsonb aunque 300 líneas antes, en el MISMO endpoint, acabe de resolver el país
  desde el perfil actualizado. Mientras tanto el swap ya le sirve comida española: un plan que le
  propone bacalao y se lo cotiza en pesos dominicanos.

  beta → DO: el plan conserva `beta_no_prices` y jamás recupera precios.

El toast de Configuración promete lo contrario en las dos direcciones.

LA POLÍTICA, ESCRITA. Aquí había que elegir y no había respuesta obvia: ¿manda el plan (snapshot
de cuando se generó) o el perfil (lo que el usuario dice hoy)? Se elige **el plan manda para lo
que YA se generó, el perfil manda para lo que se genera de nuevo**, con una excepción: el
recálculo explícito sanea un sello que YA existe. [P1-COUNTRY-STAMP-NO-FALLBACK-WRITE] supersede
la escritura de un fallback: en un plan legacy la ausencia no prueba el país de origen, por lo
que el perfil puede gobernar la lectura actual pero no convertirse en sello autoritativo.

EL SELLO SE ESCRIBE SIEMPRE, TAMBIÉN PARA 'DO'. Si sólo se estampara en beta, la AUSENCIA de la
clave significaría dos cosas distintas —«plan dominicano» y «plan anterior a este P-fix»— y esa
ambigüedad es exactamente la que hace irreparables los 2 planes vivos. Con el sello incondicional,
ausente = pre-sistema, y eso es una respuesta útil.

Cubre:
  A. El sello: se escribe siempre, con el valor de la única puerta.
  B. Los lectores post-generación prefieren el sello del plan sobre el perfil.
  C. Un plan sin sello (pre-P-fix) degrada al perfil, como hasta hoy.
  D. El recálculo sanea el régimen de planes sellados sin sellar fallbacks legacy.
  E. Byte-identidad DO y con el knob apagado.
  F. Parser-based.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GO_PATH = _BACKEND_ROOT / "graph_orchestrator.py"
_PLANS_PATH = _BACKEND_ROOT / "routers" / "plans.py"


@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


# ── A. El sello ─────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", ["DO", "ES", "MX", "US"])
def test_el_plan_se_sella_con_su_pais_siempre(knob_on, cc):
    """RED pre-fix: `plan_data` sólo llevaba `_pricing_mode`. El sello va incondicional —también
    para 'DO'— para que la AUSENCIA de la clave signifique «plan anterior a este P-fix» y no se
    confunda con «plan dominicano»."""
    from constants import stamp_plan_country
    plan = {}
    stamp_plan_country(plan, {"country": cc})
    assert plan.get("_country") == cc


def test_el_sello_pasa_por_la_unica_puerta(knob_on):
    """Un país basura no se persiste crudo: canoniza por `country_for_form_data`, que además
    aplica el knob maestro. Escribir aquí un 2º canonicalizador sería la tabla que
    P1-DIET-CANON-SSOT prohibió."""
    from constants import stamp_plan_country
    plan = {}
    stamp_plan_country(plan, {"country": "Reino de Absurdistán"})
    assert plan.get("_country") == "DO"


def test_con_el_knob_apagado_el_sello_es_dominicano(monkeypatch):
    """Rollback: apagar `MEALFIT_COUNTRY_SYSTEM` devuelve el motor a conducta dominicana, y el
    sello lo refleja en vez de mentir sobre lo que el motor hizo."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    from constants import stamp_plan_country
    plan = {}
    stamp_plan_country(plan, {"country": "ES"})
    assert plan.get("_country") == "DO"


# ── B/C. Los lectores prefieren el plan ─────────────────────────────────────────────────────────

def test_el_pais_del_plan_gana_al_del_perfil(knob_on):
    """El caso vivo: plan español, perfil que hoy dice 'DO'. Recalcular la lista de ese plan
    aplicaba reglas dominicanas a platos españoles."""
    from constants import country_for_plan
    assert country_for_plan({"_country": "ES"}, {"country": "DO"}) == "ES"


def test_un_plan_sin_sello_degrada_al_perfil(knob_on):
    """Los planes anteriores a este P-fix —incluidos los 2 beta vivos— no tienen sello. Deben
    seguir comportándose EXACTAMENTE como hasta hoy: leer el perfil. Sin este fallback el fix
    rompería todo el histórico."""
    from constants import country_for_plan
    assert country_for_plan({}, {"country": "ES"}) == "ES"
    assert country_for_plan(None, {"country": "MX"}) == "MX"


def test_sin_plan_ni_perfil_cae_a_dominicano(knob_on):
    from constants import country_for_plan
    assert country_for_plan({}, {}) == "DO"


def test_un_sello_corrupto_no_gana(knob_on):
    """Fail-safe: si el sello del plan no canoniza, se ignora y manda el perfil — un jsonb que
    alguien tocó a mano no puede secuestrar el motor."""
    from constants import country_for_plan
    assert country_for_plan({"_country": "XX"}, {"country": "ES"}) == "ES"


# ── D. El recálculo re-deriva ───────────────────────────────────────────────────────────────────

def test_el_recalculo_delega_el_regimen_sin_escribir_fallback_directo():
    """`/recalculate-shopping-list` leía `plan_data.get('_pricing_mode')` —el valor viejo— aunque
    en el mismo endpoint ya hubiera resuelto el país desde el perfil actualizado. Las dos
    verdades convivían a 300 líneas de distancia."""
    src = _PLANS_PATH.read_text(encoding="utf-8", errors="replace")
    i = src.find("_recalc_country")
    assert i > 0, "el recálculo dejó de resolver el país"
    helper_i = src.find("apply_recalc_plan_regime", i)
    assert helper_i > i, (
        "el recálculo dejó de distinguir el sello real del fallback del perfil"
    )
    assert 'pricing_mode=plan_data_fresh.get("_pricing_mode")' not in src, (
        "el recálculo volvió a costear con el régimen congelado del jsonb"
    )


# ── E. Byte-identidad ───────────────────────────────────────────────────────────────────────────

def test_el_sello_no_cambia_nada_mas_del_plan(knob_on):
    """El sello es aditivo: no toca ninguna otra clave. Un `plan_data` con contenido debe salir
    idéntico salvo por `_country` — si esto fallara, el estampado estaría pisando datos del plan
    (la clase de bug que la invariante I7 existe para evitar)."""
    from constants import stamp_plan_country
    plan = {"days": [{"day": 1}], "macros": {"protein": 120}, "_pricing_mode": "beta_no_prices"}
    antes = {k: v for k, v in plan.items()}
    stamp_plan_country(plan, {"country": "ES"})
    assert plan.pop("_country") == "ES"
    assert plan == antes


# ── F. Parser-based ─────────────────────────────────────────────────────────────────────────────

def test_el_estampado_vive_junto_al_de_pricing_mode():
    """Los dos sellos describen el mismo plan y deben escribirse en el mismo sitio: separarlos
    invita a que uno se estampe y el otro no, que es el estado en el que están los 2 planes
    vivos."""
    src = _GO_PATH.read_text(encoding="utf-8", errors="replace")
    assert "P1-PLAN-STAMPS-COUNTRY" in src
    i = src.find('result["_pricing_mode"] = _pricing_mode')
    assert i > 0, "el estampado de _pricing_mode desapareció o cambió de forma"
    assert "stamp_plan_country" in src[max(0, i - 900):i + 900], (
        "el sello de país no se escribe junto al del régimen de precios"
    )


def test_los_helpers_estan_en_el_ssot_de_pais():
    """`constants.py` es donde viven `canonicalize_country` y `country_for_form_data`. Los dos
    helpers nuevos van ahí y no en el orquestador: el router y los crons también los necesitan, y
    un helper de país importado desde `graph_orchestrator` sería la dirección de dependencia
    equivocada."""
    src = (_BACKEND_ROOT / "constants.py").read_text(encoding="utf-8", errors="replace")
    assert "def stamp_plan_country" in src
    assert "def country_for_plan" in src
    i = src.find("def country_for_plan")
    _fin = src.find("\ndef ", i + 1)
    cuerpo = src[i:_fin if _fin > 0 else len(src)]
    # El lector debe canonizar el sello, y hacerlo por la ÚNICA puerta —que además aplica el knob
    # maestro— y no llamando a `canonicalize_country` a pelo. La primera versión de este assert
    # exigía el nombre del canonicalizador y habría empujado hacia la variante PEOR: un lector que
    # canoniza pero ignora el rollback del knob.
    assert "country_for_form_data" in cuerpo, (
        "el lector no resuelve el sello por la única puerta de lectura de país del motor"
    )
    assert "COUNTRY_PROFILES" in cuerpo, "el lector no valida el sello contra los países reales"
