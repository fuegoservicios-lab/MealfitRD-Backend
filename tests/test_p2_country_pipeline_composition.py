"""[P2-COUNTRY-PIPELINE-TEST · 2026-08-21] Nada ejercitaba la COMPOSICIÓN con el knob encendido.

Los guards del sistema de países son casi todos parser-based y de una sola pieza: éste comprueba que
el planner nombre el país, aquél que el catálogo filtre, el otro que el contexto temporal no hable
del Caribe. Cada uno verde por su cuenta. Lo que faltaba es la pregunta que un usuario hace: **con
el knob encendido y un perfil español, ¿lo que se le manda al modelo es español?**

Es literalmente la lección de `P1-CULINARY-METADATA-BETA`, escrita en CLAUDE.md: *«capa 1 en
fail-open CON LOS TESTS EN VERDE (parser-based: ninguno mira el DATO)»*. Y esta ola la volvió a
pagar dos veces — el catálogo de Fase 2 era inerte para la generación, y los cinco catálogos beta
eran idénticos entre sí. Las dos habrían salido aquí.

QUÉ ES «EL PIPELINE COMPLETO» EN UN TEST SIN LLM, y por qué esta versión vale. Generar de verdad
cuesta dinero y no es determinista, así que lo que se ejercita es todo lo DETERMINISTA que decide
qué ve el modelo: el prompt del planner, el del generador de días, el catálogo verificado y el
contexto temporal, compuestos como en producción. Si esa composición sale dominicana para un
español, el plan saldrá dominicano — sin gastar un token en comprobarlo.

Lo que NO cubre, dicho: el chunk worker y la generación real. Eso sigue abierto y necesita un
entorno con base de datos y presupuesto de LLM; el gap no se cierra entero aquí.
"""
from __future__ import annotations

import pytest

@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


_BETA = ("ES", "MX", "CO", "PR", "US")

# Nombres de PLATO dominicano que sobreviven hoy en el prompt compuesto de un usuario beta, con el
# sitio del que salen. No es una lista de deseos: es lo MEDIDO, y por eso el test de abajo es una
# caracterización — si aparece uno NUEVO, falla; si desaparece uno de éstos, también, para que
# alguien actualice la nota en vez de dejarla obsoleta.
_RESIDUOS_MEDIDOS = {
    "mangu": "categoría de desayuno que asigna el Planificador («Mangú/…»)",
    "casabe": "regla de TÉCNICA P1-CASABE-NO-BOIL (no hervir una torta ya cocida)",
    "locrio": "lista de técnicas de plato fuerte prohibidas en merienda",
    "queso de hoja": "ejemplo de queso alto en sodio (regla clínica)",
}

# Marcas que NO deben aparecer nunca en un prompt beta: son instrucciones de cocinar dominicano,
# no ejemplos dentro de una regla técnica o clínica.
_PROHIBIDAS_EN_BETA = ("sancocho", "yaroa", "mofongo", "caribe", "criollo", "criolla")


def _form(cc):
    return {"country": cc}


def _prompt_compuesto(form_data: dict) -> str:
    """Lo que de verdad se le manda al modelo, compuesto COMO EN PRODUCCIÓN.

    El país se DERIVA con `country_for_form_data`, que es donde vive el knob — los builders lo
    reciben ya canónico y son knob-agnósticos por diseño. Pasarles 'ES' a pelo y esperar conducta
    dominicana con el knob apagado es medir la capa equivocada; me pasó una vez en esta misma ola
    y volvió a pasarme escribiendo este fichero."""
    from constants import country_for_form_data, strip_accents
    from graph_orchestrator import _get_verified_catalog_instruction
    from prompts.day_generator import build_day_generator_system_prompt
    from prompts.planner import build_planner_system_prompt
    from prompts.plan_generator import build_time_context
    cc = country_for_form_data(form_data or {})
    piezas = [
        build_planner_system_prompt(country=cc) or "",
        build_day_generator_system_prompt(country=cc) or "",
        build_time_context(country=cc) or "",
        _get_verified_catalog_instruction(form_data or {}) or "",
    ]
    return strip_accents(chr(10).join(piezas).lower())


# ── La composición ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", _BETA)
def test_ninguna_instruccion_de_cocinar_dominicano_llega_a_beta(knob_on, cc):
    """La pregunta que ningún guard de una pieza contesta: ¿el CONJUNTO le pide al modelo cocina
    dominicana? Estas seis marcas no son ejemplos dentro de una regla: son cocina."""
    from constants import strip_accents
    compuesto = _prompt_compuesto(_form(cc))
    coladas = sorted({m for m in _PROHIBIDAS_EN_BETA if strip_accents(m) in compuesto})
    assert not coladas, f"{cc}: el prompt compuesto le pide cocina dominicana: {coladas}"


@pytest.mark.parametrize("cc", _BETA)
def test_el_compuesto_declara_que_lo_dominicano_no_es_el_default(knob_on, cc):
    """La frase que hace inofensivos a los residuos de abajo: si el prompt dice explícitamente que
    los platos dominicanos no son requisito, un «casabe» dentro de una regla de técnica es un
    ejemplo, no una orden. Sin esa frase, dejarían de serlo."""
    assert "no son requisito" in _prompt_compuesto(_form(cc)), (
        f"{cc}: el compuesto perdió la declaración de que lo dominicano no es el default"
    )


@pytest.mark.parametrize("cc,nombre", [("ES", "espana"), ("MX", "mexico"), ("CO", "colombia")])
def test_el_prompt_compuesto_nombra_el_pais_del_usuario(knob_on, cc, nombre):
    """No basta con quitar lo dominicano: si el conjunto no nombra el país, el modelo no tiene a
    qué anclarse y produce «no-dominicano genérico» — la queja de P1-BETA-FRAGMENT-DEPTH."""
    assert nombre in _prompt_compuesto(_form(cc)), f"{cc}: el compuesto no nombra el país"


def test_dos_paises_beta_no_reciben_el_mismo_prompt(knob_on):
    """El defecto que esta ola encontró DOS veces: «no-dominicano» no es «español». Si ES y MX
    reciben lo mismo, el sistema no está haciendo su trabajo aunque cada guard suelto esté verde."""
    assert _prompt_compuesto(_form("ES")) != _prompt_compuesto(_form("MX"))


def test_los_residuos_conocidos_son_exactamente_estos(knob_on):
    """Caracterización, no umbral. Los cuatro nombres de plato que sobreviven salen de reglas de
    TÉCNICA o CLÍNICAS donde el plato es un ejemplo — quitarlos exige curar ejemplos por país, que
    es contenido (hermano de P1-BETA-FRAGMENT-DEPTH), no plomería.

    Si aparece uno NUEVO, falla: sería un residuo sin revisar. Si desaparece uno de éstos, también:
    para que alguien actualice la nota en vez de dejarla mintiendo."""
    from constants import strip_accents
    compuesto = _prompt_compuesto(_form("ES"))
    vistos = {m for m in _RESIDUOS_MEDIDOS if strip_accents(m) in compuesto}
    assert vistos == set(_RESIDUOS_MEDIDOS), (
        f"los residuos dominicanos del prompt beta cambiaron. Esperados {sorted(_RESIDUOS_MEDIDOS)}, "
        f"vistos {sorted(vistos)}. Si has quitado uno, bórralo de `_RESIDUOS_MEDIDOS`; si aparece "
        f"uno nuevo, mira de qué regla sale antes de añadirlo"
    )


# ── Byte-identidad dominicana ───────────────────────────────────────────────────────────────────

def test_el_prompt_dominicano_conserva_lo_suyo(knob_on):
    """El error opuesto —y peor— sería neutralizar de más y dejar al dominicano sin su cocina."""
    from constants import strip_accents
    compuesto = _prompt_compuesto(_form("DO"))
    presentes = {m for m in _PROHIBIDAS_EN_BETA if strip_accents(m) in compuesto}
    assert presentes, (
        "el prompt dominicano perdió TODAS las marcas criollas que el beta tiene prohibidas: la "
        "neutralización se pasó de largo y ahora RD tampoco recibe su cocina"
    )


def test_con_el_knob_apagado_todos_reciben_el_dominicano(monkeypatch):
    """Contrato de rollback del sistema entero, comprobado sobre la COMPOSICIÓN y no pieza a pieza:
    apagado, un español recibe exactamente lo que recibe un dominicano.

    Sólo funciona porque `_prompt_compuesto` deriva el país con `country_for_form_data`, que es
    donde vive el knob. Pasarle 'ES' directo a los builders daría distinto incluso apagado —son
    knob-agnósticos por diseño— y el test acusaría a código correcto."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "false")
    assert _prompt_compuesto(_form("ES")) == _prompt_compuesto(_form("DO"))


# ── Que el test no pueda quedarse vacuo ─────────────────────────────────────────────────────────

def test_la_composicion_no_esta_vacia(knob_on):
    """El guard del guard. Si una de las cuatro piezas devolviera '' —por un knob, un import roto o
    un catálogo sin DB—, los tests de arriba pasarían POR VACÍO: no encontrar «sancocho» en una
    cadena vacía no prueba nada. Es la trampa que esta ola ya vio tres veces."""
    for cc in ("DO",) + _BETA:
        compuesto = _prompt_compuesto(_form(cc))
        assert len(compuesto) > 20000, (
            f"{cc}: el prompt compuesto mide {len(compuesto)} chars — demasiado poco para que los "
            f"asertos de arriba signifiquen algo"
        )


def test_las_cuatro_piezas_aportan_algo(knob_on):
    """Corolario: que ninguna pieza concreta esté vacía. Con el catálogo caído, por ejemplo, el
    conjunto seguiría siendo grande y el test de arriba pasaría — pero la pieza que más nombres
    dominicanos puede colar sería justo la ausente."""
    from graph_orchestrator import _get_verified_catalog_instruction
    from prompts.day_generator import build_day_generator_system_prompt
    from prompts.plan_generator import build_time_context
    from prompts.planner import build_planner_system_prompt
    for nombre, pieza in (
        ("planner", build_planner_system_prompt(country="ES")),
        ("day_generator", build_day_generator_system_prompt(country="ES")),
        ("time_context", build_time_context(country="ES")),
    ):
        assert pieza and len(pieza) > 100, f"la pieza {nombre!r} vino vacía o casi"
    # El catálogo depende de la DB: se comprueba aparte y se DICE cuando falta, en vez de dejar que
    # su ausencia haga pasar los asertos de contenido en silencio.
    cat = _get_verified_catalog_instruction({"country": "ES"}) or ""
    if not cat:
        pytest.skip("catálogo verificado vacío (sin DB): los asertos de contenido lo excluyen")
    assert len(cat) > 500


def test_las_marcas_prohibidas_existen_de_verdad_en_el_dominicano(knob_on):
    """Y el guard del guard del guard: si `_PROHIBIDAS_EN_BETA` se llenara de palabras que el
    prompt dominicano tampoco tiene, este fichero sería teatro completo — no encontrarlas en beta
    no probaría nada. Se exige que al menos la mitad aparezcan en el render DO real."""
    from constants import strip_accents
    do = _prompt_compuesto(_form("DO"))
    vivas = [m for m in _PROHIBIDAS_EN_BETA if strip_accents(m) in do]
    assert len(vivas) >= len(_PROHIBIDAS_EN_BETA) // 2, (
        f"sólo {len(vivas)} de {len(_PROHIBIDAS_EN_BETA)} marcas aparecen en el prompt dominicano: "
        f"la lista se quedó obsoleta y los asertos de beta ya no prueban gran cosa. Vivas: {vivas}"
    )
