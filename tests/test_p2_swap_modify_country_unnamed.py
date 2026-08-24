"""[P2-SWAP-MODIFY-COUNTRY-UNNAMED · 2026-08-23] Los prompts de «cambiar plato» y de chat-modify
estaban neutralizados pero nunca nombraban el país.

Medido antes del arreglo:

    swap DO len=3845  tokens DO=['arepitas','casabe','mofongo','mangú']  gentilicio=['dominicanos']
    swap ES len=3904  tokens DO=[]  gentilicio=[]  'España' in? False
    swap MX len=3904  ·  swap US len=3904  ·  ES==MX: True · ES==US: True
    modify: lo mismo (DO 3559, beta 3674, ES==MX True)

O sea: el prompt ya no imponía cocina criolla, pero tampoco decía en qué país se cocina. Al modelo
se le pedía «una preparación apetecible del contexto local e internacional del usuario» **sin
decirle cuál es ese contexto** — que es lo que `P1-PROMPTS-RESIDUAL-DO` ya había resuelto para el
planner y las preferencias con `beta_prompt_country_header`, y que estas dos superficies no
heredaron.

Matiz que este test NO exagera: «ES == MX == US byte-idénticos» era cierto de la PLANTILLA, no del
prompt ENSAMBLADO — en producción el bloque de inspiración de la biblioteca sí lleva señal de país
(`P1-DISH-LIBRARY-COUNTRY`). Lo que faltaba era que la plantilla lo dijera por su cuenta.

La cabecera va DENTRO de los dos builders y antes del cacheo: `_MEAL_OPS_COUNTRY_CACHE` ya está
keyed por (superficie, país), así que este es el caso en que la clave ya estaba bien.
"""
from __future__ import annotations

import pytest

from constants import COUNTRY_PROFILES, beta_prompt_country_header
from prompts.meal_operations import (
    MODIFY_MEAL_PROMPT_TEMPLATE,
    SWAP_MEAL_PROMPT_TEMPLATE,
    build_modify_meal_prompt_template,
    build_swap_meal_prompt_template,
)

_BETA = ["ES", "MX", "US", "PR", "CO"]
_BUILDERS = {
    "swap": (build_swap_meal_prompt_template, SWAP_MEAL_PROMPT_TEMPLATE),
    "modify": (build_modify_meal_prompt_template, MODIFY_MEAL_PROMPT_TEMPLATE),
}


@pytest.mark.parametrize("superficie", list(_BUILDERS))
def test_do_devuelve_la_constante_intacta(superficie):
    """Byte-identidad DO por identidad de objeto — el ancla `is` que su docstring promete."""
    build, const = _BUILDERS[superficie]
    assert build("DO") is const
    assert build(None) is const
    assert build("marte") is const, "país desconocido ⇒ fail-safe a DO (canonicalize_country)"


@pytest.mark.parametrize("superficie", list(_BUILDERS))
@pytest.mark.parametrize("cc", _BETA)
def test_el_render_beta_nombra_su_pais(superficie, cc):
    build, _ = _BUILDERS[superficie]
    nombre = COUNTRY_PROFILES[cc]["name_es"]
    render = build(cc)
    assert nombre in render, (
        f"el prompt de {superficie} para {cc} no nombra «{nombre}»: el modelo sigue cocinando "
        "«para el contexto local del usuario» sin saber cuál es."
    )


@pytest.mark.parametrize("superficie", list(_BUILDERS))
def test_dos_paises_beta_ya_no_son_byte_identicos(superficie):
    build, _ = _BUILDERS[superficie]
    assert build("ES") != build("MX"), f"{superficie}: ES y MX siguen siendo el MISMO string"
    assert build("ES") != build("US"), f"{superficie}: ES y US siguen siendo el MISMO string"


@pytest.mark.parametrize("superficie", list(_BUILDERS))
@pytest.mark.parametrize("cc", _BETA)
def test_la_cabecera_sale_del_ssot_no_de_una_tabla_nueva(superficie, cc):
    """Una segunda tabla de gentilicios es la lección que P1-DIET-CANON-SSOT ya cobró una vez."""
    build, _ = _BUILDERS[superficie]
    assert build(cc).startswith(beta_prompt_country_header(cc)), (
        f"{superficie}/{cc}: la cabecera no es la de `constants.beta_prompt_country_header`."
    )


@pytest.mark.parametrize("superficie", list(_BUILDERS))
@pytest.mark.parametrize("cc", _BETA)
def test_el_render_sigue_siendo_formateable(superficie, cc):
    """La cabecera se antepone a una plantilla que luego pasa por `.format()`: cero llaves nuevas.

    Un `{` suelto en la cabecera convertiría el arreglo en un KeyError en producción, en el camino
    caliente del swap. Se comprueba comparando los campos de formato con los de la constante DO.
    """
    import string

    build, const = _BUILDERS[superficie]
    campos = {f for _, f, _, _ in string.Formatter().parse(build(cc)) if f}
    campos_do = {f for _, f, _, _ in string.Formatter().parse(const) if f}
    assert campos == campos_do, (
        f"{superficie}/{cc}: el render beta cambió los campos de `.format()` ({campos ^ campos_do})"
    )
