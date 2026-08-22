"""[P2-DIARY-CATALOG-COUNTRY · 2026-08-21] El catálogo del diario es ciego al país A PROPÓSITO.

La auditoría lo listó como gap: «los 60 platos criollos y los SKUs del súper dominicano se sirven a
todo el mundo sin filtro (`GET /api/catalog/dishes`)». Es cierto que no filtra. No es cierto que
deba filtrar.

LA ASIMETRÍA QUE LO DECIDE — y es la misma palabra en los dos lados, con el tiempo verbal cambiado:

  · El **generador** es PROSPECTIVO: propone lo que vas a comprar y cocinar. Ofrecerle huitlacoche
    a un español es un defecto, porque no puede comprarlo. Por eso `P1-COUNTRY-CATALOG-BY-COUNTRY`
    y `P2-SUGGEST-FOODS-COUNTRY` sí acotan por país.
  · El **diario** es RETROSPECTIVO: registra lo que YA comiste. Sólo puedes registrar un plato que
    te comiste, así que un catálogo más ancho no puede recomendarte nada equivocado — sólo puede
    faltarle el tuyo.

Filtrar aquí no arregla nada y rompe al usuario que el beta más busca: **un dominicano viviendo en
España sigue comiendo mangú**. Con `country='ES'` un filtro le quitaría del buscador el plato que
acaba de comerse, y su diario pasaría a ser incompleto o directamente falso — la comida existió y
el sistema le diría que no.

El mismo argumento cubre la otra mitad («los SKUs dominicanos»): `/catalog` sirve nombres para que
el buscador autocomplete. Un nombre de más no le da de comer a nadie; un nombre de menos le impide
anotar lo que comió.

LO QUE SÍ FALTA, y no es esto: el catálogo no tiene platos ESPAÑOLES, MEXICANOS ni COLOMBIANOS que
registrar. Un español no puede anotar «paella» de una pieza. Eso es curación de contenido —hermano
de `P1-BETA-FRAGMENT-DEPTH`— y se cierra AÑADIENDO, nunca quitando.

Este fichero no cambia código. Fija la decisión para que un futuro «cerrar P2-25» no le quite el
mangú al usuario que sí lo come.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def fs():
    import food_search as _fs
    return _fs


def test_los_platos_se_sirven_completos_sin_filtro_de_pais(fs):
    """Los 60 platos, para todo el mundo. Si alguien introduce un filtro por país aquí, este test
    lo detiene y le manda a leer la asimetría de la cabecera."""
    items = fs.dishes_for_client() or []
    assert len(items) >= 50, f"el catálogo de platos encogió a {len(items)}"


def test_la_funcion_no_acepta_pais(fs):
    """La señal más simple de que nadie ha metido el filtro: la firma no lo admite. Un `country=`
    aquí sería el primer paso del defecto, no de su arreglo."""
    params = set(inspect.signature(fs.dishes_for_client).parameters)
    assert not (params & {"country", "cc", "pais"}), (
        f"`dishes_for_client` aceptó un parámetro de país: {sorted(params)}. El diario es "
        f"RETROSPECTIVO — filtrar le quitaría a un dominicano en España el plato que se comió"
    )


def test_el_endpoint_no_deriva_pais(fs):
    """Y el endpoint tampoco lo deriva por su cuenta."""
    src = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8", errors="replace")
    i = src.index("async def api_get_catalog_dishes")
    j = src.index("\n@router", i)
    cuerpo = src[i:j]
    for prohibido in ("country_for_form_data", "canonicalize_country", "country_for_plan"):
        assert prohibido not in cuerpo, (
            f"`/catalog/dishes` empezó a derivar país ({prohibido}) — ver la asimetría "
            f"prospectivo/retrospectivo en la cabecera de este fichero"
        )


@pytest.mark.parametrize("plato", ["mangu", "sancocho", "moro", "mofongo"])
def test_un_dominicano_fuera_de_rd_sigue_pudiendo_registrar_lo_suyo(fs, plato):
    """El caso concreto que justifica la decisión: la diáspora. Es exactamente el usuario que el
    beta persigue, y un filtro por país lo dejaría sin poder anotar su propia comida."""
    from constants import strip_accents
    etiquetas = {strip_accents(str(i.get("label") or "").lower())
                 for i in (fs.dishes_for_client() or [])}
    assert any(plato in e for e in etiquetas), (
        f"{plato!r} desapareció del catálogo del diario"
    )


def test_lo_que_falta_es_anadir_no_quitar(fs):
    """Ancla del pendiente REAL, para que no se confunda con el gap descartado: hoy no hay platos
    de los países beta que registrar. Cuando los haya, este test se actualiza; mientras tanto deja
    escrito que la deuda es de contenido y se paga añadiendo."""
    from constants import strip_accents
    etiquetas = " ".join(strip_accents(str(i.get("label") or "").lower())
                         for i in (fs.dishes_for_client() or []))
    faltan = [p for p in ("paella", "pozole", "bandeja paisa", "tortilla espanola")
              if p not in etiquetas]
    assert faltan, (
        "ya hay platos de países beta en el catálogo: actualiza este test y la nota de la cabecera"
    )
