"""[P1-DIARY-STRIP-SCROLL · 2026-07-31] La tira abria por el dia MAS VIEJO.

El owner abrio el diario recien construido buscando su cena de ayer y vio
"18 … 27" con todos los rieles vacios. Era 31 de julio.

Los 14 dias miden ~597px y en el cajon caben ~423, asi que los 4 mas recientes
—hoy, ayer y los dos anteriores: los UNICOS que alguien mira— quedaban fuera de
pantalla a la derecha. Y la barra de scroll esta oculta por diseno, con lo cual
el corte se leia como "aqui se acaban los dias", no como "desliza".

*Un overflow silencioso no es una invitacion a deslizar: es una pared.*
"""

import re
from pathlib import Path

FRONT = Path(__file__).resolve().parent.parent.parent / "frontend" / "src" / "components" / "dashboard"


def test_la_tira_se_desplaza_al_dia_seleccionado():
    """[P1-DIARY-STRIP-SCROLL] Los 14 días no caben en el cajón.

    Reportado: el owner abrió el diario buscando su cena de ayer y vio
    "18 … 27" con todos los rieles vacíos. Era 31. Los días MÁS RECIENTES —los
    únicos que alguien mira— quedaban fuera de pantalla a la derecha, y la
    barra de scroll está oculta, así que el corte se leía como "no hay más".

    Se ancla al día ACTIVO y no a `scrollWidth`: así la tira también sigue al
    usuario cuando cambia de día con las flechas.
    """
    src = (FRONT / "DiaryHistory.jsx").read_text(encoding="utf-8")
    assert "activoRef" in src and "stripRef" in src, (
        "faltan las refs de la tira y del día activo"
    )
    assert "scrollTo(" in src, (
        "P1-DIARY-STRIP-SCROLL: la tira nunca se desplaza, así que abre "
        "mostrando los días MÁS VIEJOS — hoy y ayer quedan fuera de pantalla"
    )
    assert re.search(r"\[open,\s*selected,\s*dias\]", src), (
        "el desplazamiento no depende de `selected`: no seguiría al usuario al "
        "cambiar de día con las flechas"
    )


def test_el_borde_de_la_tira_avisa_de_que_hay_mas_dias():
    """Sin barra de scroll visible, un corte seco se lee como el final."""
    css = (FRONT / "DiaryHistory.module.css").read_text(encoding="utf-8")
    assert "mask-image" in css, (
        "la tira corta en seco: nada indica que se pueda deslizar"
    )
