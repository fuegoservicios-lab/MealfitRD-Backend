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


def test_el_cajon_abre_en_el_dia_de_hoy_no_en_el_mas_viejo():
    """[P1-DIARY-STRIP-SCROLL] Lo que protegía este test: al abrir el diario se ve HOY (o el día tocado), nunca los días
    más viejos.

    [CALENDARIO-SOLO-MES · 2026-11-15, a3befe13] La tira de 14 días salió por diseño: el cajón abre directo en el MES
    del día con el que se abre. El riesgo original («hoy queda fuera de pantalla») se traslada a: abrir en un mes que no
    es el de hoy, o en un día seleccionado que no es hoy.
    """
    src = (FRONT / "DiaryHistory.jsx").read_text(encoding="utf-8")
    assert "stripRef" not in src, "la tira volvió: entonces vuelve también su desplazamiento al día activo (scrollTo)"
    assert "setSelected(diaInicial || getBiologicalDateStr(new Date()))" in src, (
        "al abrir, el día seleccionado debe ser el tocado o HOY"
    )
    assert re.search(r"const base = diaInicial \? desdeISO\(diaInicial\) : getBiologicalDate\(new Date\(\)\);", src), (
        "el calendario debe abrir en el mes del día con el que se abre (hoy por defecto)"
    )
    assert re.search(r"\}, \[open, diaInicial\]\);", src), "se recalcula al ABRIR, no solo al montar"


def test_el_borde_de_la_tira_avisa_de_que_hay_mas_dias():
    """Sin barra de scroll visible, un corte seco se lee como el final."""
    css = (FRONT / "DiaryHistory.module.css").read_text(encoding="utf-8")
    assert "mask-image" in css, (
        "la tira corta en seco: nada indica que se pueda deslizar"
    )
