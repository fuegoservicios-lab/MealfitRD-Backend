"""[P2-DISPLAY-CHAT-PRUNE · 2026-08-21] El `_display` de nivel PLAN viajaba al system
prompt del coach en cada turno.

QUÉ PASÓ. `P1-PLAN-DISPLAY-I18N` escribe la traducción en DOS niveles y en la misma
mutación atómica: por comida (`plan_data.days[*].meals[*]._display`) y por plan
(`plan_data._display`, con el nombre del plan y los insights traducidos).

`_prune_plan_for_chat` (agent.py) también poda en dos sitios: filtra las claves de
`_CHAT_PLAN_PRUNE_KEYS` en el nivel superior y, aparte, recorre `days[*].meals[*]`
quitando su `_display`. Lo segundo funciona. Lo primero no lo cubría: `_display` no
estaba en la tupla, así que la copia de nivel plan sobrevivía y se serializaba al
prompt en CADA turno del chat.

POR QUÉ IMPORTA. Es exactamente el modo de fallo que la tupla ya documenta en dos
entradas suyas: `_culinary_contract_*` y los reportes de QA entraron ahí porque «sin
podarlas se serializaban al system prompt EN CADA turno». El coste crece con cada
idioma visitado —el mapa retiene una entrada por locale— y no aporta nada al
razonamiento del agente: el coach ya recibe el plan en español canónico, que es el
idioma en el que resuelve los nombres de alimento, y su PROSA la gobierna
`build_language_directive`, no este campo.

QUÉ ANCLA. Que la clave esté en la tupla, y que la poda por comida siga intacta: el
arreglo es añadir un literal, y el riesgo de un literal mal puesto es tocar la rama
que ya funcionaba.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_AGENT = _BACKEND / "agent.py"

_MARKER = "P2-DISPLAY-CHAT-PRUNE"


def _fuente() -> str:
    if not _AGENT.exists():
        pytest.skip(f"{_AGENT} no existe en este checkout")
    return _AGENT.read_text(encoding="utf-8")


def _claves_podadas(src: str) -> list[str]:
    m = re.search(r"_CHAT_PLAN_PRUNE_KEYS\s*=\s*\((.*?)\n\)", src, re.S)
    assert m, "no encuentro `_CHAT_PLAN_PRUNE_KEYS` — ¿cambió de nombre?"
    cuerpo = m.group(1)
    # Sólo las líneas de código: un comentario que cite una clave no la poda.
    codigo = "\n".join(
        ln for ln in cuerpo.splitlines() if not ln.strip().startswith("#")
    )
    return re.findall(r"[\"']([^\"']+)[\"']", codigo)


def test_display_de_nivel_plan_se_poda() -> None:
    claves = _claves_podadas(_fuente())
    assert "_display" in claves, (
        "`_display` no está en `_CHAT_PLAN_PRUNE_KEYS`. La copia de NIVEL PLAN "
        "(nombre + insights traducidos, una entrada por cada idioma visitado) se "
        "serializa al system prompt en cada turno del chat. La poda por comida, que "
        "sí existe, no la cubre: es un recorrido aparte sobre `days[*].meals[*]`. "
        f"[{_MARKER}]"
    )


def test_la_poda_por_comida_sigue_intacta() -> None:
    """NO REGRESIÓN. El arreglo es añadir un literal a una tupla; lo que puede
    romperse por accidente es la rama que YA funcionaba."""
    src = _fuente()
    assert re.search(r"k\s*!=\s*[\"']_display[\"']", src), (
        "desapareció la poda por comida de `_display` (el filtro dentro de "
        f"`days[*].meals[*]`). [{_MARKER}]"
    )


def test_la_tupla_conserva_las_demas_claves() -> None:
    """Un `_display` añadido de forma torpe podría haber sustituido a otra entrada."""
    claves = set(_claves_podadas(_fuente()))
    imprescindibles = {
        "aggregated_shopping_list",
        "_shopping_coherence_block",
        "_culinary_contract_violations",
        "dish_quality_report",
        "_review_issues_raw",
    }
    faltan = imprescindibles - claves
    assert not faltan, (
        f"la tupla perdió claves que ya estaban: {sorted(faltan)}. Todas entraron "
        f"porque engordaban cada turno del chat. [{_MARKER}]"
    )
