"""[P1-INSIGHTS-I18N · 2026-08-20] El panel de Razonamiento seguia en espanol.

Reportado con captura: cabeceras en ingles (DIAGNOSIS / ACTION PLAN / CHEF'S TIP) y el
cuerpo entero en espanol. Los TITULOS ya pasaban por `t()`; el texto lo escribe el LLM y
nadie lo pasaba por la capa `_display`.

Se traduce por el criterio que quedo fijado esa jornada: no es «lo que escribe el LLM no
se toca», es «lo que el motor usa como IDENTIFICADOR no se toca». Por el razonamiento no
resuelve nadie -- es prosa para el usuario, como el titulo del plan.

SU CONTRATO ES EL DE `recipe`, NO EL DE `plan_name`: array ALINEADO POR INDICE, misma
longitud y mismo orden o se descarta ENTERO. El panel rotula por POSICION
(0=Diagnostico, 1=Plan de Accion, 2=Tip del Chef), asi que una traduccion con un
elemento de menos no seria «peor texto» -- pondria el consejo del chef bajo el titulo de
diagnostico.

tooltip-anchor: P1-INSIGHTS-I18N
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────── el RAZONAMIENTO del plan (P1-INSIGHTS-I18N) ───────────────
#
# [P1-INSIGHTS-I18N · 2026-08-20] El panel «Diagnóstico / Plan de Acción / Tip del Chef»
# seguia en espanol con la app en ingles: los TITULOS ya pasaban por `t()`, el CUERPO lo
# escribe el LLM y nadie lo pasaba por `_display`.
#
# Se traduce por el criterio de la jornada: no es «lo que escribe el LLM no se toca», es
# «lo que el motor usa como IDENTIFICADOR no se toca». Por el razonamiento no resuelve
# nadie.
#
# SU CONTRATO ES EL DE `recipe`, NO EL DE `plan_name`: array ALINEADO POR INDICE, misma
# longitud y mismo orden o se descarta ENTERO. El panel rotula por POSICION, asi que una
# traduccion con un elemento de menos no seria «peor texto» -- pondria el consejo del
# chef bajo el titulo de diagnostico.

def test_validate_insights_exige_misma_longitud():
    from plan_display_i18n import _validate_insights
    assert _validate_insights(["a", "b"], ["x", "y"]) == ["a", "b"]
    assert _validate_insights(["a"], ["x", "y"]) is None, "acepto una traduccion incompleta"
    assert _validate_insights(["a", "b", "c"], ["x", "y"]) is None
    assert _validate_insights(["a", "   "], ["x", "y"]) is None, "acepto una entrada vacia"
    assert _validate_insights("a", ["x"]) is None
    assert _validate_insights(["a"], None) is None


def test_el_prompt_pide_los_insights_solo_cuando_los_hay():
    """Un plan sin razonamiento no debe ver el bloque: pagar tokens por una instruccion
    inaplicable es la version barata de confundir al modelo."""
    from plan_display_i18n import _build_prompt
    t = [{"name": "N", "description": "D", "recipe": ["r"], "ingredients": ["i"]}]
    con = _build_prompt(t, "en-US", insights=["uno", "dos"])
    sin = _build_prompt(t, "en-US")
    assert "INSIGHTS:" in con and "[0] uno" in con
    assert "INSIGHTS" not in sin


def test_la_directiva_de_insights_es_NATIVA_por_locale():
    """Misma leccion que P1-COACH-LANGUAGE-NATIVE: una instruccion en espanol pidiendo
    otro idioma es la senal mas debil posible."""
    from plan_display_i18n import _INSIGHTS_ADDENDUM
    assert set(_INSIGHTS_ADDENDUM) == {"en-US", "pt-BR", "fr-FR", "it-IT"}
    assert "into English" in _INSIGHTS_ADDENDUM["en-US"]
    assert "Português" in _INSIGHTS_ADDENDUM["pt-BR"]
    assert "français" in _INSIGHTS_ADDENDUM["fr-FR"]
    assert "italiano" in _INSIGHTS_ADDENDUM["it-IT"]


def test_la_directiva_exige_MISMO_ORDEN_y_MISMA_CANTIDAD():
    """Es lo que hace utilizable el array: sin esa promesa, el modelo puede resumir tres
    insights en dos y el panel rotula mal."""
    from plan_display_i18n import _INSIGHTS_ADDENDUM
    for loc, texto in _INSIGHTS_ADDENDUM.items():
        bajo = texto.lower()
        # `ordre` (frances) faltaba en la primera version de ESTA comprobacion, no en la
        # directiva: el guard acusaba a un texto correcto. Un guard con la lista de
        # idiomas incompleta es un falso positivo con pinta de hallazgo.
        assert any(x in bajo for x in ("order", "orden", "ordem", "ordine", "ordre")), loc
        assert ("number" in bajo or "quantidade" in bajo or "nombre" in bajo
                or "numero" in bajo), loc


def test_el_persist_tiene_el_MISMO_guard_TOCTOU_que_el_nombre():
    """Si una regeneracion escribio otro razonamiento mientras traduciamos, pegar la
    traduccion vieja seria peor que no traducir: el panel diria una cosa y el plan otra."""
    import io as _io
    src = _io.open(Path(__file__).resolve().parent.parent / "plan_display_i18n.py",
                   encoding="utf-8").read()
    assert 'if pd.get("insights") == insights_snapshot:' in src
    assert 'counters["insights_mismatch"] = True' in src
