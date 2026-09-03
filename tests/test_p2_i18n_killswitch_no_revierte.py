"""[P2-I18N-KILLSWITCH-NO-REVIERTE · 2026-08-23] «Kill switch TOTAL» era media verdad.

LO QUE ESTE GAP ENSEÑA, Y NO ES LO QUE YO CREÍA

Mi primer diagnóstico fue que el attach de `display_name_en` en el aggregator **no consultaba
el knob**. Era FALSO, y por una razón que ya me había costado un fallo esta misma sesión:
busqué `_plan_display_i18n_enabled()` —el accesor que vive en `plan_display_i18n.py`— y al no
encontrarlo en `shopping_calculator.py` di el attach por desprotegido. Lo consulta, por otra
vía: `_knob_env_bool("MEALFIT_PLAN_DISPLAY_I18N", True)`, en la primera línea de
`_display_name_en_for_item`.

Busqué la FORMA que esperaba en vez de la PROPIEDAD («¿alguien lee esta variable de
entorno?»). Lo destapó validar las anclas del parche ANTES de aplicarlo: una de las tres no
existía con esa forma, y esa discrepancia es lo que obligó a mirar el código de verdad.

Así que el arreglo es sólo la doc — y ahí sí había algo real: **la palabra «TOTAL»**.

El interruptor apaga el motor y el attach. NO revierte lo ya persistido: el knob vive en el
servidor y el pintado del `_display` es del cliente, así que un plan ya enriquecido se sigue
viendo traducido hasta que se regenera. Y no toca el `name_en` del catálogo, que alimenta la
BÚSQUEDA en inglés — buscar no es mostrar.

Un operador que apaga un interruptor en mitad de un incidente necesita saber exactamente qué
deja de pasar. «Total» le habría hecho esperar que lo ya servido cambiara solo.

UNA COSA QUE SÍ ESTABA BIEN Y CONVIENE NO ROMPER

El gateo vive DENTRO de `_display_name_en_for_item`, no en sus llamantes — y hay DOS (el
camino por peso y el de unidades). Gatear los call sites habría dejado uno fuera, que es
peor que no gatear: apagar el interruptor parecería funcionar.

tooltip-anchor: P2-I18N-KILLSWITCH-NO-REVIERTE
"""
from __future__ import annotations

import io
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_MARKER = "P2-I18N-KILLSWITCH-NO-REVIERTE"
_DOC = _BACKEND / "docs" / "plan_display_i18n.md"
_SHOPPING = _BACKEND / "shopping_calculator.py"


def _leer(p: Path) -> str:
    return io.open(p, encoding="utf-8").read()


def test_la_doc_ya_no_promete_un_apagado_total():
    doc = _leer(_DOC)
    assert "kill switch TOTAL" not in doc, (
        "la doc vuelve a prometer un apagado TOTAL. No lo es: el knob vive en el servidor y "
        "el pintado del `_display` es del cliente, así que un plan ya enriquecido se sigue "
        f"viendo traducido hasta que se regenera [{_MARKER}]"
    )
    assert "Qué apaga el kill switch, y qué NO" in doc, (
        "desapareció el apartado que dice qué cubre el interruptor y qué no. Sin él vuelve "
        "la promesa vaga, que es lo que había"
    )


def test_la_doc_nombra_las_dos_cosas_que_el_switch_NO_apaga():
    """Enumerar lo que NO hace es la mitad útil: lo que un operador necesita en un incidente
    es saber qué sigue pasando después de apagar."""
    doc = _leer(_DOC)
    # Se parte por el ENCABEZADO, no por el texto: la fila de la tabla de knobs REFERENCIA
    # la sección («ver «Qué apaga…»»), y esa referencia va ANTES. Partir por el texto suelto
    # cortaba en la tabla y el bloque salía vacío — este test se cazó a sí mismo.
    bloque = doc.split("### Qué apaga el kill switch, y qué NO")[1].split("###")[0]
    assert "persistido" in bloque, (
        f"la doc dejó de decir que lo ya persistido sobrevive al apagado [{_MARKER}]"
    )
    # La AFIRMACIÓN concreta, no la palabra: `name_en` y «búsqueda» aparecen varias veces en
    # el bloque, así que borrar la viñeta entera dejaba el assert satisfecho por el residuo.
    assert "buscar no es mostrar" in bloque, (
        "la doc dejó de decir que el `name_en` del catálogo sigue vivo. Alimenta la "
        "BÚSQUEDA en inglés, y buscar no es mostrar: apagar la traducción no debería dejar "
        f"a un usuario en inglés sin encontrar «chicken» [{_MARKER}]"
    )


def test_el_docstring_del_knob_no_dice_total():
    src = _leer(_BACKEND / "plan_display_i18n.py")
    assert "kill switch total." not in src, (
        f"el docstring del módulo volvió a llamarlo «total» [{_MARKER}]"
    )


def test_el_gateo_del_gloss_vive_en_la_funcion_y_no_en_sus_llamantes():
    """Hay DOS attach (peso y unidades). Gatear los call sites dejaría uno fuera, que es
    peor que no gatear: apagar el interruptor parecería funcionar."""
    src = _leer(_SHOPPING)
    cuerpo = src.split("def _display_name_en_for_item(")[1].split("\ndef ")[0]
    # La CLÁUSULA, no el nombre del knob: el docstring de la función lo menciona, así que
    # sustituir la guarda por un `if False:` dejaba este assert verde con el gateo muerto.
    assert 'if not _knob_env_bool("MEALFIT_PLAN_DISPLAY_I18N", True):' in cuerpo, (
        "`_display_name_en_for_item` dejó de consultar el knob. Sin eso, apagar la capa deja "
        f"el gloss entrando igual en cada lista nueva [{_MARKER}]"
    )
    # Y los dos llamantes siguen llamando sin condición propia: la condición es una sola.
    llamadas = re.findall(r"_name_en = _display_name_en_for_item\(master_item\)", src)
    assert len(llamadas) == 2, (
        f"cambió el número de attach del gloss ({len(llamadas)}, esperaba 2). Si añadiste un "
        f"tercero, comprueba que NO lleve su propia condición: la única vive dentro de la "
        f"función, para que un llamante nuevo nazca cubierto [{_MARKER}]"
    )
