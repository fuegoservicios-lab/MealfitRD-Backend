"""[P2-I18N-DISPLAY-CANONICO-SE-COME-EL-CONECTOR · 2026-08-23] Con el formato
«0.75 cda (5 g) de Almendras», el canónico salía como «de Almendras» y un gloss PERFECTO caía
al español.

`_extract_canonical_name` limpiaba en este orden: prefijo de cantidad → conector inicial →
paréntesis → modificador final. Con una cantidad entre paréntesis DESPUÉS de la unidad —la
forma «0.75 cda (5 g) de X», que el motor emite para cucharadas y tazas con su equivalente
en gramos— tras quitar el prefijo queda «(5 g) de Almendras»; el conector `de` ya no está al
principio y no se quita; el paréntesis se quita después, y queda «de Almendras». El gloss
del modelo dice «0.75 tbsp (5 g) of almonds (Almendras)», que es correcto, pero el
validador busca «de Almendras» como palabra dentro de la línea traducida, no lo encuentra, y
la línea cae al español.

MEDIDO sobre un plan vivo: 24,6 % de las líneas de ingredientes llevan esa forma. Una de
cada cuatro líneas correctas se descartaba.

El arreglo es de ORDEN: los paréntesis se quitan ANTES que el conector, para que el
conector vuelva a estar al principio.

tooltip-anchor: P2-I18N-DISPLAY-CANONICO-SE-COME-EL-CONECTOR
"""
from __future__ import annotations

import pytest

from plan_display_i18n import _extract_canonical_name

_MARKER = "P2-I18N-DISPLAY-CANONICO-SE-COME-EL-CONECTOR"


@pytest.mark.parametrize("linea,esperado", [
    ("0.75 cda (5 g) de Almendras", "Almendras"),
    ("2 tazas (300 g) de Arroz blanco", "Arroz blanco"),
    ("1 cdta (4 g) de Aceite de oliva", "Aceite de oliva"),
    ("½ taza (120 ml) de Leche", "Leche"),
    ("3 unidades (150 g) del Plátano verde", "Plátano verde"),
])
def test_la_cantidad_entre_parentesis_no_deja_el_conector_pegado(linea, esperado) -> None:
    assert _extract_canonical_name(linea) == esperado, (
        f"«{linea}» → {_extract_canonical_name(linea)!r}: el conector se quedó pegado al "
        f"canónico. El validador buscará «de {esperado}» en la traducción, no lo hallará, y "
        f"una línea CORRECTA caerá al español. Medido: 24,6 % de un plan vivo. [{_MARKER}]"
    )


@pytest.mark.parametrize("linea,esperado", [
    ("180 g de Pechuga de pollo", "Pechuga de pollo"),          # sin paréntesis: como antes
    ("Sal al gusto", "Sal"),                                     # modificador final
    ("2 dientes de Ajo", "Ajo"),                                 # partitivo
    ("Pechuga de pollo (sin piel)", "Pechuga de pollo"),         # paréntesis al final
    ("1 lata de Atún (en agua)", "Atún"),
])
def test_no_regresion_las_formas_de_siempre_siguen_igual(linea, esperado) -> None:
    assert _extract_canonical_name(linea) == esperado, f"regresión en «{linea}» [{_MARKER}]"
