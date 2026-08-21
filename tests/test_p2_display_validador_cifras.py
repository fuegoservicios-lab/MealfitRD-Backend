"""[P2-DISPLAY-VALIDADOR-SIN-CIFRAS · 2026-08-21] El validador de `_display` no
comprobaba NI UNA cifra.

QUÉ COMPROBABA. `_validate_and_build_display` exigía: `name` no vacío, `description`
no vacía, que `recipe` e `ingredients` tuvieran la MISMA LONGITUD que el original, y
que el nombre canónico del alimento siguiera dentro de la línea traducida. Nada más.

Así que un LLM que devuelve «1 cup» donde el original decía «180 g» pasaba entero, se
persistía, y el usuario cocinaba con la cantidad equivocada. Y no es una rareza: es la
tentación natural de un modelo al que se le pide traducir a inglés estadounidense, que
convierte unidades porque cree que ayuda.

POR QUÉ IMPORTA MÁS QUE UN RÓTULO MAL TRADUCIDO. El resto de la capa `_display` es
cosmética —si una palabra sale rara, se lee raro—. Las cantidades no: son el dato que
el usuario ejecuta con las manos. Y el motor sigue calculando macros sobre el original
en español, así que una conversión inventada NO se refleja en las kilocalorías: la
pantalla y el cálculo dejan de contar lo mismo, en silencio.

LA REGLA. Se extrae el multiconjunto de números de la línea original y de la traducida
y se exige que coincidan. El separador decimal se normaliza antes de comparar, porque
«1.5» → «1,5» en francés es una traducción CORRECTA y no puede tratarse como pérdida.

Ante un desajuste se descarta ESA LÍNEA y se cae al original español — el mismo patrón
per-línea que ya usa el check del canónico. Descartar el meal entero por una línea
sería peor: se perdería la traducción de todo lo demás.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
pdi = importlib.import_module("plan_display_i18n")

_MARKER = "P2-DISPLAY-VALIDADOR-SIN-CIFRAS"


def _original(ingredientes, receta=None) -> dict:
    return {
        "name": "Habichuelas guisadas",
        "description": "Guiso dominicano.",
        "recipe": receta if receta is not None else ["Cocinar 20 minutos."],
        "ingredients": ingredientes,
    }


def _traducido(ingredientes, receta=None) -> dict:
    return {
        "name": "Stewed red beans",
        "description": "Dominican stew.",
        "recipe": receta if receta is not None else ["Cook for 20 minutes."],
        "ingredients": ingredientes,
    }


def test_una_conversion_de_unidad_cae_al_original() -> None:
    """El caso que motiva el P-fix: el modelo «ayuda» convirtiendo gramos a tazas."""
    out = pdi._validate_and_build_display(
        _original(["180 g Habichuelas rojas"]),
        _traducido(["1 cup Red beans"]),
    )
    assert out is not None, "el meal entero no debe descartarse por una línea"
    assert out["ingredients"] == ["180 g Habichuelas rojas"], (
        "La línea con la cantidad cambiada se persistió. El usuario cocina con la "
        "cantidad equivocada y el motor sigue calculando macros sobre el original: la "
        f"pantalla y el cálculo dejan de contar lo mismo. [{_MARKER}]"
    )


def test_una_traduccion_que_conserva_la_cifra_pasa() -> None:
    """MUTACIÓN DE CONTROL. Sin esto, un validador que rechazara SIEMPRE pasaría el
    test de arriba sin probar nada.

    La línea conserva el nombre canónico español porque ese es el formato de gloss
    que el diseño exige (`Habichuelas rojas (red beans)`): el check del canónico, que
    ya existía, cae al original si el alimento desaparece de la traducción. Sin esa
    forma, este test mediría el check del canónico y no el de cifras.
    """
    out = pdi._validate_and_build_display(
        _original(["180 g Habichuelas rojas"]),
        _traducido(["180 g Habichuelas rojas (red beans)"]),
    )
    assert out["ingredients"] == ["180 g Habichuelas rojas (red beans)"]


def test_el_separador_decimal_del_idioma_no_cuenta_como_perdida() -> None:
    """«1.5» → «1,5» es lo que un francés espera leer. Tratarlo como desajuste
    convertiría el guard en un generador de falsos positivos justo en el idioma que
    más lo necesita."""
    out = pdi._validate_and_build_display(
        _original(["1.5 kg Habichuelas rojas"]),
        _traducido(["1,5 kg Habichuelas rojas (haricots rouges)"]),
    )
    assert out["ingredients"] == ["1,5 kg Habichuelas rojas (haricots rouges)"], (
        f"El cambio de separador decimal se leyó como cifra perdida. [{_MARKER}]"
    )


@pytest.mark.parametrize(
    "orig,trad,etiqueta",
    [
        ("Añadir 2 cucharadas de aceite", "Add 2 tbsp of oil", "unidad traducida, cifra intacta"),
        ("Medir 1/2 taza de arroz", "Measure 1/2 cup of rice", "fracción"),
        ("Cocinar 12-15 minutos", "Cook for 12-15 minutes", "rango"),
        ("Sofreír el sazón", "Sauté the sofrito", "sin cifras en ninguno"),
        ("Servir de inmediato", "Serve right away", "prosa sin números"),
    ],
)
def test_no_molesta_a_las_traducciones_buenas(orig, trad, etiqueta) -> None:
    """Un guard que grita con lo correcto se acaba apagando.

    Los casos van en `recipe` y no en `ingredients` A PROPÓSITO: `ingredients` ya
    tiene su propio check per-línea —el del nombre canónico— que cae al original
    cuando el alimento no aparece en la traducción. Mezclarlos aquí mediría ese check
    y no el de cifras, y el test no sabría cuál de los dos disparó.
    """
    out = pdi._validate_and_build_display(
        _original(["30 g Sal"], receta=[orig]),
        _traducido(["30 g Salt"], receta=[trad]),
    )
    assert out["recipe"] == [trad], f"falso positivo con {etiqueta} [{_MARKER}]"


def test_tambien_protege_los_pasos_de_la_receta() -> None:
    """Los tiempos y las cantidades viven también en `recipe`, y ahí no había NINGÚN
    check per-línea: el array entraba tal cual."""
    out = pdi._validate_and_build_display(
        _original(["30 g Sal"], receta=["Hornear 45 minutos a 180 grados."]),
        _traducido(["30 g Salt"], receta=["Bake for 1 hour at 350 degrees."]),
    )
    assert out["recipe"] == ["Hornear 45 minutos a 180 grados."], (
        "Un paso con los tiempos y la temperatura cambiados se persistió tal cual. "
        f"[{_MARKER}]"
    )


def test_un_paso_bien_traducido_si_pasa() -> None:
    out = pdi._validate_and_build_display(
        _original(["30 g Sal"], receta=["Hornear 45 minutos a 180 grados."]),
        _traducido(["30 g Salt"], receta=["Bake for 45 minutes at 180 degrees."]),
    )
    assert out["recipe"] == ["Bake for 45 minutes at 180 degrees."]
