"""[P1-CHAT-DIARY-WHERE · 2026-07-31] No mandes al usuario a un panel vacío.

El coach registró correctamente la cena de anoche (`days_ago=1`) y cerró con
«puedes ajustarlo o borrarlo desde **Progreso en Tiempo Real**». Pero ese panel
muestra SOLO el día de hoy, así que seguía marcando 0 comidas.

El usuario borró el registro anterior, repitió el mensaje, vio el panel en cero y
reportó "no se registró" — cuando la fila SÍ estaba en `consumed_meals`, fechada
el día anterior a las 18:51 RD. *Un registro correcto que el usuario no puede
ver es indistinguible de uno que falló*, y el propio mensaje del coach fue lo que
lo mandó a comprobarlo donde no podía estar.

La instrucción era incondicional en el prompt: no distinguía `days_ago=0` de
`days_ago>0`. Y vivía DUPLICADA en las dos variantes de
`build_tools_instructions` (stream y non-stream) — arreglar una sola habría
dejado el otro camino mintiendo igual, que es el modo de fallo que
`_CHAT_BREVITY_RULES` evita compartiendo el bloque byte-a-byte.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from prompts.chat_agent import (  # noqa: E402
    build_tools_instructions,
    build_tools_instructions_stream,
)

VARIANTES = {
    "non-stream": build_tools_instructions("u-1"),
    "stream": build_tools_instructions_stream("u-1"),
}


@pytest.mark.parametrize("nombre", sorted(VARIANTES))
def test_avisa_que_el_panel_solo_muestra_hoy(nombre: str):
    txt = VARIANTES[nombre]
    assert "P1-CHAT-DIARY-WHERE" in txt, (
        f"la variante «{nombre}» no lleva el aviso: seguiría remitiendo a "
        f"'Progreso en Tiempo Real' para una comida de otro día"
    )
    assert "days_ago" in txt.split("P1-CHAT-DIARY-WHERE")[1][:600], (
        "el aviso no ata la condición al parámetro real (`days_ago`); sin eso "
        "el modelo tiene que adivinar cuándo aplica"
    )


@pytest.mark.parametrize("nombre", sorted(VARIANTES))
def test_sigue_remitiendo_al_panel_cuando_SI_es_de_hoy(nombre: str):
    """Anti-oscilación: el caso normal (days_ago=0) no pierde su indicación.

    Sin este control, 'no lo mandes al panel' se podría leer como 'nunca
    menciones el panel', y el usuario se quedaría sin saber dónde corregir el
    caso más común.
    """
    txt = VARIANTES[nombre]
    assert "Progreso en Tiempo Real" in txt
    bloque = txt.split("P1-CHAT-DIARY-WHERE")[1][:600]
    assert "days_ago=0" in bloque, (
        "el aviso no dice explícitamente que con `days_ago=0` SÍ se remite al "
        "panel — quedaría como una prohibición general"
    )


def test_las_dos_variantes_llevan_el_mismo_aviso():
    """Duplicar la instrucción es lo que permite que una quede vieja."""
    # Acotado al FINAL REAL de la nota, no a una ventana de bytes fija: un
    # tamano fijo se sale del aviso y entra en el texto que si difiere entre
    # variantes, y el test fallaria por su propia regla de corte.
    FIN = "le remites a 'Progreso en Tiempo Real'."
    trozos = {}
    for n, t in VARIANTES.items():
        cola = t.split("P1-CHAT-DIARY-WHERE")[1]
        assert FIN in cola, f"la variante {n} no cierra el aviso como se espera"
        trozos[n] = cola[:cola.index(FIN) + len(FIN)]
    assert trozos["stream"] == trozos["non-stream"], (
        "el aviso difiere entre la variante stream y la non-stream — el chat "
        "diría una cosa u otra según el endpoint que atienda el turno"
    )
