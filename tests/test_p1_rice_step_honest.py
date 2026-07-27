"""[P1-RICE-STEP-HONEST · 2026-07-27] El arroz de relleno se contradecía, nadie lo cocía y salía tras el Montaje.

## Lo que veía el owner

En la cena "Papa Rellena de Sardinas y Queso Ricotta":

    ingrediente : 30 g de arroz blanco CRUDO
    paso 4 (tras el Montaje) : 💡 Acompaña este plato con el arroz blanco COCIDO de tus
                               ingredientes para completar las calorías del día.

## Los tres defectos, medidos sobre 92 comidas de 8 planes vivos (13 traían la nota, 14%)

  1. **Contradicción 13/13.** Todas las líneas dicen "crudo" —el display convierte a peso crudo,
     que es lo correcto para comprar— y la nota decía "cocido". Cero líneas decían "cocido".
  2. **Nadie lo cuece.** 10 de 13 no tenían ningún paso que cocinara el arroz. La nota mandaba
     acompañar con un arroz cocido que no existía cocido en ninguna parte.
  3. **Iba después del Montaje.** Se anexaba con `append`, así que el usuario leía un paso de
     cocina DESPUÉS de emplatar — justo lo que `P2-STEP-INSERT-BEFORE-MONTAJE` arregló para los
     demás pasos anexados.

Y de fondo: «para completar las calorías del día» es razonamiento INTERNO (se está rellenando un
piso calórico de ganancia muscular) filtrándose a la receta. El usuario no necesita la
contabilidad del sistema, necesita saber qué hacer con el arroz.

## El arreglo

Un paso de cocina de verdad, insertado ANTES del Montaje:

    🍚 Cuece el arroz blanco de tus ingredientes según el paquete y sírvelo como acompañante.

⚠️ SIN cantidad a propósito: el refill puede sumar gramos en pasadas posteriores y una cifra
escrita aquí quedaría stale (consolidación-proof, ya lo advertía el comentario original).

⚠️ NO se funde en el Montaje como el complemento frío (P1-CLOSER-INTO-MONTAJE): el arroz HAY que
cocerlo, y fundir en el emplatado se saltaría la cocción. Misma asimetría de siempre.

tooltip-anchor: P1-RICE-STEP-HONEST
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import graph_orchestrator as g

_SRC = Path(g.__file__).resolve().read_text(encoding="utf-8")
# Código EMITIDO, sin comentarios: el comentario del fix cita a propósito las frases que se
# eliminaron ("para completar las calorías del día"), y buscarlas en el fuente entero daba un
# falso positivo — lo cazó este mismo test en su primera versión.
_CODIGO = "\n".join(l for l in _SRC.split("\n") if not l.strip().startswith("#"))
_PASO = ("🍚 Cuece el arroz blanco de tus ingredientes según el paquete y "
         "sírvelo como acompañante.")


# ───────────── 1. los tres defectos, cerrados ─────────────

def test_la_nota_ya_no_dice_cocido():
    """13 de 13 líneas dicen 'crudo'; la nota decía 'cocido'. Contradicción universal."""
    assert "arroz blanco cocido de tus" not in _CODIGO, (
        "volvió el texto que contradice la línea de ingrediente ('crudo' vs 'cocido')"
    )


def test_no_se_filtra_el_razonamiento_interno():
    """«para completar las calorías del día» es la contabilidad del piso calórico, no una
    instrucción de cocina."""
    assert "completar las calorías del día" not in _CODIGO, (
        "el razonamiento interno del refill volvió a la receta del usuario"
    )


def test_el_paso_manda_COCER_el_arroz():
    """El defecto que más costaba: el arroz aparecía en los ingredientes y ningún paso lo cocía."""
    i = _SRC.index("P1-RICE-STEP-HONEST")
    bloque = _SRC[i:i + 2400]
    assert "Cuece el arroz blanco" in bloque, "el paso debe instruir a cocer el arroz"


def test_se_inserta_ANTES_del_montaje():
    """Se anexaba con `append` → paso de cocina después de emplatar."""
    i = _SRC.index("P1-RICE-STEP-HONEST")
    bloque = _SRC[i:i + 2400]
    assert "_insert_step_before_montaje(" in bloque, (
        "el paso del arroz debe insertarse antes del Montaje, no anexarse al final"
    )


def test_colocacion_real_antes_del_montaje():
    pasos = ["Mise en place: lava las papas.",
             "El Toque de Fuego: hornea 40 min.",
             "Montaje: sirve caliente."]
    r = g._insert_step_before_montaje(list(pasos), _PASO)
    i_arroz = r.index(_PASO)
    i_mont = next(i for i, s in enumerate(r) if s.lower().startswith("montaje"))
    assert i_arroz < i_mont, f"el arroz quedó después del Montaje: {r}"


# ───────────── 2. lo que NO debe cambiar ─────────────

def test_sigue_sin_cantidad():
    """Consolidación-proof: el refill suma gramos en pasadas posteriores; una cifra aquí
    quedaría stale."""
    i = _SRC.index("P1-RICE-STEP-HONEST")
    bloque = _SRC[i:i + 2400]
    m = re.search(r"Cuece el arroz blanco[^\"']*", bloque)
    assert m, "no se encontró el texto del paso"
    assert not re.search(r"\d+\s*g", m.group(0)), (
        f"el paso lleva una cantidad y no debe: {m.group(0)}"
    )


def test_no_se_duplica_si_la_receta_ya_menciona_el_arroz():
    """El guard original solo miraba 'arroz blanco cocido'; ahora mira 'arroz blanco', que cubre
    también las recetas que ya lo cuecen con otro wording (medido: 3 de 13 lo cocinaban)."""
    i = _SRC.index("P1-RICE-STEP-HONEST")
    bloque = _SRC[i:i + 2400]
    assert '"arroz blanco" in str(_s).lower()' in bloque, (
        "el guard anti-duplicado debe mirar 'arroz blanco', no solo 'arroz blanco cocido': si no, "
        "añade el paso a una receta que YA cuece el arroz"
    )


def test_no_es_una_nota_de_seguridad():
    """Es un paso de cocina real, no una nota ⚠/💡 — y por eso ya no lleva 💡."""
    assert g._is_recipe_safety_note_step(_PASO) is False
