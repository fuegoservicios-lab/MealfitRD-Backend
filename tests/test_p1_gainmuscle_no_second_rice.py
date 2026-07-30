"""[P1-GAINMUSCLE-NO-SECOND-RICE · 2026-07-30] Dos arroces en el mismo plato, uno con su propio
paso de cocción inyectado.

Caso vivo del owner ("ceviche de queso", plan de 30 días):

    Ingredientes:  45 g de Arroz integral · 10 g de ñame · 65 g de queso gouda · … ·
                   **40 g de arroz blanco crudo**
    Pasos:  2) …cocina arroz integral en 1 taza de agua…
            3) 🍚 Cuece el arroz blanco de tus ingredientes según el paquete y sírvelo como
               acompañante.

El refill calórico de ganancia muscular busca `"de arroz blanco cocido"` para decidir si SUMA a una
línea existente o CREA una nueva — y esa cadena es exactamente la que él mismo escribe. Un plato con
"45 g de Arroz integral" le resulta invisible ⇒ anexa el segundo arroz y le inyecta su paso.

⚠️ **Buscar solo la forma que tú mismo produces no es detectar: es reconocerte.** El guard tiene que
preguntar por el ALIMENTO (¿este plato ya lleva arroz?), no por la línea concreta que este bloque
escribe.

Se salta la comida y el bucle prueba con otra principal. Quedarse algo corto del piso calórico es el
fallo barato; servir dos arroces, no.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from constants import strip_accents as _sa

_BACKEND = Path(__file__).resolve().parents[1]
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def _ya_lleva_arroz(ingredientes):
    """Réplica exacta del predicado del guard, para poder ejercitarlo sin montar el pipeline."""
    return any(
        "arroz" in _sa(str(l).lower()) and "arroz blanco cocido" not in str(l).lower()
        for l in ingredientes
    )


# ───────────────────── el predicado ─────────────────────

@pytest.mark.parametrize("linea", [
    "45 g de Arroz integral",
    "40 g de arroz integral cocido",
    "1 taza de arroz basmati",
    "20 g de Arroz Integral",          # mayúsculas
    "½ taza de arróz integral",        # acento espurio del LLM
    "150 g de moro de guandules con arroz",
])
def test_detecta_el_arroz_que_ya_esta(linea):
    assert _ya_lleva_arroz([linea, "65 g de queso gouda"]), (
        f"{linea!r} es arroz y el plato NO debe recibir un segundo")


def test_no_confunde_su_propia_linea():
    """La línea que el refill gestiona (`arroz blanco cocido`) NO cuenta como 'otro arroz' — si
    contara, el bloque no podría nunca sumar sobre su propia siembra en la pasada final."""
    assert not _ya_lleva_arroz(["120g de arroz blanco cocido", "80 g de pollo"])


@pytest.mark.parametrize("ingredientes", [
    ["80 g de pollo", "1 taza de brócoli"],
    ["½ pedazo de ñame (≈150 g)", "1 pechuga de pollo"],
    ["2 rebanadas de pan integral", "30 g de queso gouda"],
    [],
])
def test_un_plato_sin_arroz_si_puede_recibirlo(ingredientes):
    assert not _ya_lleva_arroz(ingredientes)


def test_el_caso_vivo_completo():
    """Los ingredientes tal cual salieron en la receta del owner."""
    ceviche = ["45 g de Arroz integral", "10 g de ñame pelado", "65 g de queso gouda",
               "½ limón (jugo)", "4 ají cubanela picado finamente", "1 cebolla roja picada",
               "¼ taza de cilantro fresco picado", "¼ cda de aceite de oliva",
               "Sal al gusto", "Pimienta negra al gusto"]
    assert _ya_lleva_arroz(ceviche), "este plato no debió recibir 40 g de arroz blanco encima"


# ───────────────────── anclaje en el fuente ─────────────────────

def test_el_guard_esta_antes_de_anexar():
    """Load-bearing: si corriera DESPUÉS del append, el segundo arroz ya estaría puesto."""
    i_guard = _GO.index("P1-GAINMUSCLE-NO-SECOND-RICE")
    i_append = _GO.index('line = f"{add_g}g de arroz blanco cocido"', i_guard - 4000)
    assert i_guard < i_append, "el guard debe decidir ANTES de crear la línea"


def test_el_guard_no_mira_solo_su_propia_cadena():
    i = _GO.index("P1-GAINMUSCLE-NO-SECOND-RICE")
    seg = _GO[i:i + 2200]
    assert '"arroz" in _sa_gm(' in seg, (
        "el guard debe preguntar por el ALIMENTO, no por la línea concreta que este bloque escribe")
    assert '"arroz blanco cocido" not in' in seg, (
        "…y excluir su propia línea, o no podría sumar sobre su siembra previa")


def test_el_paso_inyectado_sigue_condicionado():
    """El paso '🍚 Cuece el arroz blanco…' solo se inyecta si la receta no lo menciona ya; eso no
    cambia. Lo que cambia es que ahora ni siquiera se llega ahí cuando hay otro arroz."""
    assert '"arroz blanco" in str(_s).lower() for _s in _rec_gm' in _GO
