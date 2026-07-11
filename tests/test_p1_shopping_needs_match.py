"""[P1-SHOPPING-NEEDS-MATCH · 2026-07-11] El filtro forward-looking de la lista de
compras matchea por CONTENCIÓN, no por igualdad exacta.

Bug vivo (primer plan modo-Nevera del owner): el PDF de la lista mostró SOLO 3 ítems
"al gusto" con "45 ingredientes ya están en tu Nevera y fueron excluidos" — con una
Nevera de 7 filas (verificado por SQL). Causa: `remainingNeedsSet` (P5-PRESENCE-
FORWARD-LOOKING, 2026-06-23) guarda LÍNEAS de receta normalizadas CON cantidades
("1½ tomate", "5 clara de huevo") mientras el check consulta el NOMBRE CANÓNICO del
ítem ("tomate") por igualdad exacta → casi nada matchea → casi toda la lista se
ocultaba como "el plan restante ya no lo usa". Los totales (RD$) se calculan de la
lista completa → PDF internamente incoherente.

Contrato:
1. El set se envuelve con un matcher de CONTENCIÓN con límites de palabra
   (" tomate " ⊂ " 1½ tomate "); el exact-match queda como fast-path.
2. Falsos positivos solo MUESTRAN de más — la dirección fail-open del diseño
   (mejor un ítem extra que esconder algo que el usuario necesita comprar).

tooltip-anchor: P1-SHOPPING-NEEDS-MATCH
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_DASH_SRC = (_BACKEND.parent / "frontend" / "src" / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")


def test_containment_matcher_wraps_the_set():
    assert "P1-SHOPPING-NEEDS-MATCH" in _DASH_SRC
    assert "const _paddedNeeds = [...remainingNeedsSet].map(k => ` ${k} `);" in _DASH_SRC, (
        "las líneas del set deben quedar padded para contención con límites de palabra"
    )
    assert "_paddedNeeds.some(line => line.includes(_pk))" in _DASH_SRC, (
        "el matcher debe probar contención — la igualdad exacta escondía ~toda la lista "
        "(PDF con 3 ítems y '45 excluidos' sobre una Nevera de 7 filas)"
    )
    assert "if (_exact.has(key)) return true;" in _DASH_SRC, "exact-match como fast-path"


def test_word_boundary_padding():
    # " sal " NO debe matchear " salsa de tomate " — el padding con espacios en AMBOS
    # lados es lo que da el límite de palabra.
    assert "const _pk = ` ${key} `;" in _DASH_SRC, (
        "la key consultada debe ir padded — sin espacios, 'sal' matchearía 'salsa'"
    )


def test_check_still_uses_has_interface():
    assert "remainingNeedsSet && !(remainingNeedsSet.has(nameKey1) || remainingNeedsSet.has(nameKey2))" in _DASH_SRC, (
        "el consumo sigue siendo .has() — el wrapper expone la misma interfaz "
        "(el path ciclo-terminado sigue siendo un Set real)"
    )
