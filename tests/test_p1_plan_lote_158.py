"""[P1-PLAN-LOTE-158 · 2026-09-22] El botón secundario responde como un botón, no como una fila.

REPORTE DEL DUEÑO (con captura): «el sombreado al pasar el mouse por encima de "Ahora no" no me
gusta, quiero algo que le quede y se vea mejor». Es el descarte de la tarjeta «¿Quieres que la IA
te arme el plan?» del modo contador.

CAUSA. Llevaba la variante `fila` del utilitario `data-hover`. `fila` está pensada para una FILA:
su respuesta es un velo tenue MÁS un anillo de 1 px que dibuja el contorno del elemento. En una
lista ese anillo es lo que te dice dónde empieza y acaba lo que vas a pulsar. En un botón, que ya
tiene forma propia, es un recuadro gris que aparece de la nada — y en el «Cancelar» del diálogo,
que ya trae su `1px solid var(--border)`, le pintaba un segundo borde pegado al primero.

  *Reutilizar una respuesta visual no es reutilizar su intención: el anillo de una fila existe
  para dibujar un límite que no se ve, y un botón ya tiene el suyo.*

ARREGLO. Variante `fantasma` para el secundario que acompaña a uno lleno: velo un punto más
fuerte (9 %, porque un botón es más superficie que una fila), el borde propio se aviva si existe
—`border-color` sobre un control con `border: 0` es inerte, así que la misma regla sirve para los
dos casos— y, sobre todo, **el texto sube a `--text-main`**. Ese salto es la señal que más se
nota y la que no existía: en esa tarjeta el rótulo vive en `--text-muted`.

Medido en el arnés con las reglas reales y los tokens del tema oscuro: con `fila` el rótulo se
queda en `rgb(148,163,184)`; con `fantasma` pasa a `rgb(241,245,249)`.

Tooltip-anchor: P1-PLAN-LOTE-158
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONT = _REPO_ROOT / "frontend"


def _leer(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip(f"El árbol del frontend no está junto a este backend ({p}).")
    return p.read_text(encoding="utf-8")


def _bloque(css: str, variante: str) -> str:
    i = css.find(f'[data-hover="{variante}"]:not(:disabled):hover {{')
    assert i != -1, f"No existe la variante `{variante}` del utilitario de hover."
    return css[i:css.index("}", i)]


def test_la_variante_del_secundario_existe_y_sube_el_texto():
    """Es la señal que `fila` no daba, y la que más se nota."""
    b = _bloque(_leer("src/index.css"), "fantasma")
    assert "color: var(--text-main)" in b, (
        "Sin el salto del texto, el hover de un botón secundario apenas se percibe: su rótulo "
        "vive en `--text-muted`."
    )


def test_el_secundario_no_dibuja_anillo():
    """Lo que sobraba. El velo es una sombra INTERIOR; el anillo de `fila`, una exterior de 1 px."""
    b = _bloque(_leer("src/index.css"), "fantasma")
    assert "inset 0 0 0 999px" in b
    assert not re.search(r",\s*0 0 0 1px", b), (
        "Volvió el anillo de contorno: en un botón es un recuadro de la nada, y en el que ya "
        "tiene borde propio son dos bordes pegados."
    )


def test_la_variante_de_fila_no_se_toca():
    """Las filas SÍ necesitan su anillo: es lo que delimita lo que se va a pulsar."""
    b = _bloque(_leer("src/index.css"), "fila")
    assert "0 0 0 1px color-mix(in srgb, var(--text-light) 55%, transparent)" in b


def test_ningun_hover_del_secundario_mueve_el_control():
    """P2-HOVER-NO-MOTION: «no quiero que ningún botón se mueva al pasarle el ratón»."""
    b = _bloque(_leer("src/index.css"), "fantasma")
    assert "transform" not in b and "translate" not in b


def test_los_tres_secundarios_la_declaran():
    """Los dos descartes de la tarjeta del contador y el cancelar del diálogo."""
    tracking = _leer("src/components/dashboard/DashboardTracking.jsx")
    assert tracking.count('data-hover="fantasma"') == 2, (
        "Son DOS ofertas con su descarte (encender el plan y reanudar el pausado); la de "
        "reanudar no declaraba ninguna variante, así que no contestaba al ratón."
    )
    assert 'data-hover="fila"' not in tracking

    dlg = _leer("src/components/common/ConfirmDialog.jsx")
    assert re.search(r'data-hover="fantasma"\s+onClick=\{onCancel\}', dlg)
    assert re.search(r'data-hover="boton"\s+onClick=\{onConfirm\}', dlg)


def test_cada_control_conserva_su_radio():
    """El velo es una sombra: toma la forma del elemento. Mismo guard que el 151."""
    assert re.search(r"\.turnOnGhost\s*\{[^}]*border-radius:",
                     _leer("src/components/dashboard/DashboardTracking.module.css"))
    assert "borderRadius: '0.8rem'" in _leer("src/components/common/ConfirmDialog.jsx")
