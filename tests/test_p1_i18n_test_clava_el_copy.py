"""[P1-I18N-TEST-CLAVA-EL-COPY · 2026-08-22] Un guard estaba siendo la RAZON de que tres
cadenas visibles siguieran en espanol en los cinco idiomas.

`Dashboard.p1_pantry_strict_consent.test.js` localizaba sus ventanas asi:

    _src.indexOf("toast.success('¡Menu Actualizado!'")
    _sliceFrom('title="¿Bloquear este plato?"', 3000)
    expect(win).toContain("loadingTitle: '👎 Registrando preferencia...'")

Envolverlas en `t()` habria hecho desaparecer el marcador, asi que el codigo de produccion
llevaba escrito, literalmente, «SIN `t()`: el test usa esta cadena como MARCADOR». Un test
no puede ser el motivo de que una pantalla no se traduzca: el ancla se cambia, el copy no.

Y habia una CUARTA de la misma clase en el mismo fichero que nadie habia nombrado: el
titulo del toast del arreglo de sodio (`Día N arreglado`), clavado por
`Dashboard.p1_fix_sodium_day.test.js`.

DOS CLASES DE ANCLA QUE ESTE GUARD PROHIBE, y las dos aparecieron en la misma tanda:

  (a) ANCLAR EL COPY. Localizar un bloque por una cadena que el usuario LEE. El dia que esa
      cadena se traduzca --que es lo que tiene que pasar-- el guard se pone rojo sin que la
      conducta empeore, y la salida barata es dejar el copy sin traducir.

  (b) VENTANA POR BYTES. `slice(idx, idx + N)` mide el TAMANO del codigo, no su estructura:
      anadir un COMENTARIO acerca el ancla al borde sin que nada avise. Medido en esta
      tanda: tres guards distintos se pusieron rojos por eso (uno por 26 bytes), y dos de
      ellos YA habian sido ampliados antes por la misma causa. Ampliarlos otra vez solo
      compra tiempo hasta la siguiente linea.

tooltip-anchor: P1-I18N-TEST-CLAVA-EL-COPY
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_SRC = _ROOT / "frontend" / "src"
_TESTS = _SRC / "__tests__"
_DASHBOARD = _SRC / "pages" / "Dashboard.jsx"

_MARKER = "P1-I18N-TEST-CLAVA-EL-COPY"

# Las cadenas liberadas. Cada una tiene que estar DENTRO de un `t(...)`.
#
# [P1-I18N-BOTON-CAMBIAR-PLATO-CLAVADO-POR-TEST · 2026-08-23] Eran cuatro y se añade la
# quinta, que era la MÁS visible de todas y se quedó fuera: el botón principal de cada
# comida del menú. `P1_weeknav_mobile_size.test.js` exigía `Cambiar Plato</span>` y el
# código llevaba escrito «SIN t(): el test exige la cadena literal». Un usuario en francés
# leía siete avisos que decían «usa *Changer de plat*» nombrando un botón que en su
# pantalla se llamaba «Cambiar Plato». El test se reancló por estructura (`data-testid`).
#
# ⚠️ «Cambiar Plato» NO está en esta lista, y es a propósito. Lo intenté y la mutación lo
# desmontó: con el rótulo del botón clavado otra vez, este guard seguía VERDE, porque el
# fichero ya tiene TRES `t('Cambiar Plato')` en otros sitios (los avisos que lo nombran en
# negrita). Este guard mide «existe un t(cadena) en el fichero», no «ESE sitio pasa por t»,
# y para una cadena que aparece en varios sitios eso es inerte. El ancla que sí distingue
# el botón es `P1_weeknav_mobile_size.test.js` (por `data-testid`, verificada por mutación).
# Añadirla aquí daría cobertura de papel.
_CADENAS_LIBERADAS = [
    "¡Menú Actualizado!",
    "¿Bloquear este plato?",
    "👎 Registrando preferencia...",
    "Día {n} arreglado",
]


def _saltar_si_no_hay_frontend() -> None:
    if not _SRC.is_dir():
        pytest.skip(f"no existe {_SRC} (¿worktree sin el repo hermano?)")


@pytest.mark.parametrize("cadena", _CADENAS_LIBERADAS)
def test_la_cadena_esta_envuelta_en_t(cadena: str) -> None:
    """Lo que el usuario lee pasa por el motor de idiomas. Sin excepciones por test."""
    _saltar_si_no_hay_frontend()
    src = io.open(_DASHBOARD, encoding="utf-8").read()
    assert cadena in src, (
        f"«{cadena}» ya no está en Dashboard.jsx — si se renombró, actualiza este guard; "
        f"si se borró, quítala de la lista. [{_MARKER}]"
    )
    # `t('…')` o `t("…")`, admitiendo espacios. Se busca la cadena precedida de la llamada.
    patron = re.compile(r"t\(\s*['\"]" + re.escape(cadena) + r"['\"]")
    assert patron.search(src), (
        f"«{cadena}» aparece en Dashboard.jsx SIN pasar por `t()`. Si un test la usa como "
        f"marcador, el arreglo es cambiar el ANCLA del test, no dejar la pantalla en "
        f"español. [{_MARKER}]"
    )


def test_ningun_test_ancla_esas_cadenas_como_marcador() -> None:
    """La otra mitad: que el ancla no vuelva.

    Si alguien reintroduce `indexOf("toast.success('¡Menú Actualizado!'")`, el copy vuelve a
    quedar preso del guard aunque hoy esté envuelto.
    """
    _saltar_si_no_hay_frontend()
    culpables = []
    for p in _TESTS.rglob("*.test.js*"):
        txt = io.open(p, encoding="utf-8").read()
        for cadena in _CADENAS_LIBERADAS:
            # Se busca la cadena dentro de un localizador (`indexOf`, `_sliceFrom`,
            # `search`), NO en un `expect(...).toContain(t('…'))`, que es legítimo.
            for m in re.finditer(r"(?:indexOf|_sliceFrom|search)\s*\(\s*(['\"])(.*?)\1", txt):
                if cadena in m.group(2):
                    culpables.append(f"{p.name}: localiza por «{cadena}»")
    assert not culpables, (
        f"Estos tests vuelven a localizar un bloque por una cadena que el usuario LEE: "
        f"{culpables}. El día que se traduzca, el guard se pone rojo sin que la conducta "
        f"empeore — y la salida barata es dejar el copy sin traducir, que es exactamente "
        f"lo que pasó. Ancla por estructura. [{_MARKER}]"
    )


# TRINQUETE de la clase (b). Puede BAJAR, nunca subir.
#
# Medido el 2026-08-22 tras convertir las ventanas que esta tanda rompió. El resto son
# decenas, y un guard que exija cero estaría rojo desde el minuto uno — que es la forma más
# rápida de que alguien lo desactive, y entonces no protege nada. Mismo criterio que el
# trinquete de español sin envolver: se acota por arriba y se baja cuando se toca el
# fichero (política boy-scout), en vez de prometer una limpieza que no se va a hacer hoy.
_PRESUPUESTOS_TOLERADOS = {
    "Dashboard.p1_pantry_strict_consent.test.js": 0,
    "Dashboard.p1_fix_sodium_day.test.js": 1,
    "DashboardPlanSelfHeal.test.js": 8,
    "History.audit_hist_8_missing_days_block.test.js": 29,
    "History.audit_hist_10_chunk_metrics_tab.test.js": 4,
}


def test_las_ventanas_por_presupuesto_de_bytes_no_aumentan() -> None:
    """La clase (b). Tres guards se rompieron por esto en una sola tanda.

    `slice(idx, idx + N)` mide el TAMAÑO del código, no su estructura: añadir un COMENTARIO
    acerca el ancla al borde sin que nada avise. Uno de los tres se pasó por 26 bytes, y dos
    de estos ficheros YA habían sido ampliados antes por la misma causa — ampliarlos otra
    vez solo compra tiempo hasta la línea siguiente.

    Lo que este trinquete impide es que la deuda CREZCA. Bajarlo es trabajo de quien toque
    el fichero: acotar por el siguiente hito estructural (el hermano, el cierre del bloque).
    """
    _saltar_si_no_hay_frontend()
    # `slice(x, x + 1234)` — un literal numérico grande como fin de ventana.
    presupuesto = re.compile(r"\.slice\(\s*\w+\s*,\s*\w+\s*\+\s*(\d{4,})\s*\)")
    subidas = []
    for nombre, tope in _PRESUPUESTOS_TOLERADOS.items():
        p = _TESTS / nombre
        if not p.exists():
            continue
        txt = io.open(p, encoding="utf-8").read()
        n = len(presupuesto.findall(txt))
        if n > tope:
            subidas.append(f"{nombre}: {n} (tolerado {tope})")
    assert not subidas, (
        f"Estas ventanas por BYTES aumentaron: {subidas}. Miden el tamaño del código y no "
        f"su estructura, así que un comentario nuevo puede dejar el ancla fuera sin que "
        f"nada avise. Acota por el siguiente hito estructural en vez de sumar bytes; si de "
        f"verdad bajaste alguna, actualiza `_PRESUPUESTOS_TOLERADOS`. [{_MARKER}]"
    )
