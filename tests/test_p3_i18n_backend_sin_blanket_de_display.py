"""[P3-I18N-BACKEND-SIN-BLANKET-DE-DISPLAY · 2026-08-22] La dirección peligrosa de la
frontera no tenía guard.

LA FRONTERA, DICHA ENTERA

`_display` es una copia de SOLO LECTURA para el frontend. El motor resuelve por el nombre
español canónico: `pantry_names_match`, el guard de coherencia receta↔lista y el backstop
clínico de alergias. La regla de `P1-I18N-DASHBOARD` no es «lo que escribe el LLM no se
toca», es **«lo que el motor usa como IDENTIFICADOR no se toca»**.

Había guards para la dirección fácil (que el frontend lo lea por UNA superficie, que se
invalide cuando cambian los gramos). La dirección peligrosa —un módulo del BACKEND leyendo
el nombre traducido para DECIDIR algo— no tenía ninguno. Es la peor porque falla en
silencio: un resolvedor que empareje por el traducido devuelve «no encontrado» para un
alimento que sí está, y eso no lanza, no alerta y no deja fila.

LA REGLA ES POR OPERACIÓN, NO POR FICHERO

Primera versión de este guard: lista blanca de ficheros. Se puso roja con once sitios
legítimos que yo no había medido —`graph_orchestrator.py`, `tools.py`, `routers/plans.py`
hacen `pop("_display")`— y la lección fue que la lista blanca estaba midiendo lo que no era.

Lo que importa no es QUIÉN toca el campo, es QUÉ hace con él:

  · **Borrarlo** (`pop`, `del`) es INVALIDAR, y es correcto en cualquier sitio: el
    `_display` espeja `ingredients` y `recipe` POR ÍNDICE, así que al reescribir los
    gramos pasa a mentir y hay que tirarlo (`P2-DISPLAY-POP-VECINO`).
  · **Leerlo** es lo que puede acabar en una decisión, y por eso va con lista blanca corta
    y una razón por fila.

LA TRAMPA QUE EL PROPIO GAP ANUNCIABA, Y UNA SEGUNDA

`'_display' in src` nace roto: hay una docena de identificadores que lo contienen sin tener
nada que ver — `_already_displayed_warnings`, `_bad_display`, `_display_completo`,
`_display_link`, `target_weight_display`. Por eso se exige la forma de ACCESO A CLAVE.

Y la segunda, que me costó un falso rojo aquí mismo: filtrar «líneas que empiezan por `#`»
no basta, porque un comentario LARGO envuelve y sus líneas siguientes no empiezan por `#`.
Se usa `tokenize`, que sabe dónde está cada comentario de verdad — y de paso no se traga un
`#` que viva dentro de una cadena, que es como un filtro casero convierte un falso positivo
en un falso VERDE.

tooltip-anchor: P3-I18N-BACKEND-SIN-BLANKET-DE-DISPLAY
"""
from __future__ import annotations

import ast
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_MARKER = "P3-I18N-BACKEND-SIN-BLANKET-DE-DISPLAY"

_CLAVE = "_display"

# Quien puede LEERLO, y por que. La razon es parte del contrato: una excepcion sin razon es
# una puerta que nadie sabe si sigue haciendo falta.
_LECTORES_PERMITIDOS = {
    "plan_display_i18n.py":
        "es el modulo SSOT: lo escribe, y lo relee para no re-pagar una traduccion que ya "
        "esta (`_ya_traducido_*`, P2-DISPLAY-REDESPACHO-SIN-FILTRO)",
    "routers/plans.py":
        "SIRVE los nombres traducidos al cliente en el preview del Historial "
        "(`display_names`). Es pasar el dato a quien lo pinta, no decidir con el",
}

_IGNORAR_DIRS = {"tests", "test_venv", "scripts", "docs", "migrations", "__pycache__"}


def _ficheros():
    for p in sorted(_BACKEND.rglob("*.py")):
        rel = p.relative_to(_BACKEND)
        if any(parte in _IGNORAR_DIRS for parte in rel.parts[:-1]):
            continue
        if p.name.startswith("test_"):
            continue
        yield p, rel.as_posix()


def _es_clave(nodo) -> bool:
    return isinstance(nodo, ast.Constant) and nodo.value == _CLAVE


def _lecturas(src: str) -> list[int]:
    """Las lineas donde se LEE `_display`, via AST.

    Se usa `ast` y no texto por dos razones que ya costaron dos versiones de este guard:

      1. Un `#` o un docstring que EXPLICAN la frontera no son codigo, y un filtro de
         lineas no distingue la continuacion de un comentario largo de una linea normal.
      2. La correccion ingenua —marcar tambien las cadenas via `tokenize`— deja el
         detector INERTE, porque `"_display"` siempre vive dentro de una cadena. Esa
         version paso sus cuatro tests y no cazaba nada.

    El AST no tiene ninguno de los dos problemas: los comentarios no existen en el arbol,
    los docstrings son sentencias y no subindices, y una clave es una clave.

    `pop` y `del` quedan fuera A PROPOSITO: borrar es invalidar, no decidir.
    """
    try:
        arbol = ast.parse(src)
    except SyntaxError:
        return []

    borrados = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Delete):
            for blanco in nodo.targets:
                if isinstance(blanco, ast.Subscript) and _es_clave(blanco.slice):
                    borrados.add((blanco.lineno, blanco.col_offset))

    lineas = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Subscript) and _es_clave(nodo.slice):
            if (nodo.lineno, nodo.col_offset) in borrados:
                continue
            lineas.append(nodo.lineno)
        elif (isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Attribute)
              and nodo.func.attr in ("get", "setdefault") and nodo.args
              and _es_clave(nodo.args[0])):
            lineas.append(nodo.lineno)
    return sorted(set(lineas))


def test_ningun_modulo_del_backend_lee_display_para_decidir():
    culpables = []
    for p, rel in _ficheros():
        if rel in _LECTORES_PERMITIDOS or p.name in _LECTORES_PERMITIDOS:
            continue
        src = p.read_text(encoding="utf-8")
        cuerpo = src.splitlines()
        for n in _lecturas(src):
            culpables.append(f"{rel}:{n}: {cuerpo[n - 1].strip()[:100]}")

    assert not culpables, (
        "Un módulo del backend está LEYENDO `_display`:\n  "
        + "\n  ".join(culpables)
        + "\n\n`_display` es una copia de solo lectura para el frontend. El motor resuelve "
        "por el nombre español canónico — `pantry_names_match`, el guard de coherencia y "
        "el backstop de alergias — y emparejar por el traducido devuelve «no encontrado» "
        "para un alimento que SÍ está: sin lanzar, sin alertar y sin dejar fila.\n"
        "BORRARLO (`pop`/`del`) es otra cosa y está permitido en todas partes: eso es "
        "invalidar, no decidir. Si de verdad necesitas LEERLO para servírselo al cliente, "
        f"añádelo a `_LECTORES_PERMITIDOS` con su razón. [{_MARKER}]"
    )


def test_los_lectores_permitidos_siguen_existiendo():
    """Una excepción a un fichero que ya no existe es una puerta abierta a un nombre libre:
    el día que alguien cree un `plan_display_i18n.py` nuevo hereda un permiso que nadie le
    dio."""
    faltan = [n for n in _LECTORES_PERMITIDOS if not (_BACKEND / n).exists()]
    assert not faltan, (
        f"Estos ficheros de `_LECTORES_PERMITIDOS` ya no existen: {faltan}. Quítalos, o el "
        f"permiso queda esperando a que alguien reutilice el nombre. [{_MARKER}]"
    )


def test_borrar_no_cuenta_como_leer():
    """`pop`/`del` son invalidación y valen en cualquier módulo. Si el detector los
    contara, once sitios legítimos se pondrían rojos y el guard acabaría desactivado."""
    for borrado in (
        'meals[i].pop("_display", None)',
        "m.pop('_display', None)",
        'del meal["_display"]',
    ):
        assert not _lecturas(borrado), f"el detector cuenta un borrado: {borrado}"


def test_el_detector_no_casa_con_los_hermanos_de_nombre():
    """La trampa que el propio gap anunciaba: `_display` ES subcadena de otras cosas."""
    for falso in (
        "_already_displayed_warnings = set()",
        'x = meta["_bad_display"]',
        'peso = plan["target_weight_display"]',
        'y = d.get("_display_completo")',
        'z = d["_display_link"]',
    ):
        assert not _lecturas(falso), f"falso positivo: {falso}"

    for real in (
        'meal["_display"]',
        "meal['_display']",
        'meal.get("_display")',
        'd.setdefault("_display", {})',
    ):
        assert _lecturas(real), f"falso negativo: {real}"
