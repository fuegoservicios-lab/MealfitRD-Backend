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
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_MARKER = "P3-I18N-BACKEND-SIN-BLANKET-DE-DISPLAY"

_CLAVE = "_display"
_re_sql = re.compile(r"\b(SELECT|WHERE|UPDATE|jsonb_set|jsonb_array_elements)\b", re.I)

# Quien puede LEERLO, y por que. La razon es parte del contrato: una excepcion sin razon es
# una puerta que nadie sabe si sigue haciendo falta.
_LECTORES_PERMITIDOS = {
    "plan_display_i18n.py":
        "es el modulo SSOT: lo escribe, y lo relee para no re-pagar una traduccion que ya "
        "esta (`_ya_traducido_*`, P2-DISPLAY-REDESPACHO-SIN-FILTRO)",
    "routers/plans.py":
        "SIRVE los nombres traducidos al cliente en el preview del Historial "
        "(`display_names`). Es pasar el dato a quien lo pinta, no decidir con el",
    # [P3-I18N-BLANKET-DISPLAY-POP-CON-USO · 2026-08-23] Lo destapó contar el `pop` con
    # uso: `meal.pop("_display", None) is not None` en `_invalidar_display_de_comidas`
    # cuenta CUÁNTAS invalidó para la telemetría. Usa el valor, sí — para saber si había
    # algo que borrar, no para decidir con lo que decía.
    "db_plans.py":
        "cuenta las invalidaciones (`popeados`) para telemetria; no lee el contenido",
}

_IGNORAR_DIRS = {"tests", "scripts", "docs", "migrations", "__pycache__"}


def _es_venv(parte: str) -> bool:
    """[P3-I18N-BLANKET-DISPLAY-POP-CON-USO · 2026-08-23] `test_venv` estaba en la lista y
    `venv`/`venv-test`/`.venv` no: el blanket escaneaba 6.002 ficheros de terceros que no
    despliega nadie. Cualquier directorio con «venv» en el nombre, o oculto, queda fuera."""
    return "venv" in parte.lower() or parte.startswith(".")


def _ficheros():
    for p in sorted(_BACKEND.rglob("*.py")):
        rel = p.relative_to(_BACKEND)
        if any(parte in _IGNORAR_DIRS or _es_venv(parte) for parte in rel.parts[:-1]):
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

    # [P3-I18N-BLANKET-DISPLAY-POP-CON-USO · 2026-08-23] Un `pop` es borrado SOLO si su
    # valor se tira: `m.pop("_display", None)` como sentencia suelta. `x = m.pop(...)`,
    # `if m.pop(...) is not None`, `f(m.pop(...))` USAN el valor — eso es leer.
    pops_sueltos = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Expr) and isinstance(nodo.value, ast.Call):
            pops_sueltos.add((nodo.value.lineno, nodo.value.col_offset))

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
        elif (isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Attribute)
              and nodo.func.attr == "pop" and nodo.args and _es_clave(nodo.args[0])
              and (nodo.lineno, nodo.col_offset) not in pops_sueltos):
            lineas.append(nodo.lineno)
        # [P3-I18N-DISPLAY-BLANKET-CIEGO-AL-SQL · 2026-08-23] Una lectura por SQL:
        # `->'_display'`, `->>'_display'` o `#>'{…,_display,…}'` dentro de una cadena. El
        # AST no ve claves ahí, y ya había una (`routers/user_data.py`, el disparador 4)
        # que decidía con ella. La cadena tiene que parecer SQL (SELECT/WHERE/jsonb_set)
        # para que la prosa que explica esta regla no la dispare.
        elif isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
            v = nodo.value
            if (("->'_display'" in v or "->>'_display'" in v or "_display,'" in v
                 or ",_display" in v or "{_display" in v)
                    and _re_sql.search(v)):
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


# ---------------------------------------------------------------------------
# [P3-I18N-DISPLAY-BLANKET-CIEGO-AL-SQL + P3-I18N-BLANKET-DISPLAY-POP-CON-USO · 2026-08-23]
# ---------------------------------------------------------------------------

def test_la_lectura_por_sql_cuenta():
    """Una consulta que proyecta `_display` para decidir con ella es una lectura aunque el
    AST no vea ninguna clave: vive dentro de una cadena. Era el caso real de
    `routers/user_data.py` (disparador 4), hoy movido al SSOT."""
    sql = (
        'row = q("""SELECT id, plan_data->\'days\'->0->\'meals\'->0->\'_display\' AS d '
        'FROM meal_plans WHERE user_id = %s""", (uid,))'
    )
    assert _lecturas(sql), "la lectura por SQL no se cuenta"
    assert _lecturas('q("SELECT plan_data->>\'_display\' FROM meal_plans WHERE id=%s")')
    # Prosa que CITA la forma no es SQL: no dispara.
    prosa = 'doc = "el blanket no veia ->\'_display\' dentro de una cadena"'
    assert not _lecturas(prosa), "la prosa que explica la regla la dispara (comentario-vence-guard)"


def test_el_pop_con_uso_cuenta_y_el_suelto_no():
    assert not _lecturas('m.pop("_display", None)'), "el pop suelto es borrado"
    for con_uso in (
        'x = m.pop("_display", None)',
        'if m.pop("_display", None) is not None:\n    n += 1',
        'f(m.pop("_display"))',
        'return m.pop("_display", {})',
    ):
        assert _lecturas(con_uso), f"el pop con uso no se cuenta: {con_uso!r}"


def test_no_se_escanea_ningun_venv():
    """Medido antes del cierre: 6.002 ficheros de `venv/` pasaban por el parser. Un blanket
    que escanea terceros mide lo que nadie despliega, y tarda lo que nadie espera."""
    rutas = [rel for _, rel in _ficheros()]
    assert rutas, "el blanket no encuentra ningún fichero"
    assert not [r for r in rutas if "venv" in r.lower() or r.startswith(".")], (
        "el blanket vuelve a entrar en un venv u oculto")
    assert len(rutas) < 400, f"{len(rutas)} ficheros: ¿volvió a entrar algún árbol de terceros?"
