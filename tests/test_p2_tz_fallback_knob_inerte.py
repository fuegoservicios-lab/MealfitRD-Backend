"""[P2-TZ-FALLBACK-KNOB-INERTE · 2026-08-23] El knob del huso por defecto no llegaba a
`db_facts`, el consumidor mas grande.

`P3-TZ-FALLBACK-SSOT` creo `constants.DEFAULT_TZ_OFFSET_MIN` (knob
`MEALFIT_DEFAULT_TZ_OFFSET_MIN`) "por si la poblacion deja de ser dominicana" y conecto tres
consumidores. El cuarto -- `db_facts`, del que cuelgan el agente proactivo,
`get_daily_nudge_count`, `get_avg_meal_hour`, `_local_date_str_for_user` y
`consultar_dia_del_plan` -- se quedo con `240` cableado en SEIS sitios: los cuatro `return 240`
de `user_tz_offset_min` y, dentro de `get_consumed_meals_today`, un `timedelta(hours=4)` mas un
`240` (y, en su rama de excepcion, los bordes literales `T04:00:00Z`/`T03:59:59Z`).

Medido antes del arreglo, con `MEALFIT_DEFAULT_TZ_OFFSET_MIN=-60`:

    constants.DEFAULT_TZ_OFFSET_MIN = -60
    literales de retorno en user_tz_offset_min: ['240','240','240','240','900','-900']
    'DEFAULT_TZ_OFFSET_MIN' in inspect.getsource(user_tz_offset_min) = False

Hoy el default del knob COINCIDE con el literal, asi que no habia error de conducta: lo que
estaba roto era la PALANCA. Un operador que la moviera veia cambiar tres caminos y no el cuarto,
que es el que alimenta el nudge y el "hoy" del coach.

QUE VIGILA ESTE FICHERO
  1. La palanca, medida por CONDUCTA: con el knob movido, los cuatro fallbacks de
     `user_tz_offset_min` y los dos defaults de `get_consumed_meals_today` siguen el knob.
  2. Que no vuelva el literal, medido por `ast` -- no por texto. Un comentario que diga "240"
     o "hours=4" NO puede satisfacer estos asserts (leccion comentario-vence-guard de este
     repo): los enteros que `ast` ve son codigo, nunca prosa.

ZONA HORARIA DE LA MAQUINA: ningun assert de aqui depende de ella. Los instantes no se
construyen en hora local del autor; lo que se comprueba es la RELACION entre el offset inyectado
y la ventana derivada (la hora del borde, nunca la fecha), que es invariante al reloj del que
corre el test. Dos tests de este repo solo pasaban donde vivia su autor; este no puede.
"""

import ast
import io

import pytest

import constants
import db_facts


# ---------------------------------------------------------------------------
# Utilidades de parseo (ast: inmune a comentarios y a prosa de docstring)
# ---------------------------------------------------------------------------

def _func_node(nombre: str) -> ast.FunctionDef:
    src = io.open(db_facts.__file__, encoding="utf-8").read()
    arbol = ast.parse(src)
    for n in ast.walk(arbol):
        if isinstance(n, ast.FunctionDef) and n.name == nombre:
            return n
    raise AssertionError(
        f"`db_facts.{nombre}` no existe - renombrada? Este guard la sigue por nombre."
    )


def _enteros(nodo: ast.AST) -> list:
    return [
        n.value for n in ast.walk(nodo)
        if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool)
    ]


def _cadenas(nodo: ast.AST) -> list:
    return [
        n.value for n in ast.walk(nodo)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]


def _nombres_llamados(nodo: ast.AST) -> set:
    out = set()
    for n in ast.walk(nodo):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            out.add(n.func.id)
    return out


# ---------------------------------------------------------------------------
# 1. El helper existe y lee el SSOT (no una copia)
# ---------------------------------------------------------------------------

def test_helper_existe_y_lee_el_ssot_de_constants():
    """El fallback de este modulo sale de `constants.DEFAULT_TZ_OFFSET_MIN`, no de un literal.

    Se comprueba por `ast` sobre el cuerpo: un docstring que MENCIONE el knob no basta.
    """
    nodo = _func_node("_fallback_tz_offset_min")
    importa_el_ssot = any(
        isinstance(n, ast.ImportFrom)
        and n.module == "constants"
        and any(a.name == "DEFAULT_TZ_OFFSET_MIN" for a in n.names)
        for n in ast.walk(nodo)
    )
    assert importa_el_ssot, (
        "`_fallback_tz_offset_min` ya no importa `DEFAULT_TZ_OFFSET_MIN` de `constants`. "
        "Sin ese import el helper es una quinta tabla del mismo hecho y el knob vuelve a ser inerte."
    )


def test_helper_sigue_al_knob(monkeypatch):
    """CONDUCTA: mover el knob mueve el helper. Revertir a `return 240` pone esto en rojo."""
    for valor in (-60, 0, 330, 240):
        monkeypatch.setattr(constants, "DEFAULT_TZ_OFFSET_MIN", valor, raising=True)
        assert db_facts._fallback_tz_offset_min() == valor, (
            f"`_fallback_tz_offset_min()` no siguio al knob movido a {valor}: la palanca sigue inerte."
        )


def test_helper_no_cachea_el_valor(monkeypatch):
    """Cachearlo lo congelaria en el primer uso: el operador lo moveria y no pasaria nada."""
    monkeypatch.setattr(constants, "DEFAULT_TZ_OFFSET_MIN", -120, raising=True)
    primero = db_facts._fallback_tz_offset_min()
    monkeypatch.setattr(constants, "DEFAULT_TZ_OFFSET_MIN", 360, raising=True)
    segundo = db_facts._fallback_tz_offset_min()
    assert (primero, segundo) == (-120, 360), (
        f"el helper devolvio {primero!r} y luego {segundo!r}: esta cacheando el knob."
    )


def test_helper_degrada_a_rd_si_el_import_falla(monkeypatch):
    """Fail-safe: si `constants` no se puede leer, la conducta es la PREVIA (240), no una excepcion."""
    import builtins
    real_import = builtins.__import__

    def _revienta(name, *a, **k):
        if name == "constants":
            raise ImportError("simulado")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _revienta)
    assert db_facts._fallback_tz_offset_min() == 240


# ---------------------------------------------------------------------------
# 2. `user_tz_offset_min` -- los CUATRO fallbacks siguen el knob
# ---------------------------------------------------------------------------

@pytest.fixture
def knob_movido(monkeypatch):
    """-60 = Espana en invierno. Elegido a proposito: distinto de 240, negativo (cruza el signo)
    y dentro del clamp [-900, 900] de `user_tz_offset_min`, para que el clamp no enmascare nada."""
    monkeypatch.setattr(constants, "DEFAULT_TZ_OFFSET_MIN", -60, raising=True)
    return -60


@pytest.mark.parametrize(
    "escenario,fila,revienta",
    [
        ("sin fila de perfil", None, False),
        ("perfil con ambas claves ausentes", {"tz": None, "tz_legacy": None}, False),
        ("perfil con valores no numericos", {"tz": "manana", "tz_legacy": "tarde"}, False),
        ("la DB revienta", None, True),
    ],
)
def test_user_tz_offset_min_degrada_al_knob(monkeypatch, knob_movido, escenario, fila, revienta):
    """Los cuatro `return` de fallback. Revertir CUALQUIERA a `return 240` pone en rojo su fila."""
    monkeypatch.setattr(db_facts, "connection_pool", object(), raising=True)

    def _q(*a, **k):
        if revienta:
            raise RuntimeError("simulado")
        return fila

    monkeypatch.setattr(db_facts, "execute_sql_query", _q, raising=True)
    got = db_facts.user_tz_offset_min("u-1")
    assert got == knob_movido, (
        f"[{escenario}] `user_tz_offset_min` devolvio {got} en vez del knob {knob_movido}: "
        "ese fallback sigue cableado a 240."
    )


def test_user_tz_offset_min_sin_pool_tambien_sigue_al_knob(monkeypatch, knob_movido):
    monkeypatch.setattr(db_facts, "connection_pool", None, raising=True)
    assert db_facts.user_tz_offset_min("u-1") == knob_movido


def test_user_tz_offset_min_no_pisa_el_valor_real_del_perfil(monkeypatch, knob_movido):
    """El fallback es fallback: un perfil con huso propio gana al knob. Sin esto, "seguir al knob"
    podria cumplirse ignorando al usuario, que seria un defecto peor que el original."""
    monkeypatch.setattr(db_facts, "connection_pool", object(), raising=True)
    monkeypatch.setattr(
        db_facts, "execute_sql_query",
        lambda *a, **k: {"tz": "360", "tz_legacy": None}, raising=True,
    )
    assert db_facts.user_tz_offset_min("u-1") == 360


def test_user_tz_offset_min_sin_enteros_240_en_el_cuerpo():
    """`ast`, no texto: un comentario que diga 240 no puede satisfacer este assert."""
    nodo = _func_node("user_tz_offset_min")
    assert 240 not in _enteros(nodo), (
        "volvio un literal `240` a `user_tz_offset_min`. El fallback tiene que salir de "
        "`_fallback_tz_offset_min()` o el knob queda inerte otra vez."
    )
    assert "_fallback_tz_offset_min" in _nombres_llamados(nodo)


# ---------------------------------------------------------------------------
# 3. `get_consumed_meals_today` -- los defaults de fecha Y de huso
# ---------------------------------------------------------------------------

def _ventana_capturada(monkeypatch, **kwargs):
    """Invoca `get_consumed_meals_today` capturando `(start_str, end_str)` del SQL."""
    capturado = {}

    def _q(query, params, **k):
        capturado["start"] = params[1]
        capturado["end"] = params[2]
        return []

    monkeypatch.setattr(db_facts, "connection_pool", object(), raising=True)
    monkeypatch.setattr(db_facts, "execute_sql_query", _q, raising=True)
    db_facts.get_consumed_meals_today("u-1", **kwargs)
    assert capturado, "no se llego al SELECT: la funcion salio antes de construir la ventana."
    return capturado["start"], capturado["end"]


@pytest.mark.parametrize(
    "offset,hhmmss_inicio,hhmmss_fin",
    [
        (240, "04:00:00", "03:59:59"),   # RD: la conducta previa, byte-identica
        (-60, "23:00:00", "22:59:59"),   # Espana en invierno
        (0, "00:00:00", "23:59:59"),     # UTC / Atlantic-Canary en invierno: 0 es un offset legitimo
        (360, "06:00:00", "05:59:59"),   # Mexico central
    ],
)
def test_get_consumed_meals_today_default_sigue_al_knob(monkeypatch, offset, hhmmss_inicio, hhmmss_fin):
    """Sin `date_str` ni `tz_offset_mins`, la ventana del dia se abre a la medianoche LOCAL del knob.

    Se asserta la HORA del borde, jamas la fecha: asi el test es independiente del reloj y de la
    zona de la maquina que lo corre (`start_str` es un instante UTC absoluto; su hora del dia solo
    depende del offset inyectado).
    """
    monkeypatch.setattr(constants, "DEFAULT_TZ_OFFSET_MIN", offset, raising=True)
    start, end = _ventana_capturada(monkeypatch)
    assert start[11:19] == hhmmss_inicio, (
        f"con el knob en {offset} la ventana abre a {start[11:19]}Z y deberia abrir a {hhmmss_inicio}Z: "
        "el default de huso sigue cableado a 240."
    )
    assert end[11:19] == hhmmss_fin


def test_get_consumed_meals_today_fecha_y_huso_no_pueden_discrepar(monkeypatch):
    """La fecha del default y el huso del default salen del MISMO sitio.

    Antes eran `timedelta(hours=4)` y `240`: dos escrituras del mismo hecho. Con el knob en -60 la
    ventana tiene que ser un dia LOCAL entero (23:59:59 exactos de ancho) y su fecha local tiene que
    ser el "hoy" local de ese mismo offset; si la fecha se calculara con un offset y los bordes con
    otro, la ventana se desplaza un dia respecto de la fecha local.
    """
    from datetime import datetime, timedelta, timezone

    offset = -60
    monkeypatch.setattr(constants, "DEFAULT_TZ_OFFSET_MIN", offset, raising=True)
    start, end = _ventana_capturada(monkeypatch)
    d0 = datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ")
    d1 = datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ")
    assert d1 - d0 == timedelta(hours=23, minutes=59, seconds=59)
    local_hoy = (datetime.now(timezone.utc) - timedelta(minutes=offset)).date()
    assert (d0 - timedelta(minutes=offset)).date() == local_hoy


def test_get_consumed_meals_today_respeta_los_argumentos_explicitos(monkeypatch):
    """El knob no puede pisar lo que el caller SI mando (contrato de `P1-DIARY-TZ-DEFAULT-RD`),
    y `0` sigue siendo un offset legitimo aunque sea falsy."""
    monkeypatch.setattr(constants, "DEFAULT_TZ_OFFSET_MIN", 240, raising=True)
    start, end = _ventana_capturada(monkeypatch, date_str="2026-03-04", tz_offset_mins=0)
    assert start == "2026-03-04T00:00:00Z"
    assert end == "2026-03-04T23:59:59Z"


def test_get_consumed_meals_today_fallback_de_fecha_basura_sigue_al_knob(monkeypatch):
    """La rama de excepcion (date_str basura) tenia TRES escrituras del mismo hecho:
    `hours=4` para el dia y los bordes literales `T04:00:00Z` / `T03:59:59Z`."""
    monkeypatch.setattr(constants, "DEFAULT_TZ_OFFSET_MIN", -60, raising=True)
    start, end = _ventana_capturada(monkeypatch, date_str="no-soy-fecha")
    assert start[11:19] == "23:00:00", (
        f"la rama de excepcion abrio a {start[11:19]}Z: los bordes literales de RD siguen ahi."
    )
    assert end[11:19] == "22:59:59"


def test_get_consumed_meals_today_sin_literales_de_rd_en_el_cuerpo():
    """`ast`: ni el entero 240, ni `timedelta(hours=4)`, ni los bordes `T04:00:00Z`/`T03:59:59Z`."""
    nodo = _func_node("get_consumed_meals_today")
    assert 240 not in _enteros(nodo), "volvio el literal 240 al default de huso."
    for s in _cadenas(nodo):
        assert "T04:00:00Z" not in s and "T03:59:59Z" not in s, (
            f"volvieron los bordes literales de RD a la rama de excepcion: {s!r}"
        )
    horas_4 = [
        n for n in ast.walk(nodo)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "timedelta"
        and any(
            kw.arg == "hours" and isinstance(kw.value, ast.Constant) and kw.value.value == 4
            for kw in n.keywords
        )
    ]
    assert not horas_4, "volvio `timedelta(hours=4)`: el default de FECHA no sigue al knob."
    assert "_fallback_tz_offset_min" in _nombres_llamados(nodo)
