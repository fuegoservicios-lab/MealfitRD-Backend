# -*- coding: utf-8 -*-
"""[P1-AUTH-CUENTA-BORRADA · 2026-09-08] Una cuenta borrada volvía entera.

## El defecto, medido en producción

El 08-sep se purgaron 7 cuentas de prueba: por cada una se borró su fila de
`user_profiles`, su `neon_auth."user"` y, en cascada, su `account` y sus `session`
(29 en el caso de esta). Horas después **una de esas cuentas generó un plan de 30
días**.

`verify_neon_jwt` valida la FIRMA contra el JWKS y nada más. Nada comprueba que el
usuario siga existiendo, así que un access token sin expirar sigue autenticando
después de borrar la cuenta — y `ensure_user_profile_exists` recrea la fila espejo
en ese mismo request. Borrar `neon_auth.session` no revoca el token ya emitido.

La huella del zombi es `email = NULL` + `full_name = NULL` con `created_at` de hoy:
el JWT no trae esos claims, así que el perfil renace vacío.

## Lo que hace este fix interesante

`P1-ACCOUNT-DELETE-IDENTITY` (2026-08-22) **ya razonó este modo de fallo**, y su
comentario en `delete_account_data` sigue ahí palabra por palabra: «mejor un perfil
sin identidad (inaccesible) que una identidad sin perfil (entra y
`ensure_user_profile_exists` lo resucita)». El orden del borrado se eligió para
apoyarse en una defensa —«un perfil sin identidad es inaccesible»— **que nunca se
implementó**. El diseño ya contaba con este guard antes de que existiera.

## La consecuencia que importa no es la fila zombi

El plan salió con `update_reason = "renewal.v1"`: el frontend reenvió el perfil de
salud entero desde localStorage (`previous_meals`, `reflection_history`,
`rejection_patterns`, `emergency_backup_plan`, historiales de adherencia y un
`grocery_cycle` de nueve días antes). La cuenta no volvió coja: volvió con memoria,
y medir con ella el camino de alta habría medido otra cosa.
"""
import ast
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_AUTH_SRC = (_BACKEND / "auth.py").read_text(encoding="utf-8")
_PROFILES_SRC = (_BACKEND / "db_profiles.py").read_text(encoding="utf-8")


def _fn(src: str, nombre: str):
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == nombre:
            return n
    return None


# ------------------------------------------------------- el veredicto de la sonda
def test_no_saber_NO_es_un_veredicto():
    """Tres valores, y solo uno cierra la puerta.

    `None` significa «no se pudo comprobar» (pool ausente, error, uuid mal formado).
    Colapsarlo a `False` convertiría un hipo de la base en un 401 para toda la flota;
    colapsarlo a `True` sería no tener guard. Debe seguir siendo un tercer valor.
    """
    import auth

    for respuesta, esperado in ((True, "u1"), (None, "u1"), (False, None)):
        auth.auth_user_row_exists = lambda _uid, _r=respuesta: _r
        assert auth._uid_si_la_identidad_vive("u1", "test") == esperado, (
            f"con la sonda devolviendo {respuesta!r} el guard debía dar {esperado!r}")


def test_el_guard_falla_ABIERTO_si_la_sonda_revienta():
    """Una excepción tampoco es un veredicto: la firma ya se validó, así que el peor
    caso aceptable es la conducta previa al fix, nunca dejar fuera a todo el mundo."""
    import auth

    def _revienta(_uid):
        raise RuntimeError("pool caído")

    auth.auth_user_row_exists = _revienta
    assert auth._uid_si_la_identidad_vive("u1", "test") == "u1"


def test_el_knob_apaga_el_guard_sin_redeploy(monkeypatch):
    """Encender un gate nuevo en la ÚNICA capa de auth sin vía de escape es apostar
    la app entera a que la consulta nunca se equivoca."""
    import auth

    auth.auth_user_row_exists = lambda _uid: False
    monkeypatch.setenv("MEALFIT_AUTH_REQUIRE_AUTH_ROW", "0")
    assert auth._uid_si_la_identidad_vive("u1", "test") == "u1", "el knob no apaga nada"
    monkeypatch.setenv("MEALFIT_AUTH_REQUIRE_AUTH_ROW", "1")
    assert auth._uid_si_la_identidad_vive("u1", "test") is None


# ------------------------------------------------------------------- la sonda
def test_la_sonda_pregunta_por_la_IDENTIDAD_no_por_el_perfil():
    """El perfil se resucita solo; la identidad es lo que la purga borra de verdad.

    Preguntar por `user_profiles` daría siempre `True` (el `ensure` la acaba de
    crear) — sería un guard que no puede fallar, que es la peor clase de guard.
    """
    fn = _fn(_PROFILES_SRC, "auth_user_row_exists")
    assert fn is not None, "auth_user_row_exists desapareció"
    doc = ast.get_docstring(fn, clean=False)
    sql = " ".join(n.value for n in ast.walk(fn)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)
                   and n.value != doc)
    assert 'neon_auth."user"' in sql, "la sonda no consulta la tabla de identidad"
    assert "user_profiles" not in sql, (
        "consulta el PERFIL, que el propio `ensure` acaba de crear: un guard que "
        "siempre aprueba")


def test_solo_se_cachea_el_POSITIVO():
    """Cachear el negativo dejaría fuera durante toda la vida del proceso a un alta
    que corriera contra la ventana entre el INSERT de Neon Auth y esta consulta."""
    import db_profiles

    db_profiles._AUTH_ROW_ALIVE_IDS.clear()
    llamadas = []

    def _falso(query, params=None, **kw):
        llamadas.append(params)
        return [] if params[0] == "muerto" else [{"vive": 1}]

    original = db_profiles.execute_sql_query
    pool_original = db_profiles.connection_pool
    db_profiles.execute_sql_query = _falso
    db_profiles.connection_pool = object()
    try:
        assert db_profiles.auth_user_row_exists("vivo") is True
        assert db_profiles.auth_user_row_exists("vivo") is True
        assert len(llamadas) == 1, "el positivo no se cacheó: una consulta por request"

        assert db_profiles.auth_user_row_exists("muerto") is False
        assert db_profiles.auth_user_row_exists("muerto") is False
        assert len(llamadas) == 3, (
            "el negativo entró al cache: un alta que corra contra la ventana de "
            "creación quedaría fuera para siempre en ese proceso")
        assert "muerto" not in db_profiles._AUTH_ROW_ALIVE_IDS
    finally:
        db_profiles.execute_sql_query = original
        db_profiles.connection_pool = pool_original
        db_profiles._AUTH_ROW_ALIVE_IDS.clear()


def test_sin_pool_no_inventa_un_veredicto():
    import db_profiles

    db_profiles._AUTH_ROW_ALIVE_IDS.clear()
    pool_original = db_profiles.connection_pool
    db_profiles.connection_pool = None
    try:
        assert db_profiles.auth_user_row_exists("quien-sea") is None
    finally:
        db_profiles.connection_pool = pool_original


# ------------------------------------------------- los CUATRO caminos, no tres
@pytest.mark.parametrize("via", ["bearer", "cookie", "header", "bearer_only"])
def test_las_cuatro_puertas_pasan_por_el_MISMO_guard(via):
    """Bearer, cookie `__Host-mf_session`, header `X-MF-Session` y el Bearer-only de
    `/api/auth/session`. Escribir la comprobación en cada una es la lección de
    `P1-DIET-CANON-SSOT`: tres tablas a mano drifean y la que se olvida es la que
    importa — aquí sería el PWA de iOS, que va por el header."""
    assert f'_uid_si_la_identidad_vive, uid, "{via}"' in _AUTH_SRC or \
           f'_uid_si_la_identidad_vive(uid, "{via}")' in _AUTH_SRC, (
        f"la puerta {via!r} devuelve identidad sin pasar por el guard")


def test_ninguna_puerta_devuelve_uid_a_pelo():
    """Ratchet: si mañana aparece un quinto camino, este test cae antes que producción.

    La primera versión de este test admitía `return uid` en una función «porque el
    guard ya rechazó más arriba» — y eso mide la BUENA VOLUNTAD del que edite
    después, no el contrato: un camino nuevo que devolviera `uid` sin pasar por
    arriba habría entrado en verde.

    Lo que se exige ahora es estructural: o el `return` invoca el guard, o el nombre
    `uid` fue **rebindeado** a su salida en esa misma función. Devolver `uid` es
    entonces devolver, por construcción, lo que el guard dejó pasar.
    """
    for nombre in ("get_verified_user_id", "get_neon_bearer_user_id"):
        fn = _fn(_AUTH_SRC, nombre)
        assert fn is not None, f"{nombre} desapareció"
        rebindeado = any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "uid" for t in n.targets)
            and "_uid_si_la_identidad_vive" in ast.unparse(n.value)
            for n in ast.walk(fn))
        for nodo in ast.walk(fn):
            if not isinstance(nodo, ast.Return) or nodo.value is None:
                continue
            crudo = ast.unparse(nodo.value)
            if crudo == "None":
                continue
            if "_uid_si_la_identidad_vive" in crudo:
                continue
            assert crudo == "uid" and rebindeado, (
                f"{nombre} tiene un `return {crudo}` que no pasa por el guard ni "
                f"devuelve un `uid` rebindeado por él")


def test_el_guard_va_ANTES_de_resucitar_el_perfil():
    """El orden ES el arreglo.

    Después del `ensure`, el perfil ya está recreado cuando decidimos rechazar: la
    fila zombi queda en la base igual, y el 401 llega tarde. Es la misma lección que
    `P1-I18N-DEAD-VEREDICTO`, donde la reparación colocada después de la consulta
    dejaba la primera pasada informando de un candidato que acababa de resolver.
    """
    for nombre in ("get_verified_user_id", "get_neon_bearer_user_id"):
        fn = _fn(_AUTH_SRC, nombre)
        cuerpo = ast.unparse(fn)
        i_guard = cuerpo.find("_uid_si_la_identidad_vive")
        i_ensure = cuerpo.find("ensure_user_profile_exists")
        assert i_guard != -1 and i_ensure != -1, f"{nombre}: falta uno de los dos"
        assert i_guard < i_ensure, (
            f"{nombre}: el guard corre DESPUÉS del ensure — el perfil ya se resucitó "
            f"cuando se decide rechazar")


def test_el_guard_no_puede_CONCEDER_identidad():
    """Preserva `P0-AUDIT-1`: el guard solo puede quitar un `sub` ya verificado.

    Si alguna vez devolviera algo que no recibió, sería una vía para inyectar
    identidad en la única capa de auth del backend.
    """
    fn = _fn(_AUTH_SRC, "_uid_si_la_identidad_vive")
    assert fn is not None
    devueltos = {ast.unparse(n.value) for n in ast.walk(fn)
                 if isinstance(n, ast.Return) and n.value is not None}
    assert devueltos <= {"uid", "None"}, (
        f"el guard devuelve algo que no es ni el uid recibido ni None: {devueltos}")


def test_el_borrado_de_cuenta_sigue_tumbando_la_identidad():
    """El guard solo sirve si la purga sigue borrando `neon_auth."user"`; si alguien
    revirtiera aquel DELETE, este fix quedaría inerte sin que nada lo dijera."""
    assert re.search(r'DELETE FROM neon_auth\."user" WHERE id = %s', _PROFILES_SRC), (
        "`delete_account_data` ya no borra la identidad: el guard no tendría a qué "
        "agarrarse")
