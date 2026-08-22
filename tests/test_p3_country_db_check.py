"""[P3-COUNTRY-DB-CHECK · 2026-08-22] El país es el único de los tres ajustes de Configuración sin
defensa en la base de datos — y el único cuyo valor nadie valida al escribirlo.

EL CONTRASTE QUE LO DELATA. `locale` nació el mismo mes, en la misma tabla, y tiene las cuatro
cosas: columna propia, `NOT NULL`, `DEFAULT` y `CHECK`. Además `PATCH /api/profile` valida su
**valor** y devuelve 400 ante un idioma desconocido. `country` no tiene ninguna de las cinco: vive
dentro del JSONB `health_profile`, y el PATCH lo mergea con `||` crudo sin mirar qué hay dentro.
El sistema de países entero —seis países, flip ejecutado el 2026-08-18— no tiene **ni una**
migración.

POR QUÉ IMPORTA, con el mecanismo exacto. `canonicalize_country` es fail-safe: cualquier cosa que
no sea uno de los seis códigos ISO cae a `'DO'`. Eso está bien como último recurso y mal como
única defensa, porque el fallo es **silencioso para el usuario**: quien acabe con `'España'` o
`'Marte'` escrito ahí recibe planes dominicanos indefinidamente y no hay nada en su pantalla que
se lo diga. `P2-COUNTRY-FAILSAFE-LOUD` ya le puso un `logger.warning` a ese momento — pero un log
avisa al operador DESPUÉS, y no impide la escritura.

Es la misma lección que la migración de `locale` escribió en su propio comentario: *la invariante
vive en la DB, no sólo en el camino que hoy resulta que la escribe*. Hoy el único escritor es el
selector de Configuración, que manda códigos canónicos. Mañana es un script de soporte, un
backfill, o el `update_form_field` del chat — que ya fue el tercer setter sin jerarquía
(`P2-TOOLS-FIELD-WHITELIST`) y que el prompt ordena llamar «OBLIGATORIO y SIN EXCEPCIÓN» ante
cualquier dato personal nuevo. Un «me mudé a España» escribiría `country='España'`.

MEDIDO EN PRODUCCIÓN HOY (SELECT read-only, 16 perfiles):

    health_profile->>'country'    NULL: 15   'DO': 1      → cero filas fuera de la lista
    jsonb_typeof(...->'country')  string: 1                → ningún valor de otro tipo
    columna locale (con CHECK)    'es-DO': 16

O sea: la migración aplica limpia y **la ausencia sigue siendo el caso normal** (15 de 16). Por eso
el CHECK admite NULL explícitamente — el fail-safe por ausencia es legítimo y no se toca. Lo que
se cierra es el otro caso: un string que no canoniza.

LO QUE ESTE P-FIX NO HACE, a propósito:

- **No promueve `country` a columna.** Sería el arreglo «completo» y es un proyecto aparte: hay
  lectores del JSONB por todo el backend y el frontend, y la ganancia sobre un CHECK en la
  expresión jsonb es marginal. El patrón «CHECK sobre expresión jsonb» ya es el de I8.
- **No canoniza en el endpoint.** Rechaza con 400. Coerción silenciosa es exactamente el «default
  sembrado» que este repo ya pagó: un valor que el sistema eligió por ti es indistinguible de uno
  que elegiste tú.
- **No se aplica a producción desde aquí.** La migración se escribe en los DOS directorios
  (P3-MIGRATIONS-SSOT) y la aplica el dueño en su ventana de deploy.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent

_NOMBRE = "p3_country_db_check_2026_08_22.sql"
_MIG_BACKEND = _BACKEND / "migrations" / _NOMBRE
_MIG_ROOT = _ROOT / "migrations" / _NOMBRE


def _sql() -> str:
    if not _MIG_BACKEND.is_file():
        pytest.fail(f"falta la migración {_NOMBRE} en backend/migrations/")
    return _MIG_BACKEND.read_text(encoding="utf-8", errors="replace")


# ── La migración ────────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sql() -> str:
    """El fichero COMPLETO. Para todo lo que sea una afirmación sobre el SQL, usa `codigo`."""
    return _sql()


@pytest.fixture(scope="module")
def codigo(sql) -> str:
    """El SQL sin comentarios.

    ⚠️ La mutación destapó por qué hace falta: el ancla del sanity comprobaba `"RAISE EXCEPTION"
    in sql` y la cabecera del propio fichero contiene esa frase explicando lo que hace. El guard
    se satisfacía con la PROSA — comentario-vence-guard, décima vez en la ola de agosto, y la
    cuarta seguida en la que el comentario culpable es mío."""
    return "\n".join(l for l in sql.splitlines() if not l.lstrip().startswith("--"))


@pytest.fixture(scope="module")
def bloque_check(codigo) -> str:
    """SÓLO la constraint. La paridad de países leía el primer `IN (...)` del fichero, que es el
    del bloque de sanity — así que una deriva en el CHECK de verdad pasaba inadvertida. Es el
    mismo «el test mira al sitio equivocado» que esta ola ya pagó en P2-14."""
    i = codigo.index("ADD CONSTRAINT user_profiles_country_supported")
    return codigo[i:codigo.index(";", i)]


def test_la_migracion_vive_en_los_dos_directorios(sql):
    """[P3-MIGRATIONS-SSOT] La raíz del workspace excluye `backend/` de su git, así que hacen falta
    ficheros FÍSICOS en ambos dirs o el `git push` de uno de los dos repos no se lleva el cambio.
    La verificación documentada compara sólo NOMBRES — por eso aquí se compara el CONTENIDO."""
    assert _MIG_ROOT.is_file(), f"falta {_NOMBRE} en migrations/ (raíz del workspace)"
    assert _MIG_ROOT.read_text(encoding="utf-8", errors="replace") == sql, (
        "las dos copias de la migración divergieron. El drift entre `migrations/` y "
        "`backend/migrations/` ya existió antes y la comprobación por nombres no puede verlo"
    )


def test_el_check_admite_la_ausencia(bloque_check):
    """15 de los 16 perfiles vivos NO tienen país, y eso es legítimo: `canonicalize_country` cae a
    'DO' por ausencia a propósito. Un CHECK que no admitiera NULL rechazaría a casi toda la base
    instalada."""
    assert re.search(r"IS NULL", bloque_check), (
        "el CHECK no admite la ausencia de país. 15 de 16 perfiles vivos están en ese caso"
    )


def test_los_paises_del_check_son_exactamente_los_del_ssot(bloque_check):
    """Paridad con `COUNTRY_PROFILES`, que es el SSOT. Un séptimo país añadido al selector sin
    tocar el CHECK dejaría a sus usuarios sin poder guardarlo — y fallaría en la base de datos,
    que es el peor sitio para enterarse."""
    import constants

    m = re.search(r"IN\s*\(([^)]*'DO'[^)]*)\)", bloque_check)
    assert m, "no encuentro la lista de países del CHECK"
    declarados = set(re.findall(r"'([A-Z]{2})'", m.group(1)))
    assert declarados == set(constants.COUNTRY_PROFILES), (
        f"el CHECK declara {sorted(declarados)} y COUNTRY_PROFILES tiene "
        f"{sorted(constants.COUNTRY_PROFILES)}"
    )


def test_la_migracion_es_idempotente(codigo):
    """[P3-MIGRATION-IDEMPOTENCE-DOC] `DROP CONSTRAINT IF EXISTS` antes de `ADD`, o una
    re-aplicación revienta."""
    i_drop = codigo.find("DROP CONSTRAINT IF EXISTS")
    i_add = codigo.find("ADD CONSTRAINT")
    assert i_drop > 0 and i_add > i_drop, (
        "la migración no hace DROP CONSTRAINT IF EXISTS antes del ADD: no es re-aplicable"
    )


def test_la_migracion_comprueba_los_datos_antes_de_imponer(codigo):
    """El sanity `DO $$ RAISE EXCEPTION`. Hoy sale limpio (0 filas fuera), pero una re-aplicación
    sobre datos futuros no tiene por qué — y este repo ya vio un dry-run abortar contra 12 filas
    preexistentes y CORRECTAS. Mejor que aborte diciendo cuántas y por qué."""
    assert "RAISE EXCEPTION" in codigo and "DO $$" in codigo, (
        "la migración impone el CHECK sin comprobar antes qué hay en los datos"
    )
    i_sanity = codigo.index("RAISE EXCEPTION")
    i_add = codigo.index("ADD CONSTRAINT")
    assert i_sanity < i_add, "el sanity va DESPUÉS del ADD: no puede avisar de nada"


def test_no_promete_haberse_aplicado(sql):
    """Una migración escrita no es una migración aplicada. El fichero tiene que decirlo, porque el
    modo de fallo de este repo es justo el contrario: dar por hecho lo cableado."""
    assert re.search(r"NO se ha aplicado|pendiente de aplicar|aplica el due", sql, re.I), (
        "la migración no declara que está pendiente de aplicar en producción"
    )


# ── El endpoint ─────────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def user_data_src() -> str:
    return (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8", errors="replace")


def _patch(monkeypatch, health_profile, uid="user-p3-country"):
    """Ejecuta el PATCH real con el UPDATE mockeado a éxito (mismo patrón que el arnés del
    trigger 4 de `test_p1_plan_display_i18n.py`)."""
    import db as db_module
    from routers.user_data import ProfilePatchBody, api_patch_profile

    monkeypatch.setattr(db_module, "execute_sql_write", lambda *a, **kw: [{"id": uid}])
    monkeypatch.setattr(db_module, "execute_sql_query", lambda *a, **kw: None)
    body = ProfilePatchBody(health_profile=health_profile)
    return asyncio.run(api_patch_profile(body=body, verified_user_id=uid))


def test_un_pais_canonico_se_acepta(monkeypatch):
    """El caso normal: el selector manda códigos ISO y tiene que seguir funcionando."""
    _patch(monkeypatch, {"country": "ES"})


def test_el_pais_dominicano_se_acepta(monkeypatch):
    """Byte-identidad: el usuario de RD no puede notar que este guard existe."""
    _patch(monkeypatch, {"country": "DO"})


def test_un_pais_desconocido_se_rechaza_con_400(monkeypatch):
    """EL CASO. Antes: `||` crudo, el valor se persistía, y `canonicalize_country` lo convertía en
    'DO' en cada lectura — plan dominicano indefinido, sin nada en pantalla que lo dijera."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _patch(monkeypatch, {"country": "España"})
    assert exc.value.status_code == 400
    assert "country" in str(exc.value.detail).lower()


def test_el_rechazo_nombra_los_paises_validos(monkeypatch):
    """Un 400 que no dice qué se esperaba obliga a leer el código para arreglarlo. El de `locale`
    ya lista los permitidos; este hace lo mismo."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _patch(monkeypatch, {"country": "Marte"})
    detalle = str(exc.value.detail)
    assert "DO" in detalle and "ES" in detalle


def test_no_canoniza_en_silencio(monkeypatch):
    """La decisión de diseño, anclada: rechazar, NO coercer. Un país que el sistema eligió por ti
    es indistinguible de uno que elegiste tú — el «default sembrado» que este repo ya pagó."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        # 'es' en minúscula NO es el código: es el idioma. Coercerlo a 'ES' sería adivinar.
        _patch(monkeypatch, {"country": "espana"})


def test_el_resto_de_health_profile_sigue_pasando_sin_mirarse(monkeypatch):
    """El alcance. `health_profile` es texto libre por diseño (super_personalization, perfil
    clínico); este P-fix valida UNA clave, no convierte el merge en un whitelist. Ensancharlo
    aquí rompería tres superficies que no tienen nada que ver."""
    _patch(monkeypatch, {"allergies": ["maní"], "notas": "lo que sea"})


def test_una_peticion_sin_country_no_toca_el_guard(monkeypatch):
    """Ausencia ≠ valor inválido. Es la distinción que `P2-COUNTRY-FAILSAFE-LOUD` ya hizo para el
    log, y tiene que valer igual aquí."""
    _patch(monkeypatch, {"country": None})
