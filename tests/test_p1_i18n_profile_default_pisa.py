"""[P1-I18N-PROFILE-DEFAULT-PISA · 2026-08-21] La autodetección de idioma era inerte
para todo usuario con sesión, y el primer login la desactivaba PARA SIEMPRE.

LA TRAZA, verificada paso a paso:

  1. Visitante nuevo anglófono → `P1-AUTO-LOCALE` lo detecta a `en-US`. Correcto.
  2. Se registra. La fila de `user_profiles` nace con el DEFAULT `'es-DO'`.
  3. `fetchProfile` llama a `syncLocaleFromProfile('es-DO')`. Como `'es-DO' !== 'en-US'`,
     entra: carga el catálogo español **y hace `_persistLocal('es-DO')`**.
  4. El español queda escrito en `localStorage`. Y la autodetección no vuelve a
     dispararse jamás: `getStoredLocale()` encuentra un valor soportado y sale antes de
     consultarla («lo guardado gana sobre lo detectado, siempre»).

O sea: la feature desplegada el 2026-08-20 funcionaba **sólo para visitantes anónimos**, y
el primer login la apagaba de forma permanente. No fallaba: hacía exactamente lo que el
código decía, y lo que el código decía estaba mal.

LA CAUSA RAÍZ ES DE ESQUEMA, no de frontend: `locale text NOT NULL DEFAULT 'es-DO'`
significa que **el perfil no puede decir «no he elegido»**. «Español» y «todavía nada» son
el mismo valor, así que el cliente no tiene forma de distinguir una preferencia real de un
default sembrado — y un default sembrado es indistinguible de una elección, que es la
misma lección que dejó `P1-COUNTRY-RENEW-OVERWRITE` con el país.

LA REPRESENTACIÓN ELEGIDA: `NULL` = no elegido (opción (a) del plan). Se descarta la
columna hermana `locale_explicit` porque añade un estado más que mantener sincronizado, y
porque `NULL` ya degrada correcto sin tocar el motor: `syncLocaleFromProfile` empieza con
`if (!isSupportedLocale(profileLocale)) return false;`.

LOS CUATRO LECTORES DEL BACKEND, verificados uno a uno antes de tocar el DEFAULT:
  · `agent.py` ×2  → `.get("locale") or "es-DO"`
  · `proactive_agent.py` → `.get("locale") or "es-DO"`
  · `cron_tasks.py` → `if _p1_i18n_locale and _p1_i18n_locale != "es-DO"` (NULL es falsy,
    y no despachar traducción para un usuario sin idioma elegido es lo correcto)
  · `routers/user_data.py` → sólo valida en la ESCRITURA
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_MARKER = "P1-I18N-PROFILE-DEFAULT-PISA"

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent

_MIGRACIONES = ("migrations", "backend/migrations")
_NOMBRE = "p1_i18n_profile_locale_nullable_2026_08_21.sql"


def _ruta(dir_rel: str) -> Path:
    return _ROOT / dir_rel / _NOMBRE


def _sql(dir_rel: str) -> str:
    p = _ruta(dir_rel)
    if not p.exists():
        pytest.fail(
            f"Falta {p}. Sin `DROP DEFAULT`, la fila de un usuario nuevo nace en "
            f"'es-DO' y el primer login apaga la autodetección para siempre. "
            f"[{_MARKER}]"
        )
    return p.read_text(encoding="utf-8")


# ============================================================
# 1 · La migración, en los DOS directorios (P3-MIGRATIONS-SSOT)
# ============================================================

@pytest.mark.parametrize("dir_rel", _MIGRACIONES)
def test_la_migracion_existe_en_los_dos_directorios(dir_rel: str) -> None:
    _sql(dir_rel)


def test_las_dos_copias_son_identicas() -> None:
    """P3-MIGRATIONS-SSOT: el workspace-root y el repo backend tienen `.gitignore`
    distintos, así que cada uno necesita el fichero FÍSICO. Un drift entre las dos es una
    migración que se aplica en un entorno y no en el otro."""
    a, b = (_ruta(d) for d in _MIGRACIONES)
    if not (a.exists() and b.exists()):
        pytest.skip("una de las dos copias no está en este checkout")
    assert a.read_bytes() == b.read_bytes(), (
        f"las dos copias de {_NOMBRE} difieren. [{_MARKER}]"
    )


@pytest.mark.parametrize("dir_rel", _MIGRACIONES)
def test_quita_el_default_que_impedia_decir_no_he_elegido(dir_rel: str) -> None:
    sql = _sql(dir_rel).lower()
    assert "drop default" in sql, (
        f"la migración no quita el DEFAULT. Mientras exista, «español» y «todavía nada» "
        f"son el mismo valor y el cliente no puede distinguirlos. [{_MARKER}]"
    )
    assert "drop not null" in sql, (
        f"la migración no quita el `NOT NULL`. Sin eso, `NULL` no se puede escribir y "
        f"«no elegido» sigue sin ser representable. [{_MARKER}]"
    )


@pytest.mark.parametrize("dir_rel", _MIGRACIONES)
def test_el_check_sigue_acotando_los_valores_no_nulos(dir_rel: str) -> None:
    """LA MITAD QUE NO SE PUEDE PERDER. Relajar la columna no puede relajar QUÉ valores
    se aceptan: un `locale` fuera de la lista rompe el `import()` del catálogo."""
    sql = _sql(dir_rel)
    m = re.search(r"CHECK\s*\((.*?)\)\s*(?:NOT\s+VALID)?\s*;", sql, re.S | re.I)
    assert m, f"la migración no vuelve a declarar el CHECK. [{_MARKER}]"
    cuerpo = m.group(1)
    for code in ("es-DO", "en-US", "pt-BR", "fr-FR", "it-IT"):
        assert code in cuerpo, f"el CHECK perdió {code}. [{_MARKER}]"
    assert re.search(r"locale\s+IS\s+NULL", cuerpo, re.I), (
        f"el CHECK no admite `NULL` explícitamente. En Postgres un CHECK con `NULL` "
        f"evalúa a NULL y la fila pasa, así que en la práctica funcionaría — pero "
        f"dejarlo implícito hace que el siguiente lector no sepa si es deliberado. "
        f"[{_MARKER}]"
    )


@pytest.mark.parametrize("dir_rel", _MIGRACIONES)
def test_es_idempotente_y_tiene_sanity(dir_rel: str) -> None:
    """P3-MIGRATION-IDEMPOTENCE-DOC."""
    sql = _sql(dir_rel)
    assert "DROP CONSTRAINT IF EXISTS" in sql, (
        f"sin `DROP CONSTRAINT IF EXISTS` antes del ADD, re-aplicarla revienta. "
        f"[{_MARKER}]"
    )
    assert "RAISE EXCEPTION" in sql, (
        f"falta el bloque de sanity. [{_MARKER}]"
    )


# ============================================================
# 2 · Los lectores toleran el NULL
# ============================================================

@pytest.mark.parametrize(
    "rel,patron",
    [
        ("agent.py", r'\.get\("locale"\)\s*or\s*"es-DO"'),
        ("proactive_agent.py", r'\.get\("locale"\)\s*or\s*"es-DO"'),
        ("cron_tasks.py", r"if\s+_p1_i18n_locale\s+and\s+_p1_i18n_locale\s*!="),
    ],
)
def test_los_lectores_del_backend_toleran_null(rel: str, patron: str) -> None:
    """Se verificaron uno a uno ANTES de tocar el DEFAULT, y se anclan para que sigan
    así: quitar el `or "es-DO"` de cualquiera lo convierte en un `None` que viaja hasta
    un prompt o hasta un `import()`."""
    p = _BACKEND / rel
    if not p.exists():
        pytest.skip(f"{rel} no existe en este checkout")
    assert re.search(patron, p.read_text(encoding="utf-8")), (
        f"{rel} ya no protege contra `locale` NULL. Desde esta migración, un perfil que "
        f"nunca eligió idioma trae `NULL`. [{_MARKER}]"
    )


# ============================================================
# 3 · El frontend: el default sembrado ya no pisa la detección
# ============================================================

def _i18n() -> str:
    p = _ROOT / "frontend" / "src" / "i18n" / "index.js"
    if not (_ROOT / "backend").is_dir() or not p.exists():
        pytest.skip("frontend no disponible en este checkout (repos hermanos)")
    return p.read_text(encoding="utf-8")


def test_sync_desde_el_perfil_ignora_un_locale_no_soportado() -> None:
    """`NULL` llega al cliente como `null`, y ahí `isSupportedLocale` lo rechaza: el
    motor conserva el idioma DETECTADO y —lo que importa— no escribe `localStorage`.

    Ese `_persistLocal` era el que hacía el daño permanente: una vez escrito, la
    detección no vuelve a consultarse nunca."""
    src = _i18n()
    m = re.search(r"export async function syncLocaleFromProfile\([^)]*\)\s*\{(.*?)\n\}",
                  src, re.S)
    assert m, f"no encontré `syncLocaleFromProfile`. [{_MARKER}]"
    cuerpo = m.group(1)
    i_guard = cuerpo.find("isSupportedLocale(profileLocale)")
    i_persist = cuerpo.find("_persistLocal")
    assert i_guard != -1, (
        f"`syncLocaleFromProfile` ya no valida el locale del perfil: un `null` entraría "
        f"y el motor caería al español. [{_MARKER}]"
    )
    assert i_persist == -1 or i_guard < i_persist, (
        f"el guard de soporte tiene que estar ANTES del `_persistLocal`. Si se persiste "
        f"primero, un default sembrado queda escrito en el navegador y apaga la "
        f"autodetección para siempre. [{_MARKER}]"
    )
