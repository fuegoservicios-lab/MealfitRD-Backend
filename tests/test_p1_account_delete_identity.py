"""[P1-ACCOUNT-DELETE-IDENTITY · 2026-08-22] «Eliminar cuenta» también borra la IDENTIDAD.

LO QUE PASABA
    `delete_account_data` borraba `user_profiles` y las tablas user-scoped, pero NO
    tocaba `neon_auth."user"`. Tras «Eliminar cuenta» desde Ajustes, la identidad
    seguía viva: el mismo correo volvía a entrar sin registrarse (código al correo o
    Google) y las sesiones abiertas en OTROS dispositivos seguían siendo válidas
    (`neon_auth.session` cuelga del user, no del perfil).

    Para Apple (guideline 5.1.1(v), borrado de cuenta dentro de la app) eso no es
    borrar la cuenta: es borrar los datos y dejar la puerta abierta. Medido el
    2026-08-22 antes del primer build iOS: 0 huérfanos en producción porque nadie
    había usado aún el botón — el hueco existía sin haber mordido.

EL FIX
    Cuando `include_profile=True` (el flujo del usuario), se borra también
    `neon_auth."user"`. `session` y `account` tienen ON DELETE CASCADE hacia `user`
    (medido en el esquema), así que con ese DELETE cae todo lo de identidad.

    Y SOLO con `include_profile=True`: el purge administrativo de datos
    (`include_profile=False`) debe dejar la cuenta viva — es «vaciar», no «cerrar».
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import db_profiles


class _FakeDB:
    def __init__(self):
        self.writes: list[tuple[str, tuple]] = []

    def execute_sql_write(self, query, params=None, returning=False, lock_timeout_ms=None):
        q = " ".join(str(query).split())
        self.writes.append((q, params))
        return [{"id": params[0]}] if returning else True

    def execute_sql_query(self, *a, **k):
        return [] if k.get("fetch_all") else None


@pytest.fixture
def db(monkeypatch):
    fake = _FakeDB()
    monkeypatch.setattr(db_profiles, "execute_sql_write", fake.execute_sql_write)
    monkeypatch.setattr(db_profiles, "execute_sql_query", fake.execute_sql_query, raising=False)
    monkeypatch.setattr(db_profiles, "connection_pool", object(), raising=False)
    monkeypatch.setattr(db_profiles, "_purge_visual_diary_storage", lambda uid: 0, raising=False)
    return fake


_UID = "11111111-2222-3333-4444-555555555555"


def _deletes_de(db, tabla_regex):
    import re
    # Sin `\b` al final: tras `"user"` la comilla no es carácter de palabra y `\b`
    # no casa (el primer RED de este fichero lo dejó en verde a medias por eso).
    return [q for q, _ in db.writes if re.search(rf"DELETE FROM {tabla_regex}(\s|$)", q)]


def test_eliminar_cuenta_borra_la_identidad_de_neon_auth(db):
    db_profiles.delete_account_data(_UID, include_profile=True)
    ident = _deletes_de(db, r'neon_auth\."user"')
    assert ident, (
        "Con include_profile=True debe ejecutarse `DELETE FROM neon_auth.\"user\"`: sin "
        "eso el correo vuelve a entrar sin registrarse y las sesiones de otros "
        "dispositivos siguen vivas (Apple 5.1.1(v))."
    )
    assert all(p and p[0] == _UID for q, p in db.writes if 'neon_auth."user"' in q), (
        "El DELETE de identidad debe ir filtrado por el user_id (I2)."
    )


def test_la_identidad_se_borra_DESPUES_del_perfil(db):
    """user_profiles referencia al user por id; borrar la identidad antes dejaría el
    perfil sin dueño si algo fallara entre medias. Orden: datos → perfil → identidad."""
    db_profiles.delete_account_data(_UID, include_profile=True)
    qs = [q for q, _ in db.writes]
    i_perfil = next(i for i, q in enumerate(qs) if "DELETE FROM user_profiles" in q)
    i_ident = next(i for i, q in enumerate(qs) if 'neon_auth."user"' in q)
    assert i_perfil < i_ident


def test_purge_de_datos_sin_perfil_NO_toca_la_identidad(db):
    """`include_profile=False` es «vaciar» (purge admin): la cuenta sigue viva."""
    db_profiles.delete_account_data(_UID, include_profile=False)
    assert not _deletes_de(db, r'neon_auth\."user"'), (
        "Un purge de datos (include_profile=False) NO debe borrar la identidad."
    )
    assert not _deletes_de(db, "user_profiles")


def test_el_resultado_informa_si_la_identidad_se_borro(db):
    r = db_profiles.delete_account_data(_UID, include_profile=True)
    assert r["deleted"].get("neon_auth_user") == 1, (
        f"El resultado debe contar la identidad borrada (para el log y la alerta): {r['deleted']}"
    )


def test_fallo_al_borrar_identidad_queda_en_errors_no_revienta(db, monkeypatch):
    """Best-effort como el resto de tablas: un fallo aquí se REGISTRA (errors) y el
    handler decide; no puede dejar a medias un borrado ya ejecutado."""
    orig = db.execute_sql_write

    def boom(query, params=None, returning=False, lock_timeout_ms=None):
        if 'neon_auth."user"' in " ".join(str(query).split()):
            raise RuntimeError("neon_auth no disponible")
        return orig(query, params, returning, lock_timeout_ms)

    monkeypatch.setattr(db_profiles, "execute_sql_write", boom)
    r = db_profiles.delete_account_data(_UID, include_profile=True)
    assert any("neon_auth" in e for e in r["errors"]), r["errors"]
