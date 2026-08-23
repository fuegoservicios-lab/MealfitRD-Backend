"""[P2-I18N-MIGRACIONES-SIN-LIBRO · 2026-08-23] Nada registraba qué migraciones se habían
aplicado a Neon: ``scripts/apply_migration.py`` ejecutaba y no dejaba rastro, y «¿está
aplicada?» era una auditoría a mano contra ``information_schema``. Medido con esa auditoría
el 2026-08-23: 110 ficheros y UNO sin aplicar en producción
(``p3_country_db_check_2026_08_22.sql``) sin que nada lo dijera.

Cierre: tabla ``public.schema_migrations`` (migración idempotente, en los DOS dirs —
P3-MIGRATIONS-SSOT), el runner anota al aplicar (``--apply``), anota sin ejecutar
(``--record``, el backfill de las 109 anteriores con la nota de CÓMO se verificó cada grupo)
y compara ficheros contra libro (``--status``: al día / PENDIENTE / aplicada con OTRO
contenido, exit ≠ 0 si hay trabajo). Aplicada a Neon y backfilled el 2026-08-23: 110 al
día, 1 pendiente.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_RUNNER = _BACKEND / "scripts" / "apply_migration.py"
_MIG = "p2_i18n_migraciones_sin_libro_2026_08_23.sql"
_MARKER = "P2-I18N-MIGRACIONES-SIN-LIBRO"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("apply_migration_under_test", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── la migración ───────────────────────────────────────────────────────────────────────────

def test_la_migracion_existe_en_los_dos_dirs_y_es_idempotente() -> None:
    backend_copy = _BACKEND / "migrations" / _MIG
    assert backend_copy.exists(), f"falta backend/migrations/{_MIG} [{_MARKER}]"
    src = backend_copy.read_text(encoding="utf-8")
    assert re.search(r"CREATE TABLE IF NOT EXISTS public\.schema_migrations", src), "CREATE sin IF NOT EXISTS"
    assert "DO $$" in src and "RAISE EXCEPTION" in src, "sin sanity check DO $$ (P3-MIGRATION-IDEMPOTENCE-DOC)"
    for col in ("name", "checksum", "applied_at"):
        assert re.search(rf"^\s*{col}\s", src, re.M), f"la tabla no declara `{col}`, que el runner usa"
    root_copy = _ROOT / "migrations" / _MIG
    if root_copy.exists():
        # Identidad de CONTENIDO: `core.autocrlf` deja una copia en CRLF y otra en LF según
        # qué árbol la tocó git; comparar bytes crudos pondría rojo un no-cambio.
        normal = lambda p: p.read_bytes().replace(b"\r\n", b"\n")
        assert normal(root_copy) == normal(backend_copy), (
            f"las dos copias de {_MIG} difieren (P3-MIGRATIONS-SSOT)")
    else:
        pytest.skip("migrations/ del workspace-root no está en este checkout (worktree)")


# ── clasificar(): la lógica de --status, pura ──────────────────────────────────────────────

def test_clasificar_separa_las_cuatro_situaciones(runner) -> None:
    ficheros = {"a.sql": "h1", "b.sql": "h2", "c.sql": "h3-nuevo"}
    libro = {"a.sql": "h1", "c.sql": "h3-viejo", "z.sql": "hz"}
    r = runner.clasificar(ficheros, libro)
    assert r == {
        "al_dia": ["a.sql"],
        "pendientes": ["b.sql"],
        "cambiadas": ["c.sql"],
        "solo_en_libro": ["z.sql"],
    }


def test_clasificar_con_libro_vacio_lo_marca_todo_pendiente(runner) -> None:
    r = runner.clasificar({"a.sql": "h", "b.sql": "h"}, {})
    assert r["pendientes"] == ["a.sql", "b.sql"] and not r["al_dia"]


def test_el_checksum_es_sha256_del_contenido(runner) -> None:
    import hashlib
    assert runner._checksum("x") == hashlib.sha256(b"x").hexdigest()
    assert runner._checksum("x") != runner._checksum("x ")


# ── el runner anota al aplicar (conducta, con un cursor falso) ─────────────────────────────

class _CursorFalso:
    def __init__(self, libro_existe=True):
        self.ejecutado = []
        self.libro_existe = libro_existe

    def execute(self, sql, params=None):
        self.ejecutado.append((sql, params))
        if "schema_migrations" in sql and not self.libro_existe:
            import psycopg
            raise psycopg.errors.UndefinedTable("relation does not exist")


def test_record_hace_upsert_por_nombre_con_checksum(runner) -> None:
    cur = _CursorFalso()
    assert runner._record(cur, "x.sql", "abc", "nota") is True
    sql, params = cur.ejecutado[-1]
    assert "INSERT INTO public.schema_migrations" in sql and "ON CONFLICT (name) DO UPDATE" in sql
    assert params[0] == "x.sql" and params[1] == "abc" and params[3] == "nota"


def test_sin_libro_avisa_y_no_revienta(runner, capsys) -> None:
    """La única razón aceptable para no anotar: el libro aún no existe. Reventar aquí
    dejaría la migración aplicada y al operador con un traceback en vez de la instrucción."""
    cur = _CursorFalso(libro_existe=False)
    assert runner._record(cur, "x.sql", "abc", None) is False
    assert _MIG in capsys.readouterr().out


def test_apply_ejecuta_el_sql_y_despues_anota(runner) -> None:
    """Guard de orden sobre el runner real: en `main`, `cur.execute(sql)` va ANTES de
    `_record(...)`. Al revés, un fichero que revienta quedaría anotado como aplicado."""
    src = _RUNNER.read_text(encoding="utf-8")
    cuerpo = src.split("def main()", 1)[1]
    i_exec = cuerpo.find("cur.execute(sql)")
    i_rec = cuerpo.find("_record(cur, name, checksum, note)")
    assert 0 < i_exec < i_rec, f"el runner anota antes de ejecutar, o no anota [{_MARKER}]"
