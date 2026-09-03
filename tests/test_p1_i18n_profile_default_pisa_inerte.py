"""[P1-I18N-PROFILE-DEFAULT-PISA-INERTE · 2026-08-22] La migracion se escribio, se
reviso, se commiteo en los dos espejos, sus 19 tests pasaban... y nunca corrio contra la
base. El arreglo llevaba un dia entero siendo INERTE.

MEDIDO contra Neon produccion el 2026-08-22, antes de aplicarla:

    SELECT is_nullable, column_default FROM information_schema.columns
     WHERE table_name='user_profiles' AND column_name='locale';
      ->  ('NO', "'es-DO'::text")

O sea: la columna seguia siendo `NOT NULL DEFAULT 'es-DO'`, con lo que la rama
`if (!data.locale)` de AssessmentContext.jsx era ESTRUCTURALMENTE INALCANZABLE -- el
perfil nunca podia traer NULL-- y la autodeteccion seguia muriendo en el primer login,
que es justo el defecto que P1-I18N-PROFILE-DEFAULT-PISA decia haber cerrado.

Y no era una migracion sola: `p1_country_system_f2_locale_comment.sql` tampoco se habia
aplicado. El COMMENT vivo de la columna seguia siendo el del 15-ago.

LA CAUSA ESTRUCTURAL, que es lo que este fichero existe para no dejar pasar otra vez:
**no hay libro de migraciones**. La unica tabla candidata en el esquema,
`checkpoint_migrations`, es de LangGraph (una sola columna `v`). Nada en la base registra
que `migrations/*.sql` se han aplicado, asi que dos pudieron quedarse fuera sin que nada
avisara -- y los tests que las anclan miden el ARCHIVO, no la base. Un guard que lee el
`.sql` no puede distinguir «escrita» de «aplicada», y esa es EXACTAMENTE la distincion que
costo el gap.

QUE ANCLA ESTE FICHERO:
  · el test de esquema VIVO (marcado e2e: necesita DB, no corre en el CI sin base), que
    es el unico que puede decir «aplicada»;
  · que el COMMENT que la migracion escribe ya no repita la afirmacion falsa
    («NO afecta al contenido generado»), corregida en P3-I18N-COMMENT-DB-ALCANCE-STALE;
  · que el test hermano parser-based DIGA en su docstring que no prueba produccion.

tooltip-anchor: P1-I18N-PROFILE-DEFAULT-PISA-INERTE
"""
from __future__ import annotations

import io
import os
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_MIGRACION = "migrations/p1_i18n_profile_locale_nullable_2026_08_21.sql"
_MARKER = "P1-I18N-PROFILE-DEFAULT-PISA-INERTE"


def _sql() -> str:
    return io.open(_ROOT / _MIGRACION, encoding="utf-8").read()


# ---------------------------------------------------------------------------
# 1 · El COMMENT que la migracion ESCRIBE tiene que ser verdad
# ---------------------------------------------------------------------------

def test_el_comment_no_repite_que_el_idioma_no_toca_el_contenido() -> None:
    """La version original decia «NO afecta al contenido generado». Falso desde el
    2026-08-17 (prosa del coach) y el 2026-08-19 (capa `_display`).

    Aplicarla tal cual habria metido en el esquema de produccion un comentario que ya
    sabiamos falso -- el mismo dano que documento `P2-I18N-DOC-ALCANCE-MIENTE`, esta vez
    en el sitio que lee quien abre la tabla.
    """
    sql = _sql()
    assert "NO afecta al contenido generado" not in sql, (
        "el COMMENT vuelve a afirmar que el locale no afecta al contenido generado. Lo "
        "afecta: gobierna la prosa del coach y dispara la capa `_display` que traduce "
        f"plan, recetas e insights. [{_MARKER}]"
    )


def test_el_comment_declara_las_tres_superficies_que_el_locale_gobierna() -> None:
    sql = _sql().lower()
    for pieza, porque in (
        ("coach", "la prosa del coach sigue el locale desde P1-COUNTRY-SYSTEM-F2 T3"),
        ("_display", "la capa que traduce plan/recetas/insights desde P1-PLAN-DISPLAY-I18N"),
    ):
        assert pieza in sql, f"el COMMENT no menciona {pieza}: {porque}. [{_MARKER}]"


def test_el_comment_preserva_la_frontera_dura() -> None:
    """Lo que el locale NO gobierna es tan load-bearing como lo que si."""
    sql = _sql()
    assert "pantry_names_match" in sql, (
        "el COMMENT no dice que los nombres de alimento siguen en espanol canonico por ser "
        f"el SSOT de pantry_names_match / coherencia / backstop de alergias. [{_MARKER}]"
    )


def test_las_dos_copias_siguen_identicas() -> None:
    """P3-MIGRATIONS-SSOT. Se compara NORMALIZANDO fin de linea: el repo tiene
    `core.autocrlf=true` y comparar bytes crudos pone el guard rojo por un \\r, que es la
    forma exacta de `P3-I18N-MIGRACION-ESPEJO-CRLF`."""
    a = io.open(_ROOT / _MIGRACION, encoding="utf-8").read().replace("\r\n", "\n")
    b = io.open(_BACKEND / "migrations" / Path(_MIGRACION).name, encoding="utf-8").read().replace("\r\n", "\n")
    assert a == b, f"los dos espejos de la migracion divergieron. [{_MARKER}]"


# ---------------------------------------------------------------------------
# 2 · El test hermano tiene que DECIR que no prueba produccion
# ---------------------------------------------------------------------------

def test_el_guard_parser_based_declara_su_limite() -> None:
    """Un guard que lee el `.sql` no puede distinguir «escrita» de «aplicada».

    Sin esa frase escrita, 19 tests en verde se leen como «el arreglo esta vivo» -- que es
    literalmente lo que paso durante un dia entero.
    """
    hermano = _BACKEND / "tests" / "test_p1_i18n_profile_default_pisa.py"
    txt = io.open(hermano, encoding="utf-8").read()
    # Se comprueba la PROPIEDAD «declara el limite», no una frase concreta: el aviso
    # tiene que nombrar las dos cosas que confundir costo el gap -- el fichero que SI
    # mide y la base que NO. Un `assert "no prueba" in txt` habria fallado por un
    # asterisco de enfasis en medio (medido: fallo asi la primera vez).
    cuerpo = re.sub(r"[^\wáéíóúñÁÉÍÓÚÑ\s]", " ", txt).lower()
    assert "archivo" in cuerpo and re.search(r"\bbase\b", cuerpo), (
        f"{hermano.name} no declara en ningun sitio que mide el ARCHIVO `.sql` y no la "
        f"BASE. Sin esa frase, 19 tests en verde se leen como «el arreglo esta vivo» -- "
        f"que es literalmente lo que paso durante 24 h. [{_MARKER}]"
    )
    assert re.search(r"\binerte\b|inalcanzable", cuerpo), (
        f"{hermano.name} no dice QUE se ve cuando la migracion no se aplica (el arreglo "
        f"queda inerte y la rama del frontend inalcanzable). [{_MARKER}]"
    )


# ---------------------------------------------------------------------------
# 3 · El esquema VIVO. El unico test que puede decir «aplicada».
# ---------------------------------------------------------------------------

@pytest.mark.e2e
def test_el_esquema_vivo_admite_no_he_elegido() -> None:
    """Necesita DB: sin ella se salta. Un skip NO es un verde -- es «no concluyente»,
    y el repo ya tiene registrado (`P1-CI-GATE-INCONCLUSIVE`) que eso no puede colapsar a
    ninguno de los dos lados. Por eso el mensaje del skip dice que hay que correrlo."""
    dsn = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        pytest.skip(
            "sin NEON_DATABASE_URL: NO se ha verificado que la migracion este aplicada. "
            "Correr con la DSN de produccion antes de dar el gap por cerrado."
        )
    psycopg = pytest.importorskip("psycopg")
    with psycopg.connect(dsn) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT is_nullable, column_default FROM information_schema.columns "
            "WHERE table_name='user_profiles' AND column_name='locale'"
        )
        fila = cur.fetchone()
        assert fila, f"no existe user_profiles.locale [{_MARKER}]"
        nullable, default = fila
        assert nullable == "YES", (
            "`user_profiles.locale` sigue siendo NOT NULL: el perfil no puede decir «no he "
            "elegido» y la autodeteccion muere en el primer login. La migracion NO esta "
            f"aplicada. [{_MARKER}]"
        )
        assert default is None, (
            f"`user_profiles.locale` conserva el DEFAULT {default!r}: un default sembrado es "
            f"indistinguible de una eleccion. La migracion NO esta aplicada. [{_MARKER}]"
        )

        cur.execute(
            "SELECT pg_get_constraintdef(con.oid) FROM pg_constraint con "
            "JOIN pg_class rel ON rel.oid = con.conrelid "
            "WHERE rel.relname = 'user_profiles' "
            "AND con.conname = 'user_profiles_locale_supported'"
        )
        chk = cur.fetchone()
        assert chk, f"falta el CHECK user_profiles_locale_supported [{_MARKER}]"
        assert "IS NULL" in chk[0], (
            f"el CHECK no admite NULL explicitamente: {chk[0]} [{_MARKER}]"
        )
