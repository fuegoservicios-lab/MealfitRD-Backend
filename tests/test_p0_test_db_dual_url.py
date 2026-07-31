"""[P0-TEST-DB-DUAL-URL · 2026-07-31] El guard de aislamiento miraba una URL y el pool
usaba la otra.

EL CEPO
`db_core` inicializa el pool con `NEON_DATABASE_URL_POOLED`, pero
`_guard_test_write_to_prod` evaluaba solo `NEON_DATABASE_URL`. Una migración a medias
—apuntar la directa a un branch de test y olvidar la pooled, que es lo natural porque la
directa es la que se ve primero— hacía que el guard concluyera "esto es base de test,
las escrituras son seguras" mientras TODAS salían a producción por la pooled. Sin marker
`e2e`, sin escape hatch y sin aviso.

Lo peor: el cepo estaba justo en el camino que el propio docstring del guard recomienda
("deja el guard listo para cuando exista una base de test real"). Dar ese paso a medias
convertía el freno en acelerador.

EL CONTRATO
Las DOS URLs deben parecer no-producción. El desacuerdo NO se asume test: se trata como
producción y se explica la causa, porque una configuración a medias es peor que
cualquiera de los dos extremos — el operador cree que está aislado.

Estos tests EJECUTAN el guard con env vars fabricadas; ninguno toca la base.
"""
import pytest

import db_core


@pytest.fixture(autouse=True)
def _sin_escapes(monkeypatch):
    """Deja solo el eje que se está midiendo: bajo pytest, sin e2e, sin escape hatch."""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_p0_test_db_dual_url")
    monkeypatch.delenv("MEALFIT_ALLOW_TEST_WRITES_TO_PROD", raising=False)
    monkeypatch.setattr(db_core, "_CURRENT_TEST_IS_E2E", False)


PROD = "postgresql://u:p@ep-divine-sun-aiaszh9z.c-4.us-east-1.aws.neon.tech/neondb"
BRANCH = "postgresql://u:p@ep-otro-branch-xyz.c-4.us-east-1.aws.neon.tech/neondb_test"


def _set(monkeypatch, directa, pooled):
    monkeypatch.setenv("NEON_DATABASE_URL", directa)
    monkeypatch.setenv("NEON_DATABASE_URL_POOLED", pooled)


def test_el_cepo_exacto_directa_en_test_pooled_en_prod(monkeypatch):
    """El caso que motivó el fix: media migración.

    Pre-fix esto DEJABA PASAR la escritura (el guard solo miraba la directa) y todo
    iba a producción por la pooled.
    """
    _set(monkeypatch, BRANCH, PROD)
    with pytest.raises(RuntimeError) as exc:
        db_core._guard_test_write_to_prod("INSERT INTO meal_plans VALUES (1)")
    msg = str(exc.value)
    assert "P0-TEST-DB-DUAL-URL" in msg
    # El mensaje debe nombrar la causa REAL, no repetir el genérico: si solo dijera
    # "no existe base de test", el operador buscaría en el sitio equivocado.
    assert "POOLED" in msg.upper()
    assert "producción" in msg


def test_el_caso_simetrico_tambien_bloquea(monkeypatch):
    """Pooled en test y directa en prod. Menos probable, mismo veredicto."""
    _set(monkeypatch, PROD, BRANCH)
    with pytest.raises(RuntimeError, match="P0-TEST-DB-DUAL-URL"):
        db_core._guard_test_write_to_prod("UPDATE user_profiles SET email='x'")


def test_las_dos_en_el_branch_permite_escribir(monkeypatch):
    """Migración COMPLETA: el guard se aparta, que es todo el objetivo del branch."""
    _set(monkeypatch, BRANCH, BRANCH.replace("ep-otro-branch", "ep-otro-branch-pooler"))
    db_core._guard_test_write_to_prod("INSERT INTO meal_plans VALUES (1)")  # no lanza


def test_las_dos_en_produccion_sigue_bloqueando_como_siempre(monkeypatch):
    """Sin regresión del comportamiento original."""
    _set(monkeypatch, PROD, PROD)
    with pytest.raises(RuntimeError) as exc:
        db_core._guard_test_write_to_prod("DELETE FROM user_profiles")
    assert "P0-TEST-DB-ISOLATION" in str(exc.value)


def test_el_predicado_devuelve_el_motivo_del_desacuerdo(monkeypatch):
    _set(monkeypatch, BRANCH, PROD)
    es_nonprod, motivo = db_core._db_target_is_nonprod()
    assert es_nonprod is False
    assert motivo and "POOLED" in motivo.upper()

    _set(monkeypatch, PROD, PROD)
    assert db_core._db_target_is_nonprod() == (False, None)   # sin desacuerdo que narrar


def test_los_escapes_previos_siguen_funcionando(monkeypatch):
    """El fix endurece un eje; no debe cerrar las salidas que ya existían."""
    _set(monkeypatch, PROD, PROD)

    monkeypatch.setenv("MEALFIT_ALLOW_TEST_WRITES_TO_PROD", "1")
    db_core._guard_test_write_to_prod("INSERT INTO x VALUES (1)")  # escape hatch
    monkeypatch.delenv("MEALFIT_ALLOW_TEST_WRITES_TO_PROD")

    monkeypatch.setattr(db_core, "_CURRENT_TEST_IS_E2E", True)
    db_core._guard_test_write_to_prod("INSERT INTO x VALUES (1)")  # marker e2e
    monkeypatch.setattr(db_core, "_CURRENT_TEST_IS_E2E", False)

    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    db_core._guard_test_write_to_prod("INSERT INTO x VALUES (1)")  # fuera de pytest
