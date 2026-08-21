"""[P1-TIME-CONTEXT-COUNTRY · 2026-08-21] El único bloque del prompt marcado «(OBLIGATORIO)» le
contaba el clima del Caribe a un usuario de Madrid.

`build_time_context()` no tenía parámetro de país y su render se inyecta —incondicionalmente— en
el contexto COMPARTIDO que alimenta al planner Y al day-generator. Lo que recibía un español el
20 de agosto de 2026:

    --- 📅 CONTEXTO ESTACIONAL Y CULTURAL (OBLIGATORIO) ---
    Hoy es Jueves, 20 de Agosto de 2026. Contexto en República Dominicana:
    - Temporada Caribeña: De Lluvia/Huracanes.
    - Clima: Hace MUCHO calor en el Caribe. Prioriza comidas más frescas, bowls, ensaladas…
    INSTRUCCIÓN: Adapta sutilmente la propuesta a este contexto…

Eso explica directamente el desayuno «Bowl Caribeño de Avena, Melón y Huevo» que el usuario
español recibió en el plan 6a4321f5, vivo en producción. Y no es sólo agosto: en octubre-noviembre
le habría sugerido «sancocho ligero» por la época de lluvias antillana, y en marzo-abril
«bacalao, tilapia, chillo» — chillo es pargo antillano.

QUÉ SE VA Y QUÉ SE QUEDA. La discriminación no es «dominicano sí / dominicano no», es
**RD-específico vs universal**:

  se va en beta   la línea «Contexto en República Dominicana», la temporada caribeña
                  (seca / de lluvias y huracanes), el hint de calor del Caribe y el de
                  sancocho — describen un clima que el usuario no tiene
  se queda        la fecha, día laboral vs fin de semana (universal), y los hints de
                  Navidad y de enero post-Navidad, que valen para los 6 países
  se queda pero
  se neutraliza   Cuaresma/Semana Santa: en España es más marcada que en RD y el bacalao es
                  SU plato de vigilia — lo que sobra es la lista de peces antillanos

Cubre:
  A. Byte-identidad dominicana (con el knob encendido y apagado).
  B. El país beta no lee el clima ni la temporada del Caribe.
  C. Lo universal sobrevive: fecha, laboral/fin de semana.
  D. El bloque sigue existiendo y sigue marcado OBLIGATORIO (no se vació de contenido).
  E. Los hints estacionales por mes, uno a uno, contra un reloj congelado.
  F. Parser-based: el call site del contexto compartido threadea el país.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GO_PATH = _BACKEND_ROOT / "graph_orchestrator.py"


@pytest.fixture(scope="module")
def pg():
    from prompts import plan_generator as _pg
    return _pg


@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


class _RelojCongelado(datetime):
    """Un reloj fijo: los hints dependen del MES, así que sin congelarlo el test aprueba o falla
    según el día en que se ejecute — la clase de flake que este repo ya pagó."""
    _fijo = datetime(2026, 8, 20, 12, 0, 0)

    @classmethod
    def now(cls, tz=None):
        return cls._fijo


@pytest.fixture
def en_agosto(monkeypatch, pg):
    monkeypatch.setattr(pg, "datetime", _RelojCongelado)


def _en_mes(monkeypatch, pg, mes):
    class _R(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, mes, 15, 12, 0, 0)
    monkeypatch.setattr(pg, "datetime", _R)


# ── A. Byte-identidad dominicana ────────────────────────────────────────────────────────────────

def test_do_es_identico_a_no_declarar_pais(pg, knob_on, en_agosto):
    """Los call sites de antes llamaban sin argumento: declarar 'DO' debe dar lo mismo."""
    assert pg.build_time_context(country="DO") == pg.build_time_context()


def test_el_pais_beta_cae_a_dominicano_con_el_knob_apagado(pg, monkeypatch, en_agosto):
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    assert pg.build_time_context(country="ES") == pg.build_time_context(country="DO")


# ── B. El Caribe no viaja a un país beta ────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", ["ES", "MX", "CO", "PR", "US"])
def test_el_pais_beta_no_lee_el_contexto_dominicano(pg, knob_on, en_agosto, cc):
    """RED pre-fix: los 5 recibían las tres líneas caribeñas."""
    bloque = pg.build_time_context(country=cc)
    assert "República Dominicana" not in bloque
    assert "Caribeña" not in bloque and "Caribe" not in bloque


def test_el_dominicano_conserva_su_contexto(pg, knob_on, en_agosto):
    """Control del anterior: en agosto, en RD, esas tres líneas son correctas."""
    bloque = pg.build_time_context(country="DO")
    assert "Contexto en República Dominicana" in bloque
    assert "Temporada Caribeña" in bloque
    assert "Caribe" in bloque


# ── C. Lo universal sobrevive ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", ["DO", "ES", "US"])
def test_la_fecha_y_el_tipo_de_dia_llegan_a_todos(pg, knob_on, en_agosto, cc):
    """Vaciar el bloque para beta habría sido el otro error: la fecha y saber si es día laboral
    gobiernan la complejidad de las recetas, y no tienen nada de dominicano."""
    bloque = pg.build_time_context(country=cc)
    assert "20 de Agosto de 2026" in bloque
    assert "DÍA LABORAL" in bloque  # 2026-08-20 es jueves


def test_el_bloque_beta_sigue_siendo_obligatorio_y_no_esta_vacio(pg, knob_on, en_agosto):
    """El bloque conserva su etiqueta y su instrucción final: se le quitó el Caribe, no el
    propósito."""
    bloque = pg.build_time_context(country="ES")
    assert "(OBLIGATORIO)" in bloque
    assert "INSTRUCCIÓN:" in bloque
    assert len(bloque.strip()) > 150


# ── D. Los hints por mes ────────────────────────────────────────────────────────────────────────

def test_en_noviembre_el_beta_no_recibe_el_sancocho(pg, knob_on, monkeypatch):
    """Época de lluvias antillana → «integrar algún caldo o sopa (ej. sancocho ligero)». El
    sancocho es el plato nacional dominicano y la lluvia es la del Caribe."""
    _en_mes(monkeypatch, pg, 11)
    assert "sancocho" not in pg.build_time_context(country="ES").lower()
    assert "sancocho" in pg.build_time_context(country="DO").lower()


def test_en_diciembre_la_navidad_llega_a_todos(pg, knob_on, monkeypatch):
    """La Navidad no es dominicana: los 6 países del sistema la celebran, y el hint (cenas
    pesadas fuera de casa → plan digestivo) vale igual en Madrid."""
    _en_mes(monkeypatch, pg, 12)
    for cc in ("DO", "ES", "MX"):
        assert "Navidad" in pg.build_time_context(country=cc)


def test_en_enero_el_reset_llega_a_todos(pg, knob_on, monkeypatch):
    _en_mes(monkeypatch, pg, 1)
    for cc in ("DO", "ES"):
        assert "post-Navidad" in pg.build_time_context(country=cc)


def test_en_semana_santa_el_beta_conserva_la_cuaresma_sin_los_peces_antillanos(pg, knob_on, monkeypatch):
    """En España la Cuaresma es MÁS marcada que en RD y el bacalao es su plato de vigilia: quitar
    el hint entero habría perdido una señal cultural real. Lo que sobra es «chillo» (pargo
    antillano) y la lista concreta de peces caribeños."""
    _en_mes(monkeypatch, pg, 4)
    es = pg.build_time_context(country="ES")
    assert "Cuaresma" in es, "la Cuaresma vale en España, no debía desaparecer"
    assert "chillo" not in es.lower(), "sigue nombrando pargo antillano"
    do = pg.build_time_context(country="DO")
    assert "chillo" in do.lower(), "el hint dominicano cambió"


# ── E. Parser-based ─────────────────────────────────────────────────────────────────────────────

def test_el_call_site_del_contexto_compartido_threadea_el_pais():
    """`_build_shared_context` ya deriva el país tres líneas más abajo (Fase 1 T3). Este guard
    impide que un refactor vuelva a llamar a `build_time_context()` sin él."""
    src = _GO_PATH.read_text(encoding="utf-8", errors="replace")
    i = src.find('"time_context": build_time_context(')
    assert i > 0, "el call site del contexto compartido desapareció o cambió de forma"
    assert "country" in src[i:i + 120], (
        "el contexto compartido volvió a construir el bloque temporal sin país"
    )


def test_el_fuente_declara_el_marker_y_la_puerta_unica(pg):
    src = (_BACKEND_ROOT / "prompts" / "plan_generator.py").read_text(encoding="utf-8", errors="replace")
    assert "P1-TIME-CONTEXT-COUNTRY" in src
    i = src.find("def build_time_context")
    assert "country_for_form_data" in src[i:i + 4000], (
        "el bloque temporal no deriva el país por la única puerta del motor"
    )
