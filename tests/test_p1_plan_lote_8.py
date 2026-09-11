"""[P1-PLAN-LOTE-8 · 2026-09-11] Octavo lote del plan de pendientes: G52 · `P2-TIME-CONTEXT-SERVER-CLOCK`.

El ÚNICO bloque del prompt marcado «(OBLIGATORIO)» —fecha, día laboral vs fin de semana, Navidad, Cuaresma— calculaba
«hoy» con el reloj del SERVIDOR (`datetime.now()` sin huso). A las 21:00 de un viernes en Santo Domingo (01:00Z) ya
renderizaba «Hoy es Sábado … Es FIN DE SEMANA … meal prep dominical». No era un defecto de los países beta: pegaba a
DO y el error escala con |offset|.

Ahora `build_time_context(country, tz_offset_min)` recibe el offset del usuario (convención getTimezoneOffset,
+240 = UTC-4) desde el contexto compartido, resuelto por `constants.tz_offset_min_for_form_data`: `tzOffset` del
cliente o `tz_offset_minutes` del perfil; un 0 explícito es UTC (un dato, P1-1); sin dato, `DEFAULT_TZ_OFFSET_MIN`
(P3-TZ-FALLBACK-SSOT) y NUNCA el default por país (T5-F1: el país es identidad culinaria, no dónde vive el usuario).
Sin offset (None) el bloque es byte-idéntico al de antes: es el camino de los tests con reloj congelado.

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def pg():
    from prompts import plan_generator as _pg
    return _pg


class _SabadoUnaDeLaMadrugadaZ(datetime):
    """Sábado 12-sep-2026 01:00Z = viernes 21:00 en Santo Domingo (UTC-4) = sábado 03:00 en Madrid (UTC+2)."""
    _fijo = datetime(2026, 9, 12, 1, 0, 0, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls._fijo if tz is not None else cls._fijo.replace(tzinfo=None)


@pytest.fixture
def madrugada_z(monkeypatch, pg):
    monkeypatch.setattr(pg, "datetime", _SabadoUnaDeLaMadrugadaZ)


# ─────────────────────────── el bloque lee el reloj del usuario ───────────────────────────

def test_g52_viernes_noche_en_santo_domingo_ya_no_es_sabado(pg, madrugada_z):
    do = pg.build_time_context(country="DO", tz_offset_min=240)
    assert "Hoy es Viernes, 11 de Septiembre de 2026" in do
    assert "DÍA LABORAL" in do and "FIN DE SEMANA" not in do


def test_g52_el_mismo_instante_en_madrid_si_es_sabado(pg, madrugada_z, monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    es = pg.build_time_context(country="ES", tz_offset_min=-120)
    assert "Hoy es Sábado, 12 de Septiembre de 2026" in es and "FIN DE SEMANA" in es
    # y un español que vive en Santo Domingo recibe SU viernes: decide el offset, no el país
    assert "Hoy es Viernes, 11 de Septiembre de 2026" in pg.build_time_context(country="ES", tz_offset_min=240)


def test_g52_utc_explicito_es_un_dato(pg, madrugada_z):
    assert "Hoy es Sábado, 12 de Septiembre de 2026" in pg.build_time_context(country="DO", tz_offset_min=0)


def test_g52_sin_offset_la_conducta_previa_es_byte_identica(pg, madrugada_z):
    """El camino None es el de los tests con reloj congelado y de los callers que sólo miden tamaño."""
    assert pg.build_time_context(country="DO") == pg.build_time_context(country="DO", tz_offset_min=None)
    assert "Sábado" in pg.build_time_context(country="DO"), "sin dato, el reloj del proceso (como siempre)"


def test_g52_el_mes_tambien_sigue_al_usuario(pg, monkeypatch):
    """1 de enero 02:00Z es todavía 31 de diciembre en Santo Domingo: Navidad, no «reset» de enero."""
    class _AnoNuevoZ(datetime):
        _fijo = datetime(2027, 1, 1, 2, 0, 0, tzinfo=timezone.utc)

        @classmethod
        def now(cls, tz=None):
            return cls._fijo if tz is not None else cls._fijo.replace(tzinfo=None)
    monkeypatch.setattr(pg, "datetime", _AnoNuevoZ)
    do = pg.build_time_context(country="DO", tz_offset_min=240)
    assert "31 de Diciembre de 2026" in do and "Época de Navidad" in do
    assert "post-Navidad" in pg.build_time_context(country="DO", tz_offset_min=0)


# ─────────────────────────── el resolvedor del offset ───────────────────────────

def test_g52_el_offset_del_formulario_manda_y_el_cero_es_un_dato():
    from constants import DEFAULT_TZ_OFFSET_MIN, tz_offset_min_for_form_data as f
    assert f({"tzOffset": 240}) == 240
    assert f({"tzOffset": "300"}) == 300 and f({"tzOffset": 300.0}) == 300
    assert f({"tzOffset": 0}) == 0, "UTC explícito es un dato, no una ausencia (P1-1)"
    assert f({"tz_offset_minutes": -60}) == -60
    assert f({"tzOffset": None, "tz_offset_minutes": 300}) == 300, "el perfil cubre al cliente que no lo mandó"
    assert f({"tzOffset": 240, "tz_offset_minutes": 300}) == 240, "el dato fresco del cliente gana"
    assert f({}) == DEFAULT_TZ_OFFSET_MIN and f(None) == DEFAULT_TZ_OFFSET_MIN and f("x") == DEFAULT_TZ_OFFSET_MIN
    assert f({"tzOffset": "garbage"}) == DEFAULT_TZ_OFFSET_MIN and f({"tzOffset": ""}) == DEFAULT_TZ_OFFSET_MIN
    assert f({"tzOffset": 99999}) == 720 and f({"tzOffset": -99999}) == -840, "clamp a un huso real"


def test_g52_nunca_el_default_por_pais():
    """T5-F1: `default_tz_offset_min` de COUNTRY_PROFILES sigue SIN lector, por diseño."""
    from constants import tz_offset_min_for_form_data as f
    assert f({"country": "ES"}) == f({"country": "DO"}) == f({})
    src = _src("constants.py")
    i = src.find("def tz_offset_min_for_form_data(")
    j = src.find("\ndef ", i + 10)
    cuerpo = src[i:j].split('"""')[-1]     # el cuerpo, sin el docstring que sí lo nombra para prohibirlo
    assert "default_tz_offset_min" not in cuerpo and "COUNTRY_PROFILES" not in cuerpo


def test_g52_el_contexto_compartido_pasa_el_offset():
    src = _src("graph_orchestrator.py")
    i = src.find('"time_context": build_time_context(')
    assert i > 0
    assert "tz_offset_min=_shared_ctx_tz" in src[i:i + 140]
    assert "_shared_ctx_tz = tz_offset_min_for_form_data(form_data)" in src


def test_marker_bumpeado():
    import app
    assert "[P1-PLAN-LOTE-8 · 2026-09-11]" in _src("app.py")
    assert app._LAST_KNOWN_PFIX.startswith("P1-PLAN-") and "2026-09-11" in app._LAST_KNOWN_PFIX
