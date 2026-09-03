"""[P1-COUNTRY-RENEWAL-PROFILE-WINS · 2026-08-18] Regresión del incidente del día del flip:
el país elegido en Configuración era pisado por la renovación del plan.

Cadena del bug (verificada en prod, user f47126cb, plan d2f2dbc6):
  1. Configuración → España: PATCH jsonb_set → health_profile.country='ES'. ✓
  2. «Renovar»: el frontend reenvía el formData del dispositivo, cuyo `country` es el
     'DO' SEMBRADO por initialFormData (el usuario jamás lo eligió).
  3. La generación lee ese país stale → plan criollo con RD$ (logs 19:11-19:14).
  4. El persist post-pipeline (`hp_data.update(data)`) escribe 'DO' de vuelta al
     perfil → el 'ES' de Configuración muere en silencio (SQL forense: country='DO').

La hidratación F2a (`_enrich_clinical_from_profile`) no aplicaba: es de las superficies
de UPDATE (swap/regenerate-day) y además su guard `if not data.get("country")` asume
que un payload CON país trae una elección — el default sembrado es indistinguible de
una elección desde el server... salvo por `update_reason`, que solo las regens
explícitas mandan. Esa es la señal que este fix explota.

Cubre:
  1. Unit: el helper `_hydrate_country_from_profile_for_submit` —
     perfil GANA con update_reason (el incidente), payload gana en wizard fresco,
     fill-si-falta, no-op para guest/sin-perfil, fail-open ante excepción DB.
  2. Parser: los DOS entry points (/analyze y /analyze/stream) invocan el helper
     tras `_close_medical_freetext_scope` (ANTES del pipeline y del merge a
     health_profile — el orden es lo que mata el clobber).
"""
import os
import re

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(BACKEND, rel), encoding="utf-8") as f:
        return f.read()


try:
    from routers.plans import _hydrate_country_from_profile_for_submit as _HYDRATE
    _IMPORT_ERR = None
except Exception as _e:  # pragma: no cover - entorno sin deps del router
    _HYDRATE = None
    _IMPORT_ERR = _e

requires_router = pytest.mark.skipif(
    _HYDRATE is None,
    reason=f"routers.plans no importable en este entorno: {_IMPORT_ERR}",
)


# --------------------------------------------------------------------------------------
# 1. Unit — semántica del helper
# --------------------------------------------------------------------------------------

@requires_router
def test_renovacion_perfil_gana_sobre_payload_stale(monkeypatch):
    """El incidente exacto: perfil ES, payload trae el 'DO' sembrado + update_reason."""
    import db
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"health_profile": {"country": "ES"}})
    data = {"country": "DO", "update_reason": "swap:variety", "age": 30}
    _HYDRATE(data, "user-1")
    assert data["country"] == "ES", "en regen explícita el país del PERFIL debe ganar"


@requires_router
def test_wizard_fresco_payload_gana(monkeypatch):
    """Sin update_reason el payload puede traer una elección RECIÉN hecha en QCountry."""
    import db
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"health_profile": {"country": "ES"}})
    data = {"country": "MX", "age": 30}
    _HYDRATE(data, "user-1")
    assert data["country"] == "MX", "un wizard completo con elección nueva NO debe ser pisado"


@requires_router
def test_fill_si_falta_sin_update_reason(monkeypatch):
    import db
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"health_profile": {"country": "CO"}})
    data = {"age": 30}
    _HYDRATE(data, "user-1")
    assert data["country"] == "CO", "payload sin país debe rellenarse del perfil"


@requires_router
def test_guest_es_noop(monkeypatch):
    import db

    def _should_not_be_called(uid):
        raise AssertionError("no debe consultar perfil para guests")

    monkeypatch.setattr(db, "get_user_profile", _should_not_be_called)
    data = {"country": "DO", "update_reason": "variety"}
    _HYDRATE(data, "guest")
    assert data["country"] == "DO"
    _HYDRATE(data, None)
    assert data["country"] == "DO"


@requires_router
def test_sin_pais_en_perfil_payload_intacto(monkeypatch):
    import db
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"health_profile": {}})
    data = {"country": "DO", "update_reason": "variety"}
    _HYDRATE(data, "user-1")
    assert data["country"] == "DO", "sin país en el perfil no hay nada que imponer"


@requires_router
def test_excepcion_db_fail_open(monkeypatch):
    import db

    def _boom(uid):
        raise RuntimeError("db caída")

    monkeypatch.setattr(db, "get_user_profile", _boom)
    data = {"country": "DO", "update_reason": "variety"}
    _HYDRATE(data, "user-1")  # no debe lanzar
    assert data["country"] == "DO", "ante error de DB el payload queda intacto (fail-open)"


@requires_router
def test_pais_invalido_del_perfil_se_rechaza_sin_canonicalizar(monkeypatch):
    """El writer rechaza prosa; no la convierte silenciosamente a DO."""
    import db
    from fastapi import HTTPException
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"health_profile": {"country": "españa"}})
    data = {"country": "DO", "update_reason": "variety"}
    with pytest.raises(HTTPException) as caught:
        _HYDRATE(data, "user-1")
    assert caught.value.status_code == 400
    assert data["country"] == "DO", "el valor inválido no debe alcanzar el payload ni coercerse"


# --------------------------------------------------------------------------------------
# 2. Parser — los dos entry points invocan el helper, en el orden correcto
# --------------------------------------------------------------------------------------

def test_ambos_entry_points_invocan_el_hidratado():
    src = _read(os.path.join("routers", "plans.py"))
    calls = re.findall(r"_hydrate_country_from_profile_for_submit\(data, verified_user_id\)", src)
    assert len(calls) == 2, (
        f"esperados EXACTAMENTE 2 call sites (/analyze y /analyze/stream), hay {len(calls)}. "
        "Si añadiste un tercer entry point de generación, debe hidratarse igual; si quitaste "
        "uno, el clobber de país renace en esa superficie."
    )
    assert "tooltip-anchor: P1-COUNTRY-RENEWAL-PROFILE-WINS" in src, (
        "el tooltip-anchor del helper desapareció — si renombraste el helper, actualiza este test"
    )


def test_hidratado_ocurre_tras_close_medical_y_antes_del_pipeline():
    """El call site debe seguir a `_close_medical_freetext_scope(data)` (entrada temprana del
    handler, antes del pipeline). Si alguien lo mueve después del launch, el merge post-pipeline
    vuelve a escribir el país stale y el bug renace con los tests unit en verde."""
    src = _read(os.path.join("routers", "plans.py"))
    closes = [m.end() for m in re.finditer(r"_close_medical_freetext_scope\(data\)", src)]
    assert len(closes) == 2, "se esperaban 2 usos de _close_medical_freetext_scope en handlers"
    for pos in closes:
        window = src[pos:pos + 600]
        assert "_hydrate_country_from_profile_for_submit(data, verified_user_id)" in window, (
            "el hidratado de país debe invocarse inmediatamente tras _close_medical_freetext_scope "
            "en AMBOS handlers (ventana de 600 chars) — moverlo más tarde arriesga correr después "
            "del snapshot que alimenta al pipeline"
        )
