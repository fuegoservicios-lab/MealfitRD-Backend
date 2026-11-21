"""[P1-PLAN-LOTE-1117 · 2026-11-20] Rango «probador» para la prueba cerrada de 14 días de Google Play: Sherwin y las
cuentas que entran por Android (cupo 11) tienen topes holgados sin tocar `plan_tier` ni lo que se vende."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import rango_probador as rp  # noqa: E402
import topes_por_nivel as tp  # noqa: E402

UID = "a9a658db-588c-486d-9ad1-40f17f4c91eb"


def test_la_fila_probador_queda_entre_plus_y_max_y_no_se_vende():
    assert "probador" not in tp.NIVELES and "probador" not in tp.tabla_publica()
    for clave in tp.CLAVES:
        assert tp.TABLA["plus"][clave] <= tp.TABLA["probador"][clave] <= tp.TABLA["ultra"][clave], clave
    assert tp.nivel_de("probador") == "probador"
    assert tp.limite("probador", "fotos_dia") == tp.TABLA["probador"]["fotos_dia"]
    assert tp.TABLA["probador"]["usd_mes"] * rp.CUPO <= 40      # el peor caso del rango entero, acotado


def test_el_rango_sube_gratis_basic_y_plus_pero_nunca_baja_max_ni_admin(monkeypatch):
    monkeypatch.setattr(rp, "es_probador", lambda uid: True)
    for nivel in ("gratis", "basic", "plus"):
        assert rp.nivel_con_rango(UID, nivel) == "probador"
    for nivel in ("ultra", "admin"):
        assert rp.nivel_con_rango(UID, nivel) == nivel
    monkeypatch.setattr(rp, "es_probador", lambda uid: False)
    assert rp.nivel_con_rango(UID, "gratis") == "gratis"


def test_vigencia_y_inscripcion():
    ahora = datetime(2026, 11, 20, tzinfo=timezone.utc)
    assert rp._vigente({"hasta": (ahora + timedelta(days=1)).isoformat()}, ahora)
    assert not rp._vigente({"hasta": (ahora - timedelta(days=1)).isoformat()}, ahora)
    assert not rp._vigente({"hasta": "basura"}, ahora) and not rp._vigente(None, ahora)
    assert rp._inscripcion_abierta(ahora)
    assert not rp._inscripcion_abierta(datetime(2027, 1, 1, tzinfo=timezone.utc))
    # solo Android, solo ids de cuenta; bajo pytest nunca escribe en la base real
    assert rp.inscribir_si_android(UID, "ios") is False
    assert rp.inscribir_si_android("guest", "android") is False
    assert rp.es_probador(UID) is False


def test_el_tier_de_los_topes_pasa_por_el_rango(monkeypatch):
    import db
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"plan_tier": "gratis"})
    monkeypatch.setattr(rp, "es_probador", lambda uid: True)
    assert tp._tier(UID) == "probador"


def test_auth_sube_los_cupos_mensuales_del_probador(monkeypatch):
    import auth
    monkeypatch.setattr(rp, "es_probador", lambda uid: True)
    assert auth._con_rango_probador(UID, "gratis", 60, "coach_mes") == tp.TABLA["probador"]["coach_mes"]
    assert auth._con_rango_probador(UID, "gratis", 4, "generacion_mes") == tp.TABLA["probador"]["generacion_mes"]
    assert auth._con_rango_probador(UID, "ultra", 2000, "coach_mes") == 2000
    monkeypatch.setattr(rp, "es_probador", lambda uid: False)
    assert auth._con_rango_probador(UID, "gratis", 60, "coach_mes") == 60


def test_cableado():
    fuente = {f: (_BACKEND / f).read_text(encoding="utf-8") for f in
              ("auth.py", "ajustes_cuenta.py", "regalos_cuenta.py", "db_profiles.py", "app.py")}
    assert fuente["auth.py"].count("_con_rango_probador(") >= 4
    assert "inscribir_si_android(uid, plataforma)" in fuente["ajustes_cuenta.py"]
    assert "_con_rango_probador" in fuente["regalos_cuenta.py"]
    assert '"probador:"' in fuente["db_profiles.py"]
    # el marcador de app.py avanza con cada lote (lo vigilan test_p3_1 y test_p2_hist_audit_14); aquí no se ancla
