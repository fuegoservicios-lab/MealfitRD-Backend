"""[P1-PLAN-LOTE-771 · 2026-09-28] Regalos de la cuenta: la tabla y las reglas (spec 2026-09-28-admin-cuentas-regalos-
design §3). Lo pagado y lo regalado nunca se mezclan: se superpone al leer, el mayor de los dos manda, y si la tabla no
se puede leer (o el knob está apagado) cada cuenta queda exactamente con lo que paga."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import regalos_cuenta as rc

_BACKEND = Path(__file__).resolve().parents[1]
_MIG = "p1_plan_lote_771_account_grants_2026_09_28.sql"
UID = "33333333-3333-3333-3333-333333333333"


def _g(kind, amount=None, plan=None, ends=None, gid="g1"):
    ahora = datetime.now(timezone.utc)
    return {"id": gid, "kind": kind, "amount": amount, "plan": plan, "starts_at": ahora - timedelta(days=1),
            "ends_at": ends, "created_at": ahora}


def test_inicio_de_mes_cruza_el_anio():
    ahora = datetime(2026, 12, 15, 10, tzinfo=timezone.utc)
    assert rc.inicio_de_mes(0, ahora) == datetime(2026, 12, 1, tzinfo=timezone.utc)
    assert rc.inicio_de_mes(1, ahora) == datetime(2027, 1, 1, tzinfo=timezone.utc)
    assert rc.inicio_de_mes(2, ahora) == datetime(2027, 2, 1, tzinfo=timezone.utc)


def test_extra_suma_solo_su_medidor():
    regalos = [_g("creditos_generacion", 20), _g("creditos_generacion", 5), _g("creditos_coach", 100),
               _g("plan", plan="plus")]
    assert rc.extra_de(regalos, "generacion") == 25 and rc.extra_de(regalos, "coach") == 100


@pytest.mark.parametrize("pagado,cortesia,esperado", [
    ("gratis", "plus", "plus"), ("basic", "plus", "plus"), ("plus", "basic", "plus"), ("ultra", "plus", "ultra"),
    ("plus", "plus", "plus"), ("admin", "ultra", "admin"), (None, "basic", "basic"), ("gratis", None, "gratis"),
    (None, None, None),
])
def test_plan_efectivo_es_el_mayor(pagado, cortesia, esperado):
    assert rc.plan_efectivo(pagado, {"plan": cortesia, "hasta": None} if cortesia else None) == esperado


def test_vigentes_filtra_en_sql_y_no_consulta_con_un_id_que_no_es_uuid(monkeypatch):
    llamadas = []
    monkeypatch.setattr(rc, "execute_sql_query", lambda q, p=None, **k: llamadas.append((" ".join(q.split()), p)) or [])
    assert rc.regalos_vigentes("guest") == [] and llamadas == []
    rc.regalos_vigentes(UID)
    q, p = llamadas[0]
    assert "revoked_at IS NULL" in q and "starts_at <= now()" in q and "(ends_at IS NULL OR ends_at > now())" in q
    assert p == (UID,)


def test_lectura_rota_o_knob_apagado_devuelve_nada(monkeypatch):
    def _rota(*a, **k):
        raise RuntimeError("relation does not exist")
    monkeypatch.setattr(rc, "execute_sql_query", _rota)
    assert rc.regalos_vigentes(UID) == []
    monkeypatch.setattr(rc, "execute_sql_query", lambda *a, **k: [_g("plan", plan="ultra")])
    monkeypatch.setenv("MEALFIT_ACCOUNT_GRANTS", "false")
    assert rc.regalos_vigentes(UID) == []
    assert rc.superponer({"id": UID, "plan_tier": "gratis"})["plan_tier"] == "gratis"


def test_superponer(monkeypatch):
    hasta = datetime(2026, 11, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(rc, "regalos_vigentes", lambda uid: [
        _g("plan", plan="plus", ends=hasta), _g("creditos_generacion", 20), _g("creditos_coach", 50)])
    p = rc.superponer({"id": UID, "plan_tier": "gratis"})
    assert p["plan_tier"] == "plus" and p["plan_tier_pagado"] == "gratis"
    assert p["cortesia"] == {"plan": "plus", "hasta": hasta.isoformat()}
    assert p["creditos_extra"] == {"generacion": 20, "coach": 50}


def test_superponer_cortesia_igual_a_lo_pagado_no_se_anuncia(monkeypatch):
    # Review Focus 3: le dieron Plus de cortesía y después pagó Plus → nada de «cortesía» en sus pantallas
    monkeypatch.setattr(rc, "regalos_vigentes", lambda uid: [_g("plan", plan="plus")])
    p = rc.superponer({"id": UID, "plan_tier": "plus"})
    assert p["plan_tier"] == "plus" and p["cortesia"] is None


def test_superponer_admin_no_lee_regalos(monkeypatch):
    monkeypatch.setattr(rc, "regalos_vigentes", lambda uid: pytest.fail("una cuenta admin no lee regalos"))
    p = rc.superponer({"id": UID, "plan_tier": "admin"})
    assert p["plan_tier"] == "admin" and p["creditos_extra"] == {"generacion": 0, "coach": 0}


def test_superponer_none():
    assert rc.superponer(None) is None


def test_migracion_idempotente_y_con_sus_reglas():
    sql = (_BACKEND / "migrations" / _MIG).read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS public.account_grants" in sql
    assert "REFERENCES public.user_profiles(id) ON DELETE CASCADE" in sql      # Review Focus 5
    for c in ("account_grants_kind_chk", "account_grants_forma_chk", "account_grants_ventana_chk",
              "account_grants_motivo_chk"):
        assert f"DROP CONSTRAINT IF EXISTS {c}" in sql and f"ADD CONSTRAINT {c}" in sql
    assert "plan IN ('basic', 'plus', 'ultra')" in sql and "amount BETWEEN 1 AND 1000" in sql
    assert "CREATE UNIQUE INDEX IF NOT EXISTS account_grants_una_cortesia_idx" in sql
    assert "WHERE kind = 'plan' AND revoked_at IS NULL" in sql
    assert "RAISE EXCEPTION" in sql
    raiz = _BACKEND.parent / "migrations" / _MIG                 # SSOT: la copia de la raíz, idéntica
    if raiz.exists():
        assert raiz.read_text(encoding="utf-8") == sql
