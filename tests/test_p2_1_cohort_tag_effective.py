"""[P2-COHORT-TAG-EFFECTIVE · 2026-07-10] `_harden_pools_canary_cohort` etiquetaba TODO plan como
cohorte "on" cuando `HARDEN_POOLS_CANARY_PCT=0` (default) — SIN mirar el master switch
`HARDEN_POOLS_ENABLED` (también False por default hoy en prod). Resultado: el 100% de los planes de la
flota llevan `harden_pools_cohort="on"` en `clinical_band` aunque `harden_day_pools()` sea un no-op total
(early-return en la línea 1) — contamina cualquier análisis futuro "¿A1 mejoró la banda?" cuando se
active el canario real, porque el histórico OFF-real ya está mal-etiquetado "on". Fix: con el master
switch OFF, la cohorte reportada es "off" (honesta — el enforcer no corrió), independiente del %
canario. Solo con el master ON aplica el bucketing determinista por usuario.
"""
import graph_orchestrator as go


def test_master_off_reports_off_regardless_of_canary_pct():
    st = {"form_data": {"user_id": "u-123"}}
    for pct in (0, 50, 100):
        if not hasattr(go, "HARDEN_POOLS_CANARY_PCT"):
            continue
        import pytest
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(go, "HARDEN_POOLS_ENABLED", False)
            mp.setattr(go, "HARDEN_POOLS_CANARY_PCT", pct)
            assert go._harden_pools_canary_cohort(st) == "off", \
                f"master OFF debe reportar 'off' incluso con PCT={pct}"


def test_master_on_pct_zero_is_on():
    st = {"form_data": {"user_id": "u-123"}}
    import pytest
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(go, "HARDEN_POOLS_ENABLED", True)
        mp.setattr(go, "HARDEN_POOLS_CANARY_PCT", 0)
        assert go._harden_pools_canary_cohort(st) == "on"


def test_master_on_pct_100_is_off():
    st = {"form_data": {"user_id": "u-123"}}
    import pytest
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(go, "HARDEN_POOLS_ENABLED", True)
        mp.setattr(go, "HARDEN_POOLS_CANARY_PCT", 100)
        assert go._harden_pools_canary_cohort(st) == "off"


def test_master_on_bucketing_stable_per_user():
    st = {"form_data": {"user_id": "u-xyz"}}
    import pytest
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(go, "HARDEN_POOLS_ENABLED", True)
        mp.setattr(go, "HARDEN_POOLS_CANARY_PCT", 50)
        a = go._harden_pools_canary_cohort(st)
        b = go._harden_pools_canary_cohort(st)
        assert a == b and a in ("on", "off")


def test_marker_present():
    with open(go.__file__, encoding="utf-8") as f:
        src = f.read()
    assert "P2-COHORT-TAG-EFFECTIVE" in src
