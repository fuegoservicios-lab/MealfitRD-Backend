"""[P1-PLAN-LOTE-1115 · 2026-11-20] Quien deja de usar la app (gratis 3 días, de pago 10) deja de gastar en bloques del
plan: el pickup del worker los salta y, al volver, el siguiente tick los recoge. La presencia la marca la autenticación."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import pausa_inactividad as p  # noqa: E402


def test_el_fragmento_frena_por_nivel_y_nunca_el_primer_bloque(monkeypatch):
    monkeypatch.delenv("MEALFIT_PAUSA_INACTIVIDAD", raising=False)
    f = p.fragmento_sql("q1")
    assert f.lstrip().startswith("-- [P1-PLAN-LOTE-1115]") and "AND NOT (" in f
    assert "COALESCE(q1.chunk_kind, '') <> 'initial_plan'" in f, "el plan recién pedido nunca espera"
    assert "THEN 3 ELSE 10 END" in f
    for fuente in ("'presencia:' || q1.user_id::text", "FROM consumed_meals", "ami.role = 'user'", "FROM meal_plans"):
        assert fuente in f
    assert "), NOW())" in f, "sin ningún dato no frena (falla abierto)"
    assert "q.user_id" in p.fragmento_sql("q") and "q1." not in p.fragmento_sql("q")


def test_knobs_y_entrada_segura(monkeypatch):
    monkeypatch.setenv("MEALFIT_PAUSA_INACTIVIDAD_DIAS_GRATIS", "5")
    monkeypatch.setenv("MEALFIT_PAUSA_INACTIVIDAD_DIAS_PAGO", "9999")
    assert "THEN 5 ELSE 90 END" in p.fragmento_sql(), "acotado a [1, 90]"
    monkeypatch.setenv("MEALFIT_PAUSA_INACTIVIDAD_DIAS_GRATIS", "abc")
    assert p.dias_gratis() == 3
    monkeypatch.setenv("MEALFIT_PAUSA_INACTIVIDAD", "off")
    assert p.fragmento_sql() == "" and p.marcar_presencia("u1") is False
    monkeypatch.delenv("MEALFIT_PAUSA_INACTIVIDAD")
    for malo in ("q1; DROP TABLE x", "Q1", "", "1q"):
        try:
            p.fragmento_sql(malo)
        except ValueError:
            continue
        raise AssertionError(f"alias aceptado: {malo!r}")


def test_la_presencia_se_anota_como_mucho_cada_30_min(monkeypatch):
    monkeypatch.delenv("MEALFIT_PAUSA_INACTIVIDAD", raising=False)
    escritas = []
    monkeypatch.setattr(p, "_escribir", lambda uid: escritas.append(uid))
    monkeypatch.setattr(p.threading, "Thread", lambda target, args, **k: type("T", (), {"start": lambda s: target(*args)})())
    p._MARCAS.clear()
    monkeypatch.setattr(p.marcar_presencia, "_en_test", True, raising=False)
    reloj = [1000.0]
    monkeypatch.setattr(p.time, "monotonic", lambda: reloj[0])
    assert p.marcar_presencia("u1") is True and p.marcar_presencia("u1") is False
    reloj[0] += 29 * 60
    assert p.marcar_presencia("u1") is False
    reloj[0] += 2 * 60
    assert p.marcar_presencia("u1") is True and p.marcar_presencia("u2") is True
    assert p.marcar_presencia("guest") is False and p.marcar_presencia(None) is False
    assert escritas == ["u1", "u1", "u2"]


def test_cableado():
    cron = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    i = cron.index('_gates_sql = (_pm_gate_sql.rstrip() if _pm_on else "")')
    assert '__import__("pausa_inactividad").fragmento_sql("q1")' in cron[i:i + 300], "el freno va en el pickup"
    j = cron.index('.replace("__AI_CONSENT_GATE__", __import__("consentimientos").fragmento_sql_permiso("q.user_id")')
    assert '__import__("pausa_inactividad").fragmento_sql("q")' in cron[j:j + 300], "la alerta de zombis lo respeta"
    auth = (_BACKEND / "auth.py").read_text(encoding="utf-8")
    k = auth.index("def _uid_si_la_identidad_vive(")
    cuerpo = auth[k:auth.index("\n# ----", k)]
    assert cuerpo.rstrip().endswith('__import__("pausa_inactividad").marcar_presencia(uid)   # [P1-PLAN-LOTE-1115] usó la app: sus bloques siguen\n    return uid')
    assert '"presencia:"' in (_BACKEND / "db_profiles.py").read_text(encoding="utf-8"), "el borrado de cuenta la purga"
