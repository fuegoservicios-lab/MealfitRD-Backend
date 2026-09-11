"""[P1-PLAN-LOTE-5 · 2026-09-11] Quinto lote del plan de pendientes (`docs/plan_pendientes_2026_09_11.md`): F5,
la auditoría de guards inertes (`docs/audits/f5_guards_inertes_2026_09_11.md`), primera tanda.

  · El escáner de alérgenos del pool (`_allergen_pool_item_banned`) era fail-OPEN: si reventaba, «no hay alérgeno»
    y el ítem entraba al plan de un alérgico. Con alergias declaradas, la duda VETA.
  · Las notas de seguridad de embarazo y por condición devolvían 0 en silencio cuando no podían evaluarse; ahora avisan.
  · `dish_registry.allergen_classes_for` sin vocabulario devolvía «sin alérgenos» en silencio; ahora avisa.
  · Cuatro funciones del god-file y dos de `db_core` con CERO llamadores, fuera.
  · El comentario de `cron_tasks` prometía un bypass (`_is_inventory_live_degraded`) que nadie invoca; ahora lo dice.
  · FALSO POSITIVO de la auditoría, corregido en el informe: la «cota absoluta de la cookie sin llamadores» SÍ se
    aplica en cada path de auth dentro de `_decode_session_cookie`. Contar llamadores no basta: la REGLA vivía en
    otro sitio. Este archivo lo fija para que nadie vuelva a «cablearla».

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ─────────────────────────── escáner de alérgenos del pool: fail-secure ───────────────────────────

def test_f5_escaner_de_alergenos_roto_veta_por_duda_solo_si_hay_alergias(monkeypatch, caplog):
    import graph_orchestrator as go

    def _boom(*_a, **_k):
        raise RuntimeError("escáner roto")

    monkeypatch.setattr(go, "_scan_allergen_violations", _boom)
    with caplog.at_level(logging.WARNING):
        assert go._allergen_pool_item_banned("Huevo", ["huevo"]) is True, "con alergias declaradas, la duda veta"
        assert go._allergen_pool_item_banned("Huevo", []) is False, "sin alergias no hay nada que vetar"
        assert go._allergen_pool_item_banned("Huevo", None) is False
    assert any("[P1-PLAN-LOTE-5]" in r.getMessage() for r in caplog.records), "el fallo del escáner se ve en el log"


def test_f5_escaner_sano_sigue_decidiendo_por_contenido():
    import graph_orchestrator as go
    assert go._allergen_pool_item_banned("Huevo", ["huevo"]) is True
    assert go._allergen_pool_item_banned("Arroz blanco", ["huevo"]) is False


# ─────────────────────────── notas clínicas y clases de alérgeno: avisan cuando no se evalúan ───────────────────────────

def test_f5_notas_clinicas_avisan_cuando_no_se_evaluan():
    src = _src("graph_orchestrator.py")
    assert "[P1-PLAN-LOTE-5] notas de seguridad de embarazo NO evaluadas" in src
    assert "[P1-PLAN-LOTE-5] notas de seguridad por condición NO evaluadas" in src
    i = src.find("def _apply_pregnancy_food_safety_annotations(")
    j = src.find("_pregnancy_catalog_risk_tokens()", i)
    assert "except Exception:\n        return 0" not in src[i:j], "el import del helper ya no falla en silencio"


def test_f5_clases_de_alergeno_sin_vocabulario_avisan(monkeypatch, caplog):
    import dish_registry as dr
    monkeypatch.setitem(sys.modules, "graph_orchestrator", None)   # `from graph_orchestrator import …` ⇒ ImportError
    with caplog.at_level(logging.WARNING, logger="dish_registry"):
        assert dr.allergen_classes_for(["Huevo", "Leche"]) == []
    assert any("vocabulario de alérgenos no disponible" in r.getMessage() for r in caplog.records)


# ─────────────────────────── muertos fuera ───────────────────────────

def test_f5_funciones_sin_llamador_fuera_del_god_file_y_de_db_core():
    import graph_orchestrator as go
    import db_core
    for fn in ("_select_ab_temp_pair", "get_circuit_breaker_snapshot", "get_progress_cb_stats_snapshot"):
        assert not hasattr(go, fn), fn
    assert hasattr(go, "_aselect_ab_temp_pair"), "la variante async (la viva) sigue"
    assert hasattr(go, "_compute_ab_temp_pair_from_rows")
    for fn in ("close_connection_pool", "aclose_connection_pool"):
        assert not hasattr(db_core, fn), fn
    src = _src("graph_orchestrator.py")
    assert "_coh_finite_delta_rv" not in src
    assert "def _select_ab_temp_pair" not in src
    assert "se borró en P1-PLAN-LOTE-5" in _src("app.py"), "el comentario del lifespan ya no cita una función que no existe"


# ─────────────────────────── el comentario del bypass dice la verdad (y sigue siéndolo) ───────────────────────────

def test_f5_el_comentario_del_bypass_dice_la_verdad():
    ct = _src("cron_tasks.py")
    assert "`_is_inventory_live_degraded()` EXISTE pero nadie la invoca" in ct
    menciones = len(re.findall(r"_is_inventory_live_degraded\(", ct))
    defs = len(re.findall(r"def _is_inventory_live_degraded\(", ct))
    assert defs == 1 and menciones == defs + 1, (
        "si alguien cablea `_is_inventory_live_degraded`, este comentario vuelve a mentir: actualízalo y quita este ancla")


# ─────────────────────────── el falso positivo: la cota de la cookie ya se aplicaba ───────────────────────────

def test_f5_la_cota_absoluta_de_la_cookie_ya_se_aplica_en_el_decodificador(monkeypatch):
    import auth as _auth
    monkeypatch.setattr(_auth, "_SESSION_SECRET", "s" * 40, raising=False)
    now = int(time.time())
    stale = _auth.mint_session_cookie("u-1", iat=now - _auth._SESSION_ABS_MAX_S - 60)
    fresh = _auth.mint_session_cookie("u-1")
    assert stale and fresh
    assert _auth.verify_session_cookie(stale) is None, "exp vigente pero iat fuera del cap ⇒ rechazada en el decodificador"
    assert _auth.verify_session_cookie(fresh) == "u-1"
    # El helper huérfano sigue (lo usan tests) pero `get_verified_user_id` NO lo necesita: no re-cablearlo.
    gv = _src("auth.py").split("async def get_verified_user_id(")[1].split("async def get_neon_bearer_user_id(")[0]
    assert "session_cookie_within_absolute_cap(" not in gv


def test_f5_informe_corregido_y_plan_al_dia():
    informe = _src("docs/audits/f5_guards_inertes_2026_09_11.md")
    assert "FALSO POSITIVO" in informe and "_decode_session_cookie" in informe
    plan = _src("docs/plan_pendientes_2026_09_11.md")
    for row in ("| F5 |", "| F2 |", "| F8 |"):
        assert row in plan, row
    assert "Falso positivo de la auditoría" in plan
    assert (_BACKEND.parent / "docs" / "superpowers" / "plans" / "2026-09-11-paises-gaps-reconciliacion.md").exists()


def test_marker_bumpeado():
    import app
    assert "[P1-PLAN-LOTE-5 · 2026-09-11]" in _src("app.py")
    assert app._LAST_KNOWN_PFIX.startswith("P1-PLAN-") and "2026-09-11" in app._LAST_KNOWN_PFIX
