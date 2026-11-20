"""[P1-PLAN-LOTE-1116 · 2026-11-20] Topes de uso por nivel en UNA tabla (`topes_por_nivel.TABLA`) y aplicados en los
cuatro sitios que gastan: generación, coach (día + mes), escáner de fotos y voz en vivo (mes + duración de sesión),
más un tope de gasto de IA del mes por nivel. Precios bajados (3,99 / 6,99 / 11,99)."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import topes_por_nivel as tp  # noqa: E402

_FRONT = Path(os.environ.get("MEALFIT_TEST_FRONTEND_SRC") or (_BACKEND.parent / "frontend" / "src"))   # override: worktrees


def _uso(monkeypatch, *, tier="gratis", mes=0.0, fotos=0, coach=0, voz_s=0.0):
    monkeypatch.delenv("MEALFIT_TOPES_POR_NIVEL", raising=False)
    monkeypatch.setattr(tp, "_tier", lambda uid: tier)
    monkeypatch.setattr(tp, "gasto_mes_usd", lambda uid: mes)
    monkeypatch.setattr(tp, "fotos_24h", lambda uid: fotos)
    monkeypatch.setattr(tp, "coach_24h", lambda uid: coach)
    monkeypatch.setattr(tp, "voz_segundos_mes", lambda uid: voz_s)


def test_la_tabla_sube_con_el_nivel_y_el_peor_caso_cabe_en_el_precio():
    for clave in tp.CLAVES:
        vals = [tp.TABLA[n][clave] for n in tp.NIVELES]
        assert vals == sorted(vals), f"{clave} no crece con el nivel: {vals}"
    # lo peor que gasta un usuario de pago, por debajo de ~45 % de lo que deja su cuota tras PayPal (3,49 % + 0,49)
    for nivel, precio in (("basic", 3.99), ("plus", 6.99), ("ultra", 11.99)):
        neto = precio * (1 - 0.0349) - 0.49
        assert tp.TABLA[nivel]["usd_mes"] <= 0.46 * neto, (nivel, tp.TABLA[nivel]["usd_mes"], neto)
    assert tp.TABLA["gratis"]["usd_mes"] <= 0.50


def test_override_por_knob_y_admin_sin_tope(monkeypatch):
    monkeypatch.setenv("MEALFIT_TOPE_GRATIS_FOTOS_DIA", "7")
    assert tp.limite("gratis", "fotos_dia") == 7
    monkeypatch.setenv("MEALFIT_TOPE_GRATIS_FOTOS_DIA", "abc")
    assert tp.limite("gratis", "fotos_dia") == tp.TABLA["gratis"]["fotos_dia"]
    assert tp.limite("admin", "fotos_dia") >= 99999 and tp.limite("loquesea", "coach_dia") == tp.TABLA["gratis"]["coach_dia"]
    with pytest.raises(KeyError):
        tp.limite("gratis", "nada")


def test_frenado_por_cada_tope(monkeypatch):
    _uso(monkeypatch)
    assert tp.frenado("u1", "foto") is None
    _uso(monkeypatch, fotos=tp.TABLA["gratis"]["fotos_dia"])
    assert tp.frenado("u1", "foto")["codigo"] == "tope_fotos_dia"
    assert tp.frenado("u1", "coach") is None, "el de fotos no frena el coach"
    _uso(monkeypatch, coach=tp.TABLA["gratis"]["coach_dia"])
    assert tp.frenado("u1", "coach")["codigo"] == "tope_coach_dia"
    _uso(monkeypatch, voz_s=tp.TABLA["gratis"]["voz_min_mes"] * 60)
    assert tp.frenado("u1", "voz")["codigo"] == "tope_voz_mes"
    _uso(monkeypatch, tier="plus", mes=tp.TABLA["plus"]["usd_mes"])
    for que in ("foto", "coach", "voz", "generacion"):
        assert tp.frenado("u1", que)["codigo"] == "tope_mes_usd", "el gasto del mes frena todo lo que cuesta"
    _uso(monkeypatch, tier="admin", mes=999, fotos=999)
    assert tp.frenado("u1", "foto") is None
    assert tp.frenado(None, "foto") is None and tp.frenado("guest", "foto") is None
    monkeypatch.setenv("MEALFIT_TOPES_POR_NIVEL", "off")
    monkeypatch.setattr(tp, "fotos_24h", lambda uid: 999)
    assert tp.frenado("u1", "foto") is None


def test_la_sesion_de_voz_no_pasa_de_lo_que_queda_del_mes(monkeypatch):
    _uso(monkeypatch, tier="basic", voz_s=0)
    assert tp.segundos_de_sesion_voz("u1") == tp.TABLA["basic"]["voz_min_sesion"] * 60
    _uso(monkeypatch, tier="basic", voz_s=tp.TABLA["basic"]["voz_min_mes"] * 60 - 90)
    assert tp.segundos_de_sesion_voz("u1") == 90
    _uso(monkeypatch, tier="basic", voz_s=tp.TABLA["basic"]["voz_min_mes"] * 60 - 5)
    assert tp.segundos_de_sesion_voz("u1") == 30, "mínimo 30 s"


def test_auth_coincide_con_la_tabla():
    auth = (_BACKEND / "auth.py").read_text(encoding="utf-8")
    for nivel, env in (("gratis", "GRATIS"), ("basic", "BASIC"), ("plus", "PLUS"), ("ultra", "ULTRA")):
        m = re.search(rf'_env_int\("MEALFIT_TIER_LIMIT_{env}",\s*(\d+)\)', auth)
        assert int(m.group(1)) == tp.TABLA[nivel]["generacion_mes"], nivel
        m = re.search(rf'os.environ.get\("MEALFIT_COACH_LIMIT_{env}",\s*"(\d+)"\)', auth)
        assert int(m.group(1)) == tp.TABLA[nivel]["coach_mes"], nivel
    assert auth.count('__import__("topes_por_nivel").frenado(verified_user_id, ') == 2   # coach + generación


def test_cableado_en_foto_voz_y_chat():
    diary = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    i = diary.index('async def api_diary_upload(')
    j = diary.index('__import__("topes_por_nivel").frenado(verified_user_id, "foto")', i)
    assert j < diary.index("actual_user_id = user_id if user_id != \"guest\" else session_id", i), "antes de leer la foto"
    live = (_BACKEND / "coach_live.py").read_text(encoding="utf-8")
    assert '_tpn.frenado(str(user_id), "voz")' in live and "tope_s=tope_s" in live and "s.tope_s" in live
    cht = (_BACKEND / "chat_hard_turn.py").read_text(encoding="utf-8")
    assert '_tpn.limite(_tpn._tier(str(user_id)), "usd_dia")' in cht
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert '"X-Tope",' in app, "la app nativa necesita leer la cabecera"


def test_paridad_con_la_pagina_de_planes():
    if not _FRONT.exists():
        pytest.skip("sin el repo hermano del frontend")
    plans = (_FRONT / "config" / "plans.js").read_text(encoding="utf-8")
    bloque = re.search(r"export const TIER_LIMITS = \{(.*?)\n\};", plans, re.S).group(1)
    for nivel in tp.NIVELES:
        fila = re.search(rf"{nivel}: \{{([^}}]*)\}}", bloque).group(1)
        for clave, valor in re.findall(r"(\w+): (\d+)", fila):
            assert int(valor) == tp.TABLA[nivel][clave], (nivel, clave)
    creditos = dict(re.findall(r"(\w+): (\d+)", re.search(r"TIER_CREDITS = \{([^}]*)\}", plans).group(1)))
    for nivel in tp.NIVELES:
        assert int(creditos[nivel]) == tp.TABLA[nivel]["generacion_mes"]
    for precio in ("'3.99'", "'6.99'", "'11.99'", "'29.99'", "'49.99'", "'89.99'"):
        assert precio in plans
