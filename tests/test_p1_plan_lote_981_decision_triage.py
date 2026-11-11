"""[P1-PLAN-LOTE-981 · P1-DECISION-TRIAGE · 2026-11-11] Gemini Decision 1 como triaje (BB-074..BB-079). HTTP mockeado: cero IA real.

Ancla: el triaje solo PREFILTRA una foto o FRENA una vez una bebida ambigua; jamás escribe por una probabilidad, y
cualquier fallo deja la conducta de siempre.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import decision_triage as dt  # noqa: E402


class _Resp:
    def __init__(self, status, data):
        self.status_code = status
        self._data = data
        self.text = str(data)

    def json(self):
        return self._data


class _Cliente:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def post(self, url, json=None, timeout=None, headers=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
        if self.exc:
            raise self.exc
        return self.resp


def _answers(bebida=0.1, media=None, intencion="new_meal_log", fusion=None):
    a = {
        dt.Q_BEBIDA: {"type": "noul", "noul": bebida, "confidence": 0.9},
        # el API ordena las opciones alfabéticamente: el mapeo es por CLAVE
        dt.Q_INTENCION: {"type": "choice", "choice": intencion, "confidence": 0.8,
                         "probabilities": {"off_topic": 0.05, "new_meal_log": 0.7, "general_coaching": 0.05,
                                           "correct_last_meal": 0.1, "append_active_meal": 0.1}},
    }
    if media is not None:
        a[dt.Q_MEDIA] = {"type": "noul", "noul": media, "confidence": 0.9}
    if fusion is not None:
        a[dt.Q_FUSION] = {"type": "noul", "noul": fusion, "confidence": 0.9}
    return {"answers": a, "usage": {"total_input_tokens": 120, "total_output_tokens": 0}}


@pytest.fixture
def entorno(monkeypatch):
    monkeypatch.setenv("MEALFIT_DECISION_TRIAGE_MODE", "guard")
    monkeypatch.setattr(dt, "_api_key", lambda: "k")
    monkeypatch.setattr(dt, "_excluido_por_region", lambda: False)
    monkeypatch.setattr(dt, "registrar_metrica", lambda *a, **k: None)
    monkeypatch.setattr(dt, "_comida_activa_de", lambda uid: None)
    import consentimientos
    monkeypatch.setattr(consentimientos, "permite_ia", lambda uid, donde="": True)
    import db
    monkeypatch.setattr(db, "log_llm_usage_event", lambda **k: None, raising=False)

    def poner(resp=None, exc=None):
        c = _Cliente(resp, exc)
        monkeypatch.setattr(dt, "_cliente", lambda: c)
        return c
    return poner


def test_off_no_hace_request(entorno, monkeypatch):
    c = entorno(_Resp(200, _answers()))
    monkeypatch.setenv("MEALFIT_DECISION_TRIAGE_MODE", "off")
    assert dt.run_triage("me tomé un jugo") is None
    assert c.calls == []
    assert dt.iniciar_turno("me tomé un jugo", "u1") is None


def test_default_es_off(monkeypatch):
    monkeypatch.delenv("MEALFIT_DECISION_TRIAGE_MODE", raising=False)
    assert dt.modo() == "off"


def test_request_contrato_real(entorno):
    c = entorno(_Resp(200, _answers(fusion=0.5)))
    res = dt.run_triage("me tomé un jugo", comida_activa="Mangú con salami")
    assert res is not None
    body = c.calls[0]["json"]
    assert c.calls[0]["url"] == "https://generativelanguage.googleapis.com/v1beta/decisions"
    assert c.calls[0]["headers"]["x-goog-api-key"] == "k"
    assert body["model"] == "gemini-decision-1"  # [P1-DECISION-TRIAGE]
    assert body["questions"][dt.Q_BEBIDA]["type"] == "noul"
    assert set(body["questions"][dt.Q_INTENCION]["criteria"]) == set(dt.INTENCIONES)
    assert dt.Q_FUSION in body["questions"]
    assert dt.Q_MEDIA not in body["questions"]
    # Eliminadas del RFC a propósito: no se piden.
    assert "criollo_fat_level" not in body["questions"] and "ready_to_commit_db" not in body["questions"]
    # Sin fechas ni horas en el contexto (el día biológico y el TTL son del código, no del modelo).
    texto = " ".join(b.get("text", "") for b in body["input"])
    assert not re.search(r"\d{1,2}:\d{2}|\d{4}-\d{2}-\d{2}", texto)


def test_sin_comida_activa_no_pregunta_fusion(entorno):
    c = entorno(_Resp(200, _answers()))
    dt.run_triage("me tomé un jugo")
    assert dt.Q_FUSION not in c.calls[0]["json"]["questions"]


@pytest.mark.parametrize("resp,exc", [
    (_Resp(500, {}), None), (_Resp(200, {"answers": {}}), None), (_Resp(200, "basura"), None),
    (None, TimeoutError("lento")),
])
def test_fail_open(entorno, resp, exc):
    entorno(resp, exc)
    assert dt.run_triage("me tomé un jugo") is None


def test_sin_key_o_ue_devuelve_none(entorno, monkeypatch):
    c = entorno(_Resp(200, _answers()))
    monkeypatch.setattr(dt, "_api_key", lambda: None)
    assert dt.run_triage("jugo") is None
    monkeypatch.setattr(dt, "_api_key", lambda: "k")
    monkeypatch.setattr(dt, "_excluido_por_region", lambda: True)
    assert dt.run_triage("jugo") is None
    assert c.calls == []


def test_choice_se_lee_por_clave(entorno):
    entorno(_Resp(200, _answers()))
    res = dt.run_triage("jugo")
    assert res.eleccion(dt.Q_INTENCION) == "new_meal_log"
    assert res.probabilidades(dt.Q_INTENCION)["new_meal_log"] == 0.7


def _turno(entorno, p):
    entorno(_Resp(200, _answers(bebida=p)))
    dt.iniciar_turno("me tomé un jugo", "u1")
    dt.resultado_del_turno(espera=2.0)


def test_guard_frena_una_vez(entorno):
    _turno(entorno, 0.9)
    txt = dt.freno_de_bebida("log_consumed_meal", [])
    assert txt and dt.FRENO_MARCA in txt

    class _Msg:
        content = txt
    assert dt.freno_de_bebida("log_consumed_meal", [_Msg()]) is None   # tope: 1 freno por comida


def test_bajo_umbral_no_frena(entorno):
    _turno(entorno, 0.79)
    assert dt.freno_de_bebida("log_consumed_meal", []) is None


def test_shadow_nunca_frena(entorno, monkeypatch):
    monkeypatch.setenv("MEALFIT_DECISION_TRIAGE_MODE", "shadow")
    _turno(entorno, 0.99)
    assert dt.freno_de_bebida("log_consumed_meal", []) is None


def test_solo_frena_log_consumed_meal_y_no_foto_ni_repeticion(entorno):
    _turno(entorno, 0.99)
    assert dt.freno_de_bebida("correct_consumed_meal", []) is None
    assert dt.freno_de_bebida("log_consumed_meal", [], turno_con_foto=True) is None
    assert dt.freno_de_bebida("log_consumed_meal", [], repeticion=True) is None


def test_media_umbral(entorno):
    entorno(_Resp(200, {"answers": {dt.Q_MEDIA: {"type": "noul", "noul": 0.10, "confidence": 0.9}}}))
    r = dt.run_triage(media={"data": b"\xff\xd8x", "mime_type": "image/jpeg"})
    assert dt.foto_rechazada(r) is True
    entorno(_Resp(200, {"answers": {dt.Q_MEDIA: {"type": "noul", "noul": 0.16, "confidence": 0.9}}}))
    r = dt.run_triage(media={"data": b"\xff\xd8x", "mime_type": "image/jpeg"})
    assert dt.foto_rechazada(r) is False
    assert dt.foto_rechazada(None) is False
    p = dt.payload_no_comida()
    assert p["is_food"] is False and p["calories"] == 0


def test_shadow_no_rechaza_foto(entorno, monkeypatch):
    entorno(_Resp(200, {"answers": {dt.Q_MEDIA: {"type": "noul", "noul": 0.01, "confidence": 0.9}}}))
    r = dt.run_triage(media={"data": b"x", "mime_type": "image/jpeg"})
    monkeypatch.setenv("MEALFIT_DECISION_TRIAGE_MODE", "shadow")
    assert dt.foto_rechazada(r) is False


# ── anclas estructurales ──
def _src(rel):
    return (_BACKEND / rel).read_text(encoding="utf-8")


def test_freno_va_despues_del_override_user_id():
    s = _src("agent.py")
    i_override = s.index('tool_args["user_id"] = _trusted_uid')
    i_freno = s.index("freno_de_bebida(")
    assert i_override < i_freno


def test_ningun_camino_escribe_por_probabilidad():
    s = _src("decision_triage.py")
    for prohibido in ("consumed_meals SET", "INSERT INTO consumed_meals", "UPDATE consumed_meals",
                      "log_consumed_meal(", "correct_consumed_meal("):
        assert prohibido not in s


def test_cableado_en_chat_y_upload():
    assert _src("routers/chat.py").count('__import__("decision_triage").iniciar_turno(prompt, user_id)') == 2
    d = _src("routers/diary.py")
    assert "_dt.foto_rechazada(_dt_res)" in d and "P1-DECISION-TRIAGE" in d


def test_reglas_deterministas_intactas():
    s = _src("decision_triage.py")
    assert "biological_date_resolver" not in s
    assert "_FUSION_TTL_MINUTOS" not in s


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · ', _src("app.py"))
    assert m and int(m.group(1)) >= 981
