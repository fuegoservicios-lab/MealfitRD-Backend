"""[P1-PLAN-LOTE-1006 · G-05 + G-17] Tracking 100% de tokens de visión, desglose por sub-fases y subida paralela.

Verifica sin gasto de IA:
1. `vision_usage_scope()` propaga usage y sub-fases (`llm_ms`, `resize_ms`, `sanitize_ms`) a través de `asyncio.gather`,
   manteniendo intacto el reset a `None` fuera de scope (contrato P2-VISION-USAGE-TOKENS).
2. `_VisionUsageCapture._extract` extrae usage desde `response_metadata` plano, `additional_kwargs`, `generation_info`
   y `kwargs`.
3. `registrar_vision_scan` persiste `fases_ms`, `modelo` y `resize` sin texto libre y conserva retrocompatibilidad.
4. `api_diary_upload` registra `llm_usage_events(node='vision_scan')` también cuando `is_food=False` o
   `analysis_failed=True` (sólo lo omite si `triage_rejected=True`), y corre `create_chat_attachment` en paralelo con
   visión cuando `purpose='chat'` (fallando con HTTP 503 si falla el adjunto).
"""
from __future__ import annotations

import asyncio
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException
from PIL import Image


def _fake_jpeg_bytes(w: int = 800, h: int = 600) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (w, h), color=(180, 90, 40)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class _FakeUpload:
    def __init__(self, payload: bytes, filename: str = "plate.jpg", content_type: str = "image/jpeg"):
        self._buf = BytesIO(payload)
        self.filename = filename
        self.content_type = content_type

    async def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)


def test_usage_capture_extracts_direct_rm_additional_kwargs_and_gen_info():
    from vision_agent import _VisionUsageCapture

    # Direct keys inside response_metadata (without nested token_usage/usage wrapper)
    cb_rm = _VisionUsageCapture()
    msg_rm = SimpleNamespace(
        usage_metadata=None,
        response_metadata={"input_tokens": 710, "output_tokens": 42, "cached_tokens": 200},
    )
    cb_rm.on_llm_end(SimpleNamespace(generations=[[SimpleNamespace(message=msg_rm)]], llm_output=None))
    assert cb_rm.usage == {"input_tokens": 710, "output_tokens": 42, "cached_tokens": 200}

    # additional_kwargs fallback
    cb_ak = _VisionUsageCapture()
    msg_ak = SimpleNamespace(
        usage_metadata=None,
        response_metadata=None,
        additional_kwargs={"token_usage": {"prompt_tokens": 400, "completion_tokens": 25}},
    )
    cb_ak.on_llm_end(SimpleNamespace(generations=[[SimpleNamespace(message=msg_ak)]], llm_output=None))
    assert cb_ak.usage == {"input_tokens": 400, "output_tokens": 25}

    # generation_info + kwargs fallback
    cb_gi = _VisionUsageCapture()
    gen_gi = SimpleNamespace(
        message=None,
        generation_info={"usage_metadata": {"input_tokens": 300, "output_tokens": 19}},
    )
    cb_gi.on_llm_end(SimpleNamespace(generations=[[gen_gi]], llm_output=None))
    assert cb_gi.usage == {"input_tokens": 300, "output_tokens": 19}

    cb_kw = _VisionUsageCapture()
    cb_kw.on_llm_end(
        SimpleNamespace(generations=[], llm_output=None),
        usage_metadata={"input_tokens": 150, "output_tokens": 12},
    )
    assert cb_kw.usage == {"input_tokens": 150, "output_tokens": 12}


def test_vision_usage_scope_propagates_across_asyncio_gather(monkeypatch):
    async def _run():
        import vision_agent as va

        class _FakeClient:
            def __init__(self):
                self._cbs = []

            def with_config(self, callbacks=None, **kw):
                self._cbs = list(callbacks or [])
                return self

            async def ainvoke(self, messages):
                await asyncio.sleep(0.01)
                msg = SimpleNamespace(usage_metadata={"input_tokens": 1450, "output_tokens": 77})
                for cb in self._cbs:
                    cb.on_llm_end(SimpleNamespace(generations=[[SimpleNamespace(message=msg)]], llm_output=None))
                return SimpleNamespace(
                    model_dump=lambda: {
                        "description": "Pollo guisado con arroz",
                        "is_food": True,
                        "meal_name": "Pollo con arroz",
                        "calories": 550,
                        "protein": 38,
                        "carbs": 52,
                        "healthy_fats": 18,
                    }
                )

        monkeypatch.setattr(va, "_resolve_vision_client", lambda schema: _FakeClient())
        monkeypatch.setattr(va, "is_vision_enabled", lambda: True)

        async def _peer_triage():
            await asyncio.sleep(0.005)
            return "triage_ok"

        raw = _fake_jpeg_bytes(600, 400)
        with va.vision_usage_scope():
            peer, out = await asyncio.gather(_peer_triage(), va.process_image_with_vision(raw))
            assert peer == "triage_ok"
            assert out["is_food"] is True
            assert va.get_last_vision_usage() == {"input_tokens": 1450, "output_tokens": 77}
            meta = va.get_last_vision_meta()
            assert meta["resize_ms"] >= 0
            assert meta["llm_ms"] >= 5.0
            assert meta["sanitize_ms"] >= 0
            assert isinstance(meta.get("resize_info"), dict)

    asyncio.run(_run())


def test_registrar_vision_scan_records_subphases_safely(monkeypatch):
    import telemetria_escaner as te

    captured = []
    monkeypatch.setattr(
        te,
        "execute_sql_write",
        lambda sql, params: captured.append((sql, params)),
    )
    te.registrar_vision_scan(
        "user-1",
        duracion_ms=2150.4,
        resultado={"is_food": True, "calories": 420, "photo_kind": "plato"},
        purpose="chat",
        fases_ms={"attachment_ms": 45.2, "resize_ms": 18.1, "llm_ms": 1990.5, "bad_nan": float("nan"), "neg": -5.0},
        modelo="gpt-6-luna",
        resize_info={"original_bytes": 500000, "sent_bytes": 90000, "resized": True, "sent_wh": "1024x768"},
    )
    assert len(captured) == 1
    import json
    meta = json.loads(captured[0][1][4])
    assert meta["resultado"] == "ok"
    assert meta["modelo"] == "gpt-6-luna"
    assert meta["fases_ms"] == {"attachment_ms": 45.2, "resize_ms": 18.1, "llm_ms": 1990.5}
    assert meta["resize"]["resized"] is True
    assert meta["resize"]["sent_wh"] == "1024x768"


def test_diary_upload_logs_usage_on_non_food_and_failure_and_runs_attachment_in_parallel(monkeypatch):
    async def _run():
        import routers.diary as rd
        import vision_agent as va
        import pais_del_estimador as pde
        import decision_triage as dt

        uid = "11111111-1111-1111-1111-111111111111"
        sid = "22222222-2222-2222-2222-222222222222"
        raw = _fake_jpeg_bytes(400, 300)

        events: list[dict] = []
        scans: list[dict] = []
        concurrency_events: list[str] = []

        async def _fake_pais(_uid):
            return "DO"

        monkeypatch.setattr(pde, "pais_del_usuario", _fake_pais)
        monkeypatch.setattr(dt, "modo", lambda: "off")
        monkeypatch.setattr(rd, "_vision_model_name", lambda: "gpt-6-luna")
        monkeypatch.setattr(rd, "get_or_create_session", lambda s, user_id=None: None)
        monkeypatch.setattr(rd, "build_chat_attachment_url", lambda aid: f"/att/{aid}")
        monkeypatch.setattr(
            rd,
            "log_llm_usage_event",
            lambda **kw: events.append(kw),
        )
        monkeypatch.setattr(
            rd,
            "registrar_vision_scan",
            lambda u, **kw: scans.append({"user_id": u, **kw}),
        )

        # Case 1: purpose='chat' + is_food=False -> MUST run in parallel AND log tokens!
        async def _slow_process_non_food(image_bytes, aclaracion=None, pais=None):
            concurrency_events.append("vision_start")
            await asyncio.sleep(0.03)
            va._set_last_vision_usage({"input_tokens": 810, "output_tokens": 22})
            va._merge_vision_meta(resize_ms=4.5, llm_ms=28.0, sanitize_ms=0.8)
            concurrency_events.append("vision_end")
            return {
                "description": "No es comida",
                "is_food": False,
                "photo_kind": "otro",
                "calories": 0,
                "protein": 0,
                "carbs": 0,
                "healthy_fats": 0,
            }

        def _slow_create_attachment(*args, **kwargs):
            import time as _t
            concurrency_events.append("att_start")
            _t.sleep(0.03)
            concurrency_events.append("att_end")
            return "att-123"

        monkeypatch.setattr(rd, "process_image_with_vision", _slow_process_non_food)
        monkeypatch.setattr(rd, "create_chat_attachment", _slow_create_attachment)

        bg = BackgroundTasks()
        res = await rd.api_diary_upload(
            background_tasks=bg,
            file=_FakeUpload(raw),
            user_id=uid,
            session_id=sid,
            purpose="chat",
            tz_offset_mins=-240,
            locale=None,
            aclaracion=None,
            verified_user_id=uid,
        )
        assert res["is_food"] is False
        assert res["attachment_id"] == "att-123"
        # Attachment started before vision finished => genuine parallel execution
        assert concurrency_events.index("att_start") < concurrency_events.index("vision_end")
        # usage logged even when is_food=False
        assert len(events) == 1
        assert events[0]["node"] == "vision_scan"
        assert events[0]["input_tokens"] == 810
        assert events[0]["output_tokens"] == 22

        # Run scheduled background tasks to verify registrar_vision_scan receives sub-phases
        for task in bg.tasks:
            task.func(*task.args, **task.kwargs)
        assert len(scans) == 1
        assert scans[0]["modelo"] == "gpt-6-luna"
        assert "llm_ms" in scans[0]["fases_ms"]
        assert "attachment_ms" in scans[0]["fases_ms"]

        # Case 2: analysis_failed=True -> still logs usage event
        events.clear()

        async def _failed_process(image_bytes, aclaracion=None, pais=None):
            va._set_last_vision_usage({"input_tokens": 500, "output_tokens": 0})
            return {
                "description": "Error analizando imagen.",
                "is_food": False,
                "analysis_failed": True,
                "calories": 0,
                "protein": 0,
                "carbs": 0,
                "healthy_fats": 0,
            }

        monkeypatch.setattr(rd, "process_image_with_vision", _failed_process)
        res_fail = await rd.api_diary_upload(
            background_tasks=BackgroundTasks(),
            file=_FakeUpload(raw),
            user_id=uid,
            session_id=sid,
            purpose="diary",
            tz_offset_mins=0,
            locale=None,
            aclaracion=None,
            verified_user_id=uid,
        )
        assert res_fail["analysis_failed"] is True
        assert len(events) == 1 and events[0]["node"] == "vision_scan"

        # Case 3: triage_rejected=True -> NO vision_scan usage event
        events.clear()
        monkeypatch.setattr(dt, "modo", lambda: "guard")

        async def _fake_triage(**kwargs):
            return SimpleNamespace(latency_ms=80, si=lambda _q: 0.05)

        monkeypatch.setattr(dt, "run_triage_async", _fake_triage)
        monkeypatch.setattr(dt, "foto_rechazada", lambda r: True)
        monkeypatch.setattr(
            dt,
            "payload_no_comida",
            lambda: {
                "description": "Foto no relacionada.",
                "is_food": False,
                "triage_rejected": True,
                "calories": 0,
                "protein": 0,
                "carbs": 0,
                "healthy_fats": 0,
            },
        )
        res_triage = await rd.api_diary_upload(
            background_tasks=BackgroundTasks(),
            file=_FakeUpload(raw),
            user_id=uid,
            session_id=sid,
            purpose="diary",
            tz_offset_mins=0,
            locale=None,
            aclaracion=None,
            verified_user_id=uid,
        )
        assert res_triage["is_food"] is False
        assert events == []

        # Case 4: attachment failure in parallel chat mode -> HTTP 503
        monkeypatch.setattr(dt, "modo", lambda: "off")
        monkeypatch.setattr(rd, "create_chat_attachment", lambda *a, **kw: None)
        with pytest.raises(HTTPException) as exc:
            await rd.api_diary_upload(
                background_tasks=BackgroundTasks(),
                file=_FakeUpload(raw),
                user_id=uid,
                session_id=sid,
                purpose="chat",
                tz_offset_mins=0,
                locale=None,
                aclaracion=None,
                verified_user_id=uid,
            )
        assert exc.value.status_code == 503

    asyncio.run(_run())
