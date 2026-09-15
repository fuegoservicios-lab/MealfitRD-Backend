"""[P1-CHAT-FACTS-AUDIT · 2026-09-14] Auditoría del extractor de hechos del coach.

Cada test FALLA contra el código previo y pasa con el arreglo. Todo mockeado: el
`.env` apunta a PRODUCCIÓN, así que ningún test llama al LLM real ni escribe en la
base (se sustituyen las funciones de `db_facts`/`fact_extractor` en su namespace).

Hallazgos cubiertos:
  1. Embedding vacío ⇒ el hecho ya no se descarta en silencio (clínico se guarda
     sin embedding; no clínico ⇒ el mensaje se re-encola).
  2. Un hecho no clínico no retira ni absorbe uno clínico; una fusión que absorbe un
     hecho clínico conserva la categoría; un id que no se le mostró al LLM no se toca.
  3. `extract_facts` distingue error (`None`) de vacío (`[]`); la cola no borra los
     ítems fallidos (tope por antigüedad + espaciado).
  4. El lock se libera solo si se adquirió, y solo el propio (token).
  5. `delete_user_fact` filtra `AND user_id = %s` (I2).
  6. Router y extractor registran su gasto en `llm_usage_events`.
  7. Guarda de invitado antes de gastar LLM.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND = Path(__file__).resolve().parent.parent

UID = "11111111-1111-1111-1111-111111111111"

import fact_extractor as _fe_mod  # noqa: E402

# Referencia a la función real, capturada antes de que el fixture la sustituya.
_ORIG_INVOKE = _fe_mod._invoke_with_shadow


@pytest.fixture
def fe(monkeypatch):
    import fact_extractor as _fe

    # Red de seguridad: nada de DB ni LLM real en ningún test de este fichero.
    def _boom(*_a, **_k):
        raise AssertionError("llamada no mockeada a DB/LLM")

    for name in (
        "save_user_fact", "search_user_facts_hybrid", "delete_user_fact",
        "acquire_fact_lock", "release_fact_lock", "enqueue_pending_fact",
        "dequeue_pending_facts", "delete_pending_facts",
    ):
        monkeypatch.setattr(_fe, name, _boom)
    monkeypatch.setattr(_fe, "_invoke_with_shadow", _boom)
    # getattr: el fixture debe poder correr contra el código previo (verificación
    # «falla antes / pasa después») sin convertir cada test en un ERROR de montaje.
    getattr(_fe, "_PENDING_RETRY_NOT_BEFORE", {}).clear()
    yield _fe
    getattr(_fe, "_PENDING_RETRY_NOT_BEFORE", {}).clear()


def _item(text, category):
    return {"fact": text, "metadata": {"category": category, "ingrediente": "", "intensidad": 3}}


class _Saves:
    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def __call__(self, user_id, fact, embedding, metadata=None):
        self.calls.append({"user_id": user_id, "fact": fact, "embedding": embedding, "metadata": metadata})
        return [{"id": f"new-{len(self.calls)}"}] if self.ok else None


# ─────────────────────────────────────────────────────────────── 1. embedding vacío
def test_1_embedding_vacio_hecho_clinico_se_guarda_sin_embedding(fe, monkeypatch):
    saves = _Saves()
    monkeypatch.setattr(fe, "get_embedding", lambda *a, **k: [])
    monkeypatch.setattr(fe, "_embeddings_enabled", lambda: True, raising=False)
    monkeypatch.setattr(fe, "save_user_fact", saves)

    complete = fe._run_fact_pipeline(UID, [_item("El usuario es alérgico al maní", "alergia")])

    assert [c["fact"] for c in saves.calls] == ["El usuario es alérgico al maní"]
    assert saves.calls[0]["embedding"] is None
    assert saves.calls[0]["metadata"]["category"] == "alergia"
    assert complete is True


def test_1_embedding_vacio_no_clinico_marca_incompleto_para_reintento(fe, monkeypatch):
    saves = _Saves()
    monkeypatch.setattr(fe, "get_embedding", lambda *a, **k: [])
    monkeypatch.setattr(fe, "_embeddings_enabled", lambda: True, raising=False)
    monkeypatch.setattr(fe, "save_user_fact", saves)

    complete = fe._run_fact_pipeline(UID, [_item("Le gusta el mangú", "preferencia")])

    assert saves.calls == []
    assert complete is False


def test_1_proveedor_desactivado_guarda_todo_sin_embedding(fe, monkeypatch):
    saves = _Saves()
    monkeypatch.setattr(fe, "get_embedding", lambda *a, **k: [])
    monkeypatch.setattr(fe, "_embeddings_enabled", lambda: False, raising=False)
    monkeypatch.setattr(fe, "save_user_fact", saves)

    complete = fe._run_fact_pipeline(UID, [_item("Le gusta el mangú", "preferencia")])

    assert [c["embedding"] for c in saves.calls] == [None]
    assert complete is True  # reintentar no cambiaría nada


def test_1_async_reencola_si_quedaron_hechos_sin_guardar(fe, monkeypatch):
    enq = []
    monkeypatch.setattr(fe, "should_extract_facts", lambda *a, **k: True)
    monkeypatch.setattr(fe, "acquire_fact_lock", lambda uid: "tok")
    monkeypatch.setattr(fe, "release_fact_lock", lambda *a, **k: None)
    monkeypatch.setattr(fe, "extract_facts", lambda *a, **k: [_item("x", "preferencia")])
    monkeypatch.setattr(fe, "_run_fact_pipeline", lambda *a, **k: False)
    monkeypatch.setattr(fe, "enqueue_pending_fact", lambda *a: enq.append(a))

    fe.async_extract_and_save_facts(UID, "me gusta el mangú", "")

    assert enq == [(UID, "me gusta el mangú", "")]


# ─────────────────────────────────────────── 2. protección clínica en el pipeline
_ALERGIA = {"id": "a1", "fact": "El usuario es alérgico al maní", "metadata": {"category": "alergia", "ingrediente": "maní"}}
_PREF = {"id": "p1", "fact": "Le gusta el pollo", "metadata": {"category": "preferencia"}}


def _setup_pipeline(fe, monkeypatch, similar, llm_result):
    saves, deletes = _Saves(), []
    monkeypatch.setattr(fe, "get_embedding", lambda *a, **k: [0.1, 0.2])
    monkeypatch.setattr(fe, "search_user_facts_hybrid", lambda *a, **k: list(similar))
    monkeypatch.setattr(fe, "_invoke_with_shadow", lambda **k: llm_result)
    monkeypatch.setattr(fe, "save_user_fact", saves)
    monkeypatch.setattr(fe, "delete_user_fact", lambda *a, **k: deletes.append(a) or [{"id": a[0]}])
    return saves, deletes


def _result(contradictions=(), merges=()):
    return SimpleNamespace(contradictions=list(contradictions), merges=list(merges))


def test_2_preferencia_no_retira_una_alergia(fe, monkeypatch):
    res = _result(contradictions=[SimpleNamespace(new_fact="Ahora le encanta el maní", ids_to_delete=["a1"])])
    saves, deletes = _setup_pipeline(fe, monkeypatch, [_ALERGIA], res)

    fe._run_fact_pipeline(UID, [_item("Ahora le encanta el maní", "preferencia")])

    assert deletes == []
    assert [c["fact"] for c in saves.calls] == ["Ahora le encanta el maní"]


def test_2_contradiccion_no_clinica_si_retira_y_filtra_por_user_id(fe, monkeypatch):
    res = _result(contradictions=[SimpleNamespace(new_fact="Ya no le gusta el pollo", ids_to_delete=["p1"])])
    _saves, deletes = _setup_pipeline(fe, monkeypatch, [_PREF], res)

    fe._run_fact_pipeline(UID, [_item("Ya no le gusta el pollo", "rechazo")])

    assert deletes == [("p1", UID)]


def test_2_id_no_mostrado_al_llm_no_se_toca(fe, monkeypatch):
    res = _result(contradictions=[SimpleNamespace(new_fact="Ya no le gusta el pollo", ids_to_delete=["p1", "ajeno-9"])])
    _saves, deletes = _setup_pipeline(fe, monkeypatch, [_PREF], res)

    fe._run_fact_pipeline(UID, [_item("Ya no le gusta el pollo", "rechazo")])

    assert deletes == [("p1", UID)]


def test_2_fusion_que_absorberia_una_alergia_desde_preferencia_se_descarta(fe, monkeypatch):
    res = _result(merges=[SimpleNamespace(
        merged_fact="Le gusta el maní tostado", ids_to_delete=["a1"], skip_new_fact="Le gusta el maní tostado",
    )])
    saves, deletes = _setup_pipeline(fe, monkeypatch, [_ALERGIA], res)

    fe._run_fact_pipeline(UID, [_item("Le gusta el maní tostado", "preferencia")])

    assert deletes == []
    # el hecho nuevo se guarda por separado (no se considera absorbido)
    assert [c["fact"] for c in saves.calls] == ["Le gusta el maní tostado"]


def test_2_fusion_que_absorbe_alergia_conserva_la_categoria_clinica(fe, monkeypatch):
    # A (alergia) tiene similares; B (alergia) no. La fusión dice absorber a1 y saltar B.
    # Antes: la metadata del fusionado salía `{}` (se buscaba B solo entre los que
    # tenían similares) → la alergia fusionada perdía `category='alergia'`.
    calls = {"n": 0}

    def _search(*a, **k):
        calls["n"] += 1
        return [dict(_ALERGIA)] if calls["n"] == 1 else []

    res = _result(merges=[SimpleNamespace(
        merged_fact="Es alérgico al maní y a los cacahuetes", ids_to_delete=["a1"],
        skip_new_fact="Tiene alergia a los cacahuetes",
    )])
    saves, deletes = _setup_pipeline(fe, monkeypatch, [], res)
    monkeypatch.setattr(fe, "search_user_facts_hybrid", _search)

    fe._run_fact_pipeline(UID, [
        _item("Es alérgico al maní", "alergia"),
        _item("Tiene alergia a los cacahuetes", "alergia"),
    ])

    merged = [c for c in saves.calls if c["fact"] == "Es alérgico al maní y a los cacahuetes"]
    assert merged and merged[0]["metadata"].get("category") == "alergia"
    assert deletes == [("a1", UID)]


def test_2_fusion_fallida_conserva_los_hechos_absorbidos(fe, monkeypatch):
    res = _result(merges=[SimpleNamespace(
        merged_fact="Le encanta el pollo asado", ids_to_delete=["p1"], skip_new_fact="Ama el pollo asado",
    )])
    _saves, deletes = _setup_pipeline(fe, monkeypatch, [_PREF], res)
    monkeypatch.setattr(fe, "save_user_fact", _Saves(ok=False))

    complete = fe._run_fact_pipeline(UID, [_item("Ama el pollo asado", "preferencia")])

    assert deletes == []  # antes: se soft-borraba ANTES de intentar guardar
    assert complete is False


def test_2_criterio_clinico_es_el_de_dreaming(fe):
    import dreaming
    assert fe._clinical_categories() == tuple(dreaming.CLINICAL_CATEGORIES)


# ─────────────────────────────────── 3. error ≠ vacío + la cola no pierde ítems
def test_3_extract_facts_devuelve_none_si_el_llm_falla(fe, monkeypatch):
    def _raise(**_k):
        raise TimeoutError("proveedor caído")

    monkeypatch.setattr(fe, "_invoke_with_shadow", _raise)
    assert fe.extract_facts("soy alérgico al maní", user_id=UID) is None


def test_3_async_encola_si_la_extraccion_falla(fe, monkeypatch):
    enq = []
    monkeypatch.setattr(fe, "should_extract_facts", lambda *a, **k: True)
    monkeypatch.setattr(fe, "acquire_fact_lock", lambda uid: "tok")
    monkeypatch.setattr(fe, "release_fact_lock", lambda *a, **k: None)
    monkeypatch.setattr(fe, "extract_facts", lambda *a, **k: None)
    monkeypatch.setattr(fe, "enqueue_pending_fact", lambda *a: enq.append(a))

    fe.async_extract_and_save_facts(UID, "soy alérgico al maní", "")

    assert enq == [(UID, "soy alérgico al maní", "")]


def test_3_process_single_extraction_lanza_si_el_llm_falla(fe, monkeypatch):
    monkeypatch.setattr(fe, "should_extract_facts", lambda *a, **k: True)
    monkeypatch.setattr(fe, "extract_facts", lambda *a, **k: None)
    with pytest.raises(Exception):  # FactExtractionIncomplete; antes: return silencioso
        fe._process_single_extraction(UID, "soy alérgico al maní")


def _queue(fe, monkeypatch, items, process):
    deleted, released = [], []
    monkeypatch.setattr(fe, "acquire_fact_lock", lambda uid: "tok-q")
    monkeypatch.setattr(fe, "release_fact_lock", lambda *a, **k: released.append(a))
    monkeypatch.setattr(fe, "dequeue_pending_facts", lambda uid: [dict(i) for i in items])
    monkeypatch.setattr(fe, "delete_pending_facts", lambda ids: deleted.append(list(ids)))
    monkeypatch.setattr(fe, "_process_single_extraction", process)
    return deleted, released


def test_3_cola_no_borra_items_que_fallaron(fe, monkeypatch):
    now = datetime.now(timezone.utc)
    items = [
        {"id": "ok-1", "message": "me gusta el mangú", "recent_history": "", "created_at": now},
        {"id": "ko-2", "message": "soy alérgico al maní", "recent_history": "", "created_at": now},
    ]

    def _process(uid, msg, hist=""):
        if "alérgico" in msg:
            raise RuntimeError("LLM caído")

    deleted, released = _queue(fe, monkeypatch, items, _process)
    fe.process_pending_queue_sync(UID)

    assert deleted == [["ok-1"]]
    assert released == [(UID, "tok-q")]


def test_3_cola_espacia_el_reintento_de_un_item_fallido(fe, monkeypatch):
    now = datetime.now(timezone.utc)
    items = [{"id": "ko-2", "message": "soy alérgico al maní", "recent_history": "", "created_at": now}]
    calls = []

    def _process(uid, msg, hist=""):
        calls.append(msg)
        raise RuntimeError("LLM caído")

    _queue(fe, monkeypatch, items, _process)
    fe.process_pending_queue_sync(UID)
    fe.process_pending_queue_sync(UID)

    assert len(calls) == 1


def test_3_cola_descarta_items_que_superan_el_tope_de_antiguedad(fe, monkeypatch):
    viejo = datetime.now(timezone.utc) - timedelta(hours=fe._pending_fact_max_age_h() + 1)
    items = [{"id": "viejo-1", "message": "x", "recent_history": "", "created_at": viejo}]

    def _process(*_a, **_k):
        raise AssertionError("no debe procesarse un ítem caducado")

    deleted, _released = _queue(fe, monkeypatch, items, _process)
    fe.process_pending_queue_sync(UID)

    assert deleted == [["viejo-1"]]


# ────────────────────────────────────────────────────────── 4. lock con token
def test_4_no_libera_el_lock_si_el_router_descarta(fe, monkeypatch):
    released = []
    monkeypatch.setattr(fe, "should_extract_facts", lambda *a, **k: False)
    monkeypatch.setattr(fe, "release_fact_lock", lambda *a, **k: released.append(a))

    fe.async_extract_and_save_facts(UID, "hola, gracias", "")

    assert released == []


def test_4_no_libera_el_lock_si_no_lo_consiguio(fe, monkeypatch):
    released, enq = [], []
    monkeypatch.setattr(fe, "should_extract_facts", lambda *a, **k: True)
    monkeypatch.setattr(fe, "acquire_fact_lock", lambda uid: False)
    monkeypatch.setattr(fe, "release_fact_lock", lambda *a, **k: released.append(a))
    monkeypatch.setattr(fe, "enqueue_pending_fact", lambda *a: enq.append(a))
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    fe.async_extract_and_save_facts(UID, "soy alérgico al maní", "")

    assert released == []
    assert len(enq) == 1


def test_4_libera_con_el_token_que_adquirio(fe, monkeypatch):
    released = []
    token = datetime(2026, 9, 14, 12, 0, 0, 123456, tzinfo=timezone.utc)
    monkeypatch.setattr(fe, "should_extract_facts", lambda *a, **k: True)
    monkeypatch.setattr(fe, "acquire_fact_lock", lambda uid: token)
    monkeypatch.setattr(fe, "release_fact_lock", lambda *a, **k: released.append(a))
    monkeypatch.setattr(fe, "extract_facts", lambda *a, **k: [])

    fe.async_extract_and_save_facts(UID, "hola, soy yo", "")

    assert released == [(UID, token)]


@pytest.fixture
def dbf(monkeypatch):
    import db_facts as _dbf

    writes = []

    def _write(sql, params=None, returning=False, **_k):
        writes.append((" ".join(sql.split()), params))
        return [{"id": "x"}] if returning else None

    def _no_query(*_a, **_k):
        raise AssertionError("no debe leer antes de escribir")

    monkeypatch.setattr(_dbf, "connection_pool", object())
    monkeypatch.setattr(_dbf, "execute_sql_write", _write)
    monkeypatch.setattr(_dbf, "execute_sql_query", _no_query)
    monkeypatch.setattr(_dbf, "_invalidate_rag_cache", lambda uid: None)
    _dbf._test_writes = writes
    return _dbf


def test_4_acquire_es_un_update_condicional_y_devuelve_token(dbf):
    token = dbf.acquire_fact_lock(UID)

    assert isinstance(token, datetime)
    sql, params = dbf._test_writes[-1]
    assert "(fact_locked_at IS NULL OR fact_locked_at < %s)" in sql
    assert params[0] == token and params[1] == UID


def test_4_release_con_token_solo_libera_el_lock_propio(dbf):
    token = datetime(2026, 9, 14, 12, 0, 0, 123456, tzinfo=timezone.utc)
    dbf.release_fact_lock(UID, token)

    sql, params = dbf._test_writes[-1]
    assert "WHERE id = %s AND fact_locked_at = %s" in sql
    assert params == (UID, token)


# ────────────────────────────────────────────────── 5. delete_user_fact con I2
def test_5_delete_user_fact_filtra_por_user_id(dbf):
    dbf.delete_user_fact("f-1", UID)

    sql, params = dbf._test_writes[-1]
    assert "WHERE id = %s AND user_id = %s" in sql
    assert params == ("f-1", UID)


def test_5_delete_user_fact_sin_user_id_no_toca_nada(dbf):
    assert dbf.delete_user_fact("f-1", "") is None
    assert dbf._test_writes == []


def test_5_save_user_fact_sin_embedding_inserta_null(dbf):
    dbf.save_user_fact(UID, "alérgico al maní", [], metadata={"category": "alergia"})

    _sql, params = dbf._test_writes[-1]
    assert params[2] is None  # antes: el literal '[]', que el cast a vector rechaza


# ───────────────────────────────────────────── 6. gasto LLM en llm_usage_events
class _FakeLLM:
    def __init__(self, parsed, **_kw):
        self._parsed = parsed

    def with_structured_output(self, schema, include_raw=False, **_kw):
        parsed, raw = self._parsed, SimpleNamespace(
            usage_metadata={"input_tokens": 120, "output_tokens": 30, "input_token_details": {"cache_read": 5}}
        )

        class _Runnable:
            def invoke(self, _prompt):
                if include_raw:
                    return {"raw": raw, "parsed": parsed, "parsing_error": None}
                return parsed

        return _Runnable()


def _capture_usage(monkeypatch):
    import db
    events = []
    monkeypatch.setattr(db, "log_llm_usage_event", lambda **kw: events.append(kw))
    return events


def test_6_router_registra_su_gasto(fe, monkeypatch):
    events = _capture_usage(monkeypatch)
    parsed = fe.RouterResult(has_relevant_info=True, confidence_score=9)
    monkeypatch.setattr(fe, "ChatGLM", lambda **kw: _FakeLLM(parsed, **kw))

    assert fe.should_extract_facts("soy alérgico al maní", user_id=UID) is True
    assert [e["node"] for e in events] == ["fact_extractor_router"]
    assert events[0]["user_id"] == UID
    assert events[0]["input_tokens"] == 120 and events[0]["output_tokens"] == 30


def test_6_extractor_registra_su_gasto(fe, monkeypatch):
    monkeypatch.setattr(fe, "_invoke_with_shadow", _ORIG_INVOKE)  # la real, no el _boom
    events = _capture_usage(monkeypatch)
    parsed = fe.FactsModel(facts=[])
    monkeypatch.setattr(fe, "ChatGLM", lambda **kw: _FakeLLM(parsed, **kw))
    monkeypatch.setattr(fe, "_should_run_shadow", lambda uid: False)

    assert fe.extract_facts("soy alérgico al maní", user_id=UID) == []
    assert [e["node"] for e in events] == ["fact_extractor_extract_facts"]


def test_6_error_de_parseo_se_relanza_no_se_confunde_con_vacio(fe):
    with pytest.raises(ValueError):
        fe._unwrap_structured(
            {"raw": SimpleNamespace(usage_metadata={}), "parsed": None, "parsing_error": ValueError("json roto")},
            model="m", node="n", user_id=None,
        )


# ─────────────────────────────────────────────────────── 7. guarda de invitado
@pytest.mark.parametrize("uid", [None, "", "guest", "GUEST "])
def test_7_invitado_no_gasta_llm(fe, monkeypatch, uid):
    def _no_llm(*_a, **_k):
        raise AssertionError("un invitado no debe llegar al router LLM")

    monkeypatch.setattr(fe, "should_extract_facts", _no_llm)
    fe.async_extract_and_save_facts(uid, "soy alérgico al maní", "")


# ───────────────────────────────────────────────────────────── anclas de texto
def test_anclas_presentes():
    for name in ("fact_extractor.py", "db_facts.py"):
        assert "P1-CHAT-FACTS-AUDIT" in (_BACKEND / name).read_text(encoding="utf-8"), name
