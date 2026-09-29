# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-843 · ronda de arreglo 1 · 2026-09-29] Los embeddings desde caminos SIN IA.

`shopping_calculator.normalize_name` manda el nombre de un alimento a Cohere en su intento 6, y a él se llega desde la
lista de compras, la Nevera, el diario y sus crons: caminos que no pasan por un endpoint de IA (sin 428). Cada punto de
entrada lleva la MARCA de su usuario (`consentimientos.embeddings_de_usuario`, `por_usuario` en los bucles de los
crons, la dependencia `embeddings_de_la_peticion` en los endpoints) y sin su permiso el intento 6 se salta: quedan los
intentos 1-5. Sin marca = permitido (el pipeline y el worker ya van filtrados antes). Base FALSA en todo.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
from fastapi import BackgroundTasks, Depends, FastAPI
from fastapi.testclient import TestClient

import consentimientos as cs
import shopping_calculator as sc
from auth import get_verified_user_id

_BACKEND = Path(__file__).resolve().parent.parent
_DOC = _BACKEND / "docs" / "consentimiento_ia.md"
CON = "11111111-2222-4333-8444-555555555555"   # permiso vigente
SIN = "22222222-3333-4444-8555-666666666666"   # nunca lo dio
RET = "33333333-4444-4555-8666-777777777777"   # lo retiró
NOMBRE = "Salsa secreta de la abuela Chucha"
_MARCA_RE = re.compile(r"^\| (POST|GET|PATCH|PUT|DELETE) \| (\S+) \| (\S+\.py) \| (\w+) \| marca \|$", re.M)


def _fila(uid):
    if uid == CON:
        return {"ai_consent_version": cs.AI_CONSENT_VERSION, "ai_consent_at": 1, "ai_cn_transfer_at": 1,
                "ai_consent_revoked_at": None}
    if uid == RET:
        return {"ai_consent_version": cs.AI_CONSENT_VERSION, "ai_consent_at": 1, "ai_cn_transfer_at": 1,
                "ai_consent_revoked_at": 2}
    return None


@pytest.fixture
def base(monkeypatch):
    """`block` y la base falsa: cada lectura del permiso queda anotada."""
    lecturas = []
    monkeypatch.setattr(cs, "_leer_fila", lambda uid: lecturas.append(uid) or _fila(uid))
    monkeypatch.setenv("MEALFIT_AI_CONSENT_GATE", "block")
    return lecturas


class _Cohere:
    def __init__(self):
        self.consultas = []

    def embed_query(self, texto):
        self.consultas.append(texto)
        return [1.0, 0.0]


@pytest.fixture
def cohere(monkeypatch):
    """Un catálogo de una fila y un Cohere falso: un nombre que los intentos 1-5 no resuelven llega al 6."""
    cliente = _Cohere()
    monkeypatch.setattr(sc, "get_master_ingredients",
                        lambda: [{"name": "Tomate", "category": "Vegetales", "aliases": ["tomates"]}])
    monkeypatch.setattr(sc, "get_semantic_cache", lambda *a, **k: {
        "embeddings_client": cliente, "vectors": [[1.0, 0.0]], "master_list": [{"name": "Tomate"}]})
    monkeypatch.setattr(sc, "_gemini_call_with_retry", lambda fn, *a, _label="", **k: fn(*a, **k))
    sc.invalidate_master_cache()
    yield cliente
    sc.invalidate_master_cache()


# ═════════════════════════════════════════════ 1. el intento 6 y la marca
def test_sin_marca_el_intento_6_llama_a_cohere_y_no_lee_la_base(cohere, base):
    assert sc.normalize_name(NOMBRE) == "Tomate"
    assert len(cohere.consultas) == 1 and base == []


@pytest.mark.parametrize("uid", [SIN, RET])
def test_con_la_marca_de_quien_no_tiene_permiso_el_nombre_no_sale(cohere, base, uid):
    with cs.embeddings_de_usuario(uid, "prueba"):
        r = sc.normalize_name(NOMBRE)
    assert cohere.consultas == [] and r != "Tomate", "sin permiso: quedan los intentos 1-5"
    assert base == [uid]


def test_con_la_marca_de_quien_si_tiene_permiso_sale(cohere, base):
    with cs.embeddings_de_usuario(CON, "prueba"):
        assert sc.normalize_name(NOMBRE) == "Tomate"
    assert len(cohere.consultas) == 1 and base == [CON]


def test_la_decision_se_toma_una_vez_por_marca(cohere, base):
    with cs.embeddings_de_usuario(SIN, "prueba"):
        for n in (NOMBRE, "Mermelada de guayaba casera", "Chenchén de maíz morado"):
            sc.normalize_name(n)
    assert cohere.consultas == [] and base == [SIN], "una lectura por clave primaria, no una por nombre"


def test_modo_log_solo_frena_la_retirada_y_off_no_lee_nada(cohere, base, monkeypatch):
    monkeypatch.setenv("MEALFIT_AI_CONSENT_GATE", "log")
    with cs.embeddings_de_usuario(SIN):
        assert sc.normalize_name(NOMBRE) == "Tomate"
    with cs.embeddings_de_usuario(RET):
        assert sc.normalize_name(NOMBRE) != "Tomate"
    monkeypatch.setenv("MEALFIT_AI_CONSENT_GATE", "off")
    base.clear()
    with cs.embeddings_de_usuario(RET):
        assert sc.normalize_name(NOMBRE) == "Tomate"
    assert base == []


def test_base_ilegible_es_no_en_block(cohere, monkeypatch):
    monkeypatch.setenv("MEALFIT_AI_CONSENT_GATE", "block")

    def _cae(uid):
        raise RuntimeError("Neon caído")

    monkeypatch.setattr(cs, "_leer_fila", _cae)
    with cs.embeddings_de_usuario(CON):
        assert sc.normalize_name(NOMBRE) != "Tomate"
    assert cohere.consultas == []


def test_el_intento_6_consulta_la_marca_antes_de_cohere():
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    i = src.index("Intento 6: Búsqueda de Similitud Semántica")
    ventana = src[i - 700:i + 500]
    assert '_cs843 = __import__("sys").modules.get("consentimientos")' in ventana
    assert "if len(n) > 3 and (_cs843 is None or _cs843.embeddings_permitidos()):" in ventana
    assert ventana.index("embeddings_permitidos()") < ventana.index("get_semantic_cache()")


# ═════════════════════════════════════════════ 2. la marca: bloque, bucle y petición
def test_la_marca_se_quita_al_salir_y_respeta_la_de_fuera(base):
    assert cs._EMBEDDINGS_DE.get() is None and cs.embeddings_permitidos() is True
    with cs.embeddings_de_usuario(CON):
        assert cs.embeddings_permitidos() is True
        with cs.embeddings_de_usuario(SIN):
            assert cs.embeddings_permitidos() is False
        assert cs.embeddings_permitidos() is True
    assert cs._EMBEDDINGS_DE.get() is None


def test_por_usuario_marca_cada_vuelta_y_limpia_al_terminar(base):
    filas = [{"user_id": CON}, {"user_id": SIN}, {"user_id": None}, "no es un dict"]
    vistos = [cs.embeddings_permitidos() for _ in cs.por_usuario(filas, donde="prueba")]
    assert vistos == [True, False, False, False], "sin cuenta en `block`: no"
    assert cs._EMBEDDINGS_DE.get() is None
    for fila in cs.por_usuario(filas):
        if fila["user_id"] == SIN:
            assert cs.embeddings_permitidos() is False
            break
    assert cs._EMBEDDINGS_DE.get() is None, "un `break` tampoco deja la marca puesta"
    assert list(cs.por_usuario(None)) == []


@pytest.fixture
def app_marcada(cohere, base):
    quien = {"uid": SIN}
    vistos = []
    app = FastAPI()
    app.dependency_overrides[get_verified_user_id] = lambda: quien["uid"]

    @app.post("/sincrono")
    def _sincrono(bt: BackgroundTasks, _emb: None = Depends(cs.embeddings_de_la_peticion)):
        bt.add_task(lambda: vistos.append(cs.embeddings_permitidos()))
        return {"r": sc.normalize_name(NOMBRE)}

    @app.post("/asincrono")
    async def _asincrono(_emb: None = Depends(cs.embeddings_de_la_peticion)):
        return {"r": await asyncio.to_thread(sc.normalize_name, NOMBRE)}

    @app.post("/sin_marca")
    def _sin_marca():
        return {"r": sc.normalize_name(NOMBRE)}

    return TestClient(app), quien, vistos


def test_la_dependencia_marca_la_peticion_entera(app_marcada, cohere):
    cliente, quien, vistos = app_marcada
    for ruta in ("/sincrono", "/asincrono"):
        assert cliente.post(ruta).json()["r"] != "Tomate", ruta
    assert cohere.consultas == [] and vistos == [False], "la tarea de fondo de la petición también lleva la marca"
    assert cliente.post("/sin_marca").json()["r"] == "Tomate", "la marca no se queda pegada entre peticiones"
    quien["uid"] = CON
    for ruta in ("/sincrono", "/asincrono"):
        assert cliente.post(ruta).json()["r"] == "Tomate", ruta
    assert vistos == [False, True]


def test_el_invitado_de_la_peticion_depende_de_su_cabecera(app_marcada, cohere, base):
    cliente, quien, _ = app_marcada
    quien["uid"] = None
    assert cliente.post("/sincrono").json()["r"] != "Tomate"
    r = cliente.post("/sincrono", headers={cs.CABECERA_INVITADO: cs.AI_CONSENT_VERSION}).json()["r"]
    assert r == "Tomate" and base == [], "el invitado se decide por la cabecera, sin leer la base"


# ═════════════════════════════════════════════ 3. los endpoints reales: la tabla ⇔ la app
def _usa(dependant, objetivo) -> bool:
    return any(d.call is objetivo or _usa(d, objetivo) for d in dependant.dependencies)


def test_las_rutas_con_la_marca_son_exactamente_las_de_la_tabla():
    import app as app_module
    from fastapi.routing import APIRoute
    reales = {(m, r.path) for r in app_module.app.routes if isinstance(r, APIRoute)
              and _usa(r.dependant, cs.embeddings_de_la_peticion) for m in r.methods}
    filas = _MARCA_RE.findall(_DOC.read_text(encoding="utf-8"))
    tabla = {(m, p) for m, p, _, _ in filas}
    assert len(tabla) >= 12, "la tabla de la marca de docs/consentimiento_ia.md no se lee"
    assert reales == tabla, f"solo en la app: {sorted(reales - tabla)} · solo en la tabla: {sorted(tabla - reales)}"
    import ast
    for _, _, fichero, funcion in filas:
        fn = next(n for n in ast.walk(ast.parse((_BACKEND / fichero).read_text(encoding="utf-8")))
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == funcion)
        seg = ast.get_source_segment((_BACKEND / fichero).read_text(encoding="utf-8"), fn)
        assert "embeddings_de_la_peticion)" in seg.split('"""')[0], f"{fichero}::{funcion} sin la marca"


@pytest.mark.parametrize("metodo, ruta", [
    ("POST", "/api/plans/recalculate-shopping-list"), ("POST", "/api/plans/restock"),
    ("POST", "/api/diary/consumed"), ("POST", "/api/diary/consumed/manual"), ("POST", "/api/diary/consumed/repeat"),
    ("POST", "/api/diary/consumed-from-plan"), ("POST", "/api/diary/consumed-from-plan/preview"),
])
def test_los_caminos_de_la_revision_llevan_la_marca(metodo, ruta):
    filas = {(m, p) for m, p, _, _ in _MARCA_RE.findall(_DOC.read_text(encoding="utf-8"))}
    assert (metodo, ruta) in filas


def test_restock_cycle_solo_lo_llama_una_tool_del_coach():
    """`restock_cycle.purchase_item_name` normaliza el nombre (intento 6), pero su ÚNICO llamador es la tool del coach
    `mark_shopping_list_purchased`, dentro del turno del chat (428). Si nace otro llamador, necesita su marca."""
    llamadores = []
    for f in list(_BACKEND.glob("*.py")) + list((_BACKEND / "routers").glob("*.py")):
        if f.name == "restock_cycle.py":
            continue
        if re.search(r"\brestock_cycle\b", f.read_text(encoding="utf-8")):
            llamadores.append(f.name)
    assert llamadores == ["tools.py"], llamadores


# ═════════════════════════════════════════════ 4. los crons y el worker de plan_jobs: por usuario, dentro del bucle
@pytest.fixture
def cron(monkeypatch, base):
    import cron_tasks as ct
    monkeypatch.setattr(ct, "execute_sql_write", lambda *a, **k: True)
    return ct


def test_los_descuentos_fallidos_se_reintentan_con_la_marca_de_cada_usuario(cron, monkeypatch):
    vistos = []
    filas = [{"id": 1, "user_id": CON, "ingredients": [{"item": "2 tazas de salsa de la abuela"}], "attempts": 0},
             {"id": 2, "user_id": SIN, "ingredients": [{"item": "1 frasco de salsa de la abuela"}], "attempts": 0}]
    monkeypatch.setattr(cron, "execute_sql_query", lambda *a, **k: filas)

    def _parse(texto):
        vistos.append(cs.embeddings_permitidos())
        raise RuntimeError("corta aquí")

    monkeypatch.setattr(sc, "_parse_quantity", _parse)
    cron._process_failed_inventory_deductions_queue()
    assert vistos == [True, False] and cs._EMBEDDINGS_DE.get() is None


def test_las_listas_pendientes_se_calculan_con_la_marca_de_cada_usuario(cron, monkeypatch):
    vistos = []
    planes = [{"id": "p1", "user_id": CON, "plan_data": {"days": []}},
              {"id": "p2", "user_id": SIN, "plan_data": {"days": []}}]
    monkeypatch.setattr(cron, "execute_sql_query",
                        lambda q, *a, **k: planes if "partial_no_shopping" in q else None)
    monkeypatch.setattr(sc, "fetch_inventory_and_consumed_for_plan", lambda *a, **k: ([], []))

    def _delta(user_id, *a, **k):
        vistos.append((user_id, cs.embeddings_permitidos()))
        raise RuntimeError("corta aquí")

    monkeypatch.setattr(sc, "get_shopping_list_delta", _delta)
    cron._process_pending_shopping_lists()
    assert vistos == [(CON, True), (SIN, False)] and cs._EMBEDDINGS_DE.get() is None


def test_la_coherencia_diaria_evalua_cada_plan_con_la_marca_de_su_dueno(cron, monkeypatch):
    import db_core
    vistos = []
    duenos = [CON, SIN, CON, SIN, CON]
    planes = [{"id": f"p{i}", "user_id": u, "plan_data": {"days": []}} for i, u in enumerate(duenos)]
    monkeypatch.setattr(db_core, "connection_pool", object())
    monkeypatch.setattr(cron, "execute_sql_query", lambda q, *a, **k: planes if "FROM public.meal_plans" in q else [])

    def _guard(plan_data, **k):
        vistos.append((k.get("plan_id_hint"), cs.embeddings_permitidos()))
        return [], set()

    monkeypatch.setattr(sc, "run_shopping_coherence_guard_and_append_history", _guard)
    cron._shopping_coherence_alert_job()
    assert vistos == [(f"p{i}", u == CON) for i, u in enumerate(duenos)]
    assert cs._EMBEDDINGS_DE.get() is None


def test_plan_jobs_consume_cada_trabajo_con_la_marca_de_su_dueno(monkeypatch, base):
    import plan_jobs as pj
    vistos = []
    trabajos = [{"id": "j1", "job_type": "shopping_projection", "user_id": SIN, "attempts": 1},
                {"id": "j2", "job_type": "shopping_projection", "user_id": CON, "attempts": 1}]
    monkeypatch.setattr(pj, "plan_jobs_enabled", lambda: True)
    monkeypatch.setattr(pj, "reclaim_stale_processing", lambda: 0)
    monkeypatch.setattr(pj, "enabled_consumers", lambda: ["shopping_projection"])
    monkeypatch.setattr(pj, "claim_plan_jobs", lambda *a, **k: list(trabajos))
    monkeypatch.setattr(pj, "heartbeat_plan_job", lambda *a, **k: True)
    monkeypatch.setattr(pj, "finish_plan_job", lambda *a, **k: True)
    monkeypatch.setattr(pj, "_emit_metric", lambda *a, **k: None)
    monkeypatch.setitem(pj.CONSUMERS, "shopping_projection",
                        lambda job: vistos.append((job["id"], cs.embeddings_permitidos())) or ("done", None, {}))
    pj.process_plan_jobs()
    assert vistos == [("j1", False), ("j2", True)] and cs._EMBEDDINGS_DE.get() is None


def test_los_tres_crons_marcan_por_usuario_dentro_del_bucle():
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    for funcion, bucle in (
        ("def _process_pending_shopping_lists(", 'for p in __import__("consentimientos").por_usuario(plans,'),
        ("def _shopping_coherence_alert_job(", 'for plan_record in __import__("consentimientos").por_usuario(plans,'),
        ("def _process_failed_inventory_deductions_queue(",
         'for row in __import__("consentimientos").por_usuario(rows,'),
    ):
        i = src.index(funcion)
        assert bucle in src[i:src.index("\ndef ", i + 10)], funcion
