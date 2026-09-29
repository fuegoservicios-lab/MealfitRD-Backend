# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-843 · 2026-09-29] Cobertura del permiso para la IA: cada endpoint que llama a un proveedor lo exige.

Tres anclas que se vigilan entre sí:
  1. La tabla canónica de `docs/consentimiento_ia.md` (fila 428 ⇔ `Depends(requiere_consentimiento_ia)`, fila «suave» ⇔
     `hay_permiso_ia`/`permite_ia`), leída del fuente de cada router.
  2. Las rutas REALES de la app: las que llevan la dependencia en su árbol son exactamente las filas 428 (así un router
     que no se registra, o una ruta que cambia de nombre, se ve).
  3. La lista de la auditoría (§A.1) está entera dentro, y ningún módulo nuevo llama a un proveedor sin clasificar.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import consentimientos as cs

_BACKEND = Path(__file__).resolve().parent.parent
_DOC = _BACKEND / "docs" / "consentimiento_ia.md"
_FILA_RE = re.compile(r"^\| (POST|GET|PATCH|PUT|DELETE) \| (\S+) \| (routers/\S+\.py) \| (\w+) \| (428|suave) \|$",
                      re.M)

# §A.1 de la auditoría 2026-09-29 (fila 4), con las rutas tal como las monta la app. «La cola de generación» es
# `POST /api/plans/generation-runs`. `PATCH /api/profile` va aparte: suave (se guarda y no se traduce).
_AUDITORIA_A1 = {
    ("POST", "/api/plans/analyze"), ("POST", "/api/plans/analyze/stream"), ("POST", "/api/plans/generation-runs"),
    ("POST", "/api/plans/swap-meal"), ("POST", "/api/plans/{plan_id}/regenerate-day"),
    ("POST", "/api/plans/{plan_id}/fix-sodium-day"), ("POST", "/api/plans/recipe/expand"),
    ("POST", "/api/plans/{plan_id}/retry-chunk/{chunk_id}"),
    ("POST", "/api/plans/{plan_id}/chunks/{chunk_id}/regenerate-simplified"),
    ("POST", "/api/chat/stream"), ("POST", "/api/chat"), ("POST", "/api/chat/message"), ("POST", "/api/chat/voz"),
    ("POST", "/api/diary/upload"), ("POST", "/api/diary/consumed/estimate-macros"),
    ("POST", "/api/diary/consumed/estimate-plate"), ("POST", "/api/diary/scan/ajuste-duda"),
    ("POST", "/api/diary/scan/ingrediente"), ("POST", "/api/inventory/photo-scan"), ("POST", "/api/help/chat"),
}


def _filas():
    filas = _FILA_RE.findall(_DOC.read_text(encoding="utf-8"))
    assert len(filas) >= 20, "la tabla canónica de docs/consentimiento_ia.md no se lee"
    return filas


def _funcion(rel: str, nombre: str):
    arbol = ast.parse((_BACKEND / rel).read_text(encoding="utf-8"))
    for n in ast.walk(arbol):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == nombre:
            return n
    raise AssertionError(f"{rel} no define {nombre}")


def _usa_la_dependencia(fn, dependencia: str) -> bool:
    defaults = list(fn.args.defaults) + [d for d in fn.args.kw_defaults if d is not None]
    return any(isinstance(d, ast.Call) and getattr(d.func, "id", "") == "Depends" and d.args
               and getattr(d.args[0], "id", "") == dependencia for d in defaults)


# ═════════════════════════════════════════════ 1. la tabla ⇔ el fuente de los routers
@pytest.mark.parametrize("metodo, ruta, fichero, funcion, tipo", _FILA_RE.findall(_DOC.read_text(encoding="utf-8")))
def test_cada_fila_de_la_tabla_esta_en_su_endpoint(metodo, ruta, fichero, funcion, tipo):
    fn = _funcion(fichero, funcion)
    if tipo == "428":
        assert _usa_la_dependencia(fn, "requiere_consentimiento_ia"), (
            f"{metodo} {ruta} ({fichero}::{funcion}) no lleva Depends(requiere_consentimiento_ia)")
    else:
        fuente = ast.get_source_segment((_BACKEND / fichero).read_text(encoding="utf-8"), fn)
        assert _usa_la_dependencia(fn, "hay_permiso_ia") or "permite_ia(" in fuente or "permite_ia," in fuente, (
            f"{metodo} {ruta} es suave pero no consulta el permiso")


def test_todo_endpoint_con_la_dependencia_esta_en_la_tabla():
    en_tabla = {(f, fn) for _, _, f, fn, _ in _filas()}
    sueltos = []
    for f in sorted((_BACKEND / "routers").glob("*.py")):
        rel = f"routers/{f.name}"
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    _usa_la_dependencia(n, "requiere_consentimiento_ia") or _usa_la_dependencia(n, "hay_permiso_ia")):
                if (rel, n.name) not in en_tabla:
                    sueltos.append(f"{rel}::{n.name}")
    assert not sueltos, f"endpoints con el permiso que faltan en docs/consentimiento_ia.md: {sueltos}"


# ═════════════════════════════════════════════ 2. las rutas reales de la app
def _usa(dependant, objetivo) -> bool:
    return any(d.call is objetivo or _usa(d, objetivo) for d in dependant.dependencies)


@pytest.fixture(scope="module")
def rutas():
    import app as app_module
    from fastapi.routing import APIRoute
    return [r for r in app_module.app.routes if isinstance(r, APIRoute)]


def test_las_rutas_con_el_428_son_exactamente_las_de_la_tabla(rutas):
    reales = {(m, r.path) for r in rutas if _usa(r.dependant, cs.requiere_consentimiento_ia) for m in r.methods}
    tabla = {(m, p) for m, p, _, _, t in _filas() if t == "428"}
    assert reales == tabla, f"solo en la app: {sorted(reales - tabla)} · solo en la tabla: {sorted(tabla - reales)}"


def test_las_suaves_consultan_el_permiso_en_la_app(rutas):
    suaves = {(m, r.path) for r in rutas if _usa(r.dependant, cs.hay_permiso_ia) for m in r.methods}
    assert {("POST", "/api/i18n/textos"), ("POST", "/api/plans/guest-display")} <= suaves


def test_la_lista_de_la_auditoria_esta_entera(rutas):
    reales = {(m, r.path) for r in rutas if _usa(r.dependant, cs.requiere_consentimiento_ia) for m in r.methods}
    assert _AUDITORIA_A1 <= reales, f"sin el permiso: {sorted(_AUDITORIA_A1 - reales)}"


def test_consents_esta_montado_y_exento_de_cuota(rutas):
    import auth
    propias = {(m, r.path): r for r in rutas if r.path.startswith("/api/consents") for m in r.methods}
    assert set(propias) == {("GET", "/api/consents"), ("POST", "/api/consents"), ("POST", "/api/consents/withdraw"),
                            ("POST", "/api/consents/guest")}
    for r in propias.values():
        assert not _usa(r.dependant, auth.verify_api_quota) and not _usa(r.dependant, cs.requiere_consentimiento_ia)


# ═════════════════════════════════════════════ 3. el cerco: quién llama a un proveedor
_PROVEEDOR_RE = re.compile(r"ChatGLM\(|build_chat_llm\(|ChatOpenAI\(|cohere\.ClientV2\(|OpenAIEmbeddings\(|"
                           r"generativelanguage\.googleapis\.com")

#: Cada módulo que construye un cliente de IA (o llama a la API de Google) y por dónde pasa el permiso. El inventario
#: línea a línea está en el informe del lote; aquí se vigila que no aparezca uno nuevo SIN clasificar.
_CERCO = {
    "agent.py": "turno del chat (/api/chat/*, 428); swap_meal por /swap-meal, /regenerate-day, /fix-sodium-day",
    "ai_helpers.py": "/recipe/expand (428); título (services._titulo_del_plan); retrospectiva (permite_ia)",
    "ajuste_de_duda.py": "/api/diary/scan/ajuste-duda (428)",
    "coach_voz.py": "/api/chat/voz (428)",
    "cron_tasks.py": "sonda «ping» del worker: sin datos, y tras la recogida con permiso",
    "dreaming.py": "consolidate_user (permite_ia)",
    "embeddings_provider.py": "cliente de Cohere: lo usan sitios ya protegidos (chat, hechos, Dreaming, proactivo…)",
    "etiqueta_web.py": "tool del coach (main, lote 767): dentro del turno del chat (428)",
    "fact_extractor.py": "async_extract_and_save_facts y la cola (permite_ia)",
    "graph_orchestrator.py": "el pipeline del plan: endpoints de planes (428) y la recogida (SQL)",
    "ingrediente_corregido.py": "/api/diary/scan/ingrediente (428)",
    "llm_provider.py": "definición de los clientes",
    "memory_manager.py": "resumen del chat: tras un turno o una generación (428)",
    "plan_display_i18n.py": "enrich_plan_display (permite_ia + claim de plan_jobs); invitado: /guest-display (suave)",
    "plato_descrito.py": "/api/diary/consumed/estimate-plate (428)",
    "proactive_agent.py": "run_proactive_checks y handle_nudge_response (permite_ia)",
    "routers/diary.py": "/api/diary/consumed/estimate-macros (428)",
    "routers/help_chat.py": "/api/help/chat (428)",
    "sentiment_classifier.py": "turno del chat (428)",
    "tools.py": "tools del coach: dentro del turno del chat (428)",
    "tools_medical.py": "revisor clínico dentro del pipeline del plan",
    "traduccion_para_mostrar.py": "/i18n/textos (suave), /diary/upload y /regenerate-day (428)",
    "vision_agent.py": "/api/diary/upload y /api/inventory/photo-scan (428)",
}


def test_ningun_modulo_nuevo_llama_a_un_proveedor_sin_clasificar():
    encontrados = set()
    for f in list(_BACKEND.glob("*.py")) + list((_BACKEND / "routers").glob("*.py")):
        rel = f.name if f.parent == _BACKEND else f"routers/{f.name}"
        if _PROVEEDOR_RE.search(f.read_text(encoding="utf-8")):
            encontrados.add(rel)
    nuevos = sorted(encontrados - set(_CERCO) - {"consentimientos.py"})
    assert not nuevos, (
        f"{nuevos} llaman a un proveedor de IA y no están clasificados. Protégelos (endpoint con "
        "Depends(requiere_consentimiento_ia) o `permite_ia(user_id, ...)` antes de la llamada en segundo plano), "
        "añádelos a docs/consentimiento_ia.md y a `_CERCO`.")


def test_los_sitios_de_segundo_plano_consultan_el_permiso():
    anclas = {
        "fact_extractor.py": ['permite_ia(user_id, "extraccion_de_hechos")', 'permite_ia(user_id, "cola_de_hechos")'],
        "dreaming.py": ['permite_ia(user_id, "dreaming")'],
        "plan_display_i18n.py": ['permite_ia(user_id, "traduccion_del_plan")'],
        "proactive_agent.py": ['permite_ia(user_id, "coach_proactivo")', 'permite_ia(user_id, "respuesta_a_aviso")',
                               'permite_ia(user_id, "jit_semana_2")'],
        "services.py": ['permite_ia(user_id, "titulo_del_plan")'],
        "cron_tasks.py": ['fragmento_sql_permiso("q1.user_id")', 'fragmento_sql_permiso("plan_chunk_queue.user_id")',
                          'fragmento_sql_permiso("q.user_id")', 'permite_ia(user_id, "retrospectiva_semanal")',
                          'permite_ia(user_id, "aprendizaje_del_bloque")'],
        "plan_jobs.py": ['condicion_sql_permiso("j.user_id")'],
    }
    for fichero, trozos in anclas.items():
        src = (_BACKEND / fichero).read_text(encoding="utf-8")
        for t in trozos:
            assert t in src, f"{fichero}: falta el gate `{t}`"
