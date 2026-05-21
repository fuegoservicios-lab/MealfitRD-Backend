"""[H2 / P3-CORRELATION-ID · 2026-05-20] Anchor + regression guards para
el bundle de correlation_id end-to-end.

Cubre 3 piezas:
  1. `correlation.py` — módulo con ContextVar + helpers + LogFilter.
  2. `app.py` — instalación del filter + format con `[corr=%(correlation_id)s]`
     + middleware FastAPI con echo de header.
  3. `bg_executor.py::submit_bg_task` — propagación via `contextvars.copy_context()`.

Sin estos 3 puntos sincronizados, los logs pierden la trazabilidad
request → bg task → cron en algún eslabón y vuelven a 30-60min para
debuggear un incident.

Tooltip-anchor: P3-CORRELATION-ID.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CORR_PY = _BACKEND_ROOT / "correlation.py"
_APP_PY = _BACKEND_ROOT / "app.py"
_BG_PY = _BACKEND_ROOT / "bg_executor.py"


# ---------------------------------------------------------------------------
# Anchor presence (P2-HIST-AUDIT-14 cross-link).
# ---------------------------------------------------------------------------


def test_anchor_present_in_correlation_module():
    src = _CORR_PY.read_text(encoding="utf-8")
    assert "P3-CORRELATION-ID" in src, (
        "Falta anchor `P3-CORRELATION-ID` en backend/correlation.py."
    )


def test_anchor_present_in_app():
    src = _APP_PY.read_text(encoding="utf-8")
    assert "P3-CORRELATION-ID" in src, (
        "Falta anchor `P3-CORRELATION-ID` en backend/app.py. Sin él "
        "un futuro reader puede 'limpiar' el middleware o el format string "
        "asumiendo que `[corr=...]` es ruido."
    )


def test_anchor_present_in_bg_executor():
    src = _BG_PY.read_text(encoding="utf-8")
    assert "P3-CORRELATION-ID" in src, (
        "Falta anchor `P3-CORRELATION-ID` en backend/bg_executor.py. "
        "Sin él un refactor podría remover `copy_context()` rompiendo "
        "la propagación request → bg task."
    )


def test_anchor_present_in_test_file():
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P3-CORRELATION-ID" in src


# ---------------------------------------------------------------------------
# Correlation module — contract.
# ---------------------------------------------------------------------------


def test_correlation_module_exports():
    """El módulo debe exponer los 5 símbolos del contrato público."""
    src = _CORR_PY.read_text(encoding="utf-8")
    for name in (
        "_correlation_id",
        "get_correlation_id",
        "set_correlation_id",
        "reset_correlation_id",
        "new_correlation_id",
        "with_correlation_id",
        "CorrelationIdFilter",
        "install_log_filter",
    ):
        assert name in src, f"correlation.py no expone `{name}`"


def test_correlation_uses_contextvar():
    """El backbone debe ser `ContextVar` (no thread-local, no global).
    ContextVar es el único primitive que se propaga a `asyncio.create_task`
    y `asyncio.to_thread` automáticamente."""
    src = _CORR_PY.read_text(encoding="utf-8")
    assert "ContextVar" in src, (
        "correlation.py NO usa ContextVar. Regresión grave — un thread-local "
        "no se propaga a async tasks; un global rompe entre requests."
    )


def test_filter_class_is_logging_filter():
    """`CorrelationIdFilter` debe heredar `logging.Filter` y tener un
    método `filter` que setea `record.correlation_id`."""
    src = _CORR_PY.read_text(encoding="utf-8")
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "CorrelationIdFilter":
            base_names = [
                ast.unparse(b) if hasattr(ast, "unparse") else "" for b in node.bases
            ]
            assert any("Filter" in b for b in base_names), (
                "CorrelationIdFilter no hereda de logging.Filter."
            )
            method_names = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            assert "filter" in method_names, (
                "CorrelationIdFilter no define `filter(record)`."
            )
            found = True
            break
    assert found, "CorrelationIdFilter no encontrado vía AST."


# ---------------------------------------------------------------------------
# app.py — basicConfig format + filter install + middleware.
# ---------------------------------------------------------------------------


def test_app_logging_format_includes_correlation_id():
    """El `format=` del `logging.basicConfig` debe interpolar
    `%(correlation_id)s` — sin él, los logs no expondrán el ID aunque
    el filter esté instalado.

    Verificación AST-based para no depender de regex con parens balanceados
    (basicConfig tiene `getattr(...)` anidado que rompe regex naïve)."""
    src = _APP_PY.read_text(encoding="utf-8")
    tree = ast.parse(src)

    found_basicConfig = False
    found_correlation_in_format = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # logging.basicConfig(...) — Attribute(Name('logging'), 'basicConfig')
        fn = node.func
        if not isinstance(fn, ast.Attribute) or fn.attr != "basicConfig":
            continue
        if not isinstance(fn.value, ast.Name) or fn.value.id != "logging":
            continue
        found_basicConfig = True
        for kw in node.keywords:
            if kw.arg == "format" and isinstance(kw.value, ast.Constant):
                if "%(correlation_id)s" in (kw.value.value or ""):
                    found_correlation_in_format = True
                    break
        break

    assert found_basicConfig, "No se encontró `logging.basicConfig(...)` en app.py"
    assert found_correlation_in_format, (
        "El format= de logging.basicConfig no incluye `%(correlation_id)s`. "
        "Sin el placeholder, el filter inyecta el atributo pero el formato "
        "lo ignora — logs sin trazabilidad request."
    )


def test_app_installs_log_filter():
    """`install_log_filter()` debe llamarse en app.py POST-basicConfig."""
    src = _APP_PY.read_text(encoding="utf-8")
    # Find positions
    basic_cfg_idx = src.find("logging.basicConfig")
    install_idx = src.find("install_log_filter()")
    assert basic_cfg_idx != -1, "logging.basicConfig no encontrado"
    assert install_idx != -1, (
        "install_log_filter() no se llama en app.py. Sin esa call, "
        "los logs no tendrán el atributo `correlation_id` y el format "
        "fallaría con KeyError."
    )
    assert install_idx > basic_cfg_idx, (
        "install_log_filter() debe llamarse DESPUÉS de basicConfig. "
        "Llamarlo antes no tiene efecto (el handler raíz aún no existe)."
    )


def test_app_has_correlation_middleware():
    """El middleware FastAPI `_correlation_id_middleware` debe existir,
    estar decorado con `@app.middleware("http")`, leer/generar el ID,
    y echo en response header."""
    src = _APP_PY.read_text(encoding="utf-8")
    assert "@app.middleware" in src, (
        "Falta `@app.middleware(...)` en app.py. Sin middleware no hay "
        "scope per-request y todos los logs llevarían `corr=-`."
    )
    assert "_correlation_id_middleware" in src, (
        "Función `_correlation_id_middleware` no encontrada por nombre. "
        "Si fue renombrada, actualizar este test y el anchor."
    )
    # El middleware DEBE leer el header (case-insensitive idealmente)
    middleware_block = re.search(
        r"async def _correlation_id_middleware[\s\S]+?(?=\n(?:async )?def |\Z)",
        src,
    )
    assert middleware_block is not None
    body = middleware_block.group(0)
    assert "X-Correlation-ID" in body, (
        "Middleware no referencia `X-Correlation-ID` header."
    )
    assert "_set_corr" in body or "set_correlation_id" in body, (
        "Middleware no setea el ContextVar via `set_correlation_id`."
    )
    assert "_reset_corr" in body or "reset_correlation_id" in body, (
        "Middleware no resetea el ContextVar al final. SIN reset, el "
        "valor leak al siguiente request del mismo worker thread."
    )
    assert "response.headers" in body, (
        "Middleware no echo `X-Correlation-ID` en response.headers. "
        "Cliente no podrá citar el ID al reportar bugs."
    )


def test_cors_exposes_correlation_header():
    """CORS `expose_headers` debe incluir `X-Correlation-ID` para que el
    browser JS del frontend pueda LEER el header de la response.

    AST-based para no romper si el bloque tiene comentarios con `)` (los
    regex naïve se cortan en el primer paréntesis cerrado dentro del
    bloque, e.g. en un comentario `(see this)`)."""
    src = _APP_PY.read_text(encoding="utf-8")
    tree = ast.parse(src)
    found_cors_call = False
    expose_headers_kw = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        # app.add_middleware(...)
        if not (isinstance(fn, ast.Attribute) and fn.attr == "add_middleware"):
            continue
        # Primer arg posicional debe ser `CORSMiddleware`
        if not node.args or not isinstance(node.args[0], ast.Name):
            continue
        if node.args[0].id != "CORSMiddleware":
            continue
        found_cors_call = True
        for kw in node.keywords:
            if kw.arg == "expose_headers":
                expose_headers_kw = kw.value
                break
        break

    assert found_cors_call, "No se encontró `app.add_middleware(CORSMiddleware, ...)`"
    assert expose_headers_kw is not None, (
        "CORS config no tiene `expose_headers=[...]`. Sin él, el browser "
        "JS no puede leer `X-Correlation-ID` de la response."
    )
    # expose_headers debe ser una List con `X-Correlation-ID` como Constant
    assert isinstance(expose_headers_kw, ast.List), (
        f"expose_headers debe ser un list literal, got {type(expose_headers_kw).__name__}"
    )
    headers_listed = [
        elt.value for elt in expose_headers_kw.elts
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    ]
    assert "X-Correlation-ID" in headers_listed, (
        f"expose_headers no contiene `X-Correlation-ID`. Headers expuestos: "
        f"{headers_listed}. Sin él, el cliente no puede mostrar el ID en bug "
        f"reports — pierdes el valor principal de correlation_id."
    )


# ---------------------------------------------------------------------------
# bg_executor — propagación via copy_context.
# ---------------------------------------------------------------------------


def test_bg_executor_uses_copy_context():
    """`submit_bg_task` debe importar `contextvars` y usar `copy_context()`
    + `ctx.run(...)` en el wrap del callable. Sin esto, bg tasks pierden
    el correlation_id del request."""
    src = _BG_PY.read_text(encoding="utf-8")
    assert "import contextvars" in src, (
        "bg_executor.py no importa `contextvars`. Sin él no hay propagación."
    )
    m = re.search(
        r"def submit_bg_task[\s\S]+?(?=\ndef |\Z)",
        src,
    )
    assert m is not None
    body = m.group(0)
    assert "copy_context" in body, (
        "submit_bg_task no llama `copy_context()`. Worker threads del "
        "ThreadPoolExecutor no heredan ContextVars automáticamente — los "
        "logs del bg task aparecerían con `corr=-` rompiendo trazabilidad."
    )
    assert "ctx.run" in body, (
        "submit_bg_task captura el context pero no lo invoca via `ctx.run(...)`. "
        "El snapshot copiado debe ejecutar el callable para restaurar los vars."
    )


# ---------------------------------------------------------------------------
# Functional check — el filter funciona en runtime.
# ---------------------------------------------------------------------------


def test_filter_injects_correlation_id_to_log_record():
    """El filter debe setear `record.correlation_id` en cada log record."""
    import logging

    # Aislar: nuevo logger sin handlers (no contamina root del test runner)
    from correlation import CorrelationIdFilter, set_correlation_id, reset_correlation_id

    test_logger = logging.getLogger("p3_corr_test_logger")
    test_logger.handlers.clear()
    test_logger.addFilter(CorrelationIdFilter())

    # Token para set + reset
    token = set_correlation_id("test1234")
    try:
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname=__file__, lineno=1,
            msg="hello", args=(), exc_info=None,
        )
        for f in test_logger.filters:
            f.filter(record)
        assert record.correlation_id == "test1234", (
            f"Filter no inyectó el ID al record. Got {getattr(record, 'correlation_id', None)!r}"
        )
    finally:
        reset_correlation_id(token)


def test_default_correlation_id_is_dash():
    """Fuera de un scope activo, `get_correlation_id()` retorna `'-'`."""
    from correlation import get_correlation_id
    # En el test runner no hay middleware activo
    assert get_correlation_id() == "-", (
        f"Default correlation_id debe ser '-', got {get_correlation_id()!r}"
    )


def test_with_correlation_id_context_manager():
    """`with_correlation_id()` setea, yieldea, y resetea."""
    from correlation import with_correlation_id, get_correlation_id

    assert get_correlation_id() == "-"
    with with_correlation_id("abc12345") as cid:
        assert cid == "abc12345"
        assert get_correlation_id() == "abc12345"
    # Tras el with, debe estar de vuelta al default
    assert get_correlation_id() == "-"


def test_with_correlation_id_generates_when_no_value():
    """`with_correlation_id()` sin arg genera uno via `new_correlation_id()`."""
    from correlation import with_correlation_id, get_correlation_id

    with with_correlation_id() as cid:
        assert cid != "-"
        assert len(cid) == 8
        assert re.match(r"^[a-f0-9]{8}$", cid)
        assert get_correlation_id() == cid


def test_new_correlation_id_uniqueness():
    """`new_correlation_id()` genera valores distintos. Collision en 8
    chars hex es ~1 en 4B — para 1k iteraciones debe haber 0."""
    from correlation import new_correlation_id

    ids = {new_correlation_id() for _ in range(1000)}
    # 1000 unique, 0 collisions esperadas (256M space)
    assert len(ids) == 1000, f"Colisión en {1000 - len(ids)} de 1000 IDs"


def test_install_log_filter_is_idempotent():
    """`install_log_filter()` no añade filtros duplicados al re-llamarse."""
    import logging
    from correlation import install_log_filter, CorrelationIdFilter

    test_logger = logging.getLogger("p3_corr_idempotent_test")
    test_logger.handlers.clear()
    test_logger.filters.clear()
    install_log_filter(test_logger)
    install_log_filter(test_logger)
    install_log_filter(test_logger)

    corr_filters = [f for f in test_logger.filters if isinstance(f, CorrelationIdFilter)]
    assert len(corr_filters) == 1, (
        f"install_log_filter no idempotente — {len(corr_filters)} filtros instalados."
    )
