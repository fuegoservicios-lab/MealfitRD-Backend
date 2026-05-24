"""[P1-PROD-AUDIT-1 · 2026-05-23] `except Exception:` en startup/init
handlers DEBEN tener `logger.error/warning/info` en el body — no silenciar.

Gap original (audit production-readiness 2026-05-23, B-P1-4):
    `app.py` tiene ~34 bare `except Exception:` blocks (Sentry init,
    scheduler init, telemetry init, fixtures init). La mayoría loguean
    pero no hay enforcement — un PR que añada un nuevo `except Exception:
    pass` silencioso al startup podría ocultar fallos críticos de init
    (Sentry quedando off, scheduler no registrando crons, DB pool no
    abriendo).

Decisión arquitectónica:
    NO migrar a `except (SpecificError1, SpecificError2)` — startup tiene
    decenas de modos de fallo distintos (network, DNS, config, permisos,
    versión de lib). El `except Exception` defensive es intencional para
    que un blip de Supabase NO mate el worker entero.

    Lo que SÍ se enforza:
      - Cada bare `except Exception:` debe tener `logger.error|warning|info|
        debug` en el body.
      - Excepciones legítimas (re-raise, return, decoradores) marcadas
        inline con `# [BARE-EXCEPT-EXEMPT: <razón>]`.

Cobertura:
    Escaneo AST de `app.py` (puede extenderse a otros archivos de startup
    en futuro). Para cada `try: ... except Exception` en module-level o
    function body de startup handlers, validar que:
      (a) El except body tiene call a `logger.<level>(...)`, O
      (b) El except body hace `raise` (re-throw), O
      (c) Hay marker `# [BARE-EXCEPT-EXEMPT: <razón>]` en las 5 líneas
          arriba del except.

Tooltip-anchor: P1-PROD-AUDIT-1-STARTUP-BARE-EXCEPT | audit 2026-05-23.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"

_EXEMPT_MARKER = re.compile(r"#\s*\[BARE-EXCEPT-EXEMPT\s*:")


def _is_logger_call(node: ast.AST) -> bool:
    """True si el node es `logger.<level>(...)` o `logging.<level>(...)`
    en la statement-list del except body.
    """
    if not isinstance(node, ast.Expr):
        return False
    call = node.value
    if not isinstance(call, ast.Call):
        return False
    f = call.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        if f.value.id in ("logger", "logging", "_logger"):
            return f.attr in ("error", "warning", "info", "debug", "exception", "critical", "log")
    return False


def _is_raise(node: ast.AST) -> bool:
    return isinstance(node, ast.Raise)


def _is_print(node: ast.AST) -> bool:
    """`print(...)` también cuenta como señal (aunque CLAUDE.md prohibe
    print en productivo, en startup handlers Sentry-init pre-Sentry-installed
    print fallback es legítimo).
    """
    if not isinstance(node, ast.Expr):
        return False
    call = node.value
    if not isinstance(call, ast.Call):
        return False
    f = call.func
    return isinstance(f, ast.Name) and f.id == "print"


def _body_has_observable_signal(body: list[ast.AST]) -> bool:
    """Recursivamente busca en el except body si hay logger.X / raise / print.

    Recursivo porque `except Exception as e: if x: logger.warn(...)` es válido.
    """
    for stmt in body:
        if _is_logger_call(stmt) or _is_raise(stmt) or _is_print(stmt):
            return True
        # Recurse en if / try / for / while bodies.
        for attr in ("body", "orelse", "handlers", "finalbody"):
            sub = getattr(stmt, attr, None)
            if isinstance(sub, list):
                if _body_has_observable_signal(sub):
                    return True
            elif isinstance(sub, list) and len(sub) > 0:
                if _body_has_observable_signal(sub):
                    return True
    return False


def _has_exempt_marker_before(source: str, lineno: int, window: int = 5) -> bool:
    """True si en las `window` líneas anteriores al lineno hay marker
    `[BARE-EXCEPT-EXEMPT: ...]`.
    """
    lines = source.split("\n")
    start = max(0, lineno - 1 - window)
    end = lineno
    window_text = "\n".join(lines[start:end])
    return bool(_EXEMPT_MARKER.search(window_text))


def test_app_py_exists():
    assert _APP_PY.exists(), f"app.py ausente en {_APP_PY}"


def test_anchor_present():
    src = _APP_PY.read_text(encoding="utf-8")
    assert "P1-PROD-AUDIT-1" in src or "BARE-EXCEPT-EXEMPT" in src, (
        "app.py NO menciona `P1-PROD-AUDIT-1` ni `BARE-EXCEPT-EXEMPT`. "
        "Si removiste el anchor, este test pierde su contexto. Restaurar."
    )


# Snapshot 2026-05-23 — líneas de app.py con bare except sin señal observable.
# Estos son tolerados como deuda existente; este test SOLO falla si aparece
# una línea NUEVA fuera de este allowlist. Cada cierre incremental (añadir
# logger.X o marker) reduce este set; el commit debe eliminar la línea del
# allowlist en el mismo PR.
_GRANDFATHERED_BARE_EXCEPT_LINES = frozenset({
    180, 200, 1206, 1297, 1300, 1313, 1333, 1363, 1383, 1394,
    1447, 1486, 1517, 1531, 1537, 1558, 1693,
})


def test_bare_except_blocks_have_observable_signal():
    """Cada `except Exception` (sin re-raise) NUEVO debe tener señal observable
    (logger.X / raise / print). Líneas grandfathered en
    `_GRANDFATHERED_BARE_EXCEPT_LINES` son tolerated baseline 2026-05-23.

    Cierre incremental: cada cleanup commit reduce el allowlist.
    """
    src = _APP_PY.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_APP_PY))

    violations = []
    grandfathered_present = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        # Solo `except Exception` o `except BaseException` o bare `except:`.
        exc_type = node.type
        is_bare = exc_type is None
        is_exception = (
            isinstance(exc_type, ast.Name) and exc_type.id in ("Exception", "BaseException")
        )
        if not (is_bare or is_exception):
            continue

        if _body_has_observable_signal(node.body):
            continue

        # Check exempt marker.
        if _has_exempt_marker_before(src, node.lineno):
            continue

        if node.lineno in _GRANDFATHERED_BARE_EXCEPT_LINES:
            grandfathered_present.append(node.lineno)
            continue

        violations.append({
            "lineno": node.lineno,
            "exc_name": "Exception" if is_exception else "<bare>",
        })

    if violations:
        detail = "\n".join(
            f"  - app.py:{v['lineno']}  (except {v['exc_name']}: ... sin señal observable)"
            for v in violations
        )
        msg = (
            f"\n[P1-PROD-AUDIT-1-STARTUP-BARE-EXCEPT] {len(violations)} NUEVO(s) except "
            f"block(s) en app.py sin señal observable (no en allowlist):\n\n{detail}\n\n"
            f"Cada `except Exception:` NUEVO debe tener uno de:\n"
            f"  (a) `logger.error|warning|info|debug(...)` para visibilidad.\n"
            f"  (b) `raise` (re-throw — defensa válida).\n"
            f"  (c) `print(...)` (solo para handlers pre-logger en startup).\n"
            f"  (d) Marker `# [BARE-EXCEPT-EXEMPT: <razón>]` en las 5 líneas\n"
            f"      arriba del except.\n\n"
            f"NO añadir al allowlist `_GRANDFATHERED_BARE_EXCEPT_LINES` — el "
            f"allowlist es snapshot del baseline 2026-05-23, NO mecanismo para "
            f"añadir excepciones nuevas. Si el bare except es legítimo, usar marker (d).\n"
        )
        pytest.fail(msg)


def test_grandfathered_allowlist_is_eventually_drained():
    """Sanity: el allowlist NO debe crecer. Imprime el count actual para que
    el progreso del cleanup sea visible en CI logs.
    """
    print(
        f"\n[P1-PROD-AUDIT-1-STARTUP-BARE-EXCEPT] Grandfathered allowlist size: "
        f"{len(_GRANDFATHERED_BARE_EXCEPT_LINES)} línea(s) tolerated como "
        f"deuda baseline 2026-05-23. Cada cleanup commit reduce este número."
    )
    # No assertion — solo visibility. El test falla solo si _GRANDFATHERED
    # cambia de manera explícita (commit visible en review).
    assert len(_GRANDFATHERED_BARE_EXCEPT_LINES) <= 20, (
        f"Grandfathered allowlist tiene {len(_GRANDFATHERED_BARE_EXCEPT_LINES)} "
        f"entries — alguien añadió >3 sin reducir. El allowlist es snapshot, "
        f"NO mecanismo de exemption ongoing. Refactor para reducir."
    )


def test_no_new_pass_only_bare_except():
    """Específico: `except Exception: pass` sin nada más = anti-patrón puro.
    No tolerado en líneas NUEVAS (allowlist grandfathered es snapshot).
    """
    src = _APP_PY.read_text(encoding="utf-8")
    # AST-based para evitar falsos positivos en strings.
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        exc_type = node.type
        is_exception = (
            isinstance(exc_type, ast.Name) and exc_type.id in ("Exception", "BaseException")
        )
        if not (exc_type is None or is_exception):
            continue
        # `body` = lista de statements. Si solo es `[Pass()]` o `[Expr(Constant)]`
        # (docstring), es pass-only.
        is_pass_only = (
            len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
        )
        if is_pass_only:
            # Permitir si tiene `# [BARE-EXCEPT-EXEMPT: pass-deliberate ...]`
            # explícito O está grandfathered.
            if _has_exempt_marker_before(src, node.lineno):
                continue
            if node.lineno in _GRANDFATHERED_BARE_EXCEPT_LINES:
                continue
            bad.append(node.lineno)
    if bad:
        pytest.fail(
            f"app.py tiene `except: pass` puro NUEVO (no en allowlist): "
            f"líneas {bad}. Esto silencia errores 100% sin trazabilidad. "
            f"Mínimo añadir `logger.debug(f'caught: {{e}}')` o marker "
            f"`# [BARE-EXCEPT-EXEMPT: <razón>]`."
        )
