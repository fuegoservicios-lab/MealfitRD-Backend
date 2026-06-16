"""[P2-A11Y-LOGGING · 2026-05-13] Anchors para silent-degradation logging
(parte del bundle P2 junto con a11y frontend gaps).

Contexto:
    Audit production-readiness 2026-05-12 detectó 4 callsites con
    `except Exception: pass` (sin logging) en hot paths del pipeline:

      1. `ai_helpers.py` — parse de `temporary_dislikes` (ISO timestamp)
         dentro de `_build_filter_context` (función que construye
         filtros de ingredientes según preferencias del usuario).
      2. `ai_helpers.py` — fetch de `variety_level` desde
         `health_profile` (fallback silencioso a `"standard"`).
      3. `ai_helpers.py` — fetch de `rejection_patterns` (memoria
         del Revisor Médico, anti-repetición de errores corregidos).
      4. `agent.py` — generación de sugerencia anti mode-collapse en
         flujo de swap de comida (variedad de proteínas/carbs).

    Modo de fallo común: bajo carga sostenida un blip transient de
    pool DB hace que TODOS estos paths fallen al mismo tiempo y el
    sistema entrega planes degradados sin **un solo log de error**.
    Sentry ve cero, métricas ven cero, usuario nota "variedad rara"
    pero SRE no puede correlacionar.

Fix:
    Cada bare `except Exception: pass` reemplazado por
    `logger.debug("[P2-SILENT-DEGRADATION] <contexto>: %s: %s",
    type(exc).__name__, str(exc)[:160])`. Mantiene el fallback
    (no re-raise — el comportamiento funcional no cambia). Permite
    grep `[P2-SILENT-DEGRADATION]` en logs para detectar burst de
    fallos durante incidentes.

Lo que este test enforza:
  A) Anchor `P2-SILENT-DEGRADATION` presente en `ai_helpers.py` y
     `agent.py` (mínimo N veces cada uno).
  B) Cero ocurrencias del patrón `except Exception:\n\s+pass` (sin
     logger en las 12 líneas siguientes) en los 2 archivos parcheados.
     Si alguien añade un nuevo bare-except sin logging, el test falla.
  C) Cada anchor `[P2-SILENT-DEGRADATION]` viene precedido por
     `except Exception as <name>:` en ≤ 6 líneas previas (el log
     pertenece a un handler de excepción, no es un log random).

Tooltip-anchor: P2-A11Y-LOGGING / P2-SILENT-DEGRADATION.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_AI_HELPERS = _BACKEND_ROOT / "ai_helpers.py"
_AGENT_PY = _BACKEND_ROOT / "agent.py"

# Archivos parcheados en P2-A11Y-LOGGING. Si añades el patrón a otro
# archivo productivo, súmalo aquí + sube min_anchors.
_PATCHED_FILES: tuple[tuple[Path, int], ...] = (
    (_AI_HELPERS, 3),  # 3 callsites: temp_dislikes, variety_level, rejection_patterns
    (_AGENT_PY, 3),    # 3 callsites: anti mode-collapse swap + generate_new_plan parse + modify_single_meal parse
)

# Pattern: bare `except Exception` seguido en la línea siguiente por
# `pass` solo (sin logger). Multiline para capturar la siguiente línea.
_BARE_EXCEPT_PASS = re.compile(
    r"except\s+Exception\s*:\s*\n[ \t]+pass\s*(?:#[^\n]*)?\s*\n",
    re.MULTILINE,
)

# [P2-A11Y-LOGGING intent] El contrato (docstring B) apunta a la degradación
# SILENCIOSA de PATHS FUNCIONALES (parse de temp_dislikes/variety_level/
# rejection_patterns en ai_helpers, sugerencia anti mode-collapse en agent).
# Un bare `except Exception: pass` cuyo `try:` envuelve operaciones BEST-EFFORT
# de observabilidad/limpieza (logging, emisión de métricas a pipeline_metrics,
# cierre de un iterator de stream) NO es degradación funcional silenciosa — su
# fallo es aceptable e intencional, y NO debe (ni puede útilmente) re-loguearse
# (un except que envuelve un `logger.debug` no puede loguear su propio fallo sin
# recursión). Estas señales identifican esos bloques para exentarlos sin debilitar
# la guarda sobre swallows de lógica funcional.
_BEST_EFFORT_TRY_BODY_SIGNALS = (
    "logger.",                # logging-of-logging (inner guard de un handler que ya loguea)
    "_emit_",                 # emisión de métricas/alertas best-effort
    "execute_sql_write",      # INSERT a pipeline_metrics / telemetría
    "INSERT INTO pipeline_metrics",
    ".close()",               # cleanup de recursos (stream_iter.close())
)


def _try_body_is_best_effort(src: str, except_start: int) -> bool:
    """Devuelve True si el `try:` que precede al `except Exception: pass` en
    `except_start` envuelve operaciones best-effort de observabilidad/limpieza.

    Localiza el `try:` correspondiente (al mismo nivel de indentación que el
    `except`) buscando hacia atrás, y examina su cuerpo en busca de las señales
    best-effort. Conservador: si no encuentra el `try:`, NO exenta (devuelve
    False → el match cuenta como violación)."""
    head = src[:except_start]
    lines = head.splitlines()
    if not lines:
        return False
    # Indentación de la línea del `except`.
    except_line = lines[-1]
    except_indent = len(except_line) - len(except_line.lstrip())
    # Buscar hacia atrás el `try:` al mismo nivel de indentación.
    body_lines: list[str] = []
    for line in reversed(lines[:-1]):
        stripped = line.strip()
        if not stripped:
            body_lines.append(line)
            continue
        cur_indent = len(line) - len(line.lstrip())
        if stripped == "try:" and cur_indent == except_indent:
            body = "\n".join(reversed(body_lines))
            return any(sig in body for sig in _BEST_EFFORT_TRY_BODY_SIGNALS)
        body_lines.append(line)
    return False


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# A) Anchor count por archivo.
@pytest.mark.parametrize("path,min_anchors", _PATCHED_FILES)
def test_a_anchor_count_meets_minimum(path: Path, min_anchors: int):
    src = _read(path)
    count = src.count("[P2-SILENT-DEGRADATION]")
    assert count >= min_anchors, (
        f"P2-A11Y-LOGGING: {path.name} tiene {count} anchors "
        f"`[P2-SILENT-DEGRADATION]`, esperaba >= {min_anchors}. "
        f"Si removiste un callsite del patrón, ajusta `_PATCHED_FILES` "
        f"en este test."
    )


# B) Cero bare-except sin logger en los archivos parcheados.
@pytest.mark.parametrize("path,_min", _PATCHED_FILES)
def test_b_no_bare_except_pass_in_patched_files(path: Path, _min: int):
    src = _read(path)
    matches = list(_BARE_EXCEPT_PASS.finditer(src))
    # Permitir matches que estén DENTRO de un docstring (raras pero
    # posibles, e.g. ejemplos). Heurística simple: el match no debe
    # estar entre triple-quotes consecutivos.
    real_bare = []
    for m in matches:
        before = src[: m.start()]
        # Si número impar de """ antes del match → estamos dentro de docstring.
        if before.count('"""') % 2 == 1 or before.count("'''") % 2 == 1:
            continue
        # Exentar bloques best-effort de observabilidad/limpieza (telemetría,
        # logging-of-logging, cierre de stream): NO son degradación funcional
        # silenciosa — ver _try_body_is_best_effort + nota de intent arriba.
        if _try_body_is_best_effort(src, m.start()):
            continue
        real_bare.append(m)
    assert not real_bare, (
        f"P2-A11Y-LOGGING: {path.name} tiene {len(real_bare)} bare "
        f"`except Exception: pass` sin logging.\n"
        f"Reemplazar por:\n"
        f"  except Exception as _exc:\n"
        f"      logger.debug(\n"
        f"          \"[P2-SILENT-DEGRADATION] <context>: %s: %s\",\n"
        f"          type(_exc).__name__, str(_exc)[:160])\n"
        f"Primer match en char offset {real_bare[0].start()} "
        f"(line ~{src[:real_bare[0].start()].count(chr(10)) + 1})."
    )


# C) Cada anchor pertenece a un except handler (no log random).
@pytest.mark.parametrize("path,_min", _PATCHED_FILES)
def test_c_anchor_inside_except_handler(path: Path, _min: int):
    src = _read(path)
    lines = src.splitlines()
    anchor_positions = [
        i for i, line in enumerate(lines)
        if "[P2-SILENT-DEGRADATION]" in line
    ]
    for line_idx in anchor_positions:
        # Mirar hasta 12 líneas hacia arriba por `except Exception`.
        window_start = max(0, line_idx - 12)
        window = lines[window_start:line_idx]
        has_except = any(
            re.search(r"except\s+Exception\s+as\s+\w+\s*:", w)
            for w in window
        )
        assert has_except, (
            f"P2-A11Y-LOGGING: anchor en {path.name}:{line_idx + 1} no "
            f"está dentro de un `except Exception as <name>:` handler "
            f"en las 12 líneas previas. Un log con este anchor pertenece "
            f"a un swallowed-exception path, no a un info/debug random.\n"
            f"Ventana revisada:\n  "
            + "\n  ".join(window)
        )
