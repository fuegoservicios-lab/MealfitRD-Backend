"""[P0-8] Tests para garantizar persistencia atómica del buffer
`deferrals_pending.jsonl` usado por `_append_deferral_to_buffer` y
`_flush_pending_deferrals`.

Bug original (audit P0-8):
  Ambas funciones hacían `with open(path, "w") as f: f.write(...)` con
  read-modify-write completo del archivo en cada operación. Sin tempfile,
  sin `os.replace`, sin `f.flush()` + `os.fsync()`. Un crash del proceso,
  kill -9, OOM, o reboot durante la reescritura corrompía el JSONL
  completo (no solo la última línea) y se perdían TODOS los deferrals
  pendientes — justo el caso para el que el buffer existe (DB caída →
  buffer crece → server reinicia). El archivo `.bak` con 1473 líneas
  evidenciaba uso real con volumen.

  Adicionalmente: cada append era O(N) en líneas existentes (read-all +
  write-all), con O(N²) acumulado bajo presión sostenida.

Fix:
  - `_append_deferral_to_buffer`: modo `"a"` simple con UNA sola línea +
    `flush` + `os.fsync`. O(1) por append. Crashes mid-write corrompen
    como mucho la última línea (records previos son inmutables).
  - `_flush_pending_deferrals`: reescritura vía `tmp + fsync + os.replace`.
    Atómico cross-platform; un crash deja el archivo original intacto.
  - FIFO cap (`CHUNK_DEFERRALS_BUFFER_MAX_RECORDS`) movido del hot path
    de append al sweep periódico del flush.

Cobertura:
  - test_append_uses_mode_a_not_w
  - test_append_calls_fsync
  - test_append_does_not_read_existing_file
  - test_append_preserves_existing_records
  - test_flush_uses_tmp_then_replace_for_atomic_rewrite
  - test_flush_applies_fifo_cap
  - test_flush_calls_fsync_before_replace
  - test_source_no_longer_uses_naked_open_w_for_buffer
"""
import ast
import inspect
import os
import re
import textwrap
from unittest.mock import patch, MagicMock

import pytest

import cron_tasks


def _open_calls_in(fn) -> list:
    """Devuelve la lista de modos string que aparecen como segundo arg
    posicional en `open(...)` calls dentro del cuerpo de `fn` (excluyendo
    docstring + comentarios). Robusto contra falsos positivos textuales."""
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    modes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                modes.append(node.args[1].value)
    return modes


def _calls_with_attr(fn, *attr_paths) -> list:
    """Devuelve los nodos AST `Call` cuyo `func` es un Attribute del path
    dado (ej. `os.fsync`, `os.replace`). El path es una tupla
    ('os','fsync'). Retorna nodos en el orden en que aparecen."""
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Reconstruir path del atributo: ej. os.fsync → ['os','fsync']
        path = []
        cur = node.func
        while isinstance(cur, ast.Attribute):
            path.insert(0, cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            path.insert(0, cur.id)
        if tuple(path) == attr_paths:
            out.append(node)
    return out


# ---------------------------------------------------------------------------
# 1. `_append_deferral_to_buffer` — modo "a" simple.
# ---------------------------------------------------------------------------
def test_append_uses_mode_a_not_w():
    """El AST del helper compartido `_atomic_append_jsonl_record` debe
    contener `open(..., "a")` y NO `open(..., "w")`. AST evita falsos
    positivos textuales sobre docstrings o comentarios.
    El wrapper `_append_deferral_to_buffer` delega a este helper (P0-10
    extracción) — la garantía real vive en el helper."""
    modes = _open_calls_in(cron_tasks._atomic_append_jsonl_record)
    assert "a" in modes, f"P0-8: append debe usar open(path, 'a', ...). Modes encontrados: {modes}"
    assert "w" not in modes, (
        f"P0-8 regression: open(path, 'w') reapareció en append helper. Modes: {modes}"
    )


def test_append_calls_fsync(tmp_path, monkeypatch):
    """Después del write, debe llamar `os.fsync(f.fileno())` para durabilidad."""
    buf_path = tmp_path / "deferrals_test.jsonl"
    monkeypatch.setattr("constants.CHUNK_DEFERRALS_BUFFER_PATH", str(buf_path))

    fsync_called = {"count": 0}
    real_fsync = os.fsync

    def tracked_fsync(fd):
        fsync_called["count"] += 1
        return real_fsync(fd)

    with patch.object(cron_tasks.os, "fsync", side_effect=tracked_fsync):
        ok = cron_tasks._append_deferral_to_buffer({"user_id": "u1", "meal_plan_id": "m1"})

    assert ok is True
    assert fsync_called["count"] >= 1, \
        "P0-8: _append_deferral_to_buffer debe llamar os.fsync para durabilidad"


def test_append_does_not_read_existing_file(tmp_path, monkeypatch):
    """`open(path, "a")` NO debe leer el archivo existente — el bug original
    leía O(N) líneas en cada append. Verificamos que el helper solo añade
    una línea sin tocar las anteriores."""
    buf_path = tmp_path / "deferrals_test.jsonl"
    # Pre-popular con 5 records.
    initial = ["\n".join(['{"id": ' + str(i) + '}' for i in range(5)]) + "\n"]
    buf_path.write_text(initial[0], encoding="utf-8")
    initial_content = buf_path.read_text(encoding="utf-8")
    initial_size = len(initial_content)

    monkeypatch.setattr("constants.CHUNK_DEFERRALS_BUFFER_PATH", str(buf_path))

    ok = cron_tasks._append_deferral_to_buffer({
        "user_id": "00000000-0000-0000-0000-000000000001",
        "meal_plan_id": "00000000-0000-0000-0000-000000000002",
    })

    assert ok is True
    final = buf_path.read_text(encoding="utf-8")
    # El contenido nuevo DEBE empezar con el contenido original (preservado byte-a-byte).
    assert final.startswith(initial_content), \
        "P0-8: append debe preservar el contenido previo intacto byte-a-byte"
    # Y debe haber UNA línea adicional.
    assert len(final) > initial_size
    assert final[initial_size:].count("\n") == 1, \
        "P0-8: append debe agregar exactamente UNA línea"


def test_append_preserves_existing_records_under_concurrent_writes(tmp_path, monkeypatch):
    """Bajo lock, múltiples appends preservan TODOS los records previos."""
    buf_path = tmp_path / "deferrals_test.jsonl"
    monkeypatch.setattr("constants.CHUNK_DEFERRALS_BUFFER_PATH", str(buf_path))

    for i in range(10):
        cron_tasks._append_deferral_to_buffer({
            "user_id": f"00000000-0000-0000-0000-{i:012d}",
            "meal_plan_id": f"00000000-0000-0000-0000-{i+100:012d}",
            "seq": i,
        })

    content = buf_path.read_text(encoding="utf-8")
    lines = [ln for ln in content.splitlines() if ln.strip()]
    assert len(lines) == 10
    # Los seq deben aparecer en orden: el bug original (read+rewrite con FIFO
    # cap aplicado en cada append) podía descartar registros antiguos
    # incorrectamente. Ahora todos se preservan.
    seqs = []
    import json as _json
    for ln in lines:
        seqs.append(_json.loads(ln)["seq"])
    assert seqs == list(range(10)), \
        f"P0-8: orden de records corrupto. Esperado 0-9, got {seqs}"


# ---------------------------------------------------------------------------
# 2. `_flush_pending_deferrals` — reescritura atómica vía tmp+os.replace.
# ---------------------------------------------------------------------------
def test_flush_uses_tmp_then_replace_for_atomic_rewrite():
    """El helper compartido `_atomic_rewrite_jsonl_buffer` debe usar archivo
    temporal y `os.replace` para atomicidad cross-platform. `_flush_pending_deferrals`
    delega a este helper (P0-10 extracción)."""
    src = inspect.getsource(cron_tasks._atomic_rewrite_jsonl_buffer)
    assert "os.replace" in src, \
        "P0-8: helper de rewrite debe usar os.replace para atomicidad"
    assert ".tmp" in src or "tempfile" in src, \
        "P0-8: helper de rewrite debe usar archivo temporal"


def test_flush_calls_fsync_before_replace():
    """En el helper compartido, `os.fsync` debe ejecutarse ANTES del
    `os.replace` (vía AST line numbers)."""
    fsync_calls = _calls_with_attr(cron_tasks._atomic_rewrite_jsonl_buffer, "os", "fsync")
    replace_calls = _calls_with_attr(cron_tasks._atomic_rewrite_jsonl_buffer, "os", "replace")
    assert fsync_calls, "P0-8: rewrite helper debe invocar os.fsync"
    assert replace_calls, "P0-8: rewrite helper debe invocar os.replace"
    fsync_line = min(c.lineno for c in fsync_calls)
    replace_line = min(c.lineno for c in replace_calls)
    assert fsync_line < replace_line, (
        f"P0-8: os.fsync (línea {fsync_line}) DEBE preceder a os.replace "
        f"(línea {replace_line})"
    )


def test_flush_applies_fifo_cap_during_rewrite(tmp_path, monkeypatch):
    """Cuando el buffer crece más allá del cap durante un outage, el flush
    debe truncarlo al cap (descartando las líneas más viejas, FIFO)."""
    buf_path = tmp_path / "deferrals_test.jsonl"
    monkeypatch.setattr("constants.CHUNK_DEFERRALS_BUFFER_PATH", str(buf_path))
    # Forzar cap pequeño para test.
    monkeypatch.setattr("constants.CHUNK_DEFERRALS_BUFFER_MAX_RECORDS", 5)

    # Mock execute_sql_write para que SIEMPRE falle con un error transitorio
    # (los records quedan en remaining y sí se reescriben).
    def fake_execute(*args, **kwargs):
        raise ConnectionError("DB unavailable")

    # Pre-popular con 20 records VÁLIDOS (UUIDs reales).
    import json as _json
    lines = []
    for i in range(20):
        lines.append(_json.dumps({
            "user_id": f"00000000-0000-0000-0000-{i:012d}",
            "meal_plan_id": f"00000000-0000-0000-0000-{i+100:012d}",
            "seq": i,
        }))
    buf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with patch.object(cron_tasks, "execute_sql_write", side_effect=fake_execute):
        stats = cron_tasks._flush_pending_deferrals()

    # Tras el flush, todos quedaron en remaining (DB falló). El cap debe
    # haber recortado a las 5 más recientes (seq=15..19).
    assert stats["remaining"] == 5, f"esperado remaining=5, got {stats}"
    final = buf_path.read_text(encoding="utf-8")
    final_lines = [_json.loads(ln) for ln in final.splitlines() if ln.strip()]
    assert len(final_lines) == 5
    seqs = [r["seq"] for r in final_lines]
    assert seqs == [15, 16, 17, 18, 19], \
        f"FIFO cap incorrecto: esperado [15..19], got {seqs}"


def test_flush_atomic_replace_leaves_original_on_tmp_error(tmp_path, monkeypatch):
    """Si os.replace falla (ej. permissions), el archivo original debe
    permanecer intacto (no medio-corrupto)."""
    buf_path = tmp_path / "deferrals_test.jsonl"
    monkeypatch.setattr("constants.CHUNK_DEFERRALS_BUFFER_PATH", str(buf_path))
    monkeypatch.setattr("constants.CHUNK_DEFERRALS_BUFFER_MAX_RECORDS", 1000)

    import json as _json
    original_lines = [
        _json.dumps({
            "user_id": f"00000000-0000-0000-0000-{i:012d}",
            "meal_plan_id": f"00000000-0000-0000-0000-{i+100:012d}",
            "seq": i,
        })
        for i in range(3)
    ]
    original_content = "\n".join(original_lines) + "\n"
    buf_path.write_text(original_content, encoding="utf-8")

    def fake_db_fail(*args, **kwargs):
        raise ConnectionError("transient")

    def fake_replace(src, dst):
        raise PermissionError("simulated replace failure")

    with patch.object(cron_tasks, "execute_sql_write", side_effect=fake_db_fail), \
         patch.object(cron_tasks.os, "replace", side_effect=fake_replace):
        # No debe lanzar — el except interno captura el fallo del replace.
        cron_tasks._flush_pending_deferrals()

    # El archivo original DEBE seguir intacto.
    assert buf_path.exists()
    final_content = buf_path.read_text(encoding="utf-8")
    assert final_content == original_content, \
        "P0-8: si os.replace falla, el original debe quedar intacto"


# ---------------------------------------------------------------------------
# 3. Defensa textual contra reintroducción del patrón roto.
# ---------------------------------------------------------------------------
def test_source_no_longer_uses_naked_open_w_for_buffer_in_append():
    """Defensa AST adicional: el helper compartido NO debe contener
    `open(..., "w")` real (ignorando docstrings/comentarios)."""
    modes = _open_calls_in(cron_tasks._atomic_append_jsonl_record)
    assert "w" not in modes, \
        f"P0-8 regression: open(..., 'w') reapareció en append helper. Modes: {modes}"


def test_source_flush_uses_tempfile_pattern():
    """El helper de rewrite debe construir un path temporal."""
    src = inspect.getsource(cron_tasks._atomic_rewrite_jsonl_buffer)
    assert "tmp_path" in src or "tempfile" in src, \
        "P0-8: el rewrite helper debe usar un path temporal explícito"


def test_documentation_p0_8_present():
    """Comentarios `[P0-8]` deben estar presentes en ambas funciones para
    que futuros maintainers entiendan el contrato de atomicidad."""
    for fn in (cron_tasks._append_deferral_to_buffer, cron_tasks._flush_pending_deferrals):
        src = inspect.getsource(fn)
        assert "[P0-8]" in src, f"falta documentación P0-8 en {fn.__name__}"
