"""[P3-4 · 2026-05-08] Tests del contrato `mutator(hp) -> dict | None | False`
en `update_user_health_profile_atomic` y su helper SSOT.

Bug original (audit 2026-05-07):
  El docstring de `update_user_health_profile_atomic` documenta que el mutator
  debe retornar `dict | None | False`. La implementación legacy era:
    `new_hp = result if isinstance(result, dict) else hp`
  Si un caller bug-introduce `return "string"` o `return ['list']` o
  `return True`, el helper SILENCIOSAMENTE persiste `hp` (in-place) sin
  log — el bug del caller queda invisible. Tests existentes (P1-ORQ-1) sólo
  cubrían los 3 valores contractuales (dict/None/False), no los anómalos.

Fix:
  1. Helper SSOT `_resolve_mutator_result(result, hp, *, user_id, path_label)`
     extraído (antes duplicado en 2 sitios: fallback path línea ~263 y atomic
     path línea ~298).
  2. Cuando `result` no es dict ni None, helper loguea
     `[P3-4/MUTATOR-CONTRACT] WARNING` con tipo + valor truncado + path_label
     (fallback/atomic) + sugerencia de investigar al caller.
  3. Backwards-compat: comportamiento end-user idéntico (cae a `hp`),
     solo añade observabilidad.

Cobertura:
  - dict → retorna dict (sin warning).
  - None → retorna hp (sin warning).
  - str/int/list/tuple/True → retorna hp + WARNING con path_label.
  - WARNING incluye tipo + repr truncado para diagnóstico.
  - SSOT: ningún call site mantiene el patrón inline `result if isinstance(...)`.
  - Helper invocado desde ambos paths (fallback + atomic).
"""
import logging
import pathlib
import re

import pytest


_DB_PROFILES = pathlib.Path(__file__).parent / "db_profiles.py"


# ---------------------------------------------------------------------------
# 1. Comportamiento del helper directo
# ---------------------------------------------------------------------------
def test_dict_result_returned_as_is():
    from db_profiles import _resolve_mutator_result
    new_hp = {"goal": "lose", "weight": 70}
    hp = {"goal": "maintain", "weight": 80}
    result = _resolve_mutator_result(new_hp, hp, user_id="u1", path_label="atomic")
    assert result is new_hp


def test_none_falls_back_to_hp_for_inplace():
    from db_profiles import _resolve_mutator_result
    hp = {"goal": "maintain", "weight": 80}
    result = _resolve_mutator_result(None, hp, user_id="u1", path_label="atomic")
    assert result is hp


def test_string_logs_warning_and_falls_back(caplog):
    from db_profiles import _resolve_mutator_result
    hp = {"goal": "maintain"}
    with caplog.at_level(logging.WARNING):
        result = _resolve_mutator_result("oops", hp, user_id="u1", path_label="atomic")
    assert result is hp
    assert any("[P3-4/MUTATOR-CONTRACT]" in rec.message for rec in caplog.records)
    assert any("str" in rec.message for rec in caplog.records)


def test_int_logs_warning_and_falls_back(caplog):
    from db_profiles import _resolve_mutator_result
    hp = {"goal": "maintain"}
    with caplog.at_level(logging.WARNING):
        result = _resolve_mutator_result(42, hp, user_id="u1", path_label="fallback")
    assert result is hp
    assert any("int" in rec.message for rec in caplog.records)


def test_list_logs_warning_and_falls_back(caplog):
    from db_profiles import _resolve_mutator_result
    hp = {"goal": "maintain"}
    with caplog.at_level(logging.WARNING):
        result = _resolve_mutator_result(["a", "b"], hp, user_id="u1", path_label="atomic")
    assert result is hp
    assert any("list" in rec.message for rec in caplog.records)


def test_true_logs_warning_and_falls_back(caplog):
    """`True` NO es False (filtrado antes), pero tampoco es dict ni None.
    Debe loguear WARNING — un caller que retorna True probablemente
    confundió el contrato (creyó que True = "yes, persist hp")."""
    from db_profiles import _resolve_mutator_result
    hp = {"goal": "maintain"}
    with caplog.at_level(logging.WARNING):
        result = _resolve_mutator_result(True, hp, user_id="u1", path_label="atomic")
    assert result is hp
    assert any("bool" in rec.message for rec in caplog.records)


def test_warning_includes_user_id_and_path_label(caplog):
    from db_profiles import _resolve_mutator_result
    hp = {}
    with caplog.at_level(logging.WARNING):
        _resolve_mutator_result(123, hp, user_id="user-xyz-789", path_label="fallback")
    msg = " ".join(rec.message for rec in caplog.records)
    assert "user-xyz-789" in msg, "WARNING debe incluir user_id para diagnóstico"
    assert "fallback" in msg, "WARNING debe incluir path_label (atomic/fallback)"


def test_warning_truncates_long_repr(caplog):
    """Si el caller retornó una estructura grande (lista de 1000 items),
    el WARNING debe truncar el repr para no inflar logs."""
    from db_profiles import _resolve_mutator_result
    hp = {}
    big_value = "x" * 5000
    with caplog.at_level(logging.WARNING):
        _resolve_mutator_result(big_value, hp, user_id="u1", path_label="atomic")
    msg = " ".join(rec.message for rec in caplog.records)
    # repr truncado a ~100 chars + comillas + escapes; razonable bajo 200.
    assert len(msg) < 1000, f"WARNING demasiado largo: {len(msg)} chars"


# ---------------------------------------------------------------------------
# 2. Estructura: SSOT — call sites usan el helper
# ---------------------------------------------------------------------------
def test_call_sites_use_helper_not_inline():
    """Los 2 call sites originales (fallback ~266, atomic ~301) deben
    invocar `_resolve_mutator_result(...)`, no el patrón inline
    `result if isinstance(result, dict) else hp` que duplicaba lógica."""
    src = _DB_PROFILES.read_text(encoding="utf-8")
    # El patrón inline ejecutable ya no debe aparecer fuera del helper.
    inline_pattern = r"new_hp\s*=\s*result\s+if\s+isinstance\(\s*result\s*,\s*dict\s*\)"
    inline_count = len(re.findall(inline_pattern, src))
    assert inline_count == 0, (
        f"Patrón legacy `new_hp = result if isinstance(result, dict)` aún "
        f"aparece {inline_count}x. Debe ser reemplazado por "
        f"`_resolve_mutator_result(...)` en los 2 call sites."
    )


def test_helper_invoked_from_both_paths():
    """`_resolve_mutator_result` debe invocarse al menos 2 veces (un call
    desde fallback path, otro desde atomic path)."""
    src = _DB_PROFILES.read_text(encoding="utf-8")
    invocations = re.findall(r"_resolve_mutator_result\s*\(", src)
    # 2 call sites + la def → al menos 3 ocurrencias.
    assert len(invocations) >= 3, (
        f"Helper invocado {len(invocations)} veces; esperado ≥3 (def + 2 call sites). "
        f"Si bajó, P3-4 puede haber sido revertido."
    )


def test_path_label_distinguishes_fallback_vs_atomic():
    """Cada call site pasa un `path_label` distinto para que el WARNING
    indique si el bug viene del path atómico o del fallback."""
    src = _DB_PROFILES.read_text(encoding="utf-8")
    # Buscar invocaciones con path_label.
    assert re.search(r'path_label\s*=\s*["\']fallback["\']', src), (
        "Call site del fallback path debe pasar `path_label='fallback'`."
    )
    assert re.search(r'path_label\s*=\s*["\']atomic["\']', src), (
        "Call site del atomic path debe pasar `path_label='atomic'`."
    )


# ---------------------------------------------------------------------------
# 3. Smoke regresivo
# ---------------------------------------------------------------------------
def test_helper_signature_uses_keyword_only_for_metadata():
    """`user_id` y `path_label` deben ser keyword-only (después de `*`) para
    evitar errores posicionales sutiles si alguien copia mal la signature."""
    import inspect
    from db_profiles import _resolve_mutator_result
    sig = inspect.signature(_resolve_mutator_result)
    params = sig.parameters
    assert params["user_id"].kind == inspect.Parameter.KEYWORD_ONLY, (
        "`user_id` debe ser keyword-only para prevenir confusión posicional."
    )
    assert params["path_label"].kind == inspect.Parameter.KEYWORD_ONLY, (
        "`path_label` debe ser keyword-only."
    )


def test_helper_returns_hp_object_identity_on_none():
    """Cuando `result is None`, el helper debe retornar el MISMO objeto `hp`
    (no una copia) — eso permite que la mutación in-place del caller se
    refleje en lo que se persiste."""
    from db_profiles import _resolve_mutator_result
    hp = {"goal": "maintain"}
    hp_id = id(hp)
    result = _resolve_mutator_result(None, hp, user_id="u1", path_label="atomic")
    assert id(result) == hp_id, (
        "Helper debe retornar el mismo objeto hp (no copia) cuando result=None, "
        "para preservar la mutación in-place del caller."
    )
