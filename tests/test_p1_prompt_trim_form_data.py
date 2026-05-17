"""[P1-PROMPT-TRIM-FORM-DATA · 2026-05-15] Regression guards para el helper
`_sanitize_form_data_for_prompt` que reduce el dump del `form_data` inyectado
al prompt de `plan_skeleton_node` y `generate_single_day`.

Pre-fix: `json.dumps(form_data, indent=2)` volcaba TODAS las claves del
form_data, incluyendo ~25 claves pipeline-internal con prefijo `_` (e.g.
`_chunk_lessons`, `_adherence_hint`, `_emotional_state`, `_days_offset`).
Estas claves ya están absorbidas en los `ctx[...]` que se inyectan
separadamente → duplicación + ruido en el prompt (~1-1.5K tokens).

Fix: helper que filtra claves con prefijo `_` antes del dump. Knob
`MEALFIT_PROMPT_TRIM_FORM_DATA` (default True, kill switch).

Tests parser-based + funcionales sobre el helper.
"""
from __future__ import annotations

import re
import sys
import importlib
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GRAPH_PATH = _BACKEND_ROOT / "graph_orchestrator.py"


def _read_graph() -> str:
    return _GRAPH_PATH.read_text(encoding="utf-8")


# ----- Knob registrado + helper definido --------------------------------

def test_knob_registered():
    text = _read_graph()
    assert "MEALFIT_PROMPT_TRIM_FORM_DATA" in text, (
        "P1-PROMPT-TRIM-FORM-DATA: knob `MEALFIT_PROMPT_TRIM_FORM_DATA` "
        "debe estar declarado en graph_orchestrator.py."
    )
    m = re.search(
        r'PROMPT_TRIM_FORM_DATA\s*=\s*_env_bool\(\s*[\"\']MEALFIT_PROMPT_TRIM_FORM_DATA[\"\']\s*,\s*(True|False)\s*\)',
        text,
    )
    assert m and m.group(1) == "True", (
        "Default debe ser `True` — el trim debe estar habilitado desde el deploy."
    )


def test_helper_defined_top_level():
    text = _read_graph()
    assert re.search(
        r"^def _sanitize_form_data_for_prompt\(form_data:\s*dict\)\s*->\s*dict:",
        text,
        re.MULTILINE,
    ), (
        "Helper `_sanitize_form_data_for_prompt(form_data: dict) -> dict` "
        "debe estar definido top-level."
    )


# ----- Callsites del helper -----------------------------------------------

def test_planner_callsite_uses_sanitize_helper():
    """`plan_skeleton_node` debe pasar `_sanitize_form_data_for_prompt(form_data)`
    a `json.dumps` en la construcción del prompt."""
    text = _read_graph()
    # Quitar líneas-comentario (empiezan con `#`) para no contar callsites
    # mencionados en docstrings/comentarios explicativos del fix.
    code_only = "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )
    direct = re.findall(r"json\.dumps\(form_data,\s*indent=2\)", code_only)
    assert len(direct) == 0, (
        f"P1-PROMPT-TRIM-FORM-DATA: encontré {len(direct)} callsite(s) en "
        f"código (sin contar comentarios) con `json.dumps(form_data, indent=2)` "
        f"directo — deben usar `_sanitize_form_data_for_prompt(form_data)`."
    )
    via_helper = re.findall(
        r"json\.dumps\(_sanitize_form_data_for_prompt\(form_data\),\s*indent=2\)",
        code_only,
    )
    assert len(via_helper) >= 2, (
        f"Esperaba ≥2 callsites via helper (planner + day_generator), "
        f"encontré {len(via_helper)}."
    )


# ----- Comportamiento funcional del helper -------------------------------
# Importamos el helper directamente. Para evitar disparar conftest (que requiere
# Supabase), insertamos el directorio backend en sys.path y hacemos un import
# ligero del módulo helpers.

@pytest.fixture(scope="module")
def helper_fn():
    """Importa solo el helper sin disparar todo graph_orchestrator (que importa
    langchain, supabase, etc. — pesado y frágil en test env).
    Estrategia: AST-extract o exec en namespace mínimo. Aquí usamos exec del
    snippet específico tras leer source — no perfecto pero suficiente para
    validar la lógica del helper, que es 5 líneas puras."""
    text = _read_graph()
    # Extraer la región del helper + el knob que usa.
    knob_m = re.search(
        r'PROMPT_TRIM_FORM_DATA\s*=\s*_env_bool\([^)]+\)',
        text,
    )
    helper_m = re.search(
        r"def _sanitize_form_data_for_prompt\(form_data:\s*dict\)\s*->\s*dict:.*?(?=\n(?:def |class |[A-Z_]+\s*=))",
        text,
        re.DOTALL,
    )
    assert knob_m and helper_m
    # Construimos un namespace mínimo: PROMPT_TRIM_FORM_DATA hardcoded a True,
    # y exec del helper.
    ns: dict = {"PROMPT_TRIM_FORM_DATA": True}
    exec(helper_m.group(0), ns)
    fn = ns["_sanitize_form_data_for_prompt"]

    def with_flag(flag: bool):
        ns["PROMPT_TRIM_FORM_DATA"] = flag
        return fn

    return with_flag


def test_helper_removes_underscore_prefixed_keys(helper_fn):
    fn = helper_fn(True)
    form = {
        "user_id": "abc",
        "medicalConditions": ["diabetes"],
        "allergies": ["maní"],
        "_chunk_lessons": {"weak_signal": True},
        "_adherence_hint": "OK",
        "_chunk_prior_meals": ["pollo guisado"],
        "_pipeline_drift_alert": True,
        "_emotional_state": "fatigued",
        "_days_offset": 7,
    }
    out = fn(form)
    # User-facing keys preservadas
    assert "user_id" in out
    assert "medicalConditions" in out
    assert "allergies" in out
    # Pipeline-internal keys removidas
    assert "_chunk_lessons" not in out
    assert "_adherence_hint" not in out
    assert "_chunk_prior_meals" not in out
    assert "_pipeline_drift_alert" not in out
    assert "_emotional_state" not in out
    assert "_days_offset" not in out


def test_helper_does_not_mutate_input(helper_fn):
    fn = helper_fn(True)
    form = {"a": 1, "_b": 2}
    _ = fn(form)
    # Original preservado: el código backend sigue leyendo `form_data["_xxx"]`
    # para routing / context build, NO debe ver el dict trimmed.
    assert form == {"a": 1, "_b": 2}, (
        "Helper debe retornar COPIA, NO mutar el original. El código backend "
        "lee form_data['_xxx'] downstream."
    )


def test_helper_passthrough_when_knob_false(helper_fn):
    """Kill switch: `PROMPT_TRIM_FORM_DATA=False` debe retornar el dict
    sin cambios (legacy behavior, rollback sin redeploy)."""
    fn = helper_fn(False)
    form = {"a": 1, "_b": 2, "_chunk_lessons": {"weak_signal": True}}
    out = fn(form)
    assert out == form, (
        "Cuando knob=False el helper DEBE retornar el form_data sin tocar "
        "(rollback path)."
    )


def test_helper_handles_non_dict_input(helper_fn):
    """Defensa: si form_data llega como None o lista (no debería pasar pero
    el code path tiene `form_data or {}` en algunos sitios), no crashear."""
    fn = helper_fn(True)
    assert fn(None) is None
    assert fn([1, 2, 3]) == [1, 2, 3]
    assert fn("strings tampoco") == "strings tampoco"


def test_helper_handles_non_string_keys(helper_fn):
    """Defensa: si por error hay una key int/None, no crashear con
    `key.startswith('_')`."""
    fn = helper_fn(True)
    form = {"valid": 1, 42: "int_key", None: "none_key", "_drop_me": "bye"}
    out = fn(form)
    assert "valid" in out
    assert 42 in out
    assert None in out
    assert "_drop_me" not in out
