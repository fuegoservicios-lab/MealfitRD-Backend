"""[P1-FIRST-PURCHASE-PAUSE · 2026-08-16] La autonomía de `initial_plan` cede UNA
vez para pedir la primera compra.

El incidente que lo motivó: un bloque generándose a las 00:06 con la Nevera a
CERO items y sin que nada se lo dijera al usuario ("¿cómo se generó el chunk 2 si
ni siquiera tenía alimentos en la nevera?"). No era un bug — era P1-CHUNK-AUTONOMY
funcionando (la lista de compras NACE del plan; bloquear sin compras interbloquea
el arranque). La decisión del dueño: la autonomía sigue siendo la regla, pero si
la lista YA fue entregada y el usuario JAMÁS marcó una compra, el siguiente bloque
se pausa UNA vez con CTA — y si no actúa, el recovery existente lo genera solo a
las 12h (TTL `CHUNK_PANTRY_EMPTY_TTL_HOURS`) en modo flexible.

Las cuatro condiciones del helper son cuatro maneras distintas de NO romper nada:
sin hechos → conducta previa; sin lista → cold start intacto (el interbloqueo que
la autonomía existe para evitar); marker presente → una pregunta por plan, no un
goteo de pausas; `is_restocked` → quien ya usa la Nevera conserva la autonomía.

La decisión vive DENTRO de `_pantry_gate_waiver_reason` (SSOT de las dos guardas,
P1-PANTRY-GATE-SSOT): ninguna guarda decide por su cuenta.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import cron_tasks as ct

_BACKEND = Path(__file__).resolve().parent.parent
_FRONTEND = _BACKEND.parent / "frontend"


def _facts(**overrides):
    base = {
        "user_id": "u-1",
        "shopping_list_delivered": True,
        "first_purchase_pause_at": None,
        "is_restocked": False,
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# 1. El helper: cuatro condiciones, cada una un no-pausar
# ─────────────────────────────────────────────────────────────────────────────

def test_aplica_con_lista_entregada_y_cero_compras():
    assert ct._first_purchase_pause_applies(_facts()) is True


def test_sin_hechos_no_opina():
    """`None` = fail-safe: la autonomía queda EXACTAMENTE como estaba."""
    assert ct._first_purchase_pause_applies(None) is False
    assert ct._first_purchase_pause_applies("no-es-dict") is False


def test_sin_lista_entregada_jamas_pausa():
    """El cold start: pedir que compre una lista que no existe es el
    interbloqueo que P2-CHUNK-AUTONOMY evita. Esta condición es la diferencia
    entre refinar el waiver y resucitar el incidente del dry-run 2026-07-10."""
    assert ct._first_purchase_pause_applies(_facts(shopping_list_delivered=False)) is False


def test_una_sola_vez_por_plan():
    assert ct._first_purchase_pause_applies(
        _facts(first_purchase_pause_at="2026-08-16T04:00:00+00:00")
    ) is False


def test_quien_ya_restockeo_conserva_la_autonomia():
    """Nevera vacía a mitad de plan de un usuario que SÍ compra = consumió lo
    que compró. Eso es la autonomía normal, no una primera compra pendiente."""
    assert ct._first_purchase_pause_applies(_facts(is_restocked=True)) is False


def test_knob_apagado_desactiva(monkeypatch):
    monkeypatch.setenv("MEALFIT_FIRST_PURCHASE_PAUSE", "false")
    assert ct._first_purchase_pause_applies(_facts()) is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. El waiver SSOT: dónde cede y dónde NO cambia nada
# ─────────────────────────────────────────────────────────────────────────────

def test_la_autonomia_cede_con_hechos_que_aplican():
    assert ct._pantry_gate_waiver_reason(
        chunk_kind="initial_plan", plan_facts=_facts()
    ) is None


def test_sin_hechos_la_autonomia_queda_intacta():
    assert ct._pantry_gate_waiver_reason(
        chunk_kind="initial_plan", plan_facts=None
    ) == "initial_plan_autonomy"


def test_con_marker_la_autonomia_vuelve():
    assert ct._pantry_gate_waiver_reason(
        chunk_kind="initial_plan",
        plan_facts=_facts(first_purchase_pause_at="2026-08-16T04:00:00+00:00"),
    ) == "initial_plan_autonomy"


def test_flexible_mode_gana_antes_que_la_primera_compra():
    """El orden de las exenciones NO cambia: un chunk ya flexibilizado por otra
    válvula (viability floor, snapshot stale) no debe re-pausarse — re-pausar
    flexibles es el loop que P1-PANTRY-GATE-SSOT cerró."""
    assert ct._pantry_gate_waiver_reason(
        chunk_kind="initial_plan",
        snapshot={"_pantry_flexible_mode": True},
        plan_facts=_facts(),
    ) == "flexible_mode"


def test_guest_gana_antes_que_la_primera_compra():
    assert ct._pantry_gate_waiver_reason(
        chunk_kind="initial_plan",
        fresh_inventory_source="guest",
        plan_facts=_facts(),
    ) == "guest"


def test_rolling_refill_no_cambia_con_ni_sin_hechos():
    """Los rolling/catchup nunca tuvieron waiver de autonomía: su pausa por
    nevera vacía es la conducta EXISTENTE y estos hechos no la tocan."""
    assert ct._pantry_gate_waiver_reason(chunk_kind="rolling_refill", plan_facts=_facts()) is None
    assert ct._pantry_gate_waiver_reason(chunk_kind="rolling_refill", plan_facts=None) is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. El loader de hechos: fail-safe por construcción
# ─────────────────────────────────────────────────────────────────────────────

def test_facts_mapea_la_fila(monkeypatch):
    monkeypatch.setattr(ct, "execute_sql_query", lambda *a, **k: {
        "user_id": "u-9",
        "shopping_list_delivered": True,
        "first_purchase_pause_at": None,
        "is_restocked_raw": "true",
    })
    f = ct._first_purchase_plan_facts("plan-x")
    assert f == {
        "user_id": "u-9",
        "shopping_list_delivered": True,
        "first_purchase_pause_at": None,
        "is_restocked": True,
    }


def test_facts_sin_fila_es_none(monkeypatch):
    monkeypatch.setattr(ct, "execute_sql_query", lambda *a, **k: None)
    assert ct._first_purchase_plan_facts("plan-x") is None


def test_facts_con_excepcion_es_none(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("db blip")
    monkeypatch.setattr(ct, "execute_sql_query", _boom)
    assert ct._first_purchase_plan_facts("plan-x") is None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Parser-based: el call site y el marker
# ─────────────────────────────────────────────────────────────────────────────

def _fuente_sin_comentarios() -> str:
    src = Path(ct.__file__).with_suffix(".py").read_text(encoding="utf-8")
    return "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))


def test_el_call_site_pasa_los_hechos_al_waiver():
    """Si el waiver deja de recibir `plan_facts`, la denegación es inalcanzable
    y el fix muere en silencio (facts=None ⇒ autonomía siempre)."""
    codigo = _fuente_sin_comentarios()
    ini = codigo.index("if _should_pause_for_empty_pantry(")
    bloque = codigo[ini: ini + 3000]
    assert "_first_purchase_plan_facts(meal_plan_id)" in bloque, (
        "El call site del pickup ya no carga los hechos del plan."
    )
    assert re.search(r"_pantry_gate_waiver_reason\([^)]*plan_facts=", bloque, re.S), (
        "El call site consulta el waiver SIN plan_facts: la autonomía nunca cede."
    )


def test_el_reason_es_awaiting_first_purchase_y_estampa_el_marker_antes():
    codigo = _fuente_sin_comentarios()
    ini = codigo.index("if _should_pause_for_empty_pantry(")
    bloque = codigo[ini: ini + 3000]
    pos_marker = bloque.find("_mark_first_purchase_pause(")
    pos_pausa = bloque.find('reason="awaiting_first_purchase"')
    assert pos_pausa != -1, (
        "La pausa de primera compra volvió al reason genérico: el banner diría "
        "que la nevera está rota cuando lo que falta es la primera compra."
    )
    assert pos_marker != -1 and pos_marker < pos_pausa, (
        "El marker una-vez-por-plan no se estampa ANTES de pausar: dos siblings "
        "concurrentes podrían pausar los dos."
    )


def test_el_marker_es_jsonb_set_con_user_id():
    """I2 (toda mutación filtra user_id) + I7 (jsonb_set quirúrgico, no full
    overwrite — un overwrite aquí necesitaría advisory lock)."""
    src = Path(ct.__file__).with_suffix(".py").read_text(encoding="utf-8")
    ini = src.index("def _mark_first_purchase_pause(")
    cuerpo = src[ini: src.index("\ndef ", ini + 10)]
    assert "jsonb_set" in cuerpo
    assert "_first_purchase_pause_at" in cuerpo
    assert re.search(r"AND\s+user_id\s*=\s*%s", cuerpo), (
        "El UPDATE del marker perdió el filtro AND user_id (invariante I2)."
    )
    assert "plan_data = %s" not in cuerpo, (
        "El marker pasó a full-overwrite: eso exige advisory lock (I7) y aquí "
        "no lo hay. Debe seguir siendo jsonb_set quirúrgico."
    )


def test_el_knob_existe_con_default_true():
    src = Path(ct.__file__).with_suffix(".py").read_text(encoding="utf-8")
    ini = src.index("def _first_purchase_pause_applies(")
    cuerpo = src[ini: src.index("\ndef ", ini + 10)]
    assert re.search(r"_env_bool\(\s*\"MEALFIT_FIRST_PURCHASE_PAUSE\"\s*,\s*True\s*\)", cuerpo), (
        "El knob MEALFIT_FIRST_PURCHASE_PAUSE con default True salió del helper."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Las superficies de copy: backend, Dashboard, i18n
# ─────────────────────────────────────────────────────────────────────────────

def test_copy_backend_para_el_reason_nuevo():
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    ini = src.find('"awaiting_first_purchase": {')
    assert ini != -1, (
        "El reason nuevo no tiene copy en el dict de notificaciones: caería al "
        "fallback empty_pantry («Tu nevera está vacía»), que describe otra cosa."
    )
    bloque = src[ini: ini + 600]
    for pieza in ('"title"', '"body"', '"cta"', '"url"'):
        assert pieza in bloque


def test_copy_dashboard_para_el_reason_nuevo():
    src = (_FRONTEND / "src" / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")
    ini = src.find("_reasonCopy = {")
    assert ini != -1
    bloque = src[ini: ini + 4000]
    assert "awaiting_first_purchase:" in bloque, (
        "El banner del Dashboard no distingue la primera compra: usaría el "
        "fallback genérico y el CTA perdería su razón."
    )


@pytest.mark.parametrize("locale", ["en-US", "pt-BR", "fr-FR", "it-IT"])
def test_i18n_cubre_el_titulo_nuevo(locale):
    import json
    catalogo = json.loads(
        (_FRONTEND / "src" / "i18n" / "locales" / f"{locale}.json").read_text(encoding="utf-8")
    )
    clave = "Tu primera compra está pendiente"
    assert catalogo.get(clave), (
        f"{locale} no traduce «{clave}»: el banner caería al español para ese idioma."
    )
