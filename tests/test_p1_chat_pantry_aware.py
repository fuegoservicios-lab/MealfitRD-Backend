"""[P1-CHAT-PANTRY-AWARE · 2026-07-12] El agente conoce la Nevera REAL, no su memoria.

Vivo (owner): "agrega otra" → el agente confirmó "¡ya van 4 leches evaporadas!"
sumando su memoria conversacional — la fila real decía 6 (el owner había
borrado/editado items desde la UI, cosa que el chat no ve).

Dos capas:
  1. Snapshot de `user_inventory` (cantidades reales AHORA) inyectado al
     system prompt en cada turno — bloque VOLÁTIL al final (no rompe el
     prefix-cache P2-CHAT-PROMPT-STATIC-PREFIX). Kill-switch:
     MEALFIT_CHAT_PANTRY_SNAPSHOT (default ON).
  2. `modify_pantry_inventory` anexa al ToolMessage el estado REAL post-cambio
     de los items tocados ("📊 Estado REAL...: Leche evaporada: 6 lata (Wala)")
     — la confirmación del agente sale de la DB, no de contar mensajes.
tooltip-anchor: P1-CHAT-PANTRY-AWARE
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "agent.py"), encoding="utf-8") as f:
    _AG = f.read()
with open(os.path.join(_BACKEND, "tools.py"), encoding="utf-8") as f:
    _TL = f.read()


# ---------------------------------------------------------------------------
# Capa 1: snapshot en el system prompt
# ---------------------------------------------------------------------------

def test_pantry_context_injected_in_both_paths():
    assert "def _build_pantry_context" in _AG
    assert _AG.count("system_prompt += _build_pantry_context(user_id)") >= 2, \
        "ambos paths (non-stream y stream) deben inyectar el snapshot"


def test_pantry_context_is_killswitchable_and_guest_safe():
    i = _AG.find("def _build_pantry_context")
    win = _AG[i:i + 2500]
    assert "MEALFIT_CHAT_PANTRY_SNAPSHOT" in win, "kill-switch sin redeploy"
    assert 'user_id == "guest"' in win, "guests no tienen user_inventory"
    assert "quantity > 0" in win and "LIMIT 120" in win, \
        "solo items presentes, capado (costo de tokens acotado)"


def test_pantry_context_functional(monkeypatch):
    import agent as ag
    import db

    def _fake_q(sql, params=None, fetch_all=False, **kw):
        assert "user_inventory" in sql
        return [
            {"ingredient_name": "Leche evaporada", "quantity": 6.0, "unit": "lata", "brand": "Wala"},
            {"ingredient_name": "Huevo", "quantity": 3.0, "unit": "cartón", "brand": None},
        ]

    monkeypatch.setattr(db, "execute_sql_query", _fake_q)
    out = ag._build_pantry_context("11111111-1111-1111-1111-111111111111")
    assert "NEVERA FÍSICA AHORA (2 items" in out
    assert "Leche evaporada 6 lata (Wala)" in out
    assert "Huevo 3 cartón" in out
    # Guests → vacío, sin tocar DB.
    assert ag._build_pantry_context("guest") == ""
    assert ag._build_pantry_context(None) == ""


def test_pantry_context_empty_fridge(monkeypatch):
    import agent as ag
    import db
    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: [])
    out = ag._build_pantry_context("11111111-1111-1111-1111-111111111111")
    assert "vacía" in out


# ---------------------------------------------------------------------------
# Capa 2: la tool devuelve el estado real post-cambio
# ---------------------------------------------------------------------------

def test_tool_appends_real_state_for_touched_items():
    i = _TL.find("def modify_pantry_inventory")
    assert i != -1
    body = _TL[i:i + 12000]
    assert "_touched_names" in body, "los items tocados se rastrean"
    assert "Estado REAL en la Nevera tras el cambio" in body, \
        "el ToolMessage lleva los totales reales — la LLM confirma con la DB"
    assert "Ya no quedan:" in body, "items agotados/eliminados también se reportan"
    # Los 3 paths alimentan el tracking.
    assert body.count("_touched_names.add") >= 3, \
        "adds, depletes y removes deben registrar sus nombres"
