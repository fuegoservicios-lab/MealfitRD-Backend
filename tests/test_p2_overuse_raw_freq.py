"""[P2-OVERUSE-RAW-FREQ · 2026-08-03] El veto textual «EVITA usar como base principal»
comparaba `freq >= OVERUSE_THRESHOLD` contra `db_freq_map` YA fatigado por recencia
(`recent_3d×3.0 + recent_7d×1.5`, `ai_helpers._apply_recency_fatigue`) — un ingrediente
usado UNA sola vez ayer da `1 + 1*3.0 + 1*1.5 = 5.5 >= 3` y entra al veto, aunque el propio
comentario de calibración (L~895-897) diga explícitamente que 1-2 usos NO deben marcarse
"PROHIBIDOS" porque el soft-penalty `1/(freq+1)` ya castiga lo suficiente. En planes de
15/30 días esto veta en el prompt los staples RECIÉN COMPRADOS.

Fix: capturar `raw_freq_map` ANTES de aplicar la fatiga; el veto textual (`used_proteins`/
`used_carbs`/`used_veggies`) se computa desde la frecuencia CRUDA. Los PESOS del sorteo
(`protein_weights = 1/(freq+1)`) siguen leyendo el mapa fatigado sin cambios — la fatiga
sigue siendo correcta para SESGAR la lotería; lo incorrecto era usarla también para el
umbral binario de veto.

Decisión medida y documentada (no solo asumida): con freq CRUDA histórica alta (4 usos)
pero SIN boost de recencia, el veto SIGUE disparando — 4 usos reales son sobreuso real,
la cruda manda en ambas direcciones. `test_cuatro_usos_sin_recencia_si_veta` ancla ese caso
explícitamente para que no se lea como regresión.

Confirmado por ejecución directa (git stash del fix, 30 semillas 0-29): el bug reproduce
en el 100% de las semillas con freq=1 reciente — no depende de random.choices, por eso
los tests no fijan seed.

Todo el archivo es OFFLINE: `get_user_ingredient_frequencies` y `get_user_profile` van
mockeados; cero conexión real a Neon.
"""
import ai_helpers as ah


def _mock_freqs(monkeypatch, base, r3, r7):
    """[P2-OVERUSE-RAW-FREQ] `_apply_recency_fatigue` llama a
    `get_user_ingredient_frequencies(user_id, days_limit=3|7)` y el flujo principal la llama
    SIN `days_limit` — un solo mock que discrimina por `days_limit` cubre ambas rutas y deja
    correr la fatiga REAL (no se mockea `_apply_recency_fatigue` en sí: este archivo prueba
    justamente su interacción con el snapshot pre-fatiga)."""
    def _fn(user_id, days_limit=None):
        if days_limit == 3:
            return dict(r3)
        if days_limit == 7:
            return dict(r7)
        return dict(base)
    monkeypatch.setattr(ah, "get_user_ingredient_frequencies", _fn, raising=False)
    monkeypatch.setattr(ah, "get_user_profile", lambda uid: {"health_profile": {}}, raising=False)


def _prompt(monkeypatch, base, r3, r7):
    _mock_freqs(monkeypatch, base, r3, r7)
    return ah.get_deterministic_variety_prompt(
        "", form_data={"mainGoal": "maintenance"}, user_id="u1")


def _bloque_veto(p: str) -> str:
    return p.split("EVITA usar como base principal")[-1] if "EVITA usar" in p else ""


def test_un_uso_reciente_no_veta(monkeypatch):
    """1 uso AYER (recent_3d=1, recent_7d=1) fatiga a 1 + 1*3.0 + 1*1.5 = 5.5 >= 3: bug
    pre-fix. Con el fix, el veto lee la cruda (freq=1 < 3) y NO debe aparecer."""
    p = _prompt(monkeypatch, {"pollo": 1}, {"pollo": 1}, {"pollo": 1})
    bloque = _bloque_veto(p)
    assert "pollo" not in bloque.lower(), (
        "1 uso ayer NO es sobreuso real; el veto textual sigue leyendo el mapa fatigado")


def test_cuatro_usos_sin_recencia_si_veta(monkeypatch):
    """4 usos históricos, CERO uso reciente (boost=0 ⇒ fatigado == crudo == 4). El veto debe
    seguir disparando en ambas direcciones: 4 usos reales SÍ son sobreuso, con o sin fix."""
    p = _prompt(monkeypatch, {"pollo": 4}, {}, {})
    bloque = _bloque_veto(p)
    assert "pollo" in bloque.lower(), (
        "4 usos reales (sin boost de recencia) deben vetarse igual: la cruda manda")


def test_knob_registrado_con_default_true():
    from knobs import _KNOBS_REGISTRY
    row = _KNOBS_REGISTRY.get("MEALFIT_OVERUSE_ON_RAW_FREQ")
    assert row is not None, "el knob debe auto-registrarse vía _env_bool (no os.environ crudo)"
    assert row["default"] is True


def test_knob_off_restaura_el_comportamiento_previo(monkeypatch):
    """Rollback sin redeploy: con el knob apagado, el veto vuelve a leer el mapa fatigado
    (comportamiento previo exacto, incluyendo el bug que este P-fix corrige)."""
    monkeypatch.setattr(ah, "OVERUSE_ON_RAW_FREQ", False)
    p = _prompt(monkeypatch, {"pollo": 1}, {"pollo": 1}, {"pollo": 1})
    bloque = _bloque_veto(p)
    assert "pollo" in bloque.lower(), (
        "con el knob OFF debe reaparecer el comportamiento previo (si no, el knob es cosmético)")
