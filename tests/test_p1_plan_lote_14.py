# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-14 · 2026-09-12] Decimocuarto lote del plan de pendientes: E2 y D8, MEDIDOS antes que construidos.

E2 · «Fase 1 gate: 2 kills + 7 días sin alertas ⇒ flip `MEALFIT_INITIAL_VIA_QUEUE`». Medido el 09-12: el flip llevaba
vivo desde ≤ 09-06 (`.env` del VPS y `.env.production` del frontend en `true`; `prod_profile.py` lo leyó el 09-06;
6/6 planes nuevos desde el 09-04 nacieron por la cola, de 2 usuarios; 0 alertas del lifecycle desde el canary del
09-02) y el plan seguía llamándolo pendiente. El gate en DB (`arq25_gate_status`) decía `ready_to_flip: false` con
`runs: 6, kills_recovered: 0`: las filas de `plan_generation_runs`/`plan_chunk_queue` del canary (usuario f47126cb)
se fueron en CASCADE con la purga de cuentas del 09-11. *Un gate que vive en filas que una purga borra no recuerda
que pasó.* Ahora el gate responde `phase`/`flip_live` leyendo el interruptor con la MISMA función que decide el 404
del endpoint, cuenta el canary sin listarlo, y avisa cuándo los contadores son informativos.

D8 · «`shopping_commercial` y medición del lag p95 de `plan_jobs`». Medido con `scripts/measure_plan_jobs_lag.py`
(read-only): la recogida del worker p95 = 101,6 s (display_i18n) / 9,4 s (shopping_projection), dentro de la cota de
2 min; el p95 «total» de 2.450 s de display_i18n eran el incidente del 09-08 (`already_enriched` → dead → revivido,
cerrado por P1-I18N-DEAD-VEREDICTO) y las cadenas `revision_changed` del reconcile, no el worker. Nueve de 14 jobs
limpios eran no-ops (0 s): meterlos en el p95 lo escondía. `shopping_commercial` no se construye: no hay consumidor
que la pida (decisión de producto, dueño).

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _script(name: str):
    """Carga `scripts/<name>.py` por RUTA, sin tocar sys.path (ratchet de LOTE-13: `scripts/` nunca en cabeza)."""
    import importlib.util
    p = _BACKEND / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_lote14_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _q_vacia(sql, params=None, fetch_one=False, fetch_all=False, **kw):
    """La DB tras la purga: ninguna fila del canary. El gate debe seguir diciendo la verdad."""
    return {} if fetch_one else []


# ─────────────────────────── E2 · el gate sabe si el flip está vivo ───────────────────────────

def test_gate_off_sin_knob_ni_canary(monkeypatch):
    monkeypatch.delenv("MEALFIT_INITIAL_VIA_QUEUE", raising=False)
    monkeypatch.delenv("MEALFIT_INITIAL_VIA_QUEUE_USERS", raising=False)
    from routers import system as S
    out = S.arq25_gate_status(_q_vacia)
    assert out["phase"] == "off" and out["flip_live"] is False and out["canary_users_configured"] == 0
    assert out["counts_scope"] == "gate"


def test_gate_canary_cuenta_sin_listar(monkeypatch):
    monkeypatch.delenv("MEALFIT_INITIAL_VIA_QUEUE", raising=False)
    u1, u2 = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"
    monkeypatch.setenv("MEALFIT_INITIAL_VIA_QUEUE_USERS", f"{u1}, {u2},{u1.upper()}")
    from routers import system as S
    out = S.arq25_gate_status(_q_vacia)
    assert out["phase"] == "canary" and out["flip_live"] is False
    assert out["canary_users_configured"] == 2, "misma uuid en dos grafías = una persona"
    assert u1 not in repr(out).lower() and u2 not in repr(out), "los uuids del canary son personas: se cuentan, no se listan"


def test_gate_flipped_marca_los_contadores_como_informativos(monkeypatch):
    monkeypatch.setenv("MEALFIT_INITIAL_VIA_QUEUE", "true")
    monkeypatch.delenv("MEALFIT_INITIAL_VIA_QUEUE_USERS", raising=False)
    from routers import system as S
    out = S.arq25_gate_status(_q_vacia)
    assert out["phase"] == "flipped" and out["flip_live"] is True
    assert out["counts_scope"] == "informational_post_flip"
    # El contrato previo sigue entero: con la DB purgada los contadores dicen la verdad («no pasa»), no se maquillan.
    assert out["runs"] == 0 and out["counts_ok"] is False and out["ready_to_flip"] is False
    for k in ("canary_since", "kills_recovered", "soak_ok", "days_since_last_lifecycle_alert", "soak_days_required",
              "duplicate_runs_same_plan", "fencing_rejected", "pending_pipeline_chunks"):
        assert k in out, k


def test_gate_lee_el_interruptor_con_el_ssot_y_no_con_otra_tabla():
    src = _src("routers/system.py")
    i = src.find("def arq25_gate_status")
    j = src.find('@router.get("/admin/arq25-gate")')
    assert 0 < i < j
    body = src[i:j]
    assert "from generation_lifecycle import initial_via_queue_enabled" in body
    assert "tooltip-anchor: arq25_gate_flip_live" in body
    assert '_env_bool("MEALFIT_INITIAL_VIA_QUEUE"' not in body, "no reimplementar el parseo del knob (P1-DIET-CANON-SSOT)"


def test_prod_profile_dice_que_el_flip_esta_vivo():
    import prod_profile
    assert prod_profile.PROD_KNOBS["MEALFIT_INITIAL_VIA_QUEUE"] == "true"
    assert prod_profile.PROD_KNOBS.get("MEALFIT_PLAN_JOBS_ENABLED") == "1"


# ─────────────────────────── D8 · la medición del lag de plan_jobs ───────────────────────────

def test_script_de_medicion_es_solo_lectura_y_repetible():
    src = _src("scripts/measure_plan_jobs_lag.py")
    assert "conn.read_only = True" in src and '"--json"' in src and "P2-LOGGER-EXEMPT" in src
    sin_comentarios = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert re.search(r"\b(INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|TRUNCATE|ALTER\s+TABLE|DROP\s+TABLE)\b",
                     sin_comentarios, re.I) is None, "sólo SELECTs"
    assert re.search(r"sys\.path\.insert\(\s*0", src) is None, "ratchet LOTE-13: scripts/ nunca en cabeza de sys.path"


def test_veredicto_juzga_la_recogida_y_no_colapsa_sin_datos():
    m = _script("measure_plan_jobs_lag")
    ok = m.veredicto({
        "display_i18n": {"p95_recogida_s": 101.6, "p95_consumo_s": 54.3, "p95_limpio_s": 180.9, "n_limpio": 5, "n_noop": 9},
        "shopping_projection": {"p95_recogida_s": 9.4, "p95_consumo_s": 1.7, "p95_limpio_s": 10.4, "n_limpio": 10, "n_noop": 6},
    }, dead_sin_alerta=0)
    assert ok["gate_ok"] is True and ok["por_tipo"]["display_i18n"]["recogida_ok"] is True
    assert ok["por_tipo"]["display_i18n"]["p95_limpio_s"] == 180.9, "el total (> 2 min) viaja como información: no decide"
    assert ok["por_tipo"]["display_i18n"]["n_noop"] == 9

    lento = m.veredicto({"display_i18n": {"p95_recogida_s": 180.0, "n_limpio": 5}}, 0)
    assert lento["gate_ok"] is False and lento["por_tipo"]["display_i18n"]["recogida_ok"] is False

    muerto = m.veredicto({"display_i18n": {"p95_recogida_s": 10.0, "n_limpio": 5}}, dead_sin_alerta=1)
    assert muerto["gate_ok"] is False and muerto["dead_sin_alerta"] == 1

    sin_datos = m.veredicto({"display_i18n": {"p95_recogida_s": None, "n_limpio": 0},
                             "shopping_projection": {"p95_recogida_s": 5.0, "n_limpio": 3}}, 0)
    assert sin_datos["gate_ok"] is None, "«no concluyente» no colapsa a ningún lado"
    assert sin_datos["por_tipo"]["display_i18n"]["recogida_ok"] is None
    assert m.veredicto({}, 0)["gate_ok"] is None
    assert m.GATE_P95_S == 120.0


def test_el_sql_separa_no_ops_de_trabajo_real_y_mide_la_recogida_con_el_heartbeat():
    src = _src("scripts/measure_plan_jobs_lag.py")
    assert "heartbeat_at - GREATEST(execute_after, created_at)" in src, "recogida = claim − momento en que era elegible"
    assert "processed_at <= created_at + INTERVAL '1 second'" in src and "AS n_noop" in src
    assert "a.alert_type = 'plan_jobs_dead' AND a.metadata->>'job_id' = j.id::text" in src, "dead SIN alerta se cuenta por job_id"


# ─────────────────────────── docs y marker ───────────────────────────

def test_los_docs_cuentan_el_lote():
    plan = _src("docs/plan_pendientes_2026_09_11.md")
    assert re.search(r"^\| E2 \| ✅ 2026-09-12 \|", plan, re.M), "E2 cerrado en el Estado del plan"
    assert re.search(r"^\| D8 \| 📏 medido", plan, re.M), "D8 medido en el Estado del plan"
    lc = _src("docs/generation_lifecycle_2_5.md")
    assert "Estado 2026-09-12" in lc and "CASCADE" in lc and "flip_live" in lc
    f5 = _src("docs/plan_jobs_f5.md")
    assert "Medición del 2026-09-12" in f5 and "measure_plan_jobs_lag.py" in f5 and "shopping_commercial" in f5


def test_marker_bumpeado():
    import app
    assert "[P1-PLAN-LOTE-14 · 2026-09-12]" in _src("app.py")
    # «no anterior a este lote», no «igual a hoy» (lección de LOTE-13).
    assert app._LAST_KNOWN_PFIX.split("·")[-1].strip() >= "2026-09-12", app._LAST_KNOWN_PFIX
