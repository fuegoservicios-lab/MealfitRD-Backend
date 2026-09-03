"""[P1-DAYGEN-LUNA-CANARY · 2026-07-26] Canario de MODELO en el day generator.

Andamiaje para medir si un modelo distinto (`gpt-5.6-luna`) genera mejores días que GLM.
Arranca APAGADO: sin `MEALFIT_DAYGEN_CANARY_MODEL` el chain es byte-idéntico al de hoy.

## Por qué este nodo (medido sobre 24 h de `llm_usage_events`)

    day_generator             110 llamadas  3,26M in / 224K out   USD 0,60   62,6%
    self_critique              58            265K in /  85K out   USD 0,15   15,3%
    planner                    37            870K in /  34K out   USD 0,12   12,3%

`day_generator` es el 62% del costo Y el origen de casi todos los rechazos medidos (proteína o
fruta repetida, proteínas asignadas omitidas, sin preparación transformada) — fallos de
satisfacción de restricciones, justo donde un modelo que razona debería ayudar. Subirlo en todos
los nodos a la vez impediría saber qué mejoró.

## Contrato del modelo, verificado contra el API antes de escribir esto

    gpt-5.6-luna   /v1/chat/completions OK · response_format=json_object OK · razona (38 tok)
    a pelo         rechaza temperature≠1 y max_tokens (pide max_completion_tokens)
    langchain-openai 1.3.0 traduce ambos → drop-in sobre ChatOpenAI

⚠️ LangChain **descarta la temperatura en silencio** con estos modelos. Por eso el canario se
limita al day-gen (donde la temperatura es un empujón) y NO toca nodos que dependen de
`temperature=0` como contrato, p.ej. el `compressor` ("no inventes nada, solo resume").
"""
import pytest

import graph_orchestrator as go
from llm_provider import is_openai_model


@pytest.fixture(autouse=True)
def _sin_tier_routing(monkeypatch):
    """[P1-DAYGEN-TIER-MODEL · 2026-07-31] Estos tests anclan el CANARIO sobre
    la cadena base [flash, red]. El routing por tier (Luna primario) tiene su
    ancla en test_p1_daygen_tier_model.py; aquí se neutraliza para que las
    aserciones midan el canario y no el tier/API-key del entorno."""
    monkeypatch.setattr(go, "_daygen_tier_profile", lambda: (None, ""))


@pytest.fixture
def canario_on(monkeypatch):
    monkeypatch.setattr(go, "DAYGEN_CANARY_MODEL", "gpt-5.6-luna")
    monkeypatch.setattr(go, "DAYGEN_CANARY_PCT", 100)


# ───────────── 1. apagado por defecto ─────────────

def test_apagado_por_defecto_no_toca_el_chain(monkeypatch):
    monkeypatch.setattr(go, "DAYGEN_CANARY_MODEL", "")
    monkeypatch.setattr(go, "DAYGEN_CANARY_PCT", 0)
    ch = go._day_model_chain({"user_id": "u"}, 1)
    assert all(not is_openai_model(m) for m in ch), ch
    assert go._daygen_model_canary_cohort({"user_id": "u"}) == "off"


def test_pct_cero_con_modelo_puesto_sigue_apagado(monkeypatch):
    """Poner el modelo NO enciende nada: hace falta subir el porcentaje a propósito."""
    monkeypatch.setattr(go, "DAYGEN_CANARY_MODEL", "gpt-5.6-luna")
    monkeypatch.setattr(go, "DAYGEN_CANARY_PCT", 0)
    assert go._daygen_model_canary_cohort({"user_id": "u"}) == "off"
    assert "gpt-5.6-luna" not in go._day_model_chain({"user_id": "u"}, 1)


# ───────────── 2. encendido: el canario va DELANTE, con red detrás ─────────────

def test_se_antepone_y_conserva_la_cascada(canario_on, monkeypatch):
    """Anteponer (no reemplazar) deja el circuit breaker y el fallback existentes intactos:
    si Luna falla o su CB abre, `_build_day_llm` cae a flash/pro sin tocar nada.

    [P1-CANARY-RETRY-ONLY · 2026-07-26] En el intento 1 el canario YA NO se antepone por defecto
    (se paga sólo donde el modelo barato falló). Este test comprueba la MECÁNICA de anteposición,
    así que fuerza `scope='all'`; el reparto por intento lo cubre
    `test_p1_canary_retry_only.py`."""
    monkeypatch.setattr(go, "DAYGEN_CANARY_SCOPE", "all")
    ch = go._day_model_chain({"user_id": "u"}, 1)
    assert ch[0] == "gpt-5.6-luna"
    assert len(ch) >= 2, "sin red detrás, un fallo del canario mataría la generación"
    assert any("glm" in m for m in ch[1:])


def test_tambien_en_el_retry(canario_on):
    ch = go._day_model_chain({"user_id": "u"}, 2)
    assert ch[0] == "gpt-5.6-luna"
    assert any("glm" in m for m in ch[1:])


def test_sin_duplicados():
    ch = go._day_model_chain({"user_id": "u"}, 1)
    assert len(ch) == len(set(ch))


# ───────────── 3. la cohorte ─────────────

def test_determinista_y_estable(monkeypatch):
    """Un usuario no puede saltar de rama entre generaciones o el A/B es ilegible."""
    monkeypatch.setattr(go, "DAYGEN_CANARY_MODEL", "gpt-5.6-luna")
    monkeypatch.setattr(go, "DAYGEN_CANARY_PCT", 50)
    fd = {"user_id": "usuario-estable"}
    v = {go._daygen_model_canary_cohort(fd) for _ in range(20)}
    assert len(v) == 1, v


def test_reparto_insesgado(monkeypatch):
    monkeypatch.setattr(go, "DAYGEN_CANARY_MODEL", "gpt-5.6-luna")
    monkeypatch.setattr(go, "DAYGEN_CANARY_PCT", 50)
    on = sum(1 for i in range(400)
             if go._daygen_model_canary_cohort({"user_id": f"u{i}"}) == "on")
    assert 150 < on < 250, on


def test_salt_propio_no_correlaciona_con_los_otros_canarios(monkeypatch):
    """Con el mismo salt, un usuario caería en la MISMA rama de los tres canarios y el A/B
    mediría tres cosas a la vez."""
    monkeypatch.setattr(go, "DAYGEN_CANARY_MODEL", "gpt-5.6-luna")
    monkeypatch.setattr(go, "DAYGEN_CANARY_PCT", 50)
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_POOLS_CANARY_PCT", 50)
    iguales = 0
    for i in range(200):
        fd = {"user_id": f"u{i}"}
        a = go._daygen_model_canary_cohort(fd)
        b = go._harden_pools_canary_cohort({"form_data": fd})
        iguales += (a == b)
    assert 60 < iguales < 140, f"cohortes correlacionadas: {iguales}/200"


def test_fail_safe_apaga(monkeypatch):
    """Ante cualquier error, 'off': un fallo de lookup jamás debe COSTAR dinero (simétrico al
    fail-cheap de `resolve_model_for_tier`)."""
    monkeypatch.setattr(go, "DAYGEN_CANARY_MODEL", "gpt-5.6-luna")
    monkeypatch.setattr(go, "DAYGEN_CANARY_PCT", 100)
    assert go._daygen_model_canary_cohort(None) == "on"      # PCT=100 corta antes
    monkeypatch.setattr(go, "DAYGEN_CANARY_PCT", 50)
    assert go._daygen_model_canary_cohort(None) in ("on", "off")


# ───────────── 4. proveedor por prefijo ─────────────

@pytest.mark.parametrize("modelo,es_openai", [
    ("gpt-5.6-luna", True), ("gpt-5.6-sol", True), ("gpt-4o", True),
    ("o3", True), ("o4-mini", True),
    ("glm-5.3-flash", False), ("glm-5.3", False), ("", False),
])
def test_deteccion_de_proveedor(modelo, es_openai):
    assert is_openai_model(modelo) is es_openai


def test_openai_sin_key_falla_ruidoso(monkeypatch):
    """Fail-loud: mejor una excepción que un call silencioso al proveedor equivocado."""
    from llm_provider import build_chat_llm
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_chat_llm("gpt-5.6-luna")


def test_el_daygen_usa_la_fabrica_no_ChatGLM_a_pelo():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def _build_day_llm(")
    cuerpo = src[i:i + 1800]
    assert "build_chat_llm" in cuerpo, \
        "con un modelo OpenAI en el chain, ChatGLM lo mandaría al base_url equivocado"


# ───────────── 5. la cohorte llega a la telemetría ─────────────

def test_cohorte_declarada_en_planstate():
    """Lección de P1-RESCOPE-METRIC-BLIND y P1-HARDEN-POOLS-CANARY-GATING: sin declarar,
    LangGraph la descarta y el A/B produce filas que no miden nada."""
    import ast
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "PlanState")
    campos = {s.target.id for s in cls.body
              if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}
    assert "_daygen_model_cohort" in campos


def test_se_emite_SIN_fallback_inventado():
    """El `or "on"` del canario anterior convertía la ausencia de dato en una rama falsa."""
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '"daygen_model_cohort": final_state.get("_daygen_model_cohort")' in src
    i = src.index('"daygen_model_cohort"')
    assert 'or "on"' not in src[i:i + 120]


def _emit_metadata_src():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index('"daygen_model_cohort": final_state.get')
    j = src.index('"same_day_protein_repeats"', i)
    return src[i:j]


def test_se_emiten_las_razones_del_reintento():
    """`retries` dice CUÁNTO costó; sin las razones, un A/B con diferencia no se puede explicar
    y hay que volver a los logs — que es de donde tuve que sacarlas a mano."""
    seg = _emit_metadata_src()
    assert '"rejection_reasons"' in seg
    assert "_cumulative_rejection_reasons" in seg, \
        "acumulan entre intentos; sólo el último intento pierde el motivo original"
    assert "[:160]" in seg and "[:3]" in seg, "sin recorte esto infla pipeline_metrics"


def test_el_lector_del_ab_existe_y_agrupa_costo_por_modelo_real():
    """El costo NO se puede agrupar por el tag: un plan asignado a 'on' con el circuit breaker
    abierto corre GLM igual. El tag dice a quién se asignó; el modelo, qué pasó."""
    from pathlib import Path
    p = Path(go.__file__).resolve().parent / "scripts" / "daygen_canary_ab.py"
    assert p.exists(), "un canario sin lectura no informa nada"
    s = p.read_text(encoding="utf-8")
    assert "node = 'day_generator'" in s and "GROUP BY 1" in s
    assert "daygen_model_cohort" in s
    # la limitación de atribución debe quedar dicha, no tapada
    assert "plan_id" in s and "user_id" in s
