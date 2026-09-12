# -*- coding: utf-8 -*-
"""[P1-MEASUREMENT-INTEGRITY · 2026-09-07] Cinco formas de que el medidor mienta a su favor.

Revisión del paquete `docs/audits/2026-09-07-coherencia-culinaria` contra el código. Sus P0-01 y
P0-02 señalaban defectos **de la capa de medición que yo mismo escribí el 6-sep**, y los cinco se
confirmaron leyendo el código. Todos comparten una forma: **cuando el medidor no puede medir,
informa del mejor valor posible.**

| defecto | antes | ahora |
|---|---|---|
| juez que no llegó a juzgar | `violations: []` + `warn_only` + sello | `status: "unavailable"`, sin sello |
| `scan_coverage` al reventar | `1.0` (cobertura perfecta) | `None` (desconocido) |
| marcador sin etiquetas | exit 0 con métricas `null` | exit 4 (incompleto) |
| identidad de comida | por franja: dos meriendas colisionan | por ocurrencia |
| línea base | `planes: 96` y nada más | huella del corpus + detección de deriva |

## Lo que hizo falta medir para no exagerar

Dos de los cinco venían descritos como «problema comprobado» y **no lo eran del todo**:

- La colisión de meriendas es **latente**: 0 de 1.182 comidas en los 96 planes vivos. Se cierra
  porque nada la impide y su pérdida sería silenciosa, no porque esté ocurriendo.
- La identidad por ocurrencia queda cerrada **solo del lado del corpus**: las violaciones de ambas
  capas identifican la comida por su FRANJA y no llevan índice, así que dos meriendas heredarían
  los mismos hallazgos. Cerrarlo entero es CUL-P0-01.

## Y uno que el paquete no vio, y era el peor

La línea base congelada el 6-sep **ya no era reproducible al día siguiente**: los MISMOS 96 planes
daban 1.186 comidas al congelarla y 1.182 catorce horas después. No es un error de conteo — el
corpus es `plan_data` VIVO y el shift del cron encoge la ventana de días de un plan existente (dos
planes mutaron a las 00:00 y 00:30 y se llevaron 4 comidas). Una re-medición no comparaba «antes vs
después del cambio»: comparaba dos corpus, con la deriva escondida dentro del delta.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import culinary_coherence as cc  # noqa: E402

_CAT = [{"name": "Pollo", "aliases": [], "category": "proteina",
         "ready_to_eat": False, "prep_methods": ["hervido"]}]
_PLAN = {"days": [{"meals": [{"meal": "almuerzo", "name": "Pollo",
                              "ingredients": ["150 g de pollo"], "recipe": ["Hierve el pollo"]}]}]}


# ── 1. cobertura: «no pude mirar» ya no es «perfecto» ─────────────────────────────────────────
def test_la_cobertura_devuelve_None_cuando_no_reconoce_ningun_alimento():
    """Antes devolvía 1,0. Es la telemetría con la que se decide escalar `warn → block`: un
    medidor ciego que informa del 100 % empuja justo hacia el lado peligroso."""
    assert cc.scan_coverage({"days": [{"meals": [{"ingredients": ["xyzzy"], "recipe": []}]}]},
                            _CAT) is None


def test_la_cobertura_devuelve_None_al_reventar():
    assert cc.scan_coverage(None, None) is None
    assert cc.scan_coverage({"days": "no soy una lista"}, _CAT) is None


def test_la_cobertura_sigue_midiendo_cuando_SI_puede():
    """El arreglo no puede convertirse en «devuelve None siempre»: eso apagaría la telemetría
    en vez de hacerla honesta."""
    cov = cc.scan_coverage(_PLAN, _CAT)
    assert cov is not None and 0.0 <= cov <= 1.0


def test_el_orquestador_persiste_el_None_en_vez_de_redondearlo():
    """`round(None, 3)` reventaría; y escribir 1,0 ahí devolvería el defecto por la puerta de
    atrás, esta vez en el dato persistido."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8", errors="ignore")
    i = src.index('plan["_culinary_contract_coverage"]')
    win = src[i:i + 220]
    assert "if _cul_cov is not None else None" in win, win[:200]


# ── 2. el juez que no llegó a juzgar ──────────────────────────────────────────────────────────
def test_un_juicio_que_no_ocurrio_no_se_guarda_como_limpio():
    """`run_culinary_judge` devuelve None con timeout/error/breaker abierto. Sin estado, eso se
    guardaba byte por byte igual que «juzgado, sin hallazgos» — y desde el sello, encima con
    apariencia de juicio válido sobre lo entregado."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8", errors="ignore")
    i = src.index("_cj_hist.append({")
    bloque = src[i:i + 600]
    assert '"status": _cj_status' in bloque, bloque[:300]
    assert '_cj_status = "judged" if _cj is not None else "unavailable"' in src


def test_un_juicio_que_no_ocurrio_NO_lleva_sello():
    """Sellar un `unavailable` afirmaría que esta versión del plan fue examinada."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8", errors="ignore")
    i = src.index("_cj_hist.append({")
    bloque = src[i:i + 600]
    assert '"judged_fingerprint": _cj_fingerprint(plan) if _cj is not None else None' in bloque


# ── 3. el marcador no puede aprobar en silencio ───────────────────────────────────────────────
def test_el_marcador_sale_con_codigo_4_si_faltan_etiquetas(tmp_path, monkeypatch):
    """Un aviso en pantalla solo lo ve quien lo lea. CI necesita el código de salida: exit 0 con
    métricas `null` es indistinguible de «medido y correcto».

    [2026-09-07] Este test corría el marcador contra el golden set REAL y se apoyaba en que
    estuviera sin etiquetar. El dueño etiquetó las 80 y se puso rojo — midiendo el ENTORNO, no el
    contrato, que es justo el defecto que este mismo fichero documenta en otros tres guards.

    Ahora fabrica un golden set incompleto y comprueba la regla sobre él. Se comprueban los DOS
    lados: incompleto ⇒ 4, completo ⇒ 0. Un guard que solo mira un lado no distingue «la regla
    funciona» de «esta rama nunca se ejecuta».
    """
    import json as _json
    import shutil

    origen = _BACKEND / "docs" / "culinary_golden_set.json"
    real = _json.loads(origen.read_text(encoding="utf-8"))

    def _correr(casos):
        raiz = tmp_path / "backend"
        (raiz / "docs").mkdir(parents=True, exist_ok=True)
        (raiz / "scripts").mkdir(parents=True, exist_ok=True)
        shutil.copy2(_BACKEND / "scripts" / "culinary_golden_score.py",
                     raiz / "scripts" / "culinary_golden_score.py")
        (raiz / "docs" / "culinary_golden_set.json").write_text(
            _json.dumps({**real, "casos": casos}, ensure_ascii=False), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(raiz / "scripts" / "culinary_golden_score.py")],
            capture_output=True, text=True, cwd=str(raiz)).returncode

    # 19 etiquetados: por debajo de MINIMO_ETIQUETAS (20)
    pocos = [dict(c, veredicto_humano=(c["veredicto_humano"] if i < 19 else ""))
             for i, c in enumerate(real["casos"])]
    assert _correr(pocos) == 4, "no avisó por código de salida con el golden set incompleto"

    # y con TODOS etiquetados sale limpio: si no, el 4 de arriba no probaría nada
    assert _correr(real["casos"]) == 0, "el marcador falla con el golden set completo"


def test_el_umbral_es_una_constante_y_no_dos_numeros_sueltos():
    """Estaba escrito a mano en el render y en el main: dos sitios que se separan al primer
    cambio, y el aviso diría una cosa mientras el código de salida hace otra."""
    src = (_BACKEND / "scripts" / "culinary_golden_score.py").read_text(encoding="utf-8")
    assert "MINIMO_ETIQUETAS = 20" in src
    assert src.count("MINIMO_ETIQUETAS") >= 3


# ── 4. identidad de la comida ─────────────────────────────────────────────────────────────────
def test_el_corpus_se_indexa_por_OCURRENCIA():
    src = (_BACKEND / "scripts" / "culinary_golden_sample.py").read_text(encoding="utf-8")
    assert 'comidas[(pid, di, mi, str(m.get("meal") or m.get("name")))]' in src
    assert "for mi, m in enumerate(d.get(\"meals\") or [])" in src


def test_el_join_con_los_hallazgos_sigue_funcionando():
    """El índice rompía el cruce: las violaciones solo traen `(día, franja)`. Un guard aparte
    porque el modo de fallo era silencioso — cero estratos, no una excepción."""
    src = (_BACKEND / "scripts" / "culinary_golden_sample.py").read_text(encoding="utf-8")
    assert "def _clave_hallazgo(k):" in src
    assert "det.get(_clave_hallazgo(k), [])" in src
    assert "juez.get(_clave_hallazgo(k), [])" in src


def test_el_limite_de_la_identidad_por_ocurrencia_queda_escrito():
    """Queda cerrada del lado del corpus y ABIERTA del lado de los hallazgos. Sin decirlo, el
    siguiente lector supondría que dos meriendas ya se distinguen de punta a punta."""
    src = (_BACKEND / "scripts" / "culinary_golden_sample.py").read_text(encoding="utf-8")
    assert "no llevan indice" in src and "CUL-P0-01" in src


# ── 5. la línea base tiene que ser reproducible ───────────────────────────────────────────────
def test_la_medicion_graba_la_huella_del_corpus():
    src = (_BACKEND / "scripts" / "culinary_baseline.py").read_text(encoding="utf-8")
    assert '"huella": huella_corpus.hexdigest()[:16]' in src
    assert 'huella_corpus.update(f"{pid}:{len(_dias)}:".encode())' in src, (
        "el nº de dias es justo lo que el shift mueve: sin el, la huella no ve la deriva")


def test_comparar_corpus_distintos_avisa_en_vez_de_restar():
    sys.path.append(str(_BACKEND / "scripts"))  # [P1-PLAN-LOTE-13] al FINAL: en cabeza, scripts/plan_gym.py sombreaba a plan_gym
    from culinary_baseline import _corpus_comparable, render
    a = {"corpus": {"huella": "aaaa"}}
    b = {"corpus": {"huella": "bbbb"}}
    assert _corpus_comparable(a, a) is True
    assert _corpus_comparable(a, b) is False
    r = {"planes": 1, "comidas": 1, "corpus": {"huella": "aaaa"},
         "determinista": {"comidas": 0, "pct": 0.0, "por_check": {}},
         "juez": {"comidas": 0, "pct": 0.0, "por_tipo": {}},
         "solapamiento": {"ambas": 0, "solo_determinista": 0, "solo_juez": 0},
         "advertencia": "x"}
    assert "EL CORPUS CAMBIO" in render(r, b)


def test_una_foto_SIN_huella_es_desconocido_no_comparable():
    """Las fotos anteriores a este P-fix no la llevan. Tratarlas como comparables presentaría la
    deriva del corpus como el efecto de un cambio de código."""
    sys.path.append(str(_BACKEND / "scripts"))  # [P1-PLAN-LOTE-13] al FINAL: en cabeza, scripts/plan_gym.py sombreaba a plan_gym
    from culinary_baseline import _corpus_comparable
    assert _corpus_comparable({"corpus": {"huella": "aaaa"}}, {"planes": 96}) is None
    assert _corpus_comparable({"corpus": {"huella": "aaaa"}}, None) is None


def test_la_foto_congelada_lleva_huella_de_una_medicion_REAL():
    """La foto se re-congeló el 7-sep, ya con huella. La condición no es que exista: es que sea
    consistente con el resto del fichero — una huella escrita a mano sería peor que ninguna,
    porque afirmaría comparabilidad sin respaldarla."""
    d = json.loads((_BACKEND / "docs" / "culinary_baseline.json").read_text(encoding="utf-8"))
    c = d["corpus"]
    assert len(c["huella"]) == 16 and all(ch in "0123456789abcdef" for ch in c["huella"])
    assert len(c["plan_ids"]) == d["planes"], (
        "los ids y el nº de planes discrepan: la huella no salió de esta medición")
    assert d["comidas"] > 500
