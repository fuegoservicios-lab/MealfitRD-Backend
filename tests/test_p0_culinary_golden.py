# -*- coding: utf-8 -*-
"""[P0-CULINARY-GOLDEN · 2026-09-06] Línea base congelada + golden set humano.

Dos capas juzgan hoy la coherencia culinaria —el contrato determinista (V1-V5) y un juez LLM— y **de
ninguna se conoce su precisión ni su recall**, porque no hay verdad de referencia. Sin ella, «el juez
señala el 19,1 % de las comidas» no dice que el 19,1 % esté mal: dice que él lo cree. Calibrar un
umbral contra esa tasa es el overfitting que este repo ya pagó en agosto.

## Lo que se congela (P0-2)

`docs/culinary_baseline.json`, medido el 2026-09-06 sobre 96 planes y 1.186 comidas:

    contrato determinista ... 133 comidas   11,2 %
    juez culinario .......... 227 comidas   19,1 %
    coinciden ...............  23

**Las dos capas apenas se solapan**: 110 comidas las ve solo el determinista, 204 solo el juez. Por
eso se publican separadas y NUNCA sumadas en un índice único — fundirlas escondería que ninguna
sustituye a la otra.

Se congela ANTES de mejorar nada porque *una medición posterior al efecto no mide el efecto*.

## El golden set (P0-1)

80 comidas estratificadas: 20 solo-determinista, 20 solo-juez, 15 ambas y **25 sin ningún hallazgo**.
Ese último estrato es el que se suele omitir y el que impide engañarse: sin comidas limpias solo se
mide precisión, y **un detector que no dispara nunca sale perfecto**.

Tres decisiones que este test ancla porque son las que hacen la medición honesta:

1. **La muestra no la elige quien la mide.** Orden por `sha256` de la clave, no por azar ni fecha.
2. **`dudoso` no cuenta en ninguna dirección.** Forzar un binario donde no lo hay contamina la medida.
3. **El marcador se niega a dar cifras con menos de 20 etiquetas**, en vez de imprimir ceros que
   parecen un resultado.

Y una que no es de este fichero pero decide su valor: **las etiquetas las pone una persona.** Si las
pusiera el modelo, el marcador mediría el acuerdo del juez consigo mismo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

_BASE = _BACKEND / "docs" / "culinary_baseline.json"
_GOLDEN = _BACKEND / "docs" / "culinary_golden_set.json"
_SAMPLE_SRC = (_BACKEND / "scripts" / "culinary_golden_sample.py").read_text(encoding="utf-8")
_SCORE_SRC = (_BACKEND / "scripts" / "culinary_golden_score.py").read_text(encoding="utf-8")


# ── la línea base ─────────────────────────────────────────────────────────────────────────────
def test_la_linea_base_esta_congelada():
    """Sin fichero congelado no hay «antes» contra el que comparar, y toda mejora futura sería una
    afirmación sin cifra."""
    assert _BASE.exists()
    d = json.loads(_BASE.read_text(encoding="utf-8"))
    assert d["comidas"] > 500 and d["planes"] > 50


def test_las_dos_capas_se_publican_por_separado():
    d = json.loads(_BASE.read_text(encoding="utf-8"))
    assert "determinista" in d and "juez" in d
    assert "solapamiento" in d
    assert "indice" not in json.dumps(d).lower().replace("indice_", ""), (
        "no se fabrica un indice unico: las dos capas apenas coinciden")


def test_el_solapamiento_es_pequenno_y_por_eso_no_se_suman():
    """23 de 314. Si algun dia se parecieran, habria que reconsiderar publicarlas juntas — pero eso
    seria una decision con datos, no una simplificacion."""
    s = json.loads(_BASE.read_text(encoding="utf-8"))["solapamiento"]
    assert s["ambas"] < min(s["solo_determinista"], s["solo_juez"])


def test_la_advertencia_viaja_con_el_dato():
    """Como campo del JSON, no como comentario: quien lea el fichero suelto tiene que encontrarla."""
    d = json.loads(_BASE.read_text(encoding="utf-8"))
    assert "overfitting" in d["advertencia"].lower()
    assert "sin verdad de referencia" in d["advertencia"].lower()


# ── el golden set ─────────────────────────────────────────────────────────────────────────────
def test_el_golden_set_existe_y_esta_estratificado():
    assert _GOLDEN.exists()
    d = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    estratos = {c["estrato"] for c in d["casos"]}
    assert estratos == {"solo_determinista", "solo_juez", "ambas", "sin_hallazgo"}


def test_hay_comidas_SIN_hallazgo_en_la_muestra():
    """El estrato que mide el RECALL. Sin él solo se mediría precisión, y un detector que no dispara
    nunca saldría perfecto."""
    d = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    limpias = [c for c in d["casos"] if c["estrato"] == "sin_hallazgo"]
    assert len(limpias) >= 20
    assert all(not c["maquina_determinista"] and not c["maquina_juez"] for c in limpias)


def test_cada_caso_trae_lo_necesario_para_juzgarlo():
    d = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    for c in d["casos"][:10]:
        assert c["nombre"] and c["ingredientes"] and c["pasos"], c["id"]
        assert "veredicto_humano" in c and "nota_humana" in c


def test_la_muestra_no_la_elige_quien_la_mide():
    """Orden por `sha256` de la clave: reproducible, y sin la tentación de escoger los casos que
    confirman lo que uno espera."""
    assert "hashlib.sha256(repr(k).encode()).hexdigest()" in _SAMPLE_SRC
    assert "random" not in _SAMPLE_SRC.lower()


def test_no_se_sobrescribe_un_golden_set_ya_etiquetado():
    """Regenerarlo tiraría el trabajo de la persona que puso las etiquetas."""
    assert "ya tiene etiquetas humanas" in _SAMPLE_SRC
    assert "return 1" in _SAMPLE_SRC


def test_el_json_se_entrega_SIN_etiquetar():
    """Si el modelo las rellenara, el marcador mediría el acuerdo del juez consigo mismo."""
    d = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    puestas = [c["id"] for c in d["casos"] if str(c.get("veredicto_humano") or "").strip()]
    assert not puestas or len(puestas) == len(d["casos"]), (
        f"etiquetas a medias, o puestas por quien no debe: {puestas[:5]}")


# ── el marcador ───────────────────────────────────────────────────────────────────────────────
def _puntuar(casos, disponibles):
    sys.path.append(str(_BACKEND / "scripts"))  # [P1-PLAN-LOTE-13] al FINAL: en cabeza, scripts/plan_gym.py sombreaba a plan_gym
    from culinary_golden_score import puntuar
    return puntuar({"casos": casos, "disponibles_por_estrato": disponibles})


def _caso(estrato, det, juez, veredicto):
    return {"estrato": estrato, "maquina_determinista": (["x"] if det else []),
            "maquina_juez": (["y"] if juez else []), "veredicto_humano": veredicto}


def test_el_marcador_cuenta_bien_lo_evidente():
    casos = [_caso("ambas", True, True, "defecto"),      # tp de las dos
             _caso("solo_determinista", True, False, "ok"),   # fp del determinista
             _caso("sin_hallazgo", False, False, "defecto")]  # fn de las dos
    r = _puntuar(casos, {"ambas": 1, "solo_determinista": 1, "sin_hallazgo": 1})
    det = r["capas"]["determinista"]["crudo"]
    assert det["tp"] == 1 and det["fp"] == 1 and det["fn"] == 1


def test_dudoso_no_cuenta_en_ninguna_direccion():
    """Forzar un binario donde la persona dijo «no sé» contamina la medición."""
    casos = [_caso("ambas", True, True, "dudoso"), _caso("ambas", True, True, "defecto")]
    r = _puntuar(casos, {"ambas": 2})
    assert r["dudosos"] == 1 and r["etiquetados"] == 1
    assert r["capas"]["determinista"]["crudo"]["tp"] == 1


def test_el_ponderado_corrige_el_sesgo_del_muestreo():
    """15 de 23 «ambas» y 25 de 919 «limpias»: sin ponderar, los estratos raros mandan y la cifra es
    inventada. Se publican las dos, crudo y ponderado."""
    casos = [_caso("ambas", True, True, "defecto"), _caso("sin_hallazgo", False, False, "defecto")]
    r = _puntuar(casos, {"ambas": 1, "sin_hallazgo": 900})
    v = r["capas"]["determinista"]
    assert v["crudo"]["recall"] == 50.0
    assert v["ponderado"]["recall"] < 1.0, "el estrato limpio pesa 900x y hunde el recall real"


def test_con_pocas_etiquetas_avisa_en_vez_de_dar_un_numero():
    assert "no significan nada" in _SCORE_SRC


def test_el_marcador_no_inventa_una_nota_de_calidad():
    """Precisión y recall son propiedades del DETECTOR; la calidad del plan necesitaría un criterio
    de gravedad que nadie ha definido. Fabricar la nota sería darle a una opinión cara de medición."""
    assert "nota" not in _SCORE_SRC.split('"""')[2].lower() or "NO calcula" in _SCORE_SRC
    for inventado in ("score_global", "nota_calidad", "puntuacion_1_10"):
        assert inventado not in _SCORE_SRC
