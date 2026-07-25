"""[P1-PAREN-GRAMS-CAP · 2026-07-25] Los caps por gramos solo veían los gramos LÍDER.

Caso vivo (plan `ea79db0e`, D1): **`½ conejo (aprox. 358 g en piezas)`** — 358 g de proteína en
una comida, por encima del techo `PORTION_CAP_PROTEIN_G=300`, y **ningún cap lo veía**. La razón
es una sola línea:

    m_g = _re.match(r"^\\s*(\\d+(?:[.,]\\d+)?)\\s*(?:g|gr|gramos)\\b", il)

`match` ancla al principio: la línea empieza por "½", así que `m_g` es None y toda la cadena de
caps por gramos (techo duro de 600, proteína 300, fruta acuosa 300) se salta entera.

Cuando la masa vive en el paréntesis, **esa masa es la cantidad real** — el número líder es una
presentación ("½ conejo", "1½ filetes"). `_resc` ya reescala el paréntesis además del número
líder, así que el resultado queda coherente por los dos lados.
"""
import pytest

import graph_orchestrator as go


@pytest.fixture(autouse=True)
def _grupo_proteina(monkeypatch):
    """Sin DB el clasificador de grupo no resuelve; se mockea para probar el cap, no el lookup."""
    monkeypatch.setattr(
        go, "_ingredient_macro_group",
        lambda s, db=None: "protein" if any(
            k in str(s).lower() for k in ("conejo", "tilapia", "calamar", "pollo")) else "other",
        raising=False,
    )
    yield


def _cap(linea):
    days = [{"day": 1, "meals": [{"name": "Plato", "ingredients": [linea]}]}]
    n = go._cap_unrealistic_portions(days)
    return days[0]["meals"][0]["ingredients"][0], n


# ───────────── 1. el caso vivo ─────────────

def test_conejo_por_encima_del_techo_de_proteina():
    out, n = _cap("½ conejo (aprox. 358 g en piezas)")
    assert n >= 1, out
    assert "300 g" in out, f"la masa baja al techo exacto: {out}"


def test_el_parentesis_se_reescala_igual_que_el_numero_lider():
    """Si solo bajara uno de los dos, la línea quedaría contradiciéndose a sí misma."""
    out, _ = _cap("½ conejo (aprox. 358 g en piezas)")
    assert out.startswith("0.4"), out          # ½ × (300/358) ≈ 0.42
    assert "358" not in out


@pytest.mark.parametrize("prefijo", ["aprox. ", "~", "≈", ""])
def test_tolera_las_formas_del_parentesis(prefijo):
    out, n = _cap(f"½ conejo ({prefijo}358 g)")
    assert n >= 1 and "300 g" in out, out


# ───────────── 2. lo que NO debe tocar ─────────────

@pytest.mark.parametrize("linea", [
    "1½ filetes de tilapia (225 g)",     # proteína BAJO el techo
    "2 batatas medianas (302 g)",        # 302 g pero no es proteína
    "6½ láminas de casabe (95 g)",       # masa razonable
    "1 taza de yuca cocida",             # sin gramos declarados
])
def test_sin_cambios(linea):
    out, _ = _cap(linea)
    assert out == linea


def test_el_camino_de_gramos_lider_sigue_intacto():
    """La regresión que hay que impedir: romper el cap que ya funcionaba."""
    out, n = _cap("505 g de calamar")
    assert n >= 1 and out.startswith("300 g"), out


def test_knob_de_rollback():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'PAREN_GRAMS_CAP = _env_bool("MEALFIT_PAREN_GRAMS_CAP", True)' in src
    assert "if not m_g and PAREN_GRAMS_CAP:" in src, (
        "el paréntesis se consulta SOLO si no hay gramos líder — no puede pisar el camino exacto"
    )
