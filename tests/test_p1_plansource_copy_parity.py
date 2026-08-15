"""[P1-PLANSOURCE-COPY-PARITY · 2026-08-09] El paso 1 hacía creer que solo una de
las dos opciones usaba IA.

Decía «Plan completo con IA» vs «Desde mi Nevera». Sin querer, eso afirmaba dos
cosas falsas: que la segunda NO usa IA, y que es menos «completa». Las dos son
generación con IA y las dos entregan un plan entero — la única diferencia es de
dónde salen los ingredientes. El owner lo reportó como «no se entiende, las dos
son con IA».

Este test persigue el SIGNIFICADO, no la redacción: el copy puede reescribirse
libremente mientras (a) ninguna etiqueta reclame la IA para sí sola y (b) las dos
opciones se presenten en paralelo.

Tooltip-anchor: P1-PLANSOURCE-COPY-PARITY
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_Q = _REPO_ROOT / "frontend" / "src" / "components" / "assessment" / "questions" / "QPlanSource.jsx"
_FLOW = _REPO_ROOT / "frontend" / "src" / "components" / "assessment" / "InteractiveAssessmentFlow.jsx"


# [P1-I18N-DASHBOARD · 2026-08-15] El atributo puede venir en dos GRAFÍAS:
# el literal de siempre (`label="…"`) o envuelto en el traductor
# (`label={t('…')}`). Lo que estos tests vigilan es el TEXTO es-DO de las dos
# etiquetas —que ninguna reclame la IA en exclusiva, que ninguna se llame
# «completa»—, y ese texto es idéntico en ambas formas: la clave del catálogo
# ES el español. Anclar a la grafía ponía los 3 tests en rojo el día que la app
# se volvió multiidioma, sin que la propiedad vigilada hubiera cambiado un ápice.
_ATTR = r'(?:"([^"]+)"|\{t\(\'([^\']+)\'\)\})'


def _attr_value(m: re.Match) -> str:
    """Devuelve el texto es-DO venga de la comilla o del `t()`."""
    return next(g for g in m.groups()[-2:] if g is not None)


def _labels() -> dict[str, str]:
    """{valor -> label} de las dos RadioCard del paso."""
    src = _Q.read_text(encoding="utf-8")
    out = {}
    for val in ("scratch", "pantry"):
        m = re.search(rf'value="{val}"[^>]*?\n\s*label={_ATTR}', src, re.DOTALL)
        assert m, (
            f"P1-PLANSOURCE-COPY-PARITY: no encuentro el label de `{val}`. Se "
            f"aceptan `label=\"…\"` y `label={{t('…')}}`; si apareció una tercera "
            f"forma, extender `_ATTR`."
        )
        out[val] = _attr_value(m)
    return out


def test_no_label_claims_the_ai_for_itself():
    """LA REGLA. Si una etiqueta nombra la IA y la otra no, el usuario deduce que
    la otra no la usa — que es exactamente el reporte que originó este P-fix."""
    labels = _labels()
    menciona = {v: ("ia" in l.lower().replace("í", "i")) for v, l in labels.items()}
    assert len(set(menciona.values())) == 1, (
        "P1-PLANSOURCE-COPY-PARITY: una etiqueta nombra la IA y la otra no "
        f"({labels}). O las dos la nombran o ninguna: si solo una la reclama, la "
        "otra parece un modo manual. Las DOS son generación con IA."
    )


def test_no_label_claims_to_be_the_complete_one():
    """«Plan COMPLETO con IA» insinuaba que el otro modo entrega menos plan. Los
    dos entregan un plan entero; lo que cambia es el origen de los ingredientes."""
    for val, label in _labels().items():
        assert "complet" not in label.lower(), (
            f"P1-PLANSOURCE-COPY-PARITY: el label de `{val}` ({label!r}) se declara "
            "«completo». Ambos modos entregan un plan completo — el adjetivo solo "
            "puede leerse como que el otro está incompleto."
        )


def test_the_step_states_the_ai_once_and_up_front():
    """Decirlo en el título (antes de las dos tarjetas) evita que la duda nazca."""
    flow = _FLOW.read_text(encoding="utf-8")
    i = flow.index("<QPlanSource")
    bloque = flow[max(0, i - 900):i]
    assert re.search(r"title:.*\bIA\b", bloque), (
        "P1-PLANSOURCE-COPY-PARITY: el título del paso ya no nombra la IA. Nombrarla "
        "UNA vez por delante de las dos opciones es lo que impide que el usuario se "
        "pregunte si una de ellas es manual."
    )
    assert "Las dos opciones" in bloque or "ambas" in bloque.lower(), (
        "P1-PLANSOURCE-COPY-PARITY: el subtítulo dejó de decir explícitamente que las "
        "DOS son con IA. Esa frase es la que cierra la ambigüedad."
    )


def test_the_claim_about_not_reading_the_pantry_stays_true():
    """El copy del modo libre afirma que NO mira la Nevera. Es literal: el
    inventario se inyecta server-side solo en modo `pantry`. Si algún día el modo
    libre también lo consultara, la frase pasaría a ser mentira — este test ata
    la afirmación al comentario que documenta el mecanismo."""
    src = _Q.read_text(encoding="utf-8")
    m = re.search(rf'value="scratch".*?desc={_ATTR}', src, re.DOTALL)
    assert m, "P1-PLANSOURCE-COPY-PARITY: no encuentro la descripción de `scratch`"
    if "no mira" in _attr_value(m).lower():
        assert "server-side" in src, (
            "P1-PLANSOURCE-COPY-PARITY: el copy afirma que el modo libre no mira la "
            "Nevera, pero el archivo ya no documenta que la inyección del inventario "
            "es exclusiva del modo `pantry`. Sin ese anclaje la afirmación queda al "
            "aire y nadie sabrá que hay que revisarla si el mecanismo cambia."
        )


def test_the_two_values_are_untouched():
    """El copy es libre; los valores NO. `scratch`/`pantry` los consumen el backend
    y varios tests (test_p1_pantry_first_plan, test_p1_pantry_wizard_step)."""
    src = _Q.read_text(encoding="utf-8")
    for val in ("scratch", "pantry"):
        assert f'value="{val}"' in src, (
            f"P1-PLANSOURCE-COPY-PARITY: desapareció el valor `{val}`. Reescribir el "
            "copy nunca debe tocar los valores — el backend enruta con ellos."
        )
