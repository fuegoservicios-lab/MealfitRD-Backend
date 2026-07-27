"""[P1-CAP-STRICTEST-WINS · 2026-07-26] El primer tope que casaba dejaba a los demás sin evaluar.

## El caso

`_cap_unrealistic_portions` aplica los topes de realismo como una **cascada de
`if factor is None`**: en cuanto una rama casa, las siguientes se saltan. Pero cada rama limita
una DIMENSIÓN distinta —gramos, tazas, cucharaditas, piezas contables— así que una línea podía
salir corregida en una y absurda en otra:

    "9½ tallos de apio (285g)"
       rama 1 (veg-volume, 250 g)  dispara   -> factor = 250/285
       rama 3 (tope de 4 tallos)   NO se evalúa
    resultado:  "8.33 tallos de apio (250g)"

Gramos arreglados, conteo intacto — y encima fraccionario, que ningún humano escribe en una
receta. El tope de 4 tallos existía desde P1-APIO-STALK-CAP (2026-07-07) y **no disparaba nunca
en una línea que declarase gramos**, que son casi todas. Es el modo de fallo "código presente,
efecto ausente" otra vez: el tope estaba escrito y no se ejecutaba.

## La regla

En la rama de CONTEO gana el más estricto: `factor = min(factor, cap/actual)`.

Se limita a esa rama a propósito. Las piezas contables son la única dimensión independiente de
la masa, así que `min()` ahí no puede quitar comida real que el tope de gramos ya haya
aprobado: recorta piezas y el gramaje lo sigue mandando la rama de gramos. Las ramas de tazas y
cucharaditas siguen en cascada.

## Verificado like-for-like

Suite completa de topes (`-k "cap or realism or portion or unrealistic or count"`, 26 archivos):

    sin el fix:  27 fallos
    con el fix:  26 fallos
    nuevos:       0        arreglados: 1

tooltip-anchor: P1-CAP-STRICTEST-WINS
"""
from __future__ import annotations

import re

import pytest

import graph_orchestrator as g


class _DB:
    """Macros proporcionales a los gramos declarados entre paréntesis."""

    def macros_from_ingredient_string(self, s):
        m = re.search(r"\((\d+)", str(s))
        gr = float(m.group(1)) if m else 0.0
        return {"protein": gr * 0.007, "carbs": gr * 0.03, "fats": gr * 0.002, "kcal": gr * 0.16}

    def lookup(self, s):
        return object()

    def _ingredient_macro_group(self, *a, **k):
        return None


def _cap(line: str) -> str:
    meal = {"name": "Plato", "meal": "Cena", "ingredients": [line], "ingredients_raw": [line],
            "protein": 5, "carbs": 30, "fats": 2, "cals": 150}
    g._cap_unrealistic_portions([{"meals": [meal]}], db=_DB())
    return meal["ingredients"][0]


def _leading_number(s: str) -> float:
    m = re.match(r"\s*(\d+(?:[.,]\d+)?)", s)
    return float(m.group(1).replace(",", ".")) if m else -1.0


# ───────────── 1. el caso que lo destapó ─────────────

def test_el_conteo_se_capea_aunque_los_gramos_ya_hayan_disparado():
    """Con gramos declarados, la rama de volumen vegetal disparaba primero y el tope de 4 tallos
    quedaba sin evaluar."""
    out = _cap("9½ tallos de apio (285g) cortados en bastones")
    assert _leading_number(out) <= 4.0, out


def test_sin_gramos_declarados_seguia_funcionando():
    """El camino que YA funcionaba no se toca: sin gramos, ninguna rama previa dispara."""
    out = _cap("9½ tallos de apio cortados en bastones")
    assert _leading_number(out) <= 4.0, out


# ───────────── 2. no se volvió más agresivo de la cuenta ─────────────

@pytest.mark.parametrize("linea", [
    "2 tallos de apio (60g)",
    "3 tomates cherry (30g)",
    "1 tallo de apio (30g)",
])
def test_lo_que_esta_dentro_del_tope_no_se_toca(linea):
    """`min()` solo puede recortar cuando el conteo EXCEDE su propio tope."""
    assert _cap(linea) == linea


def test_una_linea_sin_conteo_no_se_ve_afectada():
    linea = "250 g de pechuga de pollo"
    assert _cap(linea) == linea


def test_un_alimento_sin_tope_de_conteo_no_se_recorta_por_esta_via():
    """La tabla de topes es explícita: lo que no está en ella no lo capea esta rama."""
    linea = "12 hojas de albahaca fresca"
    assert _leading_number(_cap(linea)) == 12.0


# ───────────── 3. ancla de la clase ─────────────

def test_la_rama_de_conteo_ya_no_esta_en_la_cascada():
    """Ancla: si alguien devuelve la rama de conteo a `if factor is None`, el tope vuelve a no
    evaluarse en toda línea que declare gramos — sin que ningún test de topes falle, porque el
    síntoma solo aparece cuando DOS topes aplican a la vez."""
    import inspect
    src = inspect.getsource(g._cap_unrealistic_portions)
    # ⚠️ NADA de ventanas de bytes fijas: caducan al primer comentario que crezca (cinco tests
    # así se rompieron en una sola sesión). Se ancla entre dos límites ESTRUCTURALES: el marker
    # y la aplicación del factor, que es donde la rama de conteo termina por definición.
    i = src.index("P1-CAP-STRICTEST-WINS")
    j = src.index("if factor is not None and factor < 0.999:", i)
    bloque = src[i:j]
    assert "min(factor, _f_count)" in bloque, (
        "la rama de conteo debe quedarse con el tope MÁS ESTRICTO, no con el primero"
    )
    assert "if factor is None:" not in bloque, (
        "la rama de conteo no puede volver a la cascada"
    )


def test_las_ramas_de_tazas_y_cdtas_siguen_en_cascada():
    """El cambio se limitó a la rama de conteo — a propósito. Si alguien lo extiende a las de
    volumen, `min()` sí puede quitar comida real que el tope de gramos ya aprobó."""
    import inspect
    src = inspect.getsource(g._cap_unrealistic_portions)
    i_cup = src.index("_REALISM_CUP_LEAD_RE.match(il)")
    assert "if factor is None:" in src[max(0, i_cup - 400):i_cup]
