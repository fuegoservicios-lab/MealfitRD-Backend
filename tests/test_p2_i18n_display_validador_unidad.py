"""[P2-I18N-DISPLAY-VALIDADOR-CIEGO-A-LA-UNIDAD · 2026-08-23] «180 g» convertido a «180 oz» se
persistía entero: el validador comparaba las CIFRAS y nunca la UNIDAD.

`_conserva_las_cifras` (P2-DISPLAY-VALIDADOR-SIN-CIFRAS) protege contra una traducción que
pierda o invente un número. Pero una línea puede conservar todos sus números y cambiar de
magnitud: «180 g de pollo» → «180 oz of chicken» tiene exactamente las mismas cifras y es
cinco veces más comida. Un modelo que «localiza» unidades al traducir produce justo eso, y
el validador lo dejaba pasar como traducción buena. Y como el ingrediente canónico español
sigue dentro de la línea, el MOTOR resuelve bien; es el USUARIO el que lee una cantidad
falsa en su receta.

Estaba declarado CERRADO por reconciliación en el plan v2, y no lo estaba: el código de hoy
no mira la unidad.

LO QUE SE COMPARA Y LO QUE NO. Sólo las unidades de MASA y VOLUMEN, que son magnitud:
g/kg/mg, ml/l/cl, oz/lb. No se comparan las de cocina traducibles —taza→cup, cda→tbsp,
cdta→tsp, unidad→unit, diente→clove— porque ésas SE TRADUCEN y un guard que las exigiera
iguales tiraría todas las traducciones buenas. El criterio es: si la unidad es una
magnitud física, tiene que ser LA MISMA magnitud a los dos lados.

tooltip-anchor: P2-I18N-DISPLAY-VALIDADOR-CIEGO-A-LA-UNIDAD
"""
from __future__ import annotations

import pytest

from plan_display_i18n import _conserva_las_cifras, _conserva_la_unidad

_MARKER = "P2-I18N-DISPLAY-VALIDADOR-CIEGO-A-LA-UNIDAD"


@pytest.mark.parametrize("original,traducida", [
    ("180 g de Pechuga de pollo", "180 oz of chicken breast (Pechuga de pollo)"),
    ("250 ml de Leche", "250 fl oz of milk (Leche)"),
    ("1 kg de Arroz", "1 lb of rice (Arroz)"),
    ("500 g de Habichuelas", "500 kg de haricots (Habichuelas)"),
    ("2 l de Agua", "2 ml d'eau (Agua)"),
])
def test_una_unidad_de_magnitud_cambiada_se_descarta(original, traducida) -> None:
    """Las mismas cifras, otra magnitud: el control de cifras pasa y el de unidad no."""
    assert _conserva_las_cifras(original, traducida), "premisa: las cifras coinciden"
    assert not _conserva_la_unidad(original, traducida), (
        f"«{original}» → «{traducida}» conserva los números y CAMBIA la magnitud; se "
        f"persistiría como traducción y el usuario leería una cantidad falsa. [{_MARKER}]"
    )


@pytest.mark.parametrize("original,traducida", [
    ("180 g de Pechuga de pollo", "180 g of chicken breast (Pechuga de pollo)"),
    ("250 ml de Leche", "250 ml de lait (Leche)"),
    ("1 kg de Arroz", "1 kg di riso (Arroz)"),
    ("100 gr de Queso", "100 g de fromage (Queso)"),          # gr y g son la misma
    ("2 tazas de Arroz", "2 cups of rice (Arroz)"),            # unidad TRADUCIBLE: ok
    ("1 cda de Aceite", "1 tbsp of oil (Aceite)"),             # idem
    ("2 dientes de Ajo", "2 cloves of garlic (Ajo)"),          # partitivo traducible
    ("1 unidad de Huevo", "1 egg (Huevo)"),                    # la unidad puede desaparecer
    ("Sal al gusto", "Salt to taste (Sal)"),                   # sin unidad
])
def test_la_misma_magnitud_o_una_unidad_traducible_pasa(original, traducida) -> None:
    """La otra dirección, sin la cual el guard tira traducciones buenas."""
    assert _conserva_la_unidad(original, traducida), (
        f"«{original}» → «{traducida}» es una traducción legítima y se descartó. [{_MARKER}]"
    )


def test_el_decimal_con_coma_no_rompe_la_deteccion_de_unidad() -> None:
    assert _conserva_la_unidad("1.5 kg de Arroz", "1,5 kg de riz (Arroz)")
    assert not _conserva_la_unidad("1.5 kg de Arroz", "1,5 g de riz (Arroz)")


# ───────────────────── el validador ENTERO, no sólo el comparador ─────────────────────

def test_el_validador_cae_al_espanol_en_la_linea_con_unidad_cambiada() -> None:
    """La CONDUCTA: una línea con la magnitud cambiada se queda en español (fallback), y
    las demás se traducen. Es el contrato del validador — degradar por línea, jamás
    persistir una cantidad falsa."""
    from plan_display_i18n import _validate_and_build_display  # noqa: PLC0415

    original = {
        "name": "Pollo con arroz",
        "description": "Un plato sencillo.",
        "ingredients": ["180 g de Pechuga de pollo", "100 g de Arroz blanco"],
        "recipe": ["Cocina el pollo.", "Sirve con el arroz."],
    }
    item = {
        "i": 0,
        "name": "Chicken with rice",
        "description": "A simple dish.",
        "ingredients": ["180 oz of chicken breast (Pechuga de pollo)",   # magnitud CAMBIADA
                        "100 g of white rice (Arroz blanco)"],           # correcta
        "recipe": ["Cook the chicken.", "Serve with the rice."],
    }
    out = _validate_and_build_display(original, item)
    assert out, f"el validador descartó la comida entera en vez de degradar la línea [{_MARKER}]"
    ings = out.get("ingredients") or []
    assert ings[0] == "180 g de Pechuga de pollo", (
        f"la línea con «180 oz» se persistió como traducción: {ings[0]!r}. El usuario leería "
        f"cinco veces la cantidad real. [{_MARKER}]"
    )
    assert ings[1] == "100 g of white rice (Arroz blanco)", (
        f"la línea CORRECTA también cayó al español: el guard tira traducciones buenas [{_MARKER}]"
    )
