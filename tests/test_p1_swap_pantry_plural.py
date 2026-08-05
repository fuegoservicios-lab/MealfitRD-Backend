"""[P1-SWAP-PANTRY-PLURAL · 2026-08-05] El reparador de coherencia encuentra la
comida en la nevera aunque cambie el número.

CASO REAL QUE LO ORIGINA. El dueño pidió cambiar su desayuno. El LLM escribió
«revoltillo» en los pasos sin listar huevos, el guard de coherencia lo marcó, y
el reparador determinista de `P1-SWAP-COHERENCE-REPAIR` —que añade la línea que
falta -SOLO si el alimento está en la nevera- se declaró "fuera de nevera". Tres
intentos IDÉNTICOS, 422, plato original conservado.

Pero el usuario SÍ tenía huevos: su `user_inventory` decía «Huevo: 2 cartón (20
uds.)». El chequeo era `name in blob` (subcadena cruda) y el guard preguntaba por
el canónico PLURAL `huevos`. Como el plural es más largo que el singular,
`"huevos" in "huevo"` es False. Reproducido con la nevera real (45 ítems).

Esta era la causa dominante de los cambios de plato que no devolvían nada.

⚠️ El arreglo NO puede ser recortar plurales sobre subcadenas: esta casa tiene una
familia entera de bugs de ese tipo (`pollo` ⊂ `repollo`, `sal` ⊂ `salsa`, `res` ⊂
`fresco`, `guisa` ⊂ `guisantes`). Se empareja por TOKEN COMPLETO.

tooltip-anchor: P1-SWAP-PANTRY-PLURAL
"""
import pytest

from agent import pantry_contains_food, _pantry_singular_key


# ─────────────────────────── el caso reportado ───────────────────────────

def test_plural_del_guard_contra_singular_de_la_nevera():
    """El caso exacto: nevera «huevo», guard «huevos»."""
    blob = "huevo yogurt avena pan integral aguacate"
    assert pantry_contains_food(blob, "huevos") is True


def test_singular_del_guard_contra_plural_de_la_nevera():
    """Y la dirección contraria, que también ocurre."""
    blob = "huevos yogurt avena"
    assert pantry_contains_food(blob, "huevo") is True


def test_lo_que_no_esta_sigue_sin_estar():
    """El reparador JAMÁS puede inventar una compra."""
    blob = "huevo yogurt avena"
    assert pantry_contains_food(blob, "salmon") is False
    assert pantry_contains_food(blob, "carne de res") is False


# ──────────────── la familia de bugs de subcadena de esta casa ────────────────

@pytest.mark.parametrize("blob,comida", [
    ("repollo lechuga tomate", "pollo"),       # "pollo" ⊂ "repollo"
    ("salsa de tomate", "sal"),                # "sal" ⊂ "salsa"
    ("pescado fresco", "res"),                 # "res" ⊂ "fresco"
    ("guisantes verdes", "guisa"),             # "guisa" ⊂ "guisantes"
    ("chinola madura", "ola"),                 # "ola" ⊂ "chinola"
])
def test_no_reintroduce_los_falsos_positivos_por_subcadena(blob, comida):
    """Ninguno de estos debe casar por TOKEN.

    Si alguno diera True, el reparador añadiría a la receta un alimento que el
    usuario no tiene — exactamente lo que el modo pantry existe para impedir.
    """
    toks = [_pantry_singular_key(t) for t in comida.split()]
    disponibles = {_pantry_singular_key(t) for t in blob.split()}
    assert not all(t in disponibles for t in toks), (
        f"'{comida}' casó por token contra '{blob}' — falso positivo de la familia subcadena."
    )


# ──────────────────────────── la clave singular ────────────────────────────

@pytest.mark.parametrize("token,esperado", [
    ("huevos", "huevo"),
    ("huevo", "huevo"),
    ("res", "res"),        # 3 letras: intacto
    ("mas", "mas"),        # 3 letras: intacto
    ("sal", "sal"),        # 3 letras: intacto
    ("arroz", "arroz"),    # no termina en s
    ("", ""),
])
def test_clave_singular(token, esperado):
    assert _pantry_singular_key(token) == esperado


def test_multipalabra_exige_todos_los_tokens():
    """Un alimento de varias palabras necesita TODAS, no una."""
    assert pantry_contains_food("queso blanco fresco", "queso blanco") is True
    # 'queso' está, 'ricotta' no → no está.
    assert pantry_contains_food("queso blanco fresco", "queso ricotta") is False


def test_entradas_vacias_no_revientan():
    assert pantry_contains_food("", "huevos") is False
    assert pantry_contains_food("huevo", "") is False
    assert pantry_contains_food(None, "huevos") is False


def test_la_subcadena_previa_sigue_funcionando():
    """El arreglo solo AÑADE emparejamientos: lo que ya casaba, sigue casando."""
    assert pantry_contains_food("pechuga de pollo fresca", "pollo") is True
    assert pantry_contains_food("aceite de oliva extra virgen", "aceite de oliva") is True
