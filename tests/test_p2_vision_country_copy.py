"""[P2-VISION-COUNTRY-COPY · 2026-08-21] La limitación del escáner se aceptó CON una condición, y
la condición no se cumplió.

El escáner de comida es dominicano por diseño: `vision_agent.py` le dice al modelo «Eres un
nutricionista dominicano», le pide el nombre del plato «en español dominicano» y le da ejemplos
criollos («Los tres golpes» = mangú + …). No recibe país.

La spec de visión declaró eso **limitación aceptada de v1** — una decisión legítima; curar un
prompt de visión por país es trabajo de contenido. Pero la aceptó con una condición escrita: *«se
documenta en el copy del escáner para beta»*.

Medido: ese copy **no existía**. Ni una palabra en `ScanMealModal.jsx`. O sea que la parte barata
del trato —la que convertía una limitación en algo que el usuario puede compensar— nunca se pagó, y
lo que quedaba era la deuda de una decisión que nadie escribió.

POR QUÉ EL AVISO ES ACCIONABLE Y NO UN DESCARGO. El modal ya deja editar el nombre y las macros
antes de registrar. Decirle a un usuario de España que el escáner está calibrado con cocina
dominicana le explica por qué su plato salió con un nombre criollo y le dice exactamente qué hacer:
revisarlo antes de guardar. Sin el aviso, la única lectura posible es que el sistema se equivocó sin
motivo.

Byte-identidad dominicana: el aviso no se monta para RD, ni sin país, ni con el knob de UI apagado.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_FRONT = Path(__file__).resolve().parent.parent.parent / "frontend"
_MODAL = _FRONT / "src" / "components" / "dashboard" / "ScanMealModal.jsx"
_VISION = Path(__file__).resolve().parent.parent / "vision_agent.py"


@pytest.fixture(scope="module")
def modal() -> str:
    if not _MODAL.is_file():
        pytest.skip("ScanMealModal.jsx no está en este árbol")
    return _MODAL.read_text(encoding="utf-8", errors="replace")


def _pos_del_aviso(modal: str) -> int:
    """Posición de la CADENA que ve el usuario, no de la prosa que la explica.

    La primera versión usaba `modal.index("calibrad")` y caía en mi propio comentario JSX, que
    también habla de «contra qué está calibrado». Es la enésima vez que un comentario derrota a un
    guard en este repo — y van varias mías. El remedio es el de siempre: anclar en la forma que
    SÓLO el código tiene, aquí la llamada `t('…')`."""
    m = re.search(r"t\('El escáner está calibrad", modal)
    assert m, "no se encontró la llamada `t(...)` del aviso (¿se reescribió el copy?)"
    return m.start()


def test_el_escaner_sigue_siendo_dominicano(modal):
    """Control de la premisa: si el prompt de visión dejara de ser dominicano, el aviso sobra y
    este P-fix habría que revisarlo entero. Que falle aquí es una buena noticia, no un bug."""
    src = _VISION.read_text(encoding="utf-8", errors="replace")
    assert "nutricionista dominicano" in src, (
        "el prompt de visión ya no se declara dominicano: revisa si el aviso del modal sigue "
        "diciendo la verdad"
    )


def test_existe_el_aviso_que_la_spec_exigia(modal):
    """La condición del trato: «se documenta en el copy del escáner para beta»."""
    assert re.search(r"t\('El escáner está calibrado con cocina dominicana", modal), (
        "el escáner sigue sin decirle al usuario beta contra qué está calibrado — la condición con "
        "la que la spec aceptó la limitación"
    )


def test_el_aviso_dice_QUE_HACER(modal):
    """Un aviso que sólo se disculpa es un descargo. El modal deja editar nombre y macros, así que
    el aviso tiene que apuntar a esa acción."""
    i = _pos_del_aviso(modal)
    aviso = modal[i:i + 400]
    assert re.search(r"[Rr]evisa", aviso), (
        "el aviso no dice qué hacer: sin la acción, es un descargo de responsabilidad"
    )


def test_solo_se_monta_para_usuarios_beta(modal):
    """Byte-identidad dominicana: a un usuario de RD no le aparece nada nuevo."""
    i = _pos_del_aviso(modal)
    guarda = modal[max(0, i - 700):i]
    assert "DEFAULT_COUNTRY" in guarda and "COUNTRY_SYSTEM_UI" in guarda, (
        "el aviso no está gateado por país + knob de UI: se le mostraría también a un dominicano"
    )


def test_usa_el_ssot_de_paises(modal):
    """`coerceCountry`/`DEFAULT_COUNTRY` viven en `config/countries.js`, espejo del backend con test
    de paridad. Un `!== 'DO'` a mano aquí sería la tabla que P1-DIET-CANON-SSOT prohíbe."""
    assert re.search(r"from\s+'\.\./\.\./config/countries'", modal)
    i = _pos_del_aviso(modal)
    guarda = modal[max(0, i - 700):i]
    assert "coerceCountry" in guarda, "el modal compara el país a mano en vez de usar el SSOT"


def test_no_se_curo_un_prompt_de_vision_por_pais(modal):
    """Lo que este P-fix NO hace, dicho: no se escriben prompts de visión mexicano/español/…, que
    exigirían ejemplos de platos y equivalencias inventadas de memoria. La limitación sigue
    aceptada; lo que se paga es la mitad barata del trato."""
    src = _VISION.read_text(encoding="utf-8", errors="replace")
    assert "country" not in src.lower() or "nutricionista dominicano" in src, (
        "el prompt de visión empezó a ramificar por país: si son ejemplos curados con fuente, "
        "actualiza este test; si son de memoria, son datos fabricados"
    )
