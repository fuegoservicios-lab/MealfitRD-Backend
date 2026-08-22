"""[P2-I18N-DOC-ALCANCE-MIENTE · 2026-08-21] La doc canónica declaraba «No» lo que se
traduce desde hace dos días.

`docs/i18n_dashboard.md` fijaba el alcance del sistema de idiomas, y su tabla decía:

    | Plan, recetas, lista de compras | **No** | Las genera el LLM en español… |

seguido de un párrafo que llamaba a eso «la consecuencia honesta». Falso desde
`P1-PLAN-DISPLAY-I18N` (2026-08-19), cuyo knob nace en `True`: plan, recetas y nombre
del plan se traducen por la capa `_display`, y la lista de compras sale bilingüe.

Y el copy de Configuración se lo repetía al usuario en cinco idiomas: «Tu plan y tus
recetas siguen en español».

POR QUÉ ESTE GAP MERECE SU PROPIO TEST y no es una errata: **es el que hizo que la
primera pasada de la auditoría del sistema de idiomas dejara fuera la superficie i18n
más cara del producto.** La doc canónica se leyó, se creyó, y la capa `_display` —con
sus lotes, su validador y su telemetría— no entró en el inventario. Una doc canónica
equivocada no confunde solo a las personas.

Lo que este test ancla es que la fila diga la verdad Y enlace a donde vive el detalle.
Un test que solo exigiera «no dice No» pasaría con la fila borrada.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_MARKER = "P2-I18N-DOC-ALCANCE-MIENTE"

_BACKEND = Path(__file__).resolve().parent.parent
_DOC = _BACKEND / "docs" / "i18n_dashboard.md"


def _doc() -> str:
    if not _DOC.exists():
        pytest.skip(f"{_DOC} no existe en este checkout")
    return _DOC.read_text(encoding="utf-8")


def _fila(prefijo: str) -> str:
    for linea in _doc().split("\n"):
        if linea.startswith(f"| {prefijo}"):
            return linea
    pytest.fail(
        f"no encontré la fila «{prefijo}» en la tabla de alcance. Si se renombró, "
        f"actualiza este guard; si se borró, el alcance dejó de estar declarado. "
        f"[{_MARKER}]"
    )


def test_la_fila_del_plan_dice_que_SI_se_traduce() -> None:
    fila = _fila("Plan, recetas")
    assert "**Sí**" in fila, (
        f"la fila del plan sigue sin declarar que se traduce: {fila[:120]!r}. Es falso "
        f"desde P1-PLAN-DISPLAY-I18N (2026-08-19, knob default True). [{_MARKER}]"
    )


def test_la_fila_del_plan_enlaza_donde_vive_el_detalle() -> None:
    """Decir «Sí» sin decir CÓMO deja al lector sin saber qué esperar cuando una línea
    sale en español — que es conducta esperada del validador, no un fallo."""
    fila = _fila("Plan, recetas")
    assert "_display" in fila and "plan_display_i18n" in fila, (
        f"la fila no menciona `_display` ni enlaza `plan_display_i18n`: {fila[:150]!r}. "
        f"[{_MARKER}]"
    )
    assert "fallback" in fila.lower() or "español" in fila.lower(), (
        f"la fila no declara el fallback al español como conducta ESPERADA. Sin eso, "
        f"un lector toma por bug lo que es el validador haciendo su trabajo. [{_MARKER}]"
    )


def test_la_lista_de_compras_se_declara_bilingue() -> None:
    """Ni «Sí» ni «No»: cada línea lleva el gloss traducido Y el canónico español entre
    paréntesis, y el paréntesis es el identificador con el que resuelve el motor."""
    fila = _fila("Lista de compras")
    assert "ling" in fila.lower(), (
        f"la lista de compras no se declara bilingüe: {fila[:120]!r}. [{_MARKER}]"
    )
    assert "Habichuelas rojas" in fila or "canónico" in fila, (
        f"la fila no enseña la forma real de la línea, que es lo único que hace "
        f"entender por qué el paréntesis español no sobra. [{_MARKER}]"
    )


def test_los_nombres_del_catalogo_siguen_declarados_como_intocables() -> None:
    """LA MITAD QUE NO SE MUEVE. El riesgo de corregir esta doc es que alguien lea
    «ahora sí se traduce el plan» y arrastre los nombres de alimento con él."""
    fila = _fila("Nombres de alimentos")
    assert "**No, jamás**" in fila, (
        f"la fila de los nombres de alimento perdió su «No, jamás»: {fila[:120]!r}. "
        f"Son el SSOT del motor: `pantry_names_match`, el guard de coherencia y el "
        f"backstop de alergias resuelven por esas cadenas exactas, y dos de las tres "
        f"fallarían en silencio. [{_MARKER}]"
    )


def test_ya_no_queda_la_afirmacion_vieja_como_verdad_vigente() -> None:
    """MUTACIÓN DE CONTROL, y con una trampa que este repo ya pisó seis veces en dos
    días: la frase vieja SIGUE en el documento, citada dentro del párrafo que explica
    que era falsa. Un guard que buscara la cadena a secas fallaría contra su propia
    corrección; uno que la ignorase no cazaría una reincidencia.

    Se distingue por el contexto: la frase solo puede aparecer dentro del párrafo que la
    declara superada.
    """
    doc = _doc()
    frase = "Tu plan y tus recetas siguen en español"
    apariciones = [m.start() for m in re.finditer(re.escape(frase), doc)]
    if not apariciones:
        return  # se retiró del todo: también es correcto
    i_epigrafe = doc.find("Lo que este párrafo decía antes")
    assert i_epigrafe != -1, (
        f"la frase «{frase}» sigue en la doc pero ya no está el párrafo que la declara "
        f"superada. O es verdad otra vez —no lo es— o volvió por descuido. [{_MARKER}]"
    )
    for i in apariciones:
        assert i > i_epigrafe, (
            f"«{frase}» aparece ANTES del párrafo que la corrige (offset {i}), así que "
            f"un lector la encuentra como afirmación vigente. [{_MARKER}]"
        )


def test_el_copy_del_selector_ya_no_le_miente_al_usuario() -> None:
    """La doc es para nosotros; esto es lo que lee el usuario en Configuración."""
    settings = _BACKEND.parent / "frontend" / "src" / "pages" / "Settings.jsx"
    if not (_BACKEND.parent / "backend").is_dir() or not settings.exists():
        pytest.skip("frontend no disponible en este checkout (repos hermanos)")
    s = settings.read_text(encoding="utf-8")
    assert "Tu plan y tus recetas siguen en español" not in s, (
        f"Configuración sigue diciéndole al usuario que su plan no se traduce. No es "
        f"una imprecisión inocua: le enseña a tomar por fallo el plan traducido que sí "
        f"le llega, y por normal la línea que cae al español. [{_MARKER}]"
    )
