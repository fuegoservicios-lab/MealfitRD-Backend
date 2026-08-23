"""[P2-I18N-CALIDAD-CONCORDANCIA · 2026-08-22] Cinco cadenas mal traducidas, y las tres
cifras que la medición desmintió por el camino.

Este cierre vale sobre todo por lo que NO había que hacer. Los tres gaps de calidad que
quedaban abiertos describían números que resultaron falsos, y en dos de los tres el
diagnóstico también:

  · «266 valores de pt-BR son idénticos a su clave española» — cierto, y **casi todos
    correctos**: portugués y español comparten «Confirmar compra», «Peso máximo», «Estados
    Unidos», «Política de Uso»… De las 266, UNA sola no lo era por cognado: «Afinando sabores
    criollos…», que en portugués es «crioulos». Un valor igual a su clave pasa el gate como
    traducido, así que el número bruto no dice nada por sí solo.

  · «El "Menú" de producto se traduce de dos formas en pt-BR (12 cardápio contra 5 menu) y
    las dos se ven en la misma sesión» — no eran dos formas de la misma palabra: son DOS
    PALABRAS DISTINTAS que en español se escriben igual. 8 de los 10 «menu» eran el menú de
    NAVEGACIÓN, donde «menu» es lo correcto en portugués; solo 2 eran el menú de COMIDA.
    Un `find/replace` de menu→cardápio habría roto los ocho rótulos de navegación,
    incluidos tres `aria-label`.

  · «10 cadenas italianas dicen "Il tuo Frigo è vuota"» — quedaba UNA. La sustitución
    Dispensa→Frigo del 22-ago ya se había corregido casi entera; la superviviente llevaba
    TRES errores en la misma frase (adjetivo, participio y clítico), porque «Nevera» es
    femenino en español y «Frigo» es masculino en italiano.

Y una que ningún gap listaba, encontrada al mirar la misma frase en los cuatro idiomas:
en fr-FR el «Vérifiez-la» apuntaba a la lista cuando lo que hay que revisar es la nevera
—es de donde se quita lo que no se compró—, con el resto de la frase ya en masculino.

QUÉ ANCLA ESTE GUARD. No las cinco cadenas: la REGLA que las tres primeras violaban, que es
lo único que impide que vuelvan por otra puerta.

tooltip-anchor: P2-I18N-CALIDAD-CONCORDANCIA
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_LOCALES = _BACKEND.parent / "frontend" / "src" / "i18n" / "locales"

_MARKER = "P2-I18N-CALIDAD-CONCORDANCIA"

# Los rótulos de NAVEGACIÓN: ahí «menu» es la palabra portuguesa correcta.
_MENU_DE_NAVEGACION = re.compile(r"^(Abrir|Cerrar|Volver al) menú", re.I)


def _catalogo(loc: str) -> dict:
    p = _LOCALES / f"{loc}.json"
    if not p.exists():
        pytest.skip(f"no existe {p} (¿repo hermano sin clonar?)")
    return json.loads(io.open(p, encoding="utf-8").read())


def test_pt_br_distingue_el_menu_de_comida_del_menu_de_navegacion() -> None:
    """Dos palabras distintas que en español se escriben igual."""
    pt = _catalogo("pt-BR")

    comida_mal, navegacion_mal = [], []
    for clave, valor_raw in pt.items():
        if "menú" not in clave.lower():
            continue
        # [P2-I18N-CARDAPIO-SUPERVIVIENTE-EN-CLAVE-PLURAL · 2026-08-23] Las claves PLURALES
        # son un dict `{one, other}` y este guard las saltaba con `isinstance(valor, str)`:
        # así sobrevivió «Seu menu de {n} dias já está pronto» —menú de COMIDA— mientras el
        # guard cantaba que los 12 cardápio y los 5 menu estaban en su sitio. Se funden las
        # formas y se juzgan juntas: una sola mal y la clave es culpable.
        valor = valor_raw if isinstance(valor_raw, str) else " ".join(
            v for v in (valor_raw or {}).values() if isinstance(v, str))
        if not valor:
            continue
        es_navegacion = bool(_MENU_DE_NAVEGACION.match(clave))
        dice_cardapio = bool(re.search(r"card[áa]pio", valor, re.I))
        dice_menu = bool(re.search(r"\bmenus?\b", valor, re.I))
        if es_navegacion and dice_cardapio:
            navegacion_mal.append(clave)
        elif not es_navegacion and dice_menu and not dice_cardapio:
            comida_mal.append(clave)

    assert not comida_mal, (
        f"{len(comida_mal)} cadena(s) de pt-BR llaman «menu» al menú de COMIDA, que en "
        f"portugués es «cardápio»:\n"
        + "\n".join(f"  · {c[:80]!r}" for c in comida_mal)
        + f"\n[{_MARKER}]"
    )
    assert not navegacion_mal, (
        f"{len(navegacion_mal)} rótulo(s) de NAVEGACIÓN de pt-BR dicen «cardápio». Ahí "
        f"«menu» es lo correcto: es el widget, no la comida. Alguien hizo un find/replace "
        f"sobre la palabra en vez de sobre el sentido, y tres de esos rótulos son "
        f"`aria-label`.\n"
        + "\n".join(f"  · {c[:80]!r}" for c in navegacion_mal)
        + f"\n[{_MARKER}]"
    )


def test_ningun_catalogo_conserva_una_flexion_espanola_de_lo_criollo() -> None:
    """Lo prohibido es la FLEXIÓN, no la palabra.

    Los cuatro catálogos traducían «criollo» en una cadena y lo dejaban en español en otra
    —lo destapó este mismo guard en su primera corrida—, así que la regla no puede ser «no
    aparezca»: en inglés `criollo` es un préstamo culinario legítimo y se conserva.

    Lo que no lo es en ningún idioma destino es una forma FLEXIONADA en español (`criolla`,
    `criollos`, `criollas`): el inglés no concuerda en género, y el francés, el italiano y el
    portugués tienen su propia palabra (`créole`, `creolo`, `crioulo`). Una `criolla` suelta
    dentro de una frase italiana es siempre un descuido, nunca una decisión.
    """
    flexion = re.compile(r"criol(?:la|los|las)\b", re.I)
    sucios = []
    for loc in ("en-US", "pt-BR", "fr-FR", "it-IT"):
        for valor in _catalogo(loc).values():
            if isinstance(valor, str) and flexion.search(valor):
                sucios.append(f"{loc}: {valor[:70]!r}")
    assert not sucios, (
        f"flexión española de «criollo» en un catálogo que no es español.\n"
        + "\n".join(f"  · {s}" for s in sucios)
        + f"\n[{_MARKER}]"
    )


def test_it_it_concuerda_en_masculino_con_frigo() -> None:
    """«Nevera» es femenino en español; «Frigo» es MASCULINO en italiano."""
    # Adjetivo o participio femenino a menos de tres palabras de «Frigo». Se acota la
    # ventana a propósito: sin ella, 14 de 15 aciertos son falsos —el femenino concuerda
    # con «la lista» o «la spesa», que salen en la misma frase.
    rx = re.compile(
        r"\bFrigo\b(?:\s+\w+){0,3}\s+"
        r"(vuota|piena|pronta|aggiornata|svuotata|riempita|completa|nuova|sola)\b",
        re.I,
    )
    fallos = [
        f"{valor[:100]!r}"
        for valor in _catalogo("it-IT").values()
        if isinstance(valor, str) and rx.search(valor)
    ]
    assert not fallos, (
        f"concordancia femenina junto a «Frigo» en it-IT. La sustitución Dispensa→Frigo "
        f"cambia el género del sustantivo: hay que mover TAMBIÉN el adjetivo, el participio "
        f"y el clítico.\n" + "\n".join(f"  · {f}" for f in fallos) + f"\n[{_MARKER}]"
    )


def test_las_cinco_cadenas_corregidas_siguen_corregidas() -> None:
    """Ancla directa: las tres reglas de arriba no cubren el clítico ni el francés."""
    it = _catalogo("it-IT")
    fr = _catalogo("fr-FR")

    clave = (
        "Pasaron unos días desde el inicio de tu plan y tu Nevera seguía vacía, así que la "
        "llenamos con tu lista de compras. Revísala y quita lo que no hayas comprado."
    )
    v_it = it.get(clave, "")
    assert v_it, f"desapareció la clave del aviso de nevera vacía [{_MARKER}]"
    for esperado in ("vuoto", "riempito", "Controllalo"):
        assert esperado in v_it, (
            f"it-IT volvió a la forma femenina: falta «{esperado}» en «{v_it[:90]}». "
            f"[{_MARKER}]"
        )

    v_fr = fr.get(clave, "")
    assert v_fr, f"desapareció la clave del aviso de nevera vacía en fr-FR [{_MARKER}]"
    assert "Vérifiez-le" in v_fr, (
        f"fr-FR volvió a «Vérifiez-la»: el pronombre apunta a la lista, y lo que hay que "
        f"revisar es la nevera — es de donde se quita lo que no se compró. El resto de la "
        f"frase ya concordaba en masculino. [{_MARKER}]"
    )
