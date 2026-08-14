"""[P1-DASH-BRAND-MARK · 2026-08-14] El isotipo vuelve al dashboard, junto al wordmark.

El dueño pidió «el logo de nuestra marca en el dashboard». El dashboard ya mostraba
el WORDMARK (el texto «Bioboros»); lo que faltaba era el ISOTIPO, el brote. Vivía en
`public/mealfit-mark-dark.png` con un `Logo.jsx` que lo envolvía y la auditoría del
landing lo borró ese mismo día como «asset sin un solo consumidor» — lo cual era
cierto: nadie lo renderizaba. Se recupera del historial y ahora SÍ tiene consumidor.

Este test ancla las tres cosas que costaría re-descubrir:

1. **El nombre del archivo.** El original decía «mealfit», la marca muerta.
   P2-WORDMARK-BIOBOROS ya enseñó el precio de dejar el nombre viejo escrito en el
   árbol: el rebrand automático no alcanzó a `Logo.jsx` y el usuario vio «Mealfit» en
   una app ya rebrandeada. Un asset que se llama como la marca muerta es la misma
   trampa esperando.
2. **El par símbolo+wordmark.** Si alguien quita uno de los dos, la marca queda coja:
   el isotipo solo no dice el nombre, y este archivo existe porque el nombre solo no
   bastaba.
3. **El peso.** 47,7 KB → 6,1 KB recortando la transparencia y bajando a 128 px. Es un
   glifo plano de un color; el original traía el dibujo en el 45% central de un lienzo
   de 1254×1254. Importa porque quien lo borró estaba recortando el precache: si vuelve
   a entrar un PNG de 47 KB, este test lo dice.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_FRONT = _REPO / "frontend"
_LAYOUT = _FRONT / "src" / "components" / "dashboard" / "DashboardLayout.jsx"
_MARK_COMPONENT = _FRONT / "src" / "components" / "common" / "BrandMark.jsx"
_MARK_ASSET = _FRONT / "public" / "bioboros-mark.png"


def test_el_asset_existe_y_no_se_llama_como_la_marca_muerta():
    assert _MARK_ASSET.exists(), (
        "Falta `frontend/public/bioboros-mark.png` — el dashboard renderiza un "
        "isotipo que no está en el árbol: saldría el icono roto."
    )
    muertos = list((_FRONT / "public").glob("mealfit*"))
    assert not muertos, (
        f"Volvió un asset con el nombre de la marca muerta: {[p.name for p in muertos]}. "
        "Ver P2-WORDMARK-BIOBOROS: el nombre viejo en el árbol es lo que hizo que el "
        "usuario viera «Mealfit» en una app ya rebrandeada."
    )


def test_el_isotipo_no_vuelve_a_pesar_lo_que_pesaba():
    tam = _MARK_ASSET.stat().st_size
    assert tam <= 12_000, (
        f"`bioboros-mark.png` pesa {tam} bytes. El original pesaba 47.713 porque traía "
        f"el glifo en el 45% central de un lienzo de 1254×1254 y color verdadero para un "
        f"dibujo de UN color. Recortado a 128 px con paleta son ~6 KB. Si alguien lo "
        f"reemplaza por el original, el precache que la auditoría del landing adelgazó "
        f"vuelve a engordar."
    )


def test_el_dashboard_renderiza_isotipo_Y_wordmark():
    src = _LAYOUT.read_text(encoding="utf-8")
    codigo = "\n".join(
        ln for ln in src.splitlines()
        if not ln.lstrip().startswith("//") and not ln.lstrip().startswith("*")
    )
    assert "<BrandMark" in codigo, (
        "El dashboard dejó de renderizar el isotipo. Si fue a propósito, borra también "
        "el asset y este test — un símbolo sin consumidor es peso muerto (es exactamente "
        "por eso que la auditoría del landing borró el anterior)."
    )
    assert "<Wordmark" in codigo, (
        "El dashboard dejó de renderizar el wordmark. El isotipo SOLO no dice el nombre "
        "de la marca."
    )
    # van juntos, en el mismo bloque de marca
    i_mark, i_word = codigo.index("<BrandMark"), codigo.index("<Wordmark")
    assert abs(i_word - i_mark) < 200, (
        "El isotipo y el wordmark se separaron: dejaron de leerse como una unidad de marca."
    )


def test_el_isotipo_es_decorativo_para_lectores_de_pantalla():
    """Va SIEMPRE con el wordmark, que ya dice «Bioboros». Un alt descriptivo haría que
    un lector de pantalla anunciara la marca dos veces seguidas.

    [P1-BRAND-MARK-MONO · 2026-08-14] Dejó de ser un `<img alt="">` y pasó a ser un
    `<span>` pintado con máscara CSS, así que la afirmación cambia de sitio pero no de
    intención: sigue siendo invisible para un lector de pantalla."""
    src = _MARK_COMPONENT.read_text(encoding="utf-8")
    assert 'aria-hidden="true"' in src, "el isotipo debe llevar aria-hidden"
    assert 'role="presentation"' in src or re.search(r'alt=""', src), (
        "el isotipo debe declararse decorativo (`role=\"presentation\"` en el span, o "
        "`alt=\"\"` si vuelve a ser un <img>)"
    )


def test_la_tinta_del_isotipo_la_pone_el_tema_no_el_PNG():
    """[P1-BRAND-MARK-MONO] El wordmark es monocromo por decisión del dueño y se
    rechazaron DOS versiones con color. Un símbolo índigo al lado reintroducía ese
    acento por la puerta de atrás — y además pesaba 2,9× menos que la palabra contra el
    fondo (5,18:1 vs 15,11:1), leyéndose como adorno y no como parte del logo.

    La máscara hace que herede `currentColor`; si alguien vuelve a un `<img>` de color
    fijo, el tema claro pierde además su tinta oscura automática."""
    css = (_MARK_COMPONENT.parent / "BrandMark.module.css").read_text(encoding="utf-8")
    sin_comentarios = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    assert "currentColor" in sin_comentarios, (
        "el isotipo dejó de heredar la tinta del bloque de marca"
    )
    assert "mask" in sin_comentarios and "bioboros-mark.png" in sin_comentarios, (
        "el PNG debe entrar como MÁSCARA (su alfa es la forma), no como color"
    )
    # el respaldo para navegadores sin máscaras no puede desaparecer en silencio
    assert "@supports" in sin_comentarios, (
        "falta el respaldo `@supports`: sin máscaras el símbolo saldría invisible"
    )
