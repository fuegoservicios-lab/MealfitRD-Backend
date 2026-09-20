# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-139 · 2026-09-20] La coreografía del teclado vuelve a ser un modo de PRUEBA (apagada por defecto).

El lote 138 la encendió por defecto verificándola SOLO en el arnés del PC. El dueño, 20 minutos después: «¿es normal que
actualmente esté peor?… al abrir y cerrar, pero más cuando lo abría». Su sonda (paquete 20260920-211937):

    + 92 coreo+                                                  ← arranca el transform; el LAYOUT sigue cerrado
    +150 resize  H=509 S=335 sy=335 top=-335 caja=394            ← iOS DESPLAZA la página 335 px para enseñar el campo
    +466 relevo  H=509 S=335 sy=335 top=-335 cont=509 caja=174   ← la caja subió DOS veces (paneo + transform)
    +484 scroll  S=0 sy=0 top=0 caja=509                         ← y salta a su sitio

El camino de siempre no lo sufre: al foco pone YA el layout final y el cerrojo del documento, WebKit ve el campo visible y
no panea. La coreografía deja el layout cerrado hasta el relevo —justo lo que invita al paneo— y el arnés no tiene ni
teclado ni paneo. No se arregla a ciegas: vuelve a `/fluido`, y la llave cambia de nombre para que nadie quede encendido.

Lo demás del 138 sigue en pie (una animación por apertura, relevo sin scroll suave, miniatura inmediata).
Lección: un arnés que no reproduce el fenómeno no puede dar el visto bueno a un cambio de DEFECTO. Cero cambios de backend."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_apagada_por_defecto_y_con_llave_nueva():
    kc = _front("src/utils/keyboardChoreography.js")
    assert "return safeLocalStorageGet(CLAVE_COREOGRAFIA, null) === '1';" in kc
    assert "export const CLAVE_COREOGRAFIA = 'mf_kb_coreografia_v2';" in kc, \
        "quien guardó «1» en la llave vieja (lote 131) o pasó por el 138 tiene que arrancar APAGADO"
    assert "!== '0'" not in kc.split("export function coreografiaEncendida")[1].split("}")[0]


def test_el_camino_por_defecto_pone_layout_y_cerrojo_al_foco():
    """Lo que evita el paneo de iOS: inset + `data-kb-scroll-lock` escritos EN `anticiparApertura`, no en el relevo."""
    ap = _front("src/pages/AgentPage.jsx")
    k = ap.index("const anticiparApertura = (inset) => {")
    cuerpo = ap[k:ap.index("\n        };", k)]
    i = cuerpo.index("if (!abrirConCoreografia(")
    tramo = cuerpo[i:i + 500]
    assert "contenedor.style.setProperty('--kb-inset', `${inset}px`);" in tramo
    assert "root.toggleAttribute('data-kb-scroll-lock', true);" in tramo


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 139
