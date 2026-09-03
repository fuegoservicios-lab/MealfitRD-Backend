"""[P2-LEGAL-FECHA-MANIFIESTO-DRIFT · 2026-08-23] G71: el manifiesto de gobernanza legal
declaraba cuatro documentos más frescos de lo que el lector ve.

MEDIDO comparando manifiesto contra la fecha publicada en cada documento (8/8):

    terms ✓ · privacy ✓ · ai-policy ✓ · data-protection ✓
    medical                 manifiesto 2026-08-14   publicado 2026-07-12
    acceptable-use          manifiesto 2026-08-14   publicado 2026-06-30
    refunds                 manifiesto 2026-08-14   publicado 2026-07-12
    responsible-disclosure  manifiesto 2026-08-14   publicado 2026-06-30

CUÁL DE LAS DOS FECHAS ERA LA VERDADERA — lo decidió el historial, no el criterio. `git log` de
cada documento:

    medical, acceptable-use, responsible-disclosure → último cambio: metadatos SEO y anclas.
        El TEXTO legal no se ha tocado desde julio/junio, así que la fecha publicada es la
        cierta y el manifiesto es el que mentía. Se alinea el MANIFIESTO.
    refunds → cambió el correo al que se piden los reembolsos (18-ago) y el documento seguía
        diciendo «12 de Julio». Ahí el desfasado era el DOCUMENTO. Se alinea el DOCUMENTO.

*No se mueve la fecha de un documento legal para que cuadre con un manifiesto: eso es fabricar
una revisión que nunca ocurrió.* La dirección de la corrección la decide qué cambió de verdad.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_APEX = _BACKEND.parent.parent / "bioboros-cinematic"

pytestmark = pytest.mark.skipif(
    not _APEX.is_dir(),
    reason="el repo del apex no está clonado junto a este (es un repo hermano)",
)

_MES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
        "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11,
        "diciembre": 12}


def _fecha_publicada(ruta: Path) -> str | None:
    """La fecha que el LECTOR ve, normalizada a ISO. Es la que manda: un manifiesto interno no
    puede contradecir lo que la página dice en su primera línea."""
    s = io.open(ruta, encoding="utf-8").read()
    m = re.search(r"[Úu]ltima actualizaci[óo]n:\s*(\d{1,2})\s+de\s+(\w+),\s*(20\d\d)", s)
    if not m:
        return None
    dd, mes, yy = m.groups()
    return f"{yy}-{_MES.get(mes.lower(), 0):02d}-{int(dd):02d}"


def _manifiesto() -> list[dict]:
    return json.loads(io.open(_APEX / "contenido-legal.json", encoding="utf-8").read())["documentos"]


def test_ningun_documento_declara_una_fecha_distinta_de_la_que_publica():
    """EL contrato: si el manifiesto dice 14-ago y la página dice 12-jul, una de las dos miente
    y el usuario sólo ve una."""
    drift = []
    for d in _manifiesto():
        ruta = _APEX / d["fichero"]
        if not ruta.exists():
            continue
        pub = _fecha_publicada(ruta)
        if pub and pub != d["fecha_efectiva"]:
            drift.append(f"{d['id']}: manifiesto={d['fecha_efectiva']} publicado={pub}")
    assert not drift, "el manifiesto legal volvió a desfasarse:\n  " + "\n  ".join(drift)


def test_todos_los_documentos_del_manifiesto_publican_una_fecha():
    """Un documento legal sin fecha visible es peor que uno desfasado: no se puede saber qué
    versión aceptaste."""
    sin_fecha = [d["id"] for d in _manifiesto()
                 if (_APEX / d["fichero"]).exists() and not _fecha_publicada(_APEX / d["fichero"])]
    assert not sin_fecha, f"documentos legales sin «Última actualización» visible: {sin_fecha}"


def test_el_manifiesto_apunta_a_ficheros_que_existen():
    """Una fila que apunta a un fichero ausente declara gobernanza sobre nada."""
    faltan = [d["id"] for d in _manifiesto() if not (_APEX / d["fichero"]).exists()]
    assert not faltan, f"el manifiesto declara documentos inexistentes: {faltan}"


def test_ninguna_fecha_esta_en_el_futuro():
    """Una fecha efectiva futura significa «esto aún no rige», y el documento está publicado."""
    import datetime as _dt
    hoy = _dt.date.today().isoformat()
    futuras = [(d["id"], d["fecha_efectiva"]) for d in _manifiesto()
               if str(d.get("fecha_efectiva", "")) > hoy]
    assert not futuras, f"documentos legales con fecha efectiva futura: {futuras}"
