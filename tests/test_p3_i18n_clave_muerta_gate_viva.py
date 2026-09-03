"""[P3-I18N-CLAVE-MUERTA-QUE-EL-GATE-DECLARA-VIVA · 2026-08-23] El extractor de claves del
gate de i18n leía la fuente CRUDA, comentarios incluidos: una clave citada sólo en un
comentario («antes aquí decía t('…')») contaba como VIVA, su traducción seguía en los
cuatro catálogos y el gate cantaba «0 huérfanas». Comentario-vence-guard nº 12, y al revés
de las once anteriores: el comentario mantenía verde una clave muerta. Medido al cerrarlo:
2 claves («de 2,100» y «…»), borradas de los 4 catálogos.

Cierre: ``scripts/lib/sin-comentarios.mjs`` (máquina de estados que respeta cadenas y
templates; los comentarios pasan a blancos del mismo largo para no mover offsets) y los
tres extractores (`T_CALL`, `TN_CALL`, `KEY_DECL`) leen ``codigo``, no ``src``.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_FRONTEND = _BACKEND.parent / "frontend"
_LIB = _FRONTEND / "scripts" / "lib" / "sin-comentarios.mjs"
_CHECK = _FRONTEND / "scripts" / "i18n-check.mjs"
_MARKER = "P3-I18N-CLAVE-MUERTA-QUE-EL-GATE-DECLARA-VIVA"


def _hay_node() -> bool:
    return shutil.which("node") is not None


def _saltar_si_falta(*rutas: Path) -> None:
    for r in rutas:
        if not r.exists():
            pytest.skip(f"{r} no existe en este checkout (repos hermanos)")


def _node(codigo: str) -> str:
    tmp = _FRONTEND / "scripts" / "_t_p3_clave_muerta.mjs"
    tmp.write_text(codigo, encoding="utf-8")
    try:
        r = subprocess.run(["node", str(tmp)], cwd=str(_FRONTEND),
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    finally:
        tmp.unlink(missing_ok=True)
    assert r.returncode == 0, f"node falló:\n{r.stdout}\n{r.stderr}"
    return r.stdout


_FUENTE = (
    "// t('Clave solo en comentario de linea')\n"
    "/* t('Clave solo en bloque') */\n"
    "{/* t('Clave solo en comentario JSX') */}\n"
    "const a = t('Clave viva');\n"
    "const url = 'http://x.y/z'; // no es comentario lo de dentro de la cadena\n"
    "const tpl = `// tampoco ${t('Clave en template')}`;\n"
    "const esc = 'comilla \\\\' escapada'; t('Clave tras escape');\n"
)


@pytest.mark.skipif(not _hay_node(), reason="node no está en PATH")
def test_los_comentarios_se_vacian_y_el_codigo_queda_intacto() -> None:
    _saltar_si_falta(_LIB)
    salida = _node(
        "import { sinComentarios } from './lib/sin-comentarios.mjs';\n"
        f"const s = sinComentarios({json.dumps(_FUENTE)});\n"
        "console.log(JSON.stringify(s));\n"
    )
    limpio = json.loads(salida.strip().splitlines()[-1])
    for muerta in ("Clave solo en comentario de linea", "Clave solo en bloque", "Clave solo en comentario JSX"):
        assert muerta not in limpio, f"«{muerta}» sobrevivió al vaciado [{_MARKER}]"
    for viva in ("Clave viva", "http://x.y/z", "Clave en template", "Clave tras escape"):
        assert viva in limpio, f"«{viva}» NO es un comentario y se borró [{_MARKER}]"
    # Los offsets no se mueven: mismo largo y mismas líneas.
    assert len(limpio) == len(_FUENTE) and limpio.count("\n") == _FUENTE.count("\n")


def test_los_tres_extractores_leen_el_codigo_y_no_la_fuente_cruda() -> None:
    _saltar_si_falta(_CHECK)
    src = _CHECK.read_text(encoding="utf-8")
    assert "import { sinComentarios } from './lib/sin-comentarios.mjs'" in src
    assert "const codigo = sinComentarios(src);" in src
    for rx in ("T_CALL", "TN_CALL", "KEY_DECL"):
        assert re.search(rf"for \(const m of codigo\.matchAll\({rx}\)\)", src), (
            f"el extractor {rx} no lee `codigo` [{_MARKER}]")
        assert not re.search(rf"for \(const m of src\.matchAll\({rx}\)\)", src), (
            f"el extractor {rx} volvió a leer la fuente cruda [{_MARKER}]")
