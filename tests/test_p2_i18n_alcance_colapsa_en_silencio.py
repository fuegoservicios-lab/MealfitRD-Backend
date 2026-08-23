"""[P2-I18N-ALCANCE-COLAPSA-EN-SILENCIO · 2026-08-23] El alcance del gate de i18n nacía
de ``ENTRADAS.filter(existsSync)``: si ``main.jsx`` se renombra, la entrada desaparece
sin ruido, el grafo se recorre sólo desde ``custom-sw.js``, el alcance cae de 217
ficheros a 1 y el gate pasa de ❌ a ✅ con la cobertura «100,0 %» de un fichero.

Dos defensas fail-loud en ``scripts/i18n-alcance.mjs``: (1) una entrada declarada que
no existe lanza; (2) un alcance que no sea la MAYORÍA de ``src/`` lanza (medido: 217
dentro / 27 fuera — un colapso queda a 8× del umbral). La segunda existe porque la
primera sólo ve entradas que FALTAN, no entradas que existen y ya no importan nada.

Se mide la CONDUCTA ejecutando el módulo real con entradas inyectadas — no se renombra
``main.jsx`` en el árbol, que es compartido.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_FRONTEND = _BACKEND.parent / "frontend"
_ALCANCE = _FRONTEND / "scripts" / "i18n-alcance.mjs"
_CHECK = _FRONTEND / "scripts" / "i18n-check.mjs"
_MARKER = "P2-I18N-ALCANCE-COLAPSA-EN-SILENCIO"


def _hay_node() -> bool:
    return shutil.which("node") is not None


def _saltar_si_falta(*rutas: Path) -> None:
    for r in rutas:
        if not r.exists():
            pytest.skip(f"{r} no existe en este checkout (repos hermanos)")


def _node(codigo: str) -> subprocess.CompletedProcess:
    """Módulo ESM efímero DENTRO de `frontend/scripts/` para que los imports relativos
    resuelvan igual que en el script real (mismo patrón que el test hermano
    `test_p1_i18n_gate_ciego_sin_t.py`)."""
    tmp = _FRONTEND / "scripts" / "_t_p2_alcance_colapsa.mjs"
    tmp.write_text(codigo, encoding="utf-8")
    try:
        return subprocess.run(
            ["node", str(tmp)], cwd=str(_FRONTEND),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    finally:
        tmp.unlink(missing_ok=True)


_SONDA = """
import { clasificarAlcance, AlcanceColapsado } from './i18n-alcance.mjs';
const out = {};
try {
    const r = clasificarAlcance();
    out.sano = { dentro: r.dentro.length, fuera: r.fuera.length, tieneApp: r.dentro.includes('App.jsx') };
} catch (e) { out.sano = { error: String(e.message) }; }
for (const [nombre, entradas] of [['ausente', ['no-existe-%s.jsx']], ['colapso', ['custom-sw.js']]]) {
    try { const r = clasificarAlcance({ entradas }); out[nombre] = { paso: true, dentro: r.dentro.length }; }
    catch (e) { out[nombre] = { lanza: e instanceof AlcanceColapsado, msg: String(e.message).slice(0, 120) }; }
}
console.log(JSON.stringify(out));
""" % _MARKER


@pytest.mark.skipif(not _hay_node(), reason="node no está en PATH")
def test_las_dos_defensas_lanzan_y_el_arbol_real_pasa() -> None:
    _saltar_si_falta(_ALCANCE)
    r = _node(_SONDA)
    assert r.returncode == 0, f"la sonda de node falló:\n{r.stdout}\n{r.stderr}"
    out = json.loads(r.stdout.strip().splitlines()[-1])

    # El árbol real no debe disparar las defensas (si lo hace, el gate está roto, no el test).
    assert "error" not in out["sano"], f"el alcance real lanza: {out['sano']} [{_MARKER}]"
    assert out["sano"]["tieneApp"] and out["sano"]["dentro"] > out["sano"]["fuera"], out["sano"]

    # (1) entrada inexistente → lanza, con el marker en el mensaje (es lo que lee el operador).
    assert out["ausente"].get("lanza") is True, (
        f"una entrada que no existe se filtró en silencio: {out['ausente']} [{_MARKER}]")
    assert _MARKER in out["ausente"]["msg"]

    # (2) entrada que existe pero no alcanza la app → alcance colapsado → lanza.
    assert out["colapso"].get("lanza") is True, (
        f"el alcance colapsó a {out['colapso']} ficheros y nadie lo dijo [{_MARKER}]")
    assert "colapsado" in out["colapso"]["msg"]


def test_el_gate_no_envuelve_el_alcance_en_un_try() -> None:
    """Las defensas viven en `clasificarAlcance`; si el gate la llamase dentro de un
    `try/catch` que continúa, el colapso volvería a ser silencioso POR EL LLAMADOR.
    Se mide en el único call site."""
    _saltar_si_falta(_CHECK)
    src = _CHECK.read_text(encoding="utf-8")
    llamadas = [i for i in range(len(src)) if src.startswith("clasificarAlcance(", i)]
    llamadas = [i for i in llamadas if not src.startswith("import", src.rfind("\n", 0, i) + 1)]
    assert len(llamadas) == 1, f"esperaba 1 call site de clasificarAlcance en i18n-check.mjs, hay {len(llamadas)}"
    i = llamadas[0]
    linea_ini = src.rfind("\n", 0, i) + 1
    linea = src[linea_ini: src.find("\n", i)]
    assert linea.lstrip().startswith("const "), (
        f"la llamada a clasificarAlcance debe ser una `const` a nivel de módulo, no ir "
        f"dentro de un bloque: {linea!r} [{_MARKER}]")
    # Ventana corta hacia atrás: un `try {` abierto justo antes la envolvería.
    previo = src[max(0, linea_ini - 400): linea_ini]
    assert "try {" not in previo.split("\n")[-3:][0] if previo else True
    assert not any(l.strip().startswith("try") for l in previo.splitlines()[-3:]), (
        f"clasificarAlcance está bajo un try: {previo[-200:]!r} [{_MARKER}]")
