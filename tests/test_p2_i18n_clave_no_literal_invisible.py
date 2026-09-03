"""[P2-I18N-CLAVE-NO-LITERAL-INVISIBLE-PARA-LAS-DOS-MITADES · 2026-08-23] Una ``t()``
cuya clave no sea un literal pegado al paréntesis desaparecía de las DOS mitades del
gate de i18n: el extractor (regex ``t('…')``) no la ve, y el escáner de español sin
envolver tampoco, porque sus literales ya son argumento de ``t()``. Medido antes del
cierre: ``t(ok ? 'A' : 'B')``, ``t(`…${n}…`)`` y ``t(K)`` con ``const K = '…'`` →
claves ``[]`` y sin-envolver ``[]`` — cadena nueva en español, «100,0 %» y ✅.

Cierre: ``clavesNoLiterales(src)`` en ``scripts/i18n-sin-envolver.mjs`` (tercera
mirada, AST) y fallo DURO en ``scripts/i18n-check.mjs``. El patrón ``i18nKey``
(``t(sec.titleKey)``, ``t(KEY)`` declarada con ``i18nKey(...)``) sigue siendo legítimo
y NO se marca: medido en el árbol, las 10 no-literales vivas son todas ese patrón.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_FRONTEND = _BACKEND.parent / "frontend"
_DETECTOR = _FRONTEND / "scripts" / "i18n-sin-envolver.mjs"
_CHECK = _FRONTEND / "scripts" / "i18n-check.mjs"
_MARKER = "P2-I18N-CLAVE-NO-LITERAL-INVISIBLE-PARA-LAS-DOS-MITADES"


def _hay_node() -> bool:
    return shutil.which("node") is not None


def _saltar_si_falta(*rutas: Path) -> None:
    for r in rutas:
        if not r.exists():
            pytest.skip(f"{r} no existe en este checkout (repos hermanos)")


def _node(codigo: str) -> str:
    """Módulo ESM efímero dentro de `frontend/scripts/` (mismo patrón que
    `test_p1_i18n_gate_ciego_sin_t.py`): sus imports relativos resuelven como los reales."""
    tmp = _FRONTEND / "scripts" / "_t_p2_clave_no_literal.mjs"
    tmp.write_text(codigo, encoding="utf-8")
    try:
        r = subprocess.run(
            ["node", str(tmp)], cwd=str(_FRONTEND),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    finally:
        tmp.unlink(missing_ok=True)
    assert r.returncode == 0, f"la sonda de node falló:\n{r.stdout}\n{r.stderr}"
    return r.stdout


def _formas(fuente: str) -> list[str]:
    salida = _node(
        "import { clavesNoLiterales } from './i18n-sin-envolver.mjs';\n"
        f"const h = clavesNoLiterales({json.dumps(fuente)});\n"
        "console.log(JSON.stringify(h.map(x => x.forma).sort()));\n"
    )
    return json.loads(salida.strip().splitlines()[-1])


@pytest.mark.skipif(not _hay_node(), reason="node no está en PATH")
def test_las_formas_invisibles_se_ven() -> None:
    """Las cuatro formas medidas como invisibles, una por una, más `tn` (clave en
    posición 2 y 3, no 1)."""
    _saltar_si_falta(_DETECTOR)
    fuente = (
        "const K = 'Abre tu nevera ahora';\n"
        "export function C({ ok, n }) {\n"
        "  return <p>\n"
        "    {t(ok ? 'Guardamos tus cambios' : 'No pudimos guardar los cambios')}\n"
        "    {t(`Tienes ${n} planes guardados`)}\n"
        "    {t('Revisa los valores ' + 'del formulario')}\n"
        "    {t(K)}\n"
        "    {tn(n, cond ? 'un plan' : 'x', 'varios planes')}\n"
        "  </p>;\n"
        "}\n"
    )
    formas = _formas(fuente)
    for esperada in ("ternario", "template-con-interpolacion", "concatenacion",
                     "identificador-a-literal-pelado"):
        assert esperada in formas, f"no ve la forma {esperada!r}; vio {formas} [{_MARKER}]"
    assert formas.count("ternario") == 2, f"`tn` mira la 2.ª y 3.ª posición: {formas}"


@pytest.mark.skipif(not _hay_node(), reason="node no está en PATH")
def test_el_patron_i18nkey_no_se_marca() -> None:
    """CONTROL. Si marcase también `t(identificador)`/`t(obj.prop)`, el gate se pondría
    rojo en las 10 llamadas legítimas del árbol y alguien lo desactivaría."""
    _saltar_si_falta(_DETECTOR)
    fuente = (
        "import { i18nKey } from '../i18n';\n"
        "const TOPE = i18nKey('Tope de proteína');\n"
        "const SECCIONES = [{ rx: /^montaje:/i, titleKey: i18nKey('Montaje') }];\n"
        "export function C({ sec }) {\n"
        "  return <p>{t(sec.titleKey)}{t(TOPE)}{t(SECCIONES[0]?.titleKey)}{t('Literal normal')}{t(`sin interpolar`)}</p>;\n"
        "}\n"
    )
    assert _formas(fuente) == [], f"falsos positivos sobre el patrón i18nKey [{_MARKER}]"


@pytest.mark.skipif(not _hay_node(), reason="node no está en PATH")
def test_el_gate_falla_duro_con_un_ternario_inyectado(tmp_path: Path) -> None:
    """CONDUCTA del gate entero: no basta con que la función exista — tiene que estar
    cableada como fallo duro. Se ejecuta el gate real con un fichero inyectado en el
    alcance y se restaura el árbol pase lo que pase."""
    _saltar_si_falta(_CHECK, _DETECTOR)
    victima = _FRONTEND / "src" / "pages" / "ResetPassword.jsx"
    _saltar_si_falta(victima)
    original = victima.read_bytes()
    texto = original.decode("utf-8")
    ancla = "const ResetPassword = () => {"
    assert ancla in texto, "el ancla de inyección cambió; elige otra"
    inyectado = texto.replace(
        ancla,
        "function _Mut({ ok }) { return <p>{t(ok ? 'Guardamos tus cambios' : 'No pudimos guardar')}</p>; }\n" + ancla,
        1,
    )
    try:
        victima.write_text(inyectado, encoding="utf-8", newline="\n")
        r = subprocess.run(
            ["node", str(_CHECK), "--strict"], cwd=str(_FRONTEND),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    finally:
        victima.write_bytes(original)
    assert victima.read_bytes() == original, "el árbol no quedó restaurado"
    assert r.returncode != 0, f"el gate pasó en VERDE con un ternario dentro de t():\n{r.stdout}\n{r.stderr}"
    assert "CLAVE DE t() QUE NO PUEDE VIVIR EN UN CATÁLOGO" in (r.stdout + r.stderr)
    assert "pages/ResetPassword.jsx" in (r.stdout + r.stderr), "el hallazgo no señala fichero:línea"


def test_el_comentario_del_extractor_ya_no_promete_un_reporte_que_no_existe() -> None:
    """El extractor decía «se reportan aparte como sospechosas» y no había ni una línea
    que lo hiciera. Ahora la promesa tiene mecanismo: el gate importa y usa
    `clavesNoLiterales` y lo escala a `hardFail`."""
    _saltar_si_falta(_CHECK)
    src = _CHECK.read_text(encoding="utf-8")
    assert "import { detectarEnFuente, clavesNoLiterales } from './i18n-sin-envolver.mjs'" in src, (
        f"el gate no importa clavesNoLiterales [{_MARKER}]")
    i = src.find("for (const h of clavesNoLiterales(src))")
    assert i > 0, f"el gate no recorre clavesNoLiterales [{_MARKER}]"
    # [2026-08-23] Ventana por ESTRUCTURA, no por bytes. El `hardFail` NO está dentro del
    # `for` que recoge los hallazgos, sino en el `if (clavesOpacas.length)` que viene
    # después: se ancla a ese bloque y se corta en su `}` de columna 0. Una ventana fija
    # (`i + 600`) la desborda el primer comentario que alguien añada en medio — le pasó el
    # mismo día a `test_p1_chat_mobile_ready` con `_AP[i:i+4200]`, dos veces en el mismo sitio.
    j = src.find("if (clavesOpacas.length) {", i)
    assert j > i, f"el gate ya no escala los hallazgos en un `if (clavesOpacas.length)` [{_MARKER}]"
    fin = src.find("\n}\n", j)
    ventana = src[j: fin if fin > j else len(src)]
    assert "hardFail = true" in ventana, f"el hallazgo no escala a hardFail [{_MARKER}]"
