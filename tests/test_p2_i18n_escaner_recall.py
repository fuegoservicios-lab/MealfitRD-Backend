"""[P2-I18N-ESCANER-RECALL · 2026-08-22] El trinquete de español sin envolver llega a CERO.

═══════════════════════════════════════════════════════════════════════════════
LA CIFRA DEL GAP NO SE SOSTENÍA
═══════════════════════════════════════════════════════════════════════════════

El gap decía: «el trinquete mide lo que el detector sabe mirar; **394 literales españoles**
viven en posiciones que no inspecciona». Medido con AST sobre las cinco formas que enumera:
**13 hallazgos, 12 FUERA de alcance** (landing, legales, `/supermercado` — confirmado con
`clasificarAlcance()`), y el único dentro es `"Suscripción Básico"`, el nombre del plan en
PayPal: un identificador que viaja al proveedor y sale en el cargo del usuario. `toast(cond
? … : …)` da cero, y la asignación a variable ya la cubría el detector desde agosto.

═══════════════════════════════════════════════════════════════════════════════
EL MECANISMO SÍ ERA REAL, Y ES OTRO
═══════════════════════════════════════════════════════════════════════════════

No estaba en las posiciones: estaba en la MARCA. `pareceEspanol` exige un diacrítico o una
palabra funcional con espacio, y su comentario afirmaba que esa rama «no cuesta nada en
recall». Costaba 39 — los rótulos del panel forense del Historial («Calidad LLM»,
«Pausado», «Emergencia») no llevan ninguna de las dos, así que ese fichero reportaba CERO
hallazgos con 39 cadenas en español dentro. Más las tuplas `['id', 'Rótulo', 'tipo']`, que
no alcanzaba ningún nodo.

Al ensanchar el detector aparecieron 12 más, y **todos** resultaron ser tablas SSOT
deliberadas con su hermana traducida al lado, o cosas que no deben traducirse. Ahí estaba
el otro defecto, el que hacía inútil el número: **el trinquete mezclaba lo intencional con
lo pendiente**. 48 de los 78 eran tablas canónicas.

Con `[I18N-EXEMPT: razón]` en cada una: **78 → 0**, y con un detector MÁS ancho que el que
produjo el 78. Desde cero, la aritmética de «¿subió?» se vuelve trivial —cualquier hallazgo
es una regresión— y el gate ya lo trata como error duro sin cambiar nada: `retrocesos` marca
`hardFail` y con el baseline en `{}` todo fichero con hallazgos es un retroceso.

**Ese último punto se comprobó ANTES de escribir código**: la primera versión de este cierre
añadía una rama `else if (se.baseline === 0)` que nunca se ejecutaba, porque la rama de
retrocesos la precede y siempre gana. Se eliminó; lo que faltaba no era mecanismo, era el
MENSAJE que distingue deuda heredada de regresión propia.

tooltip-anchor: P2-I18N-ESCANER-RECALL
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_FRONT = _BACKEND.parent / "frontend"
_BASELINE = _FRONT / "scripts" / "i18n-sin-envolver.baseline.json"
_DETECTOR = _FRONT / "scripts" / "i18n-sin-envolver.mjs"
_CHECK = _FRONT / "scripts" / "i18n-check.mjs"

_MARKER = "P2-I18N-ESCANER-RECALL"


def _fuente(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"no existe {p} (¿repo hermano sin clonar?)")
    return io.open(p, encoding="utf-8").read()


def test_el_trinquete_esta_en_cero() -> None:
    datos = json.loads(_fuente(_BASELINE))
    assert datos.get("total") == 0, (
        f"el trinquete de español sin envolver dejó de estar en CERO (ahora "
        f"{datos.get('total')}). Llegó ahí el 2026-08-22 desde 78, y **48 de esos 78 eran "
        f"tablas SSOT deliberadas** que ahora llevan su `[I18N-EXEMPT: razón]`. Subirlo "
        f"vuelve a mezclar lo intencional con lo pendiente, que es exactamente por lo que "
        f"nadie pudo ver que a `History.jsx` le faltaban 39 rótulos. Si una cadena nueva no "
        f"debe traducirse, márcala; si debe, envuélvela. [{_MARKER}]"
    )
    assert datos.get("porArchivo") == {}, (
        f"el desglose por archivo del trinquete dejó de estar vacío: {datos.get('porArchivo')} "
        f"[{_MARKER}]"
    )


def test_la_marca_ancha_existe_y_no_se_mezcla_con_la_estrecha() -> None:
    src = _fuente(_DETECTOR)
    assert "export function pareceEspanolEnPosicionFuerte" in src, (
        f"desapareció la marca ancha. Sin ella, «Calidad LLM» o «Pausado» vuelven a ser "
        f"invisibles: ni tilde ni palabra funcional. [{_MARKER}]"
    )
    # La estrecha NO puede absorber la morfología: en un `return` suelto o en texto JSX el
    # falso positivo (una cadena inglesa marcada como española) no compensa, y es el caro,
    # porque enseña a desconfiar del gate.
    m = re.search(r"export function pareceEspanol\(texto\)(.*?)\n\}", src, re.S)
    assert m, f"no encontré `pareceEspanol` [{_MARKER}]"
    assert "MORFOLOGIA" not in m.group(1), (
        f"la marca ancha se coló dentro de `pareceEspanol`. Debe aplicarse SÓLO donde la "
        f"posición ya es evidencia de copy (tabla de rótulos, atributo, tupla). [{_MARKER}]"
    )


def test_el_detector_mira_las_posiciones_que_se_le_escapaban() -> None:
    src = _fuente(_DETECTOR)
    faltan = [
        etiqueta
        for etiqueta, patron in (
            ("tupla de rótulo", r"'tupla-de-rotulo'"),
            ("toast con fallback", r"'toast:fallback'"),
            ("toast ternario", r"'toast:ternario'"),
            ("expresión JSX", r"'jsx-expr'"),
        )
        if not re.search(patron, src)
    ]
    assert not faltan, (
        f"el detector dejó de mirar: {faltan}. La tupla es donde vivían 28 de los 39 rótulos "
        f"del panel forense. [{_MARKER}]"
    )


def test_el_gate_distingue_deuda_heredada_de_regresion_propia() -> None:
    src = _fuente(_CHECK)
    assert "no es deuda heredada" in src, (
        f"el gate dejó de avisar de que, con el trinquete en cero, la cadena la acaba de "
        f"introducir quien lee el error. Sin ese matiz el mensaje se lee como una deuda "
        f"vieja y se ignora. [{_MARKER}]"
    )
    # Y NO se reintroduce la rama muerta: con el baseline en cero, todo fichero con
    # hallazgos es un retroceso y la rama anterior siempre gana.
    assert "} else if (se.baseline === 0 && se.total > 0) {" not in src, (
        f"volvió la rama `else if (se.baseline === 0)`: es inalcanzable. La rama de "
        f"retrocesos la precede y, con el desglose vacío, captura todo fichero con "
        f"hallazgos. [{_MARKER}]"
    )


def test_toda_exencion_lleva_su_razon() -> None:
    """Una excepción sin motivo escrito es indistinguible de un silenciamiento por prisa."""
    sin_razon = []
    for p in sorted((_FRONT / "src").rglob("*.js")) + sorted((_FRONT / "src").rglob("*.jsx")):
        for n, linea in enumerate(io.open(p, encoding="utf-8").read().split("\n"), 1):
            if "I18N-EXEMPT" not in linea:
                continue
            m = re.search(r"\[I18N-EXEMPT:([^\]]*)\]", linea)
            if not m or len(m.group(1).strip()) < 4:
                sin_razon.append(f"{p.relative_to(_FRONT)}:{n}")
    assert not sin_razon, (
        f"exención(es) sin razón escrita (o con el marcador partido en varias líneas — se "
        f"busca por LÍNEA, no por bloque):\n"
        + "\n".join(f"  · {s}" for s in sin_razon)
        + f"\n[{_MARKER}]"
    )
