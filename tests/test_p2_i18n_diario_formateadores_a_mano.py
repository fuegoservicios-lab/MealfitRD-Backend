"""[P2-I18N-DIARIO-FORMATEADORES-A-MANO + P2-I18N-RELTIME-HISTORY-CRUDO +
P2-I18N-META-LITERAL-PLANSHOWCASE + P2-I18N-DASH-MONEDA-ARIA-PLACEHOLDER · 2026-08-22]
Cuatro formateadores escritos a mano que ninguna búsqueda por `t()` podía encontrar.

Es el patrón que más veces se ha escapado en este repo: **el defecto no está en una cadena
sin envolver, está en código que FABRICA la cadena**. Un barrido por llamadas al motor no ve
un array de iniciales, ni un `padStart` que compone una hora, ni un número tecleado dentro
de un JSX. Por eso los cuatro sobrevivieron a tres pasadas de i18n.

LOS CUATRO:

  1. `DIA_LETRA = ['D','L','M','M','J','V','S']` en el diario. Iniciales ESPAÑOLAS a nivel de
     módulo. En inglés la tira es S M T W T F S y en portugués D S T Q Q S S — así que no era
     «falta traducir», era «la semana está mal escrita». No se arregla con siete claves:
     `Intl` ya conoce la forma `narrow` de cada idioma, y 7 × 4 serían 28 oportunidades de
     teclear mal una letra que nadie revisaría.

  2. `hhmm = ${getHours()}:${getMinutes()}` — 24 h fijo para los cinco idiomas. En en-US se
     lee «3:05 PM», y esa diferencia no es cosmética: «03:05» y «15:05» son horas distintas
     para quien espera AM/PM.

  3. `_fmtRelTime` en `History.jsx` construía «hace 2h 15m» en español y lo interpolaba
     DENTRO de un `t()`: «Escalated: hace 2h 15m», media frase en cada idioma. Vivía a 3.700
     líneas de profundidad dentro del componente, que es cómo `P1-I18N-TIEMPO-RELATIVO` cerró
     a su gemelo (`shelfLife.js`) y lo dejó a él atrás.

  4. `/ 2,100` en `PlanShowcase`, con separador de millares estadounidense: en francés salía
     «1 900 / 2,100», la mitad de la línea formateada por locale y la mitad no. Y el número
     vivía DENTRO de la clave (`t('de 2,100')`), así que cambiar la meta de la demo
     huerfanaba cuatro traducciones en silencio.

  5. El `aria-label` del presupuesto del Dashboard salía de un ternario de DOS ramas escrito
     cuando sólo había dos monedas. Con el sistema de países vivo son cinco: a un español con
     EUR el lector de pantalla le decía «Presupuesto total en pesos dominicanos», y el
     placeholder le proponía «Ej. 5000» euros.

LO QUE NO SE TOCÓ, y por qué: `QBudget.jsx` tiene la misma forma de ternario pero SÍ cubre
las cinco monedas, y sus ramas están ancladas a propósito por `test_p1_country_system_f1.py`
—«byte-identidad del dark path»— que es un contrato de otro P-fix. Funciona hoy; unificarlo
es una decisión de quien lo escribió, no una corrección.

tooltip-anchor: P2-I18N-DIARIO-FORMATEADORES-A-MANO
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_FRONT = _BACKEND.parent / "frontend" / "src"

_MARKER = "P2-I18N-DIARIO-FORMATEADORES-A-MANO"


def _codigo(rel: str) -> str:
    """La fuente SIN las líneas de comentario.

    Estos guards buscan formas de CÓDIGO, y el propio comentario que explica el arreglo cita
    la forma prohibida — así que un `in` sobre el fichero entero se acusa a sí mismo. Es el
    comentario-vence-guard que este repo ya pagó varias veces; la salida no es recortar la
    prosa (documenta por qué el arreglo es el que es), es mirar donde el defecto puede vivir.
    """
    return "\n".join(
        l for l in _fuente(rel).split("\n") if not l.lstrip().startswith(("//", "*", "/*"))
    )


def _fuente(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip(f"no existe {p} (¿repo hermano sin clonar?)")
    return io.open(p, encoding="utf-8").read()


def test_el_diario_no_teclea_las_iniciales_de_la_semana() -> None:
    src = _codigo("components/dashboard/DiaryHistory.jsx")
    assert "['D', 'L', 'M', 'M', 'J', 'V', 'S']" not in src, (
        f"volvieron las iniciales españolas escritas a mano. En inglés la tira es "
        f"S M T W T F S y en portugués D S T Q Q S S: no es que falte traducir, es que la "
        f"semana está mal escrita. Usa `formatDate(d, {{ weekday: 'narrow' }})`. [{_MARKER}]"
    )
    assert "weekday: 'narrow'" in _fuente("components/dashboard/DiaryHistory.jsx"), (
        f"el diario dejó de pedirle a `Intl` la inicial del día. [{_MARKER}]"
    )


def test_el_diario_no_compone_la_hora_a_mano() -> None:
    src = _fuente("components/dashboard/DiaryHistory.jsx")
    assert not re.search(r"getHours\(\)\)\.padStart", _codigo("components/dashboard/DiaryHistory.jsx")), (
        f"volvió la hora compuesta a mano (24 h fijo). En en-US se lee «3:05 PM», y «03:05» "
        f"y «15:05» son horas DISTINTAS para quien espera AM/PM. [{_MARKER}]"
    )
    assert "timeStyle: 'short'" in src, (
        f"el diario dejó de formatear la hora por locale. [{_MARKER}]"
    )


def test_el_tiempo_relativo_del_historial_vive_en_su_modulo() -> None:
    src = _fuente("pages/History.jsx")
    assert "formatRelativeTime(iso, t, tn)" in src, (
        f"`History.jsx` volvió a construir el tiempo relativo en vez de delegar en "
        f"`utils/relativeTime.js`. Así se quedó atrás cuando su gemelo `shelfLife.js` se "
        f"arregló: el helper estaba dentro del componente y ninguna búsqueda lo alcanzaba. "
        f"[{_MARKER}]"
    )
    modulo = _fuente("utils/relativeTime.js")
    # El fallback tiene que INTERPOLAR: un `(es) => es` pelado deja `{h}` crudo en pantalla.
    assert "_tFallback = (es, vars) => _interp(es, vars)" in modulo, (
        f"el fallback de `relativeTime.js` volvió a no interpolar. Sin `t`, la pantalla "
        f"muestra `hace {{h}} h {{m}} min` literal — el mismo defecto que la nota clínica del "
        f"PDF cerró el mismo día. [{_MARKER}]"
    )


def test_la_meta_de_la_demo_no_es_un_literal_con_separador_gringo() -> None:
    src = _fuente("components/auth/PlanShowcase.jsx")
    # El literal sólo puede quedar en la prosa del comentario que explica el arreglo.
    codigo = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("//"))
    assert "2,100" not in codigo, (
        f"volvió el `2,100` tecleado. En francés la línea salía «1 900 / 2,100»: media "
        f"formateada por locale y media no, en el mismo renglón. [{_MARKER}]"
    )
    assert "META_DEMO_KCAL" in src and "formatNumber(META_DEMO_KCAL)" in src, (
        f"la meta de la demo dejó de salir de una constante formateada por locale. "
        f"[{_MARKER}]"
    )
    assert "t('de {meta}'" in src, (
        f"el número volvió a vivir DENTRO de la clave de traducción: cambiar la meta de la "
        f"demo huerfanaría las cuatro traducciones en silencio. [{_MARKER}]"
    )


def test_el_presupuesto_del_dashboard_deriva_de_la_moneda() -> None:
    src = _fuente("pages/Dashboard.jsx")
    assert "_cur === 'USD' ? t('Presupuesto total en dólares')" not in _codigo("pages/Dashboard.jsx"), (
        f"volvió el ternario de DOS ramas. Hay CINCO monedas vivas: a un español con EUR el "
        f"lector de pantalla le decía «Presupuesto total en pesos dominicanos». [{_MARKER}]"
    )
    assert "formatCurrencyName(_cur)" in src, (
        f"el `aria-label` dejó de derivar el nombre de la moneda. `Intl.DisplayNames` lo "
        f"sabe en los cinco idiomas; cinco monedas × cuatro idiomas serían veinte claves que "
        f"nadie revisaría. [{_MARKER}]"
    )
    assert re.search(r"placeholder=\{t\('Ej\. \{monto\}'", src), (
        f"el placeholder volvió a un ejemplo fijo. Sale del MÍNIMO real de esa moneda y "
        f"ciclo (`minBudgetFor`, SSOT), que además enseña el orden de magnitud que el "
        f"backend va a aceptar. [{_MARKER}]"
    )


def test_qbudget_sigue_intacto_a_proposito() -> None:
    """La frontera del cambio, escrita para que nadie la cruce por simetría."""
    src = _fuente("components/assessment/questions/QBudget.jsx")
    assert "effectiveCurrency === 'MXN'" in src and "Presupuesto total en pesos mexicanos" in src, (
        f"alguien unificó `QBudget.jsx` con el helper del Dashboard. Ahí las cinco ramas "
        f"están ancladas A PROPÓSITO por `test_p1_country_system_f1.py` («byte-identidad del "
        f"dark path»), que es un contrato de otro P-fix: funciona hoy, y unificarlo es una "
        f"decisión de quien lo escribió — no una corrección que se arrastre por simetría. "
        f"[{_MARKER}]"
    )
