"""[P2-I18N-HIST-FORENSE-ROTULOS + P2-I18N-FECHA-LANZAMIENTO-CLAVADA · 2026-08-22]
39 rótulos del panel forense, y la fecha de lanzamiento inyectada en español.

Y dos gaps del mismo grupo CERRADOS POR MEDICIÓN, sin tocar código — que es el resultado
más repetido de esta ola:

  · `P2-I18N-HIST-CONTROLES-SUELTOS` («cinco controles del modal en español, uno con la
    traducción ya escrita»): cero hallazgos del detector en los tres ficheros del Historial,
    cero huérfanas en los cuatro catálogos. Lo cerró una tanda anterior de esta misma ola.

  · `P2-I18N-SUPLEMENTOS-SIN-TRADUCIR` («cuatro de doce al italiano y una al brasileño»):
    las «sin traducir» son `Creatina`, `Magnesio`, `Colágeno`, `Probióticos`… — **cognados
    exactos**, correctos en su idioma — más `BCAA / EAA` y `Omega-3`, que son siglas. Cero
    defectos. Es la misma cuenta que ya falló con las 266 cadenas idénticas de pt-BR:
    **contar valores iguales a su clave no es contar cadenas sin traducir**.

═══════════════════════════════════════════════════════════════════════════════
LO QUE ESTE LOTE DEJÓ VER, y vale más que los 39 rótulos
═══════════════════════════════════════════════════════════════════════════════

`detectarEnFuente` —el escáner del trinquete de «español sin envolver»— reportaba
**CERO hallazgos en `History.jsx`** mientras 39 rótulos del panel forense estaban en
español. No es que fallara: es que esos rótulos viven en posiciones que no inspecciona —
valores de un objeto literal (`{ llm: 'Calidad LLM' }`) y segundos elementos de tuplas
(`['recovery_attempts', 'Reintentos recovery', 'int']`).

Un trinquete en 78 no significa «quedan 78 cadenas»: significa «el escáner ve 78». Queda
anotado en `P1-I18N-ESCANER-RECALL`.

Los 6 hallazgos que el escáner SÍ reporta en los paneles del Historial son, además, FALSOS
POSITIVOS: son los IDs canónicos de los grupos por fecha («Esta semana»), que se traducen
al PINTAR con literales — a propósito, y con su razón escrita al lado. Este guard los ancla
para que nadie los «arregle».

tooltip-anchor: P2-I18N-HIST-FORENSE-ROTULOS
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_FRONT = _BACKEND.parent / "frontend" / "src"

_MARKER = "P2-I18N-HIST-FORENSE-ROTULOS"


def _fuente(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip(f"no existe {p} (¿repo hermano sin clonar?)")
    return io.open(p, encoding="utf-8").read()


def test_los_rotulos_del_panel_forense_pasan_por_el_motor() -> None:
    """Las tuplas del catálogo: `['clave', 'Rótulo', 'tipo']`."""
    src = _fuente("pages/History.jsx")
    crudas = re.findall(r"\['[a-z0-9_]+'\s*,\s*'([^']+)'\s*,\s*'[a-z_]+'\s*\]", src)
    assert not crudas, (
        f"{len(crudas)} rótulo(s) del catálogo forense volvieron a ir sin `t()`. Viven "
        f"dentro de TUPLAS, que es una posición donde el escáner del trinquete no mira: "
        f"reportaba cero hallazgos en este fichero con 39 rótulos en español.\n"
        + "\n".join(f"  · {c!r}" for c in crudas[:12])
        + f"\n[{_MARKER}]"
    )
    # Y quedan efectivamente envueltos.
    envueltas = re.findall(r"\['[a-z0-9_]+'\s*,\s*t\('[^']+'\)\s*,\s*'[a-z_]+'\s*\]", src)
    assert len(envueltas) >= 28, (
        f"sólo {len(envueltas)} tuplas del catálogo forense pasan por `t()`; eran 28. "
        f"[{_MARKER}]"
    )


def test_el_mapa_de_tiers_pasa_por_el_motor() -> None:
    src = _fuente("pages/History.jsx")
    m = re.search(r"const _TIER_LABELS = \{(.*?)\n\s+\};", src, re.S)
    assert m, f"desapareció `_TIER_LABELS` de History.jsx [{_MARKER}]"
    crudos = re.findall(r"^\s+[a-z]+: '([^']+)',$", m.group(1), re.M)
    assert not crudos, (
        f"los rótulos de tier volvieron a ir sin `t()`: {crudos}. Es un mapa literal, otra "
        f"posición ciega para el escáner. [{_MARKER}]"
    )


def test_el_chip_de_lista_no_identifica_por_el_rotulo() -> None:
    """La `key` de React no puede salir de un texto que cambia con el idioma."""
    src = _fuente("pages/History.jsx")
    assert "key={`lcl-list-${label}`}" not in src, (
        f"la `key` de React del chip volvió a derivar del RÓTULO, que ahora está traducido. "
        f"La identidad sale del canónico, jamás del display — y dos rótulos podrían colisionar "
        f"al traducirse. [{_MARKER}]"
    )
    assert "key={`lcl-list-${id}`}" in src, (
        f"el chip dejó de identificarse por su `id` canónico. [{_MARKER}]"
    )


def test_la_fecha_de_lanzamiento_se_deriva_del_iso() -> None:
    src = _fuente("pages/Upgrade.jsx")
    assert "fecha: LAUNCH_OFFER.deadlineShort" not in src, (
        f"la píldora de lanzamiento volvió a inyectar `deadlineShort` —la cadena ESPAÑOLA de "
        f"la landing, «15 sep»— dentro de una frase traducida: «Lancement · augmente le 15 "
        f"sep». [{_MARKER}]"
    )
    # La secuencia COMPLETA de opciones, no `timeZone: 'UTC'` suelto: ese fragmento aparece
    # también en el comentario que explica por qué es load-bearing, así que un `in` pelado se
    # daba por satisfecho con mi propia prosa mientras el código no lo llevaba.
    assert "LAUNCH_OFFER.deadlineISO" in src and re.search(
        r"month: 'short', timeZone: 'UTC'", src
    ), (
        f"la fecha dejó de derivarse del ISO con `timeZone: 'UTC'`. Ese huso es load-bearing: "
        f"`new Date('2026-09-15')` es medianoche UTC y al oeste de Greenwich se formatearía "
        f"como día 14 — el mismo «¿día de quién?» que documenta el comentario de "
        f"`P3-LAUNCH-OFFER-LOCAL-DAY` al lado de la constante. [{_MARKER}]"
    )


def test_los_ids_de_grupo_del_historial_siguen_sin_traducir_a_proposito() -> None:
    """La frontera: el bucket es un ID que resulta ser su propio texto español."""
    for panel in ("components/history/HistoryDesktopPanel.jsx",
                  "components/history/HistoryMobilePanel.jsx"):
        src = _fuente(panel)
        assert 'return "Esta semana"' in src, (
            f"{panel}: alguien tradujo el bucket en `bucketOf`. Ese valor es la CLAVE con la "
            f"que se agrupan las filas y el que ordena `BUCKET_ORDER`: traducirlo río arriba "
            f"rompe el agrupamiento. Se traduce al PINTAR, en `bucketTitle`. El escáner del "
            f"trinquete los cuenta como «sin envolver» y son un FALSO POSITIVO. [{_MARKER}]"
        )
        assert 't("Esta semana")' in src, (
            f"{panel}: `bucketTitle` dejó de traducir el rótulo al pintarlo. [{_MARKER}]"
        )


def test_las_etiquetas_de_suplemento_estan_bien_y_no_hay_nada_que_arreglar() -> None:
    """Cuenta refutada: los «idénticos» son cognados y siglas.

    Se ancla para que la próxima auditoría no vuelva a abrir el mismo gap contando valores
    iguales a su clave — ya pasó dos veces en esta ola.
    """
    locales = _FRONT / "i18n" / "locales"
    if not locales.exists():
        pytest.skip("sin catálogos")

    # Los que DEBEN poder ser idénticos: siglas y cognados exactos.
    tolerados = {
        "BCAA / EAA",       # sigla, idéntica en los cinco idiomas
        "Omega-3",          # nomenclatura química
        "Creatina",         # es/it/pt comparten la palabra
        "Magnesio",         # es/it
        "Colágeno", "Multivitamínico", "Probióticos",
        "Proteína Whey", "Prot. Vegana",
    }
    # Sólo las DOCE etiquetas, no todo `t()` del fichero: un barrido al fichero entero se
    # llevaba por delante el título de la sección («Incluir Suplementos», que en portugués es
    # esa misma frase y es correcta) y acusaba a la traducción de no existir.
    src_sup = _fuente("components/assessment/questions/QSupplements.jsx")
    m = re.search(r"const getSupplementLabels = \(t\) => \(\{(.*?)\n\}\);", src_sup, re.S)
    assert m, f"desapareció `getSupplementLabels` [{_MARKER}]"
    claves = re.findall(r"t\('([^']+)'\)", m.group(1))
    assert len(claves) >= 12, f"se perdieron etiquetas de suplemento [{_MARKER}]"

    sin_traducir = []
    for p in sorted(locales.glob("*.json")):
        datos = json.loads(io.open(p, encoding="utf-8").read())
        for k in claves:
            if datos.get(k) == k and k not in tolerados:
                sin_traducir.append(f"{p.stem}: {k!r}")

    assert not sin_traducir, (
        f"etiqueta(s) de suplemento realmente sin traducir (no cognado ni sigla):\n"
        + "\n".join(f"  · {s}" for s in sin_traducir)
        + f"\n[{_MARKER}]"
    )
