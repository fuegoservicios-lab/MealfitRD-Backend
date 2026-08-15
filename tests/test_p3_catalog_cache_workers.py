"""[P3-CATALOG-CACHE-WORKERS · 2026-08-15] La caché del catálogo asume UN worker.

`routers/supermarket.py` cachea el catálogo del supermercado en memoria del
proceso, y su comentario ya explica el riesgo: con más de un worker de uvicorn, un
PATCH atendido por el worker A no limpia la caché del worker B, que seguiría
sirviendo precios viejos hasta que expire su TTL (≤5 min). Esos precios alimentan
el costeo de marca del Dashboard y de la Nevera.

LO QUE FALTABA NO ERA LA DOCUMENTACIÓN: ERA QUE ALGO FALLARA. El comentario dice
«no es hipotético-lejano: subir workers es lo primero que se toca» — y tenía razón,
porque subir `--workers` es la respuesta natural a «el backend va justo de CPU».
Quien lo haga estará mirando systemd, no `supermarket.py`, y no hay ningún punto
del camino que le ponga delante la consecuencia.

Este test es ese punto. No prohíbe escalar: prohíbe escalar **en silencio**. Si
alguien sube el número, esto falla y le dice exactamente qué hay que resolver
antes (invalidación por canal compartido: Redis pub/sub, o una fila en
`app_kv_store` que los workers consulten).

Es el mismo patrón de cross-link que `test_p2_hist_audit_14_marker_test_link.py`
usa para el marker: dos ficheros que sólo son correctos JUNTOS, y un test que los
ata.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UNIT = _REPO_ROOT / "backend" / "infra" / "systemd" / "mealfit-backend.service"
_SUPERMARKET = _REPO_ROOT / "backend" / "routers" / "supermarket.py"


def _leer(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"{p} no existe en este checkout")
    return p.read_text(encoding="utf-8")


def test_un_solo_worker_mientras_la_cache_sea_por_proceso() -> None:
    unit = _leer(_UNIT)
    m = re.search(r"--workers\s+(\d+)", unit)
    assert m, (
        "No encuentro `--workers N` en el ExecStart. Sin ese flag uvicorn arranca "
        "con 1, que es lo correcto hoy — pero hazlo explícito: este test necesita "
        "poder leerlo, y el siguiente lector también."
    )
    workers = int(m.group(1))

    # ⚠️ Se miran las líneas de CÓDIGO, no el fichero entero.
    #
    # La primera versión buscaba `redis|app_kv_store` en todo el texto y la
    # mutación de verificación (subir a `--workers 4`) PASÓ tranquilamente: el
    # comentario de `supermarket.py` menciona «Redis pub/sub o una fila en
    # app_kv_store» al describir justamente lo que HABRÍA que construir. O sea que
    # el guard se daba por satisfecho con la prosa que explica el problema.
    #
    # Es la misma trampa que ya mordió hoy en otro guard: un patrón que casa con un
    # vecino no vigila a su objetivo. Aquí el vecino era la solución hipotética.
    codigo = "\n".join(
        ln for ln in _leer(_SUPERMARKET).splitlines()
        if not ln.lstrip().startswith("#")
    )
    tiene_canal_compartido = bool(
        re.search(r"^\s*(?:from|import)\s+.*redis|redis_client|app_kv_store\s*\(|"
                  r"\bpublish\s*\(|\bsubscribe\s*\(", codigo, re.IGNORECASE | re.MULTILINE)
    )

    assert workers == 1 or tiene_canal_compartido, (
        f"`--workers {workers}` con la caché del catálogo todavía en memoria del "
        "PROCESO.\n\n"
        "Un PATCH del catálogo lo atiende UN worker y sólo invalida SU caché; los "
        "demás siguen sirviendo precios viejos hasta que expire su TTL (≤5 min). "
        "Esos precios alimentan el costeo de marca del Dashboard y de la Nevera, "
        "así que el síntoma sería «unos usuarios ven el precio nuevo y otros el "
        "viejo», que es de los más difíciles de reproducir.\n\n"
        "Para escalar: mueve la invalidación a un canal compartido (Redis pub/sub, "
        "o una fila en `app_kv_store` que los workers consulten) y este test dejará "
        "de bloquear solo. Escalar está bien; escalar en silencio, no."
    )


def test_la_cache_declara_su_supuesto_junto_al_codigo() -> None:
    """El porqué vive donde está el efecto, no sólo en este test.

    Si el comentario desaparece, alguien puede leer `supermarket.py` entero sin
    enterarse de que hay un supuesto de despliegue metido en su diseño.
    """
    src = _leer(_SUPERMARKET)
    assert re.search(r"--workers|worker", src), (
        "`routers/supermarket.py` perdió la nota sobre el número de workers. Es el "
        "único sitio donde un lector del código de la caché puede enterarse de que "
        "su corrección depende de cómo se arranca el proceso."
    )
