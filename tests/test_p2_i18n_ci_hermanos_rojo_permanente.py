"""[P2-I18N-CI-HERMANOS-ROJO-PERMANENTE + 3 · 2026-08-22] La tercera tanda de la ola final.

LO QUE ESTA TANDA ENSEÑA

1. DEJÉ DE DEDUCIR Y MIRÉ LOS RUNS. El gap decía «los otros dos CI llevan 44/44 y 98/100 en
   rojo, y ahí viven los pytest que anclan el gate de i18n». Con `gh run view --log-failed`
   resultaron ser DOS causas distintas, y una de las dos preocupaciones era falsa:

   · El CI del repo **backend** muere en 15 segundos sin ejecutar un solo test:
     `No file in .../MealfitRD-Backend matched to [backend/requirements.txt]`. Su `ci.yml`
     era una copia del workflow MONOREPO, pidiendo `backend/` y `frontend/` en un checkout
     cuya raíz ES el backend. Un CI que lleva 44 corridas en rojo sin haber ejecutado nada
     no es una red: es ruido que entrena a ignorar el CI entero.

   · El CI del repo **frontend** sí ejecuta, y **su gate de i18n está VERDE** (paso 5 del
     job `quality`). Lo que lo tumba es `npm test`. De sus tres fallos, reproduciendo con
     `CI=true TZ=UTC`, sólo UNO es real: los dos de `NativeShell` no se reproducen ni
     aislados ni en suite completa.

2. UN TEST QUE SÓLO PASA DONDE VIVE SU AUTOR. `plans.launch_offer_expiry` construye el
   instante en hora de RD, pero `isLaunchOfferActive` sigue —desde `P3-LAUNCH-OFFER-LOCAL-DAY`—
   el día LOCAL DEL USUARIO, y sin segundo argumento lo lee del reloj del PROCESO. En un
   runner en UTC, «23:59 del 15 en RD» ya es el día 16.

   Lo que lo hace didáctico: **la mitad de abajo de ese mismo fichero ya inyecta el huso**, y
   su comentario dice la lección entera — «un caso cuyo resultado depende de dónde corra no
   es una defensa, es un intermitente». La mitad de arriba es anterior y nunca la aprendió.

3. UN PANEL QUE SIEMPRE DICE CERO ES UN PANEL QUE NADIE MIRA. `pipeline_metrics` no tiene ni
   una fila con `node='plan_display_i18n'` (contra 14.835 de la semana). O sea que «la
   telemetría se escribe y no la lee nadie» es, medido, «no hay nada que leer». Eso descarta
   el arreglo obvio —un cron agregador— y deja el correcto: una alerta EMITIDA desde donde ya
   se detecta el fallo, que cuesta cero mientras la capa no corra.

4. UN PUENTE DE UN IDIOMA, DICHO COMO TAL. `name_en` está poblado al 347/347 y difiere del
   español en 329 filas (medido contra Neon), así que enviarlo arregla el buscador para
   en-US. Para fr/it/pt no hay columna que enviar. Un puente a medias presentado como puente
   completo es peor que ninguno: nadie vuelve a mirarlo.

tooltip-anchor: P2-I18N-CI-HERMANOS-ROJO-PERMANENTE
"""
from __future__ import annotations

import io
import re
from pathlib import Path
import pytest

# [P2-CI-BACKEND-SIBLINGS · 2026-09-04] Este módulo necesita el catálogo/la base de datos o el
# .env local (pasa en el checkout del dueño; en el CI sin NEON_DATABASE_URL se salta con motivo).
pytestmark = pytest.mark.needs_local_data

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_FRONT = _ROOT / "frontend"
_MARKER = "P2-I18N-CI-HERMANOS-ROJO-PERMANENTE"


def _leer(p: Path) -> str:
    return io.open(p, encoding="utf-8").read()


def _codigo_py(src: str) -> str:
    """Sin las líneas de comentario. Mi propia prosa ya ha satisfecho mis propios guards
    tres veces en un solo fichero este mes."""
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))


# ───────────────────── 1. el CI del repo backend, en su propio repo ─────────────────

def test_el_ci_del_backend_no_busca_un_subdirectorio_backend():
    yml = _leer(_BACKEND / ".github" / "workflows" / "ci.yml")
    codigo = _codigo_py(yml)
    # [P1-I18N-CI-SIN-VEREDICTO · 2026-08-23] Misma reanclada que la de abajo: las rutas
    # `backend/...` sólo son un error si el checkout no clona en `backend/`.
    clona_en_backend = bool(re.search(r"path:\s*backend\b", codigo))
    if "backend/requirements.txt" in codigo:
        assert clona_en_backend, (
            "el CI del repo BACKEND pide `backend/requirements.txt` sin clonar en `backend/`. "
            "Su raíz ES el backend: ese fichero no existe como subdirectorio y el job muere "
            f"en 15 s en `Set up Python`, sin ejecutar un solo test [{_MARKER}]"
        )
    # [P1-I18N-CI-SIN-VEREDICTO · 2026-08-23] Esto prohibía `working-directory: backend` a
    # secas, y tenía razón ENTONCES: el job clonaba en la raíz y ese subdirectorio no
    # existía. Ahora el checkout va con `path: backend` a propósito —para reproducir la
    # disposición hermana que `parents[2]` asume en 400 tests— y el subdirectorio SÍ existe.
    # La propiedad real nunca fue «jamás ese directorio»: es «las rutas del job son
    # coherentes con dónde se clona». Un `working-directory: backend` sólo es un error si
    # nadie clonó ahí.
    if "working-directory: backend" in codigo:
        assert re.search(r"path:\s*backend\b", codigo), (
            "hay un `working-directory: backend` pero el checkout NO clona en `backend/`: el "
            f"subdirectorio no existe y el job muere sin ejecutar un test [{_MARKER}]"
        )


def test_el_ci_del_backend_no_tiene_jobs_de_frontend():
    """Ese código vive en otro repo, con su propio CI. Duplicarlo aquí apuntando a un
    directorio inexistente no es redundancia: es un rojo garantizado."""
    codigo = _codigo_py(_leer(_BACKEND / ".github" / "workflows" / "ci.yml"))
    # [P2-CI-BACKEND-SIBLINGS · 2026-09-04] El frontend (repo público) SÍ se descarga en
    # `frontend/` y se instalan sus deps (`npm ci --ignore-scripts`) porque ~400 tests del
    # backend leen `../frontend/src` y sondean sus scripts de node. Eso no es un "job de
    # frontend": lo vetado sigue siendo construir/testear/lintar el frontend desde aquí.
    for paso in ("npm run build", "npm test", "vitest", "npm run lint", "playwright", "npm run test:e2e"):
        assert paso not in codigo, (
            f"volvió un job de frontend ({paso!r}) al CI del repo backend [{_MARKER}]"
        )


def test_el_ci_del_backend_no_para_en_el_primer_fallo():
    """Es la lección que el CI de la raíz ya aprendió y este espejo no: `-x` convierte cada
    corrida en UN dato en vez del mapa, y esconde todo lo que viene detrás."""
    codigo = _codigo_py(_leer(_BACKEND / ".github" / "workflows" / "ci.yml"))
    assert '-m "not e2e" -x' not in codigo, (
        f"volvió el `-x`: un rojo cualquiera esconde el resto de la suite [{_MARKER}]"
    )
    # [P1-I18N-CI-SIN-VEREDICTO · 2026-08-23] Esto EXIGÍA `--maxfail`, como alternativa al
    # `-x`. Era mejor que `-x`, y aun así dejaba el job sin veredicto: con 25 errores de
    # COLECCIÓN —ficheros que no se pueden importar, no tests rojos— el cupo se agotaba
    # antes del primer test. Medido con `gh run view`. La propiedad de este test es «la
    # corrida devuelve el mapa entero, no un dato»; `--maxfail` la cumplía a medias y
    # ahora se exige que NO esté. `test_p1_i18n_ci_sin_veredicto.py` lo ancla también.
    assert "--maxfail" not in codigo, (
        f"`--maxfail` ha vuelto: con errores de colección la corrida termina sin veredicto [{_MARKER}]"
    )


# ───────────────────── 2. el test que sólo pasaba en RD ─────────────────────────────

def test_los_casos_de_la_oferta_inyectan_el_huso():
    src = _leer(_FRONT / "src" / "__tests__" / "plans.launch_offer_expiry.test.js")
    codigo = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("//"))
    assert "RD_OFFSET" in codigo, (
        "los casos volvieron a depender del reloj del runner. `isLaunchOfferActive` sigue el "
        "día LOCAL del usuario, así que sin el segundo argumento «23:59 del último día en "
        f"RD» sólo da `true` en una máquina puesta en RD [{_MARKER}]"
    )
    sueltos = [
        l.strip() for l in codigo.splitlines()
        if "isLaunchOfferActive(enRD(" in l and "RD_OFFSET" not in l
    ]
    assert not sueltos, (
        "estas llamadas construyen el instante en hora de RD pero no inyectan el huso, así "
        f"que su resultado depende de dónde corran:\n  " + "\n  ".join(sueltos)
        + f"\n[{_MARKER}]"
    )


# ───────────────────── 3. la telemetría del display gana lector ─────────────────────

def test_el_display_emite_alerta_en_las_razones_reales():
    src = _codigo_py(_leer(_BACKEND / "plan_display_i18n.py"))
    # LAS DOS MITADES. Buscar sólo el nombre lo satisface el CALL SITE aunque la definición
    # ya no exista — renombrar `def _emit_degraded_alert` dejaba este guard verde con el
    # lector borrado. Que un nombre aparezca no dice que la función exista.
    assert "def _emit_degraded_alert(" in src, (
        "desapareció la DEFINICIÓN del lector de la telemetría: un enriquecimiento que se "
        f"cae vuelve a ser indistinguible de uno que nunca se disparó [{_MARKER}]"
    )
    assert "_emit_degraded_alert(plan_id, user_id, locale, _razon)" in src, (
        "la definición existe pero nadie la LLAMA: un lector que no se invoca es peor que "
        "no tenerlo, porque parece que hay vigilancia"
    )
    assert "plan_display_i18n_degraded" in src


def test_las_razones_benignas_no_alertan():
    """`dedupe_locked` es el caso NORMAL bajo concurrencia e `inflight_cap` es el techo de
    hilos haciendo su trabajo. Alertar por ellas fabricaría una tasa de error que no existe
    — el mismo error que `P2-I18N-OBSERVABILIDAD-CERO` evitó contando `SUPERSEDED` aparte."""
    import sys
    if str(_BACKEND) not in sys.path:
        sys.path.insert(0, str(_BACKEND))
    import plan_display_i18n as m

    for benigna in ("dedupe_locked", "inflight_cap", "disabled", "ok"):
        assert benigna in m._RAZONES_BENIGNAS, f"«{benigna}» dejó de ser benigna"
    for real in ("circuit_breaker_open", "partial_loss", "json_malformado"):
        assert real not in m._RAZONES_BENIGNAS, (
            f"«{real}» pasó a benigna: un fallo real dejaría de alertar [{_MARKER}]"
        )


def test_la_alerta_esta_documentada_y_el_escaner_la_ve():
    """El escáner de drift miraba SEIS ficheros y este módulo no estaba. Un `alert_key`
    fuera del conjunto escaneado es un `alert_key` sin contrato."""
    escaner = _leer(_BACKEND / "tests" / "test_p2_audit_4_alert_keys_documented.py")
    assert '_BACKEND / "plan_display_i18n.py"' in escaner, (
        f"el escáner volvió a dejar fuera el módulo del `_display` [{_MARKER}]"
    )
    doc = _leer(_BACKEND / "docs" / "system_alerts_resolution_table.md")
    assert "plan_display_i18n_degraded" in doc


# ───────────────────── 4. el puente del buscador ────────────────────────────────────

def test_el_catalogo_envia_el_gloss_ingles():
    src = _leer(_BACKEND / "routers" / "user_data.py")
    # [P1-COUNTRY-GLOSS-SOLO-INGLES · 2026-08-23] La proyección ganó `gloss_es`
    # y el SELECT se partió en dos líneas: el ancla pasa a la PROPIEDAD (que
    # `name_en` siga proyectado), no a la forma exacta de la lista de columnas.
    assert "slug, name, name_en," in src, (
        "el endpoint dejó de enviar `name_en`: buscar «chicken» o «rice» vuelve a devolver "
        "CERO resultados — sin error, simplemente vacío, que se lee como «ese alimento no "
        f"existe» [{_MARKER}]"
    )


def test_el_buscador_empareja_por_el_gloss_pero_selecciona_el_canonico():
    """Se ensancha por dónde se BUSCA, nunca lo que se guarda: el valor seleccionado sigue
    siendo el nombre español canónico con el que resuelve el motor."""
    src = _leer(_FRONT / "src" / "components" / "assessment" / "questions" / "QStapleFoods.jsx")
    codigo = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("//"))
    assert "norm(m.name_en || '').includes(q)" in codigo, (
        f"el buscador dejó de mirar el gloss inglés [{_MARKER}]"
    )
    assert "selectedLower.has(norm(m.name))" in codigo, (
        "lo que se selecciona dejó de ser `m.name`. Ese es el identificador con el que "
        "resuelven `pantry_names_match`, el guard de coherencia y el backstop de alergias"
    )
