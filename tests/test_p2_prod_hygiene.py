"""[P2-BACKEND-DEAD-ADMIN-ENDPOINT + P2-DEPLOY-ENV-GUARD + P2-LINT-GATE · 2026-08-14]
Tres restos de higiene que la auditoría de producción del landing dejó al aire.

1. UN `/admin/` SIN AUTENTICAR, SIN LÍMITE Y MUERTO.
   `app.py` exponía `GET /api/admin/test-proactive` sin `Depends`, sin limitador
   y alcanzable desde internet — verificado: nginx SÍ proxya `/api/admin/*` (a
   diferencia de `/admin/*`, que devuelve el shell del SPA). Y estaba muerto: su
   primera sentencia era `from test_push import trigger_manual_notification`, y
   ese módulo NO EXISTE en el repo. Cero llamantes en `frontend/src`, cero tests
   que lo referencien — `test_p2_admin_rate_limit.py` sólo escanea
   plans/system/notifications, así que `app.py` le quedaba fuera del barrido.
   No es una amplificación de I/O (escribe ~1 KB a un fichero ignorado por git);
   es superficie de ataque que no compra nada.

2. UN DEPLOY QUE INVITA A COMMITEAR UN SECRETO.
   `deploy-mealfit.ps1` busca `SENTRY_AUTH_TOKEN` en `.env`, `.env.production` y
   el `.env` del VPS. Pero `frontend/.env.production` está TRACKED a propósito
   (`.gitignore` lo exceptúa: «solo contiene VITE_*, públicas por diseño») y el
   tar del frontend NO lo excluye, así que ese fichero versionado viaja al VPS y
   es el que la línea lee allí. Añadir la variable ahí la commitea *y* la
   despliega en un solo gesto — y el propio mensaje del script («sin
   SENTRY_AUTH_TOKEN los stacks seguirán minificados») es lo que empuja a hacerlo.
   Exposición HOY: cero. Lo que se cierra es la clase entera, no un incidente.

3. UN GATE DE LINT QUE LLEVABA MESES ESPERANDO SU PROPIO ROADMAP.
   El job declara `continue-on-error: true` con su razón fechada: «245 errores…
   Roadmap: tras cleanup incremental que reduzca el count a 0, flippear a false».
   Medido el 2026-08-14: **5 errores**. Cuatro triviales (`no-unused-vars`) y uno
   real (`react-hooks/refs`: se escribía `isTopmostRef.current` DURANTE el
   render). Ese quinto NO se cierra con un `eslint-disable` — el hook gobierna el
   focus-trap y el ESC de los modales anidados, y silenciar la regla dejaría la
   escritura impura donde está.

Tooltip-anchor: P2-DEPLOY-ENV-GUARD
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_APP = _REPO_ROOT / "backend" / "app.py"
_DEPLOY = _REPO_ROOT / "deploy-mealfit.ps1"
_ENV_PROD = _REPO_ROOT / "frontend" / ".env.production"
_CI = _REPO_ROOT / "backend" / ".github" / "workflows" / "ci.yml"
_CI_FRONTEND = _REPO_ROOT / "frontend" / ".github" / "workflows" / "ci.yml"
_HOOK = _REPO_ROOT / "frontend" / "src" / "hooks" / "useModalAccessibility.js"


def _read(path: Path) -> str:
    if not path.exists():
        pytest.fail(f"[P2-PROD-HYGIENE] No existe {path.relative_to(_REPO_ROOT)}")
    return path.read_text(encoding="utf-8", errors="replace")


def _sin_comentarios_py(texto: str) -> str:
    """Quita comentarios `#` antes de buscar código.

    Necesario porque el comentario que documenta POR QUÉ se borró un endpoint
    contiene su ruta — y entonces el guard que vigila que no vuelva se dispara
    contra el fichero ya corregido. La alternativa sería escribir comentarios que
    no nombren lo que explican, que es peor que no tenerlos.
    """
    return re.sub(r"#.*$", "", texto, flags=re.MULTILINE)


def _sin_comentarios_js(texto: str) -> str:
    texto = re.sub(r"/\*.*?\*/", "", texto, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", texto, flags=re.MULTILINE)


# ---------------------------------------------------------------------------
# 1. El endpoint muerto ya no existe
# ---------------------------------------------------------------------------

def test_no_hay_endpoint_admin_sin_autenticar_en_app_py():
    app = _sin_comentarios_py(_read(_APP))
    assert "test-proactive" not in app, (
        "[P2-BACKEND-DEAD-ADMIN-ENDPOINT] Volvió `/api/admin/test-proactive` a "
        "`app.py`: una ruta `/admin/` SIN auth, SIN limitador y alcanzable desde "
        "internet (nginx sí proxya `/api/admin/*`).\n"
        "Si hace falta un disparador manual, que viva en un router con "
        "`_verify_admin_token` + `_check_admin_rate_limit`, no suelto en app.py."
    )


def test_ningun_endpoint_de_app_py_declara_una_ruta_admin_desprotegida():
    """Blanket sobre la clase, pero acotado al prefijo que SÍ llega desde internet.

    ⚠️ La primera versión vigilaba cualquier ruta con «admin» dentro y marcaba
    `/admin/knobs` y `/admin/cron-health`. Son falsos positivos, y no por
    casualidad: la auditoría los midió y **nginx no proxya `/admin/*`** — esas
    URLs devuelven el shell del SPA, así que sólo son alcanzables desde el
    localhost del VPS. Lo que sí se proxya a FastAPI es `/api/admin/*`, que es
    por donde el endpoint borrado quedaba abierto al mundo.

    Mantener la versión ancha habría obligado a añadir dos excepciones a mano y,
    peor, habría enseñado a leer el rojo de este guard como ruido.
    """
    app = _sin_comentarios_py(_read(_APP))
    sospechosos = []
    for m in re.finditer(r'@app\.(get|post|put|delete|patch)\(\s*["\'](/api/admin/[^"\']*)["\']', app):
        ruta = m.group(2)
        # Ventana del decorador + la firma que le sigue.
        firma = app[m.end(): m.end() + 400]
        if "Depends(" not in firma and "_verify_admin_token" not in firma:
            sospechosos.append(ruta)
    assert not sospechosos, (
        f"[P2-BACKEND-DEAD-ADMIN-ENDPOINT] Rutas `admin` en app.py sin dependencia "
        f"de autenticación: {sospechosos}"
    )


# ---------------------------------------------------------------------------
# 2. El fichero de entorno versionado sólo puede llevar claves públicas
# ---------------------------------------------------------------------------

def test_env_production_solo_contiene_claves_publicas():
    """Es el guard que hace CUMPLIBLE la razón que el .gitignore ya declara."""
    intrusas = []
    for linea in _read(_ENV_PROD).splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave = linea.split("=", 1)[0].strip()
        if not clave.startswith("VITE_"):
            intrusas.append(clave)
    assert not intrusas, (
        f"[P2-DEPLOY-ENV-GUARD] `frontend/.env.production` está VERSIONADO y "
        f"contiene claves que no son públicas: {intrusas}.\n"
        "Todo lo que empieza por `VITE_` acaba dentro del bundle que sirve el "
        "navegador: son públicas por diseño. Cualquier otra cosa aquí es un "
        "secreto commiteado."
    )


def test_el_deploy_verifica_ese_fichero_antes_de_subirlo():
    deploy = _read(_DEPLOY)
    assert "P2-DEPLOY-ENV-GUARD" in deploy, (
        "[P2-DEPLOY-ENV-GUARD] El deploy no comprueba `.env.production` antes de "
        "empaquetarlo.\n"
        "El test de arriba corre en CI, que es post-hoc y no bloquea; el deploy "
        "es el único punto por el que el fichero pasa SIEMPRE camino del VPS."
    )


def test_el_deploy_no_busca_el_token_en_el_fichero_versionado():
    """Leerlo de ahí es lo que invita a escribirlo ahí."""
    deploy = _read(_DEPLOY)
    m = re.search(r"^TOK=.*$", deploy, re.MULTILINE)
    assert m, "[P2-DEPLOY-ENV-GUARD] No se encontró la línea que resuelve SENTRY_AUTH_TOKEN."
    assert ".env.production" not in m.group(0), (
        "[P2-DEPLOY-ENV-GUARD] El deploy vuelve a leer `SENTRY_AUTH_TOKEN` de "
        "`.env.production`, que es un fichero VERSIONADO. Mientras lo lea de ahí, "
        "la vía más corta para «arreglar los sourcemaps» es commitear el secreto."
    )


def test_el_deploy_corre_los_tests_antes_de_empaquetar():
    """[P2-DEPLOY-CI-GATE] La red de seguridad existía pero llegaba tarde.

    La cabecera del propio script lo advertía: «Sube tu copia local TAL CUAL
    (incluye cambios sin commitear)». La CI del backend corre al hacer push —o
    sea DESPUÉS— y la del frontend sólo en `main`, así que un deploy podía
    publicar un árbol que ningún test había visto.
    """
    deploy = _read(_DEPLOY)
    assert "run_ci.ps1" in deploy, (
        "[P2-DEPLOY-CI-GATE] El deploy ya no invoca `scripts/run_ci.ps1`. Ese "
        "wrapper ya existía y reproduce los tres jobs; lo único que faltaba era "
        "llamarlo antes del `tar`."
    )
    i = deploy.find("run_ci.ps1")
    j = deploy.find("Deploy-Backend }")
    assert i != -1 and j != -1 and i < j, (
        "[P2-DEPLOY-CI-GATE] El gate quedó DESPUÉS de las llamadas a deploy. Un "
        "test que corre tras publicar no es un gate, es un informe."
    )
    assert "SkipTests" in deploy, (
        "[P2-DEPLOY-CI-GATE] Falta la válvula de escape `-SkipTests`. Sin una "
        "salida explícita para el hotfix urgente, el primer incidente a las 3 de "
        "la mañana se resuelve comentando el gate — y ya no vuelve."
    )


def test_el_gate_puede_pasar_de_verdad():
    """Un gate que nunca pasa entrena a saltárselo, y entonces no existe.

    La primera versión de este gate corría `run_ci.ps1` entero, o sea
    `pytest tests/ -x`. La suite del backend tiene una BASELINE ROJA —43 fallos
    medidos el 2026-08-14, casi todos anteriores a esta tanda— así que se paraba
    en el primero y abortaba TODOS los despliegues. El efecto real no habría sido
    más calidad: habría sido `-SkipTests` por costumbre, que es exactamente el
    fallo contra el que avisa el comentario del propio gate.

    Mientras la baseline siga roja, el gate corre lo que está verde (vitest +
    build) y lo declara por escrito. Cuando se limpie, se quita el `-SkipBackend`.
    """
    deploy = _read(_DEPLOY)
    # Ancla en la INVOCACIÓN, no en el nombre: el comentario que explica el gate
    # también dice «run_ci.ps1», y un `find` del nombre a secas caía ahí — la
    # quinta vez hoy que una prosa que describe código confunde a un guard que
    # lo busca.
    m = re.search(r"&\s*pwsh[^\n]*run_ci\.ps1[^\n]*", deploy)
    assert m, "[P2-DEPLOY-CI-GATE] No se encontró la invocación de `run_ci.ps1`."
    ventana = deploy[max(0, m.start() - 400): m.end()]
    assert "-SkipBackend" in ventana, (
        "[P2-DEPLOY-CI-GATE] El gate volvió a incluir la suite del backend.\n"
        "Si has limpiado la baseline roja (43 fallos el 2026-08-14), quita también "
        "esta aserción y su explicación — el guard existe para que el cambio sea "
        "deliberado, no para impedirlo. Si NO la has limpiado, el gate abortará "
        "todos los despliegues y el operador aprenderá a pasar `-SkipTests`."
    )


# ---------------------------------------------------------------------------
# 3. El gate de lint cierra su propio roadmap
# ---------------------------------------------------------------------------

def test_el_lint_ya_no_es_no_bloqueante():
    """[P2-I18N-CI-HERMANOS-ROJO-PERMANENTE · 2026-08-23] Reanclado al CI del FRONTEND.

    Este guard leía `backend/.github/workflows/ci.yml` y buscaba ahí un job `frontend-lint`.
    Sobrevivía sólo porque esa copia era un fósil del workflow monorepo: en el CI vivo del
    repo frontend ese job no existe — el lint corre dentro de `quality`.

    O sea que llevaba validando una FORMA que ya no era la del CI que de verdad corre. Al
    limpiar la copia monorepo se puso rojo, y eso fue el guard haciendo su último trabajo
    útil con el ancla vieja: avisar de que miraba al sitio equivocado.

    La propiedad no cambia —el lint BLOQUEA—; cambia dónde se comprueba.
    """
    ci = _read(_CI_FRONTEND)
    m = re.search(r"\n  quality:(.*?)(?=\n  \w|\Z)", ci, re.DOTALL)
    assert m, (
        "[P2-LINT-GATE] No se encontró el job `quality` en el CI del frontend, que es donde "
        "corre el lint. Si el job se renombró, este guard tiene que seguirlo — no borrarse."
    )
    # La INVOCACIÓN, no la palabra. El bloque `quality` menciona «eslint» cuatro veces en su
    # propia prosa (explicando por qué el tope es 66 y qué cambió en el plugin de hooks), así
    # que un `"eslint" in bloque` lo satisface el comentario aunque el paso ya no exista:
    # sustituir el `run` por un `echo` dejaba este assert VERDE.
    sin_comentarios = "\n".join(
        l for l in m.group(1).splitlines() if not l.lstrip().startswith("#")
    )
    assert re.search(r"run:.*\beslint\b", sin_comentarios), (
        "[P2-LINT-GATE] El job `quality` ya no INVOCA eslint: el gate de lint desapareció."
    )
    assert "continue-on-error: true" not in m.group(1), (
        "[P2-LINT-GATE] El job de lint sigue en `continue-on-error: true`.\n"
        "Su propio comentario fija el roadmap: «tras cleanup incremental que "
        "reduzca el count a 0, flippear a false». Medido el 2026-08-14 quedaban "
        "5 errores, todos cerrados en este P-fix. Un gate que nunca se activa es "
        "telemetría, no un gate."
    )


def test_el_hook_de_modales_no_escribe_su_ref_durante_el_render():
    """El 5º error de lint, que no se cerraba con un `disable`.

    ⚠️ La primera versión de este guard buscaba la asignación «al principio de
    línea con cualquier sangría», y eso es exactamente lo que sigue haciendo el
    código correcto: dentro de un `useEffect` la línea también está indentada. El
    guard habría pasado con el bug y con el arreglo — o sea, no habría medido
    nada. Lo que distingue a los dos casos no es la SANGRÍA sino el CONTEXTO.
    """
    hook = _sin_comentarios_js(_read(_HOOK))
    m = re.search(r"isTopmostRef\.current\s*=", hook)
    assert m, (
        "[P2-LINT-GATE] Desapareció la asignación de `isTopmostRef`: sin ella el "
        "modal exterior deja de poder suspenderse y ESC vuelve a cerrar las dos "
        "capas de una tecla."
    )
    # El contexto: entre la declaración del ref y la asignación tiene que abrirse
    # un `useEffect`. Si la asignación va suelta en el cuerpo del componente, ahí
    # no hay nada.
    entre = hook[hook.index("const isTopmostRef"): m.start()]
    assert "useEffect(" in entre, (
        "[P2-LINT-GATE] `useModalAccessibility` vuelve a asignar `isTopmostRef.current` "
        "en el cuerpo del hook, o sea DURANTE el render (react-hooks/refs).\n"
        "Escribir un ref en el render es impuro de verdad: con StrictMode, o si "
        "React descarta un render, la escritura ya ocurrió.\n"
        "El patrón «latest ref» dentro de un `useEffect` sin deps conserva la "
        "semántica que el propio comentario del hook describe — el handler lee el "
        "ref en el instante de la tecla, siempre posterior al commit."
    )
