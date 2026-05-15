"""[P3-AUDIT-4 · 2026-05-15] Test parser-based: el Service Worker
(`custom-sw.js`) tiene handler `pushsubscriptionchange`, y el cliente
(`pushNotifications.js` + `main.jsx`) tiene listener para el postMessage
que el SW emite.

Por qué este test:
    El browser puede invalidar/rotar credentials de push (FCM rotation,
    refresh interno, app reset) sin reinstalación del SW. Sin handler
    `pushsubscriptionchange`:
    - Notificaciones dejan de llegar al usuario hasta abrir la app.
    - Backend tiene endpoint zombie (subscription endpoint inválido) que
      genera errores 410 Gone al intentar emitir notifs.

Fix esperado (two-phase):
    FASE 1 (SW): handler `pushsubscriptionchange` que invoca
    `pushManager.subscribe({...})` con `event.oldSubscription.options
    .applicationServerKey` (el SW NO tiene VITE_VAPID_PUBLIC_KEY del
    cliente). Best-effort: try/catch silencioso.

    FASE 2 (SW): `postMessage` a clientes abiertos con
    `{type: 'pushsubscriptionchange', subscription: newSub.toJSON()}`.

    FASE 3 (cliente): listener `navigator.serviceWorker.addEventListener
    ('message', ...)` registrado durante bootstrap (main.jsx), reposta al
    backend con `subscribeToPushNotifications()` (que ya tiene auth).

Drift detection:
    - SW: `addEventListener('pushsubscriptionchange', ...)`.
    - SW: invoca `pushManager.subscribe` con `applicationServerKey`.
    - SW: usa `event.waitUntil(...)` para que el evento no aborte el SW.
    - SW: `postMessage` con `type: 'pushsubscriptionchange'`.
    - Cliente: `registerPushSubscriptionChangeListener` exportado en
      `pushNotifications.js`.
    - Cliente: `main.jsx` invoca el registro durante bootstrap.

Cross-link convention (P2-HIST-AUDIT-14): slug `p3_audit_4` matchea este
archivo. Bundle marker `P3-AUDIT-4` (último alfabético del bundle P3).

Tooltip-anchor: P3-AUDIT-4-START | gap audit 2026-05-15
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SW_JS = _REPO_ROOT / "frontend" / "src" / "custom-sw.js"
_PUSH_JS = _REPO_ROOT / "frontend" / "src" / "utils" / "pushNotifications.js"
_MAIN_JSX = _REPO_ROOT / "frontend" / "src" / "main.jsx"


@pytest.fixture(scope="module")
def sw_src() -> str:
    return _SW_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def push_src() -> str:
    return _PUSH_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def main_src() -> str:
    return _MAIN_JSX.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. SW: handler `pushsubscriptionchange` declarado
# ---------------------------------------------------------------------------
def test_sw_declares_pushsubscriptionchange_handler(sw_src: str):
    assert re.search(
        r"addEventListener\s*\(\s*['\"]pushsubscriptionchange['\"]",
        sw_src,
    ), (
        "P3-AUDIT-4 regresión: `addEventListener('pushsubscriptionchange', ...)` "
        "no encontrado en custom-sw.js. Sin handler, las notifs paran de "
        "llegar al usuario cuando el browser rota FCM credentials."
    )


def test_sw_uses_event_waituntil(sw_src: str):
    """`event.waitUntil(promise)` extiende la vida del SW mientras la
    promise resuelve. Sin esto, el SW puede matarse antes de re-subscribir
    o postear al cliente."""
    # Localizar el handler con regex específica (no la string en comments
    # narrativos). Ventana amplia para cubrir handler entero (~4500 chars).
    handler_match = re.search(
        r"addEventListener\s*\(\s*['\"]pushsubscriptionchange['\"][^)]*\)\s*=>",
        sw_src,
    )
    assert handler_match is not None, (
        "P3-AUDIT-4 regresión: handler `pushsubscriptionchange` no encontrado "
        "via regex."
    )
    handler_window = sw_src[handler_match.start():handler_match.start() + 5000]
    assert re.search(r"event\.waitUntil\s*\(", handler_window), (
        "P3-AUDIT-4 regresión: el handler `pushsubscriptionchange` no usa "
        "`event.waitUntil(...)`. Sin esto, el SW puede matarse antes de "
        "completar el re-subscribe + postMessage."
    )


def test_sw_re_subscribes_with_application_server_key(sw_src: str):
    """El SW DEBE invocar `pushManager.subscribe({applicationServerKey: ...})`
    usando el key del `event.oldSubscription`. Sin esto, no se re-suscribe
    localmente y las notifs siguen sin llegar hasta que el cliente bootstrap."""
    handler_idx = sw_src.find("pushsubscriptionchange")
    handler_window = sw_src[handler_idx:handler_idx + 3000]
    assert re.search(
        r"pushManager\.subscribe\s*\(",
        handler_window,
    ), (
        "P3-AUDIT-4 regresión: `pushManager.subscribe(...)` no encontrado "
        "en el handler. Sin re-subscribe local, el SW no recupera la "
        "subscription hasta el próximo bootstrap del cliente."
    )
    assert re.search(
        r"applicationServerKey",
        handler_window,
    ), (
        "P3-AUDIT-4 regresión: `applicationServerKey` no se pasa al "
        "`pushManager.subscribe(...)`. Sin la key, browser rechaza el "
        "subscribe."
    )


def test_sw_re_subscribes_with_oldsubscription_key(sw_src: str):
    """Específicamente debe usar `event.oldSubscription.options.
    applicationServerKey` — el SW NO tiene access al VITE_VAPID_PUBLIC_KEY
    del cliente. La oldSubscription conserva el key original."""
    handler_idx = sw_src.find("pushsubscriptionchange")
    handler_window = sw_src[handler_idx:handler_idx + 3000]
    assert re.search(
        r"oldSubscription[\s\S]{0,100}?(?:options|applicationServerKey)",
        handler_window,
    ), (
        "P3-AUDIT-4 regresión: `event.oldSubscription.options.applicationServerKey` "
        "no se referencia. El SW no puede obtener el VAPID key de otra "
        "manera (no tiene access a `import.meta.env` del cliente)."
    )


# ---------------------------------------------------------------------------
# 2. SW: postMessage a clientes con type='pushsubscriptionchange'
# ---------------------------------------------------------------------------
def test_sw_postmessage_to_clients(sw_src: str):
    handler_idx = sw_src.find("self.addEventListener('pushsubscriptionchange'")
    if handler_idx == -1:
        handler_idx = sw_src.find("pushsubscriptionchange")
    assert handler_idx > -1
    handler_window = sw_src[handler_idx:handler_idx + 3000]
    assert "clients.matchAll" in handler_window or "self.clients.matchAll" in handler_window, (
        "P3-AUDIT-4 regresión: `clients.matchAll(...)` no se invoca en el "
        "handler. Sin esto, no podemos notificar a la pestaña abierta para "
        "que reposte la nueva subscription al backend (con auth)."
    )
    assert re.search(
        r"\.postMessage\s*\(",
        handler_window,
    ), (
        "P3-AUDIT-4 regresión: `client.postMessage(...)` no se invoca. Sin "
        "esto, el cliente abierto no se entera del subscription change y no "
        "reposta al backend."
    )


def test_sw_postmessage_payload_has_correct_type(sw_src: str):
    """El payload del postMessage debe incluir `type: 'pushsubscriptionchange'`
    para que el listener del cliente lo filtre del resto de messages."""
    assert re.search(
        r"type\s*:\s*['\"]pushsubscriptionchange['\"]",
        sw_src,
    ), (
        "P3-AUDIT-4 regresión: payload del postMessage no tiene "
        "`type: 'pushsubscriptionchange'`. Sin él, el listener del cliente "
        "no puede discriminar este evento de otros messages del SW."
    )


# ---------------------------------------------------------------------------
# 3. Cliente: registerPushSubscriptionChangeListener exportado
# ---------------------------------------------------------------------------
def test_client_exports_listener_register_fn(push_src: str):
    assert re.search(
        r"export\s+function\s+registerPushSubscriptionChangeListener\s*\(",
        push_src,
    ), (
        "P3-AUDIT-4 regresión: `export function "
        "registerPushSubscriptionChangeListener(...)` no encontrado en "
        "pushNotifications.js. Sin este helper, el cliente no tiene forma "
        "de registrar el listener del postMessage del SW."
    )


def test_client_listener_filters_message_type(push_src: str):
    """El listener interno debe filtrar `event.data.type ===
    'pushsubscriptionchange'` para no reaccionar a otros messages."""
    assert re.search(
        r"event\.data\.type\s*(?:===|!==)\s*['\"]pushsubscriptionchange['\"]",
        push_src,
    ), (
        "P3-AUDIT-4 regresión: el listener del cliente no filtra el "
        "`type` del message. Reaccionaría a otros postMessage del SW y "
        "haría sync push spurious."
    )


def test_client_listener_reuses_subscribe(push_src: str):
    """Al recibir el message, el listener debe invocar
    `subscribeToPushNotifications()` que ya tiene auth + maneja el POST al
    backend (no reimplementar la lógica)."""
    # Buscar invocación de `subscribeToPushNotifications` DENTRO de la fn
    # `registerPushSubscriptionChangeListener`.
    fn_match = re.search(
        r"export\s+function\s+registerPushSubscriptionChangeListener\s*\([^)]*\)\s*\{",
        push_src,
    )
    assert fn_match
    body = push_src[fn_match.end():fn_match.end() + 2000]
    assert "subscribeToPushNotifications" in body, (
        "P3-AUDIT-4 regresión: el listener no invoca "
        "`subscribeToPushNotifications()`. Reimplementar el POST al backend "
        "duplica lógica + propenso a drift (auth, error handling, retry)."
    )


def test_client_listener_is_idempotent(push_src: str):
    """Llamar `registerPushSubscriptionChangeListener()` 2+ veces solo
    debe registrar el listener una vez. Sin guard, el handler corre N
    veces por message → N POSTs spurious al backend."""
    assert re.search(
        r"_pushSubChangeListenerRegistered",
        push_src,
    ), (
        "P3-AUDIT-4 regresión: guard de idempotencia "
        "`_pushSubChangeListenerRegistered` perdido. Sin él, registrar 2x "
        "el listener (e.g. StrictMode double-render) duplica POSTs."
    )


# ---------------------------------------------------------------------------
# 4. main.jsx invoca el registro durante bootstrap
# ---------------------------------------------------------------------------
def test_main_invokes_listener_register(main_src: str):
    assert re.search(
        r"registerPushSubscriptionChangeListener\s*\(\s*\)",
        main_src,
    ), (
        "P3-AUDIT-4 regresión: `registerPushSubscriptionChangeListener()` "
        "no se invoca en main.jsx. Sin esto, el SW emite postMessage pero "
        "el cliente no tiene listener registrado → backend nunca se entera "
        "de la nueva subscription."
    )


def test_main_imports_listener_register(main_src: str):
    assert re.search(
        r"import\s*\{[^}]*\bregisterPushSubscriptionChangeListener\b[^}]*\}\s*from",
        main_src,
    ), (
        "P3-AUDIT-4 regresión: import de `registerPushSubscriptionChangeListener` "
        "perdido en main.jsx."
    )


# ---------------------------------------------------------------------------
# 5. Anchor textual P3-AUDIT-4 presente
# ---------------------------------------------------------------------------
def test_anchor_present_sw(sw_src: str):
    assert "P3-AUDIT-4" in sw_src, (
        "P3-AUDIT-4 regresión: anchor textual perdido en custom-sw.js."
    )


def test_anchor_present_push(push_src: str):
    assert "P3-AUDIT-4" in push_src, (
        "P3-AUDIT-4 regresión: anchor textual perdido en pushNotifications.js."
    )


def test_anchor_present_main(main_src: str):
    assert "P3-AUDIT-4" in main_src, (
        "P3-AUDIT-4 regresión: anchor textual perdido en main.jsx."
    )
