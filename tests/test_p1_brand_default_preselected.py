"""[P1-BRAND-DEFAULT-PRESELECTED · 2026-07-06] El picker pre-selecciona la marca
que la LISTA está usando.

Pedido del owner: "en la lista está Wala por defecto en arroz blanco, así que
debe verse seleccionado Wala en el menú de marcas; la yuca también — así los
usuarios no se confunden". Diseño: el default se muestra con check verde HUECO
(punteado) — distinto de la preferencia manual (check sólido) — y tocarlo lo FIJA
como preferencia permanente. Fallback: ítem con UNA sola variante en el catálogo
(yuca/laurel) se muestra como default aunque el costeo fuera a granel.

Plumbing: `_pkg_from_product_row` lleva `id` del producto → `_select_market_package`
lo arrastra al envase elegido → `apply_smart_market_units` lo expone como
`brand_product_id` del ítem → el picker matchea contra `variant.id` del /match.
"""
import re
from pathlib import Path

import pytest

import shopping_calculator as sc

BRANDS_JSX = None


def _jsx():
    global BRANDS_JSX
    if BRANDS_JSX is None:
        from pathlib import Path
        BRANDS_JSX = (Path(sc.__file__).resolve().parent.parent / "frontend" / "src"
                      / "components" / "dashboard" / "SupermarketBrands.jsx").read_text(encoding="utf-8")
    return BRANDS_JSX


_MASTER = [
    {"name": "Arroz blanco", "category": "Despensa", "market_container": "paquete",
     "container_weight_g": 907.0, "price_per_lb": 40.0, "default_unit": "paquete",
     "shelf_life_days": 365, "aliases": []},
]


@pytest.fixture()
def master_stub(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: list(_MASTER))
    sc.invalidate_master_cache()
    yield
    sc.invalidate_master_cache()


# ───────────── backend: el id fluye hasta el ítem ─────────────

def test_pkg_row_carries_product_id():
    pkg = sc._pkg_from_product_row({
        "id": "abc-123", "presentation": "Funda Selecto 1 Lb", "brand": "Wala",
        "price_rd": 42.0, "size_grams": None,
    })
    assert pkg is not None and pkg["id"] == "abc-123"


def test_select_market_package_carries_id():
    sel = sc._select_market_package(900.0, [
        {"grams": 453.6, "price": 42.0, "label": "1 Lb · Wala", "unit": "funda", "id": "wala-1"},
        {"grams": 907.2, "price": 145.0, "label": "2 Lb · Cariño", "unit": "paquete", "id": "car-2"},
    ])
    assert sel is not None and sel.get("id") in ("wala-1", "car-2"), "el envase elegido conserva su id"


def test_item_exposes_brand_product_id(master_stub):
    defaults = {"arroz blanco": [
        {"grams": 907.184, "price": 145.0, "label": "2 Lb · Cariño", "unit": "paquete", "id": "car-2lb"},
    ]}
    result = sc.aggregate_and_deduct_shopping_list(
        ["800g de arroz blanco"], [], structured=True, brand_defaults=defaults,
    )
    item = next((r for r in result if isinstance(r, dict) and "arroz" in str(r.get("name", "")).lower()), None)
    assert item is not None
    assert item.get("brand_product_id") == "car-2lb", (
        f"el ítem debe decir QUÉ producto usa el costeo: {item.get('brand_product_id')}"
    )


def test_fetchers_select_product_id():
    from pathlib import Path
    src = (Path(sc.__file__).resolve().parent / "shopping_calculator.py").read_text(encoding="utf-8")
    i = src.index("def fetch_brand_pref_packages(")
    assert "sp.id::text AS id" in src[i:i + 2500], "prefs fetch debe traer el id del producto"
    j = src.index("def fetch_brand_default_packages(")
    assert "sp.id::text AS id" in src[j:j + 2500], "defaults fetch debe traer el id del producto"


# ───────────── frontend: pre-selección visual ─────────────

def test_picker_preselects_list_brand():
    jsx = _jsx()
    assert "P1-BRAND-DEFAULT-PRESELECTED" in jsx
    assert "defaultIdByKey" in jsx and "brand_product_id" in jsx
    assert "isDefault" in jsx, "la fila default se marca visualmente (check hueco)"
    assert "dashed" in jsx, "estilo DISTINTO de la preferencia manual (no confundir elección vs default)"


def test_single_variant_fallback():
    jsx = _jsx()
    assert "all.length === 1" in jsx, (
        "ítem con UNA variante (yuca/laurel) se muestra como default aunque el costeo sea a granel"
    )


def test_tapping_default_pins_it():
    jsx = _jsx()
    assert "tócala para fijarla como tu preferida" in jsx, (
        "tocar el default lo convierte en preferencia permanente (persistPref con su id)"
    )


# ── [P2-BRANDS-CHIP-CASCADE · 2026-08-15] El rótulo no se arma a trozos ──────

def _brands_src() -> str:
    p = (
        Path(__file__).resolve().parents[2]
        / "frontend" / "src" / "components" / "dashboard" / "SupermarketBrands.jsx"
    )
    assert p.exists(), f"P2-BRANDS-CHIP-CASCADE: no existe {p}"
    return p.read_text(encoding="utf-8")


def test_prefs_arrancan_del_cache_local_no_de_un_objeto_vacio():
    """El rótulo del chip se armaba en TRES pasos visibles al refrescar.

    Primero «Marcas del súper» a secas, luego «· N/M con opciones» al aterrizar
    /supermarket/match, y por último «· N elegidas» al aterrizar
    /supermarket/preferences: dos viajes de red independientes, cada uno
    reescribiendo el texto.

    El caché local (`readLocalPrefs`) ya contenía la respuesta —lo escribe este
    mismo componente en cada elección— pero solo se leía en la rama de FALLO del
    fetch. O sea: en el camino feliz el dato estaba disponible y sin usar
    mientras la UI esperaba a la red para decir lo mismo.

    Volver a `useState({})` reintroduce el tercer paso, y es un cambio que parece
    inocuo («inicializar vacío es lo normal»).
    """
    src = _brands_src()
    m = re.search(r"const \[prefs, setPrefs\] = useState\(([^)]*)\)", src)
    assert m, "P2-BRANDS-CHIP-CASCADE: no encuentro el useState de `prefs`."
    inicial = m.group(1).strip()
    assert inicial == "readLocalPrefs", (
        f"P2-BRANDS-CHIP-CASCADE: `prefs` arranca en {inicial!r}. Debe ser "
        "`readLocalPrefs` (la FUNCIÓN, sin paréntesis: inicialización perezosa, "
        "para no leer localStorage en cada render). Con `{}` el sufijo «· N "
        "elegidas» espera un segundo viaje de red y el rótulo se arma en tres "
        "pasos visibles."
    )


def test_el_servidor_sigue_mandando_sobre_el_cache_local():
    """El caché local es un ARRANQUE optimista, no la verdad.

    Si el fetch de preferencias dejara de sobrescribir `prefs`, la elección
    hecha en otro dispositivo nunca llegaría — y el bug sería mucho peor que el
    parpadeo que este cambio arregla.
    """
    src = _brands_src()
    # Se ancla a la LLAMADA (`fetchWithAuth('…'`), no a la ruta suelta: la ruta
    # aparece antes en un comentario que explica de dónde salen las preferencias,
    # y un `find` de la cadena pelada aterrizaba en esa prosa.
    #
    # Y SIN el paréntesis de cierre, aprendido a las horas de nacer: la primera
    # versión exigía `...preferences')` — la llamada exacta SIN segundo
    # argumento — y P2-BRANDS-LOAD-TIMEOUT la puso en rojo al añadirle
    # `{ signal: AbortSignal.timeout(...) }`. La llamada seguía siendo la misma;
    # lo que cambió fue su aridad. Anclar la grafía completa de una llamada es
    # prohibirle argumentos futuros.
    i_fetch = src.find("fetchWithAuth('/api/supermarket/preferences'")
    assert i_fetch != -1, (
        "P2-BRANDS-CHIP-CASCADE: desapareció la llamada "
        "`fetchWithAuth('/api/supermarket/preferences'...)`. Si cambió de forma, "
        "actualiza este ancla."
    )
    ventana = src[i_fetch:i_fetch + 900]
    assert "setPrefs(flat)" in ventana and "setPrefsSource('server')" in ventana, (
        "P2-BRANDS-CHIP-CASCADE: la respuesta del servidor ya no sobrescribe "
        "`prefs`. El caché local solo debe cubrir la ventana entre el montaje y "
        "esa respuesta; si deja de corregirse, una elección hecha en otro "
        "dispositivo no aparece nunca."
    )


# ── [P0-BRANDS-RETRY-STORM · 2026-08-15] La tormenta de reintentos ──────────

def test_el_guard_de_load_no_depende_del_estado_que_el_propio_efecto_observa():
    """16.105 respuestas 429 contra 12 doscientos, a ~12 peticiones por SEGUNDO.

    El bucle se cerraba así:
      1. `load` tenía deps `[matches, loading, names, t]` — `loading`, la variable
         del guard, decidía la IDENTIDAD de la función.
      2. El efecto de prefetch observaba `[names, load]` ⇒ cada cambio de
         identidad lo re-disparaba.
      3. El `catch` hacía setError + setLoading(false) sin tocar `matches`, así
         que al reentrar el guard PASABA y lanzaba otro fetch.

    Como este es el único llamante de /match con `fetch` pelado sin auth, su cupo
    es por IP; el bucle lo quemaba en ~2 s y el limitador (que entonces sellaba la
    ventana también con los rechazos) nunca drenaba. De ahí el 429 permanente.

    La regla general, que es lo que este guard protege: **la variable que decide
    si un efecto debe correr no puede estar en las deps de la función que ese
    efecto invoca.** Es una realimentación, y se manifiesta como inundación.
    """
    src = _brands_src()
    m = re.search(r"const load = useCallback\(.*?\n    \}, \[([^\]]*)\]\)", src, re.DOTALL)
    assert m, "P0-BRANDS-RETRY-STORM: no encuentro las deps del useCallback `load`."
    deps = [d.strip() for d in m.group(1).split(",") if d.strip()]
    for prohibida in ("loading", "matches"):
        assert prohibida not in deps, (
            f"P0-BRANDS-RETRY-STORM: `{prohibida}` volvió a las deps de `load`. "
            "El efecto de prefetch invoca `load`; si una variable del guard rota "
            "su identidad, el efecto se re-dispara solo y el panel inunda el "
            "endpoint (medido: 16.105 × 429). El guard vive en refs "
            "(`inFlightRef`/`loadedRef`) justamente para que esto no pueda pasar."
        )


def test_el_efecto_de_prefetch_no_observa_la_identidad_de_load():
    src = _brands_src()
    m = re.search(
        r"useEffect\(\(\) => \{\s*if \(names\.length > 0\) load\(\);.*?\}, \[([^\]]*)\]\)",
        src,
        re.DOTALL,
    )
    assert m, "P0-BRANDS-RETRY-STORM: no encuentro el efecto de prefetch."
    deps = [d.strip() for d in m.group(1).split(",") if d.strip()]
    assert "load" not in deps, (
        "P0-BRANDS-RETRY-STORM: el efecto volvió a observar `load`. Esa era la "
        "otra mitad del bucle: el efecto se re-dispara a sí mismo a través de la "
        "identidad de la función que llama."
    )


def test_hay_tope_de_intentos_automaticos_y_el_error_no_se_borra_solo():
    """Dos defensas independientes, y la segunda explica por qué NADIE vio el error.

    `setError(null)` al reentrar borraba el mensaje ~1 frame después de ponerlo:
    el usuario veía «Buscando marcas…» eterno y jamás el botón «Reintentar» que
    habría cortado el bucle. Un error que se limpia solo es un error invisible.
    """
    src = _brands_src()
    assert "autoAttemptRef" in src, (
        "P0-BRANDS-RETRY-STORM: desapareció el tope de intentos automáticos. Sin "
        "él, cualquier fallo persistente vuelve a ser una inundación."
    )
    assert re.search(r"if \(manual\) setError\(null\)", src), (
        "P0-BRANDS-RETRY-STORM: `setError(null)` volvió a correr en la reentrada "
        "automática. Eso hace INVISIBLE el fallo: el usuario ve un «Buscando…» "
        "eterno en vez del error con su «Reintentar»."
    )


def test_el_limitador_no_sella_la_ventana_con_peticiones_rechazadas():
    """[P1-RATELIMIT-NO-SELF-POISON] Un rechazo no puede alargar el castigo.

    En la rama Redis el `zadd` corría DENTRO del pipeline, antes de comprobar el
    cupo: cada 429 metía un timestamp nuevo en la ventana deslizante, así que un
    cliente en bucle nunca drenaba. El límite dejaba de ser «30 por minuto» y
    pasaba a ser «bloqueado mientras sigas intentando».

    La rama en memoria NUNCA tuvo el fallo (lanza antes del `append`). Esto cierra
    la asimetría: el comportamiento del limitador no debe depender de si Redis
    está configurado.
    """
    rl = (Path(__file__).resolve().parents[1] / "rate_limiter.py").read_text(encoding="utf-8")
    pipe_block = re.search(r"pipe = redis_client\.pipeline\(\)(.*?)results = pipe\.execute\(\)", rl, re.DOTALL)
    assert pipe_block, "P1-RATELIMIT-NO-SELF-POISON: no encuentro el pipeline de Redis."
    assert "zadd" not in pipe_block.group(1), (
        "P1-RATELIMIT-NO-SELF-POISON: el `zadd` volvió al pipeline, o sea ANTES "
        "de comprobar el cupo. Con eso, una petición rechazada vuelve a sellar la "
        "ventana y un cliente en bucle se auto-prorroga el 429 indefinidamente "
        "(medido: 16.105 seguidos). Debe ir DESPUÉS del `raise`."
    )
    i_raise = rl.find("status_code=429")
    i_zadd = rl.find("redis_client.zadd")
    assert i_zadd != -1 and i_zadd > i_raise, (
        "P1-RATELIMIT-NO-SELF-POISON: el `zadd` de aceptación debe vivir DESPUÉS "
        "del `raise` del 429 — solo se cuenta lo que se admite."
    )


# ── [P2-BRANDS-MATCH-CACHE · 2026-08-15] El chip nace completo ───────────────

def test_el_cache_de_matches_esta_claveado_por_la_lista_y_caduca():
    """Sin firma, el caché mentiría al cambiar la lista de compras.

    Las coincidencias de marca dependen de QUÉ ítems tiene la lista. Servir el
    caché de otra lista mostraría marcas de alimentos que ya no compras. La firma
    es el conjunto de nombres normalizado y ORDENADO — el orden no cambia qué
    marcas existen, así que ordenar evita fallos de caché gratuitos.

    Y caduca porque la respuesta lleva PRECIOS: un precio rancio es peor que un
    rótulo que tarda 200 ms.
    """
    src = _brands_src()
    assert "matchSignature" in src and "sort()" in src, (
        "P2-BRANDS-MATCH-CACHE: el caché debe estar claveado por una firma "
        "ORDENADA de los nombres de la lista."
    )
    m = re.search(r"MATCH_CACHE_TTL_MS\s*=\s*([^;]+);", src)
    assert m, "P2-BRANDS-MATCH-CACHE: el caché de /match perdió su TTL."
    assert re.search(r"parsed\.signature !== signature", src), (
        "P2-BRANDS-MATCH-CACHE: la lectura del caché ya no compara la firma. "
        "Sin esa comparación se sirven marcas de una lista de compras distinta."
    )


def test_el_cache_revalida_contra_la_red():
    """Es stale-while-revalidate, no stale-y-punto.

    Si la rama de caché marcara `loadedRef`, el fetch no correría y los precios
    se quedarían congelados hasta el próximo cambio de lista o los 15 min de TTL.
    El caché existe para cubrir el primer pintado, no para sustituir a la red.
    """
    src = _brands_src()
    m = re.search(r"if \(cached\) \{(.*?)\} else \{", src, re.DOTALL)
    assert m, "P2-BRANDS-MATCH-CACHE: no encuentro la rama de caché servida."
    rama = m.group(1)
    assert "loadedRef.current = true" not in rama, (
        "P2-BRANDS-MATCH-CACHE: la rama de caché marca `loadedRef`, así que corta "
        "la revalidación. El caché debe PINTAR y dejar que la red confirme."
    )
    assert "setLoading(true)" not in rama, (
        "P2-BRANDS-MATCH-CACHE: la rama de caché enciende `loading`, tapando con "
        "«Buscando…» un contenido que ya se puede leer."
    )


def test_el_reconcile_nunca_corre_contra_datos_de_cache():
    """LA defensa importante: el reconcile dispara un RECÁLCULO, no un repintado.

    `onPrefApplied()` recalcula la lista de compras en el backend. Comparando
    contra `matches` rancios puede detectar una discrepancia que ya no existe y
    lanzar ese trabajo por nada. La bandera se baja cuando responde la red, unos
    cientos de ms después, y entonces el reconcile corre con datos que sí valen.
    """
    src = _brands_src()
    i_efecto = src.find("if (reconcileFiredRef.current) return;")
    assert i_efecto != -1, "P2-BRANDS-MATCH-CACHE: no encuentro el efecto de reconcile."
    ventana = src[i_efecto:i_efecto + 900]
    assert "matchesFromCacheRef.current) return" in ventana, (
        "P2-BRANDS-MATCH-CACHE: el reconcile perdió su guarda contra datos de "
        "caché. Sin ella, un `matches` rancio puede disparar un recálculo de la "
        "lista de compras por una discrepancia inexistente."
    )
    i_raise = src.find("matchesFromCacheRef.current = false")
    assert i_raise != -1, (
        "P2-BRANDS-MATCH-CACHE: nadie baja la bandera de «rancio». Si no se baja, "
        "el reconcile NUNCA corre y la marca elegida se queda sin aplicar."
    )
