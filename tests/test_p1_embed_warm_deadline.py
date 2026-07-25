"""[P1-EMBED-WARM-DEADLINE · 2026-07-25] El caché semántico llevaba semanas sin poder existir.

Investigando por qué un chunk de 3-4 días expira a los 600 s apareció esta línea, repetida en
cada generación:

    🟡 Caché semántico no disponible (TimeoutError); usando Regex Fast-Path.
       Reintentos pausados 600s.

No es intermitencia de Cohere: es **aritmética** (valores vivos: 204 alimentos, lotes de 3, 3 s).

    204 alimentos ÷ EMBED_INIT_BATCH_SIZE (3)   = 68 lotes
    68 lotes × EMBED_INIT_BATCH_DELAY_S (3,0 s) = 204 s de espera deliberada
    …contra EMBED_INIT_DEADLINE_S = 30 s

⚠️ Al escribir este test puse "1.911 alimentos" y el test falló: esa cifra son las claves del
índice de ALIAS, no alimentos (`master_ingredients` tiene 204 filas). El propio test atrapó que
yo estaba dimensionando el plazo con un catálogo 9× más grande del real.

El piso del trabajo (96 s) supera al techo permitido (30 s), así que la init **nunca** puede
terminar. Y como sólo se persiste a Redis tras un init exitoso, Redis nunca recibe los vectores
(verificado en prod: 100 claves, ninguna del catálogo) → cada proceso reintenta cada 10 minutos,
quema 30 s cada vez, y toda la resolución de ingredientes cae al Regex Fast-Path.

El deadline de 30 s lo introdujo P1-EMBED-INIT-DEADLINE (2026-07-08) por una razón correcta —que
Cohere lento no bloqueara una petición del usuario— pero se aplicó también al **calentador de
arranque**, que es justo el único caller que puede esperar: corre en un daemon thread y nadie
aguarda su resultado. Su propio comentario ya decía "~100s en cold init"; el deadline posterior
lo dejó sin efecto.

⚠️ Lección: un tope de tiempo global aplicado a TODOS los callers castiga al que existía
precisamente para absorber la espera.
"""
from pathlib import Path

import pytest

import shopping_calculator as sc


_BACKEND = Path(sc.__file__).resolve().parent


# ───────────── 1. la aritmética que hacía imposible el init ─────────────

CATALOGO_VIVO = 204          # filas de master_ingredients al detectar el bug (2026-07-25)
CATALOGO_HOLGURA = 400       # tripwire: si el catálogo pasa de aquí hay que subir el knob


@pytest.mark.parametrize("catalogo", [CATALOGO_VIVO, CATALOGO_HOLGURA])
def test_el_plazo_del_warmer_supera_el_piso_de_espera(catalogo):
    """El delay entre lotes es un piso que NO depende de la red: si el plazo no lo supera, la
    init aborta siempre, pase lo que pase con Cohere.

    Se comprueba con el catálogo vivo y con uno casi al doble, para que crecerlo avise aquí antes
    de romper la resolución semántica en producción."""
    lotes = -(-catalogo // sc.EMBED_INIT_BATCH_SIZE)
    piso_s = lotes * sc.EMBED_INIT_BATCH_DELAY_S
    assert sc.EMBED_WARM_DEADLINE_S > piso_s, (
        f"catálogo {catalogo}: {lotes} lotes × {sc.EMBED_INIT_BATCH_DELAY_S}s = {piso_s}s de "
        f"espera mínima, pero el warmer sólo tiene {sc.EMBED_WARM_DEADLINE_S}s. "
        f"Sube MEALFIT_EMBED_WARM_DEADLINE_S o baja EMBED_INIT_BATCH_DELAY_S.")


def test_el_deadline_de_peticion_NO_alcanza_y_por_eso_existe_este_fix():
    """Ancla del bug: con el plazo de petición la init del catálogo vivo es imposible. Si algún
    día esto deja de ser cierto (lotes más grandes, menos delay), el warmer ya no haría falta."""
    lotes = -(-CATALOGO_VIVO // sc.EMBED_INIT_BATCH_SIZE)
    assert lotes * sc.EMBED_INIT_BATCH_DELAY_S > sc.EMBED_INIT_DEADLINE_S


def test_el_plazo_de_peticion_sigue_siendo_corto():
    """No se toca: una petición del usuario debe caer al fast-path antes que bloquear."""
    assert sc.EMBED_INIT_DEADLINE_S <= 60
    assert sc.EMBED_WARM_DEADLINE_S > sc.EMBED_INIT_DEADLINE_S


def test_floor_del_knob():
    """Bajarlo a ≤30 reabre el bug; el floor lo impide aunque el env diga otra cosa."""
    assert sc.EMBED_WARM_DEADLINE_S >= 60


# ───────────── 2. el cableado ─────────────

def test_get_semantic_cache_acepta_deadline():
    import inspect
    assert "deadline_s" in inspect.signature(sc.get_semantic_cache).parameters


def test_el_default_no_cambia_el_comportamiento_de_peticion():
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    i = src.index("def get_semantic_cache(")
    cuerpo = src[i:src.index("\ndef ", i + 10)]
    assert "deadline_s if deadline_s and deadline_s > 0 else EMBED_INIT_DEADLINE_S" in cuerpo, \
        "sin deadline explícito debe conservarse EMBED_INIT_DEADLINE_S"


def test_el_warmer_de_arranque_pasa_su_plazo():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    i = src.index("_warm_semantic_cache_bg")
    cuerpo = src[i:i + 1600]
    assert "deadline_s=EMBED_WARM_DEADLINE_S" in cuerpo, \
        "el warmer debe pedir su propio plazo, no heredar el de petición"


def test_el_warmer_sigue_siendo_no_bloqueante():
    """Darle más plazo NO puede convertirlo en algo que retrase el arranque."""
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    i = src.index("_warm_semantic_cache_bg")
    cuerpo = src[i - 400:i + 2000]
    assert "daemon=True" in cuerpo
    assert "threading.Thread" in cuerpo


# ───────────── 3. sin efectos colaterales ─────────────

def test_kill_switch_sigue_mandando(monkeypatch):
    """`MEALFIT_DISABLE_SEMANTIC_CACHE` gana sobre cualquier plazo."""
    monkeypatch.setattr(sc, "_semantic_cache_disabled", lambda: True)
    assert sc.get_semantic_cache(deadline_s=999) is None


def test_cache_en_memoria_devuelve_sin_mirar_el_plazo(monkeypatch):
    monkeypatch.setattr(sc, "_semantic_cache_disabled", lambda: False)
    monkeypatch.setattr(sc, "_semantic_cache", {"master_list": [], "vectors": []}, raising=False)
    try:
        assert sc.get_semantic_cache(deadline_s=1) is not None
    finally:
        monkeypatch.setattr(sc, "_semantic_cache", None, raising=False)
