"""[P2-COUNTRY-HOUSEKEEPING · 2026-08-21] Cuatro defectos pequeños del sistema de países que
comparten una propiedad: cada uno se cierra en menos de diez líneas y ninguno se cerró en su día
porque no dolía TODAVÍA. Van juntos porque separarlos habría costado más ceremonia que código.

1. PASILLOS DUPLICADOS EN LA LISTA. `_master_category_for_unpriced_item` devuelve la categoría
   CRUDA del master ('Vegetales') mientras la rama con precio devuelve el label del mapa de
   display ('VEGETALES'). El Dashboard agrupa por la cadena literal, así que el usuario ve una
   sección «VEGETALES» con doce ítems y, más abajo, otra «Vegetales» con sólo Acelgas. Igual con
   «PROTEÍNAS»/«Proteínas» (Almejas sola) y «FRUTAS»/«Frutas» (Membrillo solo). Es UNA línea, y es
   el arreglo con mejor relación coste/visibilidad de toda la auditoría.

2. EL TERCER SETTER DEL PAÍS. `tools.update_form_field` escribe CUALQUIER clave en
   `health_profile` sin whitelist, y el system prompt del coach le ordena llamarla «OBLIGATORIO y
   SIN EXCEPCIÓN cada vez que el usuario mencione un nuevo dato sobre sí mismo». Un «me mudé a
   España» escribe `country='España'` — que `canonicalize_country` convierte en 'DO', porque no es
   un código ISO. El usuario cree que lo cambió y el sistema lo devuelve a dominicano en silencio.
   Es la misma familia de «dos setters del mismo dato sin jerarquía» que costó
   P1-COUNTRY-RENEWAL-PROFILE-WINS, con un tercero que nadie había contado.

3. EL FAIL-SAFE MUDO. `canonicalize_country` cae a 'DO' para cualquier entrada no reconocida sin
   log, sin métrica y sin alerta. Es el comportamiento correcto para «ausente» —los 7 perfiles
   legacy no tienen país y no pasa nada— pero para un string NO VACÍO que no canoniza es una
   CORRUPCIÓN silenciosa: un usuario podría pasar semanas recibiendo planes dominicanos y el
   operador no tendría forma de enterarse. Se distinguen los dos casos.

4. EL COLD-START. `get_similar_user_patterns` mete platos de OTROS usuarios en el prompt del
   generador con etiqueta «PRIORIDAD 4», y su condición de disparo (menos de 3 registros de
   comida) es EXACTAMENTE la de todo usuario beta nuevo. El pool real hoy es 100 % dominicano.
   El primer plan de un español —el que decide si se queda— recibe sugerencias explícitas de
   generar los platos de los dominicanos.

   El gate existe (`MEALFIT_COUNTRY_COLDSTART_SEGMENT`) y está apagado, con una razón escrita que
   es CIERTA: con un solo usuario de un país nuevo, su pool queda vacío. Pero eso no implica que
   la conducta correcta sea servirle lo dominicano — implica que la correcta es **no sugerir
   nada**. Se cambia la SEMÁNTICA, no el default: país ≠ DO ⇒ filtrar siempre y devolver `[]` si
   el pool sale vacío. DO y país ausente conservan la conducta byte a byte.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent


# ── 1. Pasillos duplicados ──────────────────────────────────────────────────────────────────────

def test_el_pasillo_de_un_item_sin_precio_usa_el_mismo_label_que_los_demas(monkeypatch):
    """RED pre-fix: 'Vegetales' (crudo del master) frente a 'VEGETALES' (label de display) — dos
    secciones en la lista del usuario para el mismo pasillo del súper.

    Con catálogo MOCKEADO a propósito: la primera versión de este test leía el catálogo vivo y se
    SALTABA cuando no había DB — o sea que no verificaba nada justo en el entorno donde corre el
    gate. Un test que se salta en CI es un test que no existe."""
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "get_master_ingredients",
                        lambda *a, **k: [{"name": "Acelgas", "category": "Vegetales"}])
    crudo = sc._master_category_for_unpriced_item("Acelgas")
    assert crudo is not None, "el helper dejó de resolver un nombre que SÍ está en el catálogo"
    con_precio = sc._get_display_category("Vegetales", "Espinacas")
    assert crudo == con_precio, (
        f"la rama sin precio devuelve {crudo!r} y la rama con precio {con_precio!r}: el Dashboard "
        f"agrupa por la cadena literal, así que son DOS pasillos"
    )


def test_el_fallback_del_pasillo_sigue_siendo_el_label_historico(monkeypatch):
    """Si el master no resuelve el nombre, el caller conserva 'CATÁLOGO SIN PRECIO'. El fix no
    puede convertir un `None` en una categoría inventada."""
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "get_master_ingredients",
                        lambda *a, **k: [{"name": "Acelgas", "category": "Vegetales"}])
    assert sc._master_category_for_unpriced_item("Zzqx que no existe") is None


# ── 2. El tercer setter del país ────────────────────────────────────────────────────────────────

def test_la_tool_del_chat_no_escribe_cualquier_campo_del_perfil():
    """RED pre-fix: `_hp[field] = _new_field_value` sin whitelist. El prompt del coach ordena
    llamarla ante cualquier dato personal nuevo, así que la LLM decide el nombre del campo."""
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8", errors="replace")
    assert "P2-COUNTRY-HOUSEKEEPING" in src
    i = src.find("def update_form_field")
    assert i > 0
    _fin = src.find("\ndef ", i + 1)
    cuerpo = src[i:_fin if _fin > 0 else len(src)]
    # La whitelist vive FUERA de la función (es una constante de módulo, y tiene que estar antes
    # del decorador `@tool` o el fichero no compila — lo aprendí rompiéndolo). Lo que este guard
    # ancla es que el CUERPO consulte la puerta, que es la propiedad que importa: una constante
    # declarada y no consultada sería justo la «feature inerte» que este repo ya pagó dos veces.
    assert "_valor_de_campo_para_perfil(field, new_value)" in cuerpo, (
        "la tool sigue aceptando cualquier `field`: la LLM elige qué clave del perfil escribir"
    )
    assert "_CAMPOS_EDITABLES_POR_CHAT" in src, "desapareció la whitelist"


def test_el_pais_por_chat_se_canonicaliza_o_se_rechaza():
    """«me mudé a España» → `country='España'`, que no es ISO-3166 y cae a 'DO'. El usuario cree
    que lo cambió y el sistema lo devuelve a dominicano en silencio: peor que rechazarlo."""
    import tools
    assert hasattr(tools, "_valor_de_campo_para_perfil")
    ok, valor = tools._valor_de_campo_para_perfil("country", "España")
    assert ok and valor == "ES", f"esperaba canonicalizar a 'ES', salió {valor!r}"
    ok2, _ = tools._valor_de_campo_para_perfil("country", "Marte")
    assert not ok2, "un país que no existe debe RECHAZARSE, no caer a 'DO' en silencio"


def test_un_campo_fuera_de_la_whitelist_se_rechaza():
    import tools
    ok, _ = tools._valor_de_campo_para_perfil("subscription_tier", "ultra")
    assert not ok, "la tool no puede escribir el tier del usuario"


# ── 3. El fail-safe mudo ────────────────────────────────────────────────────────────────────────

def test_un_pais_corrupto_deja_rastro(caplog):
    """Ausente es fail-safe legítimo y silencioso; un string no vacío que no canoniza es una
    CORRUPCIÓN y merece un log. Sin él, un usuario puede pasar semanas recibiendo planes
    dominicanos sin que nadie se entere."""
    import logging
    from constants import canonicalize_country
    with caplog.at_level(logging.WARNING):
        assert canonicalize_country("Marte") == "DO"
    assert "Marte" in caplog.text or "P2-COUNTRY-HOUSEKEEPING" in caplog.text


@pytest.mark.parametrize("ausente", [None, "", "   "])
def test_un_pais_ausente_sigue_siendo_silencioso(caplog, ausente):
    """Los 7 perfiles legacy no tienen país: gritar por ellos convertiría el log en ruido y el
    guard se apagaría en una semana."""
    import logging
    from constants import canonicalize_country
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert canonicalize_country(ausente) == "DO"
    assert "P2-COUNTRY-HOUSEKEEPING" not in caplog.text


def test_un_pais_valido_no_deja_rastro(caplog):
    import logging
    from constants import canonicalize_country
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert canonicalize_country("es") == "ES"
    assert "P2-COUNTRY-HOUSEKEEPING" not in caplog.text


# ── 4. El cold-start ────────────────────────────────────────────────────────────────────────────

def test_el_coldstart_de_un_pais_beta_filtra_siempre_y_no_cae_a_lo_dominicano():
    """El gate existía y estaba apagado con una razón cierta —con un usuario, el pool queda
    vacío— pero de ahí no se sigue que haya que servirle lo dominicano: se sigue que hay que no
    sugerir nada. Se cambia la SEMÁNTICA, no el default."""
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8", errors="replace")
    assert "P2-COUNTRY-HOUSEKEEPING" in src
    i = src.find("def get_similar_user_patterns")
    assert i > 0
    _fin = src.find("\ndef ", i + 1)
    cuerpo = src[i:_fin if _fin > 0 else len(src)]
    assert "_coldstart_country_filter" in cuerpo, (
        "el cold-start sigue decidiendo el filtro por país sólo con el knob apagado por defecto"
    )


def test_el_coldstart_dominicano_conserva_su_conducta():
    """Control de byte-identidad: para DO y para país ausente el filtro no aplica, igual que
    antes — el problema (b) que el comentario del knob describe (el dominicano legacy sin campo)
    sólo existe en esa rama."""
    import cron_tasks
    assert cron_tasks._coldstart_country_filter({"country": "DO"}) is None
    assert cron_tasks._coldstart_country_filter({}) is None
    assert cron_tasks._coldstart_country_filter(None) is None


def test_el_coldstart_beta_devuelve_su_pais(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    import cron_tasks
    assert cron_tasks._coldstart_country_filter({"country": "ES"}) == "ES"


def test_el_coldstart_beta_no_aplica_con_el_knob_apagado(monkeypatch):
    """Rollback: con el sistema de países apagado, todo vuelve a la conducta previa."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    import cron_tasks
    assert cron_tasks._coldstart_country_filter({"country": "ES"}) is None
