"""[P1-DISPLAY-ECO-CONTENIDO + 2 · 2026-08-23] Lo que sólo se ve ejecutando contra datos reales.

CÓMO APARECIERON

Cerrando `P1-I18N-SIN-EVIDENCIA-PRODUCCION` — la puerta final del plan de i18n, la que
decía que ninguna cantidad de tests verdes contesta «¿esto funciona en producción?». Con
autorización explícita del dueño se ejecutó `enrich_plan_display` contra un plan REAL,
un idioma, un día, con respaldo previo y restauración verificada por hash.

La ejecución devolvió `{'enriched_meals': 4}` en 14,2 s. Y las cuatro comidas quedaron **en
español** dentro de `_display['fr-FR']` — nombres idénticos al original e ingredientes como
`¼ taza de avena (Avena)`, donde debía leerse francés. Cuatro éxitos declarados, cero
traducciones.

Ninguno de los tres defectos de abajo era visible antes: los tests usan dobles que devuelven
texto traducido POR CONSTRUCCIÓN, así que la pregunta «¿y si el modelo devuelve el original?»
no se le hacía a nadie.

LOS TRES

1. **El contenido de cada comida no tenía defensa contra ecos.** `P2-DISPLAY-ECO-NOMBRE`
   cubría el nombre del PLAN; `_validate_and_build_display` comprobaba tipos y longitudes y
   nada más. Un lote devuelto sin traducir pasaba entero, se persistía como si fuera la
   traducción, y a partir de ahí el gate de «ya traducido» decía SÍ: español para siempre.

   Se juzga por la DESCRIPCIÓN, no por el nombre. Un nombre puede coincidir legítimamente
   entre idiomas —una marca, un sustantivo propio como «Mangú»— pero una frase entera que
   sobrevive intacta a un cambio de idioma no es una traducción.

2. **El gate de «ya traducido» no sabía reconocer un eco.** Miraba que la clave existiera y
   no estuviera vacía. Medido en el único plan de producción con `_display`:

       en-US -> "Strong Flavor, Life in Balance"      (traducido)
       pt-BR -> "Sabor Forte, Vida em Equilíbrio"     (traducido)
       fr-FR -> "Sazón Fuerte, Vida en Equilibrio"    ← el ESPAÑOL, tal cual

   Ese `fr-FR` llevaba ahí desde antes del fix y **nadie lo iba a reintentar jamás**. Un eco
   persistido es peor que una ausencia: la ausencia se reintenta sola.

3. **La poda de idiomas descarta trabajo pagado — y es a propósito.** El plan real entró con
   tres locales y salió con dos: el `en-US`, con sus insights, desapareció sin aviso.

   Llegué a subir el tope de 2 a 4. **Estaba mal**, y es la segunda vez el mismo día que
   reviertía una decisión sin leer su razonamiento: `P2-DISPLAY-RETENCION-LOCALES` pesó
   explícitamente las dos presiones —re-pagar la traducción entera al alternar idiomas
   contra un jsonb que va «de cientos de KB a MB», ~60 KB por idioma en un plan de 30
   días— y dejó escrito que «2 es donde se cruzan». La pérdida es el coste conocido de esa
   decisión, no un descuido.

   Lo que sí faltaba y queda: que el umbral sea un KNOB. Era una constante, así que mover el
   equilibrio exigía redeploy — y es justo el tipo de número que se querría ajustar mirando
   datos reales.

tooltip-anchor: P1-DISPLAY-ECO-CONTENIDO
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_MARKER = "P1-DISPLAY-ECO-CONTENIDO"

_ORIGINAL = {
    "name": "Avena Tropical Cremosa con Pera y Huevo",
    "description": "Avena cremosa con pera y huevo, un desayuno tropical y saciante.",
    "recipe": ["Hervir la avena.", "Añadir la pera."],
    "ingredients": ["¼ taza de avena", "1 pera mediana"],
}


def _item(**cambios):
    base = {
        "i": 0,
        "name": _ORIGINAL["name"],
        "description": _ORIGINAL["description"],
        "recipe": list(_ORIGINAL["recipe"]),
        "ingredients": list(_ORIGINAL["ingredients"]),
    }
    base.update(cambios)
    return base


# ───────────────── 1. un lote sin traducir NO se persiste ─────────────────

def test_un_lote_devuelto_en_espanol_se_descarta():
    import plan_display_i18n as m
    assert m._validate_and_build_display(_ORIGINAL, _item()) is None, (
        "un lote devuelto SIN traducir se persistió como si fuera la traducción. A partir "
        "de ahí el gate de «ya traducido» dice SÍ y el plan se queda en español para "
        f"siempre. Medido en producción: 4 «éxitos», 0 traducciones [{_MARKER}]"
    )


def test_una_traduccion_de_verdad_sigue_pasando():
    """La mitad de control. Sin ella, apagar el validador entero pasaría el test de arriba."""
    import plan_display_i18n as m
    real = _item(
        name="Avoine tropicale crémeuse à la poire et à l'œuf",
        description="Avoine crémeuse à la poire et à l'œuf, un petit-déjeuner tropical.",
    )
    r = m._validate_and_build_display(_ORIGINAL, real)
    assert r is not None and r.get("name", "").startswith("Avoine"), (
        f"una traducción legítima dejó de aceptarse: el guard se pasó de frenada [{_MARKER}]"
    )


def test_el_nombre_repetido_por_si_solo_no_descarta():
    """Un nombre puede coincidir legítimamente entre idiomas (una marca, «Mangú»). Juzgar
    por el nombre convertiría esos platos en no-traducibles."""
    import plan_display_i18n as m
    solo_nombre_igual = _item(
        description="Avoine crémeuse à la poire et à l'œuf, un petit-déjeuner tropical."
    )
    assert m._validate_and_build_display(_ORIGINAL, solo_nombre_igual) is not None, (
        "un plato cuyo NOMBRE no cambia entre idiomas se está descartando entero, con su "
        f"descripción y su receta traducidas [{_MARKER}]"
    )


# ───────────────── 2. el gate distingue eco de traducción ─────────────────

def test_el_gate_reintenta_un_eco_persistido():
    import plan_display_i18n as m
    pd = {"_display": {
        "fr-FR": {"name": "Sazón Fuerte, Vida en Equilibrio"},
        "pt-BR": {"name": "Sabor Forte, Vida em Equilíbrio"},
    }}
    orig = "Sazón Fuerte, Vida en Equilibrio"
    assert m._plan_name_already_translated(pd, "fr-FR", original=orig) is False, (
        "el gate vuelve a dar por traducido un eco. Ese caso está VIVO en producción y "
        f"significa que nadie lo reintenta nunca [{_MARKER}]"
    )
    assert m._plan_name_already_translated(pd, "pt-BR", original=orig) is True, (
        "el gate dejó de reconocer una traducción real: se retraduciría en cada disparador, "
        "que es justo el gasto que existe para evitar"
    )


def test_sin_el_original_el_gate_degrada_a_la_conducta_previa():
    """`original` es opcional para no romper llamadores viejos — pero sin él no se puede
    distinguir un eco, y eso tiene que ser explícito, no un silencio."""
    import plan_display_i18n as m
    pd = {"_display": {"fr-FR": {"name": "Sazón Fuerte, Vida en Equilibrio"}}}
    assert m._plan_name_already_translated(pd, "fr-FR") is True


# ───────────────── 3. la poda no tira trabajo pagado ─────────────────

def test_el_tope_de_idiomas_sigue_siendo_dos_por_decision():
    """El tope NO se subió, y esa es la corrección de este P-fix.

    Ver la ejecución real destapó que un plan con tres idiomas sale con dos y pierde el
    primero — trabajo ya pagado al proveedor. La observación es correcta; la conclusión de
    subir el tope, no: `P2-DISPLAY-RETENCION-LOCALES` pesó explícitamente las dos presiones
    (re-pagar traducciones contra un jsonb que va «de cientos de KB a MB», ~60 KB por idioma
    en un plan de 30 días) y dejó escrito que «2 es donde se cruzan».

    O sea que la pérdida es el COSTE CONOCIDO de una decisión tomada, no un descuido — y
    revertirla no le toca a un arreglo de traducciones. Lo que sí faltaba, y queda, es que el
    umbral sea movible sin redeploy.
    """
    import plan_display_i18n as m
    assert m._max_locales_display() == 2, (
        "el tope de idiomas dejó de ser 2. Si es deliberado, actualiza también "
        f"`P2-DISPLAY-RETENCION-LOCALES`, que es donde vive el razonamiento [{_MARKER}]"
    )


def test_la_poda_sigue_acotando_por_encima_del_maximo():
    """El tope deja de descartar en uso normal, pero sigue siendo una red: si aparecen
    claves inesperadas, no crecen sin límite."""
    import plan_display_i18n as m
    cinco = {k: {"name": k} for k in ("en-US", "fr-FR", "pt-BR", "it-IT", "xx-XX")}
    quedan = m._podar_locales(cinco, "fr-FR")
    assert len(quedan) == m._max_locales_display()
    assert "fr-FR" in quedan, "la poda descartó el idioma ACTIVO, que es el único obligatorio"


def test_el_tope_es_un_knob():
    """Bajarlo sin redeploy es la vía de escape si algún plan se vuelve pesado — convención
    del repo para exactamente este caso."""
    src = io.open(_BACKEND / "plan_display_i18n.py", encoding="utf-8").read()
    assert 'MEALFIT_PLAN_DISPLAY_I18N_MAX_LOCALES' in src, (
        f"el tope de idiomas volvió a estar clavado en el código [{_MARKER}]"
    )
