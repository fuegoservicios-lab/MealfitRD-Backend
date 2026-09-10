# -*- coding: utf-8 -*-
"""[P1-SODIO-DEL-DIA-DETERMINISTA · 2026-09-10] El techo de sodio existía y no miraba a este camino.

## De dónde sale

Tercera ronda de juicio humano: **8 de las 19 notas** pedían bajar la sal y recalcular el sodio —
«reducir la sal de 2 g a 0,5 g», «añadirla después de escurrir la yuca para contabilizarla
correctamente». Su razón no es de sabor sino de CONTABILIDAD: la sal que se va con el agua del
hervor no se come, y contarla entera sube el sodio del plato con algo que acabó en el fregadero.

## Lo que la medición encontró debajo

`SODIUM_DAY_CEILING_MG` (2.000 mg, OMS) y su autofix viven en `assemble_plan_node`. **Un día del
módulo determinista no pasa por ahí** — ni reviewer, ni capa clínica, ni autofix —, que es la misma
clase de agujero que `P0-DEGRADED-SAFETY-SCAN` cerró para el path degradado y la misma que dejó
pasar el hígado en `P1-RETINOL-PREFORMADO`. Medido sobre los **30 días reales** del dueño con su
canario ENCENDIDO:

```
                        antes        con la sal a 0,5 g     + guard del día
días sobre el techo     13 de 30     1 de 30                0 de 29
sodio máximo            3.867 mg     2.122 mg               1.933 mg
mediana                 1.814 mg     1.265 mg               1.263 mg
```

El día que sigue sin caber lo empuja el **arenque**, salado de origen: ningún cambio de dato lo
arregla, y por eso hace falta el guard además del dato.

## Las dos decisiones de diseño

1. **Preferir, no descartar.** Dentro del bucle de candidatos, uno que se pasa del techo se guarda
   como reserva en vez de tirarse: el sodio es un presupuesto del DÍA y descartar platos sanos por
   una cuenta acumulada castiga al último slot. *Un guard que descarta el conjunto en vez del
   elemento castiga la abundancia* — la lección de `P1-CATALOGO-PROTEINA-DESAYUNO`.
2. **El techo se lee, no se copia.** `_techo_sodio()` importa `SODIUM_DAY_CEILING_MG`; escribir un
   2.000 aquí sería la segunda tabla que `P1-DIET-CANON-SSOT` prohíbe.
"""
import json
import pathlib

import pytest

import deterministic_day as dd
import dish_registry as dr


# ── el techo es UNO ──────────────────────────────────────────────────────────
def test_el_techo_sale_del_SSOT_no_de_una_copia():
    from graph_orchestrator import SODIUM_DAY_CEILING_MG
    assert dd._techo_sodio() == float(SODIUM_DAY_CEILING_MG)


def test_no_hay_un_2000_hardcodeado_en_el_camino():
    """El respaldo puede existir; lo que no puede es usarse cuando el SSOT responde."""
    import inspect
    src = inspect.getsource(dd._techo_sodio)
    assert "SODIUM_DAY_CEILING_MG" in src, "el techo dejó de leerse del sitio donde vive"


# ── el lector de sodio ───────────────────────────────────────────────────────
@pytest.mark.parametrize("plantilla,esperado", [
    ({"nutrition_per_serving": {"sodium_mg": 1813.4}}, 1813.4),
    ({"nutrition_per_serving": {}}, 0.0),
    ({"nutrition_per_serving": {"sodium_mg": None}}, 0.0),
    ({}, 0.0),
    (None, 0.0),
    ({"nutrition_per_serving": {"sodium_mg": "no soy un número"}}, 0.0),
])
def test_sodio_de_no_revienta_nunca(plantilla, esperado):
    assert dd._sodio_de(plantilla) == esperado


# ── el dato: la sal declarada ────────────────────────────────────────────────
def test_ninguna_plantilla_declara_mas_de_medio_gramo_de_sal():
    """Lo que pidió el dueño, aplicado a las 198: 34 declaraban 2 g y una 3,5 g por ración."""
    p = pathlib.Path(dr.DATA_DIR) / "dish_constituents_do.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    malos = []
    for nombre, t in (d.get("templates") or {}).items():
        for c in (t.get("constituents") or []):
            if str(c.get("name", "")).strip().lower() in ("sal", "salt", "sal marina"):
                if float(c.get("grams") or 0) > 0.5:
                    malos.append((nombre, c.get("grams")))
    assert not malos, f"{len(malos)} platos con más de 0,5 g de sal: {malos[:5]}"


def test_el_registro_compilado_bajo_de_sodio():
    """El snapshot en disco tiene que reflejar el tope, o el dato quedó a medias."""
    reg = json.loads((pathlib.Path(dr.REGISTRY_DIR) / "dish_registry_do_v1.json").read_text(encoding="utf-8"))
    peores = sorted(reg["templates"],
                    key=lambda t: -(t["nutrition_per_serving"].get("sodium_mg") or 0))[:1]
    tope = peores[0]["nutrition_per_serving"]["sodium_mg"]
    assert tope < 1600, f"«{peores[0]['name']}» sigue en {tope:.0f} mg de sodio"


# ── el guard del día ─────────────────────────────────────────────────────────
def test_un_dia_por_encima_del_techo_NO_se_sirve():
    """`None` en este módulo significa «que lo haga el LLM», que SÍ pasa por el autofix de sodio."""
    import inspect
    src = inspect.getsource(dd.build_day_for_skeleton)
    assert "_sodio_dia > _techo_sodio()" in src, (
        "el juicio final de sodio salió de `build_day_for_skeleton`: un día salado vuelve a servirse")
    assert src.index("_sodio_dia > _techo_sodio()") > src.index("if not meals"), (
        "el juicio de sodio corre antes de tener el día entero: mide una suma incompleta")


def test_el_candidato_salado_queda_de_RESERVA_no_descartado():
    """Descartar platos sanos por una cuenta acumulada dejaría al último slot sin día."""
    import inspect
    src = inspect.getsource(dd.build_day_for_skeleton)
    assert "_reserva" in src, "volvió a descartar en vez de reservar"
    assert "_reserva = (_c, _t, _na)" in src


def test_el_sodio_se_acumula_solo_cuando_la_comida_se_sirve():
    """Sumar al descartar inflaría el día con platos que nadie come."""
    import inspect
    src = inspect.getsource(dd.build_day_for_skeleton)
    cuerpo = src.split("_reserva = (_c, _t, _na)")[1]
    assert "_sodio_dia += _na" in cuerpo.split("break")[0], (
        "la suma dejó de ir pegada a la comida elegida")
