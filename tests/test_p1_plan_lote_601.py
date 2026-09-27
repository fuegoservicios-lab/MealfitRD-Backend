# backend/tests/test_p1_plan_lote_601.py
"""[P1-PLAN-LOTE-601 · 2026-09-27] La regla de aceptación del banco pasa a ser PAREADA contra ≥ 2 corridas de la base.

Medido: la MISMA base corrida dos veces dio 26,2 % y 28,8 % de mediana en calorías (grasa 31,7 % y 36,2 %); una
lectura cambia una mediana de 6,8 % entre corridas. La regla de una sola corrida, con 2 puntos de tolerancia,
rechazaba la base contra sí misma. Ahora se compara plato a plato: la diferencia de error de la candidata contra la
MEDIA de las bases, con errores topados al 100 % (un plato no decide) y un intervalo bootstrap al 90 %.
"""
import random

import pytest

import banco_analizador as ba

MACROS = ("kcal", "proteina_g", "carbs_g", "grasa_g")


def _corrida(errores_por_plato, *, fallos=(), latencia=5.0):
    filas = []
    for i, e in enumerate(errores_por_plato):
        d = f"d{i}"
        if d in fallos:
            filas.append({"dish_id": d, "fallo": "sin_respuesta", "latencia_s": latencia})
        else:
            filas.append({"dish_id": d, "fallo": None, "latencia_s": latencia, "errores": dict(e)})
    return filas


def _base(n=120, semilla=1):
    r = random.Random(semilla)
    return [{k: r.uniform(0.05, 0.6) for k in MACROS} for _ in range(n)]


def _mover(errores, **delta):
    return [{k: max(0.0, e[k] + delta.get(k, 0.0)) for k in MACROS} for e in errores]


def _ruidosa(errores, semilla=2, amp=0.04):
    """La misma base leída otra vez: cada error se mueve un poco (lo que se midió entre dos corridas reales)."""
    r = random.Random(semilla)
    return [{k: max(0.0, e[k] + r.uniform(-amp, amp)) for k in MACROS} for e in errores]


def test_la_misma_corrida_no_se_acepta_ni_se_da_por_peor():
    b = _base()
    r = ba.comparar_pareado([_corrida(b), _corrida(b)], _corrida(b))
    assert r["acepta"] is False
    for k in MACROS:
        assert r["macros"][k]["media"] == 0 and r["macros"][k]["ic90"] == [0, 0]


def test_mejora_clara_en_calorias_se_acepta():
    b = _base()
    r = ba.comparar_pareado([_corrida(b), _corrida(_ruidosa(b))], _corrida(_mover(_ruidosa(b, semilla=5), kcal=-0.15)))
    # tres lecturas con ruido de la misma base: la mejora de 15 puntos tiene que verse igual
    assert r["macros"]["kcal"]["ic90"][1] < 0
    assert r["acepta"] is True, r["motivo"]


def test_mejorar_calorias_empeorando_la_proteina_no_entra():
    b = _base()
    r = ba.comparar_pareado([_corrida(b), _corrida(b)], _corrida(_mover(b, kcal=-0.1, proteina_g=0.1)))
    assert r["acepta"] is False and "proteina_g" in r["motivo"]


def test_una_sola_base_no_basta():
    b = _base()
    r = ba.comparar_pareado([_corrida(b)], _corrida(_mover(b, kcal=-0.2)))
    assert r["acepta"] is False and "2 corridas" in r["motivo"]


def test_un_plato_disparatado_no_decide():
    b = _base(n=100)
    cand = _mover(b, kcal=-0.1)
    cand[0]["kcal"] = 25.0                                      # un error de 2.500 % en un solo plato
    r = ba.comparar_pareado([_corrida(b), _corrida(b)], _corrida(cand))
    assert r["acepta"] is True, r["motivo"]


def test_los_platos_fallidos_en_alguna_corrida_no_se_parean():
    b = _base(n=50)
    r = ba.comparar_pareado([_corrida(b), _corrida(b, fallos={"d3"})], _corrida(_mover(b, kcal=-0.1)))
    assert r["n"] == 49


def test_mas_fallos_o_mas_lentitud_no_entran():
    b = _base()
    mejor = _mover(b, kcal=-0.2)
    con_fallos = ba.comparar_pareado([_corrida(b), _corrida(b)], _corrida(mejor, fallos={f"d{i}" for i in range(10)}))
    assert con_fallos["acepta"] is False and "fallos" in con_fallos["motivo"]
    lenta = ba.comparar_pareado([_corrida(b), _corrida(b)], _corrida(mejor, latencia=7.0))
    assert lenta["acepta"] is False and "latencia" in lenta["motivo"]


def test_es_determinista():
    b, b2 = _base(), _ruidosa(_base(), semilla=3)
    c = _corrida(_mover(b, carbs_g=-0.05))
    assert ba.comparar_pareado([_corrida(b), _corrida(b2)], c) == ba.comparar_pareado([_corrida(b), _corrida(b2)], c)


def test_la_doc_trae_el_ruido_medido_y_la_regla_pareada():
    from pathlib import Path
    t = (Path(ba.__file__).resolve().parent / "docs" / "banco_analizador.md").read_text(encoding="utf-8")
    for ancla in ("Ruido del banco", "28,8 %", "scripts/banco_analizador_comparar.py", "pareada"):
        assert ancla in t, ancla


def test_sin_platos_comunes_no_revienta():
    with pytest.raises(ValueError):
        ba.comparar_pareado([[], []], [])


def test_el_cli_lee_las_corridas_y_sale_segun_la_regla(tmp_path, capsys):
    import importlib.util
    import json
    from pathlib import Path
    # por RUTA: scripts/ en cabeza de sys.path sombrea plan_gym (ratchet de test_p1_plan_lote_13)
    ruta = Path(ba.__file__).resolve().parent / "scripts" / "banco_analizador_comparar.py"
    spec = importlib.util.spec_from_file_location("banco_analizador_comparar", ruta)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    b = _base()
    rutas = {}
    for nombre, filas in (("b1", _corrida(b)), ("b2", _corrida(_ruidosa(b))), ("buena", _corrida(_mover(b, kcal=-0.15))),
                          ("igual", _corrida(b))):
        rutas[nombre] = tmp_path / f"{nombre}.json"
        rutas[nombre].write_text(json.dumps({"per_dish": filas}), encoding="utf-8")
    base = f"{rutas['b1']},{rutas['b2']}"
    assert cli.main(["--base", base, "--candidata", str(rutas["buena"])]) == 0
    assert '"acepta": true' in capsys.readouterr().out
    assert cli.main(["--base", base, "--candidata", str(rutas["igual"])]) == 1
