"""[P1-PLAN-LOTE-1107 · 1108 · 2026-11-18] Las preguntas autónomas que eligió el dueño: peso estancado con buen
cumplimiento → el coach propone un ajuste pequeño; el domingo, el coach cuenta la semana y UNA cosa para la siguiente."""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import estancamiento_peso as e  # noqa: E402
import resumen_del_domingo as r  # noqa: E402

HOY = date(2026, 11, 30)


def _hp(objetivo, *pesos):
    return {"mainGoal": objetivo,
            "weight_history": [{"date": d, "weight": w, "unit": "lb"} for d, w in pesos]}


def test_estancado_si_no_se_movio_hacia_su_objetivo_en_28_dias():
    quieto = _hp("lose_fat", ("2026-10-30", 183.0), ("2026-11-15", 182.8), ("2026-11-29", 182.6))
    assert e.estancado(quieto, HOY) == {"a": "183.0", "b": "182.6", "u": "lb", "objetivo": "lose_fat"}
    bajando = _hp("lose_fat", ("2026-10-30", 183.0), ("2026-11-29", 180.0))
    assert e.estancado(bajando, HOY) is None
    subiendo = _hp("gain_muscle", ("2026-10-30", 150.0), ("2026-11-29", 152.0))
    assert e.estancado(subiendo, HOY) is None
    assert e.estancado(_hp("gain_muscle", ("2026-10-30", 150.0), ("2026-11-29", 150.2)), HOY)["objetivo"] == "gain_muscle"


def test_no_se_propone_sin_ventana_sin_pesaje_reciente_o_con_mantenimiento():
    assert e.estancado(_hp("lose_fat", ("2026-11-10", 183.0), ("2026-11-29", 183.0)), HOY) is None, "solo 19 días"
    assert e.estancado(_hp("lose_fat", ("2026-09-30", 183.0), ("2026-11-10", 183.0)), HOY) is None, "último de hace 20 días"
    assert e.estancado(_hp("maintenance", ("2026-10-30", 183.0), ("2026-11-29", 183.0)), HOY) is None


def test_solo_con_buen_cumplimiento_si_no_el_problema_no_es_la_meta():
    sem = lambda d: {"cerrada": True, "dias_en_meta": d}
    assert e.cumplio([sem(5), sem(4), sem(5), {"cerrada": False, "dias_en_meta": 1}]) == 14 / 3
    assert e.cumplio([sem(2), sem(4), sem(5)]) is None
    assert e.cumplio([sem(5), sem(5)]) is None, "menos de 3 semanas cerradas"


def test_el_mensaje_propone_y_no_cambia_nada_solo():
    est = {"a": "183.0", "b": "182.6", "u": "lb", "objetivo": "lose_fat"}
    for loc in ("es-DO", "en-US", "pt-BR", "fr-FR", "it-IT"):
        t, c, m = e.textos(loc, est, 4.67)
        assert t and c and "183.0" in m and "150" in m and "4.7" in m
    assert "¿Cuál prefieres?" in e.textos("es-DO", est, 5.0)[2] and " 5 días" in e.textos("es-DO", est, 5.0)[2]


def test_solo_los_lunes_a_las_9():
    assert e.toca_ahora(datetime(2026, 11, 30, 13, 0, tzinfo=timezone.utc), 240)        # lunes 9:00 en RD
    assert not e.toca_ahora(datetime(2026, 11, 29, 13, 0, tzinfo=timezone.utc), 240)    # domingo


SEMANA = {"numero": 9, "cerrada": True, "dias_en_meta": 4, "estrellas": 2, "dias_registrados": 7,
          "dias_bajo": 1, "dias_sobre": 2, "proteina_media": 95.0, "termina": "2026-11-22"}


def test_el_domingo_cuenta_la_semana_y_una_sola_cosa():
    txt = r.mensaje("es-DO", SEMANA, racha=3, meta_proteina=100)
    assert txt.startswith("Tu Semana 9 cerró con 4 de 7 días en meta ★★.")
    assert "1 por debajo, 2 por encima y 0 sin registrar" in txt and "Llevas 3 días seguidos" in txt
    assert "Para la Semana 10: busca quedar dentro de tu meta un día más" in txt


def test_el_consejo_va_por_orden_registro_proteina_encima_debajo():
    assert r.consejo({**SEMANA, "dias_registrados": 3}, 100)[0] == "registro"
    assert r.consejo(SEMANA, 130) == ("proteina", {"g": 35})
    assert r.consejo({**SEMANA, "dias_sobre": 0, "dias_bajo": 3}, 100)[0] == "bajo"
    assert r.consejo({**SEMANA, "dias_sobre": 0, "dias_bajo": 0}, 100)[0] == "bien"
    assert r.consejo(SEMANA, None)[0] == "sobre", "sin meta de proteína se salta ese consejo"


def test_cinco_idiomas_y_sin_semana_cerrada_no_hay_mensaje():
    for loc in ("es-DO", "en-US", "pt-BR", "fr-FR", "it-IT"):
        assert "9" in r.mensaje(loc, SEMANA, 0, 100) and "10" in r.mensaje(loc, SEMANA, 0, 100)
    assert r.mensaje("es-DO", None) is None


def test_cableado():
    import crons_informe_y_progreso as c
    assert ("estancamiento_peso", 1, "estancamiento_peso", "revisar_estancamientos") in c.JOBS
    src = (_BACKEND / "aviso_semanal.py").read_text(encoding="utf-8")
    assert '__import__("resumen_del_domingo").escribir_en_el_chat(' in src


# ── [P1-PLAN-LOTE-1106] el suplemento que se acaba ──────────────────────────────────────────────────────────────────
import aviso_suplemento as s  # noqa: E402


def test_dias_restantes_a_su_ritmo():
    assert s.dias_restantes(5, 14) == 5.0          # una toma al día
    assert s.dias_restantes(5, 7) == 10.0          # una cada dos días
    assert s.dias_restantes(5, 2) is None, "con menos de 3 tomas en 14 días no hay ritmo"
    assert s.dias_restantes(0, 14) is None


def test_avisa_una_vez_por_pote_hasta_que_se_rellene():
    assert s.toca(4.0, 4, None) is True
    assert s.toca(6.0, 6, None) is False
    assert s.toca(3.0, 3, {"avisado_con": 4}) is False, "ya se avisó con este pote"
    assert s.toca(4.0, 8, {"avisado_con": 4}) is True, "subió: se rellenó y vuelve a avisar"


def test_textos_del_suplemento_en_cinco_idiomas():
    for loc in ("es-DO", "en-US", "pt-BR", "fr-FR", "it-IT"):
        t, c, m = s.textos(loc, "Creatina", 4.4, 4)
        assert "Creatina" in t and "4" in c and "4" in m


def test_cableado_del_suplemento():
    import crons_informe_y_progreso as c
    assert ("aviso_suplemento", 1, "aviso_suplemento", "enviar_avisos_de_suplemento") in c.JOBS
