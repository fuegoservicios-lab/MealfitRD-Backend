"""[P1-PLAN-LOTE-832 · 2026-09-29] Panel · Cuentas: el detalle de una cuenta de prueba — formulario, comidas, planes
(y uno), conversaciones (y una), las fotos del chat y la línea de tiempo de actividad.

Spec docs/superpowers/specs/2026-09-29-admin-cuentas-actividad-pruebas-design.md §4.4, §8 y §13.4 (raíz del
workspace); contrato 9 del plan. Todo con bases falsas: nada de este fichero toca Neon.

Lo que cubre cada bloque (y por qué fallaría contra una versión a medias):
  1. el router: interruptor apagado ⇒ 404; no admin ⇒ 404; cuenta sin marca ⇒ 403 y aviso pendiente ⇒ 409, los dos
     SIN rastro y sin leer nada; la marca se comprueba en CADA vista (la segunda tras salir o quitar falla); el rastro
     `ver_prueba` va después de `exigir_prueba` y antes de leer, y si no se anota: 503 sin datos;
  2. pertenencia: plan, hilo y foto de OTRA cuenta ⇒ 404 aunque existan; la respuesta del coach con `user_id` NULL sí
     sale en su hilo (el SSOT `USER_CHAT_THREAD_IDS_SQL`);
  3. la foto: solo imagen, `Cache-Control: no-store`;
  4. cada sección con la forma del contrato 9;
  5. validación: fechas AAAA-MM-DD y 90 días; `dias` 1..30; `tipos` de la lista; 500 eventos, del más nuevo al más
     viejo;
  6. forma del router y su limitador.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import admin_prueba_detalle as apd
import ajustes_cuenta
import cuentas_prueba as cp
import routers.admin as ra
from auth import get_verified_user_id
from db import USER_CHAT_THREAD_IDS_SQL
from tests.test_p1_plan_lote_830 import _BD as _BDPrueba

_BACKEND = Path(__file__).resolve().parents[1]
ADMIN = "11111111-1111-1111-1111-111111111111"
ADMIN2 = "22222222-2222-2222-2222-222222222222"
UID = "33333333-3333-3333-3333-333333333333"      # la cuenta de prueba (tiene perfil en el fake del 830)
OTRO = "55555555-5555-5555-5555-555555555555"     # otra cuenta, sin marca
BASE = f"/api/admin/cuentas/{UID}/prueba"
MOTIVO = "tester del beta cerrado"
T0 = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)          # el now() de la base falsa
_SSOT = " ".join(USER_CHAT_THREAD_IDS_SQL.split())

S1 = "a1000000-0000-4000-8000-000000000001"   # hilo de la cuenta (agent_sessions.user_id = UID)
S2 = "a1000000-0000-4000-8000-000000000002"   # hilo sin dueño en la sesión, pero con un mensaje suyo
SO = "a1000000-0000-4000-8000-000000000009"   # hilo de OTRO
F1 = "c1000000-0000-4000-8000-000000000001"   # su foto, enviada
F_SIN = "c1000000-0000-4000-8000-000000000002"   # subida y nunca enviada
F_PDF = "c1000000-0000-4000-8000-000000000003"
F_SVG = "c1000000-0000-4000-8000-000000000004"
F_JPG = "c1000000-0000-4000-8000-000000000005"   # «image/jpg», la variante histórica
F_OTRA = "c1000000-0000-4000-8000-000000000009"
P1 = "d1000000-0000-4000-8000-000000000001"
P2 = "d1000000-0000-4000-8000-000000000002"
PO = "d1000000-0000-4000-8000-000000000009"
FOTO = b"\xff\xd8\xff\xe0-foto-del-chat"


def _perfil() -> dict:
    return {
        "mainGoal": "lose_fat", "motivation": "salud", "age": "34", "gender": "female", "weight": "150",
        "weightUnit": "lb", "height": "165", "activityLevel": "moderate", "dietType": "balanced",
        "allergies": ["Maní", "Mariscos"], "otherAllergies": "", "dislikes": [], "medicalConditions": ["Hipertensión"],
        "medications": [], "country": "DO", "budget": "medium", "budgetCurrency": "DOP", "groceryDuration": "weekly",
        "cookingTime": "30min", "householdSize": 1, "scheduleType": "standard",
        "avisos_por_comida": {"desayuno": {"activo": True, "hora": "07:30"},
                              "cena": {"activo": False, "hora": "20:00"}},
        "cultureProfiles": {"main": "dominicana", "secondary": [{"profile_id": "mexicana", "intensity": 0.3}]},
        "clinical_profile": {"labs": {"tfg": 55}, "freeText": "riñón", "updatedAt": "2026-09-20T10:00:00+00:00"},
        "stapleFoods": ["Arroz"],
        "weight_history": [{"date": "2026-09-01", "weight": 152}],        # del servidor: solo en el crudo
        "claveNueva": "valor que el panel no conoce",                      # desconocida: solo en el crudo
        "avisos_raros": [1, 2],
        "_interno": True,
    }


def _comida(i, uid, at, nombre, **k):
    base = {"id": f"f1000000-0000-4000-8000-{i:012d}", "user_id": uid, "consumed_at": at, "meal_type": "almuerzo",
            "meal_name": nombre, "ingredients": ["150 g de arroz"], "calories": 500, "protein": 30, "carbs": 60,
            "healthy_fats": 10, "source": "manual", "plan_ref": None}
    base.update(k)
    return base


def _dia(*nombres):
    return {"day": 1, "meals": [{"meal": "Desayuno", "name": n, "ingredients": ["40 g de avena", "1 guineo"],
                                 "cals": 350, "protein": 12, "carbs": 60, "fats": 6,
                                 "recipe": ["Mise en place: mide la avena", "Montaje: añade el guineo"]}
                                for n in nombres]}


class _Mundo:
    """Las tablas que lee el detalle, en memoria. Entiende SOLO las consultas que emite `admin_prueba_detalle`:
    cualquier otra revienta (un SQL nuevo no pasa sin que alguien mire este fake). Aplica DE VERDAD los filtros de
    pertenencia (`user_id`, los hilos del SSOT) y las ventanas de tiempo: un filtro olvidado se ve en el resultado."""

    def __init__(self, orden=None, vacio=False):
        self.orden = orden if orden is not None else []
        self.consultas: list = []
        self.rota = None
        self.ahora = T0
        self.historial_pedido: list = []
        self.perfiles = {UID: {"health_profile": _perfil(), "locale": "es-DO"},
                         OTRO: {"health_profile": {"mainGoal": "gain_muscle"}, "locale": "en-US"}}
        (self.hechos, self.comidas, self.planes, self.bloques, self.sesiones, self.mensajes, self.adjuntos, self.ia,
         self.metricas, self.alertas, self.pesos, self.agua, self.ajustes) = ([] for _ in range(13))
        if not vacio:
            self._poblar()

    def _poblar(self):
        self.hechos = [
            {"id": "f2000000-0000-4000-8000-000000000001", "user_id": UID, "fact": "Prefiere desayunos salados",
             "created_at": T0 - timedelta(days=5), "salience_score": 0.8, "is_active": True},
            {"id": "f2000000-0000-4000-8000-000000000002", "user_id": UID, "fact": "Trabaja de noche",
             "created_at": T0 - timedelta(days=1), "salience_score": None, "is_active": True},
            {"id": "f2000000-0000-4000-8000-000000000003", "user_id": UID, "fact": "Olvidado",
             "created_at": T0 - timedelta(days=2), "salience_score": 0.5, "is_active": False},
            {"id": "f2000000-0000-4000-8000-000000000009", "user_id": OTRO, "fact": "Dato de otra cuenta",
             "created_at": T0, "salience_score": 1.0, "is_active": True},
        ]
        self.comidas = [
            _comida(1, UID, T0 - timedelta(hours=1), "Mangú con huevo", meal_type="desayuno", source="photo",
                    ingredients=["1 plátano verde", "2 huevos"], calories=420, protein=18, carbs=50, healthy_fats=14),
            # `hasta` es INCLUSIVO (días UTC): la del 29 a las 23:59 es del 29; la del 30 a las 00:00 ya no
            _comida(2, UID, datetime(2026, 9, 29, 23, 59, tzinfo=timezone.utc), "Pollo al horno", meal_type="cena",
                    source="plan_meal", plan_ref={"plan_id": P1, "day_index": 1, "meal_index": 2, "extra": "x"},
                    calories=Decimal("640.0"), protein=Decimal("38.5"), carbs=72, healthy_fats=18),
            _comida(5, UID, datetime(2026, 9, 30, 0, 0, tzinfo=timezone.utc), "Desayuno del 30"),
            _comida(3, UID, datetime(2026, 9, 28, 23, 30, tzinfo=timezone.utc), "Cena tardía", source="raro",
                    ingredients=None),
            _comida(6, UID, datetime(2026, 9, 29, 0, 0, tzinfo=timezone.utc), "Medianoche del 29"),
            _comida(4, UID, T0 - timedelta(days=40), "Comida vieja"),
            _comida(9, OTRO, T0 - timedelta(hours=2), "Comida de otra cuenta"),
        ]
        self.planes = [
            {"id": P1, "user_id": UID, "created_at": T0 - timedelta(days=2), "name": "Plan de septiembre",
             "calories": 1850, "revision": 2,
             "plan_data": {"generation_status": "partial", "days": [_dia("Avena con guineo", "Moro"), {"meals": []}]}},
            {"id": P2, "user_id": UID, "created_at": T0 - timedelta(days=20), "name": None, "calories": None,
             "revision": 1, "plan_data": "no es un objeto"},
            {"id": PO, "user_id": OTRO, "created_at": T0 - timedelta(days=1), "name": "Plan de otra cuenta",
             "calories": 2000, "revision": 1,
             "plan_data": {"generation_status": "complete", "days": [_dia("Secreto")]}},
        ]
        self.bloques = [
            {"id": "e1000000-0000-4000-8000-000000000001", "user_id": UID, "meal_plan_id": P1, "status": "completed",
             "attempts": 1, "week_number": 1, "days_offset": 0, "dead_letter_reason": None, "dead_lettered_at": None,
             "updated_at": T0 - timedelta(days=2), "created_at": T0 - timedelta(days=2)},
            {"id": "e1000000-0000-4000-8000-000000000002", "user_id": UID, "meal_plan_id": P1, "status": "failed",
             "attempts": 3, "week_number": 2, "days_offset": 7, "dead_letter_reason": "timeout del modelo",
             "dead_lettered_at": T0 - timedelta(hours=20), "updated_at": T0 - timedelta(hours=20),
             "created_at": T0 - timedelta(days=2)},
            {"id": "e1000000-0000-4000-8000-000000000003", "user_id": UID, "meal_plan_id": P1, "status": "pending",
             "attempts": 0, "week_number": 3, "days_offset": 14, "dead_letter_reason": None, "dead_lettered_at": None,
             "updated_at": T0 - timedelta(days=2), "created_at": T0 - timedelta(days=2)},
            {"id": "e1000000-0000-4000-8000-000000000009", "user_id": OTRO, "meal_plan_id": PO, "status": "failed",
             "attempts": 3, "week_number": 1, "days_offset": 0, "dead_letter_reason": "fallo ajeno",
             "dead_lettered_at": T0 - timedelta(hours=3), "updated_at": T0 - timedelta(hours=3),
             "created_at": T0 - timedelta(days=1)},
        ]
        self.sesiones = [{"id": S1, "user_id": UID}, {"id": S2, "user_id": None}, {"id": SO, "user_id": OTRO}]
        foto = [{"attachment_id": F1, "content_type": "image/jpeg", "name": "cena.jpg", "byte_size": 20}]
        self.mensajes = [
            {"id": "b1000000-0000-4000-8000-000000000001", "session_id": S1, "user_id": UID, "role": "user",
             "content": "Hola, ¿qué ceno hoy?\nTengo pollo", "created_at": T0 - timedelta(hours=3), "feedback": None,
             "attachments": foto},
            {"id": "b1000000-0000-4000-8000-000000000002", "session_id": S1, "user_id": None, "role": "model",
             "content": "**Pollo al horno** <b>con</b> ensalada", "created_at": T0 - timedelta(hours=3, minutes=-1),
             "feedback": "down", "attachments": None},
            {"id": "b1000000-0000-4000-8000-000000000003", "session_id": S1, "user_id": None, "role": "model",
             "content": "Otra idea: pescado", "created_at": T0 - timedelta(hours=3, minutes=-2), "feedback": "up",
             "attachments": []},
            {"id": "b1000000-0000-4000-8000-000000000004", "session_id": S2, "user_id": UID, "role": "user",
             "content": "x" * 300, "created_at": T0 - timedelta(days=2), "feedback": None, "attachments": None},
            {"id": "b1000000-0000-4000-8000-000000000005", "session_id": S2, "user_id": None, "role": "model",
             "content": "respuesta", "created_at": T0 - timedelta(days=2, minutes=-1), "feedback": None,
             "attachments": None},
            {"id": "b1000000-0000-4000-8000-000000000009", "session_id": SO, "user_id": OTRO, "role": "user",
             "content": "hola, soy otra cuenta", "created_at": T0 - timedelta(hours=1), "feedback": None,
             "attachments": [{"attachment_id": F_OTRA, "content_type": "image/jpeg"}]},
            {"id": "b1000000-0000-4000-8000-000000000010", "session_id": SO, "user_id": None, "role": "model",
             "content": "respuesta a otra cuenta", "created_at": T0 - timedelta(minutes=59), "feedback": "down",
             "attachments": None},
        ]
        m1 = "b1000000-0000-4000-8000-000000000001"
        self.adjuntos = [
            {"id": F1, "session_id": S1, "message_id": m1, "user_id": UID, "content": FOTO,
             "content_type": "image/jpeg"},
            {"id": F_SIN, "session_id": S1, "message_id": None, "user_id": UID, "content": FOTO,
             "content_type": "image/png"},
            {"id": F_PDF, "session_id": S1, "message_id": m1, "user_id": UID, "content": b"%PDF",
             "content_type": "application/pdf"},
            {"id": F_SVG, "session_id": S1, "message_id": m1, "user_id": UID, "content": b"<svg/>",
             "content_type": "image/svg+xml"},
            {"id": F_JPG, "session_id": S1, "message_id": m1, "user_id": UID, "content": memoryview(FOTO),
             "content_type": "image/jpg"},
            {"id": F_OTRA, "session_id": SO, "message_id": "b1000000-0000-4000-8000-000000000009", "user_id": OTRO,
             "content": b"\xff\xd8\xff-ajena", "content_type": "image/jpeg"},
        ]
        self.ia = [
            {"user_id": UID, "created_at": T0 - timedelta(hours=2), "node": "vision_scan", "model": "gemini-3.8-flash",
             "cost_usd_micros": 3100},
            {"user_id": UID, "created_at": T0 - timedelta(days=3), "node": "day_generator", "model": "glm-5.3",
             "cost_usd_micros": 250000},
            {"user_id": UID, "created_at": T0 - timedelta(days=10), "node": "planner", "model": "glm-5.3",
             "cost_usd_micros": 1000000},
            {"user_id": OTRO, "created_at": T0 - timedelta(hours=1), "node": "vision_scan", "model": "x",
             "cost_usd_micros": 999999},
        ]
        self.metricas = [
            {"user_id": UID, "created_at": T0 - timedelta(hours=2, minutes=-1), "node": "scan_outcome",
             "duration_ms": 0, "metadata": {"corregido": True, "cambiados": 1}},
            {"user_id": UID, "created_at": T0 - timedelta(days=4), "node": "plan_meal_deviation", "duration_ms": 0,
             "metadata": {"reason": "comi_otra_cosa", "meal_name": "Avena"}},
            {"user_id": OTRO, "created_at": T0 - timedelta(hours=1), "node": "scan_outcome", "duration_ms": 0,
             "metadata": {"ajena": True}},
        ]
        self.alertas = [
            {"alert_key": f"plan_quality_degraded:{UID}:{P1}", "title": "Plan entregado sin aprobar la revisión",
             "severity": "warning", "triggered_at": T0 - timedelta(days=2), "resolved_at": None},
            {"alert_key": f"plan_quality_degraded:{OTRO}:{PO}", "title": "Alerta de otra cuenta",
             "severity": "warning", "triggered_at": T0 - timedelta(hours=1), "resolved_at": None},
        ]
        self.pesos = [{"user_id": UID, "created_at": T0 - timedelta(hours=20), "weight": 180, "unit": "lb"},
                      {"user_id": OTRO, "created_at": T0 - timedelta(hours=1), "weight": 70, "unit": "kg"}]
        self.agua = [{"user_id": UID, "log_date": date(2026, 9, 28), "glasses": 6,
                      "updated_at": datetime(2026, 9, 28, 22, 0, tzinfo=timezone.utc)},
                     {"user_id": UID, "log_date": date(2026, 9, 10), "glasses": 3,
                      "updated_at": datetime(2026, 9, 10, 22, 0, tzinfo=timezone.utc)}]
        self.ajustes = [{"at": (T0 - timedelta(hours=5)).isoformat(), "clave": "water_tracker_enabled",
                         "etiqueta": "Hidratación", "antes": True, "despues": False, "origen": "coach"}]

    # ── utilidades del fake
    def hilos(self, uid):
        """Los hilos de la cuenta, con las tres fuentes del SSOT: sus sesiones, las sesiones donde escribió y su id."""
        return ({s["id"] for s in self.sesiones if s["user_id"] == uid}
                | {m["session_id"] for m in self.mensajes if m["user_id"] == uid} | {uid})

    def _hilos_de(self, q, p, i):
        assert _SSOT in q, "los hilos de la cuenta salen SIEMPRE del SSOT USER_CHAT_THREAD_IDS_SQL"
        uid = p[i]
        assert p[i:i + 3] == (uid, uid, uid), "el SSOT lleva el uid en sus tres parámetros"
        return self.hilos(uid)

    def _desde(self, dias):
        return self.ahora - timedelta(days=dias)

    @staticmethod
    def _recientes(filas, clave, limite):
        return [dict(f) for f in sorted(filas, key=lambda f: f[clave], reverse=True)[:limite]]

    def historial_ajustes(self, uid, dias):
        self.orden.append("consulta")
        self.historial_pedido.append((uid, dias))
        return [dict(a) for a in self.ajustes] if uid == UID else []

    # ── la consulta
    def query(self, q, p=None, fetch_one=False, fetch_all=False):  # noqa: C901 — un fake: una rama por sentencia
        q = " ".join(q.split())
        p = tuple(p or ())
        self.orden.append("consulta")
        self.consultas.append((q, p))
        if self.rota and self.rota in q:
            raise RuntimeError("tabla sin migrar")
        assert fetch_one or fetch_all, "toda consulta pide sus filas de forma explícita"
        if q == "SELECT health_profile, locale FROM public.user_profiles WHERE id = %s":
            assert fetch_one
            fila = self.perfiles.get(p[0])
            return dict(fila) if fila else None
        if q.startswith("SELECT id::text AS id, fact, created_at, salience_score FROM public.user_facts "
                        "WHERE user_id = %s AND is_active = TRUE ORDER BY created_at DESC"):
            uid, limite = p
            return self._recientes([h for h in self.hechos if h["user_id"] == uid and h["is_active"]],
                                   "created_at", limite)
        if "FROM public.consumed_meals WHERE user_id = %s AND consumed_at >= %s AND consumed_at < %s" in q:
            uid, inicio, fin, limite = p
            assert inicio.tzinfo is not None and fin.tzinfo is not None, "límites con zona: nada de fechas ingenuas"
            suyas = [c for c in self.comidas if c["user_id"] == uid and inicio <= c["consumed_at"] < fin]
            return self._recientes(suyas, "consumed_at", limite)
        if q.startswith("SELECT id::text AS id, created_at, name, calories, revision,"):
            uid, limite = p
            filas = []
            for x in self._recientes([x for x in self.planes if x["user_id"] == uid], "created_at", limite):
                pd = x.pop("plan_data")
                x["estado"] = pd.get("generation_status") if isinstance(pd, dict) else None
                x["dias"] = len(pd["days"]) if isinstance(pd, dict) and isinstance(pd.get("days"), list) else 0
                filas.append(x)
            return filas
        if "FROM public.plan_chunk_queue WHERE user_id = %s AND meal_plan_id = ANY(%s::uuid[])" in q:
            uid, ids, limite = p
            filas = sorted((b for b in self.bloques if b["user_id"] == uid and b["meal_plan_id"] in ids),
                           key=lambda b: (b["week_number"], b["days_offset"], b["created_at"]))
            return [{**b, "plan_id": b["meal_plan_id"]} for b in filas][:limite]
        if q.startswith("SELECT id::text AS id, name, created_at, plan_data FROM public.meal_plans "
                        "WHERE id = %s AND user_id = %s"):
            assert fetch_one
            pid, uid = p
            return next((dict(x) for x in self.planes if x["id"] == pid and x["user_id"] == uid), None)
        if q.startswith("SELECT s.id, s.inicio, s.ultimo, s.mensajes, s.pulgares_abajo, s.fotos"):
            hilos, limite = self._hilos_de(q, p, 0), p[3]
            sesiones = []
            for sid in {m["session_id"] for m in self.mensajes if m["session_id"] in hilos}:
                ms = sorted((m for m in self.mensajes if m["session_id"] == sid), key=lambda m: m["created_at"])
                primero = next((m["content"] for m in ms if m["role"] == "user"), None)
                sesiones.append({
                    "id": sid, "inicio": ms[0]["created_at"], "ultimo": ms[-1]["created_at"], "mensajes": len(ms),
                    "pulgares_abajo": sum(m["feedback"] == "down" for m in ms),
                    "fotos": sum(len(m["attachments"]) for m in ms if isinstance(m["attachments"], list)),
                    "primer_mensaje": primero[:200] if primero is not None else None})
            return sorted(sesiones, key=lambda s: s["ultimo"], reverse=True)[:limite]
        if "FROM public.agent_messages m WHERE m.session_id = %s AND m.session_id::text IN (" in q:
            sid, hilos, limite = p[0], self._hilos_de(q, p, 1), p[4]
            if sid not in hilos:
                return []
            return self._recientes([m for m in self.mensajes if m["session_id"] == sid], "created_at", limite)
        if q.startswith("SELECT a.content, a.content_type FROM public.chat_attachments a WHERE a.id = %s"):
            assert fetch_one and "a.message_id IS NOT NULL" in q
            assert "(a.user_id IS NULL OR a.user_id = %s)" in q and p[4] == p[1]
            aid, hilos, uid = p[0], self._hilos_de(q, p, 1), p[4]
            a = next((a for a in self.adjuntos if a["id"] == aid), None)
            if not a or a["message_id"] is None or a["session_id"] not in hilos or a["user_id"] not in (None, uid):
                return None
            return {"content": a["content"], "content_type": a["content_type"]}
        if q.startswith("SELECT COALESCE(sum(e.cost_usd_micros), 0) AS micros FROM public.llm_usage_events e"):
            assert fetch_one
            uid, dias = p
            return {"micros": sum(e["cost_usd_micros"] for e in self.ia
                                  if e["user_id"] == uid and e["created_at"] >= self._desde(dias))}
        return self._evento(q, p)

    def _evento(self, q, p):
        """Las consultas de la línea de tiempo: `[uid …] días límite`, siempre con la ventana de `now()`."""
        assert "now() - make_interval(days => %s)" in q, f"consulta que el fake no conoce: {q[:160]}"
        dias, limite = p[-2], p[-1]
        desde = self._desde(dias)
        if "FROM public.consumed_meals c WHERE c.user_id = %s AND c.consumed_at >=" in q:
            filas = [{**c, "at": c["consumed_at"]} for c in self.comidas if c["user_id"] == p[0]
                     and c["consumed_at"] >= desde]
        elif "FROM public.meal_plans x WHERE x.user_id = %s AND x.created_at >=" in q:
            filas = []
            for x in self.planes:
                if x["user_id"] == p[0] and x["created_at"] >= desde:
                    pd = x["plan_data"]
                    filas.append({"at": x["created_at"], "name": x["name"], "calories": x["calories"],
                                  "estado": pd.get("generation_status") if isinstance(pd, dict) else None,
                                  "dias": len(pd["days"]) if isinstance(pd, dict) else 0})
        elif "FROM public.agent_messages m WHERE m.role = 'user' AND m.session_id::text IN (" in q:
            hilos = self._hilos_de(q, p, 0)
            filas = [{"at": m["created_at"],
                      "fotos": len(m["attachments"]) if isinstance(m["attachments"], list) else 0}
                     for m in self.mensajes if m["role"] == "user" and m["session_id"] in hilos
                     and m["created_at"] >= desde]
        elif "FROM public.plan_chunk_queue q WHERE q.user_id = %s AND q.status = 'failed'" in q:
            filas = [{**b, "at": b["dead_lettered_at"] or b["updated_at"]} for b in self.bloques
                     if b["user_id"] == p[0] and b["status"] == "failed"
                     and (b["dead_lettered_at"] or b["updated_at"]) >= desde]
        elif "FROM public.llm_usage_events e WHERE e.user_id = %s AND e.created_at >=" in q:
            filas = [{**e, "at": e["created_at"]} for e in self.ia if e["user_id"] == p[0] and e["created_at"] >= desde]
        elif "FROM public.pipeline_metrics pm WHERE pm.user_id = %s AND pm.created_at >=" in q:
            filas = [{**m, "at": m["created_at"]} for m in self.metricas
                     if m["user_id"] == p[0] and m["created_at"] >= desde]
        elif "FROM public.system_alerts a WHERE strpos(a.alert_key, %s) > 0 AND a.triggered_at >=" in q:
            filas = [{**a, "at": a["triggered_at"]} for a in self.alertas
                     if p[0] in a["alert_key"] and a["triggered_at"] >= desde]
        elif "FROM public.weight_log w WHERE w.user_id = %s AND w.created_at >=" in q:
            filas = [{**w, "at": w["created_at"]} for w in self.pesos
                     if w["user_id"] == p[0] and w["created_at"] >= desde]
        elif "FROM public.water_intake_log w WHERE w.user_id = %s AND w.log_date >= (" in q:
            filas = [{**w, "at": w["updated_at"]} for w in self.agua
                     if w["user_id"] == p[0] and w["log_date"] >= desde.date()]
        else:
            raise AssertionError(f"consulta que el fake no conoce: {q[:160]}")
        return self._recientes(filas, "at", limite)

    def tipos_consultados(self):
        marcas = {"comida": "FROM public.consumed_meals c", "plan": "FROM public.meal_plans x",
                  "mensaje": "m.role = 'user' AND m.session_id::text IN", "bloque_fallido": "q.status = 'failed'",
                  "ia": "SELECT e.created_at AS at", "metrica": "FROM public.pipeline_metrics",
                  "alerta": "FROM public.system_alerts", "peso": "FROM public.weight_log",
                  "agua": "FROM public.water_intake_log"}
        vistos = {t for t, m in marcas.items() if any(m in q for q, _ in self.consultas)}
        return vistos | ({"ajuste"} if self.historial_pedido else set())


@pytest.fixture
def mundo(monkeypatch):
    m = _Mundo()
    monkeypatch.setattr(apd, "execute_sql_query", m.query)
    monkeypatch.setattr(ajustes_cuenta, "historial", m.historial_ajustes)
    return m


def _json_de(x) -> str:
    return json.dumps(x, ensure_ascii=False)


# ═════════════════════════════════════════════ 1. el router: interruptor, admin, marca, rastro
class _Panel:
    def __init__(self):
        self.rastro: list = []
        self.orden: list = []
        self.rastro_roto = False
        self.marcas = None
        self.mundo = None

    def anotar(self, admin, accion, objetivo=None, detalle=None):
        self.orden.append("rastro")
        if self.rastro_roto:
            raise RuntimeError("admin_access_log no escribe")
        self.rastro.append((accion, objetivo, detalle))

    @staticmethod
    def cliente(uid=ADMIN):
        app = FastAPI()
        app.include_router(ra.router)
        app.dependency_overrides[get_verified_user_id] = lambda: uid
        return TestClient(app)


@pytest.fixture
def panel(monkeypatch):
    monkeypatch.setenv("MEALFIT_ADMIN_PANEL", "true")
    monkeypatch.setenv("MEALFIT_ADMIN_USER_IDS", ADMIN)
    monkeypatch.setenv("MEALFIT_ADMIN_TEST_ACCOUNTS", "true")
    monkeypatch.delenv("MEALFIT_ADMIN_TEST_REQUIRE_NOTICE", raising=False)
    p = _Panel()
    # La marca DE VERDAD (`cuentas_prueba`) sobre la base falsa del lote 830: nada de caché que simular.
    p.marcas = _BDPrueba()
    monkeypatch.setattr(cp, "execute_sql_query", p.marcas.query)
    monkeypatch.setattr(cp, "execute_sql_write", p.marcas.write)
    monkeypatch.setattr(cp, "registrar_acceso", p.marcas.anotar)
    exigir = cp.exigir_prueba

    def _exigir(uid):
        p.orden.append("exigir")
        return exigir(uid)
    monkeypatch.setattr(cp, "exigir_prueba", _exigir)
    p.mundo = _Mundo(p.orden)
    monkeypatch.setattr(apd, "execute_sql_query", p.mundo.query)
    monkeypatch.setattr(ajustes_cuenta, "historial", p.mundo.historial_ajustes)
    monkeypatch.setattr(ra, "registrar_acceso", p.anotar)
    ra._PRUEBA_DETALLE_LIMITER._hits.clear()
    yield p
    ra._PRUEBA_DETALLE_LIMITER._hits.clear()


def _de_prueba(uid=UID, visto=True):
    """Marca la cuenta como de prueba (y, con `visto`, la persona ya vio el aviso en la app)."""
    cp.marcar(ADMIN, uid, MOTIVO)
    if visto:
        cp.aviso_visto(uid)


# (ruta relativa a /cuentas/{uid}/prueba, sección del rastro, objeto del rastro)
_SECCIONES = [
    ("formulario", "formulario", None),
    ("comidas?desde=2026-09-01&hasta=2026-09-29", "comidas", "2026-09-01..2026-09-29"),
    ("planes", "planes", None),
    (f"planes/{P1}", "plan", P1),
    ("conversaciones", "conversaciones", None),
    (f"conversaciones/{S1}", "conversacion", S1),
    (f"adjuntos/{F1}", "adjunto", F1),
    ("actividad?dias=7&tipos=", "actividad", "7d"),
]
_RUTAS = [r for r, _, _ in _SECCIONES]
_FUNCIONES = ("formulario", "comidas", "planes", "plan", "conversaciones", "conversacion", "adjunto", "actividad")


def _nada_se_lee(monkeypatch):
    def _no(*a, **k):
        pytest.fail("no se llega al módulo del detalle")
    for nombre in _FUNCIONES:
        monkeypatch.setattr(apd, nombre, _no)


@pytest.mark.parametrize("ruta", _RUTAS)
def test_con_el_interruptor_apagado_el_detalle_es_404(panel, monkeypatch, ruta):
    monkeypatch.delenv("MEALFIT_ADMIN_TEST_ACCOUNTS", raising=False)
    _nada_se_lee(monkeypatch)
    r = panel.cliente().get(f"{BASE}/{ruta}")
    assert r.status_code == 404 and r.json() == {"detail": "Not Found"}, "no se anuncia: el mismo 404 que un extraño"
    assert panel.rastro == [] and "exigir" not in panel.orden


@pytest.mark.parametrize("ruta", _RUTAS)
def test_quien_no_es_admin_recibe_404(panel, monkeypatch, ruta):
    _nada_se_lee(monkeypatch)
    _de_prueba()
    assert panel.cliente(ADMIN2).get(f"{BASE}/{ruta}").status_code == 404
    assert panel.rastro == []


@pytest.mark.parametrize("ruta", _RUTAS)
def test_una_cuenta_sin_marca_es_403_sin_rastro_ni_lectura(panel, ruta):
    _de_prueba(UID)                                       # la de prueba es UID; OTRO no lo es
    r = panel.cliente().get(f"/api/admin/cuentas/{OTRO}/prueba/{ruta.replace(UID, OTRO)}")
    assert (r.status_code, r.json()) == (403, {"detail": "no_es_prueba"})
    assert panel.rastro == [] and panel.mundo.consultas == [] and panel.mundo.historial_pedido == []


@pytest.mark.parametrize("ruta", _RUTAS)
def test_con_el_aviso_pendiente_es_409_sin_rastro_ni_lectura(panel, ruta):
    _de_prueba(visto=False)
    r = panel.cliente().get(f"{BASE}/{ruta}")
    assert (r.status_code, r.json()) == (409, {"detail": "aviso_pendiente"})
    assert panel.rastro == [] and panel.mundo.consultas == []


def test_sin_exigir_el_aviso_el_detalle_se_abre(panel, monkeypatch):
    monkeypatch.setenv("MEALFIT_ADMIN_TEST_REQUIRE_NOTICE", "false")
    _de_prueba(visto=False)
    assert panel.cliente().get(f"{BASE}/formulario").status_code == 200


def test_si_la_marca_no_se_puede_leer_503_sin_rastro(panel, monkeypatch):
    _de_prueba()

    def _rota(*a, **k):
        raise RuntimeError("sin DB")
    monkeypatch.setattr(cp, "execute_sql_query", _rota)
    r = panel.cliente().get(f"{BASE}/formulario")
    assert r.status_code == 503 and "campos" not in r.json()
    assert panel.rastro == [] and panel.mundo.consultas == []


@pytest.mark.parametrize("ruta,seccion,objeto", _SECCIONES)
def test_cada_seccion_anota_ver_prueba_tras_exigir_y_antes_de_leer(panel, ruta, seccion, objeto):
    _de_prueba()
    r = panel.cliente().get(f"{BASE}/{ruta}")
    assert r.status_code == 200, r.text
    assert panel.orden[:3] == ["exigir", "rastro", "consulta"]
    assert panel.rastro == [("ver_prueba", UID, {"seccion": seccion, "objeto": objeto})]


@pytest.mark.parametrize("ruta", _RUTAS)
def test_si_el_rastro_falla_503_sin_datos_y_sin_leer(panel, ruta):
    _de_prueba()
    panel.rastro_roto = True
    r = panel.cliente().get(f"{BASE}/{ruta}")
    assert r.status_code == 503 and r.json() == {"detail": "No se pudo registrar el acceso."}
    assert panel.mundo.consultas == [] and panel.mundo.historial_pedido == []


@pytest.mark.parametrize("quien", ["persona", "admin"])
def test_quitar_la_marca_corta_la_segunda_vista(panel, quien):
    """Review Focus 1: sin caché de la marca. Entre dos peticiones la marca se quita y la SEGUNDA vista falla."""
    _de_prueba()
    c = panel.cliente()
    assert c.get(f"{BASE}/conversaciones").status_code == 200
    if quien == "persona":
        assert cp.salir(UID) is True
    else:
        cp.quitar(ADMIN, UID, "terminó la prueba")
    consultas = len(panel.mundo.consultas)
    for ruta in _RUTAS:
        r = c.get(f"{BASE}/{ruta}")
        assert (r.status_code, r.json()) == (403, {"detail": "no_es_prueba"}), ruta
    assert [a for a, *_ in panel.rastro] == ["ver_prueba"], "solo la vista que se abrió deja rastro"
    assert len(panel.mundo.consultas) == consultas, "tras quitar la marca no se lee nada más"


@pytest.mark.parametrize("tabla,ruta", [
    ("FROM public.user_profiles", "formulario"), ("FROM public.consumed_meals", "comidas"),
    ("FROM public.meal_plans", "planes"), ("FROM public.meal_plans", f"planes/{P1}"),
    ("FROM public.agent_messages", "conversaciones"), ("FROM public.agent_messages", f"conversaciones/{S1}"),
    ("FROM public.chat_attachments", f"adjuntos/{F1}"), ("FROM public.pipeline_metrics", "actividad"),
])
def test_si_la_lectura_falla_503_sin_datos(panel, tabla, ruta):
    _de_prueba()
    panel.mundo.rota = tabla
    r = panel.cliente().get(f"{BASE}/{ruta}")
    assert r.status_code == 503 and set(r.json()) == {"detail"}
    assert "tabla sin migrar" not in r.text, "sin el error de la base en la respuesta"
    assert [a for a, *_ in panel.rastro] == ["ver_prueba"]


# ═════════════════════════════════════════════ 2. pertenencia
@pytest.mark.parametrize("ruta,objeto", [
    (f"planes/{PO}", PO),                                  # existe, pero es de OTRO
    ("planes/d1000000-0000-4000-8000-00000000abcd", "d1000000-0000-4000-8000-00000000abcd"),
    (f"conversaciones/{SO}", SO),
    ("conversaciones/a1000000-0000-4000-8000-00000000abcd", "a1000000-0000-4000-8000-00000000abcd"),
    (f"adjuntos/{F_OTRA}", F_OTRA),
    ("adjuntos/c1000000-0000-4000-8000-00000000abcd", "c1000000-0000-4000-8000-00000000abcd"),
])
def test_un_id_de_otra_cuenta_es_404_aunque_exista(panel, ruta, objeto):
    """Review Focus 2: plan, hilo o foto de otra cuenta ⇒ 404, lo mismo que si no existiera. El intento queda
    anotado."""
    _de_prueba()
    r = panel.cliente().get(f"{BASE}/{ruta}")
    assert r.status_code == 404
    assert "Secreto" not in r.text and "otra cuenta" not in r.text and "ajena" not in r.text
    assert panel.rastro[-1][2]["objeto"] == objeto


@pytest.mark.parametrize("ruta", ["planes/no-es-uuid", "conversaciones/1", "adjuntos/x"])
def test_un_id_que_no_es_uuid_es_422_sin_rastro(panel, ruta):
    _de_prueba()
    assert panel.cliente().get(f"{BASE}/{ruta}").status_code == 422
    assert panel.rastro == [] and panel.mundo.consultas == []


def test_el_hilo_trae_las_respuestas_del_coach_con_user_id_nulo(panel):
    """Review Focus 2: las respuestas del coach llevan `user_id` NULL; el SSOT las mete en el hilo de la cuenta."""
    _de_prueba()
    r = panel.cliente().get(f"{BASE}/conversaciones/{S1}")
    assert r.status_code == 200
    h = r.json()
    assert h["id"] == S1 and [m["rol"] for m in h["mensajes"]] == ["persona", "coach", "coach"]
    assert h["mensajes"][1]["texto"] == "**Pollo al horno** <b>con</b> ensalada"


def test_el_hilo_sin_dueno_en_la_sesion_pero_con_un_mensaje_suyo_es_suyo(panel):
    _de_prueba()
    r = panel.cliente().get(f"{BASE}/conversaciones/{S2}")
    assert r.status_code == 200 and [m["rol"] for m in r.json()["mensajes"]] == ["persona", "coach"]


# ═════════════════════════════════════════════ 3. la foto
def test_la_foto_llega_en_bytes_sin_cache(panel):
    _de_prueba()
    r = panel.cliente().get(f"{BASE}/adjuntos/{F1}")
    assert r.status_code == 200 and r.content == FOTO
    assert r.headers["content-type"] == "image/jpeg"
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-content-type-options"] == "nosniff"


def test_la_variante_image_jpg_sale_como_image_jpeg(panel):
    _de_prueba()
    r = panel.cliente().get(f"{BASE}/adjuntos/{F_JPG}")
    assert r.status_code == 200 and r.content == FOTO and r.headers["content-type"] == "image/jpeg"


@pytest.mark.parametrize("aid", [F_PDF, F_SVG, F_SIN])
def test_lo_que_no_es_una_foto_enviada_es_404(panel, aid):
    """Solo imagen (y sin SVG: abierto directamente ejecutaría script) y solo fotos ENVIADAS en la conversación."""
    _de_prueba()
    r = panel.cliente().get(f"{BASE}/adjuntos/{aid}")
    assert r.status_code == 404 and not r.content.startswith(b"%PDF") and b"<svg" not in r.content


def test_el_adjunto_sin_contenido_es_404(mundo):
    mundo.adjuntos[0]["content"] = None
    assert apd.adjunto(UID, F1) is None


# ═════════════════════════════════════════════ 4. cada sección
def _campo(campos, etiqueta):
    hallados = [c for c in campos if c["etiqueta"] == etiqueta]
    assert len(hallados) == 1, f"{etiqueta}: {hallados}"
    return hallados[0]


def test_el_formulario_por_grupos_con_el_crudo_y_la_memoria(mundo):
    r = apd.formulario(UID)
    assert set(r) == {"campos", "crudo", "memoria"}
    assert r["crudo"] == _perfil(), "el crudo es el health_profile entero"
    campos = r["campos"]
    assert all(set(c) == {"grupo", "etiqueta", "valor"} for c in campos)
    assert {c["grupo"] for c in campos} <= set(apd.GRUPOS_FORMULARIO)
    orden = [apd.GRUPOS_FORMULARIO.index(c["grupo"]) for c in campos]
    assert orden == sorted(orden), "los campos salen agrupados, en el orden de los grupos"
    assert apd.GRUPOS_FORMULARIO == ("Objetivo", "Cuerpo", "Dieta y alergias", "Salud", "Horarios",
                                     "Hogar y presupuesto", "País e idioma")
    assert _campo(campos, "Objetivo principal") == {"grupo": "Objetivo", "etiqueta": "Objetivo principal",
                                                    "valor": "lose_fat"}
    assert _campo(campos, "Alergias") == {"grupo": "Dieta y alergias", "etiqueta": "Alergias",
                                          "valor": ["Maní", "Mariscos"]}
    assert _campo(campos, "Condiciones médicas")["grupo"] == "Salud"
    assert _campo(campos, "Edad")["grupo"] == "Cuerpo"
    assert _campo(campos, "Presupuesto")["grupo"] == "Hogar y presupuesto"
    assert _campo(campos, "Desayuno") == {"grupo": "Horarios", "etiqueta": "Desayuno", "valor": "07:30"}
    assert _campo(campos, "Cena")["valor"] == "20:00 (recordatorio apagado)"
    assert _campo(campos, "País") == {"grupo": "País e idioma", "etiqueta": "País", "valor": "DO"}
    assert _campo(campos, "Idioma de la app") == {"grupo": "País e idioma", "etiqueta": "Idioma de la app",
                                                  "valor": "es-DO"}
    cocinas = _campo(campos, "Cocinas que le representan")["valor"]
    assert isinstance(cocinas, str) and "dominicana" in cocinas and "mexicana" in cocinas
    clinico = _campo(campos, "Perfil clínico avanzado")
    assert clinico["grupo"] == "Salud" and isinstance(clinico["valor"], str) and "tfg: 55" in clinico["valor"]
    assert _campo(campos, "Mis básicos")["valor"] == ["Arroz"], "el espejo del formulario vale si falta la canónica"
    texto = _json_de(campos)
    for fuera in ("valor que el panel no conoce", "weight_history", "avisos_raros", "_interno", "152"):
        assert fuera not in texto, f"lo que el panel no conoce va solo en el crudo: {fuera}"
    assert r["memoria"] == [
        {"id": "f2000000-0000-4000-8000-000000000002", "dato": "Trabaja de noche",
         "creado": (T0 - timedelta(days=1)).isoformat(), "relevancia": None},
        {"id": "f2000000-0000-4000-8000-000000000001", "dato": "Prefiere desayunos salados",
         "creado": (T0 - timedelta(days=5)).isoformat(), "relevancia": 0.8}]
    json.dumps(r)


def test_el_formulario_con_valores_raros_no_revienta(mundo):
    """Review Focus 4 en el formulario: listas donde se espera texto, null, objetos anidados, un perfil que no es un
    objeto. Nada revienta y lo que no es texto sale como texto legible."""
    mundo.perfiles[UID]["health_profile"] = {
        "allergies": None, "medications": "Metformina", "age": 34, "struggles": [{"a": 1}, "b", None],
        "dislikes": [None, ""], "avisos_por_comida": "todas a las 8", "targetWeightAuto": True,
        "budgetAmount": 25.5, "staple_foods": ["Habichuelas"], "stapleFoods": ["Otra cosa"]}
    campos = apd.formulario(UID)["campos"]
    assert _campo(campos, "Alergias")["valor"] is None and _campo(campos, "Medicamentos")["valor"] == "Metformina"
    assert _campo(campos, "Edad")["valor"] == 34 and _campo(campos, "Sin meta de peso concreta")["valor"] is True
    assert _campo(campos, "Dificultades")["valor"] == ["a: 1", "b"] and _campo(campos, "No le gusta")["valor"] == []
    assert _campo(campos, "Horas de las comidas")["valor"] == "todas a las 8"
    assert _campo(campos, "Mis básicos")["valor"] == ["Habichuelas"], "manda la clave canónica"
    json.dumps(campos)
    for raro in ("no es un objeto", ["lista"], None, 7):
        mundo.perfiles[UID]["health_profile"] = raro
        r = apd.formulario(UID)
        assert r["crudo"] == {} and [c["etiqueta"] for c in r["campos"]] == ["Idioma de la app"]
    mundo.perfiles[UID]["health_profile"] = '{"mainGoal": "maintenance"}'           # jsonb que llega como texto
    assert _campo(apd.formulario(UID)["campos"], "Objetivo principal")["valor"] == "maintenance"


def test_el_formulario_de_una_cuenta_que_no_existe(mundo):
    assert apd.formulario("77777777-7777-7777-7777-777777777777") is None
    mundo.consultas.clear()
    assert apd.formulario("guest") is None and mundo.consultas == [], "un id que no es uuid no consulta la base"


def test_el_formulario_por_http(panel):
    _de_prueba()
    r = panel.cliente().get(f"{BASE}/formulario")
    assert r.status_code == 200 and set(r.json()) == {"campos", "crudo", "memoria"}


def test_las_claves_del_formulario_existen_en_el_frontend():
    """Las claves salen del wizard y de Configuración, no de una suposición: cada una existe en el frontend."""
    src = _BACKEND.parent / "frontend" / "src"
    if not src.exists():
        pytest.skip("sin el repo del frontend al lado")
    texto = "\n".join(f.read_text(encoding="utf-8", errors="replace")
                      for f in [*src.rglob("*.jsx"), *src.rglob("*.js")])
    claves = apd.claves_del_formulario()
    assert len(claves) >= 40 and "mainGoal" in claves and "avisos_por_comida" in claves
    faltan = [k for k in claves if not re.search(rf"\b{re.escape(k)}\b", texto)]
    assert not faltan, f"claves que el frontend no conoce: {faltan}"


def test_las_comidas_van_por_dias_utc_con_hasta_incluido(mundo):
    """Aclaración del controlador (contrato 9): `hasta` es INCLUSIVO en días UTC — el frontend pide `hasta` = hoy y
    espera las comidas de hoy. Una comida de `hasta` a las 23:59 entra; la del día siguiente a las 00:00, no."""
    r = apd.comidas(UID, "2026-09-29", "2026-09-29")
    assert (r["desde"], r["hasta"]) == ("2026-09-29", "2026-09-29")
    assert [c["plato"] for c in r["comidas"]] == ["Pollo al horno", "Mangú con huevo", "Medianoche del 29"], (
        "del 29 a las 00:00 al 29 a las 23:59 (UTC), la más reciente primero")
    _, inicio, fin, limite = next(p for q, p in mundo.consultas if "FROM public.consumed_meals WHERE" in q)
    assert inicio == datetime(2026, 9, 29, tzinfo=timezone.utc) and fin == datetime(2026, 9, 30, tzinfo=timezone.utc)
    assert inicio.utcoffset() == timedelta(0) and limite == apd.MAX_COMIDAS
    dos = apd.comidas(UID, "2026-09-28", "2026-09-30")["comidas"]
    assert [c["plato"] for c in dos] == ["Desayuno del 30", "Pollo al horno", "Mangú con huevo", "Medianoche del 29",
                                         "Cena tardía"], "los dos extremos del rango entran"


def test_la_forma_de_una_comida(mundo):
    pollo, mangu, _ = apd.comidas(UID, "2026-09-29", "2026-09-29")["comidas"]
    assert pollo == {
        "id": "f1000000-0000-4000-8000-000000000002",
        "at": datetime(2026, 9, 29, 23, 59, tzinfo=timezone.utc).isoformat(),
        "tipo": "cena", "plato": "Pollo al horno", "ingredientes": ["150 g de arroz"], "kcal": 640, "proteina": 38.5,
        "carbohidratos": 72, "grasas": 18, "origen": "plan_meal",
        "plan_ref": {"plan_id": P1, "day_index": 1, "meal_index": 2}}
    assert mangu["origen"] == "photo" and mangu["plan_ref"] is None
    assert mangu["ingredientes"] == ["1 plátano verde", "2 huevos"]
    rara = apd.comidas(UID, "2026-09-28", "2026-09-28")["comidas"][0]
    assert rara["plato"] == "Cena tardía" and rara["origen"] is None and rara["ingredientes"] == []
    json.dumps(pollo)


def test_las_comidas_de_otra_cuenta_no_salen(mundo):
    r = apd.comidas(UID, "2026-08-01", "2026-09-30")
    assert "Comida de otra cuenta" not in _json_de(r) and len(r["comidas"]) == 6


def test_las_comidas_por_http_por_defecto_son_30_dias_hasta_hoy(panel):
    """Por defecto, los 30 últimos días UTC con hoy incluido."""
    _de_prueba()
    r = panel.cliente().get(f"{BASE}/comidas")
    hoy = datetime.now(timezone.utc).date()
    assert r.status_code == 200
    assert (r.json()["desde"], r.json()["hasta"]) == ((hoy - timedelta(days=29)).isoformat(), hoy.isoformat())
    assert panel.rastro[0][2] == {"seccion": "comidas",
                                  "objeto": f"{(hoy - timedelta(days=29)).isoformat()}..{hoy.isoformat()}"}


def test_los_planes(mundo):
    r = apd.planes(UID)
    assert [x["id"] for x in r["planes"]] == [P1, P2], "los suyos, del más nuevo al más viejo"
    p1, p2 = r["planes"]
    assert p1 == {
        "id": P1, "creado": (T0 - timedelta(days=2)).isoformat(), "nombre": "Plan de septiembre", "kcal": 1850,
        "estado_generacion": "partial", "dias": 2, "revision": 2,
        "bloques": {"completed": 1, "failed": 1, "pending": 1},
        "bloques_detalle": [
            {"id": "e1000000-0000-4000-8000-000000000001", "estado": "completed", "intentos": 1, "semana": 1,
             "dias_offset": 0, "motivo_fallo": None},
            {"id": "e1000000-0000-4000-8000-000000000002", "estado": "failed", "intentos": 3, "semana": 2,
             "dias_offset": 7, "motivo_fallo": "timeout del modelo"},
            {"id": "e1000000-0000-4000-8000-000000000003", "estado": "pending", "intentos": 0, "semana": 3,
             "dias_offset": 14, "motivo_fallo": None}]}
    assert (p2["estado_generacion"], p2["dias"], p2["bloques"], p2["bloques_detalle"]) == (None, 0, {}, [])
    assert "fallo ajeno" not in _json_de(r) and "Plan de otra cuenta" not in _json_de(r)
    _, (uid, ids, _lim) = next((q, p) for q, p in mundo.consultas if "FROM public.plan_chunk_queue" in q)
    assert uid == UID and sorted(ids) == sorted([P1, P2])
    json.dumps(r)


def test_los_planes_son_los_20_ultimos(mundo):
    mundo.planes = [{"id": f"d2000000-0000-4000-8000-{i:012d}", "user_id": UID, "created_at": T0 - timedelta(hours=i),
                     "name": f"Plan {i}", "calories": 1800, "revision": 1, "plan_data": {"days": []}}
                    for i in range(25)]
    r = apd.planes(UID)
    assert apd.MAX_PLANES == 20 and [x["nombre"] for x in r["planes"]] == [f"Plan {i}" for i in range(20)]


def test_sin_planes_no_se_piden_bloques(mundo):
    mundo.planes = []
    assert apd.planes(UID) == {"planes": []}
    assert not any("plan_chunk_queue" in q for q, _ in mundo.consultas)


def test_un_plan(mundo):
    r = apd.plan(UID, P1)
    assert set(r) == {"id", "nombre", "creado", "dias", "bloques"}
    assert (r["id"], r["nombre"], r["creado"]) == (P1, "Plan de septiembre", (T0 - timedelta(days=2)).isoformat())
    assert [d["dia"] for d in r["dias"]] == [1, 2] and r["dias"][1]["comidas"] == []
    avena = r["dias"][0]["comidas"][0]
    assert avena == {"tipo": "Desayuno", "plato": "Avena con guineo", "ingredientes": ["40 g de avena", "1 guineo"],
                     "kcal": 350, "macros": {"proteina": 12, "carbohidratos": 60, "grasas": 6},
                     "pasos": ["Mise en place: mide la avena", "Montaje: añade el guineo"]}
    assert [b["estado"] for b in r["bloques"]] == ["completed", "failed", "pending"]
    q, p = next((q, p) for q, p in mundo.consultas if "FROM public.meal_plans WHERE id = %s AND user_id = %s" in q)
    assert p == (P1, UID), "el plan se busca por SU id Y el de la cuenta"
    json.dumps(r)


def test_un_plan_con_datos_raros_no_revienta(mundo):
    mundo.planes[0]["plan_data"] = {"days": [None, {"meals": [None, {"name": "Solo nombre", "recipe": "un paso"}]}]}
    r = apd.plan(UID, P1)
    assert [d["dia"] for d in r["dias"]] == [1, 2] and r["dias"][0]["comidas"] == []
    solo = r["dias"][1]["comidas"][0]
    assert solo["plato"] == "Solo nombre" and solo["ingredientes"] == [] and solo["pasos"] == ["un paso"]
    assert solo["kcal"] == 0 and solo["macros"] == {"proteina": 0, "carbohidratos": 0, "grasas": 0}
    mundo.planes[1]["plan_data"] = "no es un objeto"
    assert apd.plan(UID, P2)["dias"] == []


def test_un_plan_ajeno_o_inexistente_es_none(mundo):
    assert apd.plan(UID, PO) is None and apd.plan(UID, "d1000000-0000-4000-8000-00000000abcd") is None
    assert apd.plan(UID, "no-es-uuid") is None


def test_las_conversaciones(mundo):
    r = apd.conversaciones(UID)
    assert [s["id"] for s in r["sesiones"]] == [S1, S2], "sus hilos (el SSOT), del más reciente al más viejo"
    s1, s2 = r["sesiones"]
    assert s1 == {"id": S1, "inicio": (T0 - timedelta(hours=3)).isoformat(),
                  "ultimo": (T0 - timedelta(hours=3, minutes=-2)).isoformat(), "mensajes": 3, "pulgares_abajo": 1,
                  "fotos": 1, "primer_mensaje": "Hola, ¿qué ceno hoy? Tengo pollo"}
    assert len(s2["primer_mensaje"]) == apd.MAX_PRIMER_MENSAJE == 140 and s2["primer_mensaje"].endswith("…")
    assert "hola, soy otra cuenta" not in _json_de(r)
    q, p = next((q, p) for q, p in mundo.consultas if q.startswith("SELECT s.id"))
    assert p == (UID, UID, UID, apd.MAX_SESIONES) and apd.MAX_SESIONES == 50
    assert "left(p.content, 200)" in q and "count(*) FILTER (WHERE m.feedback = 'down')" in q
    json.dumps(r)


def test_las_conversaciones_son_las_50_ultimas(mundo):
    mundo.sesiones, mundo.mensajes = [], []
    for i in range(55):
        sid = f"a2000000-0000-4000-8000-{i:012d}"
        mundo.sesiones.append({"id": sid, "user_id": UID})
        mundo.mensajes.append({"id": f"b2000000-0000-4000-8000-{i:012d}", "session_id": sid, "user_id": UID,
                               "role": "user", "content": f"mensaje {i}", "created_at": T0 - timedelta(hours=i),
                               "feedback": None, "attachments": None})
    r = apd.conversaciones(UID)
    assert [s["primer_mensaje"] for s in r["sesiones"]] == [f"mensaje {i}" for i in range(50)]


def test_un_hilo(mundo):
    h = apd.conversacion(UID, S1)
    assert set(h) == {"id", "mensajes"} and h["id"] == S1
    m1, m2, m3 = h["mensajes"]
    assert m1 == {"id": "b1000000-0000-4000-8000-000000000001", "rol": "persona",
                  "texto": "Hola, ¿qué ceno hoy?\nTengo pollo", "at": (T0 - timedelta(hours=3)).isoformat(),
                  "feedback": None, "fotos": [{"id": F1, "tipo": "image/jpeg"}]}
    assert (m2["rol"], m2["feedback"], m2["fotos"]) == ("coach", "down", [])
    assert (m3["rol"], m3["feedback"]) == ("coach", "up")
    json.dumps(h)


def test_un_hilo_largo_trae_los_ultimos_en_orden_cronologico(mundo):
    mundo.mensajes = [{"id": f"b3000000-0000-4000-8000-{i:012d}", "session_id": S1, "user_id": UID if i % 2 else None,
                       "role": "user" if i % 2 else "model", "content": f"m{i}",
                       "created_at": T0 - timedelta(minutes=i), "feedback": "raro" if i == 1 else None,
                       "attachments": [{"attachment_id": "no-uuid"}, 3]}
                      for i in range(apd.MAX_MENSAJES + 20)]
    h = apd.conversacion(UID, S1)
    assert len(h["mensajes"]) == apd.MAX_MENSAJES == 500
    assert h["mensajes"][-1]["texto"] == "m0" and h["mensajes"][0]["texto"] == f"m{apd.MAX_MENSAJES - 1}"
    assert all(m["fotos"] == [] for m in h["mensajes"]) and h["mensajes"][-2]["feedback"] is None


def test_un_hilo_ajeno_o_vacio_es_none(mundo):
    assert apd.conversacion(UID, SO) is None
    assert apd.conversacion(UID, "a1000000-0000-4000-8000-00000000abcd") is None
    assert apd.conversacion(UID, "no-uuid") is None


def test_la_foto_del_modulo(mundo):
    assert apd.adjunto(UID, F1) == (FOTO, "image/jpeg")
    assert apd.adjunto(UID, F_JPG) == (FOTO, "image/jpeg"), "memoryview → bytes; image/jpg → image/jpeg"
    for aid in (F_PDF, F_SVG, F_SIN, F_OTRA, "no-uuid"):
        assert apd.adjunto(UID, aid) is None, aid


# ═════════════════════════════════════════════ 4b. la línea de tiempo
_CLAVES_EVENTO = {"at", "tipo", "titulo", "detalle"}


def test_la_linea_de_tiempo_junta_los_diez_tipos(mundo):
    r = apd.actividad(UID, 7, "")
    assert set(r) == {"dias", "gasto_ia_usd", "eventos"} and r["dias"] == 7
    assert all(set(e) == _CLAVES_EVENTO for e in r["eventos"])
    assert all(isinstance(e["titulo"], str) and isinstance(e["detalle"], str) for e in r["eventos"])
    tipos = [e["tipo"] for e in r["eventos"]]
    assert set(tipos) == set(apd.TIPOS_DE_EVENTO), tipos
    assert mundo.tipos_consultados() == set(apd.TIPOS_DE_EVENTO)
    assert mundo.historial_pedido == [(UID, 7)]
    instantes = [datetime.fromisoformat(e["at"]) for e in r["eventos"]]
    assert instantes == sorted(instantes, reverse=True), "del más nuevo al más viejo"
    texto = _json_de(r)
    for ajeno in ("otra cuenta", "fallo ajeno", "ajena", "999999", "Alerta de otra cuenta"):
        assert ajeno not in texto, ajeno
    assert "Hola, ¿qué ceno hoy?" not in texto, "los mensajes van sin su texto: el texto está en «conversaciones»"
    json.dumps(r)


def test_cada_evento_con_su_titulo_y_detalle(mundo):
    eventos = apd.actividad(UID, 7, "")["eventos"]

    def uno(tipo, **k):
        hallados = [e for e in eventos if e["tipo"] == tipo and all(v in e[c] for c, v in k.items())]
        assert len(hallados) == 1, (tipo, k, [e for e in eventos if e["tipo"] == tipo])
        return hallados[0]
    assert uno("comida", titulo="Pollo al horno")["detalle"] == "640 kcal · cena · del plan"
    assert uno("comida", titulo="Mangú con huevo")["detalle"] == "420 kcal · desayuno · foto (escáner)"
    assert uno("plan")["titulo"] == "Plan creado «Plan de septiembre»"
    assert uno("plan")["detalle"] == "2 días · partial · 1850 kcal al día"
    mensajes = [e for e in eventos if e["tipo"] == "mensaje"]
    assert [(e["at"], e["titulo"], e["detalle"]) for e in mensajes] == [
        ((T0 - timedelta(hours=3)).isoformat(), "Mensaje al coach", "con 1 foto"),
        ((T0 - timedelta(days=2)).isoformat(), "Mensaje al coach", "")]
    assert uno("bloque_fallido")["detalle"] == "semana 2 · 3 intentos · timeout del modelo"
    assert uno("ia", titulo="Escáner de fotos")["detalle"] == "vision_scan · gemini-3.8-flash · US$0.0031"
    assert uno("ia", titulo="Generación de días")["detalle"] == "day_generator · glm-5.3 · US$0.2500"
    assert uno("metrica", titulo="Registro de un plato escaneado")["detalle"] == "corregido: sí · cambiados: 1"
    assert "comi_otra_cosa" in uno("metrica", titulo="Desvío declarado del plan")["detalle"]
    alerta = uno("alerta")
    assert alerta["titulo"] == "Plan entregado sin aprobar la revisión"
    assert alerta["detalle"] == "warning · abierta · plan_quality_degraded"
    assert uno("peso") == {"at": (T0 - timedelta(hours=20)).isoformat(), "tipo": "peso", "titulo": "Peso registrado",
                           "detalle": "180 lb"}
    assert uno("agua")["detalle"] == "6 vasos el 2026-09-28"
    assert uno("ajuste") == {"at": (T0 - timedelta(hours=5)).isoformat(), "tipo": "ajuste",
                             "titulo": "Cambió «Hidratación»", "detalle": "sí → no · lo cambió el coach"}


def test_el_periodo_acota_la_linea_de_tiempo_y_el_gasto(mundo):
    r7 = apd.actividad(UID, 7, "ia")
    assert r7["gasto_ia_usd"] == 0.25 and len(r7["eventos"]) == 2          # 3.100 + 250.000 micros
    r30 = apd.actividad(UID, 30, "ia")
    assert r30["gasto_ia_usd"] == 1.25 and len(r30["eventos"]) == 3
    r1 = apd.actividad(UID, 1, "")
    assert all(datetime.fromisoformat(e["at"]) >= T0 - timedelta(days=1) for e in r1["eventos"])
    # fuera del día: el plan y la alerta (hace 2 días), el mensaje de S2, el uso de IA de hace 3 días
    assert {e["tipo"] for e in r1["eventos"]} == {"comida", "mensaje", "bloque_fallido", "ia", "metrica", "peso",
                                                  "agua", "ajuste"}
    assert len([e for e in r1["eventos"] if e["tipo"] == "mensaje"]) == 1


def test_el_gasto_es_del_periodo_entero_aunque_se_filtre_por_tipo(mundo):
    r = apd.actividad(UID, 7, "peso")
    assert [e["tipo"] for e in r["eventos"]] == ["peso"] and r["gasto_ia_usd"] == 0.25


def test_filtrar_por_tipo_solo_consulta_esos_tipos(mundo):
    r = apd.actividad(UID, 7, "comida,ia")
    assert {e["tipo"] for e in r["eventos"]} == {"comida", "ia"}
    assert mundo.tipos_consultados() == {"comida", "ia"} and mundo.historial_pedido == []


def test_la_linea_de_tiempo_va_del_mas_nuevo_al_mas_viejo_con_tope_500():
    m = _Mundo(vacio=True)
    m.ia = [{"user_id": UID, "created_at": T0 - timedelta(minutes=2 * i), "node": "chat_call_model", "model": "glm",
             "cost_usd_micros": 10} for i in range(400)]
    m.comidas = [_comida(i, UID, T0 - timedelta(minutes=2 * i + 1), f"Plato {i}") for i in range(400)]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(apd, "execute_sql_query", m.query)
        mp.setattr(ajustes_cuenta, "historial", m.historial_ajustes)
        r = apd.actividad(UID, 30, None)
    assert apd.MAX_EVENTOS == 500 and len(r["eventos"]) == 500
    instantes = [datetime.fromisoformat(e["at"]) for e in r["eventos"]]
    assert instantes == [T0 - timedelta(minutes=k) for k in range(500)], (
        "los 500 más nuevos, del más nuevo al más viejo")
    assert all(p[-1] == apd.MAX_EVENTOS for q, p in m.consultas if "make_interval" in q and "sum(" not in q)


def test_la_linea_de_tiempo_de_una_cuenta_sin_nada(mundo):
    vacio = _Mundo(vacio=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(apd, "execute_sql_query", vacio.query)
        mp.setattr(ajustes_cuenta, "historial", vacio.historial_ajustes)
        assert apd.actividad(UID, 7, "") == {"dias": 7, "gasto_ia_usd": 0.0, "eventos": []}


def test_los_hilos_de_la_linea_de_tiempo_son_los_del_ssot(mundo):
    apd.actividad(UID, 7, "mensaje")
    q, p = next((q, p) for q, p in mundo.consultas if "m.role = 'user'" in q)
    assert p == (UID, UID, UID, 7, apd.MAX_EVENTOS)


def test_si_una_fuente_de_la_linea_de_tiempo_falla_lanza(mundo):
    """Un hueco silencioso engañaría («esta semana no comió»): si una fuente no se lee, el router da 503."""
    mundo.rota = "FROM public.weight_log"
    with pytest.raises(RuntimeError):
        apd.actividad(UID, 7, "")


def test_la_actividad_por_http_filtra_y_anota(panel):
    _de_prueba()
    r = panel.cliente().get(f"{BASE}/actividad?dias=14&tipos=ia,comida")
    assert r.status_code == 200 and {e["tipo"] for e in r.json()["eventos"]} == {"comida", "ia"}
    assert r.json()["dias"] == 14
    assert panel.rastro == [("ver_prueba", UID, {"seccion": "actividad", "objeto": "14d:comida,ia"})]


# ═════════════════════════════════════════════ 5. validación
@pytest.mark.parametrize("desde,hasta,esperado", [
    (None, None, (date(2026, 8, 31), date(2026, 9, 29))),                 # los 30 últimos, hoy incluido
    ("", "", (date(2026, 8, 31), date(2026, 9, 29))),
    ("2026-09-10", None, (date(2026, 9, 10), date(2026, 10, 9))),         # 30 días desde
    (None, "2026-09-10", (date(2026, 8, 12), date(2026, 9, 10))),         # 30 días hasta
    ("2026-09-01", "2026-11-29", (date(2026, 9, 1), date(2026, 11, 29))),  # 90 días: el tope
    ("2026-09-29", "2026-09-29", (date(2026, 9, 29), date(2026, 9, 29))),
    (date(2026, 9, 1), date(2026, 9, 2), (date(2026, 9, 1), date(2026, 9, 2))),
])
def test_el_rango_de_las_comidas(desde, hasta, esperado):
    assert apd.rango_de_comidas(desde, hasta, hoy=date(2026, 9, 29)) == esperado


@pytest.mark.parametrize("desde,hasta,detalle", [
    ("2026-09-01", "2026-11-30", "rango"),          # 91 días
    ("2026-09-02", "2026-09-01", "rango"),          # al revés
    ("2026-02-30", None, "fecha"), ("29-09-2026", None, "fecha"), ("2026-9-1", None, "fecha"),
    (None, "2026-09-29T00:00", "fecha"), ("hoy", None, "fecha"), (" 2026-09-01", None, "fecha"),
    ("20260901", None, "fecha"), ("0000-01-01", None, "fecha"),
    # en el borde del calendario: 30 días desde/hasta, o el día siguiente al final, no existen (ni un 500)
    ("9999-12-31", None, "fecha"), (None, "0001-01-05", "fecha"), ("9999-12-30", "9999-12-31", "fecha"),
])
def test_un_rango_de_comidas_invalido_es_422(desde, hasta, detalle):
    with pytest.raises(apd.ErrorDetalle) as ei:
        apd.rango_de_comidas(desde, hasta, hoy=date(2026, 9, 29))
    assert (ei.value.status, ei.value.detalle) == (422, detalle)


@pytest.mark.parametrize("consulta", ["desde=2026-02-30", "desde=2026-09-01&hasta=2026-11-30", "hasta=ayer",
                                      "desde=2026-09-05&hasta=2026-09-01", "desde=9999-12-31",
                                      "desde=9999-12-30&hasta=9999-12-31"])
def test_las_comidas_con_un_rango_invalido_son_422_sin_mirar_la_marca(panel, consulta):
    _de_prueba()
    r = panel.cliente().get(f"{BASE}/comidas?{consulta}")
    assert r.status_code == 422 and panel.rastro == [] and "exigir" not in panel.orden


@pytest.mark.parametrize("tipos,esperado", [
    (None, apd.TIPOS_DE_EVENTO), ("", apd.TIPOS_DE_EVENTO), ("  ", apd.TIPOS_DE_EVENTO),
    ("ia", ("ia",)), ("ia,comida", ("comida", "ia")), (" comida , ia ,", ("comida", "ia")),
    ("ajuste,ajuste", ("ajuste",)), (["peso", "agua"], ("peso", "agua")),
])
def test_los_tipos_de_la_actividad(tipos, esperado):
    assert apd.tipos_de_actividad(tipos) == esperado


@pytest.mark.parametrize("tipos", ["xyz", "comida,xyz", "Comida", "comida;ia", "comidas"])
def test_un_tipo_desconocido_es_422(tipos):
    with pytest.raises(apd.ErrorDetalle) as ei:
        apd.tipos_de_actividad(tipos)
    assert (ei.value.status, ei.value.detalle) == (422, "tipos")


def test_los_tipos_son_los_diez_del_contrato():
    assert apd.TIPOS_DE_EVENTO == ("comida", "plan", "mensaje", "bloque_fallido", "ia", "metrica", "alerta", "peso",
                                   "agua", "ajuste")


@pytest.mark.parametrize("dias", [0, -1, 31, 100])
def test_los_dias_de_la_actividad_fuera_de_1_a_30(dias, mundo):
    with pytest.raises(apd.ErrorDetalle) as ei:
        apd.actividad(UID, dias, "")
    assert (ei.value.status, ei.value.detalle) == (422, "dias") and mundo.consultas == []


@pytest.mark.parametrize("consulta", ["dias=0", "dias=31", "dias=x", "tipos=xyz", "tipos=comida,chat",
                                      "tipos=" + "a" * 201])
def test_la_actividad_rechaza_lo_que_no_esta_en_el_contrato(panel, consulta):
    _de_prueba()
    r = panel.cliente().get(f"{BASE}/actividad?{consulta}")
    assert r.status_code == 422 and panel.rastro == [] and "exigir" not in panel.orden


def test_la_actividad_por_defecto_son_7_dias_de_todo(panel):
    _de_prueba()
    r = panel.cliente().get(f"{BASE}/actividad")
    assert r.status_code == 200 and r.json()["dias"] == 7
    assert panel.rastro[0][2] == {"seccion": "actividad", "objeto": "7d"}
    assert panel.mundo.tipos_consultados() == set(apd.TIPOS_DE_EVENTO)


# ═════════════════════════════════════════════ 6. forma del router y limitador
_DETALLE = ["/api/admin/cuentas/{user_id}/prueba/formulario", "/api/admin/cuentas/{user_id}/prueba/comidas",
            "/api/admin/cuentas/{user_id}/prueba/planes", "/api/admin/cuentas/{user_id}/prueba/planes/{plan_id}",
            "/api/admin/cuentas/{user_id}/prueba/conversaciones",
            "/api/admin/cuentas/{user_id}/prueba/conversaciones/{session_id}",
            "/api/admin/cuentas/{user_id}/prueba/adjuntos/{attachment_id}",
            "/api/admin/cuentas/{user_id}/prueba/actividad"]


def _ruta(path, metodo="GET"):
    return next(r for r in ra.router.routes if r.path == path and metodo in r.methods)


@pytest.mark.parametrize("path", _DETALLE)
def test_cada_ruta_del_detalle_pasa_por_el_interruptor_y_su_limitador(path):
    deps = [d.dependency for d in _ruta(path).dependencies]
    assert ra._exigir_knob_pruebas in deps and ra._PRUEBA_DETALLE_LIMITER in deps
    assert deps.index(ra._exigir_knob_pruebas) < deps.index(ra._PRUEBA_DETALLE_LIMITER), "apagado, ni gasta cupo"


def test_el_detalle_son_solo_esas_ocho_rutas_de_lectura():
    rutas = {(r.path, m) for r in ra.router.routes for m in getattr(r, "methods", ())
             if r.path.startswith("/api/admin/cuentas/{user_id}/prueba/")}
    assert rutas == {(p, "GET") for p in _DETALLE} | {("/api/admin/cuentas/{user_id}/prueba/quitar", "POST")}, (
        "solo lectura: el detalle no edita nada de la cuenta de prueba")


def test_el_limitador_del_detalle_tiene_su_par_propio():
    src = (_BACKEND / "routers" / "admin.py").read_text(encoding="utf-8")
    m = re.search(r"_PRUEBA_DETALLE_LIMITER = RateLimiter\(max_calls=(\d+), period_seconds=(\d+)\)", src)
    assert m and m.groups() == ("90", "60"), "el par del contrato, con sus números a la vista"
    assert re.fullmatch(r"_[A-Z_]+_LIMITER", "_PRUEBA_DETALLE_LIMITER"), "el guard de limitadores no admite dígitos"
    pares = []
    for f in [*_BACKEND.glob("*.py"), *(_BACKEND / "routers").glob("*.py")]:
        pares += re.findall(r"RateLimiter\(\s*(?:max_calls\s*=\s*)?(\d+)\s*,\s*(?:period_seconds\s*=\s*)?(\d+)\s*\)",
                            f.read_text(encoding="utf-8"))
    assert pares.count(m.groups()) == 1, "Redis comparte la ventana por par (rl:<max>:<periodo>:<uid>)"


def test_el_detalle_no_cobra_cuota_ni_llama_a_la_ia():
    src = (_BACKEND / "admin_prueba_detalle.py").read_text(encoding="utf-8")
    assert "verify_api_quota" not in src
    importados = set(re.findall(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", src, re.M))
    ia = {"llm_provider", "agent", "tools", "langchain", "openai", "anthropic", "google", "graph_orchestrator",
          "vision_agent", "fact_extractor", "embeddings_provider"}
    assert not {i.split(".")[0] for i in importados} & ia, importados
    assert "verify_api_quota" not in (_BACKEND / "routers" / "admin.py").read_text(encoding="utf-8")


def test_el_modulo_usa_la_fachada_de_la_base_y_el_ssot_de_los_hilos():
    src = (_BACKEND / "admin_prueba_detalle.py").read_text(encoding="utf-8")
    assert "from db import USER_CHAT_THREAD_IDS_SQL, execute_sql_query" in src
    assert "execute_sql_write" not in src, "solo lectura"
