# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-838 · 2026-09-29] Marca cuentas como de PRUEBA desde la consola.

Hace lo mismo que `POST /api/admin/pruebas/lote` del panel y con la MISMA función (`cuentas_prueba.marcar_varias`): las
mismas reglas (`ya_marcada`, `salio_ella`, `no_existe`) y el mismo rastro (una fila `marcar_prueba` en
`admin_access_log` por cuenta marcada, ANTES del INSERT). Sirve para las cuentas que ya existen el día que se enciende
`MEALFIT_ADMIN_TEST_ACCOUNTS` (decisión del dueño, 29-sep: marcar todas las actuales y añadir más a mano desde el panel).
Doc: backend/docs/admin_cuentas_actividad_pruebas.md.

    # Ver qué haría (por defecto: NO escribe nada, solo lista):
    python scripts/marcar_cuentas_prueba.py --todas --motivo "beta cerrada" --admin <uuid>

    # Marcar de verdad, sin las cuentas que no deben (p. ej. el revisor de Apple):
    python scripts/marcar_cuentas_prueba.py --todas --excluir-correos a@x.com --motivo "beta cerrada" --admin <uuid> --aplicar

    # Solo esas:
    python scripts/marcar_cuentas_prueba.py --correos ana@x.com,beto@y.com --motivo "amigos" --admin <uuid> --aplicar

Reglas del script:
  · Lo primero que hace es `cuentas_prueba.activo()`: con el interruptor apagado se NIEGA (salida 2) sin tocar la base.
  · `--admin` tiene que estar en `MEALFIT_ADMIN_USER_IDS`: el rastro debe llevar a un admin real, no a un uuid cualquiera.
  · `--todas` = TODAS las filas de `user_profiles` (también el personal y el revisor de Apple: `--excluir-correos`).
  · Un correo pedido (`--correos`) o excluido (`--excluir-correos`) que no corresponde a ninguna cuenta elegida lo
    rechaza (salida 2): un correo excluido con una errata no excluiría a nadie y marcaría a quien no debía.
  · Marca en tandas de `cuentas_prueba.MAX_LOTE` (100). Si una tanda falla a mitad (rastro o base), corta con salida 1: las
    ya marcadas quedan con su rastro y volver a lanzarlo las devuelve como `ya_marcada`.
  · Solo imprime ASCII (una consola cp1252 revienta con el primer carácter fuera de ella).

Salida: 0 bien (o simulación) · 1 cortado a mitad / la base no respondió · 2 rechazado (interruptor, argumentos, correos).
"""
import argparse
import os
import sys
import uuid
from collections import Counter

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

_RESULTADOS = ("marcada", "ya_marcada", "salio_ella", "no_existe")


def _ascii(v) -> str:
    return str(v).encode("ascii", "replace").decode("ascii")


def _say(texto="") -> None:
    """LA salida del script: todo lo que se imprime pasa por aquí y sale en ASCII."""
    print(_ascii(texto))


def _cargar_entorno() -> None:
    """El `.env` del backend (el interruptor y las URLs de Neon viven ahí en el VPS), ANTES de importar `db_core`."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_BACKEND, ".env"))
    except Exception:
        pass


def _abrir_pool() -> None:
    """Fuera de FastAPI el pool de `db_core` nace CERRADO (SOP de SQL forense, como `scripts/check_pool_prices.py`): sin
    abrirlo, toda lectura sale vacía."""
    import db_core
    if getattr(db_core, "connection_pool", None) is None:
        raise RuntimeError("el pool de Neon no esta configurado (falta NEON_DATABASE_URL_POOLED?)")
    db_core.connection_pool.open()


def _leer_cuentas(correos=None) -> list:
    """`[{id, email}]` de `user_profiles`: todas (`correos=None`) o solo las de esos correos (en minúsculas). Solo lee."""
    import db
    if correos is None:
        filas = db.execute_sql_query(
            "SELECT id::text AS id, email FROM public.user_profiles ORDER BY created_at, id", fetch_all=True)
    else:
        filas = db.execute_sql_query(
            "SELECT id::text AS id, email FROM public.user_profiles WHERE lower(email) = ANY(%s::text[]) "
            "ORDER BY created_at, id", (list(correos),), fetch_all=True)
    return [dict(f) for f in filas or []]


def _correos(texto) -> list:
    """Los correos de una lista separada por comas: sin blancos, en minúsculas y sin repetir (en su orden)."""
    salida = []
    for c in str(texto or "").split(","):
        c = c.strip().lower()
        if c and c not in salida:
            salida.append(c)
    return salida


def _email(cuenta) -> str:
    return str(cuenta.get("email") or "").strip().lower()


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Marca cuentas como de prueba (misma regla y mismo rastro que el panel). "
                    "Sin --aplicar solo lista lo que haria.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--todas", action="store_true", help="todas las filas de user_profiles")
    g.add_argument("--correos", help="solo las cuentas con estos correos (separados por comas)")
    p.add_argument("--excluir-correos", default="", help="correos que NO se marcan (separados por comas)")
    p.add_argument("--motivo", required=True, help="por que se marcan (3 a 300 caracteres); queda en el rastro")
    p.add_argument("--admin", required=True, help="uuid del admin que marca; debe estar en MEALFIT_ADMIN_USER_IDS")
    p.add_argument("--aplicar", action="store_true", help="escribe de verdad (sin esto solo lista lo que haria)")
    return p


def _resumen(total: Counter, salieron: list, correo_de: dict) -> None:
    _say("[marcar_cuentas_prueba] resultado por cuenta:")
    for r in _RESULTADOS:
        _say(f"  {r}: {total.get(r, 0)}")
    if salieron:
        _say("[marcar_cuentas_prueba] salieron ellas del modo de prueba (no se remarcan solas: si piden volver, "
             "marcalas una a una desde el panel con la confirmacion de la vuelta):")
        for uid in salieron:
            _say(f"  {str(uid)[:8]}  {correo_de.get(uid, '')}")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    _cargar_entorno()
    import cuentas_prueba

    # 1. El interruptor maestro, ANTES de tocar nada (ni el pool ni la base).
    if not cuentas_prueba.activo():
        _say("RECHAZADO: el interruptor MEALFIT_ADMIN_TEST_ACCOUNTS esta apagado (es lo normal hasta publicar los "
             "textos legales). Enciendelo en el .env del VPS, reinicia mealfit-backend y vuelve a lanzar esto. "
             "No se toco nada.")
        return 2

    # 2. Argumentos que no dependen de la base.
    from admin_acceso import admin_ids
    try:
        admin = str(uuid.UUID(str(args.admin).strip()))
    except ValueError:
        _say("RECHAZADO: --admin no es un uuid.")
        return 2
    if admin.lower() not in admin_ids():
        _say("RECHAZADO: --admin no esta en MEALFIT_ADMIN_USER_IDS; el rastro tiene que llevar a un admin real.")
        return 2
    try:
        motivo = cuentas_prueba._motivo(args.motivo)      # la regla 3..300 vive en `cuentas_prueba` (SSOT)
    except cuentas_prueba.ErrorPrueba:
        _say("RECHAZADO: --motivo tiene que tener de 3 a 300 caracteres.")
        return 2
    correos = _correos(args.correos) if args.correos is not None else None
    if correos is not None and not correos:
        _say("RECHAZADO: --correos esta vacio.")
        return 2
    excluir = _correos(args.excluir_correos)

    # 3. Quiénes son (solo lectura).
    try:
        _abrir_pool()
        cuentas = _leer_cuentas(correos)
    except Exception as e:  # noqa: BLE001
        _say(f"FATAL: no se pudo leer user_profiles ({type(e).__name__}). No se marco nada.")
        return 1
    hallados = {_email(c) for c in cuentas}
    for nombre, pedidos in (("--correos", correos or []), ("--excluir-correos", excluir)):
        faltan = [c for c in pedidos if c not in hallados]
        if faltan:
            _say(f"RECHAZADO: {nombre} trae correos que no corresponden a ninguna cuenta elegida: "
                 f"{', '.join(faltan)}. Corrigelos: no se marco nada.")
            return 2
    excluidas = [c for c in cuentas if _email(c) in excluir]
    elegidas = [c for c in cuentas if _email(c) not in excluir]

    _say(f"[marcar_cuentas_prueba] modo: {'APLICAR' if args.aplicar else 'SIMULACION (no escribe nada)'}")
    _say(f"[marcar_cuentas_prueba] admin: {admin} | motivo: {motivo}")
    _say(f"[marcar_cuentas_prueba] cuentas a marcar: {len(elegidas)} | excluidas: {len(excluidas)}")
    for c in elegidas:
        _say(f"  marcar    {str(c.get('id'))[:8]}  {c.get('email')}")
    for c in excluidas:
        _say(f"  excluida  {str(c.get('id'))[:8]}  {c.get('email')}")
    if not elegidas:
        _say("[marcar_cuentas_prueba] no hay cuentas que marcar.")
        return 0
    if not args.aplicar:
        _say("[marcar_cuentas_prueba] simulacion: no se escribio nada. Con --aplicar se intenta marcar cada cuenta de "
             "arriba; `marcar` salta por su cuenta las que ya estan marcadas (ya_marcada), las que salieron ellas "
             "(salio_ella) y las que dejaron de existir (no_existe).")
        return 0

    # 4. Marcar, en tandas de MAX_LOTE, con las mismas reglas y el mismo rastro que el panel.
    ids = [str(c.get("id")) for c in elegidas]
    correo_de = {str(c.get("id")): c.get("email") for c in elegidas}
    tope = cuentas_prueba.MAX_LOTE
    tandas = [ids[i:i + tope] for i in range(0, len(ids), tope)]
    total: Counter = Counter()
    salieron: list = []
    for n, tanda in enumerate(tandas, 1):
        try:
            resultados = cuentas_prueba.marcar_varias(admin, tanda, motivo)
        except cuentas_prueba.ErrorPrueba as e:
            _say(f"CORTADO en la tanda {n} de {len(tandas)}: {e.status} {e.detalle}. Las anteriores quedaron marcadas "
                 "con su rastro; al relanzar salen como ya_marcada.")
            _resumen(total, salieron, correo_de)
            return 1
        except Exception as e:  # noqa: BLE001
            _say(f"CORTADO en la tanda {n} de {len(tandas)}: {type(e).__name__}. Las anteriores quedaron marcadas "
                 "con su rastro; al relanzar salen como ya_marcada.")
            _resumen(total, salieron, correo_de)
            return 1
        for r in resultados:
            total[r["resultado"]] += 1
            if r["resultado"] == "salio_ella":
                salieron.append(r["user_id"])
        _say(f"[marcar_cuentas_prueba] tanda {n}/{len(tandas)}: {len(tanda)} cuentas")
    _resumen(total, salieron, correo_de)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
