# Panel de administración — capa 1

[P1-PLAN-LOTE-574..579 · 2026-09-27] Spec: `docs/superpowers/specs/2026-09-27-panel-admin-design.md` (raíz del
workspace). Plan: `docs/superpowers/plans/2026-09-27-panel-admin-capa-1.md`.

## Qué es

`/admin` en el navegador (no existe en la app nativa): el dueño ve cómo va el producto con **agregados**, sin contenido
de ningún usuario. La capa 2 (conversaciones y escaneos seudonimizados) y la capa 3 (fotos cedidas) esperan a que el
dueño apruebe el texto nuevo de la Política de Privacidad; tienen plan propio.

## Quién entra (`admin_acceso.py`)

- Sesión verificada (`get_verified_user_id`) **y** el id en `MEALFIT_ADMIN_USER_IDS` (lista separada por comas en
  `/opt/mealfit/backend/.env`) **y** `MEALFIT_ADMIN_PANEL=true`.
- Sin sesión ⇒ 401. Con sesión fuera de la lista, o con el panel apagado ⇒ **404** (idéntico a una ruta inexistente: el
  panel no se anuncia). El frontend convierte ese 404 en volver al dashboard.
- Rutas bajo `prefix="/api/admin"` SIN el segmento `"/admin/"` en el decorador: ese patrón lo reserva
  `test_p2_audit_4_admin_endpoints_token_required` a los endpoints de mantenimiento con `CRON_SECRET`.

## El rastro (`admin_access_log`)

Migración `p1_plan_lote_574_admin_access_log_2026_09_27.sql`. `registrar_acceso(admin, acción, objetivo, detalle)`
escribe una fila y **lanza** si no puede: quien muestra contenido de un usuario no responde sin rastro (fail-closed).
En la capa 1 solo se anota `abrir_panel` (`GET /api/admin/yo`); las capas 2-3 anotarán cada vista.

## Las dos señales nuevas del escáner (`telemetria_escaner.py`, `pipeline_metrics`)

| node | Cuándo | metadata |
|---|---|---|
| `vision_scan_resultado` | cada análisis de foto (`/api/diary/upload`) | `resultado` (ok/error/no_comida/sin_totales), `photo_kind`, `purpose`; `duration_ms` = lo que tardó el analizador |
| `scan_outcome` | al registrar un plato escaneado (`POST /api/diary/consumed` con `scan_meta`) | conteos: `componentes, cambiados, cantidades_editadas, desmarcados, porcion, dudas, dudas_cambiadas, redescrito, nombre_editado, macros_tecleadas, kcal_ia, kcal_final` + `corregido`, `desvio_kcal` |

Sin texto libre. `scan_meta` lo arma `scanMealDishes.resumenDeCorrecciones` (se omite con la analítica desactivada) y el
backend lo acota: una forma inesperada jamás rechaza la comida. Un registro repetido (`already_logged`) no cuenta dos
veces.

## Los bloques (`admin_metricas.py`)

Uso, Escáner, Coach, Planes, Gasto de IA y Banco del analizador (tabla `analyzer_benchmark_runs`,
`backend/docs/banco_analizador.md`). Cada bloque llega redactado (título + filas `etiqueta`/`valor`, o tabla) y el
frontend solo lo pinta: una métrica nueva es solo backend. Un bloque que falla sale «No disponible» sin tumbar a los
demás.

### Lote 637-639 (28-sep): el panel se entiende

El dueño: «se ve feo y poco entendible». Medido antes del cambio: «3 activas» incluía al dueño probando; «11 alertas
abiertas» eran 6 notas `info` y 5 avisos; «Bloques en cola» eran los creados en el periodo, no los que esperan; el
Escáner pintaba porcentajes de 2 fotos; ningún número tenía contra qué compararse.

- **Secciones** (`seccion` en cada bloque, el pintor abre un título por cambio): Resumen → Requiere atención →
  Usuarios → Producto → Costes → Calidad del escáner.
- **Tipos nuevos**: `resumen` (cifra + cambio frente al periodo anterior con `tono` bueno/malo/aviso/neutro + `ayuda`),
  `avisos` (alertas agrupadas por el PREFIJO de `alert_key` —lo de detrás de «:» son ids y nunca sale—, traducidas en
  `_ALERTA`, con `nivel` crítico/aviso/info), `serie` (barras por día; por semana con más de 31 días; días del reloj de
  RD) y `embudo` (qué hicieron las cuentas nuevas del periodo). Las filas `kpis` aceptan `ayuda`.
- **Sin las cuentas admin** (las de `MEALFIT_ADMIN_USER_IDS` + `plan_tier='admin'`) en todo lo que cuenta personas; el
  gasto SÍ las incluye (el dinero sale igual) y la cola también (es salud del sistema).
- **Cola «ahora»**: programados / listos / generándose / esperan al usuario / atrasados (turno pasado hace >2 h). Los
  atrasados cuentan como aviso en el estado del sistema.
- **Escáner con menos de 20 fotos**: conteos y tiempo, sin porcentajes.
- La pestaña del navegador dice «Panel de administración» (`RouteTitle.RUTAS_INTERNAS`), sin indexar.

Tests: `test_p1_plan_lote_637.py`; frontend `lote638.test.jsx`, `lote639.test.jsx`.

## Cómo se enciende (con permiso del dueño)

1. Aplicar `p1_plan_lote_574_admin_access_log_2026_09_27.sql` (y la 572 del banco) con `scripts/apply_migration.py --apply`.
2. En `/opt/mealfit/backend/.env`: `MEALFIT_ADMIN_USER_IDS=<id de la cuenta del dueño>` y `MEALFIT_ADMIN_PANEL=true`;
   reiniciar `mealfit-backend`.
3. Comprobar: con la sesión del dueño `GET /api/admin/metricas` → 200; con otra cuenta → 404.

## Tests

`test_p1_plan_lote_574_admin_acceso.py`, `_575_telemetria_escaner.py`, `_576_admin_metricas.py`, `_577_router_admin.py`,
`_579_doc.py`; frontend `lote578.test.js`, `lote579.test.jsx`.
