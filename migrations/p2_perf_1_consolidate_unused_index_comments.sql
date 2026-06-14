-- [P2-PERF-1 · 2026-05-10] Consolidar COMMENT ON INDEX runtime → migration files.
--
-- Causa raíz:
--   El audit 2026-05-10 confirmó que 3 índices reportados como `unused_index`
--   por el advisor son falsos positivos (cubren FK ON DELETE CASCADE o sirven
--   endpoints reales). Cada uno tenía COMMENT aplicado en runtime durante
--   audits previos (P2-B 2026-05-07, P1-HIST-NEW-7 2026-05-09), PERO solo
--   `idx_chunk_lesson_telemetry_plan_week` quedó documentado en archivo de
--   migración (p1_hist_new_7_*). Las otras 2 COMMENTs viven SOLO en runtime;
--   un `db reset` desde migrations las pierde y un futuro operador no sabría
--   que esos índices son intencionales (probable drop equivocado).
--
--   Mismo patrón anti-drift que P1-NEW-A (runtime DDL recreaba dup indexes
--   por no estar en migrations) y P2-NEW-E/G (consolidación runtime DDL).
--
-- Justificación de KEEP (los 2 índices):
--
--   1. `idx_failed_inventory_deductions_user_id` (FK CASCADE → auth.users)
--      Sin este índice, eliminar un usuario auth disparaba seq-scan completo
--      de `failed_inventory_deductions` para cada cascade. Lección P2-5: el
--      advisor `unused_index` NO observa uso interno por FK durante DELETE,
--      así que reporta 0 idx_scan aunque el índice sea load-bearing.
--
--   2. `idx_nightly_rotation_queue_user_id` (FK CASCADE → user_profiles)
--      Misma razón: cubre la FK ON DELETE CASCADE. Sin él, eliminación de
--      user_profiles → seq-scan de toda la cola de rotación nocturna.
--
-- Sin cambio de comportamiento: solo añade COMMENT (metadata catalogo); el
-- índice físico ya existe. Idempotente: `COMMENT ON INDEX` reemplaza el
-- comentario previo si ya estaba aplicado en runtime — no error si re-corre.

COMMENT ON INDEX public.idx_failed_inventory_deductions_user_id IS
    '[P2-B · 2026-05-07] KEEP. Cubre FK failed_inventory_deductions_user_id_fkey -> auth.users(id) ON DELETE CASCADE. El advisor unused_index NO observa uso interno por FK; sin este indice, eliminar un usuario auth haria seq-scan. Leccion P2-5. Consolidado a migration por P2-PERF-1 2026-05-10.';

COMMENT ON INDEX public.idx_nightly_rotation_queue_user_id IS
    '[P2-B · 2026-05-07] KEEP. Cubre FK nightly_rotation_queue_user_id_fkey -> user_profiles(id) ON DELETE CASCADE. El advisor unused_index NO observa uso interno por FK. Leccion P2-5. Consolidado a migration por P2-PERF-1 2026-05-10.';
