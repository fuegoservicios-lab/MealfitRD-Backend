"""Apply additive migration and verify retry semantics in a rolled-back DB transaction."""
from pathlib import Path
from uuid import uuid4
import sys
from dotenv import load_dotenv
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv('/opt/mealfit/backend/.env')
import db_core

root = Path(__file__).resolve().parents[1]
db_core.connection_pool.open(wait=True)
with db_core.connection_pool.connection() as conn:
    conn.row_factory = dict_row
    with conn.transaction():
        conn.execute((root / 'migrations/ios_free_allowance_2026_10_09.sql').read_text())
    with conn.transaction(force_rollback=True):
        uid = conn.execute('SELECT id FROM user_profiles LIMIT 1').fetchone()['id']
        ids = [uuid4(), uuid4()]
        sql = """INSERT INTO account_grants
            (id,user_id,kind,amount,ends_at,reason,granted_by,usage_scope)
            VALUES (%s,%s,%s,%s,now()+interval '14 days','Verificación transaccional gratuita',%s,'ios_free')
            ON CONFLICT (id) DO NOTHING"""
        for _ in range(2):
            for gid, kind, amount in zip(ids, ('creditos_generacion', 'creditos_coach'), (100,1000)):
                conn.execute(sql, (gid,uid,kind,amount,uid))
        count = conn.execute('SELECT count(*) AS n FROM account_grants WHERE id = ANY(%s)', (ids,)).fetchone()['n']
        assert count == 2, 'Retry duplicated the package'
        conn.execute("UPDATE account_grants SET revoked_at=now() WHERE id=%s", (ids[0],))
        active = conn.execute('SELECT count(*) AS n FROM account_grants WHERE id = ANY(%s) AND revoked_at IS NULL', (ids,)).fetchone()['n']
        assert active == 1
db_core.connection_pool.close()
print('Migration verified; recharge/retry/revocation passed with test grants rolled back.')
