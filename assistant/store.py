import json
import ssl
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from datetime import datetime, timezone
import asyncpg

SCHEMA = '''
CREATE TABLE IF NOT EXISTS assistant_users (
 user_id BIGINT PRIMARY KEY, history JSONB NOT NULL DEFAULT '[]', memory BOOLEAN NOT NULL DEFAULT TRUE,
 detail TEXT NOT NULL DEFAULT 'auto', last_request TEXT NOT NULL DEFAULT '', retry_ready BOOLEAN NOT NULL DEFAULT FALSE,
 last_answer TEXT NOT NULL DEFAULT '', last_received TIMESTAMPTZ, day DATE, used INTEGER NOT NULL DEFAULT 0);
ALTER TABLE assistant_users ADD COLUMN IF NOT EXISTS last_answer TEXT NOT NULL DEFAULT '';
CREATE TABLE IF NOT EXISTS assistant_updates (
 update_id BIGINT PRIMARY KEY, user_id BIGINT NOT NULL, payload JSONB NOT NULL,
 state TEXT NOT NULL DEFAULT 'pending', created TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE INDEX IF NOT EXISTS assistant_updates_pending ON assistant_updates(state, update_id);
CREATE TABLE IF NOT EXISTS assistant_budget (
 day DATE PRIMARY KEY, attempts INTEGER NOT NULL DEFAULT 0, reserved BIGINT NOT NULL DEFAULT 0,
 tokens BIGINT NOT NULL DEFAULT 0, successes INTEGER NOT NULL DEFAULT 0, errors INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS assistant_attempts (
 id BIGSERIAL PRIMARY KEY, day DATE NOT NULL, model TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'started',
 tokens INTEGER NOT NULL DEFAULT 0, reserved INTEGER NOT NULL DEFAULT 0, created TIMESTAMPTZ NOT NULL DEFAULT now());
ALTER TABLE assistant_attempts ADD COLUMN IF NOT EXISTS reserved INTEGER NOT NULL DEFAULT 0;
CREATE TABLE IF NOT EXISTS assistant_control (name TEXT PRIMARY KEY, until TIMESTAMPTZ NOT NULL);
CREATE TABLE IF NOT EXISTS assistant_migrations (name TEXT PRIMARY KEY, completed TIMESTAMPTZ NOT NULL DEFAULT now());
ALTER TABLE assistant_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE assistant_updates ENABLE ROW LEVEL SECURITY;
ALTER TABLE assistant_budget ENABLE ROW LEVEL SECURITY;
ALTER TABLE assistant_attempts ENABLE ROW LEVEL SECURITY;
ALTER TABLE assistant_control ENABLE ROW LEVEL SECURITY;
ALTER TABLE assistant_migrations ENABLE ROW LEVEL SECURITY;
'''

class Store:
    def __init__(self, pool, cfg):
        self.pool, self.cfg, self.leader = pool, cfg, None

    @classmethod
    async def open(cls, cfg):
        parsed = urlparse(cfg.database_url)
        external = parsed.hostname not in ('localhost', '127.0.0.1')
        dsn = urlunparse(parsed._replace(query=urlencode([(k,v) for k,v in parse_qsl(parsed.query) if k not in ('sslmode','sslrootcert','channel_binding')]))) if external else cfg.database_url
        pool = await asyncpg.create_pool(dsn, ssl=ssl.create_default_context() if external else None, min_size=1, max_size=cfg.workers + 3,
                                        statement_cache_size=0, command_timeout=15)
        obj = cls(pool, cfg)
        try:
            await pool.execute(SCHEMA)
            # A dedicated session, not a transaction-pooler connection. Require direct DB endpoint.
            obj.leader = await pool.acquire()
            if not await obj.leader.fetchval('SELECT pg_try_advisory_lock(718492036)'):
                raise RuntimeError('Уже запущен другой экземпляр бота с этой базой.')
            await pool.execute("UPDATE assistant_updates SET state='interrupted',payload='{}'::jsonb WHERE state='processing'")
            return obj
        except BaseException:
            await obj.close()
            raise

    async def close(self):
        try:
            if self.leader is not None:
                try:
                    if not self.leader.is_closed():
                        await self.leader.execute('SELECT pg_advisory_unlock(718492036)')
                finally:
                    await self.pool.release(self.leader)
                    self.leader = None
        finally:
            await self.pool.close()

    async def user(self, uid):
        await self.pool.execute('INSERT INTO assistant_users(user_id) VALUES($1) ON CONFLICT DO NOTHING', uid)
        row = dict(await self.pool.fetchrow('SELECT * FROM assistant_users WHERE user_id=$1', uid))
        row['history'] = json.loads(row['history'])
        return row

    async def set_user(self, uid, *, history=None, memory=None, detail=None, last_request=None, retry_ready=None, last_answer=None):
        await self.user(uid)
        values = {'history': json.dumps(history, ensure_ascii=False) if history is not None else None,
                  'memory': memory, 'detail': detail, 'last_request': last_request, 'retry_ready': retry_ready, 'last_answer': last_answer}
        changes = [(name, value) for name, value in values.items() if value is not None]
        if changes:
            assignments = ','.join(f'{name}=${i+2}' for i,(name,_) in enumerate(changes))
            await self.pool.execute(f'UPDATE assistant_users SET {assignments} WHERE user_id=$1', uid, *(value for _,value in changes))

    async def admit(self, uid):
        async with self.pool.acquire() as c, c.transaction():
            await c.execute('INSERT INTO assistant_users(user_id) VALUES($1) ON CONFLICT DO NOTHING', uid)
            row = await c.fetchrow('SELECT * FROM assistant_users WHERE user_id=$1 FOR UPDATE', uid)
            now = datetime.now(timezone.utc)
            if row['last_received'] and (now - row['last_received']).total_seconds() < 2:
                return False
            used = row['used'] if row['day'] == now.date() else 0
            if used >= self.cfg.user_daily:
                return False
            await c.execute('UPDATE assistant_users SET last_received=$2,day=$3,used=$4 WHERE user_id=$1', uid, now, now.date(), used+1)
            return True

    async def enqueue(self, data):
        message = data.get('message') or (data.get('callback_query') or {}).get('message') or {}
        sender = (data.get('callback_query') or {}).get('from') or message.get('from') or {}
        uid = sender.get('id')
        if not isinstance(uid, int) or message.get('chat', {}).get('type') != 'private' or message.get('chat', {}).get('id') != uid:
            return True  # unsupported update acknowledged without retention
        if self.cfg.allowed and uid not in self.cfg.allowed:
            return True
        if not isinstance(data.get('update_id'), int):
            return False
        async with self.pool.acquire() as c, c.transaction():
            # Serialize capacity decisions across concurrent webhook requests.
            await c.execute('SELECT pg_advisory_xact_lock(718492037)')
            if await c.fetchval('SELECT 1 FROM assistant_updates WHERE update_id=$1', data['update_id']):
                return True
            count = await c.fetchval("SELECT count(*) FROM assistant_updates WHERE state IN ('pending','processing')")
            if count >= self.cfg.queue_size:
                return False
            await c.execute('INSERT INTO assistant_updates(update_id,user_id,payload) VALUES($1,$2,$3)', data['update_id'], uid, json.dumps(data))
            return True

    async def claim(self):
        async with self.pool.acquire() as c, c.transaction():
            row = await c.fetchrow("""SELECT u.* FROM assistant_updates u WHERE u.state='pending'
             AND NOT EXISTS (SELECT 1 FROM assistant_updates older WHERE older.user_id=u.user_id
               AND older.update_id<u.update_id AND older.state IN ('pending','processing'))
             ORDER BY u.update_id FOR UPDATE OF u SKIP LOCKED LIMIT 1""")
            if row:
                await c.execute("UPDATE assistant_updates SET state='processing' WHERE update_id=$1", row['update_id'])
                return row['update_id'], json.loads(row['payload'])

    async def finish(self, update_id, state='done'):
        await self.pool.execute("UPDATE assistant_updates SET state=$2,payload='{}'::jsonb WHERE update_id=$1", update_id, state)

    async def reserve(self, model, estimated):
        day = datetime.now(timezone.utc).date()
        async with self.pool.acquire() as c, c.transaction():
            await c.execute('SELECT pg_advisory_xact_lock(718492038)')
            await c.execute('INSERT INTO assistant_budget(day) VALUES($1) ON CONFLICT DO NOTHING', day)
            row = await c.fetchrow('SELECT * FROM assistant_budget WHERE day=$1 FOR UPDATE', day)
            recent = await c.fetchval("SELECT count(*) FROM assistant_attempts WHERE created>now()-interval '1 minute'")
            if recent >= 15:
                raise RuntimeError('Локальный лимит 15 API-попыток в минуту. Подождите минуту.')
            recent_tokens = await c.fetchval("SELECT COALESCE(sum(reserved),0) FROM assistant_attempts WHERE created>now()-interval '1 minute'")
            if recent_tokens + estimated > self.cfg.minute_tokens:
                raise RuntimeError('Минутный бюджет токенов занят. Подождите минуту или начните новый диалог для сокращения контекста.')
            if row['attempts'] >= self.cfg.daily_requests or row['reserved'] + estimated > self.cfg.daily_tokens:
                raise RuntimeError('Дневной бюджет бота исчерпан. Попробуйте завтра.')
            await c.execute('UPDATE assistant_budget SET attempts=attempts+1,reserved=reserved+$2 WHERE day=$1', day, estimated)
            return await c.fetchval('INSERT INTO assistant_attempts(day,model,reserved) VALUES($1,$2,$3) RETURNING id', day, model, estimated)

    async def record(self, attempt, state, tokens=0):
        async with self.pool.acquire() as c, c.transaction():
            day = await c.fetchval("UPDATE assistant_attempts SET state=$2,tokens=$3 WHERE id=$1 AND state='started' RETURNING day", attempt, state, tokens)
            if day:
                await c.execute('UPDATE assistant_budget SET tokens=tokens+$2,successes=successes+$3,errors=errors+$4 WHERE day=$1', day, tokens, int(state == 'ok'), int(state != 'ok'))

    async def cooldown(self):
        return await self.pool.fetchval("SELECT EXTRACT(EPOCH FROM until-now()) FROM assistant_control WHERE name='groq' AND until>now()") or 0

    async def pause(self, seconds):
        await self.pool.execute("INSERT INTO assistant_control(name,until) VALUES('groq', now()+$1*interval '1 second') ON CONFLICT(name) DO UPDATE SET until=GREATEST(assistant_control.until,excluded.until)", float(seconds))

    async def stats(self):
        row = await self.pool.fetchrow('SELECT * FROM assistant_budget WHERE day=$1', datetime.now(timezone.utc).date())
        return dict(row) if row else {}

    async def prune(self):
        await self.pool.execute("DELETE FROM assistant_updates WHERE created < now()-interval '7 days' AND state NOT IN ('pending','processing')")
        await self.pool.execute("DELETE FROM assistant_attempts WHERE created < now()-interval '30 days'")
        await self.pool.execute("DELETE FROM assistant_budget WHERE day < current_date-30")
