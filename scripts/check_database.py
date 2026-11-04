"""Read-only DB connectivity/TLS check; no schema or Telegram webhook changes."""
import asyncio
import os
import ssl
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
import asyncpg
from dotenv import load_dotenv

async def check():
    raw = os.getenv('DATABASE_URL', '')
    parsed = urlparse(raw)
    if parsed.scheme not in ('postgres', 'postgresql') or not parsed.hostname:
        raise ValueError('DATABASE_URL missing or invalid')
    external = parsed.hostname not in ('localhost','127.0.0.1')
    query = dict(parse_qsl(parsed.query))
    if external and query.get('sslmode') not in ('require','verify-full'):
        raise ValueError('External TLS required')
    dsn = urlunparse(parsed._replace(query=urlencode([(k,v) for k,v in query.items() if k not in ('sslmode','sslrootcert')]))) if external else raw
    conn = await asyncpg.connect(dsn, ssl=ssl.create_default_context() if external else None, timeout=15)
    try:
        async with conn.transaction(readonly=True):
            assert await conn.fetchval('SELECT 1') == 1
            secure = await conn.fetchval('SELECT ssl FROM pg_stat_ssl WHERE pid=pg_backend_pid()')
            if external and not secure:
                raise ValueError('TLS not active')
        print('PostgreSQL connection OK; TLS='+str(bool(secure))+'; no writes performed.')
    finally:
        await conn.close()

if __name__ == '__main__':
    load_dotenv()
    try:
        asyncio.run(check())
    except Exception:
        print('Database check failed. Check direct endpoint, TLS and access. Credentials suppressed.')
        raise SystemExit(1) from None
