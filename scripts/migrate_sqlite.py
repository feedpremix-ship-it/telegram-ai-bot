"""Read-only, idempotent migration of the supplied bot's SQLite messages table.
Stop the bot first. Never deletes or writes to the SQLite backup.
"""
import argparse
import asyncio
import hashlib
import os
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from assistant.config import Config
from assistant.store import Store
from bot import bounded_history
import json


def read_backup(path):
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError('Для этого инструмента размер резервной копии не должен превышать 100 MB.')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    con = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)
    try:
        columns = {r[1] for r in con.execute('PRAGMA table_info(messages)')}
        if not {'id','chat_id','role','content'} <= columns:
            raise ValueError('Ожидалась таблица messages(id, chat_id, role, content, ts).')
        users = {}
        # Positive chat ids in the supplied old schema denote private chats only.
        for (uid,) in con.execute('SELECT DISTINCT chat_id FROM messages WHERE chat_id>0'):
            rows = con.execute("SELECT role,content FROM messages WHERE chat_id=? AND role IN ('user','assistant') ORDER BY id DESC LIMIT 30", (uid,)).fetchall()
            users[uid] = bounded_history([{'role':r,'content':str(t or '')[:8000]} for r,t in reversed(rows)],12000)
        return digest, users
    finally:
        con.close()


async def migrate(path, cfg):
    digest, users = await asyncio.to_thread(read_backup, path)
    name = 'sqlite:'+digest
    store = await Store.open(cfg)  # singleton guard: refuses migration while bot is running
    try:
        async with store.pool.acquire() as con, con.transaction():
            if await con.fetchval('SELECT 1 FROM assistant_migrations WHERE name=$1', name):
                return 0
            for uid, history in users.items():
                old = await con.fetchrow('SELECT history,memory FROM assistant_users WHERE user_id=$1 FOR UPDATE', uid)
                if old and (json.loads(old['history']) or not old['memory']):
                    raise ValueError('В целевой базе уже есть история. Перенос отменён целиком без перезаписи.')
                await con.execute('INSERT INTO assistant_users(user_id,history) VALUES($1,$2) ON CONFLICT(user_id) DO UPDATE SET history=excluded.history', uid,json.dumps(history,ensure_ascii=False))
            await con.execute('INSERT INTO assistant_migrations(name) VALUES($1)', name)
        return len(users)
    finally:
        await store.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backup',help='Путь к согласованной SQLite backup-копии, не рабочей базе с WAL')
    args=parser.parse_args()
    load_dotenv()
    if not os.getenv('DATABASE_URL'):
        parser.error('Задайте DATABASE_URL в Environment или локальном .env')
    try:
        count=asyncio.run(migrate(args.backup, Config.load()))
        print(f'Перенос завершён: {count} пользователей; исходная копия не изменена.')
    except Exception:
        print('Перенос отменён. Проверьте схему backup, пустую целевую историю, остановку бота и DATABASE_URL. Исходная база сохранена.',file=sys.stderr)
        raise SystemExit(1) from None

if __name__=='__main__':
    main()
