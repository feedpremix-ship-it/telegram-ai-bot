"""Real PostgreSQL integration. Run only against a disposable test database."""
import asyncio
import os
import pytest
import pytest_asyncio
from assistant.config import Config
from assistant.store import Store
from bot import Service
from types import SimpleNamespace
from unittest.mock import AsyncMock

DSN=os.getenv('TEST_DATABASE_URL','')
pytestmark=pytest.mark.skipif(not DSN,reason='TEST_DATABASE_URL not set; disposable PostgreSQL required')

@pytest_asyncio.fixture
async def store():
    cfg=Config(database_url=DSN, queue_size=3, daily_requests=3,daily_tokens=10000)
    s=await Store.open(cfg)
    await s.pool.execute('TRUNCATE assistant_users,assistant_updates,assistant_budget,assistant_attempts,assistant_control,assistant_migrations RESTART IDENTITY')
    yield s
    await s.close()


def payload(n,uid=1):
    return {'update_id':n,'message':{'message_id':n,'date':1700000000,'chat':{'id':uid,'type':'private'},'from':{'id':uid,'is_bot':False,'first_name':'Test'},'text':'test'}}

async def test_dedup_capacity_and_user_order(store):
    assert await store.enqueue(payload(1))
    assert await store.enqueue(payload(1))
    assert await store.enqueue(payload(2))
    assert await store.enqueue(payload(3,2))
    assert not await store.enqueue(payload(4,3))
    first,second=await asyncio.gather(store.claim(),store.claim())
    claimed={first[0],second[0]}
    assert claimed=={1,3}  # two users concurrently, same user never simultaneously
    assert await store.claim() is None
    await store.finish(1)
    assert (await store.claim())[0]==2

async def test_atomic_memory_isolation_clear(store):
    await store.set_user(1,history=[{'role':'user','content':'secret'}],last_answer='answer')
    assert (await store.user(2))['history']==[]
    assert (await store.user(1))['history'][0]['content']=='secret'
    await store.set_user(1,history=[],last_request='',last_answer='')
    assert (await store.user(1))['history']==[]

async def test_budget_all_attempts_cooldown_and_spam(store):
    for status in ('ok','rate_limited','uncertain'):
        i=await store.reserve('model',100)
        await store.record(i,status,10 if status=='ok' else 0)
    with pytest.raises(RuntimeError):
        await store.reserve('model',100)
    row=await store.stats()
    assert row['attempts']==3 and row['successes']==1 and row['errors']==2 and row['tokens']==10 and row['reserved']==300
    await store.pause(5)
    assert 0 < await store.cooldown() <= 5
    assert await store.admit(99)
    assert not await store.admit(99)

async def test_parallel_budget_never_overcommits(store):
    results=await asyncio.gather(*(store.reserve('m',100) for _ in range(10)),return_exceptions=True)
    assert sum(isinstance(r,int) for r in results)==3

async def test_restart_persistence_pending_and_uncertain(store):
    await store.set_user(1,history=[{'role':'user','content':'remember'}])
    await store.enqueue(payload(1))
    await store.enqueue(payload(2,2))
    await store.claim()
    await store.close()
    reopened=await Store.open(store.cfg)
    # fixture finalizer will close the replacement, not a stale closed pool
    store.pool,store.leader=reopened.pool,reopened.leader
    assert (await store.user(1))['history'][0]['content']=='remember'
    assert await store.pool.fetchval('SELECT state FROM assistant_updates WHERE update_id=1')=='interrupted'
    assert (await store.claim())[0]==2

async def test_single_instance_guard(store):
    with pytest.raises(RuntimeError,match='другой экземпляр'):
        await Store.open(store.cfg)

async def test_invalid_users_not_retained(store):
    d=payload(1); d['message']['chat']['type']='group'
    assert await store.enqueue(d)
    assert await store.claim() is None

async def test_worker_order_and_real_persistent_chat(store):
    # Exercise actual worker + PTB JSON decoding + real PostgreSQL state, mocked Telegram/LLM.
    await store.enqueue(payload(1))
    await store.enqueue(payload(2))
    await store.enqueue(payload(3,2))
    seen=[]; in_flight=set()
    from telegram import Bot
    bot=Bot('123:test')
    async def process(update):
        uid=update.effective_user.id
        assert uid not in in_flight
        in_flight.add(uid)
        await asyncio.sleep(.02)
        u=await store.user(uid)
        await store.set_user(uid,history=u['history']+[{'role':'user','content':str(update.update_id)}])
        seen.append(update.update_id)
        in_flight.remove(uid)
    app=SimpleNamespace(bot=bot,process_update=process)
    svc=Service(store.cfg,store,None,None)
    tasks=[asyncio.create_task(svc.worker(app)) for _ in range(2)]
    try:
        for _ in range(100):
            if len(seen)==3:
                break
            await asyncio.sleep(.02)
        assert len(seen)==3 and seen.index(1)<seen.index(2)
        assert [x['content'] for x in (await store.user(1))['history']]==['1','2']
    finally:
        for t in tasks:t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)

async def test_sqlite_migration_readonly_idempotent(store,tmp_path):
    import sqlite3
    from scripts.migrate_sqlite import migrate
    backup=tmp_path/'old.db'
    with sqlite3.connect(backup) as c:
        c.execute('CREATE TABLE messages(id INTEGER PRIMARY KEY,chat_id INTEGER,role TEXT,content TEXT,ts INTEGER)')
        c.execute("INSERT INTO messages VALUES(1,7,'user','remember',0)")
        c.execute("INSERT INTO messages VALUES(2,-50,'user','group-private',0)")
    before=backup.read_bytes()
    await store.close()
    assert await migrate(backup,store.cfg)==1
    assert await migrate(backup,store.cfg)==0
    reopened=await Store.open(store.cfg)
    store.pool,store.leader=reopened.pool,reopened.leader
    assert (await store.user(7))['history'][0]['content']=='remember'
    assert backup.read_bytes()==before

async def test_durable_webhook_real_db(store):
    from bot import make_web_app
    from aiohttp.test_utils import TestClient,TestServer
    store.cfg.webhook_secret='z'*40
    svc=Service(store.cfg,store,None,None);svc.ready=True
    async with TestClient(TestServer(make_web_app(svc))) as client:
        headers={'X-Telegram-Bot-Api-Secret-Token':'z'*40}
        assert (await client.post('/webhook',headers=headers,json=payload(10))).status==200
        assert await store.pool.fetchval('SELECT state FROM assistant_updates WHERE update_id=10')=='pending'
        assert (await client.post('/webhook',headers=headers,json=payload(10))).status==200
        assert await store.pool.fetchval('SELECT count(*) FROM assistant_updates')==1

async def test_full_startup_webhook_reply_shutdown(store,monkeypatch):
    import socket
    import json
    import httpx
    from aiohttp import ClientSession
    from bot import run
    cfg=store.cfg
    cfg.token='123:test';cfg.groq_key='test'
    cfg.webhook_url='https://test.onrender.com/webhook';cfg.webhook_secret='a'*40
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));cfg.port=s.getsockname()[1]
    await store.close()
    calls=[]
    def respond(req):
        calls.append(str(req.url))
        if req.url.host=='api.groq.com':
            if req.url.path.endswith('/models'):
                return httpx.Response(200,json={'data':[{'id':cfg.models[0]}]})
            return httpx.Response(200,json={'choices':[{'message':{'content':'Привет, тест памяти'}}], 'usage':{'total_tokens':30}})
        method=req.url.path.rsplit('/',1)[-1].lower()
        if method=='getme':
            result={'id':123,'is_bot':True,'first_name':'Test','username':'test_bot'}
        elif method=='sendmessage':
            result={'message_id':99,'date':1700000000,'chat':{'id':1,'type':'private'},'text':'Привет, тест памяти'}
        else:
            result=True
        return httpx.Response(200,json={'ok':True,'result':result})
    original=httpx.AsyncClient
    def factory(*args,**kw):
        kw['transport']=httpx.MockTransport(respond)
        return original(*args,**kw)
    monkeypatch.setattr(httpx,'AsyncClient',factory)
    stops=[]
    monkeypatch.setattr(asyncio.get_running_loop(),'add_signal_handler',lambda sig,cb:stops.append(cb))
    task=asyncio.create_task(run(cfg))
    try:
        async with ClientSession() as client:
            for _ in range(100):
                try:
                    r=await client.get(f'http://127.0.0.1:{cfg.port}/healthz')
                    if r.status==200:break
                except OSError:pass
                if task.done(): await task
                await asyncio.sleep(.03)
            else:pytest.fail('startup health timeout')
            r=await client.post(f'http://127.0.0.1:{cfg.port}/webhook',json=payload(100),headers={'X-Telegram-Bot-Api-Secret-Token':cfg.webhook_secret})
            assert r.status==200
            for _ in range(100):
                if any('sendMessage' in x for x in calls):break
                await asyncio.sleep(.03)
            assert any('setWebhook' in x for x in calls)
            assert any('sendMessage' in x for x in calls)
    finally:
        if stops:stops[0]()
        else:task.cancel()
        await asyncio.wait_for(task,10)
        reopened=await Store.open(cfg)
        store.pool,store.leader=reopened.pool,reopened.leader
    assert (await store.user(1))['history'][-1]['content']=='Привет, тест памяти'

async def test_minute_token_budget_atomic(store):
    store.cfg.minute_tokens=150
    await store.reserve('m',100)
    with pytest.raises(RuntimeError,match='Минутный бюджет'):
        await store.reserve('m',100)
    assert (await store.stats())['attempts']==1

async def test_sqlite_conflict_rolls_back_and_preserves_backup(store,tmp_path):
    import sqlite3
    from scripts.migrate_sqlite import migrate
    backup=tmp_path/'old.db'
    with sqlite3.connect(backup) as c:
        c.execute('CREATE TABLE messages(id INTEGER PRIMARY KEY,chat_id INTEGER,role TEXT,content TEXT)')
        c.execute("INSERT INTO messages VALUES(1,1,'user','old')")
        c.execute("INSERT INTO messages VALUES(2,2,'user','old2')")
    await store.set_user(2,history=[{'role':'user','content':'new'}])
    await store.close()
    try:
        with pytest.raises(ValueError,match='Перенос отменён'):
            await migrate(backup,store.cfg)
    finally:
        reopened=await Store.open(store.cfg)
        store.pool,store.leader=reopened.pool,reopened.leader
    assert (await store.user(1))['history']==[]
    assert (await store.user(2))['history'][0]['content']=='new'

async def test_rls_enabled_for_all_private_tables(store):
    rows=await store.pool.fetch("SELECT relname,relrowsecurity FROM pg_class WHERE relname=ANY($1::text[])", ['assistant_users','assistant_updates','assistant_budget','assistant_attempts','assistant_control','assistant_migrations'])
    assert len(rows)==6 and all(row['relrowsecurity'] for row in rows)
