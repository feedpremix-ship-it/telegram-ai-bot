import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from aiohttp.test_utils import TestClient, TestServer
from telegram import Update

from assistant.config import Config
from assistant.output import messages, links_allowed, clean_links, utf16len
from assistant.charts import expression, render, isolated_render
from assistant.llm import LLM, UserError, retry_after
from assistant.tools import Tools
from bot import Service, make_web_app, make_application, bounded_history


def config(**kw):
    return Config(token='123:test', groq_key='test', database_url='postgresql://localhost/test', **kw)

class Memory:
    def __init__(self):
        self.users = {}
        self.attempts = []
        self.results = []
        self.pause_seconds = 0
        self.leader = SimpleNamespace(is_closed=lambda: False)
        self.pool = SimpleNamespace(fetchval=AsyncMock(return_value=1))
        self.enqueue = AsyncMock(return_value=True)
    async def user(self, uid):
        return self.users.setdefault(uid, dict(history=[],memory=True,detail='auto',last_request='',last_answer='',retry_ready=False)).copy()
    async def set_user(self, uid, **kw):
        await self.user(uid)
        self.users[uid].update(kw)
    async def admit(self, uid):
        return True
    async def cooldown(self):
        return self.pause_seconds
    async def pause(self, seconds):
        self.pause_seconds = seconds
    async def reserve(self, model, estimated):
        self.attempts.append((model,estimated))
        return len(self.attempts)
    async def record(self, attempt, state, tokens=0):
        self.results.append((attempt,state,tokens))


def update(uid=1, text='Привет'):
    msg = SimpleNamespace(text=text, reply_text=AsyncMock(),reply_photo=AsyncMock(),chat=SimpleNamespace(send_action=AsyncMock()))
    return SimpleNamespace(effective_user=SimpleNamespace(id=uid), effective_chat=SimpleNamespace(id=uid,type='private'), effective_message=msg)

@pytest.mark.parametrize('changes', [{}, {'token':''},{'database_url':'sqlite:///a'}, {'webhook_url':'http://a/webhook'},
                                      {'webhook_url':'https://a/webhook','webhook_secret':'short'}, {'workers':5},{'daily_tokens':0}])
def test_config(changes):
    cfg = config()
    for k,v in changes.items():
        setattr(cfg,k,v)
    if not changes:
        cfg.validate()
    else:
        with pytest.raises(ValueError):
            cfg.validate()


def test_config_webhook_tls():
    cfg = config(webhook_url='https://service.onrender.com/webhook',webhook_secret='x'*40)
    cfg.validate()
    cfg.database_url='postgresql://user:pass@external/db'
    with pytest.raises(ValueError):
        cfg.validate()

@pytest.mark.parametrize('text', ['Дай ссылку на Groq', 'ссылка на документацию', 'give me a link', 'https://example.com'])
def test_link_request(text):
    assert links_allowed(text)

@pytest.mark.parametrize('text', ['что такое ссылка', 'расскажи о погоде', 'Не присылай ссылки', 'Без ссылок дай ссылку на сайт'])
def test_no_link_request(text):
    assert not links_allowed(text)


def test_format_and_split():
    source = '**жирный** <script>&\n```python\nprint("<hi>")\n```\n- список\n' + '😀&'*6000
    chunks = messages(source)
    assert len(chunks)>10
    assert all(utf16len(x)<=4096 for x in chunks)
    assert '<b>жирный</b>' in ''.join(chunks)
    assert '<script>' not in ''.join(chunks)
    assert '<pre>' in ''.join(chunks)
    assert '&lt;script&gt;' in ''.join(chunks)
    assert '[ссылка скрыта]' in clean_links('https://example.com google.com')
    assert clean_links('https://example.com',True)=='https://example.com'

@pytest.mark.parametrize('bad', ["__import__('os').system('id')", 'x.__class__', '[x for x in range(4)]', '2**999999', 'open("a")', 'sin(x,1)'])
def test_chart_security(bad):
    with pytest.raises((ValueError, SyntaxError)):
        expression(bad)(2)


def test_chart_png():
    assert expression('sin(x)+x^2')(0)==0
    assert render({'kind':'function','expression':'sin(x)'})[:8]==b'\x89PNG\r\n\x1a\n'
    assert render({'kind':'bar','labels':['A','B'],'values':[1,2]})[:4]==b'\x89PNG'
    with pytest.raises(ValueError):
        render({'kind':'pie','labels':['A'],'values':[-1]})


def test_chart_process():
    assert isolated_render({'kind':'function','expression':'x**2'})[:4]==b'\x89PNG'


async def make_llm(responder, cfg=None):
    mem = Memory()
    client = httpx.AsyncClient(transport=httpx.MockTransport(responder))
    return LLM(cfg or config(),mem,client),mem,client

async def test_llm_success():
    def respond(req):
        body=json.loads(req.content)
        assert body['parallel_tool_calls'] is False
        return httpx.Response(200,json={'choices':[{'message':{'content':'Ответ'}}], 'usage':{'total_tokens':20}})
    llm,mem,client=await make_llm(respond)
    async with client:
        answer,images=await llm.answer([], 'Привет','auto',AsyncMock(),False)
    assert answer=='Ответ' and images==[]
    assert mem.results==[(1,'ok',20)]

async def test_model_fallback_sequential():
    calls=[]
    def respond(req):
        calls.append(json.loads(req.content)['model'])
        if len(calls)==1:
            return httpx.Response(400,json={'error':{'code':'model_decommissioned'}})
        return httpx.Response(200,json={'choices':[{'message':{'content':'ok'}}]})
    llm,mem,client=await make_llm(respond)
    async with client:
        assert (await llm.call([]))['content']=='ok'
    assert calls==list(config().models)
    assert mem.results[0][1]=='http_400'

@pytest.mark.parametrize('status',[401,403,429,500])
async def test_no_fallback_on_errors(status):
    llm,mem,client=await make_llm(lambda req:httpx.Response(status, headers={'Retry-After':'7'},json={'error':{}}))
    async with client:
        with pytest.raises(UserError):
            await llm.call([])
    assert len(mem.attempts)==1
    if status==429:
        assert mem.pause_seconds==7
        with pytest.raises(UserError):
            await llm.call([])
        assert len(mem.attempts)==1

async def test_timeout_no_repeat():
    def respond(req):
        raise httpx.ReadTimeout('test')
    llm,mem,client=await make_llm(respond)
    async with client:
        with pytest.raises(UserError):
            await llm.call([])
    assert len(mem.attempts)==1 and mem.results[0][1]=='uncertain'

async def test_invalid_response():
    llm,mem,client=await make_llm(lambda req:httpx.Response(200,json={'choices':[]}))
    async with client:
        with pytest.raises(UserError):
            await llm.call([])
    assert mem.results[0][1]=='invalid_response'

async def test_live_model_catalog_mock():
    llm,mem,client=await make_llm(lambda req:httpx.Response(200,json={'data':[{'id':config().models[1]}]}))
    async with client:
        await llm.verify_models()
    assert config().models[0] in llm.disabled

async def test_tool_translate_not_weather():
    seen=[]
    def respond(req):
        body=json.loads(req.content); seen.append(body)
        return httpx.Response(200,json={'choices':[{'message':{'content':'weather'}}]})
    llm,mem,client=await make_llm(respond)
    tool=AsyncMock()
    async with client:
        answer,_=await llm.answer([], 'Переведи на английский: погода', 'auto', tool, False)
    assert answer=='weather'
    tool.assert_not_awaited()
    assert seen[0]['messages'][-1]['content']=='Переведи на английский: погода'

async def test_tool_chain_weather():
    seen=[]
    def respond(req):
        body=json.loads(req.content); seen.append(body)
        msg = {'tool_calls':[{'id':'1','type':'function','function':{'name':'weather','arguments':'{"city":"Москва"}'}}]} if len(seen)==1 else {'content':'В Москве 10°C'}
        return httpx.Response(200,json={'choices':[{'message':msg}]})
    llm,mem,client=await make_llm(respond)
    tool=AsyncMock(return_value=('10°C',None))
    async with client:
        answer,_=await llm.answer([], 'Какая погода в Москве?', 'auto', tool, False)
    tool.assert_awaited_once_with('weather',{'city':'Москва'})
    assert len(seen)==2 and 'tools' not in seen[1] and '10°C' in answer

async def test_tools_search_missing_weather_ambiguous():
    client=httpx.AsyncClient(transport=httpx.MockTransport(lambda req:httpx.Response(200,json={'results':[{'name':'X','country':'A'}, {'name':'X','country':'B'}]})))
    async with client:
        tools=Tools(config(),client)
        assert 'не настроен' in (await tools.run('web_search', {'query':'x'}))[0]
        assert 'Уточни' in (await tools.run('weather', {'city':'X'}))[0]

async def test_service_memory_isolation_clear_and_settings():
    mem=Memory(); llm=SimpleNamespace(answer=AsyncMock(return_value=('**Ответ** https://example.com',[])))
    service=Service(config(),mem,llm,SimpleNamespace(run=AsyncMock()))
    one,two=update(1),update(2)
    await service.chat(one,'вопрос')
    assert len((await mem.user(1))['history'])==2
    assert (await mem.user(2))['history']==[]
    assert 'https://' not in (await mem.user(1))['history'][-1]['content']
    await service.chat(two,'дай ссылку на пример')
    assert 'https://' in (await mem.user(2))['history'][-1]['content']
    await service.menu(one,action='/clear')
    assert (await mem.user(1))['history']==[] and not (await mem.user(1))['last_request']
    await mem.set_user(1,memory=False)
    await service.chat(one,'новое')
    assert (await mem.user(1))['history']==[]

async def test_retry_uses_saved_answer_no_api():
    mem=Memory(); llm=SimpleNamespace(answer=AsyncMock())
    await mem.set_user(1,last_request='question',last_answer='saved',retry_ready=True)
    service=Service(config(),mem,llm,SimpleNamespace(run=AsyncMock()))
    await service.chat(update(1),'question',retry=True)
    llm.answer.assert_not_awaited()
    assert not (await mem.user(1))['retry_ready']

async def test_error_private_not_leaked():
    mem=Memory(); llm=SimpleNamespace(answer=AsyncMock(side_effect=RuntimeError('secret-token')))
    one=update(1)
    await Service(config(),mem,llm,SimpleNamespace(run=AsyncMock())).chat(one,'hello')
    assert all('secret-token' not in str(c) for c in one.effective_message.reply_text.call_args_list)
    assert (await mem.user(1))['retry_ready']

async def test_authorization_admin_and_callback():
    mem=Memory(); service=Service(config(admins=frozenset({99})),mem,None,None)
    one=update(1)
    await service.menu(one,action='/stats')
    assert 'администратору' in one.effective_message.reply_text.call_args.args[0]
    one.effective_chat.type='group'
    assert not service.authorized(one)
    one=update(1)
    one.callback_query=SimpleNamespace(data='detail:short',answer=AsyncMock())
    await service.callback(one,None)
    assert (await mem.user(1))['detail']=='short'
    one.callback_query.data='memory'
    await service.callback(one,None)
    assert not (await mem.user(1))['memory']

async def test_webhook_auth_payload_queue_health():
    mem=Memory(); service=Service(config(webhook_secret='x'*40),mem,None,None); service.ready=True
    async with TestClient(TestServer(make_web_app(service))) as client:
        assert (await client.get('/healthz')).status==200
        assert (await client.post('/webhook',json={'update_id':1})).status==403
        headers={'X-Telegram-Bot-Api-Secret-Token':'x'*40}
        assert (await client.post('/webhook',headers=headers,json={'update_id':1})).status==200
        mem.enqueue.assert_awaited_once()
        assert (await client.post('/webhook',headers=headers,json=[])).status==400
        mem.enqueue.return_value=False
        assert (await client.post('/webhook',headers=headers,json={'update_id':2})).status==503
        service.ready=False
        assert (await client.get('/healthz')).status==503


def test_import_application_and_context_limit():
    mem=Memory(); cfg=config(); svc=Service(cfg,mem,None,None)
    app=make_application(cfg,svc)
    assert app.handlers and app.job_queue is not None
    history=[{'role':'user','content':'a'*100},{'role':'assistant','content':'b'*100}]*20
    bounded=bounded_history(history,1000)
    assert sum(len(x['content']) for x in bounded)<=1000 and bounded[0]['role']=='user'
    assert retry_after('bad')==60

@pytest.mark.parametrize('text',['t.me/test','example.xyz/path','пример.рф','tg://resolve?domain=test','www.example.com','[name](https://example.com)'])
def test_all_url_forms_suppressed(text):
    assert not any(x in clean_links(text) for x in ['t.me','example.xyz','пример.рф','tg://','www.example','https://'])

async def test_weather_data_and_real_search_shape_mocked():
    calls=[]
    def respond(req):
        calls.append(req)
        if req.url.host=='geocoding-api.open-meteo.com':
            assert req.url.params['name']=='Нижний Новгород'
            return httpx.Response(200,json={'results':[{'name':'Нижний Новгород','latitude':56.3,'longitude':44,'country':'Россия'}]})
        if req.url.host=='api.open-meteo.com':
            return httpx.Response(200,json={'current':{'temperature_2m':8},'current_units':{'temperature_2m':'°C'},'timezone':'Europe/Moscow'})
        assert req.url.host=='api.tavily.com'
        assert json.loads(req.content)['search_depth']=='basic'
        return httpx.Response(200,json={'results':[{'title':'Current','content':'fresh data','url':'https://example.com'}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        tools=Tools(config(tavily_key='test'),client)
        result,_=await tools.run('weather',{'city':'Нижний Новгород','country_code':'RU'})
        assert '8' in result and 'Нижний Новгород' in result
        result,_=await tools.run('web_search',{'query':'новости'})
        assert 'fresh data' in result and len(calls)==3

async def test_retry_callback_consumed_only_once():
    mem=Memory();await mem.set_user(1,last_request='hello',retry_ready=True)
    llm=SimpleNamespace(answer=AsyncMock(return_value=('answer',[])))
    service=Service(config(),mem,llm,SimpleNamespace(run=AsyncMock()))
    one=update(1);one.callback_query=SimpleNamespace(data='retry',answer=AsyncMock())
    await service.callback(one,None)
    await service.callback(one,None)
    llm.answer.assert_awaited_once()

async def test_6000_synthetic_calls_bounded_concurrency():
    active=peak=0
    async def respond(req):
        nonlocal active,peak
        active+=1;peak=max(peak,active)
        await asyncio.sleep(0)
        active-=1
        return httpx.Response(200,json={'choices':[{'message':{'content':'ok'}}]})
    llm,mem,client=await make_llm(respond)
    async with client:
        results=await asyncio.gather(*(llm.call([{'role':'user','content':'test'}]) for _ in range(6000)))
    assert len(results)==6000 and peak<=2 and len(mem.attempts)==6000
    assert all(x[1]=='ok' for x in mem.results)


def test_config_uses_first_nonempty_existing_key(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEYS', ', key-one, key-two')
    assert Config.load().groq_key=='key-one'

async def test_worker_retries_terminal_write_after_db_outage():
    mem=Memory()
    mem.leader=SimpleNamespace(is_closed=lambda:False)
    mem.claim=AsyncMock(side_effect=[(1,{}),None,None,None])
    mem.finish=AsyncMock(side_effect=[RuntimeError('database temporarily down'),None])
    from telegram import Bot
    app=SimpleNamespace(bot=Bot('123:test'), process_update=AsyncMock(side_effect=RuntimeError('failed')))
    # Use a valid minimal update so failure is in process_update, not JSON parsing.
    mem.claim=AsyncMock(side_effect=[(1,{'update_id':1}),None,None,None])
    service=Service(config(),mem,None,None)
    task=asyncio.create_task(service.worker(app))
    try:
        for _ in range(100):
            if mem.finish.await_count>=2:break
            await asyncio.sleep(.02)
        assert mem.finish.await_count==2
        mem.finish.assert_awaited_with(1,'failed')
    finally:
        task.cancel()
        await asyncio.gather(task,return_exceptions=True)
