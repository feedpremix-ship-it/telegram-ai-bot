"""Render Free Telegram assistant. Start: python bot.py. Secrets only in Environment."""
import asyncio
import contextlib
import hmac
import io
import logging
import signal

import httpx
from aiohttp import web
from dotenv import load_dotenv
from telegram import InlineKeyboardButton as Button, InlineKeyboardMarkup, LinkPreviewOptions, ReplyKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import TelegramError
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters

from assistant.config import Config
from assistant.llm import LLM, UserError
from assistant.output import clean_links, links_allowed, messages
from assistant.store import Store
from assistant.tools import Tools

LOG = logging.getLogger('assistant')
MENU = ReplyKeyboardMarkup([
    ['💬 Новый диалог', '🧠 Память'], ['🧹 Очистить историю', '📊 Возможности'], ['⚙️ Настройки', '❓ Помощь']
], resize_keyboard=True)
HELP = ('Я помогу с вопросами, текстами, переводами, программированием и математикой. '
        'Короткий вопрос — короткий ответ, сложная задача — подробное объяснение.\n\n'
        'Примеры: «Переведи на английский: хорошая погода», «Построй y=sin(x) от -10 до 10», '
        '«Найди свежие новости науки» (если настроен поиск).\n'
        '/weather Москва, Россия — текущая погода; /translate текст — перевод; /search запрос — поиск.\n'
        '/new — новый диалог; /clear — удалить историю; /memory — память; /settings — длина ответа.\n\n'
        'Не присылайте пароли и секреты. Текст запроса передаётся Groq, поисковые запросы — Tavily, '
        'город — Open-Meteo. История хранится во внешней базе, если память включена. '
        'Последний запрос сохраняется для повтора даже при выключенной памяти; /clear удаляет и его. '
        'Ссылки показываю только по прямой просьбе; их доступность автоматически не проверяется. '
        'Нейросетевой генерации изображений нет. Бесплатный сервис может засыпать и отвечать с задержкой.')
RETRY = InlineKeyboardMarkup([[Button('🔄 Повторить', callback_data='retry')]])


def bounded_history(history, limit):
    history = list(history[-30:])
    while history and (history[0].get('role') != 'user' or sum(len(x['content']) for x in history) > limit):
        history.pop(0)
    return history


class Service:
    def __init__(self, cfg, store, llm, tools):
        self.cfg, self.store, self.llm, self.tools = cfg, store, llm, tools
        self.ready = False

    def authorized(self, update):
        return (update.effective_user is not None and update.effective_chat is not None
                and update.effective_chat.type == 'private'
                and update.effective_chat.id == update.effective_user.id
                and (not self.cfg.allowed or update.effective_user.id in self.cfg.allowed))

    async def send(self, msg, text, allow_links=False, keyboard=None):
        for chunk in messages(text, allow_links):
            await msg.reply_text(chunk, parse_mode=ParseMode.HTML,
                                 link_preview_options=LinkPreviewOptions(is_disabled=True), reply_markup=keyboard)

    async def menu(self, update, context=None, action=None):
        if not self.authorized(update):
            return
        msg, uid = update.effective_message, update.effective_user.id
        action = action or msg.text.split()[0].split('@')[0]
        if action in ('/new', '/clear', '💬 Новый диалог', '🧹 Очистить историю'):
            await self.store.set_user(uid, history=[], last_request='', last_answer='', retry_ready=False)
            await self.send(msg, 'История и сохранённый запрос удалены. Начнём новый диалог.', keyboard=MENU)
        elif action in ('/memory', '🧠 Память'):
            user = await self.store.user(uid)
            keyboard = InlineKeyboardMarkup([[Button('Выключить и удалить историю' if user['memory'] else 'Включить память', callback_data='memory')]])
            await self.send(msg, f"Память {'включена' if user['memory'] else 'выключена'}. Сообщений в контексте: {len(user['history'])}.\n"
                            'Память хранится в PostgreSQL. Последний запрос для повтора удаляется через /clear.', keyboard=keyboard)
        elif action in ('/settings', '⚙️ Настройки'):
            await self.send(msg, 'Выберите длину ответа:', keyboard=InlineKeyboardMarkup([
                [Button('Авто', callback_data='detail:auto'), Button('Кратко', callback_data='detail:short'), Button('Подробно', callback_data='detail:long')]]))
        elif action in ('/stats',):
            if uid not in self.cfg.admins:
                await self.send(msg, 'Статистика доступна администратору.')
                return
            row = await self.store.stats()
            await self.send(msg, f"Сегодня UTC: API-попыток {row.get('attempts',0)}, успешных {row.get('successes',0)}, "
                            f"ошибок {row.get('errors',0)}, учтённых токенов {row.get('tokens',0)}. "
                            f"Незавершённых/неопределённых {row.get('attempts',0)-row.get('successes',0)-row.get('errors',0)}. "
                            f"Консервативно зарезервировано {row.get('reserved',0)} токенов. "
                            'Незавершённые попытки также занимают резерв. Данные не заменяют биллинг провайдера.')
        elif action == '📊 Возможности':
            await self.send(msg, '💬 Вопросы и тексты\n🧠 Память отдельных личных диалогов\n🌐 Перевод\n'
                            '🌤 Текущая погода по уточнённому городу\n🔎 Поиск при наличии Tavily\n'
                            '📈 Графики функций, столбчатые и круговые диаграммы\n'
                            'Результаты ИИ могут содержать ошибки; проверяйте важные сведения.', keyboard=MENU)
        else:
            await self.send(msg, HELP, keyboard=MENU)

    async def callback(self, update, context):
        if not self.authorized(update):
            if update.callback_query:
                await update.callback_query.answer('Доступен только ваш личный диалог.', show_alert=True)
            return
        query, uid = update.callback_query, update.effective_user.id
        try:
            await query.answer()
        except TelegramError:
            pass  # cold-start callbacks can outlive Telegram's answer window
        if query.data == 'memory':
            user = await self.store.user(uid)
            await self.store.set_user(uid, memory=not user['memory'], history=[], last_request='', last_answer='', retry_ready=False)
            await self.menu(update, action='/memory')
        elif query.data in ('detail:auto','detail:short','detail:long'):
            await self.store.set_user(uid, detail=query.data.split(':')[1])
            await self.send(update.effective_message, 'Настройка сохранена.')
        elif query.data == 'retry':
            user = await self.store.user(uid)
            if not user['retry_ready'] or not user['last_request']:
                await self.send(update.effective_message, 'Нет запроса для повтора или он уже обрабатывается.')
                return
            await self.chat(update, user['last_request'], retry=True)
        else:
            await self.send(update.effective_message, 'Эта кнопка больше не поддерживается.')

    async def on_message(self, update, context):
        if not self.authorized(update):
            return
        text = (update.effective_message.text or '').strip()
        if text in {v for row in MENU.keyboard for v in [x.text for x in row]}:
            await self.menu(update, action=text)
            return
        await self.chat(update, text)

    async def tool_command(self, update, context):
        if not self.authorized(update):
            return
        command, _, text = update.effective_message.text.partition(' ')
        if not text.strip():
            await self.send(update.effective_message, 'После команды напишите город, текст для перевода или поисковый запрос.')
            return
        prefixes = {'/weather':'Узнай текущую погоду для города: ', '/translate':'Переведи следующий текст (если язык не указан: русский на английский, другой на русский): ', '/search':'Выполни поиск в интернете по запросу: '}
        await self.chat(update, prefixes[command.split('@')[0]] + text)

    async def typing(self, msg):
        while True:
            with contextlib.suppress(TelegramError):
                await msg.chat.send_action(ChatAction.TYPING)
            await asyncio.sleep(4)

    async def chat(self, update, text, retry=False):
        if not text or len(text) > 6000:
            await self.send(update.effective_message, 'Отправьте текст длиной от 1 до 6000 символов.')
            return
        uid, msg = update.effective_user.id, update.effective_message
        if not await self.store.admit(uid):
            await self.send(msg, 'Слишком много запросов: подождите несколько секунд или до обновления дневного лимита.')
            return
        user = await self.store.user(uid)
        if retry and user['last_answer']:
            # Delivery retry reuses saved output; no second paid/token-consuming LLM request.
            try:
                await self.send(msg, user['last_answer'], links_allowed(text))
                await self.store.set_user(uid, retry_ready=False)
            except TelegramError:
                await self.store.set_user(uid, retry_ready=True)
            return
        await self.store.set_user(uid, last_request=text, last_answer='', retry_ready=False)
        typing = asyncio.create_task(self.typing(msg))
        try:
            history = bounded_history(user['history'], self.cfg.context_chars) if user['memory'] else []
            answer, pictures = await self.llm.answer(history, text, user['detail'], self.tools.run, links_allowed(text))
            answer = clean_links(answer, links_allowed(text))
            new_history = bounded_history(history + [{'role':'user','content':text}, {'role':'assistant','content':answer}], self.cfg.context_chars)
            await self.store.set_user(uid, history=new_history if user['memory'] else [], last_answer=answer)
            for picture in pictures:
                with io.BytesIO(picture) as image:
                    image.name = 'chart.png'
                    await msg.reply_photo(image, caption='График построен по заданным данным.')
            await self.send(msg, answer, links_allowed(text))
        except UserError as exc:
            await self.store.set_user(uid, retry_ready=True)
            await self.send(msg, str(exc), keyboard=RETRY)
        except TelegramError:
            await self.store.set_user(uid, retry_ready=True)
            with contextlib.suppress(TelegramError):
                await self.send(msg, 'Не удалось полностью доставить ответ. Повтор использует сохранённый текст; график нужно запросить заново.', keyboard=RETRY)
        except Exception:
            LOG.error('request_failed user_id=%s; diagnostic values suppressed', uid)
            await self.store.set_user(uid, retry_ready=True)
            await self.send(msg, 'Не удалось обработать запрос. Повторите позже.', keyboard=RETRY)
        finally:
            typing.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing

    async def error(self, update, context):
        LOG.error('update_failed type=%s; details suppressed', type(context.error).__name__)
        if isinstance(update, Update) and self.authorized(update):
            with contextlib.suppress(TelegramError):
                await self.send(update.effective_message, 'Временная ошибка. Попробуйте позже.')

    async def worker(self, app):
        while True:
            item = None
            try:
                if self.store.leader.is_closed():
                    self.ready = False
                    await asyncio.sleep(1)
                    continue
                item = await self.store.claim()
                if not item:
                    await asyncio.sleep(.25)
                    continue
                update_id, data = item
                await app.process_update(Update.de_json(data, app.bot))
                await self.store.finish(update_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.error('worker_failed; details suppressed')
                if item:
                    # Do not lose the terminal transition during a transient DB outage.
                    while True:
                        try:
                            await self.store.finish(item[0], 'failed')
                            break
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            await asyncio.sleep(1)
                await asyncio.sleep(1)


def make_application(cfg, service):
    app = Application.builder().token(cfg.token).concurrent_updates(False).build()
    app.add_handler(CommandHandler(['start','help','new','clear','memory','settings','stats'], service.menu))
    app.add_handler(CommandHandler(['weather','translate','search'], service.tool_command))
    app.add_handler(CallbackQueryHandler(service.callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, service.on_message))
    app.add_error_handler(service.error)
    return app


def make_web_app(service):
    server = web.Application(client_max_size=128*1024)
    async def health(request):
        try:
            if not service.ready or service.store.leader.is_closed():
                raise RuntimeError('not ready')
            await service.store.pool.fetchval('SELECT 1')
            return web.json_response({'status':'ok'})
        except Exception:
            return web.json_response({'status':'unavailable'}, status=503)
    async def webhook(request):
        token = request.headers.get('X-Telegram-Bot-Api-Secret-Token', '')
        if not service.cfg.webhook_secret or not hmac.compare_digest(token.encode(), service.cfg.webhook_secret.encode()):
            return web.Response(status=403)
        if not service.ready:
            return web.Response(status=503)
        try:
            data = await request.json()
            if not isinstance(data, dict) or type(data.get('update_id')) is not int:
                return web.Response(status=400)
            accepted = await service.store.enqueue(data)
        except (ValueError, TypeError, AttributeError):
            return web.Response(status=400)
        except Exception:
            return web.Response(status=503)
        return web.Response(text='ok', status=200 if accepted else 503)
    server.router.add_get('/', health)
    server.router.add_get('/healthz', health)
    server.router.add_post('/webhook', webhook)
    return server


async def poll(app, store):
    offset = None
    while True:
        try:
            updates = await app.bot.get_updates(offset=offset, timeout=25, allowed_updates=['message','callback_query'])
            for update in updates:
                while not await store.enqueue(update.to_dict()):
                    await asyncio.sleep(1)
                offset = update.update_id+1
        except asyncio.CancelledError:
            raise
        except TelegramError:
            LOG.error('polling_unavailable; details suppressed')
            await asyncio.sleep(5)


async def run(cfg):
    cfg.validate()
    store = await Store.open(cfg)
    tasks, runner, app = [], None, None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(55, connect=10), limits=httpx.Limits(max_connections=8), follow_redirects=False) as client:
            llm = LLM(cfg, store, client)
            await llm.verify_models()
            service = Service(cfg, store, llm, Tools(cfg, client))
            app = make_application(cfg, service)
            await app.initialize()
            await app.start()
            runner = web.AppRunner(make_web_app(service))
            await runner.setup()
            await web.TCPSite(runner, '0.0.0.0', cfg.port).start()
            if cfg.webhook_url:
                await app.bot.set_webhook(cfg.webhook_url, secret_token=cfg.webhook_secret, allowed_updates=['message','callback_query'],
                                          max_connections=1, drop_pending_updates=False)
            else:
                await app.bot.delete_webhook(drop_pending_updates=False)
                tasks.append(asyncio.create_task(poll(app, store)))
            await store.prune()
            async def maintenance():
                while True:
                    await asyncio.sleep(3600)
                    await store.prune()
            tasks.append(asyncio.create_task(maintenance()))
            tasks.extend(asyncio.create_task(service.worker(app)) for _ in range(cfg.workers))
            service.ready = True
            LOG.info('ready mode=%s workers=%s', 'webhook' if cfg.webhook_url else 'polling', cfg.workers)
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                with contextlib.suppress(NotImplementedError):
                    loop.add_signal_handler(sig, stop.set)
            await stop.wait()
            service.ready = False
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if runner:
            await runner.cleanup()
        if app:
            if app.running:
                await app.stop()
            await app.shutdown()
        await store.close()


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    for name in ('httpx','httpcore','telegram','asyncpg'):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    try:
        asyncio.run(run(Config.load()))
    except (ValueError, UserError) as exc:
        LOG.error('%s', exc)
        raise SystemExit(1) from None
    except Exception:
        LOG.error('startup_failed: проверьте Environment, базу и сетевой доступ. Секреты и детали подключения скрыты.')
        raise SystemExit(1) from None

if __name__ == '__main__':
    main()
