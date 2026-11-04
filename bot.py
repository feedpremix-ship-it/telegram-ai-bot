"""Telegram-бот с ИИ на Groq (бесплатно). Погода, курсы, поиск, память, статистика."""
import asyncio
import html
import json
import logging
import os
import re
import time
import sqlite3
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from dotenv import load_dotenv
from telegram import LinkPreviewOptions, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ───────────────────────── Настройки ─────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
GROQ_MODELS = [m.strip() for m in os.getenv("GROQ_MODELS", "deepseek-r1-distill-llama-70b,llama-3.3-70b-versatile").split(",") if m.strip()]

MAX_HISTORY = int(os.getenv("MAX_HISTORY", "40"))
ALLOWED_USER_IDS = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").replace(" ", "").split(",") if x}

DEFAULT_CITY = os.getenv("DEFAULT_CITY", "").strip()
DEFAULT_COUNTRY = os.getenv("DEFAULT_COUNTRY", "RU").upper()

AUTO_SEARCH = os.getenv("AUTO_SEARCH", "1") == "1"
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
SEARCH_DEPTH = os.getenv("SEARCH_DEPTH", "advanced")
MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "60"))
STAGGER_SECONDS = int(os.getenv("STAGGER_SECONDS", "20"))

MEMORY_DB = os.getenv("MEMORY_DB", "./data/bot.db")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
PORT = int(os.getenv("PORT", "10000"))

# ───────────────────────── Промпты ─────────────────────────
STYLE = """
КАК ОТВЕЧАТЬ:
- Общайся как живой, добрый и остроумный друг. Никакого канцелярита.
- Лёгкий юмор уместен, если тема не серьёзная (болезнь, горе, тревога — без шуток).
- Длина по умолчанию: 4–8 коротких строк, не больше 700 символов. Подробнее только если попросили.
- Сначала прямой ответ, потом 2-4 пункта.
- Оформление: заголовок с эмодзи (**жирным**), пункты с эмодзи-иконками. Без таблиц и линий.
- Отвечай на языке пользователя. Не выдумывай факты, цены, ссылки и цитаты.
"""
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Ты — универсальный помощник в Telegram. " + STYLE)
SEARCH_SYSTEM = "Отвечай на вопрос, опираясь на данные из интернета. Если данных нет, честно скажи. Не пересказывай источники." + STYLE
TRANSLATE_SYSTEM = "Переведи текст. Если язык не указан: не на русском → на русский; на русском → на английский. Формат: **🌐 Перевод** \n> текст"

# ───────────────────────── База данных ─────────────────────────
os.makedirs(os.path.dirname(MEMORY_DB) or ".", exist_ok=True)
def _db():
    con = sqlite3.connect(MEMORY_DB, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    return con

def _init_db():
    with _db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER, role TEXT, content TEXT, ts INTEGER);
        CREATE TABLE IF NOT EXISTS profile (chat_id INTEGER PRIMARY KEY, data TEXT);
        CREATE TABLE IF NOT EXISTS quota (key_name TEXT, day TEXT, used INTEGER DEFAULT 0, tokens INTEGER DEFAULT 0, PRIMARY KEY (key_name, day));
        """)
_init_db()

def _load_history_sync(chat_id):
    with _db() as c:
        rows = c.execute("SELECT role, content FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT ?", (chat_id, MAX_HISTORY)).fetchall()
    return [{"role": r, "content": t} for r, t in reversed(rows)]

def _append_message_sync(chat_id, role, content):
    with _db() as c:
        c.execute("INSERT INTO messages(chat_id, role, content, ts) VALUES(?,?,?,?)", (chat_id, role, content[:8000], int(time.time())))
        c.execute("DELETE FROM messages WHERE chat_id=? AND id NOT IN (SELECT id FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT ?)", (chat_id, chat_id, MAX_HISTORY))

async def load_history(chat_id): return await asyncio.to_thread(_load_history_sync, chat_id)
async def append_message(chat_id, role, content): await asyncio.to_thread(_append_message_sync, chat_id, role, content)

def _get_quota_sync():
    with _db() as c:
        return c.execute("SELECT key_name, used, tokens FROM quota WHERE day=?", (datetime.now(timezone.utc).strftime("%Y-%m-%d"),)).fetchall()

async def get_quota(): return await asyncio.to_thread(_get_quota_sync)

def _bump_quota_sync(key_name, tokens=0):
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _db() as c:
        c.execute("INSERT INTO quota(key_name, day, used, tokens) VALUES(?,?,1,?) ON CONFLICT(key_name, day) DO UPDATE SET used=used+1, tokens=tokens+excluded.tokens", (key_name, day, tokens))

async def bump_quota(key_name, tokens=0): await asyncio.to_thread(_bump_quota_sync, key_name, tokens)

# ───────────────────────── Нейросеть ─────────────────────────
CANDIDATES = [{"name": f"groq{i}:{m}", "base": "https://api.groq.com/openai/v1", "key": k, "key_name": f"groq{i}", "model": m} for i, k in enumerate(GROQ_API_KEYS, 1) for m in GROQ_MODELS]
MODEL_SKIP = {}
MODEL_BUSY = {}

def pick_candidates():
    now = time.time()
    ok = [c for c in CANDIDATES if MODEL_SKIP.get(c["name"], 0) <= now] or CANDIDATES
    return sorted(ok, key=lambda c: MODEL_BUSY.get(c["name"], 0) > 0)

async def _call_candidate(c, history, system):
    import httpx
    name, code = c["name"], None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(MODEL_TIMEOUT, connect=10)) as client:
            resp = await client.post(c["base"] + "/chat/completions", headers={"Authorization": "Bearer " + c["key"]}, json={"model": c["model"], "temperature": TEMPERATURE, "messages": [{"role": "system", "content": system or SYSTEM_PROMPT}] + history})
        if resp.status_code >= 400:
            code = resp.status_code
            raise RuntimeError(f"код {code}: {resp.text[:100]}")
        data = resp.json()
        text = (data["choices"][0]["message"].get("content") or "").strip()
        text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.S | re.I).strip()
        if not text: raise RuntimeError("пустой ответ")
        asyncio.create_task(bump_quota(c["key_name"], (data.get("usage") or {}).get("total_tokens", 0)))
        return text
    except Exception as e:
        reason, now = str(e).lower(), time.time()
        if code in (401, 403): MODEL_SKIP[name] = now + 86400
        elif code == 429: MODEL_SKIP[name] = now + (86400 if "day" in reason else 300)
        else: MODEL_SKIP[name] = now + 60
        raise RuntimeError(str(e))

async def ask_chain(history, system=None):
    cands = pick_candidates()
    if not cands: raise RuntimeError("Нет доступных моделей")
    tasks, errors, next_idx = {}, [], 0
    def start_next():
        nonlocal next_idx
        c = cands[next_idx]; next_idx += 1
        MODEL_BUSY[c["name"]] = MODEL_BUSY.get(c["name"], 0) + 1
        task = asyncio.create_task(_call_candidate(c, history, system))
        task.add_done_callback(lambda _t, n=c["name"]: MODEL_BUSY.__setitem__(n, max(0, MODEL_BUSY.get(n, 0) - 1)))
        tasks[task] = c["name"]
    start_next()
    try:
        while tasks:
            done, _ = await asyncio.wait(list(tasks), timeout=STAGGER_SECONDS, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                name = tasks.pop(t)
                try: return t.result()
                except Exception as e: errors.append(f"{name}: {e}")
            if (not done or not tasks) and next_idx < len(cands): start_next()
    finally:
        for t in tasks: t.cancel()
    raise RuntimeError("Модели не ответили")

async def ask(history, system=None): return await ask_chain(history, system)

# ───────────────────────── Оформление ─────────────────────────
NUMS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣"]
NO_PREVIEW = LinkPreviewOptions(is_disabled=True)

def esc(text): return html.escape(str(text or ""), quote=False)
def shorten(text, n): return text if len(text) <= n else text[:n-1] + "…"
def strip_tags(text): return html.unescape(re.sub(r"<[^>]+>", "", text))

def beautify(text):
    out = []
    for line in (text or "").split("\n"):
        m = re.match(r"^(\s*)(\d{1,2})[.)]\s+(.*)$", line)
        if m:
            n = int(m.group(2))
            out.append(f"{m.group(1)}{NUMS[n-1] if 1<=n<=len(NUMS) else f'{n}.'} {m.group(3)}")
        elif line.strip().startswith("- "):
            out.append(line.replace("- ", "🔹 ", 1))
        else:
            out.append(line)
    return "\n".join(out)

def md_to_html(text):
    t = html.escape(text or "", quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)
    return t

async def send_html(msg, text, keyboard=None):
    await msg.reply_text(text, parse_mode=ParseMode.HTML, link_preview_options=NO_PREVIEW, reply_markup=keyboard)

class Working:
    FRAMES = ["🙂 .", "🤔 ..", "🙂 ...", "🤔 ...."]
    def __init__(self, msg): self.msg, self.ack, self.task = msg, None, None
    async def _animate(self):
        i = 0
        while True:
            await asyncio.sleep(2); i += 1
            try: await self.ack.edit_text(self.FRAMES[i % len(self.FRAMES)])
            except: pass
    async def __aenter__(self):
        self.ack = await self.msg.reply_text(self.FRAMES[0])
        self.task = asyncio.create_task(self._animate()); return self
    async def __aexit__(self, *exc):
        if self.task: self.task.cancel()
        try: await self.ack.delete()
        except: pass

# ───────────────────────── Сценарии ─────────────────────────
WEATHER_RE = re.compile(r"погод|прогноз|weather|температура|градус|дожд|зонт", re.I)
TRANSLATE_RE = re.compile(r"перевед|перевод|translate|по-английски|по-русски", re.I)

async def do_weather(msg, text):
    chat_id = msg.chat_id
    # Ищем ЛЮБЫЕ русские буквы в тексте, если нашли - берем слово после "в" или просто все слова, которые не "погода" и т.д.
    words = re.findall(r"[A-Za-zА-Яа-яЁё\-]{3,}", text.lower())
    stop_words = {"погода", "погоду", "какая", "сегодня", "завтра", "прогноз", "скажи", "покажи"}
    city = [w.capitalize() for w in words if w not in stop_words]
    
    if not city:
        await msg.reply_text("🌍 Для какого города показать погоду?")
        return

    # Пытаемся искать каждый вариант (на случай, если город из двух слов)
    for c in city:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": c, "count": 1, "language": "ru"})
                res = r.json().get("results")
                if res:
                    geo = res[0]
                    r2 = await client.get("https://api.open-meteo.com/v1/forecast", params={
                        "latitude": geo["latitude"], "longitude": geo["longitude"],
                        "current": "temperature_2m,weather_code", "timezone": "auto"
                    })
                    cur = r2.json()["current"]
                    card = f"🌍 <b>Погода: {esc(geo['name'])}</b>\n\n🌡 Сейчас: <b>{round(cur['temperature_2m'])}°C</b>"
                    await send_html(msg, card)
                    return
        except Exception:
            pass
    
    await msg.reply_text("🤔 Не смог найти этот город на карте. Попробуй написать иначе.")

async def do_translate(msg, text):
    rest = TRANSLATE_RE.sub("", text).strip()
    if len(rest) < 2:
        await msg.reply_text("🌐 Какой текст перевести?")
        return
    async with Working(msg):
        answer = await ask([{"role": "user", "content": text}], TRANSLATE_SYSTEM)
    await send_html(msg, md_to_html(answer))
    await append_message(msg.chat_id, "user", text)
    await append_message(msg.chat_id, "assistant", answer)

async def do_chat(msg, text):
    chat_id = msg.chat_id
    history = await load_history(chat_id)
    async with Working(msg):
        try:
            answer = await ask(history + [{"role": "user", "content": text}])
        except Exception as e:
            await msg.reply_text(f"😔 Ошибка: {str(e)[:100]}")
            return
        answer = beautify(answer)
    await send_html(msg, md_to_html(answer))
    await append_message(chat_id, "user", text)
    await append_message(chat_id, "assistant", answer)

# ───────────────────────── Бот ─────────────────────────
HELP_TEXT = "👋 <b>Привет! Я твой карманный всезнайка.</b>\n\n🌤 <b>Погода</b> — «погода в Москве»\n🌐 <b>Перевод</b> — «переведи на английский: привет»\n💬 <b>Разговор</b> — просто напиши вопрос\n📊 <b>/stats</b> — расход"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_html(update.message, HELP_TEXT)

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if ALLOWED_USER_IDS and update.effective_user.id not in ALLOWED_USER_IDS: return
    rows = await get_quota()
    lines = ["📊 <b>Статистика за сегодня</b>", ""]
    if not rows: lines.append("Пока ни одного запроса.")
    for key_name, used, tokens in rows:
        lines.append(f"🔑 <b>{esc(key_name)}</b>: {used} запросов, {tokens} токенов")
    await send_html(update.message, "\n".join(lines))

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.text: return
    if ALLOWED_USER_IDS and update.effective_user.id not in ALLOWED_USER_IDS:
        await msg.reply_text("Извините, этот бот закрытый."); return

    text = msg.text.strip()
    try:
        if WEATHER_RE.search(text):
            await do_weather(msg, text)
        elif TRANSLATE_RE.search(text):
            await do_translate(msg, text)
        else:
            await do_chat(msg, text)
    except Exception as e:
        logging.exception("Ошибка")
        await msg.reply_text(f"😔 Что-то пошло не так: {str(e)[:100]}")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).concurrent_updates(True).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    if WEBHOOK_URL:
        from aiohttp import web
        async def handler(request):
            data = await request.json()
            update = Update.de_json(data, app.bot)
            await app.update_queue.put(update)
            return web.Response(text="ok")
        async def run():
            await app.initialize()
            await app.bot.set_webhook(WEBHOOK_URL, allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
            await app.start()
            server = web.Application()
            server.router.add_post("/webhook", handler)
            runner = web.AppRunner(server)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", PORT)
            await site.start()
            logging.info("Webhook на порту %s", PORT)
            await asyncio.Event().wait()
        asyncio.run(run())
    else:
        app.run_polling()

if __name__ == "__main__":
    main()