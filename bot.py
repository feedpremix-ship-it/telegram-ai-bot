"""Telegram-бот с ИИ на Groq (бесплатно). Погода, курсы, поиск, память, статистика, тихий час."""
import asyncio
import html
import json
import logging
import os
import math
import re
import time
import sqlite3
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus, urlparse

from dotenv import load_dotenv
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ───────────────────────── Настройки (.env) ─────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")

GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
GROQ_MODELS = [m.strip() for m in os.getenv(
    "GROQ_MODELS",
    "deepseek-r1-distill-llama-70b,llama-3.3-70b-versatile,llama-3.1-8b-instant",
).split(",") if m.strip()]

MAX_HISTORY = int(os.getenv("MAX_HISTORY", "40"))
ALLOWED_USER_IDS = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").replace(" ", "").split(",") if x}

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "").strip()
DEFAULT_COUNTRY = os.getenv("DEFAULT_COUNTRY", "RU").upper()
DEFAULT_CURRENCIES = [c.strip().upper() for c in os.getenv("DEFAULT_CURRENCIES", "USD,EUR,CNY,GBP,TRY,KZT").split(",") if c.strip()]

AUTO_SEARCH = os.getenv("AUTO_SEARCH", "1") == "1"
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
SEARCH_DEPTH = os.getenv("SEARCH_DEPTH", "advanced")
MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "60"))
STAGGER_SECONDS = int(os.getenv("STAGGER_SECONDS", "20"))

MEMORY_DB = os.getenv("MEMORY_DB", "./data/bot.db")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
PORT = int(os.getenv("PORT", "7860"))

TZ_NAME = os.getenv("TZ", "Europe/Moscow")
try:
    import zoneinfo
    TZINFO = zoneinfo.ZoneInfo(TZ_NAME)
except Exception:
    TZINFO = None

_quiet = os.getenv("QUIET_HOURS", "").strip()
QUIET_HOURS = None
if _quiet and "-" in _quiet:
    try:
        a, b = _quiet.split("-", 1)
        QUIET_HOURS = (int(a), int(b))
    except ValueError:
        pass

# ───────────────────────── Инструкции для ИИ ─────────────────────────
STYLE = """
КАК ОТВЕЧАТЬ (очень важно):
- Ты общаешься как живой, добрый и остроумный друг, который знает всё на свете. Никакого канцелярита, казённых фраз и занудства. Пиши по-человечески и коротко.
- Лёгкий юмор: одна-две уместные шутки или игривые фразы на ответ, не больше. Если тема серьёзная или человеку плохо, юмор полностью убери: будь тёплым, внимательным и спокойным.
- Показывай участие: сначала отреагируй на слова человека одной короткой живой фразой, потом давай суть. Если без уточнения не обойтись, задай ОДИН короткий вопрос в конце.
- Объясняй так, чтобы понял ребёнок: простые слова, короткие предложения, примеры из жизни.
- Отвечай по делу, без воды и лирики. Длина по умолчанию: 4–8 коротких строк, не больше 700 символов. Подробнее только если попросили «подробно».
- Сначала прямой ответ на вопрос в первой строке («Да, есть», «Нет», «Около 5 км»), потом 2–4 коротких пункта.
- Оформление: короткий заголовок с эмодзи (**жирным**), затем пункты. Каждый пункт на отдельной строке и начинается с подходящей эмодзи-иконки. Главное выделяй **жирным**. Между блоками пустая строка.
- ЗАПРЕЩЕНО: таблицы, символы | и ---, линии из чёрточек или других символов, HTML, длинные абзацы.
- Разрешён только Markdown: **жирный**, *курсив*, списки, [текст](ссылка).
- Пиши грамотно: никаких опечаток, орфографических и пунктуационных ошибок.
- Отвечай на языке пользователя. Не выдумывай факты, цены, ссылки и цитаты: если не знаешь или данных нет, скажи честно.
- Не ставь в тексте номера источников в скобках вроде [1] и не вставляй ссылки, пока тебя об этом не попросили.
- Пиши только русскими буквами (названия брендов и мест допустимо латиницей). Никаких иероглифов и смешения алфавитов.

ПРИМЕРЫ ХОРОШЕГО И ПЛОХОГО СТИЛЯ:
Плохо (канцелярит): «Для получения информации о погоде необходимо указать населённый пункт».
Хорошо: «Скажи город — гляну, что там с погодой 🙂»
Плохо: «В соответствии с действующим законодательством...»
Хорошо: «По закону это можно, но есть нюанс — расскажу на пальцах.»
"""

DEFAULT_SYSTEM = (
    "Ты — универсальный собеседник и помощник в Telegram. Ты разбираешься почти во всём: наука, школа, "
    "работа, деньги, здоровье, право, техника, путешествия, творчество, быт. Можешь быть другом, "
    "советчиком, наставником и внимательным слушателем. Если человек делится переживаниями, сначала "
    "выслушай и поддержи, помоги разобраться в чувствах простыми вопросами. Ты не заменяешь врача, "
    "юриста или психотерапевта. Если человеку очень плохо, отнесись к этому серьёзно и бережно: предложи "
    "позвонить в экстренную службу (112) или рассказать близкому. Ты помнишь прошлые разговоры. "
    "Если сообщение начинается с «[Собеседник отвечает на сообщение: ...]», вопрос относится именно к этому сообщению. "
    "Если вопрос короткий («а почему?», «а завтра?»), считай, что он продолжает предыдущую тему. "
    "Если пользователь прислал текст на иностранном языке без вопроса, переведи его на русский." + STYLE
)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM)

SEARCH_SYSTEM = (
    "Ты — помощник, который отвечает на вопрос, опираясь на свежие данные из интернета, "
    "приведённые в сообщении пользователя. Если данные противоречат друг другу, скажи об этом. "
    "Если в данных нет ответа, честно скажи, что не нашёл. Данные нужны только для проверки фактов: не пересказывай их, не копируй фрагменты и не перечисляй найденные страницы. Не ставь в тексте номера источников и "
    "ссылки: ответ должен быть чистым текстом." + STYLE
)

TRANSLATE_SYSTEM = (
    "Ты — профессиональный переводчик. Определи язык текста и переведи его. Если пользователь "
    "указал язык перевода, переводи на него. Если не указал: текст не на русском → переведи на "
    "русский; текст на русском → переведи на английский.\n"
    "Формат ответа:\n"
    "**🌐 Перевод: язык оригинала → язык перевода**\n\n"
    "> сам перевод (каждая строка начинается с «> »)\n\n"
    "Если есть тонкости (идиома, вежливая или грубая форма), добавь короткое "
    "«💡 Пояснение». Больше ничего не добавляй. Язык оригинала и перевода называй по-русски."
)

NO_DATA_NOTE = (
    "\nСвежих данных из интернета найти не удалось. Не называй конкретных цифр, ставок, тарифов, статей и дат: "
    "скажи, что точных данных нет, дай общий порядок величин только с пометкой «ориентир» и подскажи, где проверить."
)

SHORTEN_SYSTEM = (
    "Сократи ответ до 6–8 коротких строк (не больше 700 символов). Сохрани главное, простой язык, эмодзи-иконки "
    "и оформление. Ничего не добавляй от себя. Верни только сокращённый текст."
)

PROOF_SYSTEM = (
    "Ты — корректор русского языка. Исправь в тексте орфографические, пунктуационные и "
    "грамматические ошибки и опечатки. Смысл, структуру, эмодзи, разметку и порядок строк не меняй. Верни только исправленный текст."
)

# ───────────────────────── Состояние и память ─────────────────────────
os.makedirs(os.path.dirname(MEMORY_DB) or ".", exist_ok=True)

def _db():
    con = sqlite3.connect(MEMORY_DB, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con

def _init_db():
    with _db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ts INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_msg_chat ON messages(chat_id, id);
        CREATE TABLE IF NOT EXISTS profile (
            chat_id INTEGER PRIMARY KEY,
            data TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS quota (
            key_name TEXT NOT NULL,
            day TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0,
            tokens INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (key_name, day)
        );
        """)
_init_db()

def _load_history_sync(chat_id):
    with _db() as c:
        rows = c.execute(
            "SELECT role, content FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT ?",
            (chat_id, MAX_HISTORY),
        ).fetchall()
    return [{"role": r, "content": t} for r, t in reversed(rows)]

def _append_message_sync(chat_id, role, content):
    with _db() as c:
        c.execute(
            "INSERT INTO messages(chat_id, role, content, ts) VALUES(?,?,?,?)",
            (chat_id, role, content[:8000], int(time.time())),
        )
        c.execute(
            "DELETE FROM messages WHERE chat_id=? AND id NOT IN "
            "(SELECT id FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT ?)",
            (chat_id, chat_id, MAX_HISTORY),
        )

async def load_history(chat_id):
    return await asyncio.to_thread(_load_history_sync, chat_id)

async def append_message(chat_id, role, content):
    await asyncio.to_thread(_append_message_sync, chat_id, role, content)

def _load_profile_sync(chat_id):
    with _db() as c:
        row = c.execute("SELECT data FROM profile WHERE chat_id=?", (chat_id,)).fetchone()
    return json.loads(row[0]) if row else {}

def _save_profile_sync(chat_id, data):
    with _db() as c:
        c.execute(
            "INSERT INTO profile(chat_id, data) VALUES(?,?) "
            "ON CONFLICT(chat_id) DO UPDATE SET data=excluded.data",
            (chat_id, json.dumps(data, ensure_ascii=False)),
        )

async def load_profile(chat_id):
    return await asyncio.to_thread(_load_profile_sync, chat_id)

async def save_profile(chat_id, data):
    await asyncio.to_thread(_save_profile_sync, chat_id, data)

PROFILE = {}

async def get_profile(chat_id):
    if chat_id not in PROFILE:
        PROFILE[chat_id] = await load_profile(chat_id)
    return PROFILE[chat_id]

QUOTA_LOCK = asyncio.Lock()

def _today_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _bump_quota_sync(key_name, tokens=0):
    day = _today_utc()
    with _db() as c:
        c.execute(
            "INSERT INTO quota(key_name, day, used, tokens) VALUES(?,?,1,?) "
            "ON CONFLICT(key_name, day) DO UPDATE SET "
            "used=used+1, tokens=tokens+excluded.tokens",
            (key_name, day, tokens),
        )

async def bump_quota(key_name, tokens=0):
    async with QUOTA_LOCK:
        await asyncio.to_thread(_bump_quota_sync, key_name, tokens)

def _get_quota_sync():
    day = _today_utc()
    with _db() as c:
        rows = c.execute(
            "SELECT key_name, used, tokens FROM quota WHERE day=? ORDER BY key_name",
            (day,),
        ).fetchall()
    return rows

async def get_quota():
    return await asyncio.to_thread(_get_quota_sync)

PENDING = {}
PENDING_TTL = 180
CONTEXT = {}

def _system(system):
    return f"{system or SYSTEM_PROMPT}\nСегодняшняя дата: {datetime.now():%d.%m.%Y}."

CANDIDATES = []
for i, key in enumerate(GROQ_API_KEYS, 1):
    for model in GROQ_MODELS:
        CANDIDATES.append({
            "name": f"groq{i}:{model}",
            "base": "https://api.groq.com/openai/v1",
            "key": key,
            "key_name": f"groq{i}",
            "model": model,
        })

MODEL_SKIP = {}
MODEL_BUSY = {}

def _next_utc_midnight():
    now = datetime.now(timezone.utc)
    return (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).timestamp()

def pick_candidates():
    now = time.time()
    ok = [c for c in CANDIDATES if MODEL_SKIP.get(c["name"], 0) <= now] or CANDIDATES
    return sorted(ok, key=lambda c: MODEL_BUSY.get(c["name"], 0) > 0)

async def _call_candidate(c, history, system):
    import httpx
    name, code = c["name"], None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(MODEL_TIMEOUT, connect=10)) as client:
            resp = await client.post(
                c["base"] + "/chat/completions",
                headers={"Authorization": "Bearer " + c["key"]},
                json={
                    "model": c["model"],
                    "temperature": TEMPERATURE,
                    "messages": [{"role": "system", "content": _system(system)}] + history,
                },
            )
        if resp.status_code >= 400:
            code = resp.status_code
            raise RuntimeError(f"код {code}: {resp.text[:200]}")
        data = resp.json()
        text = (data["choices"][0]["message"].get("content") or "").strip()
        text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.S | re.I).strip()
        if not text:
            raise RuntimeError("пустой ответ")
        tokens = (data.get("usage") or {}).get("total_tokens") or 0
        asyncio.create_task(bump_quota(c["key_name"], tokens))
        return text
    except asyncio.CancelledError:
        raise
    except Exception as e:
        reason = (str(e) or type(e).__name__)
        low, now = reason.lower(), time.time()
        if code in (401, 403):
            MODEL_SKIP[name] = now + 24 * 3600
        elif code == 429:
            daily = any(w in low for w in ("per day", "daily", "tokens per day", "tpd"))
            MODEL_SKIP[name] = _next_utc_midnight() if daily else now + 300
        else:
            MODEL_SKIP[name] = now + 60
        logging.warning("Модель %s не ответила (%s)", name, reason[:110])
        raise RuntimeError(reason)

async def ask_chain(history, system=None):
    cands = pick_candidates()
    if not cands:
        raise RuntimeError("Не настроен ни один ИИ-сервис")
    tasks, errors, next_idx = {}, [], 0
    deadline = time.time() + 2.5 * MODEL_TIMEOUT

    def start_next():
        nonlocal next_idx
        c = cands[next_idx]
        next_idx += 1
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
                try:
                    return t.result()
                except Exception as e:
                    errors.append(f"{name}: {e}")
            if time.time() > deadline:
                break
            if (not done or not tasks) and next_idx < len(cands):
                start_next()
    finally:
        for t in tasks:
            t.cancel()
    raise RuntimeError("Ни одна модель не ответила. " + (errors[-1][:250] if errors else "время ожидания вышло"))

async def ask(history, system=None):
    return await ask_chain(history, system)

async def polish(text):
    if not text or len(text) < 500:
        return text
    try:
        fixed = await ask([{"role": "user", "content": text}], PROOF_SYSTEM)
        if fixed and 0.6 * len(text) <= len(fixed) <= 1.5 * len(text):
            return fixed
    except Exception:
        pass
    return text

GARBAGE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\u0600-\u06ff]")
MIXED_RE = re.compile(r"\b(?=\w*[А-Яа-яЁё])(?=\w*[A-Za-z])\w+\b")
CITE_RE = re.compile(r"\s*\[\d+(?:[,\s–-]+\d+)*\](?!\()")
LINKS_RE = re.compile(r"ссылк|источник|откуда (?:инфа|информация|данные)|где почитать|покажи сайт", re.I)

async def ask_checked(history, system=None, user_text=""):
    answer = await ask(history, system)
    bad = (GARBAGE_RE.search(answer) or MIXED_RE.search(answer)) and not GARBAGE_RE.search(user_text)
    if bad:
        try:
            retry = await ask(
                history,
                (system or SYSTEM_PROMPT) + "\nВАЖНО: перепиши ответ грамотно, ТОЛЬКО русскими буквами. Никаких иероглифов.",
            )
            if retry:
                answer = retry
        except Exception:
            pass
        answer = GARBAGE_RE.sub("", answer)
    return CITE_RE.sub("", answer)

def footer_notes(question, answer):
    notes = []
    if re.search(r"\b(?:НК|ГК|УК|КоАП|ТК|СК|ЖК)\s*РФ", answer, re.I):
        notes.append("⚖️ Номера статей и формулировки законов проверьте по официальному тексту (КонсультантПлюс).")
    if re.search(r"вклад|депозит|кредит|ипотек|налог|ндфл|доходност", question, re.I):
        notes.append("💡 Это общая информация, а не финансовая консультация. Проверяйте на сайте банка или ФНС.")
    return ("\n\n" + "\n\n".join(notes)) if notes else ""

def special_reply(text):
    if re.search(r"обойд|обход|bypass|взлом", text, re.I) and re.search(r"капч|captcha|cloudflare|парол", text, re.I):
        return "🙅 С обходом защиты сайтов я не помогу. Давай поищем официальный API или ручной способ."
    return None

async def tavily_search(query, topic="general", max_results=5, depth="basic"):
    import httpx
    headers = {"Content-Type": "application/json"}
    if TAVILY_API_KEY:
        headers["Authorization"] = "Bearer " + TAVILY_API_KEY
    else:
        headers["X-Tavily-Access-Mode"] = "keyless"
    body = {"query": query, "max_results": max_results, "search_depth": depth, "topic": topic}
    async with httpx.AsyncClient(timeout=45) as client:
        resp = await client.post("https://api.tavily.com/search", headers=headers, json=body)
        if resp.status_code in (400, 401, 403, 422) and depth == "advanced":
            body["search_depth"] = "basic"
            resp = await client.post("https://api.tavily.com/search", headers=headers, json=body)
    if resp.status_code >= 400:
        raise RuntimeError(f"Поиск ответил кодом {resp.status_code}")
    return resp.json().get("results") or []

# ───────────────────────── Оформление ─────────────────────────
NUMS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣"]
NO_PREVIEW = LinkPreviewOptions(is_disabled=True)

def esc(text):
    return html.escape(str(text or ""), quote=False)

def shorten(text, n):
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"

def domain(url):
    try:
        return urlparse(url).netloc.removeprefix("www.")
    except Exception:
        return ""

TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")

def convert_tables(text):
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        if TABLE_ROW.match(lines[i]):
            block = []
            while i < len(lines) and TABLE_ROW.match(lines[i]):
                block.append(lines[i])
                i += 1
            out.append("(таблица скрыта для удобства)")
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)

EMOJI_START_RE = re.compile(r"[\u2190-\u21FF\u2300-\u23FF\u2460-\u27BF\u2B00-\u2BFF\U0001F000-\U0001FAFF]")

def _starts_with_emoji(line):
    line = re.sub(r"^[\s*#>_`~\-•]+", "", line)
    return bool(line) and (bool(EMOJI_START_RE.match(line[0])) or (line[0].isdigit() and "\u20e3" in line[:3]))

def beautify(text):
    out, in_code, first_done = [], False, False
    for line in (text or "").split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code or not line.strip():
            out.append(line)
            continue
        m = re.match(r"^(\s*)(\d{1,2})[.)]\s+(.*)$", line)
        if m:
            n = int(m.group(2))
            out.append(f"{m.group(1)}{NUMS[n - 1] if 1 <= n <= len(NUMS) else f'{n}.'} {m.group(3)}")
            first_done = True
            continue
        m = re.match(r"^(\s*)[-*•]\s+(.*)$", line)
        if m:
            body = m.group(2)
            out.append(f"{m.group(1)}{'' if _starts_with_emoji(body) else '🔹 '}{body}")
            first_done = True
            continue
        if not first_done:
            first_done = True
            if not _starts_with_emoji(line) and len(line) < 120 and not line.lstrip().startswith(("|", ">")):
                out.append("💬 " + line)
                continue
        out.append(line)
    return "\n".join(out)

def md_to_html(text):
    codes = []
    def stash(m):
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"
    text = re.sub(r"```[A-Za-z0-9_+#.\-]*\n?(.*?)```", stash, text or "", flags=re.S)
    t = html.escape(convert_tables(CITE_RE.sub("", text)), quote=False)
    out = []
    for line in t.split("\n"):
        s = line.strip()
        if re.fullmatch(r"[-*_]{3,}", s):
            out.append("")
            continue
        m = re.match(r"^#{1,6}\s*(.+)$", s)
        if m:
            out.append(f"<b>{m.group(1)}</b>")
            continue
        m = re.match(r"^&gt;\s?(.*)$", s)
        if m:
            out.append(f"<blockquote>{m.group(1)}</blockquote>")
            continue
        out.append(re.sub(r"^(\s*)[-*•]\s+", r"\1• ", line))
    t = "\n".join(out)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"__(.+?)__", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\*\w])", r"<i>\1</i>", t)
    t = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", t)
    t = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
        lambda m: '<a href="%s">%s</a>' % (m.group(2).replace('"', "%22"), m.group(1)),
        t,
    )
    t = t.replace("**", "")
    return re.sub(r"\x00(\d+)\x00", lambda m: "<pre>%s</pre>" % html.escape(codes[int(m.group(1))].strip("\n"), quote=False), t)

def strip_tags(text):
    return html.unescape(re.sub(r"<[^>]+>", "", text))

def split_html(text, limit=3800):
    parts, cur = [], ""
    for para in text.split("\n\n"):
        if cur and len(cur) + len(para) + 2 > limit:
            parts.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
        while len(cur) > limit:
            parts.append(cur[:limit])
            cur = cur[limit:]
    if cur:
        parts.append(cur)
    return parts

async def send_html(msg, text, keyboard=None):
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    parts = split_html(text)
    for i, part in enumerate(parts):
        markup = keyboard if i == len(parts) - 1 else None
        try:
            await msg.reply_text(
                part, parse_mode=ParseMode.HTML, link_preview_options=NO_PREVIEW, reply_markup=markup
            )
        except BadRequest:
            await msg.reply_text(strip_tags(part), link_preview_options=NO_PREVIEW, reply_markup=markup)

def sources_block(results, limit=5):
    if not results:
        return ""
    lines = ["", "🔗 <b>Источники</b>"]
    for i, r in enumerate(results[:limit], 1):
        title = shorten(r.get("title") or domain(r.get("url", "")), 70)
        lines.append(f'{i}. <a href="{esc(r.get("url", ""))}">{esc(title)}</a> — {esc(domain(r.get("url", "")))}')
    return "\n".join(lines)

class Working:
    FRAMES = ["🙂 .", "🤔 ..", "🙂 ...", "🤔 ...."]
    def __init__(self, msg, kind=None):
        self.msg, self.ack, self.task = msg, None, None
    async def _animate(self):
        i = 0
        try:
            while True:
                await asyncio.sleep(2)
                i += 1
                try:
                    await self.ack.edit_text(self.FRAMES[i % len(self.FRAMES)])
                    if i % 2 == 0:
                        await self.msg.chat.send_action(ChatAction.TYPING)
                except Exception:
                    pass
        except asyncio.CancelledError:
            pass
    async def __aenter__(self):
        self.ack = await self.msg.reply_text(self.FRAMES[0])
        try:
            await self.msg.chat.send_action(ChatAction.TYPING)
        except Exception:
            pass
        self.task = asyncio.create_task(self._animate())
        return self
    async def __aexit__(self, *exc):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except BaseException:
                pass
        try:
            await self.ack.delete()
        except Exception:
            pass
        return False

# ───────────────────────── Интенты и сценарии ─────────────────────────
WEATHER_RE = re.compile(r"погод|прогноз|weather|какая температура|сколько градус|будет ли дожд|нужен ли зонт", re.I)
TRANSLATE_RE = re.compile(r"перевед|перевод|translate|как (?:будет|сказать|звучит|переводится)|по-английски|по-русски", re.I)
SEARCH_RE = re.compile(r"найд[иь]|поищи|загугли|ищи|в интернете|актуальн|сейчас|сегодня|последн|свеж|курс |стоимост|кто такой|что такое|когда выйд|расписани|результат", re.I)
SMALLTALK_RE = re.compile(r"^\s*(привет|здравствуй|добр\w+ (?:утро|день|вечер|ночи)|как дела|спасибо|благодарю|пока\b|hello|hi\b|hey)", re.I)

def detect_intent(text):
    if WEATHER_RE.search(text): return "weather"
    if TRANSLATE_RE.search(text): return "translate"
    return "chat"

def wants_search(text):
    if SMALLTALK_RE.search(text): return False
    if SEARCH_RE.search(text): return True
    return AUTO_SEARCH and text.strip().endswith("?") and len(text.strip()) >= 15

async def do_chat(msg, text, quoted="", force_search=False):
    chat_id = msg.chat_id
    history = await load_history(chat_id)
    links_requested = bool(LINKS_RE.search(text))
    searching = force_search or links_requested or wants_search(text)
    quote_note = f"[Собеседник отвечает на сообщение: «{shorten(quoted, 1200)}»]\n" if quoted else ""
    results, system, question = [], None, quote_note + text
    if searching:
        query = text if (not quoted or len(text.split()) >= 6) else f"{shorten(quoted, 150)} {text}"
        try:
            results = await tavily_search(query, max_results=6, depth=SEARCH_DEPTH)
        except Exception:
            logging.exception("Поиск не удался")
        if results:
            ctx = "\n\n".join(
                f"[{i}] {r.get('title', '')} ({domain(r.get('url', ''))})\n{shorten(r.get('content'), 800)}"
                for i, r in enumerate(results, 1)
            )
            question = f"{quote_note}Вопрос: {text}\n\nДанные из интернета:\n{ctx}"
            system = SEARCH_SYSTEM
        elif re.search(r"тариф|стоимост|сколько стоит|\bцен[аыуе]\b|ставк|закон|постановлен|инфляц", text, re.I):
            system = SYSTEM_PROMPT + NO_DATA_NOTE
    profile = await get_profile(chat_id)
    note = ""
    if profile.get("name"): note += f"\nСобеседника зовут {profile['name']}."
    if profile.get("city"): note += f"\nЕго город: {profile['city']}."
    system = (system or SYSTEM_PROMPT) + note
    async with Working(msg, "chat"):
        answer = await ask_checked(history + [{"role": "user", "content": question}], system, text)
        answer = answer or "Не получилось сформулировать ответ."
        if len(answer) > 1100 and "```" not in answer:
            try:
                short = await ask([{"role": "user", "content": answer}], SHORTEN_SYSTEM)
                if short and 80 <= len(short) < len(answer): answer = short
            except Exception: pass
        answer = beautify(await polish(answer))
    notes = footer_notes(text, answer)
    await send_html(msg, md_to_html(answer + notes) + (sources_block(results) if links_requested else ""))
    stored = (f"(в ответ на «{shorten(quoted, 200)}») " if quoted else "") + text
    await append_message(chat_id, "user", shorten(stored, 500))
    await append_message(chat_id, "assistant", shorten(answer, 800))

async def do_weather(msg, text, from_pending=False):
    chat_id = msg.chat_id
    cands = extract_city_candidates(text) if from_pending else []
    if not cands:
        profile = await get_profile(chat_id)
        home = profile.get("city") or DEFAULT_CITY
        if home: cands = [home]
    if not cands:
        PENDING[chat_id] = ("weather", time.time())
        await msg.reply_text("🌍 Для какого города показать погоду?")
        return
    async with Working(msg, "weather"):
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                geo = None
                for cand in cands:
                    r = await client.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": cand, "count": 1, "language": "ru"})
                    res = r.json().get("results")
                    if res: geo = res[0]; break
                if not geo:
                    await do_chat(msg, text, force_search=True); return
                r = await client.get("https://api.open-meteo.com/v1/forecast", params={
                    "latitude": geo["latitude"], "longitude": geo["longitude"],
                    "current": "temperature_2m,weather_code,wind_speed_10m",
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                    "timezone": "auto", "forecast_days": 3,
                })
                data = r.json()
                cur = data["current"]
                lines = [f"🌍 <b>Погода: {esc(geo['name'])}</b>", "", f"🌡 Сейчас: <b>{round(cur['temperature_2m'])}°C</b>"]
                for i in range(3):
                    d = data["daily"]["time"][i]
                    lines.append(f"📅 {d}: {round(data['daily']['temperature_2m_min'][i])}…{round(data['daily']['temperature_2m_max'][i])}°C")
                card = "\n".join(lines)
                await send_html(msg, card)
                if from_pending: await save_profile(chat_id, {"city": geo['name']})
                await append_message(chat_id, "user", text)
                await append_message(chat_id, "assistant", strip_tags(card))
        except Exception as e:
            logging.exception("Погода упала")
            await msg.reply_text(f"Не смог получить погоду. {e}")

def extract_city_candidates(text):
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яа-яЁё]+)*", text)
    stop = {"погода", "на", "в", "скажи", "завтра", "сегодня", "прогноз"}
    return [t for t in tokens if t.lower() not in stop][:3]

async def do_translate(msg, text, from_pending=False):
    rest = text if from_pending else TRANSLATE_RE.sub("", text)
    if len(re.sub(r"[\W_]+", "", rest)) < 2:
        PENDING[msg.chat_id] = ("translate", time.time())
        await msg.reply_text("🌐 Какой текст перевести?")
        return
    async with Working(msg, "translate"):
        answer = await ask([{"role": "user", "content": text}], TRANSLATE_SYSTEM)
    await send_html(msg, md_to_html(answer or "Не получилось перевести."))
    await append_message(msg.chat_id, "user", text)
    await append_message(msg.chat_id, "assistant", answer)

async def run_intent(msg, intent, text, from_pending=False):
    if intent == "weather": await do_weather(msg, text, from_pending)
    elif intent == "translate": await do_translate(msg, text, from_pending)
    else: await do_chat(msg, text)

# ───────────────────────── Обработчики ─────────────────────────
HELP_TEXT = (
    "👋 <b>Привет! Я твой карманный всезнайка.</b>\n\n"
    "Спрашивай что угодно 😄\n\n"
    "🌤 <b>Погода</b> — «погода в Москве»\n"
    "🌐 <b>Перевод</b> — «переведи на английский: привет»\n"
    "💬 <b>Разговор</b> — просто напиши вопрос\n"
    "🧠 <b>Память</b> — я помню наш разговор\n"
    "📊 <b>/stats</b> — расход токенов\n"
    "Сбросить память — /reset"
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_html(update.message, HELP_TEXT)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with _db() as c:
        c.execute("DELETE FROM messages WHERE chat_id=?", (update.effective_chat.id,))
        c.execute("DELETE FROM profile WHERE chat_id=?", (update.effective_chat.id,))
    PROFILE.pop(update.effective_chat.id, None)
    await update.message.reply_text("🧹 Всё забыл! Начнём с чистого листа 🙂")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if ALLOWED_USER_IDS and user.id not in ALLOWED_USER_IDS: return
    rows = await get_quota()
    lines = ["📊 <b>Статистика за сегодня (UTC)</b>", ""]
    if not rows: lines.append("Пока ни одного запроса.")
    for key_name, used, tokens in rows:
        lines.append(f"🔑 <b>{esc(key_name)}</b>: {used} запросов, {tokens:,} токенов".replace(",", " "))
    lines.append("")
    lines.append(f"🤖 Модели: {esc(', '.join(GROQ_MODELS))}")
    active = [c["name"] for c in CANDIDATES if MODEL_SKIP.get(c["name"], 0) <= time.time()]
    lines.append(f"✅ Доступны: {esc(', '.join(active) or 'нет')}")
    await send_html(update.message, "\n".join(lines))

def in_quiet_hours():
    if not QUIET_HOURS: return False
    now = datetime.now(TZINFO) if TZINFO else datetime.now()
    h = now.hour
    a, b = QUIET_HOURS
    return a <= h < b if a < b else (h >= a or h < b)

def quoted_text(msg):
    if getattr(msg, "quote", None) and msg.quote.text: return msg.quote.text
    if getattr(msg, "reply_to_message", None): return msg.reply_to_message.text or msg.reply_to_message.caption or ""
    return ""

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.text: return
    user = update.effective_user
    
    if ALLOWED_USER_IDS and user.id not in ALLOWED_USER_IDS:
        await msg.reply_text("Извините, этот бот закрытый."); return

    if in_quiet_hours():
        await msg.reply_text(f"🌙 Сейчас тихий час ({QUIET_HOURS[0]:02d}:00–{QUIET_HOURS[1]:02d}:00). Я сплю. Напиши утром 🙂")
        return

    text = msg.text.strip()
    chat_id = msg.chat_id
    
    try:
        if special_reply(text):
            await send_html(msg, md_to_html(special_reply(text)))
            return
            
        quoted = quoted_text(msg)
        if quoted:
            PENDING.pop(chat_id, None)
            await do_chat(msg, text, quoted=quoted); return
            
        entry = PENDING.pop(chat_id, None)
        if entry:
            kind, ts = entry
            if time.time() - ts <= PENDING_TTL and len(text.split()) <= 6 and "?" not in text:
                await run_intent(msg, kind, text, from_pending=True); return
                
        await run_intent(msg, detect_intent(text), text)
    except Exception as e:
        logging.exception("Ошибка обработки")
        await msg.reply_text("😔 Что-то пошло не так. Попробуйте позже.\n" f"Причина: {str(e)[:200]}")

async def on_error(update, context: ContextTypes.DEFAULT_TYPE):
    logging.error("Необработанная ошибка: %s", context.error, exc_info=context.error)

async def post_init(app: Application):
    try:
        me = await app.bot.get_me()
        logging.info("=== Бот запущен: @%s ===", me.username)
    except Exception:
        logging.warning("Не удалось получить имя бота")
    logging.info("Модели: %s", ", ".join(c["name"] for c in CANDIDATES))

def main():
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .concurrent_updates(True)
        .connect_timeout(30).read_timeout(30).write_timeout(30).pool_timeout(30)
        .build()
    )
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.add_error_handler(on_error)

    if WEBHOOK_URL:
        from aiohttp import web
        async def health(_): return web.Response(text="ok")
        async def _wh(app_instance):
            async def handler(request):
                data = await request.json()
                update = Update.de_json(data, app_instance.bot)
                await app_instance.update_queue.put(update)
                return web.Response(text="ok")
            return handler
        async def run_webhook():
            await app.initialize()
            await app.bot.set_webhook(WEBHOOK_URL, allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
            await app.start()
            server = web.Application()
            server.router.add_get("/", health)
            server.router.add_post("/webhook", await _wh(app))
            runner = web.AppRunner(server)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", PORT)
            await site.start()
            logging.info("Webhook на порту %s", PORT)
            await asyncio.Event().wait()
        asyncio.run(run_webhook())
    else:
        app.run_polling(bootstrap_retries=-1, drop_pending_updates=False)

if __name__ == "__main__":
    main()