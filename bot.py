"""Telegram-бот с ИИ на Groq. Максимально простой и надежный."""
import asyncio
import html
import logging
import os
import re
import time
from datetime import datetime

from dotenv import load_dotenv
from telegram import Update, LinkPreviewOptions
from telegram.constants import ParseMode, ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

load_dotenv()
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
GROQ_MODELS = [m.strip() for m in os.getenv("GROQ_MODELS", "deepseek-r1-distill-llama-70b,llama-3.3-70b-versatile").split(",") if m.strip()]
ALLOWED_USER_IDS = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").replace(" ", "").split(",") if x}
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
PORT = int(os.getenv("PORT", "10000"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

STYLE = """
КАК ОТВЕЧАТЬ:
- Общайся как живой, добрый и остроумный друг. Никакого канцелярита.
- Лёгкий юмор уместен, если тема не серьёзная.
- Длина: 4–8 коротких строк, не больше 700 символов.
- Сначала прямой ответ, потом 2-4 пункта.
- Оформление: заголовок с эмодзи (**жирным**), пункты с эмодзи-иконками. Без таблиц.
"""
SYSTEM_PROMPT = "Ты — универсальный помощник в Telegram. " + STYLE

MODEL_INDEX = 0
KEY_INDEX = 0

async def ask_ai(history, system=SYSTEM_PROMPT):
    global MODEL_INDEX, KEY_INDEX
    import httpx
    if not GROQ_API_KEYS or not GROQ_MODELS:
        return "❌ Ошибка: не настроены ключи или модели Groq."
    
    # Пробуем модели по кругу
    for _ in range(len(GROQ_MODELS) * len(GROQ_API_KEYS)):
        model = GROQ_MODELS[MODEL_INDEX % len(GROQ_MODELS)]
        key = GROQ_API_KEYS[KEY_INDEX % len(GROQ_API_KEYS)]
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": model,
                        "temperature": 0.4,
                        "messages": [{"role": "system", "content": system}] + history
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"]
                    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.S | re.I).strip()
                    return text
                elif resp.status_code == 429:
                    MODEL_INDEX += 1
                    time.sleep(1)
                else:
                    logging.warning(f"Groq {model} error {resp.status_code}: {resp.text[:100]}")
                    MODEL_INDEX += 1
        except Exception as e:
            logging.warning(f"Groq connection error: {e}")
            MODEL_INDEX += 1
    return "😔 Все модели ИИ сейчас недоступны. Попробуйте позже."

async def search_web(query):
    if not TAVILY_API_KEY:
        return ""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
                json={"query": query, "max_results": 3}
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                return "\n".join([f"{r.get('title')}: {r.get('content')[:300]}" for r in results])
    except Exception as e:
        logging.warning(f"Search error: {e}")
    return ""

def md_to_html(text):
    t = html.escape(text or "", quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return t

async def do_chat(msg, text):
    chat_id = msg.chat_id
    await msg.chat.send_action(ChatAction.TYPING)
    
    # Простейшая логика: если в тексте есть "погода", ищем в интернете
    if "погод" in text.lower():
        web_data = await search_web(f"погода {text}")
        if web_data:
            text = f"Вопрос: {text}\n\nДанные из интернета: {web_data}"
    
    history = [{"role": "user", "content": text}]
    answer = await ask_ai(history)
    
    await msg.reply_text(md_to_html(answer), parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 <b>Привет! Я твой карманный всезнайка.</b>\n\n"
        "Просто напиши мне любой вопрос, и я отвечу!\n"
        "Умею искать информацию в интернете и общаться на любые темы.",
        parse_mode=ParseMode.HTML
    )

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.text: return
    if ALLOWED_USER_IDS and update.effective_user.id not in ALLOWED_USER_IDS:
        await msg.reply_text("Извините, этот бот закрытый."); return

    try:
        await do_chat(msg, msg.text.strip())
    except Exception as e:
        logging.error(f"Handler error: {e}")
        await msg.reply_text("😔 Произошла ошибка при обработке запроса.")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).concurrent_updates(True).build()
    app.add_handler(CommandHandler(["start", "help"], start))
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
