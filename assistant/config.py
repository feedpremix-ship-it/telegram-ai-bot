import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

@dataclass
class Config:
    token: str = ''
    groq_key: str = ''
    database_url: str = ''
    webhook_url: str = ''
    webhook_secret: str = ''
    models: tuple = ('openai/gpt-oss-120b', 'openai/gpt-oss-20b')
    tavily_key: str = ''
    port: int = 10000
    workers: int = 2
    queue_size: int = 200
    daily_requests: int = 900
    daily_tokens: int = 180000
    minute_tokens: int = 30000
    max_output: int = 1200
    context_chars: int = 12000
    user_daily: int = 50
    admins: frozenset = field(default_factory=frozenset)
    allowed: frozenset = field(default_factory=frozenset)

    @classmethod
    def load(cls):
        def ids(name):
            return frozenset(int(x.strip()) for x in os.getenv(name, '').split(',') if x.strip())
        keys = os.getenv('GROQ_API_KEYS', os.getenv('GROQ_API_KEY', '')).split(',')
        return cls(token=os.getenv('TELEGRAM_TOKEN', '').strip(), groq_key=next((key.strip() for key in keys if key.strip()), ''),
                   database_url=os.getenv('DATABASE_URL', ''), webhook_url=os.getenv('WEBHOOK_URL', '').strip(),
                   webhook_secret=os.getenv('WEBHOOK_SECRET', ''), tavily_key=os.getenv('TAVILY_API_KEY', ''),
                   models=tuple(x.strip() for x in os.getenv('GROQ_MODELS', 'openai/gpt-oss-120b,openai/gpt-oss-20b').split(',') if x.strip()),
                   port=int(os.getenv('PORT', '10000')), workers=int(os.getenv('WORKERS', '2')),
                   queue_size=int(os.getenv('QUEUE_SIZE', '200')), daily_requests=int(os.getenv('DAILY_REQUEST_LIMIT', '900')),
                   daily_tokens=int(os.getenv('DAILY_TOKEN_LIMIT', '180000')), minute_tokens=int(os.getenv('MINUTE_TOKEN_LIMIT', '30000')), max_output=int(os.getenv('MAX_OUTPUT_TOKENS', '1200')),
                   context_chars=int(os.getenv('CONTEXT_CHARS', '12000')), user_daily=int(os.getenv('USER_DAILY_LIMIT', '50')),
                   admins=ids('ADMIN_USER_IDS'), allowed=ids('ALLOWED_USER_IDS'))

    def validate(self):
        if not self.token or not self.groq_key or not self.database_url:
            raise ValueError('Нужны TELEGRAM_TOKEN, GROQ_API_KEYS и DATABASE_URL в Environment.')
        db = urlparse(self.database_url)
        if db.scheme not in ('postgres', 'postgresql') or not db.hostname:
            raise ValueError('DATABASE_URL должен быть адресом PostgreSQL.')
        if db.hostname not in ('localhost', '127.0.0.1') and 'sslmode=require' not in self.database_url and 'sslmode=verify-full' not in self.database_url:
            raise ValueError('Для внешней PostgreSQL укажите sslmode=require или verify-full.')
        if self.webhook_url:
            url = urlparse(self.webhook_url)
            if url.scheme != 'https' or url.path != '/webhook' or url.query or url.fragment or url.username:
                raise ValueError('WEBHOOK_URL: https://имя-сервиса.onrender.com/webhook без query.')
            if not 32 <= len(self.webhook_secret) <= 256 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in self.webhook_secret):
                raise ValueError('WEBHOOK_SECRET: 32–256 символов A-Z, a-z, 0-9, _ или -.')
        if not self.models or not all(0 < n for n in (self.queue_size, self.daily_requests, self.daily_tokens, self.minute_tokens, self.max_output, self.context_chars, self.user_daily)):
            raise ValueError('Лимиты должны быть положительными; список моделей не пустым.')
        if not 1 <= self.workers <= 4 or not 1 <= self.port <= 65535 or self.max_output > 4096 or self.context_chars > 24000:
            raise ValueError('Проверьте WORKERS (1–4), PORT, MAX_OUTPUT_TOKENS (до 4096), CONTEXT_CHARS (до 24000).')
