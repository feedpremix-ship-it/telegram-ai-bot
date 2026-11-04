"""Render only a small safe subset of Markdown; split BEFORE HTML rendering."""
import html
import re

URL = re.compile(r'(?i)(?:https?://|ftp://|tg://|www\.)[^\s<>\]\)]+|\b(?:[a-zа-яё0-9-]+\.)+(?:[a-zа-яё]{2,63})(?::[0-9]+)?(?:/[^\s<>]*)?')
LINK_REQUEST = re.compile(r'(?i)(?:дай|дайте|пришли|пришлите|покажи|укажи|нужна|нужен|нужны|предоставь|отправь).{0,35}(?:ссылк|url|адрес сайта)|(?:ссылк\w*|url)\s+(?:на|для)|(?:give|send|show|provide|share).{0,25}(?:link|url)|https?://')

def links_allowed(text):
    return bool(LINK_REQUEST.search(text)) and not bool(re.search(r'(?i)(?:без|не\s+(?:давай|показывай|присылай|добавляй)).{0,15}ссыл', text))

def clean_links(text, allowed=False):
    if allowed:
        return text
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    return URL.sub('[ссылка скрыта]', text)

def utf16len(text):
    return len(text.encode('utf-16-le')) // 2

def take(text, limit):
    n = size = 0
    for ch in text:
        width = 2 if ord(ch) > 0xffff else 1
        if size + width > limit:
            break
        n += 1
        size += width
    return text[:n], text[n:]

def inline(text):
    # Escape first, then introduce only our own tags; raw model HTML never executes.
    tokens = re.split(r'(`[^`\n]+`)', text)
    out = []
    for part in tokens:
        if part.startswith('`') and part.endswith('`'):
            out.append('<code>' + html.escape(part[1:-1], quote=False) + '</code>')
        else:
            part = html.escape(part, quote=False)
            part = re.sub(r'\*\*([^*\n]+)\*\*', r'<b>\1</b>', part)
            part = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'<i>\1</i>', part)
            out.append(part)
    return ''.join(out)

def messages(text, allow_links=False):
    text = clean_links(text or 'Пустой ответ.', allow_links)
    # Each rendered part stays below 4096 UTF-16 units, even counting HTML entities.
    # 700 source units * 5 (worst-case & -> &amp;) plus tags remains safe.
    result = []
    for index, block in enumerate(text.split('```')):
        code = index % 2 == 1
        if code and '\n' in block:
            first, rest = block.split('\n', 1)
            if re.fullmatch(r'[\w+#.-]{0,24}', first):
                block = rest
        while block:
            chunk, block = take(block, 700)
            result.append('<pre>' + html.escape(chunk, quote=False) + '</pre>' if code else inline(chunk))
    return result or ['Пустой ответ.']
