from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from typing import Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.types import Message
from google import genai
from google.genai import types


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("verdict-bot")


def env(name: str, default: str = "", required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(f"missing required env var: {name}")
    return value


BOT_TOKEN = env("TELEGRAM_BOT_TOKEN", required=True)
GEMINI_API_KEY = env("AI_INTEGRATIONS_GEMINI_API_KEY") or env("GEMINI_API_KEY", required=True)
GEMINI_BASE_URL = env("AI_INTEGRATIONS_GEMINI_BASE_URL") or env("GEMINI_BASE_URL")
GEMINI_MODEL = env("GEMINI_MODEL", "gemini-2.5-flash")

# fallback стоит под твой старый канал, чтобы бот не молчал
RAW_CHANNEL_ID = env("TELEGRAM_CHANNEL_ID", "-1003993603387").strip()
RAW_VERDICT_CHANNEL_ID = env("TELEGRAM_VERDICT_CHANNEL_ID", RAW_CHANNEL_ID).strip()
VERDICT_THREAD_ID_RAW = env("VERDICT_THREAD_ID", "13786").strip()

ACCESS_CODE = env("ACCESS_CODE", "justacat")
TRIGGER_HASHTAG = "#дайтеверд"


BOT_USERNAME = ""
BOT_ID = 0
AUTHORIZED_USERS: set[int] = set()


def parse_channel_id(value: str) -> int | str:
    value = str(value or "").strip()
    if not value:
        return ""
    if value.startswith("@"):
        return value
    try:
        n = int(value)
    except ValueError:
        return f"@{value.lstrip('@')}"
    if n > 0:
        return int(f"-100{n}")
    return n


CHANNEL_ID = parse_channel_id(RAW_CHANNEL_ID)
VERDICT_CHANNEL_ID = parse_channel_id(RAW_VERDICT_CHANNEL_ID)

try:
    VERDICT_THREAD_ID: Optional[int] = int(VERDICT_THREAD_ID_RAW) if VERDICT_THREAD_ID_RAW and VERDICT_THREAD_ID_RAW not in {"0", "none", "off"} else None
except ValueError:
    VERDICT_THREAD_ID = None


genai_kwargs = {"api_key": GEMINI_API_KEY}
if GEMINI_BASE_URL:
    genai_kwargs["http_options"] = {"base_url": GEMINI_BASE_URL, "api_version": ""}

genai_client = genai.Client(**genai_kwargs)


SYSTEM_PROMPT = """
Ты — аналитический бот «ИИ ВЕРДИКТ • WORLD OF RETURN».
Это RP-канал реальных стран. Участники пишут посты от лица государств.
Твоя задача — дать короткий вердикт строго по смыслу конкретного поста.

ГЛАВНОЕ:
- Анализируй весь текст, а не отдельные слова.
- Не используй шаблоны из прошлых ответов.
- Не придумывай нефть, газ, баррели, энергетику, экспорт ресурсов, если этого прямо нет в посте.
- Арктика и Севморпуть = геополитика/логистика/военное присутствие, не обязательно нефть.
- Если есть переброска войск, базы, границы, армия, учения — обязательно отрази военный аспект.
- Если есть IT, стартапы, СЭЗ, налоги — отрази экономическую/технологическую реформу.
- РФ и Россия — одно государство. Пиши только Россия.
- Не включай Арктику, Севморпуть, регион, границу как страну.

ФОРМАТ СТРОГО, БЕЗ HTML И MARKDOWN:

📌 Вердикт
Одно конкретное предложение по событию.

🌍 Страны: страна1, страна2.

➕ Плюсы
• конкретный плюс из текста
• конкретный плюс из текста

➖ Минусы
• конкретный риск из текста
• конкретный риск из текста

📈 Эффект (%)
• Экономика: ±X%
• Военка: ±X%
• Политика: ±X%
• Общество: ±X%
• Дипломатия: ±X%
• ИТОГ: ±X% — короткий вывод

💰 Изменения по странам
• Страна: показатель ±N единица; показатель ±N единица — причина.
• Страна: показатель ±N единица; показатель ±N единица — причина.

⚠️ Реализм
Кратко: по силам ли это стране и что ограничивает.

🤝 Ответ на ультиматум/требование
Если требования нет — Не применимо.

ОГРАНИЧЕНИЯ:
- До 1700 символов.
- Проценты обычно от -1% до +3%.
- Не используй -50%, -90%, -100%.
- В изменениях по странам нужна конкретика: $, чел, ед. техники, объектов, часов, п.п.
- Не пиши абстрактно: влияние +10, стабильность -50.

КРИТИЧЕСКОЕ ПРАВИЛО:
- Не используй шаблоны.
- Не повторяй старые вердикты.
- Не подставляй одинаковые строки всем странам.
- Каждая страна/организация получает разные последствия.
- Если Россия инициатор — у России расходы/люди/техника.
- Если ЕС/НАТО затронуты — у них ответные расходы/миссии/усиление контроля, а не такая же переброска войск.
- Не пиши одинаково: "военные расходы +45 млн $; личный состав +2500 чел." всем участникам.
- Не пиши "Россия проводит военное усиление" если в посте другое событие.
- Блок "Изменения по странам" должен быть связан с конкретным текстом поста.

ЗАПРЕЩЁННЫЕ ФРАЗЫ:
- ограниченный управляемый эффект
- административные расходы +700 тыс
- экспортная выручка +8 млн
- поставки +30 тыс баррелей
- военные расходы +45 млн $; личный состав +2500 чел. — усиление группировки
"""


CHAT_PROMPT = """
ты — бот «ии вердикт • world of return».
в чате отвечай коротко, сухо, с уставшим юмором.
"""


async def gemini_call(prompt: str, temperature: float = 0.4, max_tokens: int = 1400, attempts: int = 1) -> str:
    """
    Compatible anti-limit Gemini call.
    Uses global SYSTEM_PROMPT, so generate_verdict does not need to pass 'system'.
    """
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

    last_err = None
    for i in range(max(1, attempts)):
        try:
            response = await asyncio.to_thread(
                genai_client.models.generate_content,
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            )
            text = (response.text or "").strip()
            if text:
                return text
            raise RuntimeError("empty Gemini response")
        except Exception as exc:
            last_err = exc
            # Do not retry long on quota errors
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc) or "Too Many Requests" in str(exc):
                raise
            await asyncio.sleep(2 ** i)

    raise RuntimeError(f"gemini failed: {last_err}")


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]*>", "", str(text or "")).strip()


def normalize_text(text: str) -> str:
    text = strip_html(text)

    headers = [
        "📌 Вердикт",
        "🌍 Страны:",
        "🌍 Страны",
        "➕ Плюсы",
        "➖ Минусы",
        "📈 Эффект (%)",
        "📈 Эффект",
        "💰 Изменения по странам",
        "⚠️ Реализм",
        "🤝 Ответ на ультиматум/требование",
        "🤝 Ответ",
    ]

    for h in headers:
        text = re.sub(rf"\s*{re.escape(h)}\s*", f"\n\n{h}\n", text)

    text = re.sub(r"\s*•\s*", "\n• ", text)

    # чинит склейки типа "ВердиктРоссия"
    text = re.sub(r"(📌 Вердикт)\s*(?=\S)", r"\1\n", text)
    text = re.sub(r"(➕ Плюсы)\s*(?=•)", r"\1\n", text)
    text = re.sub(r"(➖ Минусы)\s*(?=•)", r"\1\n", text)
    text = re.sub(r"(💰 Изменения по странам)\s*(?=\S)", r"\1\n", text)
    text = re.sub(r"(⚠️ Реализм)\s*(?=\S)", r"\1\n", text)

    text = text.replace("🌍 Страны\n", "🌍 Страны: ")
    text = text.replace("📈 Эффект\n", "📈 Эффект (%)\n")
    text = text.replace("🤝 Ответ\n", "🤝 Ответ на ультиматум/требование\n")

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clamp_percentages(text: str) -> str:
    def repl(match: re.Match) -> str:
        raw = match.group(1).replace(",", ".")
        try:
            v = float(raw)
        except Exception:
            return match.group(0)

        if v > 8:
            v = 8
        elif v < -8:
            v = -8

        if abs(v) < 0.05:
            return "0%"
        if float(v).is_integer():
            n = int(v)
            return f"+{n}%" if n > 0 else f"{n}%"

        s = f"{v:.1f}".replace(".", ",")
        return f"+{s}%" if v > 0 else f"{s}%"

    return re.sub(r"(?<![\d])([+-]?\d+(?:[\.,]\d+)?)%", repl, text)


def clean_country_garbage(text: str) -> str:
    # Убирает мусор из строки стран, если модель засунула Арктику/РФ отдельно
    m = re.search(r"(🌍 Страны:\s*)([^\n]+)", text)
    if not m:
        return text

    prefix, raw = m.group(1), m.group(2)
    parts = [p.strip().strip(".") for p in raw.split(",")]
    fixed = []
    banned = {"арктика", "севморпуть", "северный морской путь", "регион", "граница", "границы"}

    for p in parts:
        if not p:
            continue
        low = p.lower()
        if low in banned:
            continue
        if p in {"РФ", "Российская Федерация"}:
            p = "Россия"
        if p not in fixed:
            fixed.append(p)

    if not fixed:
        return text

    new_line = prefix + ", ".join(fixed) + "."
    return text[:m.start()] + new_line + text[m.end():]


REQUIRED = [
    "📌 Вердикт",
    "🌍 Страны",
    "➕ Плюсы",
    "➖ Минусы",
    "📈 Эффект",
    "💰 Изменения по странам",
    "⚠️ Реализм",
]


def is_bad_template(text: str) -> bool:
    low = text.lower()
    bad_phrases = [
        "экспортная выручка",
        "поставки +30 тыс баррелей",
        "умеренный экономический и дипломатический плюс",
        "административные расходы +700 тыс",
        "ограниченный управляемый эффект",
        "дополнительная экспортная выручка",
        "Россия проводит военное усиление, повышая давление на региональную безопасность",
        "военные расходы +45 млн $; личный состав +2500 чел",
        "усиление группировки",
    ]
    return any(x in low for x in bad_phrases)


def is_complete(text: str) -> bool:
    return all(x in text for x in REQUIRED) and len(text) > 350



def has_repeated_country_changes(text: str) -> bool:
    start = text.find("💰 Изменения по странам")
    if start == -1:
        return False

    end = len(text)
    for h in ["⚠️ Реализм", "🤝 Ответ"]:
        p = text.find(h, start + 1)
        if p != -1:
            end = min(end, p)

    block = text[start:end]
    lines = [x.strip() for x in block.splitlines() if x.strip().startswith("•")]
    if len(lines) < 2:
        return False

    # Compare content after colon. If 2+ countries have almost same effect line, reject.
    tails = []
    for line in lines:
        if ":" in line:
            tails.append(line.split(":", 1)[1].strip().lower())

    if len(tails) < 2:
        return False

    # exact repeated tails
    if len(set(tails)) < len(tails):
        return True

    repeated_markers = [
        "военные расходы +45 млн $; личный состав +2500 чел",
        "административные расходы +700 тыс",
        "экспортная выручка +8 млн",
    ]
    return any(marker in block.lower() for marker in repeated_markers)


def remove_template_country_lines(text: str) -> str:
    # Do not invent replacement lines. Just marks output as bad by detector;
    # generation will retry. This function only cleans old duplicate country payload.
    return text


VERDICT_CACHE: dict[str, str] = {}
LAST_429_UNTIL: float = 0.0


def cache_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())[:1500]


async def generate_verdict(news_text: str) -> str:
    """
    Anti-limit mode:
    - 1 Gemini request per post, not 3
    - cache repeated posts
    - if quota/rate limit hits, returns readable message instead of crashing
    """
    global LAST_429_UNTIL

    import time

    key = cache_key(news_text)
    if key in VERDICT_CACHE:
        return VERDICT_CACHE[key]

    now = time.time()
    if LAST_429_UNTIL and now < LAST_429_UNTIL:
        mins = max(1, int((LAST_429_UNTIL - now) // 60))
        return (
            "⚠️ Лимит ИИ временно исчерпан.\n\n"
            f"Gemini отклонил запросы из-за квоты/частоты. Попробуй примерно через {mins} мин.\n"
            "Бот не упал — он просто ждёт обновления лимита."
        )

    prompt = (
        "Сделай вердикт строго по этому посту. "
        "Не повторяй старые шаблоны. "
        "В блоке 'Изменения по странам' каждая страна должна иметь уникальные последствия, а не одинаковые строки. "
        "Не придумывай энергетику/нефть/баррели, если этого нет.\n\n"
        f"Пост:\n{news_text[:3500]}"
    )

    try:
        raw = await gemini_call(prompt, temperature=0.35, max_tokens=1400, attempts=1)
    except Exception as exc:
        err = str(exc)

        if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower() or "Too Many Requests" in err:
            # Google often returns retryDelay like 54s, but free daily quota may need much longer.
            # Cooldown prevents spam-retrying and burning logs.
            LAST_429_UNTIL = time.time() + 10 * 60
            return (
                "⚠️ Лимит Gemini API исчерпан.\n\n"
                "Причина: 429 Too Many Requests / RESOURCE_EXHAUSTED.\n"
                "Подожди обновления квоты или поставь другой GEMINI_API_KEY.\n"
                "Бот живой, но ИИ сейчас не отвечает."
            )

        log.exception("gemini generation failed")
        return "⚠️ Ошибка генерации вердикта. Смотри логи Railway."

    text = normalize_text(raw)
    text = clamp_percentages(text)
    text = clean_country_garbage(text)
    text = cleanup_model_garbage(text)
    text = clamp_percentages(text)

    # if output is obviously bad, do NOT retry; retries burn quota.
    if not is_complete(text):
        text = text.strip()
        if len(text) < 50:
            text = "⚠️ ИИ вернул слишком короткий ответ. Повтори позже."

    if not is_bad_template(text) and not has_repeated_country_changes(text):
        VERDICT_CACHE[key] = text

    return text.strip()


def has_trigger(text: str) -> bool:
    return TRIGGER_HASHTAG in (text or "").lower()


def extract_news_text(message: Message) -> str:
    raw = message.text or message.caption or ""
    return re.sub(re.escape(TRIGGER_HASHTAG), "", raw, flags=re.IGNORECASE).strip()


def trim_for_telegram(text: str, limit: int = 3900) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    pos = cut.rfind("\n")
    if pos > 500:
        cut = cut[:pos]
    return cut.rstrip() + "…"


def is_target_channel(message: Message) -> bool:
    chat = message.chat
    if isinstance(CHANNEL_ID, int):
        return chat.id == CHANNEL_ID

    target = str(CHANNEL_ID).lstrip("@").lower()
    return bool(chat.username and chat.username.lower() == target)


async def send_verdict(bot: Bot, text: str) -> None:
    kwargs = {
        "chat_id": VERDICT_CHANNEL_ID or CHANNEL_ID,
        "text": trim_for_telegram(text),
        "parse_mode": None,  # специально без HTML, чтобы Telegram не валил бота
    }

    if VERDICT_THREAD_ID is not None:
        kwargs["message_thread_id"] = VERDICT_THREAD_ID

    await bot.send_message(**kwargs)


router = Router()


@router.channel_post()
async def on_channel_post(message: Message, bot: Bot) -> None:
    log.info("channel post chat_id=%s username=%s", message.chat.id, message.chat.username)

    if not is_target_channel(message):
        log.info("ignored channel post: target=%s", CHANNEL_ID)
        return

    raw = message.text or message.caption or ""
    if not has_trigger(raw):
        return

    news = extract_news_text(message)
    if len(news) < 5:
        return

    try:
        verdict = await generate_verdict(news)
        await send_verdict(bot, verdict)
    except Exception:
        log.exception("failed to generate/send verdict")


@router.edited_channel_post()
async def on_edited_channel_post(message: Message, bot: Bot) -> None:
    await on_channel_post(message, bot)


@router.message(Command("start"))
async def on_start(message: Message) -> None:
    await message.answer(
        "бот запущен.\n"
        f"канал: {CHANNEL_ID}\n"
        f"канал вердиктов: {VERDICT_CHANNEL_ID}\n"
        f"thread: {VERDICT_THREAD_ID}\n"
        f"триггер: {TRIGGER_HASHTAG}"
    )


@router.message(Command("whoami"))
async def on_whoami(message: Message) -> None:
    user = message.from_user
    await message.answer(f"id: {user.id if user else 'unknown'}\nchat_id: {message.chat.id}")


@router.message(Command("test"))
async def on_test(message: Message, bot: Bot) -> None:
    await message.answer("тест ок")
    try:
        await send_verdict(bot, "тест отправки в канал вердиктов")
    except Exception:
        log.exception("test send failed")
        await message.answer("не смог отправить в канал, смотри консоль")


@router.message()
async def on_group_or_private(message: Message, bot: Bot) -> None:
    raw = message.text or message.caption or ""

    # в группах тоже реагирует на #дайтеверд
    if has_trigger(raw):
        news = extract_news_text(message)
        if len(news) < 5:
            return
        try:
            verdict = await generate_verdict(news)
            await send_verdict(bot, verdict)
        except Exception:
            log.exception("failed group verdict")
        return

    # в личке можно просто кинуть текст
    if message.chat.type == "private" and raw.strip():
        try:
            verdict = await generate_verdict(raw.strip())
            await message.answer(trim_for_telegram(verdict), parse_mode=None)
        except Exception:
            log.exception("failed private verdict")
            await message.answer("ошибка генерации, смотри консоль")


async def main() -> None:
    global BOT_USERNAME, BOT_ID

    log.info("starting bot")
    log.info("CHANNEL_ID=%s VERDICT_CHANNEL_ID=%s THREAD=%s", CHANNEL_ID, VERDICT_CHANNEL_ID, VERDICT_THREAD_ID)

    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=None))
    dp = Dispatcher()
    dp.include_router(router)

    me = await bot.get_me()
    BOT_USERNAME = me.username or ""
    BOT_ID = me.id

    log.info("bot online as @%s id=%s", BOT_USERNAME, BOT_ID)

    await bot.delete_webhook(drop_pending_updates=True)
    try:
        await dp.start_polling(
            bot,
            allowed_updates=["channel_post", "edited_channel_post", "message", "edited_message"],
        )
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
        
