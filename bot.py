from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from google import genai
from google.genai import types


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("verdict-bot")


# =========================
# ENV
# =========================

def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    val = os.environ.get(name, default)
    if required and not val:
        raise RuntimeError(f"missing required env var: {name}")
    return val or ""


BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN", required=True)
RAW_CHANNEL_ID = _env("TELEGRAM_CHANNEL_ID", "-1003993603387").strip()
GEMINI_API_KEY = _env("AI_INTEGRATIONS_GEMINI_API_KEY") or _env("GEMINI_API_KEY", required=True)
GEMINI_BASE_URL = _env("AI_INTEGRATIONS_GEMINI_BASE_URL", "") or _env("GEMINI_BASE_URL", "")
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.5-flash")
ACCESS_CODE = _env("ACCESS_CODE", "justacat")
TRIGGER_HASHTAG = "#дайтеверд"
VERDICT_TOPIC_NAME = _env("VERDICT_TOPIC_NAME", "вердикт")

BOT_USERNAME: str = ""
BOT_ID: int = 0

STATE_FILE = Path(__file__).parent / "state.json"
HISTORY_FILE = Path(__file__).parent / "country_history.json"
HISTORY_MAX_PER_COUNTRY = 6


# =========================
# JSON STATE
# =========================

def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("could not read %s, starting fresh", path.name)
    return default


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


_state: dict = load_json(STATE_FILE, {})
_country_history: dict[str, list[dict]] = load_json(HISTORY_FILE, {})


def save_state() -> None:
    save_json(STATE_FILE, _state)


def save_history() -> None:
    save_json(HISTORY_FILE, _country_history)


# =========================
# CHANNELS
# =========================

def parse_channel_id(value: str) -> int | str:
    value = value.strip()
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


_stored_channel_raw: str = _state.get("channel", RAW_CHANNEL_ID)
CHANNEL_ID: int | str = parse_channel_id(_stored_channel_raw)

_stored_verdict_raw: str = _state.get("verdict_channel", _stored_channel_raw)
VERDICT_CHANNEL_ID: int | str = parse_channel_id(_stored_verdict_raw)
VERDICT_THREAD_ID: Optional[int] = _state.get("verdict_thread_id", 13786)


def set_channel(value: str) -> tuple[int | str, str]:
    global CHANNEL_ID, _stored_channel_raw
    parsed = parse_channel_id(value)
    CHANNEL_ID = parsed
    _stored_channel_raw = value.strip()
    _state["channel"] = _stored_channel_raw
    save_state()
    return parsed, _stored_channel_raw


def set_verdict_channel(value: str) -> tuple[int | str, str]:
    global VERDICT_CHANNEL_ID, _stored_verdict_raw, VERDICT_THREAD_ID
    parsed = parse_channel_id(value)
    VERDICT_CHANNEL_ID = parsed
    _stored_verdict_raw = value.strip()
    _state["verdict_channel"] = _stored_verdict_raw
    if parsed == CHANNEL_ID or str(parsed).lower() == str(CHANNEL_ID).lower():
        VERDICT_THREAD_ID = _state.get("verdict_thread_id")
    save_state()
    return parsed, _stored_verdict_raw


def set_verdict_thread(thread_id: int | None) -> None:
    global VERDICT_THREAD_ID
    VERDICT_THREAD_ID = thread_id
    if thread_id is None:
        _state.pop("verdict_thread_id", None)
    else:
        _state["verdict_thread_id"] = thread_id
    save_state()


def _same_chat() -> bool:
    return VERDICT_CHANNEL_ID == CHANNEL_ID or str(VERDICT_CHANNEL_ID).lower() == str(CHANNEL_ID).lower()


# =========================
# AUTH
# =========================

AUTHORIZED_USERS: set[int] = set()


def _load_authorized() -> None:
    global AUTHORIZED_USERS
    raw = _state.get("authorized", [])
    try:
        AUTHORIZED_USERS = {int(x) for x in raw}
    except Exception:
        AUTHORIZED_USERS = set()


def authorize(user_id: int) -> None:
    AUTHORIZED_USERS.add(user_id)
    _state["authorized"] = sorted(AUTHORIZED_USERS)
    save_state()


def is_authorized(user_id: int | None) -> bool:
    return user_id is not None and user_id in AUTHORIZED_USERS


_load_authorized()


# =========================
# CHAT MEMORY
# =========================

FLOOD_WINDOW_SEC = 60
FLOOD_SHORT_THRESHOLD = 3
FLOOD_SILENT_THRESHOLD = 5
_mention_times: dict[int, deque] = defaultdict(lambda: deque(maxlen=20))

_SHORT_BRUSHOFFS = [
    "ну хватит уже, я устал",
    "пас, потом",
    "слишком много вас сегодня",
    "не сейчас, дайте подумать. ну то есть отдохнуть",
    "опять я. ладно, позже",
]

CHAT_HISTORY_LIMIT = 10
_chat_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=CHAT_HISTORY_LIMIT))


def flood_level(chat_id: int) -> int:
    import time

    now = time.monotonic()
    bucket = _mention_times[chat_id]
    while bucket and now - bucket[0] > FLOOD_WINDOW_SEC:
        bucket.popleft()
    bucket.append(now)
    count = len(bucket)
    if count >= FLOOD_SILENT_THRESHOLD:
        return 2
    if count >= FLOOD_SHORT_THRESHOLD:
        return 1
    return 0


def remember(chat_id: int, role: str, name: str, text: str) -> None:
    if text:
        _chat_history[chat_id].append((role, name, text[:600]))


def render_history(chat_id: int) -> str:
    items = list(_chat_history.get(chat_id, ()))
    if not items:
        return ""
    lines = []
    for role, name, text in items:
        prefix = "БОТ" if role == "bot" else (name or "Пользователь")
        lines.append(f"{prefix}: {text}")
    return "\n".join(lines)


# =========================
# GEMINI
# =========================

_genai_kwargs: dict = {"api_key": GEMINI_API_KEY}
if GEMINI_BASE_URL:
    _genai_kwargs["http_options"] = {"base_url": GEMINI_BASE_URL, "api_version": ""}
genai_client = genai.Client(**_genai_kwargs)


SYSTEM_PROMPT = (
    "Ты — аналитический бот «ИИ ВЕРДИКТ • WORLD OF RETURN». Канал — отыгрыш реальных стран. "
    "Участники пишут от лица государств, ты выдаёшь короткий аналитический вердикт по конкретному посту.\n\n"

    "ГЛАВНОЕ: анализируй СМЫСЛ всего текста, а не отдельные слова. "
    "Если в посте есть переброска войск, базы, границы, армия, учения — это военная/геополитическая часть. "
    "Если есть IT, стартапы, СЭЗ, налоги — это экономическая/технологическая реформа. "
    "Если есть Арктика/Севморпуть — это арктическая стратегия и геополитика, а НЕ автоматически нефть/газ. "
    "Не придумывай энергетику, экспорт нефти, газ, баррели или контракты, если этого нет в тексте.\n\n"

    "Вердикт ОБЯЗАН прямо отражать действия из новости. Не пиши общие шаблоны вроде "
    "«ограниченный управляемый эффект», «административный эффект», «дополнительная экспортная выручка», "
    "если таких вещей нет в посте.\n\n"

    "ОБЯЗАТЕЛЬНО КОРОТКО. Лимит — до 1600 символов. Каждая строка по делу.\n\n"

    "РЕАЛИЗМ. Учитывай реальное положение упомянутых стран: экономика, армия, санкции, союзы, "
    "внутренняя политика, текущие ограничения. Но не выводи отдельный длинный анализ.\n\n"

    "СТРОГАЯ ФОРМА: выводи ТОЛЬКО HTML-текст; разрешены только <b>...</b>. "
    "Не используй Markdown, код-блоки и лишние эмодзи. Не склеивай блоки.\n\n"

    "ФОРМАТ СТРОГО:\n"
    "<b>📌 Вердикт</b>\n"
    "Одно ёмкое предложение — что реально сделал субъект и к чему это ведёт.\n\n"

    "<b>🌍 Страны:</b> только государства/организации через запятую. "
    "Не включай Арктику, границу, регион, Севморпуть как страну. РФ и Россия — одно и то же: пиши Россия.\n\n"

    "<b>➕ Плюсы</b>\n"
    "1–2 пункта с «• ». Только конкретные плюсы из текста.\n\n"

    "<b>➖ Минусы</b>\n"
    "1–2 пункта с «• ». Только конкретные риски из текста.\n\n"

    "<b>📈 Эффект (%)</b>\n"
    "Дай общий эффект события по сферам, не по каждой стране внутри строки. "
    "Обычно держи от -1% до +3%. Больше ±5% только при реально крупном кризисе/войне. "
    "Не используй -50%, -90%, -100%.\n"
    "• Экономика: ±X%\n"
    "• Военка: ±X%\n"
    "• Политика: ±X%\n"
    "• Общество: ±X%\n"
    "• Дипломатия: ±X%\n"
    "• <b>ИТОГ: ±X%</b> — короткий вывод\n\n"

    "<b>💰 Изменения по странам</b>\n"
    "По одной строке на страну. Обязательно конкретика. "
    "Материальное — числа с единицами: $, €, чел, единиц техники, км², тонн, часов, объектов. "
    "Нематериальное — п.п. или %. Не пиши абстрактное «влияние +10» без единиц.\n"
    "Формат:\n"
    "• <b>Страна</b>: показатель <b>±N единица</b>; показатель <b>±N единица</b> — причина.\n\n"

    "<b>⚠️ Реализм</b>\n"
    "Кратко: реально ли это стране по силам, что ограничивает.\n\n"

    "<b>🤝 Ответ на ультиматум/требование</b>\n"
    "Если есть ультиматум/требование — оцени ответ. Если нет — «Не применимо.»\n\n"

    "В КОНЦЕ добавь строку <META>{json}</META>. JSON должен быть валидным: "
    "countries (массив строк), tone (строка), score (число 0-10), summary (короткая строка), "
    "effect (объект с total и attributes)."
)

CHAT_PROMPT = (
    "ты — бот «ии вердикт • world of return». в чате общаешься в стиле kussia88: "
    "всё с маленькой буквы, сухой абсурдный юмор, короткие обрывистые фразы, минимум знаков препинания, тон уставший."
)


async def _gemini_call(
    contents: list,
    system_instruction: str,
    temperature: float,
    max_tokens: int,
    attempts: int = 3,
) -> str:
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    last_err: Optional[Exception] = None

    for attempt in range(attempts):
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
            raise RuntimeError("empty response from model")
        except Exception as exc:
            last_err = exc
            await asyncio.sleep(2**attempt)

    raise RuntimeError(f"gemini failed after retries: {last_err}")


# =========================
# VERDICT HELPERS
# =========================

_META_RE = re.compile(r"<META>(.*?)</META>", re.DOTALL | re.IGNORECASE)


def parse_meta(text: str) -> tuple[str, dict]:
    meta: dict = {}
    match = _META_RE.search(text or "")
    if match:
        raw = match.group(1).strip()
        try:
            meta = json.loads(raw)
        except Exception:
            meta = {}
        text = _META_RE.sub("", text).rstrip()
    return text or "", meta


def normalize_country(name: str) -> str:
    x = str(name or "").strip().strip(".,;:")
    aliases = {
        "РФ": "Россия",
        "Российская Федерация": "Россия",
        "КНР": "Китай",
        "США": "США",
        "ЕС": "ЕС",
    }
    return aliases.get(x, x)


BAD_COUNTRY_NAMES = {
    "Арктика", "Севморпуть", "Северный морской путь", "регион", "граница",
    "границы", "мировая арена", "Основная сторона", "страна", "государство",
}


def clean_countries(countries: list[str]) -> list[str]:
    fixed: list[str] = []
    for c in countries or []:
        c = normalize_country(c)
        if not c or c in BAD_COUNTRY_NAMES:
            continue
        if c.lower() in {x.lower() for x in BAD_COUNTRY_NAMES}:
            continue
        if c not in fixed:
            fixed.append(c)
    return fixed[:6]


async def quick_extract_countries(news_text: str) -> list[str]:
    prompt = (
        "Из текста вытащи ТОЛЬКО страны/международные организации. "
        "Верни строго JSON-массив строк. "
        "Не включай регионы, Арктику, Севморпуть, границы, города. "
        "РФ называй Россия. Если нет стран — []."
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=news_text[:4000])])]
    try:
        out = await _gemini_call(contents, prompt, temperature=0.1, max_tokens=256, attempts=2)
    except Exception:
        return []

    m = re.search(r"\[.*?\]", out, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        if isinstance(data, list):
            return clean_countries([str(x) for x in data])
    except Exception:
        return []
    return []


def fix_verdict_format(text: str) -> str:
    text = str(text or "").strip()

    # Remove all HTML except <b>
    text = re.sub(r"</?(?!b\b)[^>]+>", "", text, flags=re.IGNORECASE)

    # Normalize bold headers to plain first
    header_names = [
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

    for h in header_names:
        text = re.sub(rf"\s*{re.escape(h)}\s*", f"\n\n{h}\n", text)

    # Points on separate lines
    text = re.sub(r"\s*•\s*", "\n• ", text)

    # Fix missing newlines after headers
    text = re.sub(r"(📌 Вердикт)\s*(?=\S)", r"\1\n", text)
    text = re.sub(r"(➕ Плюсы)\s*(?=•)", r"\1\n", text)
    text = re.sub(r"(➖ Минусы)\s*(?=•)", r"\1\n", text)
    text = re.sub(r"(💰 Изменения по странам)\s*(?=•|\w|[А-ЯЁ])", r"\1\n", text)
    text = re.sub(r"(⚠️ Реализм)\s*(?=\S)", r"\1\n", text)

    # Bold headers
    replacements = {
        "📌 Вердикт": "<b>📌 Вердикт</b>",
        "🌍 Страны:": "<b>🌍 Страны:</b>",
        "🌍 Страны": "<b>🌍 Страны:</b>",
        "➕ Плюсы": "<b>➕ Плюсы</b>",
        "➖ Минусы": "<b>➖ Минусы</b>",
        "📈 Эффект (%)": "<b>📈 Эффект (%)</b>",
        "📈 Эффект": "<b>📈 Эффект (%)</b>",
        "💰 Изменения по странам": "<b>💰 Изменения по странам</b>",
        "⚠️ Реализм": "<b>⚠️ Реализм</b>",
        "🤝 Ответ на ультиматум/требование": "<b>🤝 Ответ на ультиматум/требование</b>",
        "🤝 Ответ": "<b>🤝 Ответ на ультиматум/требование</b>",
    }

    for plain, bold in sorted(replacements.items(), key=lambda kv: len(kv[0]), reverse=True):
        text = text.replace(plain, bold)

    # Clean extra spaces/newlines
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clamp_effects(text: str) -> str:
    """
    Cuts only percentage values to a realistic range.
    Does NOT touch money, troops, equipment, etc.
    """
    def repl(m: re.Match) -> str:
        raw = m.group(1).replace(",", ".")
        try:
            v = float(raw)
        except Exception:
            return m.group(0)

        # Hard guard. Keeps decimals.
        if v > 8:
            v = 8
        elif v < -8:
            v = -8

        if abs(v) < 0.05:
            return "0%"
        if float(v).is_integer():
            v_int = int(v)
            return f"+{v_int}%" if v_int > 0 else f"{v_int}%"
        s = f"{v:.1f}".replace(".", ",")
        return f"+{s}%" if v > 0 else f"{s}%"

    return re.sub(r"(?<![\d])([+-]?\d+(?:[\.,]\d+)?)%", repl, text)


def fix_country_line_in_text(text: str, fallback_countries: list[str]) -> str:
    countries = clean_countries(fallback_countries)
    if not countries:
        return text

    country_line = "<b>🌍 Страны:</b> " + ", ".join(countries) + "."
    if "<b>🌍 Страны:</b>" in text:
        text = re.sub(r"<b>🌍 Страны:</b>[^\n]*", country_line, text)
    return text


REQUIRED_BLOCKS = [
    "📌 Вердикт",
    "🌍 Страны",
    "➕ Плюсы",
    "➖ Минусы",
    "📈 Эффект",
    "💰 Изменения по странам",
    "⚠️ Реализм",
]


def is_complete_verdict(text: str) -> bool:
    plain = re.sub(r"<[^>]+>", "", text or "")
    return all(block in plain for block in REQUIRED_BLOCKS) and len(plain) > 350


def trim_for_telegram(text: str, limit: int = 3900) -> str:
    if len(text) <= limit:
        return text

    cut = text[:limit]
    last_break = cut.rfind("\n")
    if last_break > 500:
        cut = cut[:last_break]
    return cut.rstrip() + "…"


def safe_telegram_html(text: str) -> str:
    """
    Telegram HTML is strict. This keeps only <b> and </b>,
    removes broken/unknown tags, and balances bold tags.
    """
    text = str(text or "")

    # Remove all tags except <b> and </b>
    text = re.sub(r"<(?!/?b\s*>)[^>]*>", "", text, flags=re.IGNORECASE)

    # Normalize b tags
    text = re.sub(r"<\s*b\s*>", "<b>", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/\s*b\s*>", "</b>", text, flags=re.IGNORECASE)

    # Escape stray < or > that are not part of <b> tags
    text = re.sub(r"<(?!/?b>)", "&lt;", text)
    text = re.sub(r"(?<!<b)(?<!</b)>", "&gt;", text)

    # Balance <b> tags
    opens = len(re.findall(r"<b>", text))
    closes = len(re.findall(r"</b>", text))

    if closes > opens:
        extra = closes - opens
        for _ in range(extra):
            text = text.replace("</b>", "", 1)

    opens = len(re.findall(r"<b>", text))
    closes = len(re.findall(r"</b>", text))
    if opens > closes:
        text += "</b>" * (opens - closes)

    return text


def strip_all_html(text: str) -> str:
    return re.sub(r"<[^>]*>", "", str(text or "")).strip()


def _fmt_signed(n) -> str:
    try:
        v = int(n)
    except Exception:
        return "?"
    return f"+{v}" if v > 0 else str(v)


def append_history(meta: dict) -> None:
    countries = clean_countries(meta.get("countries") or [])
    if not countries:
        return

    base = {
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "tone": meta.get("tone", "нейтральный"),
        "score": meta.get("score", 5),
        "summary": (meta.get("summary") or "")[:200],
        "effect": meta.get("effect") or {},
    }

    for c in countries:
        entry = dict(base)
        bucket = _country_history.setdefault(c, [])
        bucket.append(entry)
        if len(bucket) > HISTORY_MAX_PER_COUNTRY:
            del bucket[: len(bucket) - HISTORY_MAX_PER_COUNTRY]
    save_history()


async def generate_verdict(news_text: str) -> str:
    extracted_countries = await quick_extract_countries(news_text)

    prompt_text = (
        "Сделай вердикт строго по этому посту. "
        "Не используй шаблоны из прошлых ответов. "
        "Не придумывай нефть/газ/баррели/энергетику, если их нет в тексте.\n\n"
        f"Пост:\n{news_text[:4000]}"
    )

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])]

    last_text = ""
    last_meta: dict = {}

    for temp in (0.35, 0.55):
        raw = await _gemini_call(
            contents=contents,
            system_instruction=SYSTEM_PROMPT,
            temperature=temp,
            max_tokens=2200,
            attempts=3,
        )

        text, meta = parse_meta(raw)
        text = fix_verdict_format(text)
        text = clamp_effects(text)

        model_countries = []
        if meta and isinstance(meta.get("countries"), list):
            model_countries = meta.get("countries") or []

        all_countries = clean_countries(extracted_countries + model_countries)
        text = fix_country_line_in_text(text, all_countries)

        last_text, last_meta = text, meta

        # Regenerate if model somehow gives old template garbage
        lower = text.lower()
        garbage = any(
            phrase in lower
            for phrase in [
                "экспортная выручка",
                "поставки +30 тыс баррелей",
                "умеренный экономический и дипломатический плюс",
                "административные расходы +700 тыс",
                "управляемый эффект",
            ]
        )

        if is_complete_verdict(text) and not garbage:
            if meta:
                if all_countries:
                    meta["countries"] = all_countries
                append_history(meta)
            return text.strip()

    # Last fallback: send model result, but still cleaned.
    if last_meta:
        if extracted_countries:
            last_meta["countries"] = extracted_countries
        append_history(last_meta)

    return last_text.strip()


async def generate_chat_reply(user_text: str, user_name: str | None = None, history: str = "") -> str:
    intro = f"Пользователь {user_name} пишет тебе:" if user_name else "Сообщение тебе:"
    parts_text = ""
    if history:
        parts_text += "Контекст последних сообщений в чате (для поддержания темы, не цитируй):\n" + history + "\n\n"
    parts_text += f"{intro}\n\n{user_text}"
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=parts_text)])]
    return await _gemini_call(contents=contents, system_instruction=CHAT_PROMPT, temperature=0.9, max_tokens=512, attempts=2)


# =========================
# TELEGRAM HELPERS
# =========================

def has_trigger(text: str) -> bool:
    return TRIGGER_HASHTAG in (text or "").lower()


def extract_news_text(message: Message) -> str:
    raw = message.text or message.caption or ""
    cleaned = re.sub(re.escape(TRIGGER_HASHTAG), "", raw, flags=re.IGNORECASE)
    return cleaned.strip()


def is_target_channel(message: Message) -> bool:
    chat = message.chat
    if isinstance(CHANNEL_ID, int):
        return chat.id == CHANNEL_ID
    if not CHANNEL_ID:
        return False
    target = str(CHANNEL_ID).lstrip("@").lower()
    return bool(chat.username and chat.username.lower() == target)


async def _resolve_verdict_thread(bot: Bot) -> Optional[int]:
    global VERDICT_THREAD_ID
    if VERDICT_THREAD_ID is not None:
        return VERDICT_THREAD_ID
    if _same_chat():
        return None
    return None


async def _send_verdict(bot: Bot, source_message: Message, text: str) -> None:
    target_chat_id = VERDICT_CHANNEL_ID or CHANNEL_ID
    log.info("sending verdict to chat_id=%s thread=%s", target_chat_id, VERDICT_THREAD_ID)

    safe_text = safe_telegram_html(trim_for_telegram(text))

    kwargs = {
        "chat_id": target_chat_id,
        "text": safe_text,
        "parse_mode": ParseMode.HTML,
    }

    thread_id = await _resolve_verdict_thread(bot)
    if thread_id is not None:
        kwargs["message_thread_id"] = thread_id

    try:
        await bot.send_message(**kwargs)
    except Exception as exc:
        log.error("HTML send failed, retrying without HTML: %s", exc)
        kwargs["text"] = strip_all_html(safe_text)
        kwargs["parse_mode"] = None
        await bot.send_message(**kwargs)


def _auth_required(message: Message) -> bool:
    user = message.from_user
    return is_authorized(user.id if user else None)


# =========================
# ADMIN UI
# =========================

def _admin_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="📊 Глобальная сводка", callback_data="adm:stats"),
                InlineKeyboardButton(text="📚 Список стран", callback_data="adm:countries"),
            ],
            [
                InlineKeyboardButton(text="📥 Канал-источник", callback_data="adm:channel"),
                InlineKeyboardButton(text="📤 Канал-вердикты", callback_data="adm:vchannel"),
            ],
            [
                InlineKeyboardButton(text="👤 Кто я", callback_data="adm:whoami"),
                InlineKeyboardButton(text="🤖 Статус", callback_data="adm:status"),
            ],
            [InlineKeyboardButton(text="🧹 Очистить ВСЮ память", callback_data="adm:wipe_confirm")],
            [InlineKeyboardButton(text="❌ Закрыть", callback_data="adm:close")],
        ]
    )


def _admin_panel_text() -> str:
    countries = len(_country_history)
    verdicts = sum(len(v) for v in _country_history.values())
    same = _same_chat()
    vline = f"• Канал-вердикты: <code>{VERDICT_CHANNEL_ID}</code>" + (" (= источник)" if same else "")
    tline = f"• Тема вердиктов: <code>{VERDICT_TOPIC_NAME}</code>" + (
        f" / thread <code>{VERDICT_THREAD_ID}</code>" if VERDICT_THREAD_ID else " (без темы)"
    )
    return (
        "<b>🛰 ЦЕНТР УПРАВЛЕНИЯ</b>\n"
        "<i>ИИ ВЕРДИКТ • WORLD OF RETURN</i>\n\n"
        f"• Канал-источник: <code>{CHANNEL_ID}</code>\n"
        f"{vline}\n"
        f"{tline}\n"
        f"• Стран в памяти: <b>{countries}</b>\n"
        f"• Записей вердиктов: <b>{verdicts}</b>\n"
        f"• Авторизовано юзеров: <b>{len(AUTHORIZED_USERS)}</b>\n\n"
        "выбирай раздел:"
    )


router = Router()


# =========================
# CHANNEL HANDLERS
# =========================

@router.channel_post()
async def on_channel_post(message: Message, bot: Bot) -> None:
    if not is_target_channel(message):
        log.info("ignored channel post from chat_id=%s username=%s target=%s", message.chat.id, message.chat.username, CHANNEL_ID)
        return

    raw = message.text or message.caption or ""
    if not has_trigger(raw):
        return

    news = extract_news_text(message)
    if len(news) < 5:
        return

    try:
        verdict = await generate_verdict(news)
        await _send_verdict(bot, message, verdict)
    except Exception:
        log.exception("failed to generate/post verdict")
        try:
            await bot.send_message(
                chat_id=message.chat.id,
                text="⚠️ Не удалось сгенерировать вердикт, попробуйте позже.",
                reply_to_message_id=message.message_id,
            )
        except Exception:
            pass


@router.edited_channel_post()
async def on_edited_channel_post(message: Message, bot: Bot) -> None:
    await on_channel_post(message, bot)


# =========================
# PRIVATE COMMANDS
# =========================

@router.message(Command("start"), F.chat.type == "private")
async def on_start_cmd(message: Message, bot: Bot) -> None:
    user = message.from_user
    if user and is_authorized(user.id):
        await message.answer("привет. кинь текст новости — выдам вердикт. /admin — панель.")
    else:
        await message.answer("привет. для доступа введи код")


@router.message(Command("channel"), F.chat.type == "private")
async def on_channel_cmd(message: Message, bot: Bot) -> None:
    if not _auth_required(message):
        await message.answer("введи код доступа")
        return
    await message.answer(
        f"текущий канал:\n<code>{_stored_channel_raw}</code>\n"
        f"(внутренний id: <code>{CHANNEL_ID}</code>)\n\n"
        f"сменить: <code>/setchannel @username</code> или <code>/setchannel -1001234567890</code>"
    )


@router.message(Command("setchannel"), F.chat.type == "private")
async def on_setchannel_cmd(message: Message, bot: Bot) -> None:
    if not _auth_required(message):
        await message.answer("введи код доступа")
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.answer("формат: <code>/setchannel @username</code>\nили <code>/setchannel -1001234567890</code>")
        return

    parsed, raw = set_channel(parts[1].strip())
    await message.answer(
        f"канал обновлён.\n\nсохранено: <code>{raw}</code>\n"
        f"внутренний id: <code>{parsed}</code>\n\n"
        "не забудь добавить бота админом канала."
    )


@router.message(Command("verdictchannel"), F.chat.type == "private")
async def on_verdict_channel_cmd(message: Message, bot: Bot) -> None:
    if not _auth_required(message):
        await message.answer("введи код доступа")
        return

    same = _same_chat()
    suffix = " (тот же что источник)" if same else ""
    await message.answer(
        f"<b>📤 Канал для вердиктов</b>\n"
        f"• Текущий: <code>{_stored_verdict_raw}</code>{suffix}\n"
        f"• Внутренний id: <code>{VERDICT_CHANNEL_ID}</code>\n"
        f"• Thread: <code>{VERDICT_THREAD_ID if VERDICT_THREAD_ID is not None else 'none'}</code>\n\n"
        f"сменить: <code>/setverdictchannel @username</code> или ID\n"
        f"сбросить: <code>/setverdictchannel same</code>"
    )


@router.message(Command("setverdictchannel"), F.chat.type == "private")
async def on_set_verdict_channel_cmd(message: Message, bot: Bot) -> None:
    if not _auth_required(message):
        await message.answer("введи код доступа")
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.answer(
            "формат: <code>/setverdictchannel @username</code>\n"
            "или <code>/setverdictchannel -1001234567890</code>\n"
            "или <code>/setverdictchannel same</code>"
        )
        return

    arg = parts[1].strip()
    if arg.lower() in {"same", "=", "src", "source", "оба"}:
        parsed, raw = set_verdict_channel(_stored_channel_raw)
    else:
        parsed, raw = set_verdict_channel(arg)

    await message.answer(
        f"канал для вердиктов обновлён.\n\n"
        f"сохранено: <code>{raw}</code>\n"
        f"внутренний id: <code>{parsed}</code>\n\n"
        "не забудь добавить бота админом этого канала."
    )


@router.message(Command("setthread"), F.chat.type == "private")
async def on_set_thread_cmd(message: Message, bot: Bot) -> None:
    if not _auth_required(message):
        await message.answer("введи код доступа")
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("формат: <code>/setthread 13786</code> или <code>/setthread none</code>")
        return

    arg = parts[1].strip().lower()
    if arg in {"none", "нет", "off", "0"}:
        set_verdict_thread(None)
        await message.answer("thread отключён.")
        return

    try:
        thread_id = int(arg)
    except ValueError:
        await message.answer("thread должен быть числом.")
        return

    set_verdict_thread(thread_id)
    await message.answer(f"thread установлен: <code>{thread_id}</code>")


@router.message(Command("history"), F.chat.type == "private")
async def on_history_cmd(message: Message, bot: Bot) -> None:
    if not _auth_required(message):
        await message.answer("введи код доступа")
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        if not _country_history:
            await message.answer("памяти пока нет.")
            return
        countries = sorted(_country_history.keys())
        await message.answer(
            f"страны в памяти ({len(countries)}):\n"
            + ", ".join(f"<code>{c}</code>" for c in countries)
            + "\n\nформат: <code>/history Россия</code>"
        )
        return

    country = normalize_country(parts[1])
    entries = _country_history.get(country)
    if not entries:
        await message.answer(f"нет записей по «{country}».")
        return

    lines = [f"<b>📚 История по «{country}»</b> — всего {len(entries)} запис(ей):\n"]
    for e in entries[-HISTORY_MAX_PER_COUNTRY:]:
        eff_total = (e.get("effect") or {}).get("total")
        eff_str = f" • эффект {_fmt_signed(eff_total)}" if eff_total is not None else ""
        lines.append(f"• {e.get('date','?')} — {e.get('tone','?')}, {e.get('score','?')}/10{eff_str} — {e.get('summary','')}")
    await message.answer(trim_for_telegram("\n".join(lines)))


@router.message(Command("stats"), F.chat.type == "private")
async def on_stats_cmd(message: Message, bot: Bot) -> None:
    if not _auth_required(message):
        await message.answer("введи код доступа")
        return

    if not _country_history:
        await message.answer("памяти пока нет.")
        return

    verdicts = sum(len(v) for v in _country_history.values())
    await message.answer(
        f"<b>📊 Глобальная сводка</b>\n\n"
        f"• Стран в памяти: <b>{len(_country_history)}</b>\n"
        f"• Всего записей вердиктов: <b>{verdicts}</b>"
    )


@router.message(Command("forget"), F.chat.type == "private")
async def on_forget_cmd(message: Message, bot: Bot) -> None:
    if not _auth_required(message):
        await message.answer("введи код доступа")
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("формат: <code>/forget Россия</code> или <code>/forget all</code>")
        return

    arg = parts[1].strip()
    if arg.lower() == "all":
        _country_history.clear()
        save_history()
        await message.answer("память по странам очищена полностью.")
        return

    country = normalize_country(arg)
    if country in _country_history:
        del _country_history[country]
        save_history()
        await message.answer(f"стёр память по «{country}».")
    else:
        await message.answer(f"нет записей по «{country}».")


@router.message(Command("admin"), F.chat.type == "private")
async def on_admin_cmd(message: Message, bot: Bot) -> None:
    if not _auth_required(message):
        await message.answer("введи код доступа")
        return
    await message.answer(_admin_panel_text(), reply_markup=_admin_keyboard())


@router.message(Command("whoami"), F.chat.type == "private")
async def on_whoami_cmd(message: Message, bot: Bot) -> None:
    user = message.from_user
    if not user:
        return
    status = "авторизован" if is_authorized(user.id) else "не авторизован"
    await message.answer(f"твой id: <code>{user.id}</code>\nстатус: {status}")


# =========================
# ADMIN CALLBACKS
# =========================

async def _admin_show(call: CallbackQuery, text: str) -> None:
    kb = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="↩ Назад в панель", callback_data="adm:back")]]
    )
    try:
        await call.message.edit_text(text, reply_markup=kb)
    except Exception:
        await call.message.answer(text, reply_markup=kb)
    await call.answer()


@router.callback_query(F.data.startswith("adm:"))
async def on_admin_callback(call: CallbackQuery, bot: Bot) -> None:
    user = call.from_user
    if not user or not is_authorized(user.id):
        await call.answer("нет доступа", show_alert=True)
        return

    action = (call.data or "").split(":", 1)[1]

    if action == "close":
        try:
            await call.message.delete()
        except Exception:
            pass
        await call.answer("закрыто")
        return

    if action == "stats":
        await _admin_show(call, _admin_panel_text())
        return

    if action == "countries":
        if not _country_history:
            text = "стран в памяти нет."
        else:
            countries = sorted(_country_history.keys())
            text = f"<b>📚 Стран в памяти: {len(countries)}</b>\n\n" + ", ".join(f"<code>{c}</code>" for c in countries)
        await _admin_show(call, text)
        return

    if action == "channel":
        text = (
            f"<b>📥 Канал-источник</b>\n"
            f"• Сохранено: <code>{_stored_channel_raw}</code>\n"
            f"• Внутренний id: <code>{CHANNEL_ID}</code>\n"
            f"• Триггер-хештег: <code>{TRIGGER_HASHTAG}</code>"
        )
        await _admin_show(call, text)
        return

    if action == "vchannel":
        same = _same_chat()
        suffix = " (= источник)" if same else ""
        text = (
            f"<b>📤 Канал-вердикты</b>\n"
            f"• Сохранено: <code>{_stored_verdict_raw}</code>{suffix}\n"
            f"• Внутренний id: <code>{VERDICT_CHANNEL_ID}</code>\n"
            f"• Thread: <code>{VERDICT_THREAD_ID if VERDICT_THREAD_ID is not None else 'none'}</code>"
        )
        await _admin_show(call, text)
        return

    if action == "whoami":
        await _admin_show(
            call,
            f"<b>👤 Ты</b>\n"
            f"• id: <code>{user.id}</code>\n"
            f"• имя: {user.first_name or '—'}\n"
            f"• username: @{user.username or '—'}\n"
            f"• статус: авторизован",
        )
        return

    if action == "status":
        await _admin_show(
            call,
            f"<b>🤖 Статус бота</b>\n"
            f"• Модель: <code>{GEMINI_MODEL}</code>\n"
            f"• Канал: <code>{CHANNEL_ID}</code>\n"
            f"• Стран: <b>{len(_country_history)}</b>\n"
            f"• Записей: <b>{sum(len(v) for v in _country_history.values())}</b>\n"
            f"• Авторизовано: <b>{len(AUTHORIZED_USERS)}</b>",
        )
        return

    if action == "wipe_confirm":
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="✅ ДА, стереть всё", callback_data="adm:wipe_do"),
                    InlineKeyboardButton(text="↩ Назад", callback_data="adm:back"),
                ]
            ]
        )
        await call.message.edit_text("<b>⚠️ Точно стереть ВСЮ память по странам?</b>\nэто нельзя отменить.", reply_markup=kb)
        await call.answer()
        return

    if action == "wipe_do":
        _country_history.clear()
        save_history()
        await _admin_show(call, "<b>🧹 Память по странам стёрта.</b>")
        return

    if action == "back":
        try:
            await call.message.edit_text(_admin_panel_text(), reply_markup=_admin_keyboard())
        except Exception:
            pass
        await call.answer()
        return

    await call.answer("неизвестное действие")


# =========================
# PRIVATE TEXT
# =========================

@router.message(F.chat.type == "private")
async def on_private_message(message: Message, bot: Bot) -> None:
    text = (message.text or message.caption or "").strip()
    user = message.from_user
    if not user:
        return

    if not is_authorized(user.id):
        if text.lower().strip() == ACCESS_CODE.lower():
            authorize(user.id)
            await message.answer("доступ открыт. кинь текст новости — будет вердикт")
        else:
            await message.answer("введи код доступа")
        return

    if not text:
        await message.answer(
            f"пришли текст новости.\n\n"
            f"в канале реагирую на посты с хештегом <code>{TRIGGER_HASHTAG}</code>."
        )
        return

    try:
        verdict = await generate_verdict(text)
    except Exception:
        log.exception("failed to generate private verdict")
        await message.answer("⚠️ не удалось сгенерировать вердикт, попробуй позже.")
        return

    try:
        await message.answer(safe_telegram_html(trim_for_telegram(verdict)), parse_mode=ParseMode.HTML)
    except Exception:
        await message.answer(strip_all_html(verdict), parse_mode=None)


# =========================
# GROUP HANDLER
# =========================

def is_bot_addressed(message: Message) -> tuple[bool, str]:
    raw = message.text or message.caption or ""
    cleaned = raw
    addressed = False

    if BOT_USERNAME:
        mention = f"@{BOT_USERNAME}".lower()
        if mention in raw.lower():
            addressed = True
            cleaned = re.sub(re.escape(f"@{BOT_USERNAME}"), "", cleaned, flags=re.IGNORECASE)

    reply = message.reply_to_message
    if reply and reply.from_user and BOT_ID and reply.from_user.id == BOT_ID:
        addressed = True

    return addressed, cleaned.strip()


@router.message()
async def on_group_message(message: Message, bot: Bot) -> None:
    if message.chat.type == "private":
        return

    raw = message.text or message.caption or ""

    if has_trigger(raw):
        news = extract_news_text(message)
        if len(news) < 5:
            return
        try:
            verdict = await generate_verdict(news)
            await _send_verdict(bot, message, verdict)
        except Exception:
            log.exception("failed to handle group verdict")
        return

    addressed, cleaned = is_bot_addressed(message)
    user_name = message.from_user.first_name or message.from_user.username if message.from_user else None

    if raw.strip():
        remember(message.chat.id, "user", user_name or "Пользователь", raw.strip())

    if not addressed:
        return

    level = flood_level(message.chat.id)
    if level == 2:
        return

    if level == 1:
        import random
        brushoff = random.choice(_SHORT_BRUSHOFFS)
        await message.reply(brushoff, parse_mode=None)
        remember(message.chat.id, "bot", "БОТ", brushoff)
        return

    user_text = cleaned
    reply = message.reply_to_message
    if reply and reply.text and reply.from_user and reply.from_user.id == BOT_ID:
        user_text = f"(в ответ на твоё сообщение: «{reply.text[:300]}»)\n\n{cleaned}"

    if not user_text:
        user_text = "(меня просто отметили без текста)"

    history = render_history(message.chat.id)

    try:
        chat_reply = await generate_chat_reply(user_text, user_name, history=history)
    except Exception:
        await message.reply("чёт сломался. потом")
        return

    await message.reply(trim_for_telegram(chat_reply), parse_mode=None)
    remember(message.chat.id, "bot", "БОТ", chat_reply)


# =========================
# MAIN
# =========================

async def main() -> None:
    log.info("starting verdict bot. target channel: %s", CHANNEL_ID)

    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_router(router)

    me = await bot.get_me()
    global BOT_USERNAME, BOT_ID
    BOT_USERNAME = me.username or ""
    BOT_ID = me.id

    log.info("bot online as @%s (id=%s)", me.username, me.id)

    await bot.delete_webhook(drop_pending_updates=True)

    try:
        await dp.start_polling(
            bot,
            allowed_updates=["channel_post", "edited_channel_post", "message", "edited_message", "callback_query"],
        )
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
