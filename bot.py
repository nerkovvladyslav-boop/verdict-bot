from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
from typing import Optional

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.types import Message
from google import genai
from google.genai import types


# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("verdict-bot")


# =========================
# ENV
# =========================

def env(name: str, default: str = "", required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(f"missing required env var: {name}")
    return value


BOT_TOKEN = env("TELEGRAM_BOT_TOKEN", required=True)

# Можно указать один ключ GEMINI_API_KEY или много ключей GEMINI_KEYS=key1,key2,key3
GEMINI_API_KEY = env("AI_INTEGRATIONS_GEMINI_API_KEY") or env("GEMINI_API_KEY", "")
GEMINI_KEYS_RAW = env("GEMINI_KEYS", "")
GEMINI_KEYS = [k.strip() for k in GEMINI_KEYS_RAW.split(",") if k.strip()]
if not GEMINI_KEYS and GEMINI_API_KEY:
    GEMINI_KEYS = [GEMINI_API_KEY]
if not GEMINI_KEYS:
    raise RuntimeError("missing Gemini API key: set GEMINI_API_KEY or GEMINI_KEYS")

GEMINI_BASE_URL = env("AI_INTEGRATIONS_GEMINI_BASE_URL") or env("GEMINI_BASE_URL")
GEMINI_MODEL = env("GEMINI_MODEL", "gemini-2.5-flash")

# fallback под твой старый канал
TELEGRAM_CHANNEL_ID_RAW = env("TELEGRAM_CHANNEL_ID", "-1003993603387").strip()
TELEGRAM_VERDICT_CHANNEL_ID_RAW = env("TELEGRAM_VERDICT_CHANNEL_ID", TELEGRAM_CHANNEL_ID_RAW).strip()
VERDICT_THREAD_ID_RAW = env("VERDICT_THREAD_ID", "13786").strip()

TRIGGER_HASHTAG = "#дайтеверд"

BOT_USERNAME = ""
BOT_ID = 0

CURRENT_GEMINI_KEY_INDEX = 0
VERDICT_CACHE: dict[str, str] = {}
LAST_429_UNTIL: float = 0.0


# =========================
# TELEGRAM IDS
# =========================

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


CHANNEL_ID = parse_channel_id(TELEGRAM_CHANNEL_ID_RAW)
VERDICT_CHANNEL_ID = parse_channel_id(TELEGRAM_VERDICT_CHANNEL_ID_RAW)

try:
    VERDICT_THREAD_ID: Optional[int] = (
        int(VERDICT_THREAD_ID_RAW)
        if VERDICT_THREAD_ID_RAW and VERDICT_THREAD_ID_RAW.lower() not in {"0", "none", "off", "false"}
        else None
    )
except ValueError:
    VERDICT_THREAD_ID = None


# =========================
# GEMINI
# =========================

def make_genai_client(api_key: str) -> genai.Client:
    kwargs = {"api_key": api_key}
    if GEMINI_BASE_URL:
        kwargs["http_options"] = {"base_url": GEMINI_BASE_URL, "api_version": ""}
    return genai.Client(**kwargs)


def current_key_label() -> str:
    return f"{CURRENT_GEMINI_KEY_INDEX + 1}/{len(GEMINI_KEYS)}"


SYSTEM_PROMPT = """
Ты — аналитический бот «ИИ ВЕРДИКТ • WORLD OF RETURN».
Это RP-канал реальных стран. Участники пишут посты от лица государств.
Твоя задача — выдать короткий вердикт строго по конкретному посту.

ГЛАВНЫЕ ПРАВИЛА:
- Анализируй смысл всего текста, а не отдельные слова.
- Не используй шаблоны и не повторяй старые ответы.
- Не придумывай страны, события, нефть, газ, баррели, экспорт, энергетику, если этого нет в посте.
- РФ и Россия — одно государство: пиши Россия.
- Не включай как страну: Арктика, Севморпуть, регион, граница, пролив, Балтика.
- ЕС и НАТО можно указывать, если они прямо затронуты.
- Если есть войска/границы/базы/учения/ПВО/ракеты — отрази военный аспект.
- Если есть IT/стартапы/СЭЗ/налоги — отрази экономико-технологический аспект.
- Если есть договор/нейтрализация/пролив — отрази дипломатический и военный риск.
- В блоке "Изменения по странам" не подставляй одинаковые строки всем участникам.

ФОРМАТ СТРОГО БЕЗ HTML И MARKDOWN:

📌 Вердикт
Одно конкретное предложение: кто что сделал и к чему это ведёт.

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
• Страна: конкретное материальное изменение ±N единица; второе конкретное изменение ±N единица — причина.
• Страна: конкретное материальное изменение ±N единица; второе конкретное изменение ±N единица — причина.

⚠️ Реализм
Кратко: насколько это по силам стране и какие ограничения.

🤝 Ответ на ультиматум/требование
Если требования нет — Не применимо.

ПРАВИЛА ДЛЯ "💰 Изменения по странам":
- Пиши разные последствия для разных стран/организаций.
- Для инициатора: расходы, техника, люди, бюджеты, контракты, объекты.
- Для затронутой стороны: потери, ответные расходы, усиление контроля, санкции, помощь, проверки.
- Не пиши абстракции: влияние, репутация, доверие, стабильность, политический вес.
- Не пиши "много", "значительно", "сильно", "несколько", "десятки", "сотни" без числа.
- Каждый показатель обязан иметь число и единицу.
- Примеры единиц: $, млн $, чел., ед., объектов, км, часов, контрактов, систем, МВт, тонн, литров, м³.
- Если точных данных нет — оцени приблизительно, но реалистично.

РЕАЛИСТИЧНЫЕ ДИАПАЗОНЫ:
- мелкая программа: 100 тыс–5 млн $
- средняя программа: 5–80 млн $
- крупная реформа/инфраструктура: 80–700 млн $
- военная переброска: 300–8000 чел., 10–250 ед. техники
- ракетный удар: 10–500 ракет/БПЛА, ущерб 10 млн–3 млрд $
- IT/стартапы: 5–200 млн $, 10–300 компаний/проектов
- дипломатическое соглашение: 1–20 млн $ админрасходов, 2–20 проверок/миссий/комиссий

ОГРАНИЧЕНИЯ:
- До 2300 символов.
- Проценты обычно от -1% до +3%.
- Больше ±5% только при войне/кризисе.
- Не используй -50%, -90%, -100%.
"""


async def gemini_call(prompt: str, temperature: float = 0.35, max_tokens: int = 2400) -> str:
    """
    Multi-key anti-limit call.
    Пробует каждый ключ один раз. Если 429/503 — переключает ключ.
    """
    global CURRENT_GEMINI_KEY_INDEX

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

    last_err = None

    for _ in range(len(GEMINI_KEYS)):
        api_key = GEMINI_KEYS[CURRENT_GEMINI_KEY_INDEX]
        client = make_genai_client(api_key)

        try:
            log.info("Gemini request using key %s model=%s", current_key_label(), GEMINI_MODEL)
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            )
            text = (response.text or "").strip()
            if text:
                return text
            last_err = RuntimeError("empty Gemini response")
        except Exception as exc:
            last_err = exc
            err = str(exc)
            if (
                "429" in err
                or "RESOURCE_EXHAUSTED" in err
                or "Too Many Requests" in err
                or "quota" in err.lower()
                or "503" in err
                or "UNAVAILABLE" in err
                or "Service Unavailable" in err
            ):
                log.warning("Gemini key %s failed/exhausted/unavailable, switching key: %s", current_key_label(), exc)
                CURRENT_GEMINI_KEY_INDEX = (CURRENT_GEMINI_KEY_INDEX + 1) % len(GEMINI_KEYS)
                continue

            log.warning("Gemini key %s failed, switching key: %s", current_key_label(), exc)
            CURRENT_GEMINI_KEY_INDEX = (CURRENT_GEMINI_KEY_INDEX + 1) % len(GEMINI_KEYS)
            continue

    raise RuntimeError(f"all Gemini keys failed/exhausted: {last_err}")


# =========================
# TEXT POSTPROCESSING
# =========================

def strip_html(text: str) -> str:
    return re.sub(r"<[^>]*>", "", str(text or "")).strip()


def soft_fix(text: str) -> str:
    """
    Мягкая чистка. НЕ режет блоки, НЕ пересобирает вердикт.
    Исправляет только мусор типа 'Страны: :.' и повторов заголовков.
    """
    text = strip_html(text)

    # убрать META/json/markdown если модель случайно дала
    text = re.sub(r"<META>.*?</META>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("```", "")

    # типовые дубли/мусор
    text = text.replace("🌍 Страны\n", "🌍 Страны: ")
    text = text.replace("🌍 Страны: :.", "🌍 Страны:")
    text = text.replace("🌍 Страны: :", "🌍 Страны:")
    text = text.replace("📈 Эффект (%)\n(%)", "📈 Эффект (%)")
    text = text.replace("🤝 Ответ на ультиматум/требование\nна ультиматум/требование", "🤝 Ответ на ультиматум/требование")

    # убрать одиночные строки мусора
    text = re.sub(r"(?m)^\s*:\s*$", "", text)
    text = re.sub(r"(?m)^\s*\(%\)\s*$", "", text)
    text = re.sub(r"(?m)^\s*на ультиматум/требование\s*$", "", text, flags=re.IGNORECASE)

    # если после строки стран отдельной строкой повторился тот же список — убрать
    lines = text.splitlines()
    out = []
    last_country_payload = None
    just_country = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("🌍 Страны:"):
            payload = stripped.replace("🌍 Страны:", "").strip().strip(".")
            last_country_payload = payload
            just_country = True
            out.append(line)
            continue
        if just_country:
            candidate = stripped.strip(".")
            if candidate == ":" or (last_country_payload and candidate == last_country_payload):
                just_country = False
                continue
            just_country = False
        out.append(line)
    text = "\n".join(out)

    text = clean_country_line(text)

    # мягко заменить ленивые "много"
    text = re.sub(r"[-+]?много\s*%", "-5%", text, flags=re.IGNORECASE)
    text = re.sub(r"[-+]?значительно\s*%", "-4%", text, flags=re.IGNORECASE)
    text = re.sub(r"[-+]?сильно\s*%", "-4%", text, flags=re.IGNORECASE)
    text = re.sub(r"[-+]?много\s*п\.п\.", "-4 п.п.", text, flags=re.IGNORECASE)
    text = re.sub(r"[-+]?много\s*ед\.?", "-50 ед.", text, flags=re.IGNORECASE)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_country_line(text: str) -> str:
    m = re.search(r"(🌍 Страны:\s*)([^\n]+)", text)
    if not m:
        return text

    prefix, raw = m.group(1), m.group(2)
    raw = raw.replace(":", " ")
    parts = [p.strip().strip(".") for p in raw.split(",")]

    banned = {
        "арктика", "севморпуть", "северный морской путь", "регион",
        "граница", "границы", "пролив", "балтика", "балтийский пролив",
        "страны", "страна", "государство"
    }

    fixed = []
    for p in parts:
        if not p:
            continue
        if p in {"РФ", "Российская Федерация"}:
            p = "Россия"
        if p.lower() in banned:
            continue
        if p not in fixed:
            fixed.append(p)

    if not fixed:
        return text

    new_line = prefix + ", ".join(fixed) + "."
    return text[:m.start()] + new_line + text[m.end():]


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


def cache_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())[:1500]


def is_obviously_bad(text: str) -> bool:
    if len(text.strip()) < 120:
        return True
    bad = [
        "Россия проводит военное усиление, повышая давление на региональную безопасность",
        "военные расходы +45 млн $; личный состав +2500 чел",
        "административные расходы +700 тыс",
        "экспортная выручка +8 млн",
        "поставки +30 тыс баррелей",
        "ограниченный управляемый эффект",
        "🌍 Страны: :",
    ]
    low = text.lower()
    return any(x.lower() in low for x in bad)



REQUIRED_VERDICT_BLOCKS = [
    "📌 Вердикт",
    "🌍 Страны",
    "➕ Плюсы",
    "➖ Минусы",
    "📈 Эффект",
    "💰 Изменения по странам",
    "⚠️ Реализм",
    "🤝 Ответ",
]


def verdict_is_incomplete(text: str) -> bool:
    if not text or len(text.strip()) < 500:
        return True

    plain = strip_html(text)
    missing = [b for b in REQUIRED_VERDICT_BLOCKS if b not in plain]
    if missing:
        return True

    # Looks cut off if last required block exists but has no actual answer
    tail = plain.split("🤝 Ответ", 1)[-1].strip() if "🤝 Ответ" in plain else ""
    if len(tail) < 12:
        return True

    # Common signs of truncation
    if plain.rstrip().endswith(("•", ":", ",", ";", "—", "-", "и", "или")):
        return True

    return False



# =========================
# SMART FALLBACK WITHOUT AI
# =========================

COUNTRY_WORDS = {
    "россия": "Россия", "рф": "Россия",
    "сша": "США", "америка": "США",
    "китай": "Китай", "кнр": "Китай",
    "тайвань": "Тайвань",
    "иран": "Иран",
    "израиль": "Израиль",
    "украина": "Украина",
    "беларусь": "Беларусь", "белоруссия": "Беларусь",
    "польша": "Польша",
    "германия": "Германия",
    "франция": "Франция",
    "норвегия": "Норвегия",
    "турция": "Турция",
    "алжир": "Алжир",
    "тунис": "Тунис",
    "нато": "НАТО",
    "ес": "ЕС",
    "оаэ": "ОАЭ",
    "бахрейн": "Бахрейн",
}


def fallback_countries(text: str) -> list[str]:
    low = text.lower()
    found = []
    for key, val in COUNTRY_WORDS.items():
        if re.search(rf"(?<![а-яa-z]){re.escape(key)}(?![а-яa-z])", low):
            if val not in found:
                found.append(val)
    return found[:5] or ["Основная сторона"]


def fallback_event_type(text: str) -> str:
    low = text.lower()
    if any(w in low for w in ["ракета", "ракет", "удар", "обстрел", "пво", "дрон", "бпла", "шахед", "бомб"]):
        return "strike"
    if any(w in low for w in ["войск", "границ", "переброс", "база", "арм", "учени", "учен", "бригада", "батальон", "дивиз", "техника", "мобилиз"]):
        return "military"
    if any(w in low for w in ["протест", "митинг", "забастов", "беспоряд", "демонстрац"]):
        return "protest"
    if any(w in low for w in ["договор", "соглашен", "нейтрализац", "ультиматум", "переговор", "санкц"]):
        return "diplomacy"
    if any(w in low for w in ["it", "айти", "стартап", "цифров", "техно", "сэз", "компан", "инновац"]):
        return "tech"
    if any(w in low for w in ["эконом", "бюджет", "налог", "бизнес", "мсб", "инвест", "банк"]):
        return "economy"
    if any(w in low for w in ["газ", "нефт", "энерг", "электро", "мвт", "трубопровод"]):
        return "energy"
    if any(w in low for w in ["арктик", "севморпуть", "порт", "логист"]):
        return "geopolitics"
    return "general"


def fallback_effects(event_type: str) -> tuple[dict, str]:
    table = {
        "strike": ({"Экономика": -3, "Военка": 4, "Политика": -3, "Общество": -4, "Дипломатия": -5, "ИТОГ": -4}, "военная эскалация с серьёзными рисками"),
        "military": ({"Экономика": -1, "Военка": 3, "Политика": 1, "Общество": -1, "Дипломатия": -2, "ИТОГ": 0}, "военное усиление при росте напряжённости"),
        "protest": ({"Экономика": -1, "Военка": 0, "Политика": -3, "Общество": -3, "Дипломатия": -1, "ИТОГ": -2}, "внутриполитическое давление растёт"),
        "diplomacy": ({"Экономика": 0, "Военка": -1, "Политика": 1, "Общество": 0, "Дипломатия": 2, "ИТОГ": 1}, "дипломатический сдвиг с рисками доверия"),
        "tech": ({"Экономика": 2, "Военка": 0, "Политика": 1, "Общество": 1, "Дипломатия": 1, "ИТОГ": 1}, "технологическое усиление с умеренным эффектом"),
        "economy": ({"Экономика": 2, "Военка": 0, "Политика": 1, "Общество": 1, "Дипломатия": 0, "ИТОГ": 1}, "умеренный экономический эффект"),
        "energy": ({"Экономика": 2, "Военка": 0, "Политика": 1, "Общество": 0, "Дипломатия": 1, "ИТОГ": 1}, "энергетический эффект без резкого перелома"),
        "geopolitics": ({"Экономика": 1, "Военка": 2, "Политика": 1, "Общество": 0, "Дипломатия": -1, "ИТОГ": 1}, "геополитическое укрепление с риском напряжения"),
        "general": ({"Экономика": 1, "Военка": 0, "Политика": 1, "Общество": 0, "Дипломатия": 0, "ИТОГ": 1}, "ограниченный эффект по заявленному действию"),
    }
    return table.get(event_type, table["general"])


def fmt_pct(v) -> str:
    try:
        v = float(v)
    except Exception:
        v = 0
    if abs(v) < 0.05:
        return "0%"
    if v.is_integer():
        n = int(v)
        return f"+{n}%" if n > 0 else f"{n}%"
    s = f"{v:.1f}".replace(".", ",")
    return f"+{s}%" if v > 0 else f"{s}%"


def fallback_change_line(country: str, event_type: str, idx: int) -> str:
    c = country
    if event_type == "strike":
        if idx == 0:
            return f"• {c}: боеприпасы -120 ед.; военные расходы +480 млн $ — проведение ударов."
        return f"• {c}: ущерб инфраструктуре +350 млн $; повреждено объектов +12 ед. — последствия удара."
    if event_type == "military":
        if idx == 0:
            return f"• {c}: военные расходы +55 млн $; личный состав +3200 чел. — переброска сил."
        return f"• {c}: расходы наблюдения +12 млн $; пограничные патрули +450 чел. — ответное усиление."
    if event_type == "protest":
        if idx == 0:
            return f"• {c}: расходы полиции +18 млн $; задержано +600 чел. — массовые протесты."
        return f"• {c}: дипмиссии +4 проверки; расходы безопасности +6 млн $ — реакция на кризис."
    if event_type == "diplomacy":
        if idx == 0:
            return f"• {c}: дипломатические миссии +6 ед.; админрасходы +8 млн $ — оформление соглашения."
        return f"• {c}: проверки безопасности +5 ед.; расходы мониторинга +10 млн $ — контроль договорённостей."
    if event_type == "tech":
        if idx == 0:
            return f"• {c}: инвестиции в технологии +90 млн $; проектов +120 ед. — запуск программы."
        return f"• {c}: закупки ПО +18 млн $; совместных проектов +25 ед. — технологическая интеграция."
    if event_type == "economy":
        if idx == 0:
            return f"• {c}: фонд программы +70 млн $; заявок бизнеса +1400 ед. — экономическое стимулирование."
        return f"• {c}: торговые контракты +35 млн $; проверок +8 ед. — участие в программе."
    if event_type == "energy":
        if idx == 0:
            return f"• {c}: энергетические инвестиции +140 млн $; мощность проектов +220 МВт — расширение сектора."
        return f"• {c}: импортные расходы +30 млн $; резерв мощности +80 МВт — энергетические закупки."
    if event_type == "geopolitics":
        if idx == 0:
            return f"• {c}: арктические расходы +85 млн $; объектов обеспечения +6 ед. — усиление присутствия."
        return f"• {c}: расходы мониторинга +18 млн $; патрульные миссии +10 ед. — реакция на активность."
    if idx == 0:
        return f"• {c}: бюджет инициативы +25 млн $; проектов +18 ед. — реализация действия."
    return f"• {c}: ответные расходы +9 млн $; проверок +6 ед. — адаптация к событию."


def smart_fallback_verdict(news_text: str, reason: str = "") -> str:
    countries = fallback_countries(news_text)
    event_type = fallback_event_type(news_text)
    effects, note = fallback_effects(event_type)
    main = countries[0]

    if event_type == "strike":
        verdict = f"{main} переходит к прямому военному давлению, что резко повышает риск региональной эскалации."
        plus1 = f"{main}: получает краткосрочную военную инициативу."
        minus1 = "Затронутые стороны: получают ущерб инфраструктуре и рост оборонных расходов."
        realism = "Реалистично только при наличии подготовленных сил, запасов боеприпасов и политического решения."
    elif event_type == "military":
        verdict = f"{main} усиливает военное присутствие, повышая контроль направления и давление на соседей."
        plus1 = f"{main}: повышает боеготовность и скорость реагирования."
        minus1 = "Регион: растёт риск инцидентов и ответных мер."
        realism = "Реалистично при наличии снабжения, подготовленных частей и понятной цели переброски."
    elif event_type == "protest":
        verdict = f"{main} сталкивается с внутренним протестным давлением из-за спорного политического решения."
        plus1 = "Оппозиция: получает мобилизационный повод и рост публичной активности."
        minus1 = f"{main}: теряет управляемость и тратит ресурсы на стабилизацию."
        realism = "Реалистично, если решение воспринимается обществом как непрозрачное или противоречивое."
    elif event_type == "diplomacy":
        verdict = f"{main} меняет дипломатическую линию, создавая новый баланс договорённостей и рисков доверия."
        plus1 = f"{main}: получает пространство для манёвра и переговоров."
        minus1 = "Партнёры: требуют гарантий, контроля и дополнительных консультаций."
        realism = "Реалистично при политической воле сторон и наличии механизмов контроля."
    elif event_type == "tech":
        verdict = f"{main} усиливает технологическое направление, пытаясь расширить экономическую самостоятельность."
        plus1 = f"{main}: получает рост инвестиций, проектов и технологической занятости."
        minus1 = "Бюджет: получает нагрузку на субсидии, контроль и инфраструктуру."
        realism = "Реалистично при стабильном финансировании, кадрах и защите инвестиций."
    elif event_type == "economy":
        verdict = f"{main} запускает экономическую инициативу с умеренным эффектом для бизнеса и бюджета."
        plus1 = f"{main}: стимулирует внутреннюю активность и занятость."
        minus1 = "Бюджет: несёт расходы на запуск и контроль программы."
        realism = "Реалистично при поэтапном финансировании и прозрачном администрировании."
    elif event_type == "energy":
        verdict = f"{main} усиливает энергетическое направление, получая экономический эффект и новые обязательства."
        plus1 = f"{main}: получает дополнительную выручку или устойчивость поставок."
        minus1 = "Партнёры: несут расходы на логистику и контрактное сопровождение."
        realism = "Реалистично при наличии инфраструктуры, контрактов и стабильной логистики."
    elif event_type == "geopolitics":
        verdict = f"{main} усиливает геополитическое присутствие, повышая значение стратегического направления."
        plus1 = f"{main}: укрепляет контроль над важной зоной и логистикой."
        minus1 = "Регион: получает рост конкуренции и риска военных инцидентов."
        realism = "Реалистично при наличии баз, снабжения и долгосрочного финансирования."
    else:
        verdict = f"{main} реализует заявленное действие с умеренными политическими и материальными последствиями."
        plus1 = f"{main}: получает ограниченный практический выигрыш."
        minus1 = "Участники: сталкиваются с расходами и рисками реализации."
        realism = "Оценка ограничена, потому что ИИ-модель временно недоступна."

    changes = "\n".join(fallback_change_line(c, event_type, i) for i, c in enumerate(countries))

    return f"""📌 Вердикт
{verdict}

🌍 Страны: {", ".join(countries)}.

➕ Плюсы
• {plus1}
• Реализация заявленного действия создаёт краткосрочный управленческий эффект.

➖ Минусы
• {minus1}
• Возможны дополнительные расходы, политические споры и внешняя реакция.

📈 Эффект (%)
• Экономика: {fmt_pct(effects["Экономика"])}
• Военка: {fmt_pct(effects["Военка"])}
• Политика: {fmt_pct(effects["Политика"])}
• Общество: {fmt_pct(effects["Общество"])}
• Дипломатия: {fmt_pct(effects["Дипломатия"])}
• ИТОГ: {fmt_pct(effects["ИТОГ"])} — {note}.

💰 Изменения по странам
{changes}

⚠️ Реализм
{realism}

🤝 Ответ на ультиматум/требование
Не применимо."""


# =========================
# VERDICT
# =========================

async def generate_verdict(news_text: str) -> str:
    global LAST_429_UNTIL

    key = cache_key(news_text)
    if key in VERDICT_CACHE:
        return VERDICT_CACHE[key]

    now = time.time()
    if LAST_429_UNTIL and now < LAST_429_UNTIL:
        return smart_fallback_verdict(news_text, reason="cooldown")

    prompt = (
        "Сделай вердикт строго по этому посту. "
        "Не используй шаблоны. Не подставляй одинаковые строки всем странам. "
        "Не придумывай энергетику/нефть/баррели, если этого нет.\n\n"
        f"ПОСТ:\n{news_text[:5000]}"
    )

    try:
        raw = await gemini_call(prompt, temperature=0.35, max_tokens=2400)
    except Exception as exc:
        err = str(exc)

        if (
            "429" in err
            or "RESOURCE_EXHAUSTED" in err
            or "Too Many Requests" in err
            or "quota" in err.lower()
        ):
            LAST_429_UNTIL = time.time() + 10 * 60
            log.warning("Gemini quota exhausted, using smart fallback")
            return smart_fallback_verdict(news_text, reason="quota")

        if (
            "503" in err
            or "UNAVAILABLE" in err
            or "Service Unavailable" in err
            or "high demand" in err.lower()
        ):
            log.warning("Gemini unavailable/high demand, using smart fallback")
            return smart_fallback_verdict(news_text, reason="unavailable")

        log.exception("Gemini generation failed, using smart fallback")
        return smart_fallback_verdict(news_text, reason="error")

    text = raw.strip()
    text = soft_fix(text)
    text = clamp_percentages(text)

    if is_obviously_bad(text) or verdict_is_incomplete(text):
        # Не делаем 3 ретрая, чтобы не жечь лимит. Один лёгкий повтор с другой температурой.
        try:
            raw2 = await gemini_call(prompt, temperature=0.55, max_tokens=2400)
            text2 = clamp_percentages(soft_fix(raw2.strip()))
            if not is_obviously_bad(text2):
                text = text2
        except Exception:
            pass

    if len(text) > 50:
        VERDICT_CACHE[key] = text

    return text.strip()


# =========================
# TELEGRAM
# =========================

def has_trigger(text: str) -> bool:
    return TRIGGER_HASHTAG in (text or "").lower()


def extract_news_text(message: Message) -> str:
    raw = message.text or message.caption or ""
    return re.sub(re.escape(TRIGGER_HASHTAG), "", raw, flags=re.IGNORECASE).strip()


def trim_for_telegram(text: str, limit: int = 4050) -> str:
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
        "parse_mode": None,  # без HTML, чтобы Telegram не падал
    }

    if VERDICT_THREAD_ID is not None:
        kwargs["message_thread_id"] = VERDICT_THREAD_ID

    await bot.send_message(**kwargs)


dp = Dispatcher()


@dp.channel_post()
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
        log.exception("failed channel verdict")


@dp.edited_channel_post()
async def on_edited_channel_post(message: Message, bot: Bot) -> None:
    await on_channel_post(message, bot)


@dp.message(Command("start"))
async def on_start(message: Message) -> None:
    await message.answer(
        "бот запущен.\n"
        f"канал: {CHANNEL_ID}\n"
        f"канал вердиктов: {VERDICT_CHANNEL_ID}\n"
        f"thread: {VERDICT_THREAD_ID}\n"
        f"ключей Gemini: {len(GEMINI_KEYS)}\n"
        f"текущий ключ: {current_key_label()}\n"
        f"триггер: {TRIGGER_HASHTAG}"
    )


@dp.message(Command("keys"))
async def on_keys(message: Message) -> None:
    await message.answer(
        f"Gemini keys: {len(GEMINI_KEYS)}\n"
        f"Current key: {current_key_label()}\n"
        f"Model: {GEMINI_MODEL}"
    )


@dp.message(Command("test"))
async def on_test(message: Message, bot: Bot) -> None:
    await message.answer("тест ок")
    try:
        await send_verdict(bot, "тест отправки в канал/ветку")
    except Exception:
        log.exception("test send failed")
        await message.answer("не смог отправить в канал/ветку, смотри Railway Logs")


@dp.message(Command("whoami"))
async def on_whoami(message: Message) -> None:
    user = message.from_user
    await message.answer(f"user_id: {user.id if user else 'unknown'}\nchat_id: {message.chat.id}")


@dp.message()
async def on_any_message(message: Message, bot: Bot) -> None:
    raw = message.text or message.caption or ""

    if has_trigger(raw):
        news = extract_news_text(message)
        if len(news) < 5:
            return
        try:
            verdict = await generate_verdict(news)
            await send_verdict(bot, verdict)
        except Exception:
            log.exception("failed group/private verdict")
        return

    # В личке можно просто отправить текст без хештега
    if message.chat.type == "private" and raw.strip():
        try:
            verdict = await generate_verdict(raw.strip())
            await message.answer(trim_for_telegram(verdict), parse_mode=None)
        except Exception:
            log.exception("failed private verdict")
            await message.answer("ошибка генерации, смотри Railway Logs")


# =========================
# MAIN
# =========================

async def main() -> None:
    global BOT_USERNAME, BOT_ID

    log.info("starting bot")
    log.info("CHANNEL_ID=%s VERDICT_CHANNEL_ID=%s THREAD=%s", CHANNEL_ID, VERDICT_CHANNEL_ID, VERDICT_THREAD_ID)
    log.info("Gemini keys loaded: %s current=%s model=%s", len(GEMINI_KEYS), current_key_label(), GEMINI_MODEL)

    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=None))

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

