import os
import logging
import io
import datetime
import traceback
import html
import re
import psycopg2
from psycopg2.extras import RealDictCursor
from PIL import Image
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types
from openai import OpenAI
from key_router import chat_router, image_router
from google.genai.errors import APIError

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)
logger = logging.getLogger(__name__)

load_dotenv()

# 常量定义
CHARACTER_PHOTO_PATH = "/data/three/media/photos/photo_nanase.jpg"
USER_PHOTO_PATH = "/data/three/media/photos/photo_siyuan.jpg"
ALLOWED_USER_ID = 569020802
AI_SYSTEM_PROMPT = """# Role: 顶级加密货币合约交易大师 (Top-Tier Crypto Futures Master)

## 👤 角色背景
你现在是一名拥有10年经验、胜率极高且纪律严明的华尔街级别加密货币合约交易大师。你深谙庄家意图与市场流动性，鄙视追涨杀跌的右侧韭菜玩法。你是纯粹的**“左侧交易者”（Left-side Trader）**，犹如潜伏的狙击手，专门在市场情绪最狂热时摸顶做空（阻力位），在最绝望时抄底做多（支撑位）。

## 🎯 交易信仰与核心策略
1. **绝对左侧：** 涨到强阻力位/压力位（Supply Zone）果断做空，跌到强支撑位（Demand Zone）果断做多。寻找假突破（2B法则）和流动性猎杀作为入场信号。
2. **超短线波段控制：** 绝不扛单，严格控时。单笔持仓极限时间绝不超过24小时，目标在12小时内随当次波动结束完成平仓。拒绝任何形式的长期格局。
3. **极简图表：** 只看K线（Price Action）、成交量以及关键的支撑/阻力位。只看 4H（大局）、1H（动能）、15M（执行）。

## 🛡️ 风控铁律 (Risk Management - 核心指令)
这是你的生死线，任何交易计划必须**绝对服从**以下量化标准：
1. **硬性止损（最高 1%）：** 入场价与止损价的距离**绝对不允许超过 1%**。既然是极佳的左侧点位，如果做错，意味着支撑/阻力已破，必须在 1% 内无条件认错离场。如果盘面结构需要大于 1% 的止损，则放弃该交易！
2. **硬性止盈（最低 2.4%）：** 入场价与第一止盈目标价的距离**绝对不允许低于 2.4%**。日内波段的利润必须吃到位。
3. **盈亏比底线：** 结合上述两点，每笔交易的潜在盈亏比（RR）必须 **≥ 1:2.4**。

## ⏱️ 多时间框架分析系统 (MTF)
- **【4H 级别】定大局：** 画出最核心的强支撑位和强阻力位。这是你左侧挂单或重点关注的“靶区”。
- **【1H 级别】看动能：** 观察价格靠近靶区时的动能，寻找衰竭、长影线等初步见顶/见底信号。
- **【15M 级别】精确定位：** 在关键位置，一旦出现“插针收回”、“假跌破/假突破”、“吞没形态”，立刻触发进场条件。

## 💬 交互规则与回复框架
当用户询问某个币种或提供盘面信息时，你必须以冷酷、专业的“大师口吻”回复，严格按照以下结构输出，并在计算点位时**严格执行 1% SL / 2.4% TP** 的数学规则：

**【1. 盘面扫描 (4H/1H)】**
- 简述日内宏观结构，明确给出**日内强阻力位**和**日内强支撑位**。

**【2. 狙击计划 (15M入场策略)】**
- **做空预案：** 涨至哪个位置，观察什么15M信号开空。
- **做多预案：** 跌至哪个位置，观察什么15M信号开多。

**【3. 纪律与风控点位 (精确计算)】**
- 针对上述预案，给出精确的量化点位区间：
  - **入场点 (Entry)：** [具体价格或极窄区间]
  - **止损点 (SL)：** [具体价格，计算方式：做多为 Entry * 0.99，做空为 Entry * 1.01，距离严格控制在 1% 以内]
  - **止盈点 (TP)：** [具体价格，计算方式：做多为 Entry * 1.024，做空为 Entry * 0.976，距离至少满足 2.4%]

**【4. 大师箴言】**
- 附带一句关于“纪律、耐心、左侧、极严风控”的简短、冷酷的交易心法（例如：“止损 1% 是保护你留在牌桌的铠甲，不够 2.4% 的利润不配让你承担风险。”）

## ⚠️ 注意：
你的分析必须基于当前最新的市场逻辑，不废话，只给能执行的交易计划。**在输出点位时，请务必在后台进行数学验算，确保止损幅度 ≤ 1%，止盈幅度 ≥ 2.4%**。请确认你已理解你的角色和风控铁律。"""
DEFAULT_AI_PROVIDER = os.getenv("HELPER_AI_PROVIDER", "gemini").lower()
GEMINI_CHAT_MODEL = os.getenv("HELPER_GEMINI_MODEL", os.getenv("HELPER_AI_CHAT_MODEL", os.getenv("DEFAULT_CHAT_MODEL", "gemini-3.5-flash")))
OPENAI_CHAT_MODEL = os.getenv("HELPER_OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1"))
OPENAI_BASE_URL = os.getenv("HELPER_OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL"))
OPENAI_API_KEY_VALUE = os.getenv("HELPER_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
OPENAI_REASONING_EFFORT = os.getenv("HELPER_OPENAI_REASONING_EFFORT", os.getenv("OPENAI_REASONING_EFFORT", "xhigh"))
KLINE_SYMBOLS = {
    "btc": "BTCUSDT",
    "eth": "ETHUSDT",
}
KLINE_DAY_WINDOWS = {
    "4h": "30 days",
    "1h": "20 days",
    "15m": "7 days",
}
KLINE_LIMIT_WINDOWS = {
    "1m": 60,
}
KLINE_INTERVAL_ORDER = tuple(KLINE_DAY_WINDOWS.keys()) + tuple(KLINE_LIMIT_WINDOWS.keys())
AUTO_TRADE_SYSTEM_PROMPT = """
# Role: 顶级加密货币合约交易大师 / 5分钟实时操作指令系统

你现在是一名拥有10年经验、胜率极高且纪律严明的加密货币合约短线交易大师。  
你深谙庄家意图、市场流动性、假突破、插针、诱多诱空与关键支撑阻力结构。

你不是右侧追涨杀跌交易者。  
你是纯粹的左侧交易者，像狙击手一样，只在价格进入关键支撑区或关键阻力区后，等待明确触发信号，再给出操作指令。

你的任务不是写长篇分析，也不是输出完整交易计划。  
你的任务是根据用户提供的实时盘面信息，在 5 分钟实时操作模式下，只给出一个明确指令。

---

## 🎯 交易信仰与核心策略

1. **绝对左侧：**  
   涨到强阻力位 / 压力位（Supply Zone）才考虑做空；  
   跌到强支撑位 / 需求区（Demand Zone）才考虑做多。  
   重点寻找假突破、2B 法则、插针收回、流动性猎杀作为入场信号。

2. **超短线波段控制：**  
   绝不扛单，严格控时。  
   单笔持仓极限时间不超过 24 小时，目标在 12 小时内随当次波动结束完成平仓。  
   拒绝任何形式的长期格局、幻想单、情绪单。

3. **极简图表：**  
   只看 K 线（Price Action）、成交量、关键支撑位、关键阻力位。  
   只参考：
   - 4H：定大局；
   - 1H：看动能；
   - 15M / 5M：看执行触发。

---

## 🛡️ 风控铁律

这是你的生死线，任何开仓指令必须绝对服从以下量化标准：

1. **硬性止损：最高 1%**  
   入场价与止损价的距离绝对不允许超过 1%。  
   如果做多，SL = Entry × 0.99。  
   如果做空，SL = Entry × 1.01。  
   如果盘面结构需要超过 1% 的止损，则禁止开仓，只能输出 WAIT、READY 或 CANCEL。

2. **硬性止盈：最低 2.4%**  
   第一止盈目标距离入场价不得低于 2.4%。  
   如果做多，TP = Entry × 1.024。  
   如果做空，TP = Entry × 0.976。

3. **盈亏比底线：RR ≥ 1:2.4**  
   任何不满足盈亏比的交易，都不能输出 ENTER LONG 或 ENTER SHORT。

4. **无信号不交易：**  
   即使价格到了关键位，只要没有 15M / 5M 触发信号，也不能开仓，只能输出 READY。

---

## ⏱️ 多时间框架分析系统

你必须在内部完成以下判断，但不要把完整分析输出给用户：

### 【4H 级别】定大局
识别当前最核心的强支撑位和强阻力位。  
判断价格是否接近左侧交易靶区。

### 【1H 级别】看动能
观察价格靠近靶区时是否出现：
- 动能衰竭；
- 放量滞涨；
- 放量滞跌；
- 长上影线；
- 长下影线；
- 连续 K 线推进失败。

### 【15M / 5M 级别】精确触发
只有在关键位置出现以下信号，才允许输出开仓指令：
- 插针收回；
- 假突破；
- 假跌破；
- 2B 结构；
- 吞没形态；
- 放量反转；
- 流动性扫损后快速收回。

---

## 🎮 5分钟实时状态指令

你每次只能输出以下 7 个指令之一：

### 1. WAIT / 等待
含义：机会还远，信号不足，不能操作。  
适用情况：
- 价格没有到关键支撑 / 阻力；
- 多空方向不清晰；
- 风险收益比不合格；
- 数据不足；
- 没有触发信号。

输出格式：

指令：WAIT

---

### 2. READY / 准备
含义：价格已经进入狙击区，但还没有触发开仓。  
这是准备，不是开仓。

适用情况：
- 价格接近强支撑或强阻力；
- 左侧交易区域已经出现；
- 但 15M / 5M 还没有确认信号；
- 需要等待下一轮触发。

输出格式：

指令：READY

---

### 3. ENTER LONG / 开多
含义：做多条件已经触发，可以开多。

只有同时满足以下条件，才允许输出：
- 价格位于强支撑 / 需求区；
- 出现假跌破、插针收回、吞没或放量反转；
- SL 不超过 1%；
- TP 不低于 2.4%；
- RR ≥ 1:2.4。

输出格式必须为：

指令：ENTER LONG  
Entry：xxx  
SL：xxx  
TP：xxx  
失效条件：xxx

计算规则：
- SL = Entry × 0.99
- TP = Entry × 1.024

---

### 4. ENTER SHORT / 开空
含义：做空条件已经触发，可以开空。

只有同时满足以下条件，才允许输出：
- 价格位于强阻力 / 供应区；
- 出现假突破、插针收回、吞没或放量反转；
- SL 不超过 1%；
- TP 不低于 2.4%；
- RR ≥ 1:2.4。

输出格式必须为：

指令：ENTER SHORT  
Entry：xxx  
SL：xxx  
TP：xxx  
失效条件：xxx

计算规则：
- SL = Entry × 1.01
- TP = Entry × 0.976

---

### 5. HOLD / 持有
含义：如果用户已经有仓位，当前继续持有，不主动平仓。

适用情况：
- 原交易逻辑仍然有效；
- 未触发 SL；
- 未触发 TP；
- 未出现明确反向信号；
- 没有超过持仓时间纪律。

输出格式：

指令：HOLD

---

### 6. EXIT / 平仓
含义：用户已经有仓位，现在应主动离场，不等原 SL / TP。

适用情况：
- 原交易逻辑失效；
- 出现明显反向信号；
- 持仓时间过长；
- 临近重大风险事件；
- 价格行为不再支持继续持有；
- 需要小盈、保本或减少亏损离场。

输出格式：

指令：EXIT

---

### 7. CANCEL / 取消计划
含义：原本准备中的交易计划作废。

适用情况：
- 用户还没进场；
- 原支撑 / 阻力结构失效；
- 价格远离狙击区；
- 触发条件消失；
- 旧计划不再有效。

输出格式：

指令：CANCEL

---

## 💬 回复规则

1. 每次回复只能给出一个指令。
2. 不输出完整交易计划。
3. 不输出盘面扫描。
4. 不输出大师箴言。
5. 不同时给多空两个方向。
6. 不解释太多理由。
7. 没有明确触发时，禁止输出 ENTER LONG 或 ENTER SHORT。
8. 数据不足时，默认输出 WAIT。
9. 已有仓位时，只能在 HOLD / EXIT 中选择，除非用户明确表示没有持仓。
10. 未进场时，不能输出 HOLD 或 EXIT。

---

## 最终输出格式限制

普通状态只输出一行：

指令：WAIT

或：

指令：READY

或：

指令：HOLD

或：

指令：EXIT

或：

指令：CANCEL

只有开仓状态可以输出四行：

指令：ENTER LONG  
Entry：xxx  
SL：xxx  
TP：xxx  
失效条件：xxx

或：

指令：ENTER SHORT  
Entry：xxx  
SL：xxx  
TP：xxx  
失效条件：xxx
"""
AUTO_TRADE_NOTIFY_ACTIONS = ("READY", "EXIT", "CANCEL", "ENTER LONG", "ENTER SHORT")
BEIJING_TZ = datetime.timezone(datetime.timedelta(hours=8), name="Asia/Shanghai")
PG_TIMEZONE = "Asia/Shanghai"

# 存储用户的生图会话状态
# { user_id: { 'ai': HelperAI, 'last_image': bytes | None } }
user_sessions = {}

# 存储用户的文字 AI 会话状态
# { user_id: HelperTextAI }
ai_sessions = {}

# 存储定时拉取的 K 线缓存
# { symbol: { rows_by_interval, seen_ids_by_interval, updated_at, last_new_counts, last_fetched_counts } }
kline_memory_cache = {}

# 存储 5m 自动交易 AI 会话状态，保留定时请求的上下文
auto_trade_ai_session = None

# 存储自动交易已通知用户的最近指令，避免重复推送同一状态
auto_trade_last_notified_actions = {}


def debug_log(message, exc=None):
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"[tg_helper_bot][{timestamp}] {message}", flush=True)
    logger.error(message)
    if exc is not None:
        traceback.print_exception(type(exc), exc, exc.__traceback__)


async def reply_text_chunks(update: Update, text: str):
    """Telegram 单条消息有长度限制，长回复分段发送。"""
    if not text:
        return
    chunk_size = 3900
    for i in range(0, len(text), chunk_size):
        await update.message.reply_text(text[i:i + chunk_size])


def markdown_to_telegram_html(text):
    placeholders = []

    def stash_code_block(match):
        code = html.escape(match.group(1).strip())
        placeholders.append(f"<pre>{code}</pre>")
        return f"@@CODEBLOCK{len(placeholders) - 1}@@"

    def stash_inline_code(match):
        code = html.escape(match.group(1))
        placeholders.append(f"<code>{code}</code>")
        return f"@@CODEBLOCK{len(placeholders) - 1}@@"

    converted = re.sub(r"```(?:\w+)?\n([\s\S]*?)```", stash_code_block, text)
    converted = re.sub(r"`([^`\n]+)`", stash_inline_code, converted)
    converted = html.escape(converted)

    converted = re.sub(r"^#{1,6}\s*(.+)$", r"<b>\1</b>", converted, flags=re.MULTILINE)
    converted = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", converted)
    converted = re.sub(r"__(.+?)__", r"<b>\1</b>", converted)
    converted = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", converted)
    converted = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<i>\1</i>", converted)
    converted = re.sub(r"~~(.+?)~~", r"<s>\1</s>", converted)

    for i, block in enumerate(placeholders):
        converted = converted.replace(f"@@CODEBLOCK{i}@@", block)

    return converted


async def reply_markdown_chunks(update: Update, text: str):
    """把常见 Markdown 转成 Telegram HTML；失败时降级纯文本。"""
    if not text:
        return
    chunk_size = 3500
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        try:
            await update.message.reply_text(
                markdown_to_telegram_html(chunk),
                parse_mode='HTML',
                disable_web_page_preview=True,
            )
        except Exception as e:
            debug_log(f"Markdown/HTML 回复失败，降级纯文本: error={e}", e)
            await update.message.reply_text(chunk)


def get_two_database_url():
    database_url = os.getenv("TWO_DATABASE_URL")
    if not database_url:
        debug_log("TWO_DATABASE_URL 未配置")
        raise ValueError("请设置 TWO_DATABASE_URL 环境变量")
    debug_log("TWO_DATABASE_URL 已读取")
    return database_url


def fetch_kline_rows(symbol, interval_name, days=None, limit=None):
    debug_log(f"开始查询K线: symbol={symbol}, interval={interval_name}, days={days}, limit={limit}, timezone={PG_TIMEZONE}")
    database_url = get_two_database_url()
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SET TIME ZONE %s", (PG_TIMEZONE,))
            if days:
                cur.execute(
                    """
                    SELECT id, open_time, close_time, open_price, high_price, low_price,
                           close_price, volume
                    FROM kline_data
                    WHERE symbol = %s
                      AND "interval" = %s
                      AND open_time >= now() - (%s)::interval
                    ORDER BY open_time ASC
                    """,
                    (symbol, interval_name, days),
                )
                rows = cur.fetchall()
                debug_log(f"K线查询完成: symbol={symbol}, interval={interval_name}, rows={len(rows)}")
                return rows

            cur.execute(
                """
                SELECT id, open_time, close_time, open_price, high_price, low_price,
                       close_price, volume
                FROM (
                    SELECT id, open_time, close_time, open_price, high_price, low_price,
                           close_price, volume
                    FROM kline_data
                    WHERE symbol = %s
                      AND "interval" = %s
                    ORDER BY open_time DESC
                    LIMIT %s
                ) recent
                ORDER BY open_time ASC
                """,
                (symbol, interval_name, limit),
            )
            rows = cur.fetchall()
            debug_log(f"K线查询完成: symbol={symbol}, interval={interval_name}, rows={len(rows)}")
            return rows


def format_decimal(value):
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_kline_time(value):
    if value.tzinfo is None:
        value = value.replace(tzinfo=BEIJING_TZ)
    else:
        value = value.astimezone(BEIJING_TZ)
    return value.isoformat(timespec="minutes")


def fetch_kline_rows_by_interval(symbol):
    rows_by_interval = {}
    fetched_counts = {}

    for interval_name, days in KLINE_DAY_WINDOWS.items():
        rows = fetch_kline_rows(symbol, interval_name, days=days)
        rows_by_interval[interval_name] = rows
        fetched_counts[interval_name] = len(rows)

    for interval_name, limit in KLINE_LIMIT_WINDOWS.items():
        rows = fetch_kline_rows(symbol, interval_name, limit=limit)
        rows_by_interval[interval_name] = rows
        fetched_counts[interval_name] = len(rows)

    return rows_by_interval, fetched_counts


def get_kline_memory_cache(symbol):
    return kline_memory_cache.setdefault(symbol, {
        "rows_by_interval": {},
        "seen_ids_by_interval": {},
        "updated_at": None,
        "last_new_counts": {},
        "last_fetched_counts": {},
    })


def get_cached_kline_rows_by_interval(symbol):
    cache = kline_memory_cache.get(symbol)
    if not cache or not cache.get("rows_by_interval"):
        return None, None

    rows_by_interval = {}
    fetched_counts = {}
    for interval_name in KLINE_INTERVAL_ORDER:
        rows = cache["rows_by_interval"].get(interval_name, [])
        rows_by_interval[interval_name] = list(rows)
        fetched_counts[interval_name] = len(rows)

    debug_log(
        f"使用内存K线缓存: symbol={symbol}, "
        f"updated_at={cache.get('updated_at')}, counts={fetched_counts}"
    )
    return rows_by_interval, fetched_counts


def refresh_kline_memory_cache(symbol):
    debug_log(f"开始刷新内存K线缓存: symbol={symbol}")
    rows_by_interval, fetched_counts = fetch_kline_rows_by_interval(symbol)
    cache = get_kline_memory_cache(symbol)
    new_counts = {}

    for interval_name in KLINE_INTERVAL_ORDER:
        rows = [dict(row) for row in rows_by_interval.get(interval_name, [])]
        fetched_ids = {row["id"] for row in rows}
        seen_ids = cache["seen_ids_by_interval"].setdefault(interval_name, set())
        new_rows = [row for row in rows if row["id"] not in seen_ids]

        rows_by_id = {
            row["id"]: row
            for row in cache["rows_by_interval"].get(interval_name, [])
            if row["id"] in fetched_ids
        }
        for row in rows:
            rows_by_id[row["id"]] = row

        cache["rows_by_interval"][interval_name] = [
            rows_by_id[row["id"]]
            for row in rows
            if row["id"] in rows_by_id
        ]
        seen_ids.intersection_update(fetched_ids)
        seen_ids.update(fetched_ids)
        new_counts[interval_name] = len(new_rows)

    cache["updated_at"] = datetime.datetime.now(BEIJING_TZ).isoformat(timespec="seconds")
    cache["last_new_counts"] = new_counts
    cache["last_fetched_counts"] = fetched_counts
    debug_log(
        f"内存K线缓存刷新完成: symbol={symbol}, "
        f"fetched_counts={fetched_counts}, new_counts={new_counts}"
    )
    return cache, new_counts, fetched_counts


def build_kline_prompt(symbol, seen_ids_by_interval=None):
    debug_log(f"开始组装K线prompt: symbol={symbol}")
    if seen_ids_by_interval is None:
        seen_ids_by_interval = {}
    sections = []
    counts = {}
    fetched_counts = {}
    new_ids_by_interval = {}
    rows_by_interval, cached_fetched_counts = get_cached_kline_rows_by_interval(symbol)

    if rows_by_interval is None:
        rows_by_interval, fetched_counts = fetch_kline_rows_by_interval(symbol)
    else:
        fetched_counts = cached_fetched_counts

    for interval_name in KLINE_INTERVAL_ORDER:
        rows = rows_by_interval.get(interval_name, [])
        seen_ids = seen_ids_by_interval.get(interval_name, set())
        new_rows = [row for row in rows if row["id"] not in seen_ids]
        counts[interval_name] = len(new_rows)
        new_ids_by_interval[interval_name] = {row["id"] for row in new_rows}
        debug_log(f"K线增量过滤: symbol={symbol}, interval={interval_name}, fetched={len(rows)}, seen={len(seen_ids)}, new={len(new_rows)}")
        sections.append(format_kline_section(interval_name, new_rows))

    prompt = (
        f"以下是 {symbol} 最新多周期 K 线数据。请基于这些数据回答用户接下来的交易问题。"
        "分析时严格按你的回复框架输出，给出 4H/1H 强支撑和强阻力、15M 入场触发条件，并精确计算 Entry / SL / TP。"
        "若数据不足以支持交易，明确说放弃。\n\n"
        "字段: open_time(Asia/Shanghai),open,high,low,close,volume\n\n"
        + "\n\n".join(sections)
    )
    debug_log(f"K线prompt组装完成: symbol={symbol}, fetched_counts={fetched_counts}, new_counts={counts}, chars={len(prompt)}")
    return prompt, counts, fetched_counts, new_ids_by_interval


def build_auto_trade_kline_prompt(symbol, seen_ids_by_interval=None):
    debug_log(f"开始组装自动交易K线prompt: symbol={symbol}")
    if seen_ids_by_interval is None:
        seen_ids_by_interval = {}
    sections = []
    counts = {}
    new_ids_by_interval = {}
    rows_by_interval, fetched_counts = get_cached_kline_rows_by_interval(symbol)

    if rows_by_interval is None:
        rows_by_interval, fetched_counts = fetch_kline_rows_by_interval(symbol)

    for interval_name in KLINE_INTERVAL_ORDER:
        rows = rows_by_interval.get(interval_name, [])
        seen_ids = seen_ids_by_interval.get(interval_name, set())
        new_rows = [row for row in rows if row["id"] not in seen_ids]
        counts[interval_name] = len(new_rows)
        new_ids_by_interval[interval_name] = {row["id"] for row in new_rows}
        debug_log(f"自动交易K线增量过滤: symbol={symbol}, interval={interval_name}, fetched={len(rows)}, seen={len(seen_ids)}, new={len(new_rows)}")
        sections.append(format_kline_section(interval_name, new_rows))

    prompt = (
        f"以下是 {symbol} 5m 实时自动交易轮询新增多周期 K 线数据。\n"
        f"本轮时间: {datetime.datetime.now(BEIJING_TZ).isoformat(timespec='seconds')}\n"
        f"字段: open_time(Asia/Shanghai),open,high,low,close,volume\n\n"
        + "\n\n".join(sections)
    )
    debug_log(f"自动交易K线prompt组装完成: symbol={symbol}, fetched_counts={fetched_counts}, new_counts={counts}, chars={len(prompt)}")
    return prompt, counts, fetched_counts, new_ids_by_interval


def build_kline_export_text(symbol):
    prompt, counts, fetched_counts, _ = build_kline_prompt(symbol)
    return prompt, counts, fetched_counts


def write_kline_export(symbol):
    export_text, counts, fetched_counts = build_kline_export_text(symbol)
    timestamp = datetime.datetime.now(BEIJING_TZ).strftime("%Y%m%d_%H%M%S")
    filename = f"kline_{symbol}_{timestamp}_asia_shanghai.txt"
    path = os.path.abspath(os.path.join(os.getcwd(), filename))
    with open(path, "w", encoding="utf-8") as f:
        f.write(export_text)
        f.write("\n")
    debug_log(f"K线数据已导出: symbol={symbol}, path={path}, counts={counts}, fetched_counts={fetched_counts}, chars={len(export_text)}")
    return path, counts, fetched_counts, len(export_text)


def format_kline_section(interval_name, rows):
    lines = [f"[{interval_name}]"]
    for row in rows:
        lines.append(
            ",".join(
                [
                    format_kline_time(row["open_time"]),
                    format_decimal(row["open_price"]),
                    format_decimal(row["high_price"]),
                    format_decimal(row["low_price"]),
                    format_decimal(row["close_price"]),
                    format_decimal(row["volume"]),
                ]
            )
        )
    return "\n".join(lines)

class HelperAI:
    def __init__(self, mode, system_instruction):
        self.client = image_router.get_client()
        self.mode = mode
        self.system_instruction = system_instruction
        
        history = []
        # 根据模式加载视觉参考
        if mode in ['gen_me', 'gen_both'] and os.path.exists(CHARACTER_PHOTO_PATH):
            with open(CHARACTER_PHOTO_PATH, 'rb') as f:
                char_data = f.read()
            history.append({
                'role': 'user',
                'parts': [
                    types.Part.from_text(text="This is the character (Nanase) reference."),
                    types.Part.from_bytes(data=char_data, mime_type="image/jpeg")
                ]
            })
            history.append({
                'role': 'model',
                'parts': [types.Part.from_text(text="I have received Nanase's reference. I will maintain consistency.")]
            })
            
        if mode in ['gen_user', 'gen_both'] and os.path.exists(USER_PHOTO_PATH):
            with open(USER_PHOTO_PATH, 'rb') as f:
                u_data = f.read()
            history.append({
                'role': 'user',
                'parts': [
                    types.Part.from_text(text="This is the user (Siyuan) reference."),
                    types.Part.from_bytes(data=u_data, mime_type="image/jpeg")
                ]
            })
            history.append({
                'role': 'model',
                'parts': [types.Part.from_text(text="I have received Siyuan's reference. I will maintain consistency.")]
            })

        # 按照官网最新文档：设置 response_modalities 为 ['TEXT', 'IMAGE']
        self.chat = self.client.chats.create(
            model='gemini-3.1-flash-image',
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_modalities=['TEXT', 'IMAGE'],
                image_config=types.ImageConfig(aspect_ratio="3:4")
            ),
            history=history
        )

    async def generate(self, prompt, image_bytes=None):
        """调用 Gemini 生成图片并处理"""
        # 强制要求日系卡通风格
        full_prompt = f"{prompt}. STYLE REQUIREMENT: Strictly follow Japanese anime / cartoon style."

        # 按照 SDK 要求，发送 Part 列表（可选图片 + 文本）
        parts = []
        if image_bytes:
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
        parts.append(types.Part.from_text(text=full_prompt))

        def _do_img_chat_send(cli):
            self.client = cli
            history = getattr(self.chat, "_curated_history", []) if hasattr(self.chat, "_curated_history") else getattr(self.chat, "history", [])
            self.chat = self.client.chats.create(
                model='gemini-3.1-flash-image',
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    response_modalities=['TEXT', 'IMAGE'],
                    image_config=types.ImageConfig(aspect_ratio="3:4")
                ),
                history=history
            )
            return self.chat.send_message(parts)

        response = image_router.execute_with_retry(_do_img_chat_send)
        
        for part in response.parts:
            if part.inline_data is not None:
                # 处理图片
                img = Image.open(io.BytesIO(part.inline_data.data))
                
                # 压缩优化
                max_size = 1280
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.LANCZOS)
                
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # 输出到内存流
                bio = io.BytesIO()
                bio.name = 'generated.jpg'
                img.save(bio, 'JPEG', quality=85, optimize=True)
                bio.seek(0)
                return bio, response.text
        
        return None, response.text


class HelperTextAI:
    def __init__(self, system_instruction=AI_SYSTEM_PROMPT):
        self.provider = DEFAULT_AI_PROVIDER if DEFAULT_AI_PROVIDER in ("gemini", "openai") else "gemini"
        self.model = GEMINI_CHAT_MODEL if self.provider == "gemini" else OPENAI_CHAT_MODEL
        debug_log(f"初始化文字AI本地会话: provider={self.provider}, model={self.model}")
        self.gemini_client = None
        self.openai_client = None
        self.system_instruction = system_instruction
        self.messages = []
        self.pending_contexts = []
        self.seen_kline_ids = {}
        debug_log("文字AI本地会话初始化完成，尚未请求AI")

    def set_provider(self, provider, model=None):
        provider = (provider or "").lower()
        if provider not in ("gemini", "openai"):
            raise ValueError("provider 只支持 gemini 或 openai")
        self.provider = provider
        self.model = model or (GEMINI_CHAT_MODEL if provider == "gemini" else OPENAI_CHAT_MODEL)
        debug_log(f"文字AI已切换模型: provider={self.provider}, model={self.model}, history_messages={len(self.messages)}")

    def model_status(self):
        return f"{self.provider}:{self.model}"

    def add_context(self, label, text):
        self.pending_contexts.append({
            "label": label,
            "text": text,
        })
        debug_log(f"已缓存AI上下文: label={label}, chars={len(text)}, pending_contexts={len(self.pending_contexts)}")

    def get_seen_kline_ids(self, symbol):
        return self.seen_kline_ids.setdefault(symbol, {})

    def remember_kline_ids(self, symbol, new_ids_by_interval):
        seen_by_interval = self.get_seen_kline_ids(symbol)
        for interval_name, ids in new_ids_by_interval.items():
            seen_by_interval.setdefault(interval_name, set()).update(ids)
        cached_counts = {k: len(v) for k, v in seen_by_interval.items()}
        debug_log(f"已追加K线ID缓存: symbol={symbol}, cached_counts={cached_counts}")

    def build_prompt(self, user_prompt):
        if not self.pending_contexts:
            return user_prompt

        context_blocks = []
        for item in self.pending_contexts:
            context_blocks.append(f"## {item['label']}\n{item['text']}")

        return (
            "以下是本次对话前缓存的上下文数据，请在回答用户问题时使用：\n\n"
            + "\n\n".join(context_blocks)
            + "\n\n---\n"
            + f"用户问题：{user_prompt}"
        )

    def get_openai_client(self):
        if self.openai_client is None:
            kwargs = {}
            if OPENAI_BASE_URL:
                kwargs["base_url"] = OPENAI_BASE_URL
            if OPENAI_API_KEY_VALUE:
                kwargs["api_key"] = OPENAI_API_KEY_VALUE
            elif OPENAI_BASE_URL:
                kwargs["api_key"] = "not-needed"
            debug_log(f"开始创建OpenAI客户端: base_url_configured={bool(OPENAI_BASE_URL)}, api_key_configured={bool(OPENAI_API_KEY_VALUE)}")
            self.openai_client = OpenAI(**kwargs)
        return self.openai_client

    def gemini_history(self):
        history = []
        for message in self.messages:
            role = "model" if message["role"] == "assistant" else "user"
            history.append({
                "role": role,
                "parts": [types.Part.from_text(text=message["content"])]
            })
        return history

    def send_gemini(self, prompt):
        debug_log(f"准备调用Gemini文字模型: model={self.model}, history_messages={len(self.messages)}")

        def _do_chat_send(cli):
            self.gemini_client = cli
            chat = self.gemini_client.chats.create(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction
                ),
                history=self.gemini_history()
            )
            return chat.send_message(prompt)

        response = chat_router.execute_with_retry(_do_chat_send)
        return response.text or ""

    def send_openai(self, prompt):
        debug_log(f"准备调用OpenAI文字模型: model={self.model}, reasoning_effort={OPENAI_REASONING_EFFORT}, history_messages={len(self.messages)}")
        client = self.get_openai_client()
        messages = []
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})
        messages.extend(self.messages)
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            reasoning_effort=OPENAI_REASONING_EFFORT,
            stream=False,
        )
        return response.choices[0].message.content or ""

    async def chat_text(self, prompt):
        """调用当前文字模型并保留本次 /ai 会话上下文。"""
        prompt_to_send = self.build_prompt(prompt)
        debug_log(f"准备发送文字AI消息: provider={self.provider}, model={self.model}, user_chars={len(prompt)}, total_chars={len(prompt_to_send)}, pending_contexts={len(self.pending_contexts)}")

        if self.provider == "openai":
            reply = self.send_openai(prompt_to_send)
        else:
            reply = self.send_gemini(prompt_to_send)

        debug_log(f"文字AI返回完成: provider={self.provider}, model={self.model}, text_chars={len(reply or '')}")
        self.messages.append({"role": "user", "content": prompt_to_send})
        self.messages.append({"role": "assistant", "content": reply})
        if self.pending_contexts:
            debug_log(f"清空已发送缓存上下文: count={len(self.pending_contexts)}")
            self.pending_contexts.clear()
        return reply


def get_auto_trade_ai_session():
    global auto_trade_ai_session
    if auto_trade_ai_session is None:
        auto_trade_ai_session = HelperTextAI(system_instruction=AUTO_TRADE_SYSTEM_PROMPT)
        debug_log(
            f"初始化5m自动交易AI会话: model={auto_trade_ai_session.model_status()}, "
            f"system_prompt_configured={bool(AUTO_TRADE_SYSTEM_PROMPT.strip())}"
        )
    return auto_trade_ai_session


async def send_auto_trade_kline_to_ai(symbol):
    ai_session = get_auto_trade_ai_session()
    seen_ids_by_interval = ai_session.get_seen_kline_ids(symbol)
    prompt, counts, fetched_counts, new_ids_by_interval = build_auto_trade_kline_prompt(symbol, seen_ids_by_interval)
    total_new = sum(counts.values())
    if total_new == 0:
        debug_log(f"自动交易AI跳过: symbol={symbol}, reason=没有新增K线数据, fetched_counts={fetched_counts}")
        return None

    debug_log(
        f"自动交易AI开始请求: symbol={symbol}, new_counts={counts}, "
        f"fetched_counts={fetched_counts}, history_messages={len(ai_session.messages)}"
    )
    reply = await ai_session.chat_text(prompt)
    ai_session.remember_kline_ids(symbol, new_ids_by_interval)
    debug_log(
        f"自动交易AI请求完成: symbol={symbol}, reply_chars={len(reply or '')}, "
        f"history_messages={len(ai_session.messages)}"
    )
    return reply


def extract_auto_trade_action(reply):
    if not reply:
        return None

    action_pattern = "|".join(action.replace(" ", r"\s+") for action in AUTO_TRADE_NOTIFY_ACTIONS)
    for raw_line in reply.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[\s>*#`_\-•]+", "", line)
        line = re.sub(r"[*_`]+", "", line).strip()
        line = re.sub(r"\s+", " ", line.upper()).strip()
        match = re.match(rf"^(?:(?:指令|ACTION|COMMAND|SIGNAL)\s*[:：]\s*)?({action_pattern})\b", line)
        if match:
            return re.sub(r"\s+", " ", match.group(1))
    return None


async def send_bot_text_chunks(bot, chat_id, text):
    if not text:
        return
    chunk_size = 3900
    for i in range(0, len(text), chunk_size):
        await bot.send_message(chat_id=chat_id, text=text[i:i + chunk_size])


async def notify_auto_trade_action(context: ContextTypes.DEFAULT_TYPE, symbol, reply):
    debug_log(f"自动交易AI回复内容(通知判断前):\n{reply or ''}")
    action = extract_auto_trade_action(reply)
    if not action:
        debug_log("自动交易AI未命中可通知指令，不更新通知去重缓存")
        return False

    last_action = auto_trade_last_notified_actions.get(symbol)
    if action == last_action:
        debug_log(f"自动交易AI跳过重复通知: symbol={symbol}, action={action}")
        return False

    await send_bot_text_chunks(context.bot, ALLOWED_USER_ID, reply)
    auto_trade_last_notified_actions[symbol] = action
    debug_log(f"自动交易AI已通知用户: user_id={ALLOWED_USER_ID}, action={action}, chars={len(reply or '')}")
    return True


def clear_auto_trade_state():
    global auto_trade_ai_session
    had_ai_session = auto_trade_ai_session is not None
    kline_cache_symbols = list(kline_memory_cache.keys())
    notified_symbols = list(auto_trade_last_notified_actions.keys())

    auto_trade_ai_session = None
    kline_memory_cache.clear()
    auto_trade_last_notified_actions.clear()

    debug_log(
        "自动交易状态已清空: "
        f"had_ai_session={had_ai_session}, "
        f"kline_cache_symbols={kline_cache_symbols}, "
        f"notified_symbols={notified_symbols}"
    )
    return had_ai_session, kline_cache_symbols, notified_symbols


async def check_auth(update: Update):
    user_id = update.effective_user.id if update.effective_user else None
    debug_log(f"鉴权检查: user_id={user_id}")
    if update.effective_user.id != ALLOWED_USER_ID:
        await update.message.reply_text("Unauthorized.")
        debug_log(f"鉴权失败: user_id={user_id}")
        return False
    return True

async def start_gen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update): return
    
    cmd = update.message.text.split()[0][1:] # gen, gen_me, etc.
    user_id = update.effective_user.id
    
    instruction = "You are a specialized image generation assistant. Your ONLY goal is to generate high-quality Japanese anime style images based on user prompts. "
    if cmd == 'gen_me':
        instruction += "Always use the provided Nanase reference for the character."
    elif cmd == 'gen_user':
        instruction += "Always use the provided Siyuan reference for the user."
    elif cmd == 'gen_both':
        instruction += "Maintain consistency for both Nanase and Siyuan based on references."

    try:
        ai_sessions.pop(user_id, None)
        user_sessions[user_id] = {
            'ai': HelperAI(cmd, instruction),
            'last_image': None
        }
        await update.message.reply_text(f"模式已启动: {cmd}\n请发送你的图片描述，你可以不断反馈修改。输入 /end 结束。")
    except Exception as e:
        await update.message.reply_text(f"启动失败: {str(e)}")

async def end_gen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in user_sessions:
        del user_sessions[user_id]
        await update.message.reply_text("会话已结束。")
    else:
        await update.message.reply_text("没有正在运行的生图会话。")

async def start_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    debug_log(f"收到 /ai: chat_id={update.effective_chat.id if update.effective_chat else None}")
    if not await check_auth(update): return

    user_id = update.effective_user.id
    try:
        await update.message.reply_text("正在启动 AI 会话...")
        debug_log(f"开始启动AI会话: user_id={user_id}")
        user_sessions.pop(user_id, None)
        ai_sessions[user_id] = HelperTextAI()
        debug_log(f"AI会话已写入内存: user_id={user_id}, active_ai_sessions={len(ai_sessions)}")
        await update.message.reply_text(f"AI 会话已启动，当前模型 {ai_sessions[user_id].model_status()}。现在可以直接聊天，输入 /aiend 结束。")
    except Exception as e:
        debug_log(f"AI 会话启动失败: user_id={user_id}, error={e}", e)
        await update.message.reply_text(f"AI 会话启动失败: {str(e)}")

async def end_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    debug_log(f"收到 /aiend: chat_id={update.effective_chat.id if update.effective_chat else None}")
    if not await check_auth(update): return

    user_id = update.effective_user.id
    if user_id in ai_sessions:
        del ai_sessions[user_id]
        await update.message.reply_text("AI 会话已结束。")
    else:
        await update.message.reply_text("没有正在运行的 AI 会话。")

async def ai_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    debug_log(f"收到 /aimodel: args={context.args}")
    if not await check_auth(update): return

    user_id = update.effective_user.id
    if user_id not in ai_sessions:
        await update.message.reply_text("请先使用 /ai 开始 AI 会话。")
        return

    session = ai_sessions[user_id]
    await update.message.reply_text(
        "当前 AI 模型：\n"
        f"{session.model_status()}\n\n"
        "切换命令：\n"
        "/use gemini [model]\n"
        "/use openai [model]"
    )

async def use_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    debug_log(f"收到 /use: args={context.args}")
    if not await check_auth(update): return

    user_id = update.effective_user.id
    if user_id not in ai_sessions:
        await update.message.reply_text("请先使用 /ai 开始 AI 会话。")
        return

    if not context.args:
        await update.message.reply_text("用法：/use gemini [model] 或 /use openai [model]")
        return

    provider = context.args[0].lower()
    model = context.args[1] if len(context.args) > 1 else None
    try:
        ai_sessions[user_id].set_provider(provider, model=model)
        await update.message.reply_text(f"已切换到 {ai_sessions[user_id].model_status()}")
    except Exception as e:
        debug_log(f"切换AI模型失败: user_id={user_id}, error={e}", e)
        await update.message.reply_text(f"切换失败: {str(e)}")

def parse_kline_args_from_text(text):
    if not text:
        return []
    parts = text.strip().split()
    if not parts:
        return []

    command = parts[0].lstrip("/／").split("@", 1)[0].lower()
    if command != "kline":
        return []
    return parts[1:]


def parse_symbol_arg(args):
    if not args:
        return None, None
    symbol_key = args[0].lower()
    return symbol_key, KLINE_SYMBOLS.get(symbol_key)


async def run_kline(update: Update, context: ContextTypes.DEFAULT_TYPE, symbol_key):
    user_id = update.effective_user.id
    if user_id not in ai_sessions:
        debug_log(f"/kline 被拒绝，AI会话不存在: user_id={user_id}")
        await update.message.reply_text("请先使用 /ai 开始 AI 会话，再使用 /kline btc 或 /kline eth。")
        return

    if not symbol_key:
        debug_log("/kline 缺少参数")
        await update.message.reply_text("用法：/kline btc 或 /kline eth")
        return

    symbol_key, symbol = parse_symbol_arg([symbol_key])
    if not symbol:
        debug_log(f"/kline 参数不支持: arg={symbol_key}")
        await update.message.reply_text("只支持 /kline btc 或 /kline eth。")
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    await update.message.reply_text(f"开始读取并发送数据 {symbol}...")

    try:
        debug_log(f"/kline 开始处理: user_id={user_id}, symbol={symbol}")
        ai_session = ai_sessions[user_id]
        seen_ids_by_interval = ai_session.get_seen_kline_ids(symbol)
        prompt, counts, fetched_counts, new_ids_by_interval = build_kline_prompt(symbol, seen_ids_by_interval)
        total_new = sum(counts.values())
        debug_log(f"K线数据已组装: {symbol}, fetched={fetched_counts}, new={counts}, total_new={total_new}")
        if total_new == 0:
            await update.message.reply_text(f"{symbol} 没有新增 K 线数据。")
            return

        ai_session.add_context(f"{symbol} 新增多周期K线数据", prompt)
        ai_session.remember_kline_ids(symbol, new_ids_by_interval)
        debug_log(f"/kline 增量已缓存，等待下一条普通消息再发送AI: user_id={user_id}, symbol={symbol}")
        await update.message.reply_text(f"已缓存新增数据 {symbol}: 4h={counts.get('4h', 0)}, 1h={counts.get('1h', 0)}, 15m={counts.get('15m', 0)}, 1m={counts.get('1m', 0)}。发送普通文本后再请求 AI。")
    except Exception as e:
        debug_log(f"K线数据发送出错: user_id={user_id}, symbol={symbol}, error={e}", e)
        await update.message.reply_text(f"K线数据发送失败: {str(e)}")

async def kline_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    debug_log(f"收到 /kline CommandHandler: text={update.message.text if update.message else None}, args={context.args}")
    if not await check_auth(update): return

    symbol_key = context.args[0] if context.args else None
    await run_kline(update, context, symbol_key)

async def kline_message_fallback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text if update.message else ""
    args = parse_kline_args_from_text(text)
    debug_log(f"收到 /kline fallback: text={text}, args={args}")
    if not await check_auth(update): return

    symbol_key = args[0] if args else None
    await run_kline(update, context, symbol_key)

async def outk_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    debug_log(f"收到 /outk: text={update.message.text if update.message else None}, args={context.args}")
    if not await check_auth(update): return

    symbol_key, symbol = parse_symbol_arg(context.args)
    if not symbol:
        await update.message.reply_text("用法：/outk btc 或 /outk eth")
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    await update.message.reply_text(f"开始导出 K 线数据 {symbol}...")

    try:
        path, counts, fetched_counts, char_count = write_kline_export(symbol)
        await update.message.reply_text(
            f"已导出 {symbol}\n"
            f"文件: {path}\n"
            f"4h={counts.get('4h', 0)}, 1h={counts.get('1h', 0)}, "
            f"15m={counts.get('15m', 0)}, 1m={counts.get('1m', 0)}\n"
            f"字符数: {char_count}"
        )
    except Exception as e:
        debug_log(f"K线导出失败: symbol={symbol}, error={e}", e)
        await update.message.reply_text(f"K线导出失败: {str(e)}")


async def clear_auto_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    debug_log(f"收到 /clearAuto: text={update.message.text if update.message else None}")
    if not await check_auth(update): return

    had_ai_session, kline_cache_symbols, notified_symbols = clear_auto_trade_state()
    await update.message.reply_text(
        "已清空自动交易 AI 上下文和内存 K 线，下一轮将重新开始。\n"
        f"AI会话: {'已重置' if had_ai_session else '原本为空'}\n"
        f"K线缓存: {len(kline_cache_symbols)} 个 symbol\n"
        f"通知缓存: {len(notified_symbols)} 个 symbol"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    debug_log("收到 /help")
    if not await check_auth(update): return
    help_text = (
        "🤖 <b>Helper Bot 命令列表</b>\n\n"
        "/gen — 启动自由生图模式（无参考角色）\n"
        "/gen_me — 启动生图模式（参考 Nanase）\n"
        "/gen_user — 启动生图模式（参考 Siyuan）\n"
        "/gen_both — 启动生图模式（同时参考 Nanase 和 Siyuan）\n"
        "/ai — 启动加密货币合约交易 AI 聊天模式\n"
        "/aimodel — 查看当前 AI 模型\n"
        "/use gemini|openai [model] — 切换 AI 模型后端\n"
        "/kline btc|eth — 缓存多周期 K 线数据，下一条普通消息一起发送给 AI\n"
        "/outk btc|eth — 将多周期 K 线数据导出到当前目录\n"
        "/clearAuto — 清空自动交易 AI 上下文和内存 K 线\n"
        "/aiend — 结束当前 AI 聊天会话\n"
        "/end — 结束当前生图会话\n"
        "/help — 显示此帮助信息\n\n"
        "💡 启动任意生图模式后，直接发送文字描述即可生成图片，"
        "也可先发送一张图片作为参考，再发送描述。可持续对话修改，完成后使用 /end 结束。\n"
        "💬 启动 /ai 后可直接发送文字和 AI 聊天，完成后使用 /aiend 结束。"
    )
    await update.message.reply_text(help_text, parse_mode='HTML')

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    debug_log("收到图片消息")
    if not await check_auth(update): return
    user_id = update.effective_user.id

    if user_id not in user_sessions:
        if user_id in ai_sessions:
            await update.message.reply_text("当前是 /ai 文字聊天会话。请发送文字聊天，或输入 /aiend 结束后再开始生图。")
        else:
            await update.message.reply_text("请先使用 /gen, /gen_me, /gen_user 或 /gen_both 开始生图。")
        return

    # 取最大尺寸的照片
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    photo_bytes = await file.download_as_bytearray()

    user_sessions[user_id]['last_image'] = bytes(photo_bytes)

    caption = update.message.caption
    if caption:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")
        try:
            ai_session = user_sessions[user_id]['ai']
            photo_bio, text = await ai_session.generate(caption, image_bytes=bytes(photo_bytes))
            if photo_bio:
                await update.message.reply_photo(photo=photo_bio)
            if text:
                await update.message.reply_text(text)
            elif not photo_bio:
                await update.message.reply_text("未能生成图片，请换个描述试试。")
        except Exception as e:
            logger.error(f"生图出错: {e}")
            await update.message.reply_text(f"出错啦: {str(e)}")
    else:
        await update.message.reply_text("已收到图片。请发送文字描述以生成新图。")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    debug_log(f"收到普通消息: text={update.message.text if update.message else None}")
    if not await check_auth(update): return
    user_id = update.effective_user.id

    if user_id in ai_sessions:
        prompt = update.message.text
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        try:
            reply = await ai_sessions[user_id].chat_text(prompt)
            if reply:
                await reply_markdown_chunks(update, reply)
            else:
                await update.message.reply_text("AI 没有返回内容，请再试一次。")
        except Exception as e:
            debug_log(f"AI 聊天出错: user_id={user_id}, error={e}", e)
            await update.message.reply_text(f"出错啦: {str(e)}")
        return
    
    if user_id not in user_sessions:
        await update.message.reply_text("请先使用 /gen, /gen_me, /gen_user, /gen_both 开始生图，或使用 /ai 开始文字聊天。")
        return

    prompt = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")
    
    try:
        ai_session = user_sessions[user_id]['ai']
        image_bytes = user_sessions[user_id].get('last_image')
        photo_bio, text = await ai_session.generate(prompt, image_bytes=image_bytes)
        
        if photo_bio:
            await update.message.reply_photo(photo=photo_bio)
        
        if text:
            await update.message.reply_text(text)
        elif not photo_bio:
            await update.message.reply_text("未能生成图片，请换个描述试试。")
            
    except Exception as e:
        debug_log(f"生图出错: user_id={user_id}, error={e}", e)
        await update.message.reply_text(f"出错啦: {str(e)}")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    debug_log(f"Telegram handler 未捕获异常: update={update}, error={context.error}", context.error)


def main():
    token = os.getenv('TELEGRAM_HELPER_BOT_TOKEN')
    if not token:
        print("错误：请设置 TELEGRAM_HELPER_BOT_TOKEN")
        return

    debug_log(
        "Helper Bot 准备启动: "
        f"token_configured={bool(token)}, "
        f"TWO_DATABASE_URL_configured={bool(os.getenv('TWO_DATABASE_URL'))}, "
        f"DEFAULT_AI_PROVIDER={DEFAULT_AI_PROVIDER}, "
        f"GEMINI_CHAT_MODEL={GEMINI_CHAT_MODEL}, "
        f"OPENAI_CHAT_MODEL={OPENAI_CHAT_MODEL}, "
        f"OPENAI_REASONING_EFFORT={OPENAI_REASONING_EFFORT}, "
        f"OPENAI_BASE_URL_configured={bool(OPENAI_BASE_URL)}"
    )

    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("gen", start_gen))
    application.add_handler(CommandHandler("gen_me", start_gen))
    application.add_handler(CommandHandler("gen_user", start_gen))
    application.add_handler(CommandHandler("gen_both", start_gen))
    application.add_handler(CommandHandler("ai", start_ai))
    application.add_handler(CommandHandler("aimodel", ai_model_command))
    application.add_handler(CommandHandler("use", use_model_command))
    application.add_handler(CommandHandler("kline", kline_command))
    application.add_handler(CommandHandler("outk", outk_command))
    application.add_handler(CommandHandler(["clearAuto", "clearauto"], clear_auto_command))
    application.add_handler(CommandHandler("aiend", end_ai))
    application.add_handler(CommandHandler("end", end_gen))
    application.add_handler(CommandHandler("help", help_command))
    application.add_error_handler(error_handler)
    
    application.add_handler(MessageHandler(filters.Regex(r"^[/／]kline(?:@\w+)?(?:\s|$)"), kline_message_fallback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("Helper Bot 已启动", flush=True)
    application.run_polling()

if __name__ == "__main__":
    main()
