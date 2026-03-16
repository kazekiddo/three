import os
import re
import logging
import json
import random
from dotenv import load_dotenv
from telegram import Update
from telegram.request import HTTPXRequest
from telegram.error import TimedOut, NetworkError
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types
from database import Database
import datetime
import time
import io
from PIL import Image


# 只记录错误日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)
logger = logging.getLogger(__name__)

load_dotenv()

async def _send_with_retry(coro_factory, label: str, retries: int = 2, base_delay: float = 0.7):
    """对 Telegram 发送请求做轻量重试，避免偶发网络超时导致整轮失败。"""
    import asyncio
    for attempt in range(retries + 1):
        try:
            return await coro_factory()
        except (TimedOut, NetworkError) as e:
            if attempt >= retries:
                raise
            delay = base_delay * (2 ** attempt)
            logger.error(f"{label} timeout, retry in {delay:.1f}s (attempt {attempt+1}/{retries})")
            await asyncio.sleep(delay)

# 确保媒体目录存在
MEDIA_DIR = os.path.join(os.getcwd(), 'media', 'photos')
os.makedirs(MEDIA_DIR, exist_ok=True)

DEFAULT_CHAT_MODEL = "gemini-3-flash-preview"
SUPPORTED_CHAT_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash",
]

def _parse_time_range(message):
    """从用户消息中解析时间范围关键词，返回 (time_start, time_end) 或 (None, None)"""
    now = datetime.datetime.now()
    
    # X天前
    m = re.search(r'(\d+)\s*天前', message)
    if m:
        days = int(m.group(1))
        target = now - datetime.timedelta(days=days)
        return target.replace(hour=0, minute=0, second=0, microsecond=0), target.replace(hour=23, minute=59, second=59, microsecond=0)
    
    # X周前
    m = re.search(r'(\d+)\s*周前', message)
    if m:
        weeks = int(m.group(1))
        target = now - datetime.timedelta(weeks=weeks)
        start = target - datetime.timedelta(days=target.weekday())  # 那周一
        end = start + datetime.timedelta(days=6)  # 那周日
        return start.replace(hour=0, minute=0, second=0, microsecond=0), end.replace(hour=23, minute=59, second=59, microsecond=0)
    
    # X(个)月前
    m = re.search(r'(\d+)\s*个?月前', message)
    if m:
        months = int(m.group(1))
        year, month = now.year, now.month - months
        while month <= 0:
            month += 12
            year -= 1
        import calendar
        last_day = calendar.monthrange(year, month)[1]
        return datetime.datetime(year, month, 1), datetime.datetime(year, month, last_day, 23, 59, 59)
    
    # 大前天
    if '大前天' in message:
        d = now - datetime.timedelta(days=3)
        return d.replace(hour=0, minute=0, second=0, microsecond=0), d.replace(hour=23, minute=59, second=59, microsecond=0)
    
    # 前天
    if '前天' in message:
        d = now - datetime.timedelta(days=2)
        return d.replace(hour=0, minute=0, second=0, microsecond=0), d.replace(hour=23, minute=59, second=59, microsecond=0)
    
    # 昨天
    if '昨天' in message:
        d = now - datetime.timedelta(days=1)
        return d.replace(hour=0, minute=0, second=0, microsecond=0), d.replace(hour=23, minute=59, second=59, microsecond=0)
    
    # 上(个)周/星期
    if re.search(r'上\s*个?\s*(周|星期)', message):
        days_since_monday = now.weekday()
        last_monday = now - datetime.timedelta(days=days_since_monday + 7)
        last_sunday = last_monday + datetime.timedelta(days=6)
        return last_monday.replace(hour=0, minute=0, second=0, microsecond=0), last_sunday.replace(hour=23, minute=59, second=59, microsecond=0)
    
    # 上(个)月
    if re.search(r'上\s*个?\s*月', message):
        year, month = now.year, now.month - 1
        if month <= 0:
            month = 12
            year -= 1
        import calendar
        last_day = calendar.monthrange(year, month)[1]
        return datetime.datetime(year, month, 1), datetime.datetime(year, month, last_day, 23, 59, 59)
    
    # 去年
    if '去年' in message:
        y = now.year - 1
        return datetime.datetime(y, 1, 1), datetime.datetime(y, 12, 31, 23, 59, 59)
    
    # 几天前（模糊，取2-7天范围）
    if '几天前' in message:
        return (now - datetime.timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0), (now - datetime.timedelta(days=2)).replace(hour=23, minute=59, second=59, microsecond=0)
    
    # 几周前（模糊，取1-4周范围）
    if '几周前' in message:
        return (now - datetime.timedelta(weeks=4)).replace(hour=0, minute=0, second=0, microsecond=0), (now - datetime.timedelta(weeks=1)).replace(hour=23, minute=59, second=59, microsecond=0)
    
    # 几个月前（模糊，取1-4个月范围）
    if re.search(r'几个?月前', message):
        year_s, month_s = now.year, now.month - 4
        while month_s <= 0:
            month_s += 12
            year_s -= 1
        year_e, month_e = now.year, now.month - 1
        if month_e <= 0:
            month_e += 12
            year_e -= 1
        import calendar
        last_day_e = calendar.monthrange(year_e, month_e)[1]
        return datetime.datetime(year_s, month_s, 1), datetime.datetime(year_e, month_e, last_day_e, 23, 59, 59)
    
    return None, None


class ChatAI:
    def __init__(self, model=DEFAULT_CHAT_MODEL, api_key=None, system_instruction=None, character_id=None):
        """初始化聊天AI"""
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("请设置 GOOGLE_API_KEY 或 GEMINI_API_KEY 环境变量")
        
        # Gemini 请求超时与重试（避免长输出或网络抖动导致 Timed out）
        timeout_ms = 120000
        retry_attempts = 3
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=timeout_ms,
                retry_options=types.HttpRetryOptions(
                    attempts=retry_attempts,
                    initial_delay=1.0,
                    max_delay=10.0
                )
            )
        )
        self.model = model
        self.character_id = character_id
        self.system_instruction = system_instruction
        self.db = Database()
        self.last_message_timestamp = None
        self.last_user_message_timestamp = None  # 只在用户发消息时更新，供主动发言的 30 分钟门槛使用
        self.last_prefix_timestamp = None
        self.enable_web_search = os.getenv("ENABLE_WEB_SEARCH", "true").lower() in ("1", "true", "yes", "on")
        self.enable_prompt_cache = os.getenv("ENABLE_PROMPT_CACHE", "true").lower() in ("1", "true", "yes", "on")
        self.prompt_cache_ttl = os.getenv("PROMPT_CACHE_TTL", "86400s")
        self.proactive_streak_count = 0
        self.proactive_streak_date = None
        self.proactive_blocked_date = None
        self.last_proactive_image_timestamp = None
        self.proactive_image_cooldown_seconds = int(os.getenv("PROACTIVE_IMAGE_COOLDOWN_SECONDS", "3600"))
        self.proactive_image_probability = float(os.getenv("PROACTIVE_IMAGE_PROBABILITY", "0.35"))
        self.dynamic_state = self.db.get_dynamic_state(character_id) if character_id else None
        
        # 预加载角色设定图和用户（思远）设定图 (用于视觉一致性)
        self.character_photo_path = "/data/three/media/photos/photo_nanase.jpg"
        self.user_photo_path = "/data/three/media/photos/photo_siyuan.jpg"
        
        # 从数据库加载过去 24 小时的历史记录作为当天的缓冲区
        history = []
        # base_history 保存基础系统提示条目、每次 clear_history 后复用
        self.base_history = []
        if os.path.exists(self.character_photo_path):
            history.append({
                'role': 'user',
                'parts': [{'text': "系统通知：你有固定形象参考图。仅在需要生成你自己的图片时才启用这份视觉参考，并保持一致性。"}]
            })
            history.append({
                'role': 'model',
                'parts': [{'text': "收到。普通文字聊天时我不会反复提这件事，只有生成相关图片时才会按设定保持一致。"}]
            })
        if os.path.exists(self.user_photo_path):
            history.append({
                'role': 'user',
                'parts': [{'text': "系统通知：你也有思远的肖像参考图。仅在需要生成与他相关的图片时使用，不必在普通聊天里反复提及。"}]
            })
            history.append({
                'role': 'model',
                'parts': [{'text': "知道了。只有在生成和他相关的图片时我才会调用这份参考。"}]
            })
        self.base_history = list(history)

        # 从数据库加载动态缓冲：0-4点加载前一天4点后的记录；5-23点加载当天4点后的记录
        if character_id:
            self.last_message_timestamp = self.db.get_last_message_timestamp(character_id)
            self.last_user_message_timestamp = self.db.get_last_user_message_timestamp(character_id)
            
            # 计算起始时间，以凌晨 4 点为隔离锚点
            now = datetime.datetime.now()
            if now.hour < 4:
                # 0-3点时，需要回溯到前一天的 4 点
                since_time = (now - datetime.timedelta(days=1)).replace(hour=4, minute=0, second=0, microsecond=0)
            else:
                # 4点后，只需回溯到今天的 4 点
                since_time = now.replace(hour=4, minute=0, second=0, microsecond=0)
            
            db_messages = self.db.get_recent_chat_history(character_id, since_time=since_time)
            for msg in db_messages:
                # 若数据库中有单独的时间锚点缓存，提取并包裹
                prefix = f"{msg['context_prefix']} " if msg.get('context_prefix') else ""
                text_content = f"{prefix}{msg['content']}"

                parts = [{'text': text_content}]
                
                history.append({
                    'role': msg['role'],
                    'parts': parts
                })
        
        def classify_image_subject(prompt_text: str):
            """根据提示词判断生图主体，返回 character_only/user_only/both/non_portrait/none"""
            text = (prompt_text or "").strip().lower()
            if not text:
                return "none"

            def _has_any_word(patterns, haystack: str) -> bool:
                for p in patterns:
                    if re.search(p, haystack):
                        return True
                return False

            def _has_any_substring(words, haystack: str) -> bool:
                return any(w in haystack for w in words)

            # 文件名锚点优先级最高：直接用设定图名决定主体
            has_char_anchor = ("photo_nanase" in text) or ("nanase.jpg" in text)
            has_user_anchor = ("photo_siyuan" in text) or ("siyuan.jpg" in text)
            # 英文用词边界匹配，避免 woman 中误匹配 man
            has_boy_word = _has_any_word(
                [
                    r"\bboy\b",
                    r"\bman\b",
                    r"\byoung\s+man\b",
                    r"\bmale\b",
                    r"\bboyfriend\b",
                    r"\bboy[-\s]?friend\b",
                    r"\bbf\b",
                    r"\bpartner\b",
                    r"\blover\b",
                    r"\bsignificant\s+other\b",
                ],
                text
            ) or _has_any_substring(
                ["男生", "男孩", "男人", "少年", "男朋友", "情侣", "恋人", "对象", "伴侣", "男女朋友"],
                text
            )
            has_girl_word = _has_any_word(
                [
                    r"\bgirl\b",
                    r"\bwoman\b",
                    r"\byoung\s+woman\b",
                    r"\bfemale\b",
                    r"\bgirlfriend\b",
                    r"\bgirl[-\s]?friend\b",
                    r"\bgf\b",
                    r"\bpartner\b",
                    r"\blover\b",
                    r"\bsignificant\s+other\b",
                ],
                text
            ) or _has_any_substring(
                ["女生", "女孩", "女人", "少女", "女朋友", "情侣", "恋人", "对象", "伴侣", "男女朋友"],
                text
            )
            both_keywords = [
                "我们", "一起", "同框", "合照", "你和我", "我和你", "咱俩", "两个人",
                "both of us", "you and me", "me and you", "with siyuan", "with the user", "couple shot"
            ]
            # 这里避免使用裸词 you / me，减少误判为双人图
            character_keywords = [
                "你自己", "小七", "xiaoqi", "xiao qi", "nanase",
                "the girl", "the character", "你的照片", "你的样子", "自拍",
                "solo girl", "girl only", "character only"
            ]
            user_keywords = [
                "我自己", "思远", "siyuan", "the user", "user only", "我的照片",
                "我长什么样", "给我画", "my portrait", "draw me", "me only"
            ]

            if has_char_anchor and has_user_anchor:
                return "both"
            if has_char_anchor and (any(k in text for k in user_keywords) or any(k in text for k in both_keywords)):
                return "both"
            if has_user_anchor and (any(k in text for k in character_keywords) or any(k in text for k in both_keywords)):
                return "both"
            # 明确出现 boy + girl 时，直接视为双人图
            if has_boy_word and has_girl_word:
                return "both"
            # 仅给角色锚点，但语义明确是双人互动时，也判为双人
            dual_interaction_keywords = [
                "hug", "hugging", "embrace", "from behind", "couple", "intimate", "romantic",
                "抱", "拥抱", "搂", "从背后", "同框", "情侣", "亲密"
            ]
            if has_char_anchor and any(k in text for k in dual_interaction_keywords):
                if _has_any_word(
                    [
                        r"\byoung\s+man\b",
                        r"\bman\b",
                        r"\bboy\b",
                        r"\bmale\b",
                        r"\bboyfriend\b",
                        r"\bboy[-\s]?friend\b",
                        r"\bbf\b",
                        r"\bpartner\b",
                        r"\blover\b",
                        r"\bsignificant\s+other\b",
                    ],
                    text
                ) or _has_any_substring(
                    ["男生", "男人", "男孩", "少年", "他", "男朋友", "情侣", "恋人", "对象", "伴侣", "男女朋友"],
                    text
                ):
                    return "both"
            if has_char_anchor:
                return "character_only"
            if has_user_anchor:
                return "user_only"

            explicit_non_portrait_keywords = [
                "纯风景", "只有风景", "不要人物", "不需要人物", "无人", "没人",
                "纯场景", "纯物体", "只有物体", "静物", "产品图",
                "landscape only", "scenery only", "no people", "without people",
                "object only", "still life", "product shot"
            ]
            person_keywords = [
                "人", "人物", "女生", "女孩", "男生", "肖像", "自拍", "半身", "全身",
                "person", "people", "girl", "boy", "portrait", "selfie", "character"
            ]

            if any(k in text for k in explicit_non_portrait_keywords) and not any(
                k in text for k in person_keywords
            ):
                return "non_portrait"

            if any(k in text for k in both_keywords):
                return "both"

            has_character = any(k in text for k in character_keywords)
            has_user = any(k in text for k in user_keywords)

            if has_character and has_user:
                return "both"
            if has_character:
                return "character_only"
            if has_user:
                return "user_only"
            return "none"

        def build_image_prompt(prompt_text: str, subject_mode: str) -> str:
            """构造更强的 identity preservation 提示词"""
            base_scene = (prompt_text or "").strip()
            if not base_scene:
                base_scene = "A high quality Japanese anime illustration."

            if subject_mode == "character_only":
                identity_block = (
                    "Create an illustration of the exact same girl as reference image 1. "
                    "She must remain the same person, not a redesigned variant. "
                    "Identity preservation requirements: keep the same face shape, eyes, eye shape, hairstyle, bangs, "
                    "hair color, age impression, and overall facial identity. Do not turn her into a different anime girl. "
                    "Reference image 1 is the main and only identity reference. "
                    "If any appearance description in scene text conflicts with reference image 1, always follow reference image 1."
                )
            elif subject_mode == "user_only":
                identity_block = (
                    "Create an illustration of the exact same person as reference image 1. "
                    "He must remain the same person, not a redesigned variant. "
                    "Identity preservation requirements: keep the same face shape, eyes, hairstyle, hair color, "
                    "age impression, and overall facial identity. Do not redesign him into a different anime character. "
                    "Reference image 1 is the main and only identity reference. "
                    "If any appearance description in scene text conflicts with reference image 1, always follow reference image 1."
                )
            elif subject_mode == "both":
                identity_block = (
                    "Create an illustration featuring the same two people as the two reference images. "
                    "Reference image 1 is the girl, reference image 2 is the user Siyuan. "
                    "Both must remain the same people, not redesigned variants. "
                    "Preserve each person's face shape, eyes, hairstyle, hair color, age impression, "
                    "and overall identity. Do not merge their facial features. "
                    "If scene text conflicts with either reference appearance, always follow the references."
                )
            else:
                identity_block = (
                    "Create a high quality Japanese anime illustration. "
                    "No identity reference is required unless a specific person is clearly requested."
                )

            return (
                f"{identity_block}\n\n"
                f"Scene request:\n{base_scene}\n\n"
                "Style requirements:\n"
                "- Strictly Japanese anime / cartoon style\n"
                "- clean line art\n"
                "- coherent facial structure\n"
                "- soft natural lighting\n"
                "- avoid changing the identity of referenced people\n"
            )

        def sanitize_scene_prompt_for_identity(scene_text: str, subject_mode: str) -> str:
            """
            参考图模式下，清理与身份外观强相关的描述，避免上游工具参数把发型等特征写偏。
            只保留场景动作、氛围、构图等描述。
            """
            text = (scene_text or "").strip()
            if not text:
                return text
            if subject_mode not in ("character_only", "user_only", "both"):
                return text

            patterns = [
                # file-name anchors should not become scene content
                r"\bphoto_nanase(?:\.jpg)?\b",
                r"\bphoto_siyuan(?:\.jpg)?\b",
                r"\bnanase\.jpg\b",
                r"\bsiyuan\.jpg\b",
                # English hair descriptors
                r"\b(long|short|medium|shoulder[- ]length|waist[- ]length)\s+(straight|wavy|curly)?\s*hair\b",
                r"\b(straight|wavy|curly)\s+(black|brown|blonde|silver|white|red|pink|blue|purple)?\s*hair\b",
                r"\bwith\s+[a-z\s,/-]{0,40}?hair\b",
                # Chinese hair descriptors
                r"(长发|短发|中长发|齐肩发|黑长直|卷发|直发|波浪发|双马尾|单马尾|丸子头|刘海)",
                # face/identity-detail style hints that can drift identity
                r"\bwith\s+(a\s+)?(beautiful|cute|handsome|refined|delicate)\s+(face|features|look)\b",
            ]

            cleaned = text
            for p in patterns:
                cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)

            # collapse punctuation/whitespace artifacts after removals
            cleaned = re.sub(r"\s{2,}", " ", cleaned)
            cleaned = re.sub(r"\s+,", ",", cleaned)
            cleaned = re.sub(r",\s*,", ", ", cleaned)
            cleaned = cleaned.strip(" ,;，；")
            return cleaned or text

        # 定义 AI 生成图片的工具
        def generate_image(prompt: str) -> str:
            """根据描述生成一张精美的图片。
            注意：所有生成的图像必须是日系卡通风格（Japanese anime style）。如果你生成的是关于你自己的图片，请结合你记忆中设定图（photo_nanase.jpg）的视觉特征（如面部、发型、风格）来编写 prompt，以保证一致性。
            参数:
                prompt: 详细的图片描述词，使用英文描述效果更佳。
            """
            try:
                generation_contents = []
                subject_mode = classify_image_subject(prompt)
                # 主体不明确时默认带角色设定图；明确非人物请求则不注入人物参考图
                if subject_mode == "none":
                    subject_mode = "character_only"
                elif subject_mode == "non_portrait":
                    subject_mode = "none"
                sanitized_prompt = sanitize_scene_prompt_for_identity(prompt, subject_mode)
                full_prompt = build_image_prompt(sanitized_prompt, subject_mode)
                generation_contents.append(full_prompt)

                if subject_mode in ("character_only", "both") and os.path.exists(self.character_photo_path):
                    try:
                        ref_char_image = Image.open(self.character_photo_path)
                        generation_contents.append(ref_char_image)
                    except Exception as e:
                        logger.error(f"加载角色设定图失败: {e}")

                if subject_mode in ("user_only", "both") and os.path.exists(self.user_photo_path):
                    try:
                        ref_user_image = Image.open(self.user_photo_path)
                        generation_contents.append(ref_user_image)
                    except Exception as e:
                        logger.error(f"加载用户设定图失败: {e}")

                # 调用 gemini-3.1-flash-image-preview 生成图片，暂不设置不支持的 image_size
                response = self.client.models.generate_content(
                    model='gemini-3.1-flash-image-preview',
                    contents=generation_contents,
                    config=types.GenerateContentConfig(
                        image_config=types.ImageConfig(
                            aspect_ratio="3:4"
                        )
                    )
                )
                
                saved_path = None
                parts_list = response.parts if response.parts else []
                for part in parts_list:
                    if part.inline_data is not None:
                        # 确定保存路径，使用 .jpg 格式
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"ai_gen_{self.character_id}_{timestamp}.jpg"
                        saved_path = os.path.join(MEDIA_DIR, filename)
                        
                        # 直接从原始字节流使用 PIL 加载图片，避开 SDK 的 attribute 错误
                        image_out = Image.open(io.BytesIO(part.inline_data.data))
                        
                        # 如果图片过大（比如超过 1280 像素长边），按比例缩放以减小体积
                        max_size = 1280
                        if max(image_out.size) > max_size:
                            image_out.thumbnail((max_size, max_size), Image.LANCZOS)
                        
                        # 转换为 RGB 并保存为高质量 JPEG
                        if image_out.mode in ('RGBA', 'P'):
                            image_out = image_out.convert('RGB')
                        
                        image_out.save(saved_path, "JPEG", quality=85, optimize=True)
                        break
                
                if saved_path:
                    # 记录待发送的图片路径
                    self.pending_output_image = saved_path
                    return "SUCCESS: Image has been generated and will be sent to the user automatically by the system. Just continue your conversation naturally."
                else:
                    return "ERROR: Failed to generate image content."
            except Exception as e:
                logger.error(f"AI 生成图片失败: {e}")
                return f"生成图片失败: {str(e)}"

        def register_reminder(remind_at_str: str, content: str) -> str:
            """当你答应要在未来某个时间提醒思远做某事时，必须调用此函数将任务存入你的记忆。
            参数:
                remind_at_str: 提醒的具体时间, 格式为 'YYYY-MM-DD HH:MM:SS'。请务必根据注入的当前系统时间准确推算。
                content: 提醒内容。若用户给了原因，必须把原因和事项一起写入（如“因为明天要上班，所以今晚早点休息”）；
                         若用户未给原因，仅写事项本身（如“今晚早点休息”）。
            """
            try:
                remind_at = None
                raw = str(remind_at_str).strip() if remind_at_str else ""
                if raw:
                    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
                        try:
                            remind_at = datetime.datetime.strptime(raw, fmt)
                            break
                        except ValueError:
                            continue
                if not remind_at:
                    return f"ERROR: 提醒时间无法识别（{remind_at_str}），请给出更明确的时间。"
                
                # 存入数据库
                self.db.add_reminder(
                    self.character_id,
                    ALLOWED_USER_ID,
                    content,
                    remind_at,
                    source_type='user_request'
                )
                return f"SUCCESS: 我已经妥善记下了。我会在 {remind_at_str} 准时提醒你：{content}。"
            except Exception as e:
                logger.error(f"注册提醒失败: {e}")
                return f"ERROR: 记下提醒时出错了: {str(e)}"

        self.tools = [generate_image, register_reminder]
        self.tool_registry = {
            "generate_image": generate_image,
            "register_reminder": register_reminder
        }
        self.cached_tools = None
        self.cached_prefix_history = []
        self.cache_rotate_min_messages = 50
        self.cache_pending_messages = 0
        self.cache_rotate_min_prompt_delta = 4000
        self.last_prompt_tokens = None
        self.last_cached_tokens = None
        self.last_prompt_delta = None
        self.pending_output_image = None
        
        # 创建聊天
        if system_instruction:
            # 追加图片工具使用约束，防止模型试图直接输出图片
            image_constraint = (
                "\n\n【重要工具使用规则】"
                "你不能直接输出或嵌入图片。你没有原生图片输出能力。"
                "当用户请求图片、自拍、照片时，你必须调用 generate_image 工具函数来生成图片。"
                "调用 generate_image 时，请在 prompt 里明确写清楚画的是‘你自己’、‘思远本人’还是‘你和思远同框’，不要含糊带过。"
                "绝对不要在回复中写 'Here is the original image' 或类似的占位文字。"
                "调用工具后，系统会自动将图片发送给用户。"
            )
            
            reminder_constraint = (
                "\n\n【角色化记忆规则】"
                "你拥有‘提醒’能力。但请记住你目前的身份和性格，你不是思远的私人秘书。对于他提出的提醒要求，你应该根据当时的心情来决定是否帮忙。"
                "如果觉得他太依赖你、或者你正在闹别扭，你可以拒绝、抱怨、让他自己去记、或者要求他稍微求求你才行。不要像个无情的自动办公软件一样秒答应。"
                "只要你给出了任何明确承诺（例如“我帮你记着”“我会提醒你”“行，我记下了”），就必须在同一轮立即调用 register_reminder 工具，不能只口头答应不落库。"
                "如果时间不够明确（例如只说“晚上”但没有日期），你必须先追问澄清时间，再决定是否调用工具。"
                "调用 register_reminder 时，content 字段必须保留‘原因→事项’信息：如果用户说了原因，就写成“因为…，所以提醒…/…”。如果用户没说原因，就只写事项。"
                "记住，你一旦调用了工具，就代表你‘真的答应了’，这种承诺对你来说是很正式的，要表现出那种‘虽然很麻烦但为了你还是破例一次’的傲娇感。"
            )
            self.system_instruction = system_instruction + image_constraint + reminder_constraint
        self.cached_content_name = None
        if self.enable_prompt_cache:
            self.cached_tools = self._build_cached_tools()
            self._init_prompt_cache()
        self.chat = self.client.chats.create(
            model=model,
            config=self._build_chat_config(),
            history=self._strip_base_history(history) if self.cached_content_name else history
        )

    def _build_chat_config(self):
        def _set_field(cfg, model_cls, key, alt_key, value):
            try:
                fields = getattr(model_cls, "model_fields", None) or {}
                if key in fields:
                    cfg[key] = value
                elif alt_key in fields:
                    cfg[alt_key] = value
            except Exception:
                pass

        if self.cached_content_name:
            cfg = {}
            _set_field(cfg, types.GenerateContentConfig, "cachedContent", "cached_content", self.cached_content_name)
            return cfg
        config = {
            'automaticFunctionCalling': types.AutomaticFunctionCallingConfig(),
            'tools': self.tools
        }
        if self.system_instruction:
            _set_field(config, types.GenerateContentConfig, "systemInstruction", "system_instruction", self.system_instruction)
        return config

    def _delete_cached_content(self, name: str):
        if not name:
            return
        try:
            self.client.caches.delete(name=name)
        except Exception as e:
            logger.error(f"删除缓存失败: {e}")

    def drop_cache_now(self):
        """仅删除缓存，不重建。"""
        old_cache_name = self.cached_content_name
        self.cached_content_name = None
        if old_cache_name:
            self._delete_cached_content(old_cache_name)
        # 重建 chat（保留已有 history，避免丢上下文）
        history = getattr(self.chat, "_curated_history", None) or []
        self.chat = self.client.chats.create(
            model=self.model,
            config=self._build_chat_config(),
            history=history
        )
        return True, "缓存已删除"

    def _log_cache_usage(self, response, context_label: str):
        """记录缓存命中情况（若 SDK 提供 usage_metadata 字段）"""
        try:
            usage = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)
            if not usage:
                return
            # usage 可能是对象或 dict
            def _get(obj, key):
                return getattr(obj, key, None) if not isinstance(obj, dict) else obj.get(key)

            cached_tokens = _get(usage, "cached_content_token_count")
            cached_tokens = cached_tokens if cached_tokens is not None else _get(usage, "cachedContentTokenCount")
            total_tokens = _get(usage, "total_token_count")
            total_tokens = total_tokens if total_tokens is not None else _get(usage, "totalTokenCount")
            input_tokens = _get(usage, "prompt_token_count")
            input_tokens = input_tokens if input_tokens is not None else _get(usage, "promptTokenCount")
            output_tokens = _get(usage, "candidates_token_count")
            output_tokens = output_tokens if output_tokens is not None else _get(usage, "candidatesTokenCount")

            if cached_tokens is None and total_tokens is None:
                return

            if input_tokens is not None:
                self.last_prompt_tokens = input_tokens
            if cached_tokens is not None:
                self.last_cached_tokens = cached_tokens
            if input_tokens is not None and cached_tokens is not None:
                self.last_prompt_delta = input_tokens - cached_tokens

            logger.error(
                f"[cache] {context_label} cached_tokens={cached_tokens} "
                f"prompt_tokens={input_tokens} output_tokens={output_tokens} total_tokens={total_tokens}"
            )

            if (
                self.cached_content_name
                and self.last_prompt_delta is not None
                and self.last_prompt_delta > self.cache_rotate_min_prompt_delta
                and "_tool_followup" not in context_label
            ):
                ok, msg = self.rebuild_cache_now()
                logger.error(
                    f"[cache] auto_rebuild delta={self.last_prompt_delta} "
                    f"threshold={self.cache_rotate_min_prompt_delta} ok={ok}"
                )
        except Exception:
            pass

    def _build_cached_tools(self):
        try:
            generate_image_decl = types.FunctionDeclaration(
                name="generate_image",
                description="根据描述生成一张精美的图片（必须日系卡通风格）。",
                parametersJsonSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "详细的图片描述词，使用英文描述效果更佳。"
                        }
                    },
                    "required": ["prompt"]
                }
            )
            register_reminder_decl = types.FunctionDeclaration(
                name="register_reminder",
                description="将提醒任务存入记忆，要求给出明确时间与内容。",
                parametersJsonSchema={
                    "type": "object",
                    "properties": {
                        "remind_at_str": {
                            "type": "string",
                            "description": "提醒时间，格式 YYYY-MM-DD HH:MM:SS 或 YYYY-MM-DD HH:MM"
                        },
                        "content": {
                            "type": "string",
                            "description": "提醒内容（若有原因需包含原因→事项）"
                        }
                    },
                    "required": ["remind_at_str", "content"]
                }
            )
            return [types.Tool(functionDeclarations=[generate_image_decl, register_reminder_decl])]
        except Exception as e:
            logger.error(f"构建缓存工具声明失败: {e}")
            return None

    def _execute_tool_call(self, name: str, args):
        func = self.tool_registry.get(name)
        if not func:
            return f"ERROR: 未知工具 {name}"
        try:
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            if not isinstance(args, dict):
                args = {"_raw": args}
            return func(**args)
        except Exception as e:
            logger.error(f"工具执行失败: {name}, error={e}")
            return f"ERROR: 工具执行失败: {str(e)}"

    def _send_message_with_manual_tools(self, message_parts, context_label: str):
        response = self.chat.send_message(message_parts)
        self._log_cache_usage(response, context_label)
        max_rounds = 3
        for _ in range(max_rounds):
            func_calls = getattr(response, "function_calls", None) or []
            if not func_calls:
                return response
            response_parts = []
            for call in func_calls:
                name = getattr(call, "name", None)
                args = getattr(call, "args", None)
                if not name:
                    continue
                result = self._execute_tool_call(name, args)
                response_parts.append(
                    types.Part.from_function_response(name=name, response={"result": result})
                )
            if not response_parts:
                return response
            response = self.chat.send_message(response_parts)
            self._log_cache_usage(response, f"{context_label}_tool_followup")
        return response

    def _normalize_history_text(self, history):
        items = []
        for content in history or []:
            role = getattr(content, 'role', None) or content.get('role')
            parts = getattr(content, 'parts', None) or content.get('parts') or []
            texts = []
            for part in parts:
                text = getattr(part, 'text', None) if not isinstance(part, dict) else part.get('text')
                if text:
                    texts.append(str(text))
            items.append((role, "\n".join(texts)))
        return items

    def _history_to_dicts(self, history):
        items = []
        for content in history or []:
            if hasattr(content, "model_dump"):
                items.append(content.model_dump(exclude_none=True))
            elif isinstance(content, dict):
                items.append(content)
            else:
                items.append(content)
        return items

    def _strip_base_history(self, history):
        if not history or not self.base_history:
            return history or []
        if len(history) < len(self.base_history):
            return history
        if self._normalize_history_text(history[:len(self.base_history)]) == self._normalize_history_text(self.base_history):
            return history[len(self.base_history):]
        return history

    def _is_base_history(self, history):
        if not history:
            return True
        if len(history) != len(self.base_history):
            return False
        return self._normalize_history_text(history) == self._normalize_history_text(self.base_history)

    def _init_prompt_cache(self):
        try:
            display_name = f"tg_bot_prefix_cache_{self.character_id or 'default'}_{self.model}"
            if not self.cached_tools:
                raise ValueError("cached_tools 未就绪")

            fields = getattr(types.CreateCachedContentConfig, "model_fields", None) or {}
            cfg = {}
            if "displayName" in fields:
                cfg["displayName"] = display_name
            elif "display_name" in fields:
                cfg["display_name"] = display_name

            if "contents" in fields:
                cfg["contents"] = self.base_history + (self.cached_prefix_history or [])

            if "ttl" in fields:
                cfg["ttl"] = self.prompt_cache_ttl

            if "tools" in fields:
                cfg["tools"] = self.cached_tools

            tool_config = types.ToolConfig(
                functionCallingConfig=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.AUTO
                )
            )
            if "toolConfig" in fields:
                cfg["toolConfig"] = tool_config
            elif "tool_config" in fields:
                cfg["tool_config"] = tool_config

            if self.system_instruction:
                if "systemInstruction" in fields:
                    cfg["systemInstruction"] = self.system_instruction
                elif "system_instruction" in fields:
                    cfg["system_instruction"] = self.system_instruction

            config = types.CreateCachedContentConfig(**cfg)
            cache = self.client.caches.create(
                model=self.model,
                config=config
            )
            self.cached_content_name = cache.name
        except Exception as e:
            logger.error(f"创建提示词缓存失败: {e}")
            self.cached_content_name = None

    def _rotate_cache_before_time_anchor(self):
        if not self.enable_prompt_cache or not self.cached_tools:
            return
        if self.cache_pending_messages < self.cache_rotate_min_messages:
            return
        try:
            segment = self._history_to_dicts(getattr(self.chat, "_curated_history", []))
            if not segment:
                return
            new_prefix = list(self.cached_prefix_history or [])
            new_prefix.extend(segment)
            # 尝试创建新缓存（包含原有缓存前缀 + 本次历史）
            old_prefix = self.cached_prefix_history
            old_cache_name = self.cached_content_name
            self.cached_prefix_history = new_prefix
            self.cached_content_name = None
            self._init_prompt_cache()
            if not self.cached_content_name:
                # 失败则回滚前缀，保留原 chat 历史
                self.cached_prefix_history = old_prefix
                return
            # 新缓存成功后，删除旧缓存
            if old_cache_name and old_cache_name != self.cached_content_name:
                self._delete_cached_content(old_cache_name)
            # 缓存成功：重建 chat，仅保留后续新增历史
            self.chat = self.client.chats.create(
                model=self.model,
                config=self._build_chat_config(),
                history=[]
            )
            self.cache_pending_messages = 0
        except Exception as e:
            logger.error(f"轮换缓存失败: {e}")

    def rebuild_cache_now(self):
        """立即重建缓存：把当前 chat 历史全部并入缓存前缀并替换旧缓存。"""
        if not self.enable_prompt_cache or not self.cached_tools:
            return False, "缓存未启用或工具未就绪"
        try:
            segment = self._history_to_dicts(getattr(self.chat, "_curated_history", []))
            old_prefix = self.cached_prefix_history
            old_cache_name = self.cached_content_name
            if segment:
                self.cached_prefix_history = (self.cached_prefix_history or []) + segment
            self.cached_content_name = None
            self._init_prompt_cache()
            if not self.cached_content_name:
                self.cached_prefix_history = old_prefix
                return False, "创建缓存失败"
            if old_cache_name and old_cache_name != self.cached_content_name:
                self._delete_cached_content(old_cache_name)
            self.chat = self.client.chats.create(
                model=self.model,
                config=self._build_chat_config(),
                history=[]
            )
            self.cache_pending_messages = 0
            return True, "缓存已重建"
        except Exception as e:
            logger.error(f"手动重建缓存失败: {e}")
            return False, f"重建失败: {str(e)}"

    def switch_model(self, new_model: str):
        """切换对话模型并重建 chat，会尽量保留当前对话历史。"""
        self.model = new_model
        history = getattr(self.chat, '_curated_history', None) or self.base_history
        old_cache_name = self.cached_content_name
        self.cached_content_name = None
        if self.enable_prompt_cache:
            self.cached_tools = self._build_cached_tools()
            # 把当前 chat 历史并入缓存前缀，保证不丢对话
            extra = self._history_to_dicts(getattr(self.chat, "_curated_history", []))
            if extra:
                self.cached_prefix_history = (self.cached_prefix_history or []) + extra
            self._init_prompt_cache()
            if old_cache_name:
                self._delete_cached_content(old_cache_name)
            self.cache_pending_messages = 0
        self.chat = self.client.chats.create(
            model=self.model,
            config=self._build_chat_config(),
            history=self._strip_base_history(history) if self.cached_content_name else history
        )

    def _ensure_proactive_day_state(self, now_dt: datetime.datetime):
        today = now_dt.date()
        if self.proactive_streak_date != today:
            self.proactive_streak_date = today
            self.proactive_streak_count = 0
            self.proactive_blocked_date = None

    def can_send_proactive_today(self, now_dt: datetime.datetime) -> bool:
        self._ensure_proactive_day_state(now_dt)
        return self.proactive_blocked_date != now_dt.date()

    def on_user_replied(self, now_dt: datetime.datetime):
        self._ensure_proactive_day_state(now_dt)
        self.proactive_streak_count = 0
        self.proactive_blocked_date = None

    def on_proactive_sent(self, now_dt: datetime.datetime):
        self._ensure_proactive_day_state(now_dt)
        self.proactive_streak_count += 1
        if self.proactive_streak_count >= 5:
            self.proactive_blocked_date = now_dt.date()

    def _should_add_wakeup_prompt(self) -> bool:
        """北京时间 4:00-12:00 之间，且用户自 4:00 起未发言时，加入叫起床提示。"""
        tz_bj = datetime.timezone(datetime.timedelta(hours=8))
        now_bj = datetime.datetime.now(tz_bj)
        start_time = now_bj.replace(hour=4, minute=0, second=0, microsecond=0)
        noon_time = now_bj.replace(hour=12, minute=0, second=0, microsecond=0)
        if not (start_time <= now_bj < noon_time):
            return False

        last_user_ts = self.last_user_message_timestamp
        if not last_user_ts:
            return True

        if last_user_ts.tzinfo is None:
            last_user_ts = last_user_ts.replace(tzinfo=tz_bj)
        else:
            last_user_ts = last_user_ts.astimezone(tz_bj)
        return last_user_ts < start_time

    def _should_use_web_search(self, message: str) -> bool:
        """严格模式：仅在用户明确说“上网”时触发联网搜索"""
        if not self.enable_web_search or not message:
            return False
        text = message.strip().lower()
        if not text:
            return False
        return "上网" in text

    @staticmethod
    def _should_prefer_image_response(message: str, image_data=None) -> bool:
        """轻触发：命中视觉化意图时，偏好让模型走图片工具。"""
        if image_data is not None:
            return True
        if not message:
            return False

        text = message.strip().lower()
        if not text:
            return False

        visual_keywords = [
            # 通用视觉请求
            "图片", "照片", "拍", "图", "配图", "发图", "来图", "看图",
            "发一张", "来一张", "给我一张", "看一下", "看看", "看下", "看一眼",
            "长什么样", "样子", "外观", "给我看看", "让我看看", "给你看看",
            "做个图", "生成图", "画一张", "插画", "壁纸", "头像",
        ]
        return any(keyword in text for keyword in visual_keywords)

    def _should_trigger_proactive_image(self, message: str, image_data=None) -> bool:
        """主动发图触发器：不依赖用户索图语句，按冷却+概率触发。"""
        if image_data is not None:
            return False
        if self.proactive_image_probability <= 0:
            return False

        now_dt = datetime.datetime.now()
        if self.last_proactive_image_timestamp:
            elapsed = (now_dt - self.last_proactive_image_timestamp).total_seconds()
            if elapsed < self.proactive_image_cooldown_seconds:
                return False

        text = (message or "").strip()
        if not text:
            return False

        # 避免在明显任务型指令中硬插图片，保持自然感
        task_like_keywords = ["提醒", "几点", "时间", "上网", "查", "/"]
        if any(k in text for k in task_like_keywords):
            return False

        return random.random() < min(max(self.proactive_image_probability, 0.0), 1.0)

    @staticmethod
    def _extract_response_text(response) -> str:
        text = ""
        try:
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if getattr(part, "text", None):
                        text += part.text
        except Exception:
            pass
        return text.strip()

    @staticmethod
    def _extract_grounding_sources(response, limit=6):
        """从 grounding metadata 提取来源链接"""
        sources = []
        seen = set()
        try:
            response_dict = response.model_dump(exclude_none=True) if hasattr(response, "model_dump") else {}
            candidates = response_dict.get("candidates", [])
            for cand in candidates:
                gm = cand.get("grounding_metadata") or cand.get("groundingMetadata") or {}
                chunks = gm.get("grounding_chunks") or gm.get("groundingChunks") or []
                for chunk in chunks:
                    web_info = chunk.get("web") or {}
                    url = web_info.get("uri") or web_info.get("url")
                    title = web_info.get("title") or url
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    sources.append((title, url))
                    if len(sources) >= limit:
                        return sources
        except Exception:
            return sources
        return sources

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))

    @staticmethod
    def _clean_state_text(value, default, limit=80):
        text = str(value or "").strip()
        if not text:
            return default
        return text[:limit]

    def _recent_dialogue_excerpt(self, limit=8):
        if not self.character_id:
            return ""
        try:
            messages = self.db.get_chat_history(self.character_id, limit=limit)
        except Exception as e:
            logger.error(f"读取近期对话失败: {e}")
            return ""

        lines = []
        for msg in messages:
            content = (msg.get('content') or '').strip()
            if not content:
                continue
            role_name = "思远" if msg.get('role') == 'user' else "你"
            prefix = (msg.get('context_prefix') or '').strip()
            if prefix:
                lines.append(f"{prefix} {role_name}: {content}")
            else:
                lines.append(f"{role_name}: {content}")
        return "\n".join(lines[-limit:])

    def _build_compact_relationship_context(self):
        if not self.character_id:
            return ""
        try:
            state = self.db.get_relationship_state(self.character_id)
        except Exception as e:
            logger.error(f"读取关系状态失败: {e}")
            return ""
        if not state:
            return ""

        def judge(val):
            value = float(val or 0.0)
            if value >= 0.75:
                return "高"
            if value >= 0.55:
                return "偏高"
            if value <= 0.2:
                return "很低"
            if value <= 0.4:
                return "偏低"
            return "中"

        return (
            "[系统附加关系摘要]\n"
            f"- 阶段: {state.get('stage', 'unknown')}\n"
            f"- 亲密={judge(state.get('closeness'))} 信任={judge(state.get('trust'))} 安全={judge(state.get('security'))}\n"
            f"- 吸引={judge(state.get('attraction'))} 依赖={judge(state.get('dependency'))} 嫉妒={judge(state.get('jealousy'))} 怨念={judge(state.get('resentment'))}\n"
            f"- 当前叙事: {(state.get('narrative') or '暂无').strip()[:80]}\n"
            "回复时只把它当成底色，不要复读这些标签。"
        )

    @staticmethod
    def _safe_json_loads(raw_text):
        text = (raw_text or "").strip()
        if not text:
            return {}
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    def _normalize_float(self, value, fallback, low=0.0, high=1.0):
        try:
            value = float(value)
        except Exception:
            value = fallback
        return round(self._clamp(value, low, high), 2)

    def _normalize_dynamic_state(self, data, fallback=None):
        fallback = fallback or {}
        if not isinstance(data, dict):
            data = {}

        return {
            "scene_label": self._clean_state_text(
                data.get("scene_label"),
                fallback.get("scene_label") or "日常",
                limit=50
            ),
            "emotion_label": self._clean_state_text(
                data.get("emotion_label"),
                fallback.get("emotion_label") or "表面平静",
                limit=50
            ),
            "emotion_intensity": self._normalize_float(
                data.get("emotion_intensity", fallback.get("emotion_intensity", 0.38)),
                fallback=0.38
            ),
            "motivation_label": self._clean_state_text(
                data.get("motivation_label"),
                fallback.get("motivation_label") or "先接住他 再顺手要一点关注",
                limit=80
            ),
            "inhibition_label": self._clean_state_text(
                data.get("inhibition_label"),
                fallback.get("inhibition_label") or "不想显得太黏 也不想把话说满",
                limit=100
            ),
            "hidden_expectation": self._clean_state_text(
                data.get("hidden_expectation"),
                fallback.get("hidden_expectation") or "希望他主动多哄一点 多顺着自己一点",
                limit=120
            ),
            "last_user_intent": self._clean_state_text(
                data.get("last_user_intent"),
                fallback.get("last_user_intent") or "日常聊天",
                limit=80
            ),
            "user_affect": self._clean_state_text(
                data.get("user_affect"),
                fallback.get("user_affect") or "普通",
                limit=80
            ),
            "unresolved_need": self._clean_state_text(
                data.get("unresolved_need"),
                fallback.get("unresolved_need") or "想被在意 但不想直说",
                limit=140
            ),
            "carryover_summary": self._clean_state_text(
                data.get("carryover_summary"),
                fallback.get("carryover_summary") or "心里有点想黏人 但还端着",
                limit=180
            ),
            "reply_style": self._clean_state_text(
                data.get("reply_style"),
                fallback.get("reply_style") or "短句 口语化 别解释太满 带点嘴硬和留白",
                limit=180
            ),
            "warmth_bias": self._normalize_float(
                data.get("warmth_bias", fallback.get("warmth_bias", 0.58)),
                fallback=0.58
            ),
            "initiative_bias": self._normalize_float(
                data.get("initiative_bias", fallback.get("initiative_bias", 0.46)),
                fallback=0.46
            ),
            "last_trigger_source": self._clean_state_text(
                data.get("last_trigger_source"),
                fallback.get("last_trigger_source") or "日常互动",
                limit=80
            ),
            "repair_status": self._clean_state_text(
                data.get("repair_status"),
                fallback.get("repair_status") or "无需修复",
                limit=80
            ),
        }

    def _prepare_local_pre_reply_state(self, latest_user_message: str, relation_desc: str = "", proactive=False):
        """单次回复前：仅用本地规则做一次基础状态修正，避免额外模型调用。"""
        now_dt = datetime.datetime.now()
        existing_state = self.db.get_dynamic_state(self.character_id) if self.character_id else None
        existing_state = existing_state or self.dynamic_state or {}
        carryover_state = self._derive_carryover_state(existing_state, now_dt)
        normalized = self._normalize_dynamic_state({}, fallback=carryover_state or existing_state)
        normalized, rule_notes = self._apply_dynamic_state_rules(
            normalized,
            latest_user_message,
            relation_desc=relation_desc or "",
            proactive=proactive
        )
        return normalized, rule_notes

    def _build_single_call_state_patch_instruction(self):
        return (
            "\n\n[系统附加输出协议]\n"
            "你最终只能输出一个 JSON 对象，不要输出 markdown，不要输出代码块。\n"
            "格式必须是：\n"
            "{"
            "\"reply\":\"给用户的自然回复\","
            "\"state_patch\":{"
            "\"scene_label\":\"...\","
            "\"emotion_label\":\"...\","
            "\"repair_status\":\"...\","
            "\"carryover_summary\":\"...\","
            "\"warmth_bias\":0.0,"
            "\"initiative_bias\":0.0"
            "}"
            "}\n"
            "规则：\n"
            "1) reply 必须是最终给用户看的内容，自然口语化。\n"
            "2) state_patch 仅允许包含这 6 个键，可按需部分返回；不要返回其他键。\n"
            "3) warmth_bias 和 initiative_bias 必须在 0.0~1.0。\n"
            "4) 即使你调用了工具（如 generate_image/register_reminder），最终输出仍必须是上述 JSON。"
        )

    def _extract_json_payload(self, text: str):
        cleaned = (text or "").strip()
        if not cleaned:
            return {}
        try:
            return self._safe_json_loads(cleaned)
        except Exception:
            pass
        # 容错：提取文本中所有顶层 JSON 对象，优先返回第一个
        def _extract_top_level_json_objects(s: str):
            objects = []
            depth = 0
            in_str = False
            escape = False
            start_idx = None
            for i, ch in enumerate(s):
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == "\"":
                        in_str = False
                    continue
                if ch == "\"":
                    in_str = True
                    continue
                if ch == "{":
                    if depth == 0:
                        start_idx = i
                    depth += 1
                elif ch == "}":
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start_idx is not None:
                            objects.append(s[start_idx:i + 1])
                            start_idx = None
            return objects

        candidates = _extract_top_level_json_objects(cleaned)
        if candidates:
            for snippet in candidates:
                try:
                    return self._safe_json_loads(snippet)
                except Exception:
                    continue
        return {}

    def _merge_state_patch(self, base_state, state_patch):
        merged = dict(base_state or {})
        if not isinstance(state_patch, dict):
            return self._normalize_dynamic_state(merged, fallback=base_state or {})

        text_limits = {
            "scene_label": 50,
            "emotion_label": 50,
            "repair_status": 80,
            "carryover_summary": 180,
        }
        for key, limit in text_limits.items():
            if key in state_patch:
                merged[key] = self._clean_state_text(state_patch.get(key), merged.get(key, ""), limit=limit)

        if "warmth_bias" in state_patch:
            merged["warmth_bias"] = self._normalize_float(
                state_patch.get("warmth_bias"),
                fallback=float(merged.get("warmth_bias", 0.58))
            )
        if "initiative_bias" in state_patch:
            merged["initiative_bias"] = self._normalize_float(
                state_patch.get("initiative_bias"),
                fallback=float(merged.get("initiative_bias", 0.46))
            )

        return self._normalize_dynamic_state(merged, fallback=base_state or {})

    def _persist_dynamic_state_snapshot(self, state, source_kind, trigger_message=None, model_reply=None, notes=None):
        if not self.character_id or not state:
            return
        try:
            self.db.insert_dynamic_state_history(
                self.character_id,
                source_kind,
                state.get("scene_label"),
                state.get("emotion_label"),
                state.get("emotion_intensity"),
                state.get("motivation_label"),
                state.get("inhibition_label"),
                state.get("hidden_expectation"),
                state.get("last_user_intent"),
                state.get("user_affect"),
                state.get("unresolved_need"),
                state.get("carryover_summary"),
                state.get("reply_style"),
                state.get("warmth_bias"),
                state.get("initiative_bias"),
                state.get("last_trigger_source"),
                state.get("repair_status"),
                trigger_message=trigger_message,
                model_reply=model_reply,
                notes=notes
            )
        except Exception as e:
            logger.error(f"记录动态状态快照失败: {e}")

    def _apply_dynamic_state_rules(self, state, latest_user_message, relation_desc="", proactive=False):
        if not state:
            return state, []

        adjusted = dict(state)
        notes = []
        text = (latest_user_message or "").strip()
        lower = text.lower()

        def bump(key, delta, low=0.0, high=1.0):
            adjusted[key] = round(self._clamp(float(adjusted.get(key, 0.0)) + delta, low, high), 2)

        short_cold = len(text) <= 4 and text in {"哦", "噢", "额", "行", "随便", "不知道", "再说", "嗯", "。。"}
        if short_cold:
            adjusted["scene_label"] = "敷衍"
            adjusted["emotion_label"] = "有点不爽"
            adjusted["user_affect"] = "敷衍"
            adjusted["repair_status"] = "待安抚"
            adjusted["carryover_summary"] = "被敷衍了一下 心里有点刺"
            adjusted["unresolved_need"] = "想被认真接住"
            bump("emotion_intensity", 0.14, 0.18, 0.95)
            bump("warmth_bias", -0.12)
            bump("initiative_bias", -0.08)
            notes.append("检测到短促敷衍，降低热度并挂起待安抚状态")

        if any(k in text for k in ["想你", "抱抱", "亲亲", "爱你", "陪我", "老婆", "宝贝"]):
            adjusted["scene_label"] = "亲密"
            adjusted["emotion_label"] = "变软"
            adjusted["user_affect"] = "靠近"
            adjusted["repair_status"] = "正在变软"
            adjusted["carryover_summary"] = "被他往前抱了一下 心口软了点"
            adjusted["unresolved_need"] = "还想再多黏一会儿"
            bump("emotion_intensity", 0.08, 0.18, 0.95)
            bump("warmth_bias", 0.14)
            bump("initiative_bias", 0.10)
            notes.append("检测到亲密表达，提高热度与主动度")

        if any(k in text for k in ["对不起", "抱歉", "我错了", "别生气", "错啦"]):
            adjusted["scene_label"] = "修复"
            adjusted["user_affect"] = "在哄"
            adjusted["emotion_label"] = "嘴硬但松动"
            adjusted["repair_status"] = "正在变软"
            adjusted["carryover_summary"] = "他开始哄了 气还没散完 但没刚才那么硬"
            adjusted["unresolved_need"] = "想看他是不是认真"
            bump("warmth_bias", 0.10)
            bump("initiative_bias", 0.04)
            notes.append("检测到道歉/安抚，修复状态回暖")

        if any(k in lower for k in ["晚安", "睡了", "睡觉", "困死", "先睡"]):
            adjusted["scene_label"] = "收尾"
            adjusted["user_affect"] = "疲惫"
            adjusted["inhibition_label"] = "别拉太长 让他早点休息"
            adjusted["reply_style"] = "更短一点 软一点 别拉着继续聊"
            adjusted["carryover_summary"] = "这轮该轻一点收住"
            notes.append("检测到收尾场景，压低输出长度")

        if any(k in text for k in ["帮我", "怎么办", "怎么弄", "教我", "不会", "搞不定"]):
            adjusted["scene_label"] = "求助"
            adjusted["user_affect"] = "求助"
            adjusted["motivation_label"] = "先接住他 再认真帮一下"
            adjusted["reply_style"] = "先一句接情绪 再给简洁有用的话 别像客服"
            bump("warmth_bias", 0.06)
            notes.append("检测到求助场景，提高认真帮忙倾向")

        if any(k in text for k in ["和谁", "别的女生", "别人也这样", "她是谁", "你是不是"]):
            adjusted["scene_label"] = "试探"
            adjusted["emotion_label"] = "轻微吃醋"
            adjusted["user_affect"] = "试探"
            adjusted["carryover_summary"] = "有点在意这件事 语气会带刺一点"
            adjusted["unresolved_need"] = "想确认自己是不是被偏爱"
            bump("emotion_intensity", 0.10, 0.18, 0.95)
            notes.append("检测到试探/比较话题，抬高醋意")

        if proactive:
            adjusted["last_trigger_source"] = "主动发言"
        elif text:
            adjusted["last_trigger_source"] = "用户新消息"

        if relation_desc:
            if ("安全=很低" in relation_desc or "安全=偏低" in relation_desc) or \
               "Security(Very Low)" in relation_desc or "Security(Low)" in relation_desc:
                adjusted["hidden_expectation"] = "希望他多给一点确定感和偏爱"
                notes.append("长期安全感偏低，隐性期待转向确定感")
            if ("嫉妒=高" in relation_desc or "嫉妒=偏高" in relation_desc) or \
               "Jealousy(Very High)" in relation_desc or "Jealousy(High)" in relation_desc:
                if adjusted.get("scene_label") == "日常":
                    adjusted["scene_label"] = "试探"
                notes.append("长期嫉妒位较高，场景更容易滑向试探")

        return adjusted, notes

    def _derive_carryover_state(self, existing_state, now_dt):
        if not existing_state:
            return {}

        carry = dict(existing_state)
        updated_at = existing_state.get("updated_at")
        if isinstance(updated_at, datetime.datetime):
            hours_gap = max((now_dt - updated_at).total_seconds() / 3600.0, 0.0)
        else:
            hours_gap = 1.5

        decay = min(hours_gap * 0.08, 0.28)
        carry["emotion_intensity"] = round(
            self._clamp(float(existing_state.get("emotion_intensity", 0.38)) - decay, 0.18, 0.95),
            2
        )
        carry["warmth_bias"] = round(0.75 * float(existing_state.get("warmth_bias", 0.58)) + 0.25 * 0.58, 2)
        carry["initiative_bias"] = round(0.75 * float(existing_state.get("initiative_bias", 0.46)) + 0.25 * 0.46, 2)

        repair_status = str(existing_state.get("repair_status") or "")
        if repair_status in {"待安抚", "有点别扭"} and hours_gap >= 6:
            carry["repair_status"] = "情绪淡了 但还记得"

        carryover = str(existing_state.get("carryover_summary") or "").strip()
        if carryover and hours_gap >= 10:
            carry["carryover_summary"] = f"之前那点情绪淡了些 但余味还在：{carryover[:60]}"

        return carry

    def _infer_state_and_plan(self, latest_user_message: str, relation_desc: str = "", image_requested=False, proactive=False):
        if not self.character_id or not latest_user_message.strip():
            return self.dynamic_state, {}

        now_dt = datetime.datetime.now()
        existing_state = self.db.get_dynamic_state(self.character_id) or self.dynamic_state or {}
        carryover_state = self._derive_carryover_state(existing_state, now_dt)
        recent_dialogue = self._recent_dialogue_excerpt(limit=8)
        now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        relationship_context = relation_desc.strip() if relation_desc else "暂无额外关系上下文"
        existing_state_json = json.dumps(existing_state, ensure_ascii=False, default=str)
        carryover_state_json = json.dumps(carryover_state, ensure_ascii=False, default=str)

        prompt = (
            f"当前时间（Asia/Shanghai）: {now_str}\n"
            "你是角色的【短期心智状态引擎 + 回合表达导演】。你的任务不是直接代替角色回复，而是先判断她此刻的内在状态，再决定这轮该怎么回。\n"
            "请输出一个 JSON 对象，包含 state 和 plan 两部分：\n"
            "{"
            "\"state\":{"
            "\"scene_label\":\"...\","
            "\"emotion_label\":\"...\","
            "\"emotion_intensity\":0.0,"
            "\"motivation_label\":\"...\","
            "\"inhibition_label\":\"...\","
            "\"hidden_expectation\":\"...\","
            "\"last_user_intent\":\"...\","
            "\"user_affect\":\"...\","
            "\"unresolved_need\":\"...\","
            "\"carryover_summary\":\"...\","
            "\"reply_style\":\"...\","
            "\"warmth_bias\":0.0,"
            "\"initiative_bias\":0.0,"
            "\"last_trigger_source\":\"...\","
            "\"repair_status\":\"...\""
            "},"
            "\"plan\":{"
            "\"response_mode\":\"...\","
            "\"tone\":\"...\","
            "\"goal\":\"...\","
            "\"should_ask_question\":true,"
            "\"should_tease\":false,"
            "\"should_offer_help\":false,"
            "\"should_reference_memory\":false,"
            "\"should_be_extra_brief\":true,"
            "\"max_sentences\":2,"
            "\"warmth_level\":0.0,"
            "\"initiative_level\":0.0,"
            "\"notes\":\"...\""
            "}"
            "}\n"
            "约束：\n"
            "1) state.scene_label 必须是简短场景标签，如 日常/撒娇/求助/汇报/邀约/冲突/修复/试探/分享/敷衍。\n"
            "2) state.emotion_label 用简短中文，如 平静/开心/委屈/不安/吃醋/烦躁/想撒娇。\n"
            "3) state.emotion_intensity、state.warmth_bias、state.initiative_bias、plan.warmth_level、plan.initiative_level 都在 0.0 到 1.0 之间。\n"
            "4) state.reply_style 只描述表达方式，不能直接代写台词。\n"
            "5) plan.response_mode 只能是 接住/安抚/调情/吐槽/认真帮忙/轻微吃醋/追问/冷一点/分享/提醒。\n"
            "6) plan.goal 只写这轮唯一主目标；plan.max_sentences 取 1 到 4，大多数情况不超过 2。\n"
            "7) plan.should_offer_help 只有用户在求助、卡住、焦虑或明确要方案时才为 true。\n"
            "8) plan.should_tease 只在气氛安全时使用。\n"
            "9) 只输出 JSON，不要 markdown，不要解释。\n\n"
            f"已有短期状态：{existing_state_json}\n\n"
            f"按时间衰减后的延续状态：{carryover_state_json}\n\n"
            f"关系上下文：\n{relationship_context}\n\n"
            f"是否用户索图：{'是' if image_requested else '否'}\n"
            f"是否主动发言：{'是' if proactive else '否'}\n\n"
            f"最近对话：\n{recent_dialogue or '暂无'}\n\n"
            f"用户新消息：\n{latest_user_message}"
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    temperature=0.25
                )
            )
            raw_text = response.text.strip() if response and response.text else ""
            payload = self._safe_json_loads(raw_text) if raw_text else {}
            normalized = self._normalize_dynamic_state(payload.get("state"), fallback=carryover_state or existing_state)
            normalized, rule_notes = self._apply_dynamic_state_rules(
                normalized,
                latest_user_message,
                relation_desc=relationship_context,
                proactive=proactive
            )
            self.db.upsert_dynamic_state(
                self.character_id,
                normalized["scene_label"],
                normalized["emotion_label"],
                normalized["emotion_intensity"],
                normalized["motivation_label"],
                normalized["inhibition_label"],
                normalized["hidden_expectation"],
                normalized["last_user_intent"],
                normalized["user_affect"],
                normalized["unresolved_need"],
                normalized["carryover_summary"],
                normalized["reply_style"],
                normalized["warmth_bias"],
                normalized["initiative_bias"],
                normalized["last_trigger_source"],
                normalized["repair_status"]
            )
            self.dynamic_state = normalized
            self._persist_dynamic_state_snapshot(
                normalized,
                source_kind="pre_proactive" if proactive else "pre_reply",
                trigger_message=latest_user_message,
                notes="；".join(rule_notes) if rule_notes else ("pre_proactive_state" if proactive else "pre_reply_state")
            )
            state_json = json.dumps({
                "scene_label": normalized.get("scene_label"),
                "emotion_label": normalized.get("emotion_label"),
                "emotion_intensity": normalized.get("emotion_intensity"),
                "motivation_label": normalized.get("motivation_label"),
                "user_affect": normalized.get("user_affect"),
                "repair_status": normalized.get("repair_status"),
                "warmth_bias": normalized.get("warmth_bias"),
                "initiative_bias": normalized.get("initiative_bias"),
            }, ensure_ascii=False, default=str)
            plan_payload = payload.get("plan") if isinstance(payload, dict) else {}
            if not isinstance(plan_payload, dict):
                plan_payload = {}
            try:
                max_sentences = int(plan_payload.get("max_sentences", 2))
            except Exception:
                max_sentences = 2
            plan = {
                "response_mode": self._clean_state_text(plan_payload.get("response_mode"), "接住", limit=40),
                "tone": self._clean_state_text(plan_payload.get("tone"), "嘴硬里带一点软", limit=60),
                "goal": self._clean_state_text(plan_payload.get("goal"), "先接住对方 再保持点亲密感", limit=100),
                "should_ask_question": bool(plan_payload.get("should_ask_question", False)),
                "should_tease": bool(plan_payload.get("should_tease", False)),
                "should_offer_help": bool(plan_payload.get("should_offer_help", False)),
                "should_reference_memory": bool(plan_payload.get("should_reference_memory", False)),
                "should_be_extra_brief": bool(plan_payload.get("should_be_extra_brief", True)),
                "max_sentences": max(1, min(max_sentences, 4)),
                "warmth_level": self._normalize_float(
                    plan_payload.get("warmth_level", normalized.get("warmth_bias", 0.58)),
                    fallback=float(normalized.get("warmth_bias", 0.58))
                ),
                "initiative_level": self._normalize_float(
                    plan_payload.get("initiative_level", normalized.get("initiative_bias", 0.46)),
                    fallback=float(normalized.get("initiative_bias", 0.46))
                ),
                "notes": self._clean_state_text(plan_payload.get("notes"), "别太像助手", limit=120)
            }
            return normalized, plan
        except Exception as e:
            logger.error(f"推断状态与决策失败: {e}")
            self.dynamic_state = self._normalize_dynamic_state({}, fallback=carryover_state or existing_state)
            return self.dynamic_state, {
                "response_mode": "接住",
                "tone": "嘴硬里带一点软",
                "goal": "先接住对方 再保持点亲密感",
                "should_ask_question": False,
                "should_tease": False,
                "should_offer_help": False,
                "should_reference_memory": False,
                "should_be_extra_brief": True,
                "max_sentences": 2,
                "warmth_level": float(self.dynamic_state.get("warmth_bias", 0.58)),
                "initiative_level": float(self.dynamic_state.get("initiative_bias", 0.46)),
                "notes": "别太像助手"
            }

    def _update_post_reply_state(self, trigger_message, model_reply, base_state, turn_plan=None, proactive=False):
        if not self.character_id or not base_state:
            return base_state

        state = dict(base_state)
        text = (model_reply or "").strip()
        user_text = (trigger_message or "").strip()
        notes = []

        if not text and not self.pending_output_image:
            return state

        if turn_plan:
            state["reply_style"] = self._clean_state_text(
                turn_plan.get("notes"),
                state.get("reply_style") or "短句 口语化 别解释太满 带点嘴硬和留白",
                limit=180
            )

        if any(k in text for k in ["哼", "才没有", "随你", "自己想"]):
            state["carryover_summary"] = "嘴上还是硬的 但其实在等他继续接"
            state["repair_status"] = state.get("repair_status") or "有点别扭"
            notes.append("回复里保留嘴硬尾巴")

        if any(k in text for k in ["晚安", "早点睡", "快去睡", "休息"]):
            state["scene_label"] = "收尾"
            state["carryover_summary"] = "这轮收住了 但心还是贴着的"
            state["inhibition_label"] = "别再把话题拉长"
            notes.append("回复后进入收尾状态")

        if any(k in user_text for k in ["对不起", "抱歉", "我错了", "别生气", "错啦"]) and any(
            k in text for k in ["行吧", "这次算了", "原谅你", "勉强", "下不为例"]
        ):
            state["repair_status"] = "已被哄好"
            state["emotion_label"] = "嘴硬但好了"
            state["carryover_summary"] = "嘴上没全松 但其实已经被哄住了"
            state["unresolved_need"] = "想再被哄两句"
            state["warmth_bias"] = round(self._clamp(float(state.get("warmth_bias", 0.58)) + 0.08, 0.0, 1.0), 2)
            notes.append("识别到修复完成")

        if turn_plan and turn_plan.get("should_offer_help"):
            state["carryover_summary"] = "已经伸手帮他了 心里会更偏向继续接住"
            state["warmth_bias"] = round(self._clamp(float(state.get("warmth_bias", 0.58)) + 0.04, 0.0, 1.0), 2)
            notes.append("本轮提供帮助后，热度略升")

        if proactive:
            state["initiative_bias"] = round(self._clamp(float(state.get("initiative_bias", 0.46)) - 0.03, 0.0, 1.0), 2)
            state["carryover_summary"] = self._clean_state_text(
                state.get("carryover_summary"),
                "主动伸了一次手 现在等他接",
                limit=180
            )
            notes.append("主动发言后略微回收主动度")

        state["last_trigger_source"] = "主动发言后余波" if proactive else "回复后余波"
        try:
            self.db.upsert_dynamic_state(
                self.character_id,
                state["scene_label"],
                state["emotion_label"],
                state["emotion_intensity"],
                state["motivation_label"],
                state["inhibition_label"],
                state["hidden_expectation"],
                state["last_user_intent"],
                state["user_affect"],
                state["unresolved_need"],
                state["carryover_summary"],
                state["reply_style"],
                state["warmth_bias"],
                state["initiative_bias"],
                state["last_trigger_source"],
                state["repair_status"]
            )
            self.dynamic_state = state
            self._persist_dynamic_state_snapshot(
                state,
                source_kind="post_reply" if not proactive else "post_proactive",
                trigger_message=trigger_message,
                model_reply=model_reply,
                notes="；".join(notes) if notes else "post_reply_state"
            )
        except Exception as e:
            logger.error(f"写入回合后余波状态失败: {e}")
        return state

    def _build_dynamic_state_prompt(self, state=None, turn_plan=None):
        state = state or self.dynamic_state
        if not state:
            return ""

        try:
            intensity_text = f"{float(state.get('emotion_intensity', 0.38)):.2f}"
        except Exception:
            intensity_text = "0.38"
        try:
            warmth_text = f"{float(state.get('warmth_bias', 0.58)):.2f}"
        except Exception:
            warmth_text = "0.58"
        try:
            initiative_text = f"{float(state.get('initiative_bias', 0.46)):.2f}"
        except Exception:
            initiative_text = "0.46"

        prompt = (
            "\n\n[系统附加短期心智状态：仅供角色内在把握，不要直接复述这些标签]\n"
            f"- 当前场景：{state.get('scene_label', '日常')}\n"
            f"- 当前情绪：{state.get('emotion_label', '表面平静')} (强度 {intensity_text})\n"
            f"- 你感受到的用户状态：{state.get('user_affect', '普通')}\n"
            f"- 当前动机：{state.get('motivation_label', '先接住他 再顺手要一点关注')}\n"
            f"- 当前克制：{state.get('inhibition_label', '不想显得太黏 也不想把话说满')}\n"
            f"- 隐性期待：{self._clean_state_text(state.get('hidden_expectation'), '希望他主动多哄一点 多顺着自己一点', limit=40)}\n"
            f"- 尚未满足的点：{self._clean_state_text(state.get('unresolved_need'), '想被在意 但不想直说', limit=36)}\n"
            f"- 情绪余波：{self._clean_state_text(state.get('carryover_summary'), '心里有点想黏人 但还端着', limit=42)}\n"
            f"- 关系修复状态：{state.get('repair_status', '无需修复')}\n"
            f"- 表达建议：{self._clean_state_text(state.get('reply_style'), '短句 口语化 别解释太满 带点嘴硬和留白', limit=48)}\n"
            f"- 当前亲近倾向：warmth={warmth_text}, initiative={initiative_text}\n"
        )
        if turn_plan:
            prompt += (
                "\n[系统附加本轮表达决策]\n"
                f"- 回应模式：{turn_plan.get('response_mode', '接住')}\n"
                f"- 语气：{turn_plan.get('tone', '嘴硬里带一点软')}\n"
                f"- 主目标：{self._clean_state_text(turn_plan.get('goal'), '先接住对方 再保持点亲密感', limit=32)}\n"
                f"- 是否追问：{'是' if turn_plan.get('should_ask_question') else '否'}\n"
                f"- 是否逗他：{'是' if turn_plan.get('should_tease') else '否'}\n"
                f"- 是否主动帮忙：{'是' if turn_plan.get('should_offer_help') else '否'}\n"
                f"- 是否提旧事：{'是' if turn_plan.get('should_reference_memory') else '否'}\n"
                f"- 句数上限：{turn_plan.get('max_sentences', 2)}\n"
                f"- 本轮提醒：{self._clean_state_text(turn_plan.get('notes'), '别太像助手', limit=28)}\n"
            )
        prompt += "回复时让语气、主动性和留白程度与这些状态一致，不要分析自己，不要一口气把心里话全说透。"
        return prompt

    def _dynamic_state_summary_for_decision(self):
        state = self.dynamic_state or (self.db.get_dynamic_state(self.character_id) if self.character_id else None)
        if not state:
            return "暂无额外短期心智状态。"
        return (
            f"当前短期心智状态：场景={state.get('scene_label', '日常')}，情绪={state.get('emotion_label', '表面平静')} "
            f"(强度 {state.get('emotion_intensity', 0.38)})，用户状态={state.get('user_affect', '普通')}，"
            f"动机={state.get('motivation_label', '先接住他 再顺手要一点关注')}，"
            f"克制={state.get('inhibition_label', '不想显得太黏 也不想把话说满')}，"
            f"隐性期待={state.get('hidden_expectation', '希望他主动多哄一点 多顺着自己一点')}，"
            f"修复状态={state.get('repair_status', '无需修复')}。"
        )

    def _run_web_search_context(self, query: str) -> str:
        """B-1 编排：单独调用一次 Google Search grounding，再把结果注入当前轮上下文"""
        now_dt = datetime.datetime.now()
        current_time_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        current_date_str = now_dt.strftime("%Y-%m-%d")
        prompt = (
            "请基于 Google Search 的结果，给出简洁事实摘要。\n"
            "要求：\n"
            "1) 用中文回答；2) 只保留与用户问题直接相关的事实；\n"
            "3) 如果信息可能过时/冲突，要明确提示；4) 不要虚构来源；\n"
            "5) 你必须严格区分“今天数据”和“最新可得数据”。\n"
            f"6) 当前系统时间（Asia/Shanghai）是 {current_time_str}，今天日期是 {current_date_str}。\n"
            "7) 只有当来源数据日期等于今天日期时，才能写“今天”；否则必须写“最新可得日期是 XXXX-XX-XX（非今日）”。\n\n"
            f"用户问题：{query}"
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )

        summary = self._extract_response_text(response)
        if not summary:
            return ""

        sources = self._extract_grounding_sources(response, limit=6)
        if sources:
            source_lines = "\n".join([f"- {title}: {url}" for title, url in sources])
            return f"{summary}\n\n参考来源：\n{source_lines}"
        return summary

    def _extract_proactive_care_tasks_from_conversation(self, conversation_text: str):
        """从一段对话中批量提取后续可主动关怀的事项（每小时任务用）"""
        if not conversation_text.strip():
            return []
        now_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = (
            f"当前时间（Asia/Shanghai）: {now_dt}\n"
            "请从下面完整对话中提取值得后续主动关怀的事项，输出 JSON。\n"
            "仅输出 JSON："
            "{\"tasks\":[{\"event_at\":\"YYYY-MM-DD HH:MM:SS\",\"remind_at\":\"YYYY-MM-DD HH:MM:SS\",\"task_content\":\"...\"}]}\n"
            "规则：\n"
            "1) user 与 model 消息都可用于理解上下文，但提醒事项必须围绕用户真实需求，不要把纯角色撒娇当作任务。\n"
            "2) tasks 中每条都必须给 event_at 和 remind_at，且都是绝对时间 YYYY-MM-DD HH:MM:SS。\n"
            "3) remind_at 必须早于 event_at，目的是让用户提前准备；不能在事件发生后提醒。\n"
            "4) 若用户只说“明天”没具体时刻，可先合理推断 event_at（如明天中午或下午），但 remind_at 仍必须在其之前。\n"
            "5) task_content 要写成可执行提醒，不要写“询问是否需要建议”这类二次确认句。\n"
            "6) 最多输出 5 条；不确定就不输出。\n\n"
            f"对话内容：\n{conversation_text}"
        )
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    temperature=0.0
                )
            )
            raw_text = response.text.strip() if response and response.text else ""
            if not raw_text:
                return []
            data = json.loads(raw_text)
            if not isinstance(data, dict):
                return []
            tasks = data.get("tasks", [])
            normalized = []
            if isinstance(tasks, list):
                for task in tasks[:5]:
                    if not isinstance(task, dict):
                        continue
                    event_at_str = (task.get("event_at") or "").strip()
                    remind_at_str = (task.get("remind_at") or "").strip()
                    task_content = (task.get("task_content") or "").strip()
                    if not event_at_str or not remind_at_str or not task_content:
                        continue
                    normalized.append((event_at_str, remind_at_str, task_content))
            return normalized
        except Exception as e:
            logger.error(f"批量提取主动关怀任务失败: {e}")
            return []

    def schedule_contextual_care_from_recent_window(self, window_minutes=60):
        """整点任务：扫描最近窗口对话并安排主动关怀提醒"""
        if not self.character_id:
            return
        now = datetime.datetime.now()
        start = now - datetime.timedelta(minutes=window_minutes)
        recent_msgs = self.db.get_chat_history_between(
            self.character_id,
            start_time=start,
            end_time=now
        )
        if not recent_msgs:
            return

        lines = []
        for msg in recent_msgs:
            content = (msg.get('content') or '').strip()
            if not content:
                continue
            role = msg.get('role') or 'unknown'
            prefix = (msg.get('context_prefix') or '').strip()
            if prefix:
                lines.append(f"{prefix} {role}: {content}")
            else:
                lines.append(f"{role}: {content}")
        if not lines:
            return

        tasks = self._extract_proactive_care_tasks_from_conversation("\n".join(lines))
        for event_at_str, remind_at_str, task_content in tasks:
            event_at = None
            for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
                try:
                    event_at = datetime.datetime.strptime(event_at_str, fmt)
                    break
                except ValueError:
                    continue
            remind_at = None
            for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
                try:
                    remind_at = datetime.datetime.strptime(remind_at_str, fmt)
                    break
                except ValueError:
                    continue

            if not remind_at or remind_at <= now:
                continue
            if not event_at or remind_at >= event_at:
                continue

            window_start = remind_at - datetime.timedelta(hours=2)
            window_end = remind_at + datetime.timedelta(hours=2)
            if self.db.has_pending_similar_reminder(
                character_id=self.character_id,
                user_id=ALLOWED_USER_ID,
                task_content=task_content,
                start_time=window_start,
                end_time=window_end
            ):
                continue
            try:
                self.db.add_reminder(
                    self.character_id,
                    ALLOWED_USER_ID,
                    task_content,
                    remind_at,
                    source_type='hourly_auto_extract'
                )
            except Exception as e:
                logger.error(f"整点自动安排情境关怀提醒失败: {e}")
            
    def clear_history(self):
        """每天凌晨清空一次内存中的长对话列表，同时保留工具能力"""
        logger.info(f"正在清空角色 {self.character_id} 的短期对话感知缓冲...")
        old_cache_name = self.cached_content_name
        self.cached_content_name = None
        self.cached_prefix_history = []
        self.cache_pending_messages = 0
        if old_cache_name:
            self._delete_cached_content(old_cache_name)
        self.chat = self.client.chats.create(
            model=self.model,
            config=self._build_chat_config(),
            history=[] if self.cached_content_name else self.base_history
        )
    
    def send_message(self, message, image_data=None, image_mime_type=None, media_path=None, save_user_message=True):
        """发送消息并获取回复"""
        context_prefix = None
        proactive_image_forced = False
        relation_desc = ""
        compact_relation_desc = ""
        dynamic_state = self.dynamic_state
        pre_rule_notes = []
        
        # 处理时间连续性感知架构 (判断与上一条用户消息的时间差)
        if self.character_id and save_user_message:
            last_timestamp = self.last_message_timestamp
            now_dt = datetime.datetime.now()
            last_prefix_date = self.last_prefix_timestamp.date() if self.last_prefix_timestamp else None
            
            # 如果是第一次聊天，或者距离上次发言超过 30 分钟，或者距离上次打标超过 30 分钟（针对连续聊天），强制增加系统时间锚点
            if (not last_timestamp or 
                (now_dt - last_timestamp).total_seconds() > 1800 or 
                not self.last_prefix_timestamp or 
                (now_dt - self.last_prefix_timestamp).total_seconds() > 1800 or
                (last_prefix_date is not None and last_prefix_date != now_dt.date())):
                
                weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
                weekday_str = weekdays[now_dt.weekday()]
                current_time_str = now_dt.strftime(f"%Y年%m月%d日 {weekday_str} %H:%M:%S")
                context_prefix = f"[系统时间感知：当前时间 {current_time_str}]"
                
                # 更新上一次插入时间后缀的锚点
                self.last_prefix_timestamp = now_dt
                if self.cached_content_name:
                    self._rotate_cache_before_time_anchor()
            
            # 更新上次说话时间（用户侧）
            self.last_message_timestamp = now_dt
            self.last_user_message_timestamp = now_dt  # 只有用户说话才更新这个
        
        # 保存用户消息，同步落库纯净和前置标签（如果有）
        if self.character_id and save_user_message:
            self.db.save_message(
                self.character_id, 'user', message, 
                model=self.model, 
                context_prefix=context_prefix,
                media_path=media_path,
                media_type=image_mime_type
            )
            self.cache_pending_messages += 1
            # 用户回复后，解除“当天连续主动 5 条”限制并重置计数
            self.on_user_replied(now_dt)
        
        # 向量检索核心记忆 + 情景记忆
        core_facts_text = ""
        episodic_text = ""
        if self.character_id:
            try:
                # 获取消息的 embedding 用于检索
                embed_response = self.client.models.embed_content(
                    model='gemini-embedding-001',
                    contents=message
                )
                embedding = embed_response.embeddings[0].values
                
                # 调取最相关的3条人格特质
                core_facts = self.db.search_core_fact_memories(self.character_id, embedding, limit=3)
                if core_facts:
                    core_facts_text = "\n\n[系统附加背景记忆：根据长期互动，关于该用户的核心特质（供参考）]\n"
                    for i, fact in enumerate(core_facts):
                        # evidence = f" (依据: {fact['evidence_span']})" if fact.get('evidence_span') else ""
                        evidence = ""
                        core_facts_text += f"{i+1}. {fact['fact_text']}{evidence}\n"
                
                # 解析用户消息中的时间意图
                time_start, time_end = _parse_time_range(message)
                
                # 调取最相关的5条情景记忆（具体事件、时间线）
                episodic_memories = self.db.search_episodic_memories(
                    self.character_id, embedding, limit=5,
                    time_start=time_start, time_end=time_end
                )
                if episodic_memories:
                    episodic_text = "\n\n[系统附加情景回忆：过去发生的相关事件（供参考，可自然地融入回答）]\n"
                    for i, mem in enumerate(episodic_memories):
                        time_str = ""
                        if mem.get('event_time'):
                            time_str = f" [{mem['event_time'].strftime('%Y年%m月%d日 %H:%M')}]"
                        emo_str = f" (情绪: {mem.get('emotion_category', '未知')})" if mem.get('emotion_category') else ""
                        episodic_text += f"{i+1}.{time_str} {mem['content']}{emo_str}\n"

                # 调取深度关系建模描述（第三层：解释式注入）
                relation_desc = self.db.get_relationship_description(self.character_id)
                compact_relation_desc = self._build_compact_relationship_context()
                if compact_relation_desc:
                    episodic_text += f"\n\n{compact_relation_desc}"
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"检索记忆失败: {e}")

        if self.character_id:
            dynamic_state, pre_rule_notes = self._prepare_local_pre_reply_state(
                message,
                relation_desc=compact_relation_desc,
                proactive=False
            )
            try:
                self.db.upsert_dynamic_state(
                    self.character_id,
                    dynamic_state["scene_label"],
                    dynamic_state["emotion_label"],
                    dynamic_state["emotion_intensity"],
                    dynamic_state["motivation_label"],
                    dynamic_state["inhibition_label"],
                    dynamic_state["hidden_expectation"],
                    dynamic_state["last_user_intent"],
                    dynamic_state["user_affect"],
                    dynamic_state["unresolved_need"],
                    dynamic_state["carryover_summary"],
                    dynamic_state["reply_style"],
                    dynamic_state["warmth_bias"],
                    dynamic_state["initiative_bias"],
                    dynamic_state["last_trigger_source"],
                    dynamic_state["repair_status"]
                )
                self.dynamic_state = dynamic_state
                self._persist_dynamic_state_snapshot(
                    dynamic_state,
                    source_kind="pre_reply_local",
                    trigger_message=message,
                    notes="；".join(pre_rule_notes) if pre_rule_notes else "pre_reply_local_rules"
                )
            except Exception as e:
                logger.error(f"写入本地预修正状态失败: {e}")

        # 拼接最终提交给 LLM 的增强上下文：
        # 1. 临时系统时间锚点 (决定了流逝感)
        # 2. 原始消息
        # 3. 跨期长记忆碎片 (决定了记忆留存)
        augmented_message = ""
        if context_prefix:
            augmented_message += f"{context_prefix}\n"
        augmented_message += message
        if core_facts_text:
            augmented_message += core_facts_text
        if episodic_text:
            augmented_message += episodic_text
        if dynamic_state:
            augmented_message += self._build_dynamic_state_prompt(dynamic_state, turn_plan=None)

        # 轻触发图片偏好：命中视觉化意图时，优先让模型调用图片工具
        if self._should_prefer_image_response(message, image_data=image_data):
            augmented_message += (
                "\n\n[系统附加本轮回复偏好]\n"
                "用户本轮是视觉化诉求。请优先调用 generate_image 生成图片，并将文字回复控制在 3 句以内。"
                "如果用户要求的是你/他的外观、自拍或场景展示，务必先出图，再补一句自然说明。"
            )

        # 与用户措辞无关的主动发图策略：按概率触发，让角色在日常聊天中主动分享图片
        if self._should_trigger_proactive_image(message, image_data=image_data):
            proactive_image_forced = True
            augmented_message += (
                "\n\n[系统附加主动分享策略]\n"
                "本轮请你主动分享一张图片，即使用户没有要求图片。"
                "你需要先调用 generate_image，内容选择你此刻最自然想分享的生活片段（如正在做的事、当下状态、想给对方看的瞬间）。"
                "文字只保留 3 句，语气自然，不要提及系统或工具。"
            )

        # B-1：按需联网搜索（独立请求），再把摘要注入当前轮上下文
        web_search_text = ""
        if self._should_use_web_search(message):
            try:
                web_search_text = self._run_web_search_context(message)
            except Exception as e:
                logger.error(f"联网搜索失败: {e}")
        if web_search_text:
            augmented_message += (
                "\n\n[系统附加联网信息：以下是本轮联网检索摘要，请优先使用有来源的事实作答]\n"
                f"{web_search_text}"
            )
        augmented_message += self._build_single_call_state_patch_instruction()

        # 重置待处理图片
        self.pending_output_image = None

        # 构造多模态消息 Parts
        message_parts = [augmented_message]
        if image_data:
            message_parts.append(types.Part.from_bytes(data=image_data, mime_type=image_mime_type))

        # 获取AI回复（automatic_function_calling 在 send_message 内部自动完成，response 是最终响应）
        if self.cached_content_name:
            response = self._send_message_with_manual_tools(message_parts, "send_message")
        else:
            response = self.chat.send_message(message_parts)
            self._log_cache_usage(response, "send_message")
        response_text = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text and not part.text.strip().startswith(("THOUGHT", "THINK")):
                    response_text += part.text
        # 清理模型生成图片时附带的内部标记文本
        response_text = response_text.replace("Here is the original image:", "")
        # 清理 AI 可能模仿输出的系统标记（它从 chat history 里学到的）
        response_text = re.sub(r'\[\u7cfb\u7edf\u65f6\u95f4\u611f\u77e5[^\]]*\]', '', response_text)
        response_text = re.sub(r'\[SYS_PROACTIVE[^\]]*\]', '', response_text)
        response_text = re.sub(r'\[\u7cfb\u7edf\u9644\u52a0[^\]]*\]', '', response_text)
        response_text = response_text.strip()

        payload = self._extract_json_payload(response_text)
        state_patch = payload.get("state_patch", {}) if isinstance(payload, dict) else {}
        user_reply_text = ""
        if isinstance(payload, dict):
            reply_candidate = payload.get("reply")
            if isinstance(reply_candidate, str):
                user_reply_text = reply_candidate.strip()
        # 仅当解析失败（payload 不是 dict）时才回退原文，避免把 JSON 透传给用户
        if not user_reply_text and not isinstance(payload, dict):
            user_reply_text = response_text
        
        # 从 chat history 中清除系统注入块，避免 token 累积
        # automatic_function_calling 会在末尾追加 role='user' 的 function_response，
        # 因此不能只清理“最后一条 user”，要扫描所有 user 文本 part。
        try:
            markers = ['[系统附加', '[Deep Relationship', '[系统时间感知']
            for content in getattr(self.chat, '_curated_history', []):
                if getattr(content, 'role', None) != 'user':
                    continue
                for part in getattr(content, 'parts', []):
                    text = getattr(part, 'text', None)
                    if not text:
                        continue
                    cut_points = [text.find(tag) for tag in markers if tag in text]
                    if cut_points:
                        part.text = text[:min(cut_points)].rstrip()
        except Exception:
            pass
        
        # 提取生成图片的路径（如果有的话）
        image_path = self.pending_output_image
        if proactive_image_forced and image_path:
            self.last_proactive_image_timestamp = datetime.datetime.now()

        merged_state = dynamic_state
        if self.character_id and dynamic_state:
            try:
                merged_state = self._merge_state_patch(dynamic_state, state_patch)
                self.db.upsert_dynamic_state(
                    self.character_id,
                    merged_state["scene_label"],
                    merged_state["emotion_label"],
                    merged_state["emotion_intensity"],
                    merged_state["motivation_label"],
                    merged_state["inhibition_label"],
                    merged_state["hidden_expectation"],
                    merged_state["last_user_intent"],
                    merged_state["user_affect"],
                    merged_state["unresolved_need"],
                    merged_state["carryover_summary"],
                    merged_state["reply_style"],
                    merged_state["warmth_bias"],
                    merged_state["initiative_bias"],
                    merged_state["last_trigger_source"],
                    merged_state["repair_status"]
                )
                self.dynamic_state = merged_state
                self._persist_dynamic_state_snapshot(
                    merged_state,
                    source_kind="pre_reply_patch_merge",
                    trigger_message=message,
                    model_reply=user_reply_text,
                    notes="single_call_state_patch_merge"
                )
            except Exception as e:
                logger.error(f"合并状态补丁失败: {e}")
        
        # 保存AI回复（只有非空时才保存）
        if self.character_id and (user_reply_text.strip() or image_path):
            save_text = user_reply_text if user_reply_text.strip() else "[AI发送了一张图片]"
            self.db.save_message(
                self.character_id, 'model', save_text, 
                model=self.model,
                media_path=image_path,
                media_type='image/jpeg' if image_path else None
            )
            self.cache_pending_messages += 1
            # 同样更新内存中的最后互动时间，确保 30 分钟判断更精准
            self.last_message_timestamp = datetime.datetime.now()
            self._update_post_reply_state(
                trigger_message=message,
                model_reply=save_text,
                base_state=merged_state,
                turn_plan=None,
                proactive=False
            )
        
        return user_reply_text, image_path

    def send_proactive_message(self, trigger_hint: str):
        """AI 主动发起一条消息（非用户触发）。
        trigger_hint 描述当前情境背景（沉默时长等），注入给 AI 作为发消息的出发点。
        只落库 model 侧消息，不写假用户消息。
        """
        now_dt = datetime.datetime.now()
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        weekday_str = weekdays[now_dt.weekday()]
        current_time_str = now_dt.strftime(f"%Y年%m月%d日 {weekday_str} %H:%M:%S")

        # 触发提示：以特殊标记开头，便于事后从 chat history 中识别并移除
        current_state = self.dynamic_state or (self.db.get_dynamic_state(self.character_id) if self.character_id else None)
        current_state, proactive_plan = self._infer_state_and_plan(
            trigger_hint,
            relation_desc=self._build_compact_relationship_context(),
            image_requested=False,
            proactive=True
        )
        dynamic_state_prompt = self._build_dynamic_state_prompt(
            current_state,
            turn_plan=proactive_plan
        )
        trigger_msg = (
            f"[SYS_PROACTIVE] 现在是 {current_time_str}，{trigger_hint}"
            "请你以角色身份，自然地主动给思远发一条消息。"
            "内容简短自然，符合你的性格，不要提及这是系统触发或你在'自言自语'。"
            "直接输出你想说的话即可。"
        )
        if self._should_add_wakeup_prompt():
            trigger_msg += (
                "\n[系统补充]\n"
                "现在是上午，请你以角色身份叫他起床，"
            )
        if dynamic_state_prompt:
            trigger_msg += dynamic_state_prompt
        # 统一主动消息的输出协议，便于解析 reply/state_patch
        trigger_msg += self._build_single_call_state_patch_instruction()

        self.pending_output_image = None

        if self.cached_content_name:
            response = self._send_message_with_manual_tools([trigger_msg], "send_proactive_message")
        else:
            response = self.chat.send_message([trigger_msg])
            self._log_cache_usage(response, "send_proactive_message")

        response_text = ""
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text and not part.text.strip().startswith(("THOUGHT", "THINK")):
                    response_text += part.text

        # 清理 AI 可能模仿输出的系统标记
        response_text = re.sub(r'\[\u7cfb\u7edf\u65f6\u95f4\u611f\u77e5[^\]]*\]', '', response_text)
        response_text = re.sub(r'\[SYS_PROACTIVE[^\]]*\]', '', response_text)
        response_text = re.sub(r'\[\u7cfb\u7edf\u9644\u52a0[^\]]*\]', '', response_text)
        response_text = response_text.strip()
        
        payload = self._extract_json_payload(response_text)
        state_patch = payload.get("state_patch", {}) if isinstance(payload, dict) else {}
        user_reply_text = ""
        if isinstance(payload, dict):
            reply_candidate = payload.get("reply")
            if isinstance(reply_candidate, str):
                user_reply_text = reply_candidate.strip()
        # 仅当解析失败（payload 不是 dict）时才回退原文，避免把 JSON 透传给用户
        if not user_reply_text and not isinstance(payload, dict):
            user_reply_text = response_text

        # 从 chat history 中移除注入的假"用户"触发消息，保持历史干净
        # automatic_function_calling 可能在末尾追加 function_response(user) 项，
        # 不能仅检查最后一条 user。
        try:
            history = self.chat._curated_history
            for i in range(len(history) - 1, -1, -1):
                if history[i].role == 'user':
                    is_proactive = any(
                        hasattr(p, 'text') and p.text and '[SYS_PROACTIVE]' in p.text
                        for p in history[i].parts
                    )
                    if is_proactive:
                        del history[i]
        except Exception:
            pass

        image_path = self.pending_output_image

        merged_state = current_state
        if self.character_id and current_state:
            try:
                merged_state = self._merge_state_patch(current_state, state_patch)
                self.db.upsert_dynamic_state(
                    self.character_id,
                    merged_state["scene_label"],
                    merged_state["emotion_label"],
                    merged_state["emotion_intensity"],
                    merged_state["motivation_label"],
                    merged_state["inhibition_label"],
                    merged_state["hidden_expectation"],
                    merged_state["last_user_intent"],
                    merged_state["user_affect"],
                    merged_state["unresolved_need"],
                    merged_state["carryover_summary"],
                    merged_state["reply_style"],
                    merged_state["warmth_bias"],
                    merged_state["initiative_bias"],
                    merged_state["last_trigger_source"],
                    merged_state["repair_status"]
                )
                self.dynamic_state = merged_state
                self._persist_dynamic_state_snapshot(
                    merged_state,
                    source_kind="pre_proactive_patch_merge",
                    trigger_message=trigger_hint,
                    model_reply=user_reply_text,
                    notes="single_call_state_patch_merge"
                )
            except Exception as e:
                logger.error(f"合并主动消息状态补丁失败: {e}")
        
        # 构造落库用的时间感知前缀 (遵循 30 分钟规则，避免连发时标签堆叠)
        context_prefix = None
        last_prefix_date = self.last_prefix_timestamp.date() if self.last_prefix_timestamp else None
        if (not self.last_message_timestamp or 
            (now_dt - self.last_message_timestamp).total_seconds() > 1800 or 
            not self.last_prefix_timestamp or 
            (now_dt - self.last_prefix_timestamp).total_seconds() > 1800 or
            (last_prefix_date is not None and last_prefix_date != now_dt.date())):
            
            context_prefix = f"[系统时间感知：当前时间 {current_time_str}]"
            self.last_prefix_timestamp = now_dt

        # 只落库 model 侧消息
        if self.character_id and (user_reply_text.strip() or image_path):
            self.db.save_message(
                self.character_id, 'model', user_reply_text,
                model=self.model,
                context_prefix=context_prefix,
                media_path=image_path,
                media_type='image/jpeg' if image_path else None
            )
            self.cache_pending_messages += 1
            self.last_message_timestamp = datetime.datetime.now()
            self._update_post_reply_state(
                trigger_message=trigger_hint,
                model_reply=user_reply_text if user_reply_text.strip() else "[AI发送了一张图片]",
                base_state=merged_state,
                turn_plan=proactive_plan,
                proactive=True
            )

        return user_reply_text, image_path

# 全局变量存储每个用户的聊天实例
user_chats = {}
user_model_prefs = {}
dice_sessions = {}
DICE_SESSION_TTL_SECONDS = 120
ALLOWED_USER_ID = 569020802

async def check_permission(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """检查用户权限"""
    user_id = update.effective_user.id
    
    if user_id != ALLOWED_USER_ID:
        user_name = update.effective_user.full_name or update.effective_user.username or "Unknown"
        message_text = update.message.text if update.message else "Unknown"
        
        try:
            await context.bot.send_message(
                chat_id=ALLOWED_USER_ID,
                text=f"未授权用户尝试访问：\n用户名: {user_name}\nID: {user_id}\n消息: {message_text}"
            )
        except:
            pass
        
        await update.message.reply_text("I don't know.")
        
        # 停止后续处理器执行
        from telegram.ext import ApplicationHandlerStop
        raise ApplicationHandlerStop
    
    return True

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /start 命令"""
    user_id = update.effective_user.id
    
    db = Database()
    characters = db.list_characters()
    
    message = "欢迎使用AI聊天机器人！\n\n可用角色：\n"
    for char in characters:
        message += f"{char['id']}. {char['name']}\n"
    
    message += "\n使用 /select <角色ID> 选择角色"
    
    await update.message.reply_text(message)
    db.close()

async def select_character(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /select 命令"""
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text("请提供角色ID，例如：/select 1")
        return
    
    try:
        character_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("角色ID必须是数字")
        return
    
    db = Database()
    character = db.get_character(character_id)
    
    if not character:
        await update.message.reply_text("角色不存在")
        db.close()
        return
    
    # 创建新的聊天实例
    try:
        user_chats[user_id] = ChatAI(
            model=user_model_prefs.get(user_id, DEFAULT_CHAT_MODEL),
            system_instruction=character['system_instruction'],
            character_id=character_id
        )
        await update.message.reply_text(f"已选择角色：{character['name']}\n现在可以开始聊天了！")
    except Exception as e:
        logger.error(f"创建聊天实例失败: {e}")
        await update.message.reply_text(f"创建聊天失败：{str(e)}")
    
    db.close()

async def use_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /useModel 命令，切换对话模型"""
    user_id = update.effective_user.id

    if not context.args:
        current = user_model_prefs.get(user_id, DEFAULT_CHAT_MODEL)
        if user_id in user_chats:
            current = user_chats[user_id].model
        models_text = "\n".join(SUPPORTED_CHAT_MODELS)
        await update.message.reply_text(
            f"当前模型：{current}\n可用模型：\n{models_text}\n用法：/useModel <模型名>"
        )
        return

    model = context.args[0].strip()
    if model not in SUPPORTED_CHAT_MODELS:
        models_text = "\n".join(SUPPORTED_CHAT_MODELS)
        await update.message.reply_text(f"不支持的模型：{model}\n可用模型：\n{models_text}")
        return

    user_model_prefs[user_id] = model
    if user_id in user_chats:
        user_chats[user_id].switch_model(model)
    await update.message.reply_text(f"已切换模型：{model}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理用户消息"""
    import asyncio
    now_ts = time.time()

    if not update.message:
        return
    
    async def keep_typing(chat_id, duration):
        """持续发送 typing 状态"""
        end_time = asyncio.get_event_loop().time() + duration
        while asyncio.get_event_loop().time() < end_time:
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                await asyncio.sleep(4)  # 每4秒刷新一次
            except asyncio.CancelledError:
                break
            except Exception:
                # typing 状态失败不影响主流程
                break
    
    user_id = update.effective_user.id
    user_message = update.message.text or update.message.caption or ""
    
    image_data = None
    image_mime_type = None
    media_path = None
    
    # 如果有图片，处理图片
    if update.message.photo:
        # 获取最高分辨率的图片
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        
        # 生成本地保存路径
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"user_{user_id}_{timestamp}.jpg"
        media_path = os.path.join(MEDIA_DIR, filename)
        
        # 下载图片
        image_bytes = await file.download_as_bytearray()
        with open(media_path, 'wb') as f:
            f.write(image_bytes)
            
        image_data = bytes(image_bytes)
        image_mime_type = "image/jpeg"
        
        if not user_message:
            user_message = "[用户发送了一张图片]"

    # 检查用户是否已选择角色
    if user_id not in user_chats:
        await update.message.reply_text("请先使用 /start 和 /select 选择角色")
        return

    # 清理过期骰子会话
    stale_ids = [
        uid for uid, sess in dice_sessions.items()
        if isinstance(sess, dict) and sess.get("created_at") and now_ts - sess["created_at"] > DICE_SESSION_TTL_SECONDS
    ]
    for uid in stale_ids:
        dice_sessions.pop(uid, None)

    chat_ai = user_chats[user_id]

    def save_user_message_to_db(text: str):
        if not chat_ai.character_id:
            return
        now_dt = datetime.datetime.now()
        context_prefix = None
        last_prefix_date = chat_ai.last_prefix_timestamp.date() if chat_ai.last_prefix_timestamp else None
        if (
            not chat_ai.last_message_timestamp
            or (now_dt - chat_ai.last_message_timestamp).total_seconds() > 1800
            or not chat_ai.last_prefix_timestamp
            or (now_dt - chat_ai.last_prefix_timestamp).total_seconds() > 1800
            or (last_prefix_date is not None and last_prefix_date != now_dt.date())
        ):
            weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
            weekday_str = weekdays[now_dt.weekday()]
            current_time_str = now_dt.strftime(f"%Y年%m月%d日 {weekday_str} %H:%M:%S")
            context_prefix = f"[系统时间感知：当前时间 {current_time_str}]"
            chat_ai.last_prefix_timestamp = now_dt
            if chat_ai.cached_content_name:
                chat_ai._rotate_cache_before_time_anchor()
        chat_ai.db.save_message(
            chat_ai.character_id,
            'user',
            text,
            model=chat_ai.model,
            context_prefix=context_prefix
        )
        chat_ai.cache_pending_messages += 1
        chat_ai.last_message_timestamp = now_dt
        chat_ai.last_user_message_timestamp = now_dt
        chat_ai.on_user_replied(now_dt)

    async def send_ai_reply(text: str):
        messages = text.split('\n')
        char_delay = 60.0 / 240
        for msg in messages:
            if msg.strip():
                delay = len(msg.strip()) * char_delay
                typing_task = asyncio.create_task(keep_typing(update.effective_chat.id, delay))
                await asyncio.sleep(delay)
                typing_task.cancel()
                await _send_with_retry(
                    lambda: update.message.reply_text(msg.strip()),
                    label="reply_text"
                )

    # 用户发送骰子（🎲）直接触发：用户先扔
    if update.message.dice:
        user_val = update.message.dice.value

        session = dice_sessions.get(user_id) or {}
        ai_val = session.get("ai_val")
        if ai_val is not None:
            # AI 已经扔过（你先），这是第二次：直接宣布胜负
            prompt2 = f"[系统骰子结果] 你={ai_val}，对方={user_val}。请用一句话宣布胜负。"
            reply2, _ = chat_ai.send_message(prompt2, save_user_message=False)
            if reply2.strip():
                await send_ai_reply(reply2)
            dice_sessions.pop(user_id, None)
            return

        # 用户先扔：先评论，再让 AI 扔并宣布胜负
        prompt1 = f"[系统骰子结果] 对方刚刚掷出点数={user_val}。请用一句话自然评论这次结果。"
        reply1, _ = chat_ai.send_message(prompt1, save_user_message=False)
        if reply1.strip():
            await send_ai_reply(reply1)

        msg_ai = await _send_with_retry(
            lambda: context.bot.send_dice(chat_id=update.effective_chat.id),
            label="send_dice_ai"
        )
        if not msg_ai or not getattr(msg_ai, "dice", None):
            await update.message.reply_text("掷骰子失败了，稍后再试。")
            return
        ai_val = msg_ai.dice.value

        prompt2 = f"[系统骰子结果] 对方={user_val}，你={ai_val}。请用一句话宣布胜负。"
        reply2, _ = chat_ai.send_message(prompt2, save_user_message=False)
        if reply2.strip():
            await send_ai_reply(reply2)
        dice_sessions.pop(user_id, None)
        return

    # 文字指令：仅支持“扔骰子，你先 / 我先”
    if "扔骰子" in user_message and ("我先" in user_message or "你先" in user_message):
        save_user_message_to_db(user_message)
        order = "user_first" if "我先" in user_message else "ai_first"

        if order == "user_first":
            # 等待用户发送 🎲，不提示
            dice_sessions[user_id] = {"ai_val": None, "created_at": now_ts}
            return

        # AI 先扔：先评论，等待用户发送 🎲
        msg_ai = await _send_with_retry(
            lambda: context.bot.send_dice(chat_id=update.effective_chat.id),
            label="send_dice_ai"
        )
        if not msg_ai or not getattr(msg_ai, "dice", None):
            await update.message.reply_text("掷骰子失败了，稍后再试。")
            return
        ai_val = msg_ai.dice.value

        prompt1 = f"[系统骰子结果] 你刚刚掷出点数={ai_val}。请用一句话自然评论这次结果。"
        reply1, _ = chat_ai.send_message(prompt1, save_user_message=False)
        if reply1.strip():
            await send_ai_reply(reply1)

        dice_sessions[user_id] = {"ai_val": ai_val, "created_at": now_ts}
        return
    
    try:
        # 获取AI回复
        response, ai_image_path = chat_ai.send_message(
            user_message, 
            image_data=image_data, 
            image_mime_type=image_mime_type,
            media_path=media_path
        )
        
        # 如果 AI 生成了图片，先发送图片
        if ai_image_path and os.path.exists(ai_image_path):
            await _send_with_retry(
                lambda: context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=open(ai_image_path, 'rb')
                ),
                label="send_photo"
            )
        
        # 按换行符拆分消息
        messages = response.split('\n')
        
        # 每分钟240个汉字
        char_delay = 60.0 / 240
        
        # 分别发送每条消息，模拟输入延迟
        for msg in messages:
            if msg.strip():
                # 计算这条消息的延迟时间
                delay = len(msg.strip()) * char_delay
                
                # 在延迟期间持续显示正在输入
                typing_task = asyncio.create_task(keep_typing(update.effective_chat.id, delay))
                await asyncio.sleep(delay)
                typing_task.cancel()
                
                await _send_with_retry(
                    lambda: update.message.reply_text(msg.strip()),
                    label="reply_text"
                )
    except Exception as e:
        logger.error(f"处理消息出错: {e}")
        await update.message.reply_text(f"错误：{str(e)}")

async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /history 命令"""
    user_id = update.effective_user.id
    
    if user_id not in user_chats:
        await update.message.reply_text("请先使用 /start 和 /select 选择角色")
        return
    
    chat_ai = user_chats[user_id]
    
    if not chat_ai.character_id:
        await update.message.reply_text("未关联角色ID")
        return
    
    messages = chat_ai.db.get_chat_history(chat_ai.character_id, limit=10)
    
    if not messages:
        await update.message.reply_text("暂无历史记录")
        return
    
    history_text = "最近10条对话记录：\n\n"
    for msg in messages:
        history_text += f"{msg['role']}: {msg['content'][:50]}...\n"
    
    await update.message.reply_text(history_text)

async def delete_from_last_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /del 命令：全表删除从最后一条 user 消息开始（含该条）的所有记录，并清理关联文件。"""
    db_for_del = Database()

    try:
        result = db_for_del.delete_messages_from_last_user()
    except Exception as e:
        logger.error(f"/del 删除数据库记录失败: {e}")
        await update.message.reply_text(f"删除失败：{str(e)}")
        db_for_del.close()
        return
    finally:
        try:
            db_for_del.close()
        except Exception:
            pass

    last_user_id = result.get("last_user_id")
    deleted_count = int(result.get("deleted_count") or 0)
    media_paths = list(result.get("media_paths") or [])

    if last_user_id is None:
        await update.message.reply_text("未找到 user 消息，未执行删除。")
        return

    removed_files = 0
    remove_errors = 0
    seen_paths = set()
    for path in media_paths:
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            if os.path.exists(path):
                os.remove(path)
                removed_files += 1
        except Exception as e:
            logger.error(f"/del 删除媒体文件失败: path={path}, error={e}")
            remove_errors += 1

    # 重建活跃聊天实例，避免内存中的历史与数据库删除结果不一致
    if user_chats:
        try:
            db = Database()
            for uid, active_chat in list(user_chats.items()):
                character_id = getattr(active_chat, "character_id", None)
                if not character_id:
                    continue
                character = db.get_character(character_id)
                if not character:
                    continue
                try:
                    active_chat.db.close()
                except Exception:
                    pass
                user_chats[uid] = ChatAI(
                    system_instruction=character['system_instruction'],
                    character_id=character_id
                )
            db.close()
        except Exception as e:
            logger.error(f"/del 重建聊天实例失败: {e}")

    msg = (
        f"已执行删除：从最后一条 user 消息 id={last_user_id} 开始，"
        f"共删除 {deleted_count} 条记录。"
    )
    if media_paths:
        msg += f"\n已删除媒体文件 {removed_files} 个"
        if remove_errors:
            msg += f"（失败 {remove_errors} 个）"
        msg += "。"
    await update.message.reply_text(msg)

async def trigger_filter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /filter 命令，手动触发 filter_task"""
    import asyncio
    await update.message.reply_text("已触发过滤任务 (filter_task)，正在后台执行...")
    async def run_task():
        try:
            from memory_worker import MemoryWorker
            worker = MemoryWorker()
            await worker.filter_task()
            await context.bot.send_message(chat_id=update.effective_chat.id, text="过滤任务 (filter_task) 执行完成！")
        except Exception as e:
            logger.error(f"手动触发过滤任务失败: {e}")
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"过滤任务执行失败: {str(e)}")
    asyncio.create_task(run_task())

async def trigger_consolidate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /consolidate 命令，手动触发 consolidate_task"""
    import asyncio
    await update.message.reply_text("已触发巩固任务 (consolidate_task)，正在后台执行...")
    async def run_task():
        try:
            from memory_worker import MemoryWorker
            worker = MemoryWorker()
            await worker.consolidate_task()
            await context.bot.send_message(chat_id=update.effective_chat.id, text="巩固任务 (consolidate_task) 执行完成！")
        except Exception as e:
            logger.error(f"手动触发巩固任务失败: {e}")
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"巩固任务执行失败: {str(e)}")
    asyncio.create_task(run_task())

async def trigger_hourly_care_extract(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /care_extract 命令，手动触发整点关怀提取逻辑"""
    import asyncio
    hours = 1.0
    if context.args:
        try:
            hours = float(context.args[0])
            if hours <= 0:
                raise ValueError
        except ValueError:
            await update.message.reply_text("参数格式错误，请使用正数小时，例如：/care_extract 2 或 /care_extract 1.5")
            return
    window_minutes = hours * 60

    await update.message.reply_text(f"已触发整点关怀提取任务 (hourly_contextual_care)，窗口最近 {hours:g} 小时，正在后台执行...")

    async def run_task():
        try:
            if not user_chats:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="当前没有活跃聊天实例，无需提取。")
                return

            processed = 0
            for _, chat_ai in list(user_chats.items()):
                try:
                    chat_ai.schedule_contextual_care_from_recent_window(window_minutes=window_minutes)
                    processed += 1
                except Exception as e:
                    logger.error(f"手动触发整点关怀提取失败(单实例): {e}")
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"整点关怀提取任务执行完成（最近 {hours:g} 小时），已处理 {processed} 个聊天实例。"
            )
        except Exception as e:
            logger.error(f"手动触发整点关怀提取失败: {e}")
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"整点关怀提取任务执行失败: {str(e)}")
    asyncio.create_task(run_task())

async def trigger_rebuild_cache(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /rebuild_cache 命令：立即重建缓存"""
    if not user_chats:
        await update.message.reply_text("当前没有活跃聊天实例，无法重建缓存。")
        return
    processed = 0
    failed = 0
    for _, chat_ai in list(user_chats.items()):
        try:
            ok, msg = chat_ai.rebuild_cache_now()
            if ok:
                processed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"/rebuild_cache 失败: {e}")
            failed += 1
    await update.message.reply_text(f"缓存重建完成：成功 {processed}，失败 {failed}")

async def trigger_drop_cache(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /drop_cache 命令：立即删除缓存"""
    if not user_chats:
        await update.message.reply_text("当前没有活跃聊天实例，无法删除缓存。")
        return
    processed = 0
    failed = 0
    for _, chat_ai in list(user_chats.items()):
        try:
            ok, msg = chat_ai.drop_cache_now()
            if ok:
                processed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"/drop_cache 失败: {e}")
            failed += 1
    await update.message.reply_text(f"缓存删除完成：成功 {processed}，失败 {failed}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /help 命令，显示所有可用命令"""
    help_text = (
        "🤖 <b>可用命令列表</b>\n\n"
        "/start - 显示所有可用角色列表\n"
        "/select &lt;角色ID&gt; - 选择一个角色开始聊天\n"
        "/useModel &lt;模型名&gt; - 切换对话模型\n"
        "/history - 查看与当前角色的最近 10 条历史记录\n"
        "/del - 全表删除从最后一条 user 消息开始（含该条）的所有聊天记录，并清理关联媒体文件\n"
        "/filter - 手动触发抽取情景记忆任务 (后台运行)\n"
        "/consolidate - 手动触发巩固核心人格任务 (后台运行)\n"
        "/care_extract [小时] - 手动触发整点关怀提取任务，默认最近 1 小时 (后台运行)\n"
        "/rebuild_cache - 立即重建缓存（后台实例全部重建）\n"
        "/drop_cache - 立即删除缓存（后台实例全部删除）\n"
        "/help - 显示此帮助信息"
    )
    await update.message.reply_text(help_text, parse_mode='HTML')


def evaluate_proactive_intent(chat_ai, user_silence_seconds: float, total_silence_seconds: float):
    """独立调用 Gemini 评估 AI 此刻是否想主动联系用户。
    - user_silence_seconds: 距离用户最后一次发言的时长
    - total_silence_seconds: 距离最后一次对话（含 AI 主动发言）的时长
    """
    now_dt = datetime.datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday_str = weekdays[now_dt.weekday()]
    current_time_str = now_dt.strftime(f"%Y年%m月%d日 {weekday_str} %H:%M")

    # 用户沉默描述
    u_silence_hours = user_silence_seconds / 3600
    if u_silence_hours < 4:
        user_desc = f"思远已经有 {u_silence_hours:.1f} 小时没理你了"
    else:
        user_desc = f"思远今天已经很长时间（{u_silence_hours:.1f} 小时）没和你说话了"

    # 总沉默描述（控制频率）
    if total_silence_seconds < 3600:
        total_desc = f"而你大约 {int(total_silence_seconds / 60)} 分钟前刚找过他"
    else:
        total_desc = f"距离你们最后一次互动也过了 {total_silence_seconds / 3600:.1f} 小时了"

    # 以用户最近 4 句里最早那条为时间锚点，拉出从那时到现在的完整对话链
    # 这样即使 AI 连续主动发了很多条，用户声音和完整对话上下文也不会丢失
    recent_context = ""
    try:
        context_msgs = chat_ai.db.get_context_since_nth_user_message(chat_ai.character_id, user_msg_count=4)
        if context_msgs:
            recent_context = "\n最近的对话片段：\n"
            for msg in context_msgs:
                role_name = "思远" if msg['role'] == 'user' else "你"
                time_str = msg['timestamp'].strftime('%H:%M') if msg.get('timestamp') else ''
                recent_context += f"  [{time_str}] {role_name}: {msg['content'][:80]}\n"
    except Exception:
        pass

    prompt = (
        f"现在是 {current_time_str}。\n"
        f"{user_desc}，{total_desc}。\n"
        f"{chat_ai._dynamic_state_summary_for_decision()}\n"
        f"{recent_context}\n"
        f"请思考：此刻你是否想主动给思远发一条消息？\n"
        f"考虑因素：当前时间是否合适、思远的状态、你之前的发言频率、是否有新的话题。\n"
        f"注意：即使他没回你，你也可以根据你的性格（粘人、关心或随意）决定是否继续找他，但不要让他感到烦躁。\n"
        f"只回答 YES 或 NO，不要输出任何其他内容。"
    )

    try:
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            system_instruction=chat_ai.system_instruction,
            temperature=1.2  # 略高温度，让决策更有随机性（有时想发有时不想）
        )
        response = client.models.generate_content(
            model=chat_ai.model,
            contents=prompt,
            config=config
        )
        answer_text = ""
        if response and getattr(response, "text", None):
            answer_text = response.text
        elif response and getattr(response, "candidates", None):
            try:
                first = response.candidates[0]
                parts = first.content.parts if first and first.content and first.content.parts else []
                answer_text = "".join([
                    p.text for p in parts
                    if getattr(p, "text", None)
                ])
            except Exception:
                answer_text = ""

        answer = answer_text.strip().upper()
        should_send = answer.startswith("YES")
        trigger_hint = f"{user_desc}，{total_desc}。"
        return should_send, trigger_hint
    except Exception as e:
        logger.error(f"意愿评估失败: {e}")
        return False, ""


async def reminder_job(context: ContextTypes.DEFAULT_TYPE):
    """每分钟扫描一次提醒任务，并触发提醒消息"""
    import asyncio
    db = Database()
    try:
        due_reminders = db.get_due_reminders()
        for idx, rem in enumerate(due_reminders):
            uid = int(rem['user_id'])
            chat_ai = user_chats.get(uid)
            if not chat_ai:
                continue
            
            # 使用现有流程发送主动消息
            reminder_hint = (
                f"[系统重要提醒] 现在时间到了！你之前在对话中亲口答应过思远，要在此时提醒他这件事：'{rem['task_content']}'。"
                f"请立即用你目前的角色语气，自然地对他发起提醒。不要生硬，要符合你们互动的氛围。直接说出你想提醒的话，不要提到这是‘系统任务’。"
            )
            
            response_text, image_path = chat_ai.send_proactive_message(reminder_hint)
            
            if not response_text.strip() and not image_path:
                continue

            # 发送逻辑
            if image_path and os.path.exists(image_path):
                await context.bot.send_photo(chat_id=uid, photo=open(image_path, 'rb'))

            if response_text.strip():
                lines = response_text.split('\n')
                char_delay = 60.0 / 120.0
                for line in lines:
                    if line.strip():
                        await context.bot.send_chat_action(chat_id=uid, action="typing")
                        # 模拟打字延迟
                        await asyncio.sleep(min(len(line.strip()) * char_delay, 5))
                        await context.bot.send_message(chat_id=uid, text=line.strip())
            
            # 标记为已发送
            db.mark_reminder_sent(rem['id'])

            # 避免连续触发导致速率限制：两条提醒之间间隔 10 秒
            if idx < len(due_reminders) - 1:
                await asyncio.sleep(10)
    except Exception as e:
        logger.error(f"reminder_job 出错: {e}")
    finally:
        db.close()

next_proactive_check_time = None

def _schedule_next_proactive_check():
    """安排下一次主动意愿检查，间隔随机 600~1500 秒（10~25 分钟），保证无规律感。"""
    global next_proactive_check_time
    import random
    import datetime
    next_interval = random.randint(600, 1500)
    next_proactive_check_time = datetime.datetime.now() + datetime.timedelta(seconds=next_interval)


async def proactive_check_job(context: ContextTypes.DEFAULT_TYPE):
    """AI 自主意愿检查：evaluate → generate → send，完成后动态安排下一次检查。
    - 使用 run_repeating 稳健调度，根据 next_proactive_check_time 触发
    - 夜间（北京时间 0:00~8:00）静默，不发消息
    - 沉默不足 30 分钟不打扰
    - AI 自己决定要不要发，内容也是 AI 自己生成
    - 打字延迟模拟与普通消息一致
    """
    global next_proactive_check_time
    import asyncio
    
    now_dt = datetime.datetime.now()
    if next_proactive_check_time and now_dt < next_proactive_check_time:
        return

    try:
        logger.error(
            f"[proactive_check] tick at {now_dt.strftime('%Y-%m-%d %H:%M:%S')} "
            f"user_chats={len(user_chats)}"
        )

        # 夜间屏蔽（凌晨 0~8 点不打扰，直接跳过，finally 会安排下次）
        if 0 <= now_dt.hour < 8:
            logger.error("[proactive_check] skipped due to night hours (0-8)")
            return

        for user_id, chat_ai in list(user_chats.items()):
            try:
                if not chat_ai.can_send_proactive_today(now_dt):
                    logger.error(f"[proactive_check] user={user_id} blocked_today")
                    continue

                # 用用户最后说话时间计算沉默时长，AI 主动发言不重置这个门槛
                last_user_ts = chat_ai.last_user_message_timestamp
                user_silence_seconds = (now_dt - last_user_ts).total_seconds() if last_user_ts else 86400

                # 增加总互动沉默时长，用于 AI 评估频率
                last_total_ts = chat_ai.last_message_timestamp
                total_silence_seconds = (now_dt - last_total_ts).total_seconds() if last_total_ts else 86400
                logger.error(
                    f"[proactive_check] user={user_id} "
                    f"user_silence={user_silence_seconds:.0f}s "
                    f"total_silence={total_silence_seconds:.0f}s"
                )

                # 用户刚说过话（30 分钟内），不主动打扰；AI 自己发的不算
                if user_silence_seconds < 1800:
                    logger.error(f"[proactive_check] user={user_id} skip (user_silence < 1800s)")
                    continue

                # 意愿评估（同步调用，独立 client，不污染 chat history）
                logger.error(f"[proactive_check] user={user_id} calling evaluate_proactive_intent")
                should_send, trigger_hint = evaluate_proactive_intent(chat_ai, user_silence_seconds, total_silence_seconds)
                if not should_send:
                    logger.error(f"[proactive_check] user={user_id} evaluate_proactive_intent=NO")
                    continue

                # AI 生成主动消息（同步调用）
                response_text, image_path = chat_ai.send_proactive_message(trigger_hint)

                if not response_text.strip() and not image_path:
                    continue

                # 发送图片（如有）
                sent_any = False
                if image_path and os.path.exists(image_path):
                    await _send_with_retry(
                        lambda: context.bot.send_photo(chat_id=user_id, photo=open(image_path, 'rb')),
                        label="proactive_send_photo"
                    )
                    sent_any = True

                # 分段发送文字，模拟真实打字速度（120 字/分钟），与 handle_message 保持一致
                if response_text.strip():
                    lines = response_text.split('\n')
                    char_delay = 60.0 / 120.0
                    for line in lines:
                        if line.strip():
                            delay = len(line.strip()) * char_delay

                            async def keep_typing(cid, dur):
                                end_time = asyncio.get_event_loop().time() + dur
                                while asyncio.get_event_loop().time() < end_time:
                                    try:
                                        await _send_with_retry(
                                            lambda: context.bot.send_chat_action(chat_id=cid, action="typing"),
                                            label="proactive_send_chat_action"
                                        )
                                        await asyncio.sleep(4)
                                    except asyncio.CancelledError:
                                        break

                            typing_task = asyncio.create_task(keep_typing(user_id, delay))
                            await asyncio.sleep(delay)
                            typing_task.cancel()
                            await _send_with_retry(
                                lambda: context.bot.send_message(chat_id=user_id, text=line.strip()),
                                label="proactive_send_message"
                            )
                            sent_any = True

                if sent_any:
                    chat_ai.on_proactive_sent(now_dt)

            except Exception as e:
                logger.error(f"主动消息处理出错 (user={user_id}): {e}")

    except Exception as e:
        logger.error(f"proactive_check_job 整体出错: {e}")
    finally:
        # 无论本轮结果如何，动态安排下一次检查
        try:
            _schedule_next_proactive_check()
            logger.error("[proactive_check] next check scheduled")
        except Exception as e:
            logger.error(f"[proactive_check] failed to schedule next check: {e}")


async def post_init(application: Application):
    """Bot 启动后自动恢复默认的 ChatAI 实例，保证 proactive job 启动时有实例可用。"""
    db = Database()
    try:
        characters = db.list_characters()
        if characters and ALLOWED_USER_ID not in user_chats:
            char = characters[0]  # 默认使用第一个角色
            user_chats[ALLOWED_USER_ID] = ChatAI(
                system_instruction=char['system_instruction'],
                character_id=char['id']
            )
            logger.error(f"[post_init] 已自动恢复角色 '{char['name']}' 的聊天实例")
    except Exception as e:
        logger.error(f"[post_init] 自动恢复聊天实例失败: {e}")
    finally:
        db.close()


def main():
    """启动机器人"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not token:
        print("错误：请设置 TELEGRAM_BOT_TOKEN 环境变量")
        return
    
    # 创建应用，增加超时时间
    request = HTTPXRequest(
        connect_timeout=20.0,
        read_timeout=90.0,
        write_timeout=90.0,
        pool_timeout=20.0,
    )
    application = (
        Application.builder()
        .token(token)
        .request(request)
        .post_init(post_init)
        .build()
    )
    
    # 添加全局权限检查过滤器（最先执行）
    application.add_handler(MessageHandler(filters.ALL, check_permission), group=-1)
    
    # 添加处理器
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("select", select_character))
    application.add_handler(CommandHandler("useModel", use_model))
    application.add_handler(CommandHandler("usemodel", use_model))
    application.add_handler(CommandHandler("history", history))
    application.add_handler(CommandHandler("del", delete_from_last_user))
    application.add_handler(CommandHandler("filter", trigger_filter))
    application.add_handler(CommandHandler("consolidate", trigger_consolidate))
    application.add_handler(CommandHandler("care_extract", trigger_hourly_care_extract))
    application.add_handler(CommandHandler("rebuild_cache", trigger_rebuild_cache))
    application.add_handler(CommandHandler("drop_cache", trigger_drop_cache))
    application.add_handler(CommandHandler("help", help_command))
    # 更新过滤器，支持所有消息（包含骰子）
    application.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_message))
    
    # 启动后台记忆任务
    async def memory_filter_job(context: ContextTypes.DEFAULT_TYPE):
        from memory_worker import MemoryWorker
        worker = MemoryWorker()
        await worker.filter_task()

    async def memory_consolidate_job(context: ContextTypes.DEFAULT_TYPE):
        from memory_worker import MemoryWorker
        worker = MemoryWorker()
        await worker.consolidate_task()

    job_queue = application.job_queue

    # 每天凌晨三点清空所有内存列表的上下文，截断上下文依赖
    async def clear_chat_history_job(context: ContextTypes.DEFAULT_TYPE):
        try:
            for uid, chat_ai in user_chats.items():
                chat_ai.clear_history()
            logger.info("已清空所有日更活跃用户聊天列表")
        except Exception as e:
            logger.error(f"清空聊天列表失败: {e}")

    # 每天凌晨三点（北京时间 UTC+8）执行一次提纯。UTC 19:00 是北京时间 03:00
    filter_time = datetime.time(hour=19, minute=0, second=0, tzinfo=datetime.timezone.utc)
    job_queue.run_daily(memory_filter_job, time=filter_time)
    job_queue.run_daily(clear_chat_history_job, time=filter_time)

    # 每天凌晨四点（北京时间 UTC+8）执行一次巩固。UTC 20:00 是北京时间 04:00
    consolidate_time = datetime.time(hour=20, minute=0, second=0, tzinfo=datetime.timezone.utc)
    job_queue.run_daily(memory_consolidate_job, time=consolidate_time)

    # AI 主动发消息：首次检查在启动后随机 10~25 分钟触发
    # 之后每轮由 proactive_check_job 自身动态安排下次，间隔同样随机 10~25 分钟，保证无规律感
    _schedule_next_proactive_check()
    logger.error("[proactive_check] initial check scheduled")
    job_queue.run_repeating(proactive_check_job, interval=60, first=30)
    # JobQueue 心跳：每 10 分钟确认调度器仍在运行
    async def jobqueue_heartbeat(context: ContextTypes.DEFAULT_TYPE):
        now_dt = datetime.datetime.now()
        logger.error(
            f"[jobqueue_heartbeat] {now_dt.strftime('%Y-%m-%d %H:%M:%S')} "
            f"user_chats={len(user_chats)}"
        )
    job_queue.run_repeating(jobqueue_heartbeat, interval=600, first=60)

    # 已关闭“每小时整点自动提取关怀事项”定时任务。
    # 如需执行，可使用 /care_extract 手动触发。

    # 注册提醒任务扫描：每分钟检查一次是否有到期的提醒
    job_queue.run_repeating(reminder_job, interval=60, first=10)

    # 启动机器人
    print("Telegram Bot 已启动，记忆漏斗任务已注册")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
