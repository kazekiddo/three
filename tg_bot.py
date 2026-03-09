import os
import re
import logging
import json
import random
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types
from database import Database
import datetime
import io
from PIL import Image


# 只记录错误日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)
logger = logging.getLogger(__name__)

load_dotenv()

# 确保媒体目录存在
MEDIA_DIR = os.path.join(os.getcwd(), 'media', 'photos')
os.makedirs(MEDIA_DIR, exist_ok=True)

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
    def __init__(self, model="gemini-2.5-flash", api_key=None, system_instruction=None, character_id=None):
        """初始化聊天AI"""
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("请设置 GOOGLE_API_KEY 或 GEMINI_API_KEY 环境变量")
        
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.character_id = character_id
        self.system_instruction = system_instruction
        self.db = Database()
        self.last_message_timestamp = None
        self.last_user_message_timestamp = None  # 只在用户发消息时更新，供主动发言的 30 分钟门槛使用
        self.last_prefix_timestamp = None
        self.enable_web_search = os.getenv("ENABLE_WEB_SEARCH", "true").lower() in ("1", "true", "yes", "on")
        self.proactive_streak_count = 0
        self.proactive_streak_date = None
        self.proactive_blocked_date = None
        self.last_proactive_image_timestamp = None
        self.proactive_image_cooldown_seconds = int(os.getenv("PROACTIVE_IMAGE_COOLDOWN_SECONDS", "3600"))
        self.proactive_image_probability = float(os.getenv("PROACTIVE_IMAGE_PROBABILITY", "0.35"))
        self.runtime_history_limit = max(0, int(os.getenv("RUNTIME_HISTORY_LIMIT", "6")))
        self.dialogue_excerpt_limit = max(0, int(os.getenv("DIALOGUE_EXCERPT_LIMIT", "4")))
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

        # 仅恢复最后互动时间，不在初始化阶段回灌当天长历史
        if character_id:
            self.last_message_timestamp = self.db.get_last_message_timestamp(character_id)
            self.last_user_message_timestamp = self.db.get_last_user_message_timestamp(character_id)
        
        def classify_image_subject(prompt_text: str):
            """根据提示词判断生图主体，返回 character_only/user_only/both/none"""
            text = (prompt_text or "").strip().lower()
            if not text:
                return "none"

            both_keywords = [
                "我们", "一起", "同框", "合照", "你和我", "我和你", "咱俩", "两个人",
                "both of us", "together", "couple", "with me", "with siyuan", "with the user"
            ]
            character_keywords = [
                "你", "你自己", "小七", "nanase", "girl", "the girl", "the character",
                "自拍", "你的照片", "你的样子", "你在", "you", "yourself"
            ]
            user_keywords = [
                "我", "我自己", "思远", "siyuan", "the user", "user only", "我的照片",
                "我长什么样", "给我画", "me", "myself"
            ]

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
                    "Reference image 1 is the main and only identity reference."
                )
            elif subject_mode == "user_only":
                identity_block = (
                    "Create an illustration of the exact same person as reference image 1. "
                    "He must remain the same person, not a redesigned variant. "
                    "Identity preservation requirements: keep the same face shape, eyes, hairstyle, hair color, "
                    "age impression, and overall facial identity. Do not redesign him into a different anime character. "
                    "Reference image 1 is the main and only identity reference."
                )
            elif subject_mode == "both":
                identity_block = (
                    "Create an illustration featuring the same two people as the two reference images. "
                    "Reference image 1 is the girl, reference image 2 is the user Siyuan. "
                    "Both must remain the same people, not redesigned variants. "
                    "Preserve each person's face shape, eyes, hairstyle, hair color, age impression, "
                    "and overall identity. Do not merge their facial features."
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
                full_prompt = build_image_prompt(prompt, subject_mode)
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
        self.pending_output_image = None
        
        # 创建运行配置（不再依赖持久 chat session，改为每轮显式生成）
        config = {'automatic_function_calling': {}}
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
            config['system_instruction'] = self.system_instruction
        config['tools'] = self.tools
        self.runtime_config = config
        self.base_history = list(history)

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

    def _recent_dialogue_excerpt(self, limit=None):
        if not self.character_id:
            return ""
        if limit is None:
            limit = self.dialogue_excerpt_limit
        if limit <= 0:
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

    def _build_runtime_history(self, limit=None):
        history = list(self.base_history)
        if not self.character_id:
            return history
        if limit is None:
            limit = self.runtime_history_limit
        if limit <= 0:
            return history
        try:
            messages = self.db.get_chat_history(self.character_id, limit=limit)
        except Exception as e:
            logger.error(f"读取运行时历史失败: {e}")
            return history

        for msg in messages:
            content = (msg.get('content') or '').strip()
            if not content:
                continue
            prefix = f"{msg['context_prefix']} " if msg.get('context_prefix') else ""
            history.append({
                'role': msg.get('role') or 'user',
                'parts': [{'text': f"{prefix}{content}"}]
            })
        return history

    def _generate_with_runtime_history(self, user_parts, history_limit=None):
        contents = self._build_runtime_history(limit=history_limit)
        contents.append({
            'role': 'user',
            'parts': user_parts
        })
        return self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**self.runtime_config)
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
        recent_dialogue = self._recent_dialogue_excerpt()
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
        """显式生成模式下无需重建 chat session，仅保留基础历史"""
        logger.info(f"角色 {self.character_id} 运行于短上下文模式，无需清空持久 chat session")
    
    def send_message(self, message, image_data=None, image_mime_type=None, media_path=None):
        """发送消息并获取回复"""
        context_prefix = None
        proactive_image_forced = False
        relation_desc = ""
        compact_relation_desc = ""
        dynamic_state = self.dynamic_state
        turn_plan = None
        
        # 处理时间连续性感知架构 (判断与上一条用户消息的时间差)
        if self.character_id:
            last_timestamp = self.last_message_timestamp
            now_dt = datetime.datetime.now()
            
            # 如果是第一次聊天，或者距离上次发言超过 30 分钟，或者距离上次打标超过 30 分钟（针对连续聊天），强制增加系统时间锚点
            if (not last_timestamp or 
                (now_dt - last_timestamp).total_seconds() > 1800 or 
                not self.last_prefix_timestamp or 
                (now_dt - self.last_prefix_timestamp).total_seconds() > 1800):
                
                weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
                weekday_str = weekdays[now_dt.weekday()]
                current_time_str = now_dt.strftime(f"%Y年%m月%d日 {weekday_str} %H:%M:%S")
                context_prefix = f"[系统时间感知：当前时间 {current_time_str}]"
                
                # 更新上一次插入时间后缀的锚点
                self.last_prefix_timestamp = now_dt
            
            # 更新上次说话时间（用户侧）
            self.last_message_timestamp = now_dt
            self.last_user_message_timestamp = now_dt  # 只有用户说话才更新这个
        
        # 保存用户消息，同步落库纯净和前置标签（如果有）
        if self.character_id:
            self.db.save_message(
                self.character_id, 'user', message, 
                model=self.model, 
                context_prefix=context_prefix,
                media_path=media_path,
                media_type=image_mime_type
            )
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
                        evidence = f" (依据: {fact['evidence_span']})" if fact.get('evidence_span') else ""
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
            dynamic_state, turn_plan = self._infer_state_and_plan(
                message,
                relation_desc=compact_relation_desc,
                image_requested=self._should_prefer_image_response(message, image_data=image_data),
                proactive=False
            )

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
            augmented_message += self._build_dynamic_state_prompt(dynamic_state, turn_plan=turn_plan)

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

        # 重置待处理图片
        self.pending_output_image = None

        # 构造多模态消息 Parts
        message_parts = [{'text': augmented_message}]
        if image_data:
            message_parts.append(types.Part.from_bytes(data=image_data, mime_type=image_mime_type))

        # 显式生成：只携带短历史，避免持久 chat session 无限累积 token
        response = self._generate_with_runtime_history(message_parts, history_limit=12)
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
        
        # 提取生成图片的路径（如果有的话）
        image_path = self.pending_output_image
        if proactive_image_forced and image_path:
            self.last_proactive_image_timestamp = datetime.datetime.now()
        
        # 保存AI回复（只有非空时才保存）
        if self.character_id and (response_text.strip() or image_path):
            save_text = response_text if response_text.strip() else "[AI发送了一张图片]"
            self.db.save_message(
                self.character_id, 'model', save_text, 
                model=self.model,
                media_path=image_path,
                media_type='image/jpeg' if image_path else None
            )
            # 同样更新内存中的最后互动时间，确保 30 分钟判断更精准
            self.last_message_timestamp = datetime.datetime.now()
            self._update_post_reply_state(
                trigger_message=message,
                model_reply=save_text,
                base_state=dynamic_state,
                turn_plan=turn_plan,
                proactive=False
            )
        
        return response_text, image_path

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
        if dynamic_state_prompt:
            trigger_msg += dynamic_state_prompt

        self.pending_output_image = None

        response = self._generate_with_runtime_history([{'text': trigger_msg}], history_limit=12)

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

        image_path = self.pending_output_image
        
        # 构造落库用的时间感知前缀 (遵循 30 分钟规则，避免连发时标签堆叠)
        context_prefix = None
        if (not self.last_message_timestamp or 
            (now_dt - self.last_message_timestamp).total_seconds() > 1800 or 
            not self.last_prefix_timestamp or 
            (now_dt - self.last_prefix_timestamp).total_seconds() > 1800):
            
            context_prefix = f"[系统时间感知：当前时间 {current_time_str}]"
            self.last_prefix_timestamp = now_dt

        # 只落库 model 侧消息
        if self.character_id and (response_text.strip() or image_path):
            self.db.save_message(
                self.character_id, 'model', response_text,
                model=self.model,
                context_prefix=context_prefix,
                media_path=image_path,
                media_type='image/jpeg' if image_path else None
            )
            self.last_message_timestamp = datetime.datetime.now()
            self._update_post_reply_state(
                trigger_message=trigger_hint,
                model_reply=response_text if response_text.strip() else "[AI发送了一张图片]",
                base_state=current_state,
                turn_plan=proactive_plan,
                proactive=True
            )

        return response_text, image_path

# 全局变量存储每个用户的聊天实例
user_chats = {}
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
            system_instruction=character['system_instruction'],
            character_id=character_id
        )
        await update.message.reply_text(f"已选择角色：{character['name']}\n现在可以开始聊天了！")
    except Exception as e:
        logger.error(f"创建聊天实例失败: {e}")
        await update.message.reply_text(f"创建聊天失败：{str(e)}")
    
    db.close()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理用户消息"""
    import asyncio
    
    async def keep_typing(chat_id, duration):
        """持续发送 typing 状态"""
        end_time = asyncio.get_event_loop().time() + duration
        while asyncio.get_event_loop().time() < end_time:
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                await asyncio.sleep(4)  # 每4秒刷新一次
            except asyncio.CancelledError:
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
    
    try:
        # 获取AI回复
        chat_ai = user_chats[user_id]
        response, ai_image_path = chat_ai.send_message(
            user_message, 
            image_data=image_data, 
            image_mime_type=image_mime_type,
            media_path=media_path
        )
        
        # 如果 AI 生成了图片，先发送图片
        if ai_image_path and os.path.exists(ai_image_path):
            await context.bot.send_photo(
                chat_id=update.effective_chat.id, 
                photo=open(ai_image_path, 'rb')
            )
        
        # 按换行符拆分消息
        messages = response.split('\n')
        
        # 每分钟120个汉字
        char_delay = 60.0 / 120.0
        
        # 分别发送每条消息，模拟输入延迟
        for msg in messages:
            if msg.strip():
                # 计算这条消息的延迟时间
                delay = len(msg.strip()) * char_delay
                
                # 在延迟期间持续显示正在输入
                typing_task = asyncio.create_task(keep_typing(update.effective_chat.id, delay))
                await asyncio.sleep(delay)
                typing_task.cancel()
                
                await update.message.reply_text(msg.strip())
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

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /help 命令，显示所有可用命令"""
    help_text = (
        "🤖 <b>可用命令列表</b>\n\n"
        "/start - 显示所有可用角色列表\n"
        "/select &lt;角色ID&gt; - 选择一个角色开始聊天\n"
        "/history - 查看与当前角色的最近 10 条历史记录\n"
        "/filter - 手动触发抽取情景记忆任务 (后台运行)\n"
        "/consolidate - 手动触发巩固核心人格任务 (后台运行)\n"
        "/care_extract [小时] - 手动触发整点关怀提取任务，默认最近 1 小时 (后台运行)\n"
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


def _schedule_next_proactive_check(context):
    """安排下一次主动意愿检查，间隔随机 600~1500 秒（10~25 分钟），保证无规律感。"""
    import random
    next_interval = random.randint(600, 1500)
    context.application.job_queue.run_once(proactive_check_job, when=next_interval)


async def proactive_check_job(context: ContextTypes.DEFAULT_TYPE):
    """AI 自主意愿检查：evaluate → generate → send，完成后动态安排下一次检查。
    - 夜间（北京时间 0:00~7:00）静默，不发消息
    - 沉默不足 30 分钟不打扰
    - AI 自己决定要不要发，内容也是 AI 自己生成
    - 打字延迟模拟与普通消息一致
    """
    import asyncio
    try:
        now_dt = datetime.datetime.now()

        # 夜间屏蔽（凌晨 0~7 点不打扰，直接跳过，finally 会安排下次）
        if 0 <= now_dt.hour < 7:
            return

        for user_id, chat_ai in list(user_chats.items()):
            try:
                if not chat_ai.can_send_proactive_today(now_dt):
                    continue

                # 用用户最后说话时间计算沉默时长，AI 主动发言不重置这个门槛
                last_user_ts = chat_ai.last_user_message_timestamp
                user_silence_seconds = (now_dt - last_user_ts).total_seconds() if last_user_ts else 86400

                # 增加总互动沉默时长，用于 AI 评估频率
                last_total_ts = chat_ai.last_message_timestamp
                total_silence_seconds = (now_dt - last_total_ts).total_seconds() if last_total_ts else 86400

                # 用户刚说过话（30 分钟内），不主动打扰；AI 自己发的不算
                if user_silence_seconds < 1800:
                    continue

                # 意愿评估（同步调用，独立 client，不污染 chat history）
                should_send, trigger_hint = evaluate_proactive_intent(chat_ai, user_silence_seconds, total_silence_seconds)
                if not should_send:
                    continue

                # AI 生成主动消息（同步调用）
                response_text, image_path = chat_ai.send_proactive_message(trigger_hint)

                if not response_text.strip() and not image_path:
                    continue

                # 发送图片（如有）
                sent_any = False
                if image_path and os.path.exists(image_path):
                    await context.bot.send_photo(chat_id=user_id, photo=open(image_path, 'rb'))
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
                                        await context.bot.send_chat_action(chat_id=cid, action="typing")
                                        await asyncio.sleep(4)
                                    except asyncio.CancelledError:
                                        break

                            typing_task = asyncio.create_task(keep_typing(user_id, delay))
                            await asyncio.sleep(delay)
                            typing_task.cancel()
                            await context.bot.send_message(chat_id=user_id, text=line.strip())
                            sent_any = True

                if sent_any:
                    chat_ai.on_proactive_sent(now_dt)

            except Exception as e:
                logger.error(f"主动消息处理出错 (user={user_id}): {e}")

    except Exception as e:
        logger.error(f"proactive_check_job 整体出错: {e}")
    finally:
        # 无论本轮结果如何，动态安排下一次检查
        _schedule_next_proactive_check(context)


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
    application = (
        Application.builder()
        .token(token)
        .connect_timeout(30.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .pool_timeout(30.0)
        .post_init(post_init)
        .build()
    )
    
    # 添加全局权限检查过滤器（最先执行）
    application.add_handler(MessageHandler(filters.ALL, check_permission), group=-1)
    
    # 添加处理器
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("select", select_character))
    application.add_handler(CommandHandler("history", history))
    application.add_handler(CommandHandler("filter", trigger_filter))
    application.add_handler(CommandHandler("consolidate", trigger_consolidate))
    application.add_handler(CommandHandler("care_extract", trigger_hourly_care_extract))
    application.add_handler(CommandHandler("help", help_command))
    # 更新过滤器，支持文字和图片（MESSAGE_TYPE.PHOTO）
    application.add_handler(MessageHandler((filters.TEXT | filters.PHOTO) & ~filters.COMMAND, handle_message))
    
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
    import random
    job_queue.run_once(proactive_check_job, when=random.randint(600, 1500))

    # 已关闭“每小时整点自动提取关怀事项”定时任务。
    # 如需执行，可使用 /care_extract 手动触发。

    # 注册提醒任务扫描：每分钟检查一次是否有到期的提醒
    job_queue.run_repeating(reminder_job, interval=60, first=10)

    # 启动机器人
    print("Telegram Bot 已启动，记忆漏斗任务已注册")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
