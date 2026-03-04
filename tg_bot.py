import os
import re
import logging
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
    def __init__(self, model="gemini-2.5-flash, api_key=None, system_instruction=None, character_id=None):
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
        
        # 预加载角色设定图和用户（思远）设定图 (用于视觉一致性)
        self.character_photo_path = "/data/three/media/photos/photo_nanase.jpg"
        self.user_photo_path = "/data/three/media/photos/photo_siyuan.jpg"
        
        character_photo_part = None
        if os.path.exists(self.character_photo_path):
            try:
                with open(self.character_photo_path, 'rb') as f:
                    photo_data = f.read()
                character_photo_part = types.Part.from_bytes(data=photo_data, mime_type='image/jpeg')
            except Exception as e:
                logger.error(f"加载设定图出错: {e}")

        user_photo_part = None
        if os.path.exists(self.user_photo_path):
            try:
                with open(self.user_photo_path, 'rb') as f:
                    u_photo_data = f.read()
                user_photo_part = types.Part.from_bytes(data=u_photo_data, mime_type='image/jpeg')
            except Exception as e:
                logger.error(f"加载用户设定图出错: {e}")

        # 从数据库加载过去 24 小时的历史记录作为当天的缓冲区
        history = []
        # base_history 保存设定图注入、每次 clear_history 后复用
        self.base_history = []
        
        # 1. 注入角色设定图
        if character_photo_part:
            history.append({
                'role': 'user',
                'parts': [
                    {'text': "系统通知：以下是你的官方形象设定图（Self-Image Reference）。请仔细观察并分析你的外貌、发型、面部特征及穿着。在后续对话中，如果你需要生成关于自己的图片，请务必保持视觉一致。"},
                    character_photo_part
                ]
            })
            history.append({
                'role': 'model',
                'parts': [{'text': "我已经看到了我的设定图。我会记住我的面部特征和整体形象，并在需要生成我的照片时保持一致性。"}]
            })
            self.base_history = list(history)  # 浅拷贝当前的设定图条目

        # 2. 注入用户（思远）设定图
        if user_photo_part:
            history.append({
                'role': 'user',
                'parts': [
                    {'text': "系统通知：以下是当前对话用户（名：思远）的肖像参考图。请通过此图建立对用户的视觉认知。当你生成关于用户的照片时，请以此为基准。"},
                    user_photo_part
                ]
            })
            history.append({
                'role': 'model',
                'parts': [{'text': "收到了，我已经看到了思远的肖像图。我会记住他的样子。"}]
            })
            self.base_history = list(history)  # 更新，包含两张设定图

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
                
                # 如果有图片，从本地磁盘加载
                if msg.get('media_path') and os.path.exists(msg['media_path']):
                    try:
                        with open(msg['media_path'], 'rb') as f:
                            img_data = f.read()
                        parts.append(types.Part.from_bytes(data=img_data, mime_type=msg.get('media_type', 'image/jpeg')))
                    except Exception as e:
                        logger.error(f"加载历史图片失败: {e}")

                history.append({
                    'role': msg['role'],
                    'parts': parts
                })
        
        # 定义 AI 生成图片的工具
        def generate_image(prompt: str) -> str:
            """根据描述生成一张精美的图片。
            注意：所有生成的图像必须是日系卡通风格（Japanese anime style）。如果你生成的是关于你自己的图片，请结合你记忆中设定图（photo_nanase.jpg）的视觉特征（如面部、发型、风格）来编写 prompt，以保证一致性。
            参数:
                prompt: 详细的图片描述词，使用英文描述效果更佳。
            """
            try:
                # 按照用户提供的“图片编辑/视觉参考”逻辑：采用 [Prompt, Image1, Image2...] 列表形式
                generation_contents = []
                
                # 读取 prompt，并在 prompt 中补充提示，强制要求日系卡通风格，并指明参考图归属
                full_prompt = (
                    f"{prompt}. "
                    "STYLE REQUIREMENT: Strictly follow Japanese anime / cartoon style. "
                    "(Reference image 1 is the character Nanase, Reference image 2 is the user Siyuan)"
                )
                generation_contents.append(full_prompt)
                
                # 1. 加载角色设定图 (Nanase)
                if os.path.exists(self.character_photo_path):
                    try:
                        ref_char_image = Image.open(self.character_photo_path)
                        generation_contents.append(ref_char_image)
                    except Exception as e:
                        logger.error(f"加载角色设定图失败: {e}")
                
                # 2. 加载用户设定图 (Siyuan)
                if os.path.exists(self.user_photo_path):
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
                for part in response.parts:
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

        self.tools = [generate_image]
        self.pending_output_image = None
        
        # 创建聊天
        config = {'automatic_function_calling': {}}
        if system_instruction:
            # 追加图片工具使用约束，防止模型试图直接输出图片
            image_constraint = (
                "\n\n【重要工具使用规则】"
                "你不能直接输出或嵌入图片。你没有原生图片输出能力。"
                "当用户请求图片、自拍、照片时，你必须调用 generate_image 工具函数来生成图片。"
                "绝对不要在回复中写 'Here is the original image' 或类似的占位文字。"
                "调用工具后，系统会自动将图片发送给用户。"
            )
            self.system_instruction = system_instruction + image_constraint
            config['system_instruction'] = self.system_instruction
        config['tools'] = self.tools

        self.chat = self.client.chats.create(
            model=model,
            config=config,
            history=history
        )
            
    def clear_history(self):
        """每天凌晨清空一次内存中的长对话列表，同时保留工具能力"""
        logger.info(f"正在清空角色 {self.character_id} 的短期对话感知缓冲...")
        
        config = {'automatic_function_calling': {}}
        if self.system_instruction:
            config['system_instruction'] = self.system_instruction
        config['tools'] = self.tools

        self.chat = self.client.chats.create(
            model=self.model,
            config=config,
            history=self.base_history
        )
    
    def send_message(self, message, image_data=None, image_mime_type=None, media_path=None):
        """发送消息并获取回复"""
        context_prefix = None
        
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
                        episodic_text += f"{i+1}.{time_str} {mem['content']}\n"
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"检索记忆失败: {e}")

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

        # 重置待处理图片
        self.pending_output_image = None

        # 构造多模态消息 Parts
        message_parts = [augmented_message]
        if image_data:
            message_parts.append(types.Part.from_bytes(data=image_data, mime_type=image_mime_type))

        # 获取AI回复（automatic_function_calling 在 send_message 内部自动完成，response 是最终响应）
        response = self.chat.send_message(message_parts)
        response_text = ""
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text and not part.text.strip().startswith(("THOUGHT", "THINK")):
                    response_text += part.text
        # 清理模型生成图片时附带的内部标记文本
        response_text = response_text.replace("Here is the original image:", "").strip()
        
        # 从 chat history 中清除记忆块，避免 token 累积
        # 记忆只在当前轮对模型可见，发送后即清除
        try:
            for content in reversed(self.chat._curated_history):
                if content.role == 'user':
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text and '\n\n[系统附加' in part.text:
                            part.text = part.text[:part.text.index('\n\n[系统附加')]
                    break
        except Exception:
            pass
        
        # 提取生成图片的路径（如果有的话）
        image_path = self.pending_output_image
        
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
        trigger_msg = (
            f"[SYS_PROACTIVE] 现在是 {current_time_str}，{trigger_hint}"
            "请你以角色身份，自然地主动给思远发一条消息。"
            "内容简短自然，符合你的性格，不要提及这是系统触发或你在'自言自语'。"
            "直接输出你想说的话即可。"
        )

        self.pending_output_image = None

        response = self.chat.send_message([trigger_msg])

        response_text = ""
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text and not part.text.strip().startswith(("THOUGHT", "THINK")):
                    response_text += part.text

        # 从 chat history 中移除注入的假"用户"触发消息，保持历史干净
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
                    break
        except Exception:
            pass

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

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /help 命令，显示所有可用命令"""
    help_text = (
        "🤖 <b>可用命令列表</b>\n\n"
        "/start - 显示所有可用角色列表\n"
        "/select &lt;角色ID&gt; - 选择一个角色开始聊天\n"
        "/history - 查看与当前角色的最近 10 条历史记录\n"
        "/filter - 手动触发抽取情景记忆任务 (后台运行)\n"
        "/consolidate - 手动触发巩固核心人格任务 (后台运行)\n"
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
        answer = response.text.strip().upper()
        should_send = answer.startswith("YES")
        trigger_hint = f"{user_desc}，{total_desc}。"
        return should_send, trigger_hint
    except Exception as e:
        logger.error(f"意愿评估失败: {e}")
        return False, ""


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
                if image_path and os.path.exists(image_path):
                    await context.bot.send_photo(chat_id=user_id, photo=open(image_path, 'rb'))

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

    # 启动机器人
    print("Telegram Bot 已启动，记忆漏斗任务已注册")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
