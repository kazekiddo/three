import os
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

        # 从数据库加载动态缓冲：0-4点加载前一天4点后的记录；5-23点加载当天4点后的记录
        if character_id:
            self.last_message_timestamp = self.db.get_last_message_timestamp(character_id)
            
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
            config['system_instruction'] = system_instruction
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
            config=config
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
            
            # 更新上次说话时间为本次
            self.last_message_timestamp = now_dt
        
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
                
                # 调取最相关的5条情景记忆（具体事件、时间线）
                episodic_memories = self.db.search_episodic_memories(self.character_id, embedding, limit=5)
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

        # 获取AI回复，过滤掉模型的思考过程（THOUGHT part）
        response = self.chat.send_message(message_parts)
        response_text = ""
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text and not part.text.strip().startswith("THOUGHT"):
                    response_text += part.text
        
        # 提取生成图片的路径（如果有的话）
        image_path = self.pending_output_image
        
        # 保存AI回复（只有非空时才保存）
        if self.character_id and (response_text.strip() or image_path):
            self.db.save_message(
                self.character_id, 'model', response_text, 
                model=self.model,
                media_path=image_path,
                media_type='image/jpeg' if image_path else None
            )
            # 同样更新内存中的最后互动时间，确保 30 分钟判断更精准
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

    # 启动机器人
    print("Telegram Bot 已启动，记忆漏斗任务已注册")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
