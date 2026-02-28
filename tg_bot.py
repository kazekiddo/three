import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from database import Database

# 只记录错误日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)
logger = logging.getLogger(__name__)

load_dotenv()

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
        self.db = Database()
        
        # 创建配置
        if system_instruction:
            self.chat = self.client.chats.create(
                model=model,
                config={'system_instruction': system_instruction}
            )
        else:
            self.chat = self.client.chats.create(model=model)
    
    def send_message(self, message):
        """发送消息并获取回复"""
        # 保存用户消息
        if self.character_id:
            self.db.save_message(self.character_id, 'user', message, self.model)
        
        # 获取AI回复
        response = self.chat.send_message(message)
        response_text = response.text
        
        # 保存AI回复
        if self.character_id:
            self.db.save_message(self.character_id, 'assistant', response_text, self.model)
        
        return response_text

# 全局变量存储每个用户的聊天实例
user_chats = {}

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
    user_chats[user_id] = ChatAI(
        system_instruction=character['system_instruction'],
        character_id=character_id
    )
    
    await update.message.reply_text(f"已选择角色：{character['name']}\n现在可以开始聊天了！")
    db.close()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理用户消息"""
    import asyncio
    
    user_id = update.effective_user.id
    user_message = update.message.text
    user_name = update.effective_user.full_name or update.effective_user.username or "Unknown"
    
    # 白名单检查
    ALLOWED_USER_ID = 569020802
    if user_id != ALLOWED_USER_ID:
        # 通知管理员
        try:
            await context.bot.send_message(
                chat_id=ALLOWED_USER_ID,
                text=f"未授权用户尝试访问：\n用户名: {user_name}\nID: {user_id}\n消息: {user_message}"
            )
        except:
            pass
        
        # 回复未授权用户
        await update.message.reply_text("I don't know.")
        return
    
    # 检查用户是否已选择角色
    if user_id not in user_chats:
        await update.message.reply_text("请先使用 /start 和 /select 选择角色")
        return
    
    try:
        # 获取AI回复
        chat_ai = user_chats[user_id]
        response = chat_ai.send_message(user_message)
        
        # 按换行符拆分消息
        messages = response.split('\n')
        
        # 每分钟80个汉字，即每个字0.75秒
        char_delay = 60.0 / 80.0
        
        # 分别发送每条消息，模拟输入延迟
        for msg in messages:
            if msg.strip():
                # 计算这条消息的延迟时间
                delay = len(msg.strip()) * char_delay
                await asyncio.sleep(delay)
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
        history_text += f"{msg['role']}: {msg['message'][:50]}...\n"
    
    await update.message.reply_text(history_text)

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
    
    # 添加处理器
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("select", select_character))
    application.add_handler(CommandHandler("history", history))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # 启动机器人
    print("Telegram Bot 已启动")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
