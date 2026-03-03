import os
import logging
import io
import datetime
from PIL import Image
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types

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

# 存储用户的生图会话状态
# { user_id: { 'chat': genai.Chat, 'mode': str } }
user_sessions = {}

class HelperAI:
    def __init__(self, mode, system_instruction):
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("请设置 GOOGLE_API_KEY 或 GEMINI_API_KEY")
        
        self.client = genai.Client(api_key=api_key)
        self.mode = mode
        
        history = []
        # 根据模式加载视觉参考
        if mode in ['gen_me', 'gen_both'] and os.path.exists(CHARACTER_PHOTO_PATH):
            img = Image.open(CHARACTER_PHOTO_PATH)
            history.append({
                'role': 'user',
                'parts': ["This is the character (Nanase) reference.", img]
            })
            history.append({
                'role': 'model',
                'parts': ["I have received Nanase's reference. I will maintain consistency."]
            })
            
        if mode in ['gen_user', 'gen_both'] and os.path.exists(USER_PHOTO_PATH):
            img = Image.open(USER_PHOTO_PATH)
            history.append({
                'role': 'user',
                'parts': ["This is the user (Siyuan) reference.", img]
            })
            history.append({
                'role': 'model',
                'parts': ["I have received Siyuan's reference. I will maintain consistency."]
            })

        self.chat = self.client.chats.create(
            model='gemini-3.1-flash-image-preview',
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                image_config=types.ImageConfig(aspect_ratio="3:4")
            ),
            history=history
        )

    async def generate(self, prompt):
        """调用 Gemini 生成图片并处理"""
        # 强制要求日系卡通风格
        full_prompt = f"{prompt}. STYLE REQUIREMENT: Strictly follow Japanese anime / cartoon style."
        
        response = self.chat.send_message(full_prompt)
        
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

async def check_auth(update: Update):
    if update.effective_user.id != ALLOWED_USER_ID:
        await update.message.reply_text("Unauthorized.")
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
        user_sessions[user_id] = HelperAI(cmd, instruction)
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

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update): return
    user_id = update.effective_user.id
    
    if user_id not in user_sessions:
        await update.message.reply_text("请先使用 /gen, /gen_me, /gen_user 或 /gen_both 开始生图。")
        return

    prompt = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")
    
    try:
        ai_session = user_sessions[user_id]
        photo_bio, text = await ai_session.generate(prompt)
        
        if photo_bio:
            await update.message.reply_photo(photo=photo_bio)
        
        if text:
            await update.message.reply_text(text)
        elif not photo_bio:
            await update.message.reply_text("未能生成图片，请换个描述试试。")
            
    except Exception as e:
        logger.error(f"生图出错: {e}")
        await update.message.reply_text(f"出错啦: {str(e)}")

def main():
    token = os.getenv('TELEGRAM_HELPER_BOT_TOKEN')
    if not token:
        print("错误：请设置 TELEGRAM_HELPER_BOT_TOKEN")
        return

    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("gen", start_gen))
    application.add_handler(CommandHandler("gen_me", start_gen))
    application.add_handler(CommandHandler("gen_user", start_gen))
    application.add_handler(CommandHandler("gen_both", start_gen))
    application.add_handler(CommandHandler("end", end_gen))
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Helper Bot 已启动")
    application.run_polling()

if __name__ == "__main__":
    main()
