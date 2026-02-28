import os
from dotenv import load_dotenv
from google import genai
from database import Database

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
    
    def get_history(self):
        """获取对话历史"""
        history = []
        for message in self.chat.get_history():
            history.append({
                'role': message.role,
                'content': message.parts[0].text
            })
        return history
    
    def print_history(self):
        """打印对话历史"""
        for message in self.chat.get_history():
            print(f'{message.role}: {message.parts[0].text}')
            print('-' * 50)
    
    def print_db_history(self):
        """从数据库打印历史"""
        if not self.character_id:
            print("未关联角色ID")
            return
        
        messages = self.db.get_chat_history(self.character_id)
        for msg in messages:
            print(f"{msg['timestamp']} - {msg['role']}: {msg['message']}")
            print('-' * 50)

def main():
    db = Database()
    
    # 列出所有角色
    print("=== 可用角色 ===")
    characters = db.list_characters()
    for char in characters:
        print(f"{char['id']}. {char['name']}")
    
    # 选择角色
    print("\n请选择角色ID（直接回车使用ID=1）:")
    choice = input("> ").strip()
    character_id = int(choice) if choice.isdigit() else 1
    
    # 获取角色设定
    character = db.get_character(character_id)
    if not character:
        print("角色不存在")
        return
    
    print(f"\n已选择角色: {character['name']}\n")
    
    # 创建聊天实例
    chat_ai = ChatAI(
        system_instruction=character['system_instruction'],
        character_id=character_id
    )
    
    print("聊天AI已启动！输入 'quit' 或 'exit' 退出，输入 'history' 查看对话历史\n")
    
    while True:
        user_input = input("你: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("再见！")
            db.close()
            break
        
        if user_input.lower() == 'history':
            print("\n=== 数据库历史记录 ===")
            chat_ai.print_db_history()
            continue
        
        if not user_input:
            continue
        
        try:
            response = chat_ai.send_message(user_input)
            print(f"AI: {response}\n")
        except Exception as e:
            print(f"错误: {e}\n")

if __name__ == "__main__":
    main()
