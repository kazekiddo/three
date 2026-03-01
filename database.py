import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

class Database:
    def __init__(self, database_url=None):
        """初始化数据库连接"""
        if database_url is None:
            database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            raise ValueError("请设置 DATABASE_URL 环境变量")
        
        self.database_url = database_url
        self.conn = None
    
    def connect(self):
        """连接数据库"""
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(self.database_url)
        return self.conn
    
    def close(self):
        """关闭数据库连接"""
        if self.conn and not self.conn.closed:
            self.conn.close()
    
    def get_character(self, character_id):
        """获取角色设定"""
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM character_settings WHERE id = %s",
                (character_id,)
            )
            return cur.fetchone()
    
    def list_characters(self):
        """列出所有角色"""
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM character_settings ORDER BY id")
            return cur.fetchall()
    
    def create_character(self, name, system_instruction):
        """创建新角色"""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO character_settings (name, system_instruction) VALUES (%s, %s) RETURNING id",
                (name, system_instruction)
            )
            character_id = cur.fetchone()[0]
            conn.commit()
            return character_id
    
    def save_message(self, character_id, role, content, model=None):
        """保存聊天消息"""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO chat_messages (character_id, role, content, model, timestamp) 
                   VALUES (%s, %s, %s, %s, %s)""",
                (character_id, role, content, model, datetime.now())
            )
            conn.commit()
    
    def get_chat_history(self, character_id, limit=50):
        """获取聊天历史"""
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT role, content, model, timestamp 
                   FROM chat_messages 
                   WHERE character_id = %s 
                   ORDER BY timestamp DESC 
                   LIMIT %s""",
                (character_id, limit)
            )
            messages = cur.fetchall()
            return list(reversed(messages))
