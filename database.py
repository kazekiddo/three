import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import numpy as np
import logging
from pgvector.psycopg2 import register_vector

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
            # 注册 pgvector 支持
            try:
                register_vector(self.conn)
            except psycopg2.errors.UndefinedObject:
                # 如果没有开启矢量扩展，可以忽略或打印警告
                logging.warning("pgvector 扩展可能未启用")
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
    
    def save_message(self, character_id, role, content, model=None, context_prefix=None, media_path=None, media_type=None):
        """保存聊天消息"""
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO chat_messages (character_id, role, content, context_prefix, model, media_path, media_type, timestamp) 
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (character_id, role, content, context_prefix, model, media_path, media_type, datetime.now())
                )
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
    
    def get_chat_history(self, character_id, limit=50):
        """获取聊天历史"""
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT role, content, context_prefix, model, media_path, media_type, timestamp 
                   FROM chat_messages 
                   WHERE character_id = %s 
                   ORDER BY timestamp DESC 
                   LIMIT %s""",
                (character_id, limit)
            )
            messages = cur.fetchall()
            return list(reversed(messages))

    def get_recent_chat_history(self, character_id, since_time):
        """获取指定时间点之后的完整对话历史，用于每日充当感知缓冲层"""
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT role, content, context_prefix, model, media_path, media_type, timestamp 
                   FROM chat_messages 
                   WHERE character_id = %s 
                     AND timestamp >= %s
                   ORDER BY timestamp ASC""",
                (character_id, since_time)
            )
            return cur.fetchall()

    # --- 记忆漏斗方法开始 ---

    def get_unextracted_messages(self, character_id, limit=5000):
        """获取指定角色未处理的原始对话"""
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT id, character_id, role, content, context_prefix, timestamp 
                   FROM chat_messages 
                   WHERE character_id = %s AND is_extracted = false 
                   ORDER BY timestamp ASC 
                   LIMIT %s""",
                (character_id, limit)
            )
            return cur.fetchall()

    def mark_messages_extracted(self, message_ids):
        """将原始对话标记为已提取"""
        if not message_ids:
            return
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_messages SET is_extracted = true WHERE id = ANY(%s)",
                (list(message_ids),)
            )
            conn.commit()

    def get_last_message_timestamp(self, character_id):
         """获取该角色最后一次发言（不论user还是model）的时间"""
         conn = self.connect()
         with conn.cursor() as cur:
             cur.execute(
                 """SELECT timestamp FROM chat_messages 
                    WHERE character_id = %s 
                    ORDER BY timestamp DESC LIMIT 1""",
                 (character_id,)
             )
             res = cur.fetchone()
             return res[0] if res else None

    def get_last_user_message_timestamp(self, character_id):
        """获取该角色用户（role='user'）最后一次发言的时间，用于 proactive 30 分钟门槛的持久化恢复。"""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT timestamp FROM chat_messages
                   WHERE character_id = %s AND role = 'user'
                   ORDER BY timestamp DESC LIMIT 1""",
                (character_id,)
            )
            res = cur.fetchone()
            return res[0] if res else None


    def get_context_since_nth_user_message(self, character_id, user_msg_count=4):
        """以用户最近 user_msg_count 句发言中最早那条的时间点为锚，
        返回从那个时间点到现在的所有对话（user + model 都包含）。
        即使 AI 连续主动发了很多条，用户的声音和完整对话链路也不会丢失。
        """
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 第一步：找用户最近 user_msg_count 条消息，取最早的时间戳
            cur.execute(
                """SELECT timestamp FROM chat_messages
                   WHERE character_id = %s AND role = 'user'
                   ORDER BY timestamp DESC
                   LIMIT %s""",
                (character_id, user_msg_count)
            )
            rows = cur.fetchall()
            if not rows:
                return []

            since_time = min(r['timestamp'] for r in rows)

            # 第二步：从那个时间点到现在，拉出所有消息（含 AI 回复和主动发言）
            cur.execute(
                """SELECT role, content, timestamp
                   FROM chat_messages
                   WHERE character_id = %s AND timestamp >= %s
                   ORDER BY timestamp ASC""",
                (character_id, since_time)
            )
            return cur.fetchall()


    def save_episodic_memory(self, character_id, content, emotion_intensity, promotion_candidate=True, embedding=None, event_time=None):
        """保存提纯后的情景记忆（含向量和事件时间）"""
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO episodic_memories (character_id, content, emotion_intensity, promotion_candidate, embedding, event_time)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (character_id, content, emotion_intensity, promotion_candidate,
                     np.array(embedding) if embedding is not None else None,
                     event_time)
                )
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise e

    def search_episodic_memories(self, character_id, query_embedding, limit=5, time_start=None, time_end=None):
        """向量检索最相关的情景记忆，支持时间范围过滤"""
        conn = self.connect()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if time_start and time_end:
                    # 指定时间范围：按纯相似度排序，不使用时间衰减
                    cur.execute(
                        """SELECT id, content, emotion_intensity, event_time, created_at,
                                  1 - (embedding <=> %s::vector) AS similarity
                           FROM episodic_memories 
                           WHERE character_id = %s 
                             AND embedding IS NOT NULL
                             AND event_time IS NOT NULL
                             AND event_time BETWEEN %s AND %s
                           ORDER BY embedding <=> %s::vector
                           LIMIT %s""",
                        (np.array(query_embedding), character_id, time_start, time_end, np.array(query_embedding), limit)
                    )
                else:
                    # 默认：余弦相似度 × 时间衰减因子（半衰期30天）
                    cur.execute(
                        """SELECT id, content, emotion_intensity, event_time, created_at,
                                  1 - (embedding <=> %s::vector) AS similarity
                           FROM episodic_memories 
                           WHERE character_id = %s 
                             AND embedding IS NOT NULL
                           ORDER BY (1 - (embedding <=> %s::vector)) * EXP(-0.023 * EXTRACT(EPOCH FROM (NOW() - COALESCE(event_time, created_at))) / 86400) DESC
                           LIMIT %s""",
                        (np.array(query_embedding), character_id, np.array(query_embedding), limit)
                    )
                return cur.fetchall()
        except Exception as e:
            conn.rollback()
            raise e

    def get_all_characters_with_episodic(self):
         """获取所有有情景记忆的角色 ID"""
         conn = self.connect()
         with conn.cursor(cursor_factory=RealDictCursor) as cur:
             cur.execute("SELECT DISTINCT character_id FROM episodic_memories")
             return [row['character_id'] for row in cur.fetchall()]

    def get_unconsolidated_episodic_memories(self, character_id, limit=100):
        """获取尚未合并巩固到核心人格的情景记忆"""
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT id, content, emotion_intensity 
                   FROM episodic_memories 
                   WHERE character_id = %s 
                     AND promotion_candidate = true 
                     AND is_consolidated = false
                   ORDER BY created_at ASC
                   LIMIT %s""",
                (character_id, limit)
            )
            return cur.fetchall()

    def mark_episodic_consolidated(self, episodic_ids):
        """标记情景记忆为已合并巩固"""
        if not episodic_ids: return
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE episodic_memories SET is_consolidated = true WHERE id = ANY(%s)",
                (episodic_ids,)
            )
            conn.commit()

    def get_active_core_facts(self, character_id):
        """获取角色所有活跃（未归档）的核心特质"""
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT id, fact_text, category, stability_score, evidence_span 
                   FROM core_fact_memories 
                   WHERE character_id = %s AND is_archived = false""",
                (character_id,)
            )
            return cur.fetchall()

    def save_core_fact_memory(self, character_id, fact_text, embedding, category, stability_score, evidence_span):
        """保存核心人格特征"""
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                # pgvector 支持 numpy arrays 或者 lists 作为输入
                cur.execute(
                    """INSERT INTO core_fact_memories 
                       (character_id, fact_text, embedding, category, stability_score, evidence_span)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (character_id, fact_text, np.array(embedding), category, stability_score, evidence_span)
                )
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise e

    def search_core_fact_memories(self, character_id, query_embedding, limit=3):
        """向量检索最相关的性格核心特征（未归档且相关度高）"""
        conn = self.connect()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # 用余弦距离，<-> 是 Euclidean, <=> 是余弦距离, <#> 是内积
                # 搜索相似度高的事实，并要求未归档且 validation_score > 0.3
                cur.execute(
                    """SELECT id, fact_text, evidence_span, validation_score, stability_score, 1 - (embedding <=> %s::vector) AS similarity 
                       FROM core_fact_memories 
                       WHERE character_id = %s 
                         AND is_archived = false
                         AND validation_score > 0.3
                       ORDER BY embedding <=> %s::vector 
                       LIMIT %s""",
                    (np.array(query_embedding), character_id, np.array(query_embedding), limit)
                )
                return cur.fetchall()
        except Exception as e:
            conn.rollback()
            raise e

    def get_similar_core_fact(self, character_id, embedding, threshold=0.85):
        """查找是否有非常相似的已有核心特征，返回 id"""
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT id, stability_score 
                   FROM core_fact_memories 
                   WHERE character_id = %s 
                     AND is_archived = false
                     AND (1 - (embedding <=> %s::vector)) > %s
                   LIMIT 1""",
                (character_id, np.array(embedding), threshold)
            )
            return cur.fetchone()

    def update_core_fact_memory(self, fact_id, stability_score, evidence_span):
        """更新已有的核心人格特征（演进分数和证据）"""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE core_fact_memories 
                   SET stability_score = (stability_score + %s) / 2.0,
                       evidence_span = CONCAT(evidence_span, ' | ', %s),
                       updated_at = CURRENT_TIMESTAMP
                   WHERE id = %s""",
                (stability_score, evidence_span, fact_id)
            )
            conn.commit()

    def update_validation_score(self, fact_id, delta):
        """更新验证分，如果低于0.3则归档"""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE core_fact_memories 
                   SET validation_score = validation_score + %s 
                   WHERE id = %s 
                   RETURNING validation_score""",
                (delta, fact_id)
            )
            score = cur.fetchone()[0]
            if score < 0.3:
                cur.execute("UPDATE core_fact_memories SET is_archived = true WHERE id = %s", (fact_id,))
            conn.commit()
    # --- 记忆漏斗方法结束 ---
