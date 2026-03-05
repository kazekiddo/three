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
            with self.conn.cursor() as cur:
                # 强制当前会话使用北京时间
                cur.execute("SET TIME ZONE 'Asia/Shanghai';")
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


    def save_episodic_memory(self, character_id, content, emotion_intensity, promotion_candidate=True, embedding=None, event_time=None, causal_link_id=None, emotion_category=None):
        """保存提纯后的情景记忆（支持因果链和情绪分类）"""
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO episodic_memories (character_id, content, emotion_intensity, promotion_candidate, embedding, event_time, causal_link_id, emotion_category)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (character_id, content, emotion_intensity, promotion_candidate,
                     np.array(embedding) if embedding is not None else None,
                     event_time, causal_link_id, emotion_category)
                )
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise e

    def get_relationship_state(self, character_id):
        """获取当前关系完整状态，包含依恋类型"""
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT s.*, c.attachment_style 
                   FROM relationship_states s
                   JOIN character_settings c ON s.character_id = c.id
                   WHERE s.character_id = %s""", 
                (character_id,)
            )
            return cur.fetchone()

    def update_relationship_advanced(self, character_id, events, shock_events=None, new_narrative=None):
        """
        全量级演进系统：
        1. 依恋类型修正 (Attachment Logic)
        2. 离散事件 delta 惯性更新
        3. 冲击性事件 (Shock Events) 乘法计算
        4. 动量更新 (Momentum EMA)
        5. 状态机阈值晋升 (Stage State Machine)
        """
        import math
        conn = self.connect()
        state = self.get_relationship_state(character_id)
        if not state: return

        attach = state['attachment_style']
        MAP = {"strong_positive": 0.08, "positive": 0.04, "neutral": 0, "negative": -0.04, "strong_negative": -0.08}
        
        # --- 1. 计算离散事件累积 ---
        total_deltas = {k: 0.0 for k in ['closeness', 'trust', 'resentment', 'dependency', 'attraction', 'respect', 'security', 'jealousy']}
        for e in events:
            t, i = e.get('target'), e.get('intensity')
            if t in total_deltas and i in MAP:
                delta = MAP[i]
                # 依恋类型修正逻辑
                if attach == 'anxious' and t == 'security' and delta < 0:
                    total_deltas['dependency'] += 0.04
                    total_deltas['jealousy'] += 0.04
                    total_deltas['resentment'] += 0.02
                elif attach == 'avoidant' and delta > 0 and t == 'closeness':
                    delta *= 0.5 # 回避型对亲密增加更迟钝
                    total_deltas['dependency'] -= 0.02
                total_deltas[t] += delta

        # --- 2. 动量计算 (EMA) ---
        avg_delta = sum(total_deltas.values()) / len(total_deltas)
        new_momentum = 0.8 * float(state['momentum']) + avg_delta
        new_momentum = max(-1.0, min(1.0, new_momentum))

        # --- 3. 基础演进逻辑 ---
        new_vals = {}
        decay = math.exp(-0.05 * ((datetime.now() - state['last_updated']).total_seconds() / 86400.0))
        for k in total_deltas.keys():
            v, d = float(state[k]), total_deltas[k]
            if k == 'resentment': v *= decay
            v = v + d * (1.0 - v) if d > 0 else v + d * v
            new_vals[k] = max(0.0, min(1.0, v))

        # --- 4. 冲击性事件 (乘法影响) ---
        if shock_events:
            for s in shock_events:
                if s == 'betrayal':
                    new_vals['trust'] *= 0.5
                    new_vals['respect'] *= 0.7
                    new_vals['resentment'] = min(1.0, new_vals['resentment'] + 0.3)
                elif s == 'confession':
                    new_vals['attraction'] = min(1.0, new_vals['attraction'] + 0.2)
                    new_vals['closeness'] = min(1.0, new_vals['closeness'] + 0.1)

        # --- 5. 状态机自动晋升 ---
        stages = ['stranger', 'acquaintance', 'friend', 'close_friend', 'romantic', 'partner']
        curr_idx = stages.index(state['stage']) if state['stage'] in stages else 0
        new_stage = state['stage']
        if curr_idx < len(stages) - 1:
            next_stage = stages[curr_idx + 1]
            # 晋升阈值定义
            thresholds = {
                'acquaintance': {'closeness': 0.2},
                'friend': {'closeness': 0.35, 'trust': 0.3},
                'close_friend': {'closeness': 0.5, 'trust': 0.5, 'respect': 0.4},
                'romantic': {'closeness': 0.6, 'attraction': 0.65, 'trust': 0.5},
                'partner': {'closeness': 0.8, 'attraction': 0.8, 'trust': 0.8, 'security': 0.7}
            }
            t = thresholds.get(next_stage)
            if t and all(new_vals[k] >= v for k, v in t.items()):
                new_stage = next_stage

        # --- 6. 持久化 ---
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE relationship_states 
                   SET closeness=%s, trust=%s, resentment=%s, dependency=%s, attraction=%s, 
                       respect=%s, security=%s, jealousy=%s, momentum=%s, stage=%s, 
                       narrative=COALESCE(%s, narrative), last_updated=CURRENT_TIMESTAMP
                   WHERE character_id=%s""",
                (new_vals['closeness'], new_vals['trust'], new_vals['resentment'], 
                 new_vals['dependency'], new_vals['attraction'], new_vals['respect'], 
                 new_vals['security'], new_vals['jealousy'], new_momentum, new_stage, 
                 new_narrative, character_id)
            )
            conn.commit()

    def get_relationship_description(self, character_id):
        """将数值、动量及叙事摘要转换为增强型 Prompt"""
        s = self.get_relationship_state(character_id)
        if not s: return ""

        def judge(val): return "high" if val > 0.7 else ("low" if val < 0.3 else "mid")
        momentum_desc = "Heating up (positive trend)" if s['momentum'] > 0.1 else ("Cooling down (negative trend)" if s['momentum'] < -0.1 else "Stable")
        
        desc = (
            f"[Deep Relationship Model]\n"
            f"Current Stage: {s['stage']} | Trend: {momentum_desc}\n"
            f"Narrative: {s['narrative']}\n"
            f"Attachment Style: {s['attachment_style']}\n"
            f"Key Markers: Closeness({judge(s['closeness'])}), Trust({judge(s['trust'])}), Attraction({judge(s['attraction'])}), Security({judge(s['security'])}), Jealousy({judge(s['jealousy'])})"
        )
        return desc

    def get_recent_episodic_ids(self, character_id, limit=5):
        """获取最近的几个情景记忆 ID 及其简述，用于建立因果链"""
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, content FROM episodic_memories WHERE character_id = %s ORDER BY created_at DESC LIMIT %s",
                (character_id, limit)
            )
            return cur.fetchall()

    def search_episodic_memories(self, character_id, query_embedding, limit=5, time_start=None, time_end=None):
        """向量检索最相关的情景记忆，支持时间范围过滤"""
        conn = self.connect()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if time_start and time_end:
                    # 指定时间范围：按纯相似度排序，不使用时间衰减
                    cur.execute(
                        """SELECT id, content, emotion_intensity, event_time, emotion_category, causal_link_id,
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
                        """SELECT id, content, emotion_intensity, event_time, emotion_category, causal_link_id,
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
    # --- 提醒任务方法开始 ---

    def add_reminder(self, character_id, user_id, task_content, remind_at):
        """保存提醒任务"""
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO reminders (character_id, user_id, task_content, remind_at) 
                       VALUES (%s, %s, %s, %s)""",
                    (character_id, user_id, task_content, remind_at)
                )
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise e

    def get_due_reminders(self):
        """获取到期需发送的提醒"""
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT id, character_id, user_id, task_content 
                   FROM reminders 
                   WHERE status = 'pending' AND remind_at <= NOW() 
                   ORDER BY remind_at ASC"""
            )
            return cur.fetchall()

    def mark_reminder_sent(self, reminder_id):
        """标记提醒已发送"""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE reminders SET status = 'sent' WHERE id = %s",
                (reminder_id,)
            )
            conn.commit()
    # --- 提醒任务方法结束 ---
