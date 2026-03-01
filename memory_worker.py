import os
import json
import logging
import asyncio
from datetime import datetime
from google import genai
from pydantic import BaseModel
from database import Database

logger = logging.getLogger(__name__)

class EpisodicMemorySummary(BaseModel):
    content: str
    emotion_intensity: float
    promotion_candidate: bool

class CoreFactMemory(BaseModel):
    fact_text: str
    category: str
    stability_score: float

class MemoryWorker:
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        self.client = genai.Client(api_key=api_key)
        self.db = Database()

    async def filter_task(self):
        """每日凌晨执行（如北京时间 3:00）：将未处理的消息提取为情景记忆"""
        logger.info("开始执行过滤任务（提取情景记忆）")
        try:
            characters = self.db.list_characters()
            if not characters:
                return

            for char in characters:
                cid = char['id']
                # 放宽每次提取的消息上限，由于积攒了一整天，记录可能会比较多
                msgs = self.db.get_unextracted_messages(cid, limit=5000)
                
                if not msgs:
                    continue

                conversation = "\\n".join([f"{m['role']}: {m['content']}" for m in msgs])
                
                prompt = (
                    "分析以下对话，提取其中值得记忆的核心情景和用户状态。\\n"
                    "如果是纯日常寒暄或废话，忽略即可。\\n"
                    f"对话记录：\\n{conversation}\\n\\n"
                    "请以JSON格式返回（数组），每个对象包含：\\n"
                    "- content: 提纯后的事实，例如'用户因为工作压力大而失眠'\\n"
                    "- emotion_intensity: 1-10的情绪评分\\n"
                    "- promotion_candidate: 是否可能反映长期人格（涉及童年、偏好、习惯等为true，纯废话为false）\\n"
                )

                try:
                    # 使用 structured output 提取
                    response = self.client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config={
                            'response_mime_type': 'application/json'
                        }
                    )
                    
                    if response.text:
                        results = json.loads(response.text)
                        for res in results:
                            # 忽略纯废话或明确被拒绝的提升
                            if not res.get('content') or not res.get('promotion_candidate'):
                                continue
                            self.db.save_episodic_memory(
                                character_id=cid,
                                content=res['content'],
                                emotion_intensity=float(res.get('emotion_intensity', 5.0)),
                                promotion_candidate=res.get('promotion_candidate', True)
                            )
                except Exception as e:
                    logger.error(f"提取情景记忆失败 对于角色 {cid}: {e}")
                    continue

                processed_ids = [m['id'] for m in msgs]
                # 标记为已处理
                if processed_ids:
                    self.db.mark_messages_extracted(processed_ids)
                
        except Exception as e:
            logger.error(f"执行过滤任务失败: {e}")

    async def consolidate_task(self):
        """每日执行：将情景记忆巩固为核心长期记忆"""
        logger.info("开始执行巩固任务（提炼核心人格）")
        try:
            char_ids = self.db.get_all_characters_with_episodic()
            for cid in char_ids:
                memories = self.db.get_recent_episodic_memories(cid, days=3) # 根据过去3天评估
                if len(memories) < 3:
                    continue # 事件太少，先不晋升

                events_text = "\\n".join([f"- {m['content']} (情绪值: {m['emotion_intensity']})" for m in memories])
                
                prompt = (
                    "阅读以下最近发生的用户情境记忆事件。\\n"
                    "寻找在这些事件中反复出现的主题、情绪模式或稳定性特质（如发生了3次以上类似的事情）。\\n"
                    f"事件记录：\\n{events_text}\\n\\n"
                    "如果有可以抽象为核心人格特质的发现，请以JSON格式返回（数组），每个对象包含：\\n"
                    "- fact_text: 高度抽象的结论，例如'用户存在持续的外貌焦虑'\\n"
                    "- category: 类别（如'性格', '偏好', '痛点'）\\n"
                    "- stability_score: 初始稳定分（根据此事发生频率给出，0.0-1.0）\\n"
                    "- evidence_span: 简要记录是根据哪些具体事件得出的\\n"
                )

                try:
                    response = self.client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config={
                            'response_mime_type': 'application/json'
                        }
                    )
                    
                    if response.text:
                        results = json.loads(response.text)
                        for res in results:
                            # 计算向量并存入数据库
                            fact_text = res.get('fact_text')
                            if not fact_text:
                                continue
                                
                            embed_response = self.client.models.embed_content(
                                model='gemini-embedding-001',
                                contents=fact_text
                            )
                            embedding = embed_response.embeddings[0].values
                            
                            self.db.save_core_fact_memory(
                                character_id=cid,
                                fact_text=fact_text,
                                embedding=embedding,
                                category=res.get('category', '一般'),
                                stability_score=float(res.get('stability_score', 0.5)),
                                evidence_span=res.get('evidence_span', '')
                            )
                except Exception as e:
                    logger.error(f"巩固核心记忆失败 对于角色 {cid}: {e}")
                    
        except Exception as e:
            logger.error(f"执行巩固任务失败: {e}")
