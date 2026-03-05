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

                conversation = ""
                for m in msgs:
                    prefix = f"{m['context_prefix']} " if m.get('context_prefix') else ""
                    conversation += f"{prefix}{m['role']}: {m['content']}\n"
                
                # 获取最近的事件用于因果链参考
                recent_events = self.db.get_recent_episodic_ids(cid, limit=5)
                recent_events_text = "\n".join([f"ID: {e['id']} | 内容: {e['content']}" for e in recent_events])

                # 获取当前心理基准（状态感知）
                curr_state = self.db.get_relationship_state(cid)
                state_text = "未知"
                if curr_state:
                    state_text = (
                        f"当前阶段: {curr_state.get('stage')} | 依恋风格: {curr_state.get('attachment_style')}\n"
                        f"分值: 亲密({curr_state.get('closeness', 0):.2f}), 信任({curr_state.get('trust', 0):.2f}), 吸引({curr_state.get('attraction', 0):.2f}), "
                        f"依赖({curr_state.get('dependency', 0):.2f}), 尊重({curr_state.get('respect', 0):.2f}), 怨念({curr_state.get('resentment', 0):.2f}), "
                        f"安全({curr_state.get('security', 0):.2f}), 嫉妒({curr_state.get('jealousy', 0):.2f})"
                    )

                prompt = (
                    "你是角色的【内心旁白系统】，负责记录她对关系的真实感受变化。请根据当前状态和最新对话，推断她的情绪起伏、关系影响和记忆碎片。\n\n"
                    f"【目前她的内心基准】\n{state_text}\n\n"
                    "请遵循以下“内心运行规则”进行分析：\n"
                    "1. **记忆的时间锚点 (Memories)**：\n"
                    "   - 必须通过消息中的 [系统时间感知] 标签推断事件的具体发生时间。\n"
                    "   - 【content】必须包含时间上下文（如：'3月2日晚上，用户没有履行承诺...'）。\n"
                    "   - 【event_time】必须填入推断的日期时间字符串。\n"
                    "2. **拟人化叙事 (Narrative)**：避免心理学术语，像真人一样回顾感受。参考分值：\n"
                    "   - 0.0-0.2: 轻松愉快 | 0.2-0.4: 小摩擦 | 0.4-0.6: 有些不满 | 0.6-0.8: 明显矛盾 | 0.8-1.0: 严重冲突。\n"
                    "   - 严禁过度戏剧化，0.6 的怨念只是“挺不爽的”，不是“世界毁灭”。\n"
                    "3. **关系稳定性机制 (Events)**：\n"
                    "   - 单次分析中，【relational_events】最多只能影响 2-3 个维度。\n"
                    "   - targets 必须选自: [closeness, trust, attraction, dependency, respect, resentment, security, jealousy]。\n"
                    "   - intensities 必须选自: [strong_positive, positive, neutral, negative, strong_negative]。\n"
                    "4. **时间衰减与冲击限制**：\n"
                    "   - 怨念会随时间淡化。若表现出和解或日常化，应产生正向事件对冲怨念。\n"
                    "   - 【shock_events】仅在重大转折（正式表白、确立关系、正式分手、严重背叛）时触发。\n\n"
                    f"【最近内心碎片】\n{recent_events_text}\n\n"
                    f"【最新对话片段】\n{conversation}\n\n"
                    "请以 JSON 格式返回她当下的内心报告：\n"
                    "{\n"
                    "  \"memories\": [{ \"content\": \"...\", \"event_time\": \"...\", \"emotion_intensity\": 1-10, \"emotion_category\": \"...\" }],\n"
                    "  \"relational_events\": [{ \"target\": \"trust\", \"intensity\": \"negative\" }],\n"
                    "  \"shock_events\": [],\n"
                    "  \"relationship_narrative\": \"用内心独白的口吻写一句感性总结...\",\n"
                    "  \"short_term_mood\": { \"mood\": 0.6, \"anger\": 0.1, \"affection\": 0.2, \"jealousy\": 0.1, \"sadness\": 0.4, \"anxiety\": 0.3 }\n"
                    "}"
                )

                try:
                    response = self.client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config={'response_mime_type': 'application/json'}
                    )
                    
                    if response.text:
                        data = json.loads(response.text)
                        
                        # 1. 保存情景记忆
                        results = data.get('memories', [])
                        for res in results:
                            content = res.get('content')
                            if not content: continue
                            
                            try:
                                embed_res = self.client.models.embed_content(model='gemini-embedding-001', contents=content)
                                embedding = embed_res.embeddings[0].values
                            except: embedding = None

                            event_time = None
                            if res.get('event_time'):
                                try: event_time = datetime.fromisoformat(res['event_time'])
                                except: pass

                            self.db.save_episodic_memory(
                                character_id=cid,
                                content=content,
                                emotion_intensity=float(res.get('emotion_intensity', 5.0)),
                                promotion_candidate=res.get('promotion_candidate', False),
                                embedding=embedding,
                                event_time=event_time,
                                causal_link_id=res.get('causal_link_id'),
                                emotion_category=res.get('emotion_category')
                            )
                        
                        # 2. 调用高级演进系统
                        self.db.update_relationship_advanced(
                            cid, 
                            events=data.get('relational_events', []),
                            shock_events=data.get('shock_events', []),
                            new_narrative=data.get('relationship_narrative')
                        )
                except Exception as e:
                    logger.error(f"提取情景及事件失败 对于角色 {cid}: {e}")
                    continue

                processed_ids = [m['id'] for m in msgs]
                # 标记为已处理
                if processed_ids:
                    self.db.mark_messages_extracted(processed_ids)
                
        except Exception as e:
            logger.error(f"执行过滤任务失败: {e}")

    async def consolidate_task(self):
        """每日执行：将新产生的碎片情景记忆，增量式地演进为核心长期人格"""
        logger.info("开始执行增量巩固任务...")
        try:
            characters = self.db.list_characters()
            for char in characters:
                cid = char['id']
                # 1. 捞出昨天甚至更早还没被处理的新事件
                new_memories = self.db.get_unconsolidated_episodic_memories(cid, limit=100)
                if not new_memories:
                    continue

                # 2. 捞出目前已有的核心人格画像作为“基准”
                existing_facts = self.db.get_active_core_facts(cid)
                
                # 组装 Prompt
                existing_text = "\n".join([f"ID:{f['id']} | {f['fact_text']} (稳定性: {f['stability_score']})" for f in existing_facts]) or "暂无已知特征"
                new_events_text = "\n".join([f"- {m['content']} (情绪值: {m['emotion_intensity']})" for m in new_memories])
                
                prompt = (
                    "你是一位高级心理学家，负责分析用户的最新动态并演进其核心人格画像。\n\n"
                    "【已知的核心人格特征】\n"
                    f"{existing_text}\n\n"
                    "【新产生的事件碎片】\n"
                    f"{new_events_text}\n\n"
                    "任务：分析新事件对已知特征的影响。请输出一个 JSON 列表，每个对象包含：\n"
                    "1. action: \n"
                    "   - 'update': 强化了已知特征 (需提供 existing_id)。\n"
                    "   - 'new': 发现了全新的、值得长期记住的特征。\n"
                    "   - 'contradict': 新事件反驳了已知特征 (需提供 existing_id)。\n"
                    "2. existing_id: (仅针对 update/contradict)\n"
                    "3. fact_text: (仅针对 new) 简练的特征描述。\n"
                    "4. category: (仅针对 new) '性格'/'偏好'/'习惯'/'痛点'等。\n"
                    "5. stability_score: (0.0-1.0) 该事件反映的特质强度。\n"
                    "6. evidence_span: 对该判断的简短依据。\n"
                )

                try:
                    response = self.client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config={'response_mime_type': 'application/json'}
                    )
                    
                    if response.text:
                        results = json.loads(response.text)
                        for res in results:
                            action = res.get('action')
                            
                            if action == 'update' and res.get('existing_id'):
                                # 强化已有特征
                                self.db.update_core_fact_memory(
                                    fact_id=res['existing_id'],
                                    stability_score=float(res.get('stability_score', 0.5)),
                                    evidence_span=res.get('evidence_span', '日常重复确认')
                                )
                            elif action == 'contradict' and res.get('existing_id'):
                                # 反驳已有特征，扣分
                                self.db.update_validation_score(res['existing_id'], -0.2)
                            elif action == 'new':
                                # 插入新特征
                                fact_text = res.get('fact_text')
                                if fact_text:
                                    embed_res = self.client.models.embed_content(
                                        model='gemini-embedding-001',
                                        contents=fact_text
                                    )
                                    embedding = embed_res.embeddings[0].values
                                    self.db.save_core_fact_memory(
                                        character_id=cid,
                                        fact_text=fact_text,
                                        embedding=embedding,
                                        category=res.get('category', '一般'),
                                        stability_score=float(res.get('stability_score', 0.5)),
                                        evidence_span=res.get('evidence_span', '')
                                    )
                    
                    # 3. 标记这些情景记忆为已处理
                    processed_ids = [m['id'] for m in new_memories]
                    self.db.mark_episodic_consolidated(processed_ids)
                    
                except Exception as e:
                    logger.error(f"合并任务子逻辑失败 {cid}: {e}")
                    
        except Exception as e:
            logger.error(f"执行增量巩固任务失败: {e}")
