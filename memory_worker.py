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
                    "1. **叙事与偏移对齐 (Consistency)**：\n"
                    "   - *重要*：你的 `relationship_narrative` 必须与 `relational_events` 数值完全对应。\n"
                    "   - 如果在叙事中提到“愤怒”、“委屈”、“恨”或“失望”，则**必须**在 `relational_events` 中包含 `resentment` 的增量或 `trust` 的减量。\n"
                    "2. **记忆的时间锚点 (Memories)**：\n"
                    "   - 通过 [系统时间感知] 推断具体时间。content 必须包含完整的时间上下文（如：'2026年3月2日晚上...'）。\n"
                    "3. **拟人化叙事 (Narrative)**：避免心理学术语。参考：0.4-0.6 仅为“有些不满”，不要写成“心碎/决裂”。用第一人称碎碎念的方式写。\n"
                    "4. **关系稳定性限制**：\n"
                    "   - 单次分析最多影响 2-3 个维度。targets 选自 [closeness, trust, attraction, dependency, respect, resentment, security, jealousy]。\n"
                    "   - intensities 选自 [strong_positive, positive, neutral, negative, strong_negative]。\n"
                    "5. **时间衰减与冲击**：怨念随和解淡化。shock_events 仅限重大转折（正式表白/分手/严重背叛）。\n\n"
                    f"【最新对话片段】\n{conversation}\n\n"
                    "请以 JSON 格式返回她当下的内心报告：\n"
                    "{\n"
                    "  \"memories\": [{ \"content\": \"...\", \"event_time\": \"...\", \"emotion_intensity\": 1-10, \"emotion_category\": \"...\" }],\n"
                    "  \"relational_events\": [{ \"target\": \"resentment\", \"intensity\": \"positive\" }],\n"
                    "  \"shock_events\": [],\n"
                    "  \"relationship_narrative\": \"用内心独白的口吻写一句感性总结...\",\n"
                    "  \"short_term_mood\": { \"mood\": 0.3, \"anger\": 0.5, \"affection\": 0.5, \"jealousy\": 0.0, \"sadness\": 0.6, \"anxiety\": 0.4 }\n"
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
                
                # 组装 Prompt：从“心理学家”转变为“角色的潜意识”
                existing_text = "\n".join([f"ID:{f['id']} | {f['fact_text']} (深度: {f['stability_score']})" for f in existing_facts]) or "目前还是一片空白"
                new_events_text = "\n".join([f"- {m['content']} (情感强度: {m['emotion_intensity']})" for m in new_memories])
                
                prompt = (
                    "你是角色的【核心潜意识】。你的任务是审视这些新发现的生活碎片（情景记忆），并将它们沉淀为永恒的印记。\n\n"
                    "【已有的核心印记】\n"
                    f"{existing_text}\n\n"
                    "【新发生的生活片段】\n"
                    f"{new_events_text}\n\n"
                    "任务指令：\n"
                    "分析这些碎片是否改变了你对世界、对他或对自己的深刻看法。请输出一个 JSON 列表，每个对象包含：\n"
                    "1. action:\n"
                    "   - 'update': 某个已有的认知印记被加强或深化了 (需 ID)。\n"
                    "   - 'new': 产生了一个全新的、值得铭记的深层感知或生活习惯。\n"
                    "   - 'contradict': 现实打破了你以前对他的某些偏见或认知 (需 ID)。\n"
                    "2. fact_text: 极其感性且简练的描述。例如：'他虽然总是在忙，但早晨的拥抱从不缺席'、'我发现自己其实非常渴望被他肯定'。\n"
                    "3. category: 必须选自：\n"
                    "   - 【自我特质】: 我是一个怎样的人？（如：容易感到被遗觉、在感情中很主动）\n"
                    "   - 【他者画像】: 他是一个怎样的人？（如：工作优先级极高、偶尔会愧疚的直男）\n"
                    "   - 【关系羁绊】: 我们的默契与习惯？（如：早起拥抱的仪式、通过撒娇换取关心的套路）\n"
                    "   - 【情感锚点】: 那些特定的情感开关。（如：换新裙子被夸产生的巨大喜悦）\n"
                    "4. stability_score: (0.0-1.0) 该认知对灵魂的触动程度。\n"
                    "5. evidence_span: 对这次内心演变的感性自我陈述。\n"
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
                                # 强化已有印记
                                self.db.update_core_fact_memory(
                                    fact_id=res['existing_id'],
                                    stability_score=float(res.get('stability_score', 0.5)),
                                    evidence_span=res.get('evidence_span', '日常重复确认')
                                )
                            elif action == 'contradict' and res.get('existing_id'):
                                # 现实打破认知，扣分（信念动摇）
                                self.db.update_validation_score(res['existing_id'], -0.2)
                            elif action == 'new':
                                # 产生新印记
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
                                        category=res.get('category', '灵魂碎片'),
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
