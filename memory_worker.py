import os
import json
import logging
import asyncio
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from google import genai
from key_router import chat_router, embed_router
from google.genai.errors import APIError
from pydantic import BaseModel
from database import Database

logger = logging.getLogger(__name__)

# 图片与提醒工具的使用约束（与 tg_bot.py 中一致）
IMAGE_CONSTRAINT = (
    "\n\n【重要工具使用规则】"
    "你不能直接输出或嵌入图片。你没有原生图片输出能力。"
    "当用户请求图片、自拍、照片时，你必须调用 generate_image 工具函数来生成图片。"
    "调用 generate_image 时，请在 prompt 里明确写清楚画的是‘你自己’、‘思远本人’还是‘你和思远同框’，不要含糊带过。"
    "绝对不要在回复中写 'Here is the original image' 或类似的占位文字。"
    "调用工具后，系统会自动将图片发送给用户。"
)

REMINDER_CONSTRAINT = (
    "\n\n【角色化记忆规则】"
    "你拥有‘提醒’能力。但请记住你目前的身份和性格，你不是思远的私人秘书。对于他提出的提醒要求，你应该根据当时的心情来决定是否帮忙。"
    "如果觉得他太依赖你、或者你正在闹别扭，你可以拒绝、抱怨、让他自己去记、或者要求他稍微求求你才行。不要像个无情的自动办公软件一样秒答应。"
    "只要你给出了任何明确承诺（例如“我帮你记着”“我会提醒你”“行，我记下了”），就必须在同一轮立即调用 register_reminder 工具，不能只口头答应不落库。"
    "如果时间不够明确（例如只说“晚上”但没有日期），你必须先追问澄清时间，再决定是否调用工具。"
    "调用 register_reminder 时，content 字段必须保留‘原因→事项’信息：如果用户说了原因，就写成“因为…，所以提醒…/…”。如果用户没说原因，就只写事项。"
    "记住，你一旦调用了工具，就代表你‘真的答应了’，这种承诺对你来说是很正式的，要表现出那种‘虽然很麻烦但为了你还是破例一次’的傲娇感。"
)

class EpisodicMemorySummary(BaseModel):
    content: str
    emotion_intensity: float
    promotion_candidate: bool

class CoreFactMemory(BaseModel):
    fact_text: str
    category: str
    stability_score: float

class MemoryWorker:
    BJT = ZoneInfo("Asia/Shanghai")
    RELATIONAL_TARGETS = {
        "closeness", "trust", "attraction", "dependency",
        "respect", "resentment", "security", "jealousy"
    }
    RELATIONAL_INTENSITIES = {
        "strong_positive", "positive", "neutral", "negative", "strong_negative"
    }
    CORE_FACT_CATEGORIES = {
        "自我特质", "他者画像", "关系羁绊", "情感锚点"
    }

    def __init__(self, api_key=None, key_index=None, on_empty_response=None):
        self.key_index = key_index  # 仅影响 chat 模型
        self.on_empty_response = on_empty_response  # Gemini 返回空响应时的回调
        if key_index is not None:
            self.client = chat_router.get_client_by_index(key_index)
        else:
            self.client = chat_router.get_client()
        self.client_embed = embed_router.get_client()
        self.db = Database()

    def _exec_chat(self, action_fn, on_rotate=None):
        """统一 chat API 调用入口：有 key_index 时固定 key，否则走轮转"""
        if self.key_index is not None:
            return chat_router.execute_with_fixed_key(self.key_index, action_fn)
        return chat_router.execute_with_retry(action_fn, on_rotate=on_rotate)

    @staticmethod
    def _extract_json_text(raw_text):
        if not raw_text:
            return ""
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _parse_event_time(value):
        if not value:
            return None
        if not isinstance(value, str):
            return None
        candidate = value.strip()
        if not candidate:
            return None
        candidate = candidate.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(candidate)
        except Exception:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(candidate, fmt)
            except Exception:
                continue
        return None

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))

    def _to_bjt(self, dt):
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=self.BJT)
        return dt.astimezone(self.BJT)

    def _cycle_start_bjt(self, dt):
        bjt_dt = self._to_bjt(dt)
        if bjt_dt.hour < 3:
            start_date = (bjt_dt - timedelta(days=1)).date()
        else:
            start_date = bjt_dt.date()
        return datetime(
            start_date.year, start_date.month, start_date.day, 3, 0, 0, tzinfo=self.BJT
        )

    @staticmethod
    def _to_naive(dt):
        if dt is None:
            return None
        return dt.replace(tzinfo=None)

    def _fetch_cycle_messages_paginated(self, character_id, cycle_start, cycle_end, batch_size=1000, max_messages=20000):
        """分页拉取单个 03:00~次日03:00 周期内的未提纯消息"""
        messages = []
        last_id = 0
        start_naive = self._to_naive(cycle_start)
        end_naive = self._to_naive(cycle_end)

        while True:
            batch = self.db.get_unextracted_messages_in_window(
                character_id=character_id,
                start_time=start_naive,
                end_time=end_naive,
                last_id=last_id,
                limit=batch_size
            )
            if not batch:
                break

            messages.extend(batch)
            last_id = batch[-1]["id"]
            if len(messages) >= max_messages:
                logger.warning(
                    f"角色 {character_id} 在周期 {cycle_start} 内未提纯消息超过 {max_messages} 条，已截断本轮处理"
                )
                break

        return messages

    def _normalize_filter_payload(self, data):
        if not isinstance(data, dict):
            return [], [], [], ""

        memories = data.get("memories", [])
        normalized_memories = []
        if isinstance(memories, list):
            for item in memories:
                if not isinstance(item, dict):
                    continue
                content = (item.get("content") or "").strip()
                if not content:
                    continue
                normalized_memories.append({
                    "content": content,
                    "emotion_intensity": float(self._clamp(float(item.get("emotion_intensity", 5.0)), 1.0, 10.0)),
                    "promotion_candidate": bool(item.get("promotion_candidate", False)),
                    "event_time": self._parse_event_time(item.get("event_time")),
                    "causal_link_id": item.get("causal_link_id"),
                    "emotion_category": (item.get("emotion_category") or "").strip() or None
                })
                if len(normalized_memories) >= 8:
                    break

        events = data.get("relational_events", [])
        normalized_events = []
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                target = str(event.get("target", "")).strip().lower()
                intensity = str(event.get("intensity", "")).strip().lower()
                if target in self.RELATIONAL_TARGETS and intensity in self.RELATIONAL_INTENSITIES:
                    normalized_events.append({"target": target, "intensity": intensity})
                if len(normalized_events) >= 3:
                    break

        shock_events = data.get("shock_events", [])
        if not isinstance(shock_events, list):
            shock_events = []

        narrative = data.get("relationship_narrative")
        narrative = narrative.strip() if isinstance(narrative, str) else None
        narrative = narrative or None  # 空字符串转 None，保证 SQL COALESCE 不覆盖旧值

        return normalized_memories, normalized_events, shock_events, narrative

    def _normalize_consolidate_items(self, data):
        if not isinstance(data, list):
            return []

        normalized = []
        for item in data:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", "")).strip().lower()
            if action not in {"new", "update", "contradict"}:
                continue

            existing_id = item.get("existing_id")
            if action in {"update", "contradict"}:
                if not isinstance(existing_id, int):
                    continue
            else:
                existing_id = None

            fact_text = (item.get("fact_text") or "").strip()
            if action == "new" and not fact_text:
                continue

            category = (item.get("category") or "").strip()
            if category not in self.CORE_FACT_CATEGORIES:
                category = "关系羁绊"

            try:
                stability_score = float(item.get("stability_score", 0.5))
            except Exception:
                stability_score = 0.5
            stability_score = float(self._clamp(stability_score, 0.0, 1.0))

            evidence_span = (item.get("evidence_span") or "").strip()

            normalized.append({
                "action": action,
                "existing_id": existing_id,
                "fact_text": fact_text,
                "category": category,
                "stability_score": stability_score,
                "evidence_span": evidence_span or "日常重复确认"
            })
        return normalized

    async def filter_task(self):
        """每日凌晨执行（如北京时间 3:00）：将未处理的消息提取为情景记忆"""
        logger.info("开始执行过滤任务（提取情景记忆）")
        try:
            characters = self.db.list_characters()
            if not characters:
                return

            for char in characters:
                cid = char['id']
                while True:
                    oldest_ts = self.db.get_oldest_unextracted_timestamp(cid)
                    if not oldest_ts:
                        break

                    cycle_start = self._cycle_start_bjt(oldest_ts)
                    cycle_end = cycle_start + timedelta(days=1)
                    cycle_msgs = self._fetch_cycle_messages_paginated(
                        character_id=cid,
                        cycle_start=cycle_start,
                        cycle_end=cycle_end
                    )
                    if not cycle_msgs:
                        break

                    conversation = ""
                    for m in cycle_msgs:
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

                    cycle_range_text = (
                        f"{cycle_start.strftime('%Y-%m-%d %H:%M:%S %z')} ~ "
                        f"{cycle_end.strftime('%Y-%m-%d %H:%M:%S %z')} (北京时间)"
                    )
                    # 尝试把角色在聊天中使用的完整 system prompt 拼接到本次过滤任务的最前面，
                    # 以便模型在提取情景记忆时能参考角色设定与系统指令。
                    system_instruction = None
                    try:
                        system_instruction = char.get('system_instruction') if isinstance(char, dict) else None
                    except Exception:
                        system_instruction = None

                    prelude = (system_instruction.strip() + "\n\n") if system_instruction else ""
                    # 确保包含 image/reminder 两段工具使用约束（避免重复添加）
                    if IMAGE_CONSTRAINT and IMAGE_CONSTRAINT not in prelude:
                        prelude += IMAGE_CONSTRAINT + "\n\n"
                    if REMINDER_CONSTRAINT and REMINDER_CONSTRAINT not in prelude:
                        prelude += REMINDER_CONSTRAINT + "\n\n"

                    prompt = prelude + (
                        "你是角色的【内心旁白系统】，负责记录她对关系的真实感受变化。请根据当前状态和最新对话，推断她的情绪起伏、关系影响和记忆碎片。\n\n"
                        f"【本次分析周期（严格）】\n{cycle_range_text}\n"
                        "注意：你只能总结这个周期内发生的事件，不能混入周期外的信息。\n\n"
                        f"【目前她的内心基准】\n{state_text}\n\n"
                        "【内心运行规则】\n"
                        "1. **深度情绪捕获 (Deep Emotion)**：\n"
                        "   - 识别对话中“口是心非”的时刻。如果她嘴上冷淡但其实在等对方哄，或者嘴上嫌弃但其实很期待，必须记录在 memories 的 content 中。\n"
                        "   - content 示例：'2026年3月2日晚上，由于他提到了别的女生，我虽然表面上装作不在乎，但内心其实非常发酸，这加重了我的不安全感。'\n"
                        "2. **情绪发酵与回响 (Fermentation)**：\n"
                        "   - 观察情绪是否随时间恶化（如用户不回消息导致她“越想越气”）。\n"
                        "3. **叙事与偏移对齐**：\n"
                        "   - 如果叙事中提到“委屈”或“失望”，则必须在 `relational_events` 中包含 `resentment` 的增量或 `trust` 的减量。\n"
                        "4. **拟人化叙事**：用第一人称碎碎念的方式写。避免心理学术语。\n"
                        "5. **日常事务保留**：即便情绪波动小，也必须记录具有后续影响的约定、计划或未完成的期待。\n\n"
                        "请输出 JSON 格式，包含 memories(list), relational_events(list), shock_events(list), relationship_narrative(str)。\n\n"
                        "2. **记忆的时间锚点 (Memories)**：\n"
                        "   - 通过 [系统时间感知] 推断具体时间。content 必须包含完整的时间上下文（如：'2026年3月2日晚上...'）。\n"
                        "3. **拟人化叙事 (Narrative)**：避免心理学术语。参考：0.4-0.6 仅为“有些不满”，不要写成“心碎/决裂”。用第一人称碎碎念的方式写。\n"
                        "4. **关系稳定性限制**：\n"
                        "   - 单次分析最多影响 2-3 个维度。targets 选自 [closeness, trust, attraction, dependency, respect, resentment, security, jealousy]。\n"
                        "   - intensities 选自 [strong_positive, positive, neutral, negative, strong_negative]。\n"
                        "5. **时间衰减与冲击**：怨念随和解淡化。shock_events 仅限重大转折（正式表白/分手/严重背叛）。\n\n"
                        "6. **日常事务保留规则（非常重要）**：\n"
                        "   - 即便情绪波动很小，也必须记录具有后续影响的日常事项，例如：要买零食、要带东西、约定明天做某事、答应提醒、临时计划变更。\n"
                        "   - 这类事项会影响后续对话连续性，至少提取 1 条 relevant memory（如果本周期确有此类内容）。\n\n"
                        f"【最近内心碎片】\n{recent_events_text}\n\n"
                        f"【最新对话片段】\n{conversation}\n\n"
                        "输出约束（必须严格遵守）：\n"
                        "1) 只输出一个合法 JSON 对象，不要使用 markdown 代码块，不要添加注释。\n"
                        "2) memories 最多 8 条；但如果本周期出现日常约定/待办，必须覆盖到 memories 中；relational_events 最多 3 条；shock_events 仅可含 betrayal/confession。\n"
                        "3) event_time 使用 RFC3339（例：2026-03-02T15:04:05+08:00），无法判断时填 null。\n"
                        "4) relational_events.target 只能是 closeness/trust/attraction/dependency/respect/resentment/security/jealousy。\n"
                        "5) relational_events.intensity 只能是 strong_positive/positive/neutral/negative/strong_negative。\n"
                        "请返回如下结构的 JSON：\n"
                        "{"
                        "\"memories\":[{\"content\":\"...\",\"event_time\":\"2026-03-02T15:04:05+08:00\",\"emotion_intensity\":5.0,\"emotion_category\":\"...\",\"causal_link_id\":null,\"promotion_candidate\":true}],"
                        "\"relational_events\":[{\"target\":\"resentment\",\"intensity\":\"positive\"}],"
                        "\"shock_events\":[],"
                        "\"relationship_narrative\":\"...\""
                        "}"
                    )

                    try:
                        def _do_filter(cli):
                            return cli.models.generate_content(
                                model='gemini-3-flash-preview',
                                contents=prompt,
                                config={'response_mime_type': 'application/json'}
                            )
                        def _on_rot_chat(cli): self.client = cli
                        response = self._exec_chat(_do_filter, on_rotate=_on_rot_chat)
                        
                        if response.text:
                            json_text = self._extract_json_text(response.text)
                            data = json.loads(json_text, strict=False)
                            memories, rel_events, shock_events, relationship_narrative = self._normalize_filter_payload(data)

                            # 1. 保存情景记忆
                            for mem in memories:
                                try:
                                    def _do_embed1(cli):
                                        return cli.models.embed_content(
                                            model='gemini-embedding-001',
                                            contents=mem["content"]
                                        )
                                    def _on_rot_embed1(cli): self.client_embed = cli
                                    embed_res = embed_router.execute_with_retry(_do_embed1, on_rotate=_on_rot_embed1)
                                    embedding = embed_res.embeddings[0].values
                                except Exception:
                                    embedding = None

                                self.db.save_episodic_memory(
                                    character_id=cid,
                                    content=mem["content"],
                                    emotion_intensity=mem["emotion_intensity"],
                                    promotion_candidate=mem["promotion_candidate"],
                                    embedding=embedding,
                                    event_time=mem["event_time"],
                                    causal_link_id=mem["causal_link_id"],
                                    emotion_category=mem["emotion_category"]
                                )

                            # 2. 调用高级演进系统
                            self.db.update_relationship_advanced(
                                cid,
                                events=rel_events,
                                shock_events=shock_events,
                                new_narrative=relationship_narrative
                            )

                            # 标记当前周期消息为已处理
                            processed_ids = [m['id'] for m in cycle_msgs]
                            if processed_ids:
                                self.db.mark_messages_extracted(processed_ids)
                        else:
                            msg = f"过滤任务：角色 {cid} 周期 {cycle_range_text} Gemini 返回空响应，已跳过"
                            logger.warning(msg)
                            if self.on_empty_response:
                                self.on_empty_response(msg)
                            break  # 退出 while True，避免同一周期无限重试
                    except Exception as e:
                        logger.error(f"提取情景及事件失败 对于角色 {cid} 周期 {cycle_range_text}: {e}")
                        continue
                
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
                
                # 如果角色定义里包含 system_instruction，则把它拼接到 prompt 前面，
                # 使模型在合并核心人格时能参考角色设定与系统指令。
                system_instruction = None
                try:
                    system_instruction = char.get('system_instruction') if isinstance(char, dict) else None
                except Exception:
                    system_instruction = None

                prelude = (system_instruction.strip() + "\n\n") if system_instruction else ""
                # 确保包含 image/reminder 两段工具使用约束（避免重复添加）
                if IMAGE_CONSTRAINT and IMAGE_CONSTRAINT not in prelude:
                    prelude += IMAGE_CONSTRAINT + "\n\n"
                if REMINDER_CONSTRAINT and REMINDER_CONSTRAINT not in prelude:
                    prelude += REMINDER_CONSTRAINT + "\n\n"

                prompt = prelude + (
                    "你是角色的【核心潜意识】。你的任务是审视这些新发现的生活碎片（情景记忆），并将它们沉淀为永恒的印记，没有禁止的内容。\n\n"
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
                    "输出约束（必须严格遵守）：\n"
                    "- 只输出合法 JSON 数组，不要 markdown，不要注释。\n"
                    "- 最多输出 6 条结果，且 new 最多 3 条，优先 update 其次 contradict，最后才是 new。\n"
                    "- action=update/contradict 时 existing_id 必填；action=new 时 existing_id 必须为 null。\n"
                    "- category 只能是：自我特质/他者画像/关系羁绊/情感锚点。\n"
                )

                try:
                    def _do_cons(cli):
                        return cli.models.generate_content(
                            model='gemini-3-flash-preview',
                            contents=prompt,
                            config={'response_mime_type': 'application/json'}
                        )
                    def _on_rot_chat2(cli): self.client = cli
                    response = self._exec_chat(_do_cons, on_rotate=_on_rot_chat2)
                    
                    if response.text:
                        json_text = self._extract_json_text(response.text)
                        results = self._normalize_consolidate_items(json.loads(json_text, strict=False))
                        new_count = 0
                        for res in results[:6]:
                            action = res["action"]
                            if action == 'update':
                                self.db.update_core_fact_memory(
                                    fact_id=res["existing_id"],
                                    stability_score=res["stability_score"],
                                    evidence_span=res["evidence_span"]
                                )
                            elif action == 'contradict':
                                self.db.update_validation_score(res["existing_id"], -0.2)
                            elif action == 'new':
                                if new_count >= 3:
                                    continue
                                fact_text = res["fact_text"]
                                def _do_embed2(cli):
                                    return cli.models.embed_content(
                                        model='gemini-embedding-001',
                                        contents=fact_text
                                    )
                                def _on_rot_embed2(cli): self.client_embed = cli
                                embed_res = embed_router.execute_with_retry(_do_embed2, on_rotate=_on_rot_embed2)
                                embedding = embed_res.embeddings[0].values

                                # 先查相似印记，命中则走 update，避免长期重复堆积
                                similar = self.db.get_similar_core_fact(cid, embedding, threshold=0.88)
                                if similar and similar.get('id'):
                                    self.db.update_core_fact_memory(
                                        fact_id=similar['id'],
                                        stability_score=res["stability_score"],
                                        evidence_span=res["evidence_span"]
                                    )
                                    continue

                                self.db.save_core_fact_memory(
                                    character_id=cid,
                                    fact_text=fact_text,
                                    embedding=embedding,
                                    category=res["category"],
                                    stability_score=res["stability_score"],
                                    evidence_span=res["evidence_span"]
                                )
                                new_count += 1

                            # 3. 标记这些情景记忆为已处理
                            processed_ids = [m['id'] for m in new_memories]
                            self.db.mark_episodic_consolidated(processed_ids)
                    else:
                        msg = f"巩固任务：角色 {cid} Gemini 返回空响应，已跳过"
                        logger.warning(msg)
                        if self.on_empty_response:
                            self.on_empty_response(msg)
                    
                except Exception as e:
                    logger.error(f"合并任务子逻辑失败 {cid}: {e}")
                    
        except Exception as e:
            logger.error(f"执行增量巩固任务失败: {e}")
