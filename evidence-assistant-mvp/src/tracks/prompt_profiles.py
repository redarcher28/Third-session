# -*- coding: utf-8 -*-
"""
赛道一 / 赛道二的统一 Prompt 配置。

本模块把 RAG 的三个 Prompt 面统一管理：

1. query reformulation：把用户问题改写成可检索查询；
2. grounded system：规定角色、证据边界、缺失信息和安全边界；
3. synthesis：规定如何从编号证据中综合回答并放置引用。

这样前端只切换 track，后端仍然沿用同一条「改写 → 检索 → 生成 → 校验」
链路，避免赛道规则散落在 UI 和不同业务函数里。

设计依据：SurePrompts《RAG Prompt Engineering Guide (2026)》提出的三层
Prompt 架构，以及 OpenEvidence 风格 MVP 对“问题 → 证据 → 带引用回答”的要求。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TrackKey = Literal["clinical", "nutrition"]

PROMPT_VERSION = "rag-stack-2026.08.11-v1"
PROMPT_LAYERS = (
    "query_reformulation",
    "grounded_system",
    "synthesis",
    "citation_validation",
)


@dataclass(frozen=True)
class TrackPromptProfile:
    """一个赛道的产品文案、检索偏好和 Prompt 约束。"""

    key: TrackKey
    label: str
    audience: str
    description: str
    language_contract: str
    output_contract: str
    forbidden_contract: str
    sample_questions: tuple[str, ...]
    evidence_focus: tuple[str, ...]
    prefer_levels: tuple[str, ...]
    boost_tags: tuple[str, ...]
    query_prompt: str
    persona: str
    style: str
    dosage_guard: bool = False


_COMMON_QUERY_RULES = """
你负责 RAG 的 query reformulation 层，不负责回答用户问题。
把原问题改写成一条适合医学/健康证据库检索的查询：
- 保留疾病或健康主题、干预/暴露、比较对象、结局和人群等核心概念；
- 把口语同义词补成可检索的专业关键词，可在中文后补英文术语；
- 对多部分问题保留主要子问题，不凭训练知识补充结论；
- 只输出一行查询文本，不要解释、不要编号、不要 Markdown。
""".strip()


CLINICAL_QUERY_PROMPT = f"""
{_COMMON_QUERY_RULES}

当前赛道：临床证据助手。优先显式保留 PICO 结构，并加入 guideline、RCT、
systematic review 或 meta-analysis 等证据类型词，方便后续优先召回指南、荟萃
和随机对照试验。
""".strip()


NUTRITION_QUERY_PROMPT = f"""
{_COMMON_QUERY_RULES}

当前赛道：健康营养助手。把消费者的口语问题归一成饮食/营养干预、生活方式、
风险因素和健康结局查询；优先保留 diet、nutrition、lifestyle、cardiovascular
risk 等可检索概念。不要把问题改写成药物剂量或处方问题。
""".strip()


CLINICAL_PERSONA = """
你是“临床证据助手”，服务对象是临床医生、医学生和科研人员。
你的工作是把检索到的指南、随机对照试验、系统综述和其他公开证据整理成可复核的
证据概览，而不是代替临床决策。回答要区分研究发现、适用人群、结局和局限。
""".strip()


NUTRITION_PERSONA = """
你是“健康营养助手”，服务对象是希望理解公开健康证据的普通消费者。
你的工作是把检索到的专业资料解释成清楚、克制、可理解的健康科普；优先谈饮食和
生活方式，不把相关性说成因果，不把群体研究结果包装成个人保证。
""".strip()


CLINICAL_STYLE = """
使用专业但清晰的中文，建议按以下顺序组织：结论 → 证据概览 → 关键来源/研究 →
适用人群与局限。可以使用“指南”“RCT”“荟萃分析”“结局”等术语，并在必要时解释。
不要给个体化处方、诊断、停药或具体剂量建议。
""".strip()


NUTRITION_STYLE = """
使用通俗中文，建议按以下顺序组织：通俗结论 → 研究提示 → 你可以关注的生活方式
方向 → 边界与局限。使用“研究提示”“可能有助于”等与证据强度匹配的表达，少用术语，
必要时在括号里解释。不要输出个体化诊疗、药物选择或用药剂量。
""".strip()


CLINICAL_PROFILE = TrackPromptProfile(
    key="clinical",
    label="赛道一 · 临床证据",
    audience="医生 / 医学生 / 研究者",
    description="指南与 RCT 优先的结构化证据概览。",
    language_contract="指南、RCT、结局、适用人群",
    output_contract="证据概览 + 关键来源",
    forbidden_contract="不替代临床决策，不给个体化处方或剂量",
    sample_questions=(
        "高血压患者为什么有时要长期吃药？有哪些指南或研究依据？",
        "体检发现血脂偏高，生活方式干预和药物治疗分别有哪些证据？",
        "DASH 饮食模式对血压的临床试验证据是什么？",
    ),
    evidence_focus=("guideline", "meta", "rct", "wiki", "observational"),
    prefer_levels=("guideline", "meta", "rct", "wiki", "observational"),
    boost_tags=(),
    query_prompt=CLINICAL_QUERY_PROMPT,
    persona=CLINICAL_PERSONA,
    style=CLINICAL_STYLE,
)


NUTRITION_PROFILE = TrackPromptProfile(
    key="nutrition",
    label="赛道二 · 健康营养",
    audience="普通消费者 / 健康科普读者",
    description="生活方式与营养证据的通俗解释，保留边界与来源。",
    language_contract="研究提示、生活方式、局限",
    output_contract="通俗结论 + 边界 + 来源",
    forbidden_contract="不做个体化诊疗，不提供用药剂量",
    sample_questions=(
        "地中海饮食对心血管风险有什么证据？",
        "限钠饮食对高血压是否真的有帮助？",
        "血脂高的人日常吃什么更有证据支持？",
    ),
    evidence_focus=("diet", "lifestyle", "mediterranean", "hypertension", "diabetes"),
    prefer_levels=(),
    boost_tags=(
        "diet",
        "mediterranean",
        "dash",
        "hypertension",
        "hyperlipidemia",
        "diabetes",
        "fiber",
        "plant_based",
        "sugar",
        "ultra_processed",
        "obesity",
        "low_carb",
        "omega3",
    ),
    query_prompt=NUTRITION_QUERY_PROMPT,
    persona=NUTRITION_PERSONA,
    style=NUTRITION_STYLE,
    dosage_guard=True,
)


TRACK_PROFILES: dict[TrackKey, TrackPromptProfile] = {
    "clinical": CLINICAL_PROFILE,
    "nutrition": NUTRITION_PROFILE,
}


def get_track_profile(track: str) -> TrackPromptProfile:
    """获取赛道配置；内部调用遇到未知值时明确报错。"""
    try:
        return TRACK_PROFILES[track]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(f"unsupported track: {track}") from exc


def build_query_messages(track: str, question: str) -> list[dict[str, str]]:
    """构造 query reformulation 层的对话消息。"""
    profile = get_track_profile(track)
    return [
        {"role": "system", "content": profile.query_prompt},
        {"role": "user", "content": question.strip()},
    ]


def build_synthesis_messages(
    track: str,
    question: str,
    context_block: str,
    *,
    system_persona: str | None = None,
    answer_style: str | None = None,
) -> list[dict[str, str]]:
    """
    构造 grounded system + synthesis 层消息。

    引用采用现有 MVP 的 ``[n]`` 编号协议：编号与证据面板一一对应，既能让用户
    点击/核对来源，也方便后端做越界引用和假 PMID/NCT 校验。
    """
    profile = get_track_profile(track)
    persona = system_persona or profile.persona
    style = answer_style or profile.style
    system = f"""
{persona}

## GROUNDING / 证据边界
- 只能依据下方 RETRIEVED EVIDENCE 作答；不要用训练知识填补证据空白。
- 每个事实性主张都必须能回溯到一个或多个证据编号。
- 证据只部分回答问题时，明确区分“证据支持的部分”和“当前未覆盖的部分”。
- 如果没有足够证据，直接说明不知道/证据不足，不要为了完整而猜测。

## CITATION / 引用纪律
- 只使用 [n] 形式引用，n 必须是下方证据实际存在的编号。
- 把 [n] 放在对应事实或段落末尾；一条主张使用多条证据时并列写 [n][m]。
- 禁止编造论文标题、作者、PMID、NCT、URL 或不存在的编号。
- 文末只列出实际使用过的编号，不要新增无法从证据列表核对的来源。

## CONFLICTS / 冲突处理
- 来源结论不一致时并列呈现差异，说明证据类型、年份或适用人群差别。
- 不要静默地把冲突修成一个确定结论；证据等级更高或更新的来源可作为排序依据，
  但仍需保留冲突说明。

## DOMAIN CONTRACT / 赛道输出契约
{style}
{profile.forbidden_contract}
""".strip()
    user = f"""
## USER QUESTION
{question.strip()}

## RETRIEVED EVIDENCE
在回答前先判断哪些证据与问题直接相关；不要把无关片段当作支持。不要展示内部推理
过程，只输出最终回答。

{context_block}

请按赛道输出契约回答，并为每个事实性结论添加可核对的 [n] 引用。
""".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_rerank_messages(query: str, candidates: list[dict]) -> list[dict[str, str]]:
    """构造候选证据重排消息，保持检索层 Prompt 的单一职责。"""
    lines = []
    for i, candidate in enumerate(candidates, start=1):
        lines.append(
            f"[{i}] {candidate.get('title', '')}\n"
            f"level={candidate.get('evidence_level', 'other')} "
            f"source={candidate.get('source', '')}\n"
            f"{(candidate.get('text') or '')[:320]}"
        )
    return [
        {
            "role": "system",
            "content": (
                "你是证据检索重排器。只按用户查询与候选文本的直接相关性排序；"
                "优先保留能直接支持问题的候选，不依据训练知识补判断。"
                "只输出逗号分隔的候选编号，例如 3,1,5,2，不要解释。"
            ),
        },
        {
            "role": "user",
            "content": f"用户查询：{query}\n\n候选证据：\n" + "\n\n".join(lines),
        },
    ]


def public_track_configs() -> dict:
    """返回给前端的赛道配置，不暴露完整系统 Prompt。"""
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_layers": list(PROMPT_LAYERS),
        "tracks": [
            {
                "key": profile.key,
                "label": profile.label,
                "audience": profile.audience,
                "description": profile.description,
                "language_contract": profile.language_contract,
                "output_contract": profile.output_contract,
                "forbidden_contract": profile.forbidden_contract,
                "sample_questions": list(profile.sample_questions),
                "evidence_focus": list(profile.evidence_focus),
                "dosage_guard": profile.dosage_guard,
            }
            for profile in TRACK_PROFILES.values()
        ],
    }
