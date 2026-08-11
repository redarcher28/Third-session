# 证据智能助手 MVP

OpenEvidence 风格的三赛道证据助手：临床证据助手、健康营养助手、RAG vs 通用大模型对比评测。

> **伦理边界**：仅供学习与演示，**不构成医疗建议**，不用于真实诊疗，不处理真实患者隐私。引用请人工复核。

## 团队协作文档

任务分配、流程图、时序图、架构与函数接口清单见：

- [docs/团队任务与架构说明.md](docs/团队任务与架构说明.md)

**待完善接口已分散到各业务模块文件末尾**（格式：函数签名 + 中文备注 + `NotImplementedError`），不再使用集中式 stubs 文件。认领时直接打开对应 `.py` 搜索「【待完善】」。

## 功能概览

| 赛道 | 说明 |
|------|------|
| 一 · 临床 | 专业结构化回答 + 证据面板 + 引用校验 |
| 二 · 营养 | 通俗科普改写 + 同一知识库可追溯引用 |
| 三 · 评测 | Baseline（纯 LLM）vs RAG 假引用/覆盖率对比 |

共用底座：PubMed / ClinicalTrials.gov / Europe PMC / 本地种子摘要 → 切分 → Chroma → 混合检索 → 带引用生成。

赛道一 / 二支持 **多轮对话**：像 DeepSeek / ChatGPT 一样连续追问，历史上下文会自动带入查询改写、证据检索与回答生成。

## 快速开始

```bash
cd evidence-assistant-mvp
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Linux: cp .env.example .env
```

编辑 `.env`，填入 OpenAI 兼容接口：

```env
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
# 可选：Embedding 独立端点（DeepSeek 无 embedding 接口时留空则回退离线哈希）
EMBEDDING_BASE_URL=
EMBEDDING_API_KEY=
```

使用 DeepSeek 时，聊天部分可直接配置：

```env
LLM_API_KEY=sk-你的deepseek-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
EMBEDDING_MODE=offline
```

未配置有效 key 时会进入**离线占位模式**（哈希 embedding + 模板回答），仍可演示全流程。

### 1. 构建知识库

仅本地种子（无外网 API，推荐先跑通）：

```bash
python scripts/build_kb.py --skip-live
```

拉取公开数据源（需网络）：

```bash
python scripts/build_kb.py
```

可将 PDF / Markdown 放入 `data/raw/local/` 后重建知识库。

### 2. 冒烟测试

```bash
python scripts/smoke_demo.py
```

### 3. 启动界面 / API

```bash
# Streamlit 聊天式多轮对话演示
streamlit run src/app/ui.py

# FastAPI
uvicorn src.app.api:app --reload --port 8000
# POST /ask  {"question":"...","track":"clinical","history":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
# POST /eval/run
```

### 4. 跑评测（赛道三）

```bash
python scripts/run_eval.py
```

结果写入 `data/eval/results/benchmark_results.json` 与 `benchmark_summary.md`。

## 演示路径建议

1. **临床 Tab**：问「高血压为什么要长期吃药」→ 展示证据卡片与 `[n]` 引用 → 继续追问「那饮食上要注意什么？」观察多轮上下文衔接。
2. **营养 Tab**：问「地中海饮食证据」→ 对比更通俗的表述，引用仍可点开 → 追问「那我每天吃多少盐更合适？」验证产品边界与多轮语境。
3. **评测 Tab**：运行评测 → 看假引用率 / 要点覆盖柱状图 → 打开「火星尘埃」「紫水晶」等应拒答 case。

## 项目结构

```
evidence-assistant-mvp/
  src/ingest/      # 数据采集
  src/kb/          # 切分、Chroma、Wiki
  src/retrieval/   # 混合检索 + 重排
  src/generation/  # 带引用生成 / Baseline
  src/tools/       # 引用校验、在线补检索
  src/tracks/      # 三赛道逻辑与评测
  src/app/         # FastAPI + Streamlit
  scripts/         # build_kb / run_eval / smoke_demo
  data/eval/       # 测试集
```

## 数据来源

- PubMed E-utilities
- ClinicalTrials.gov Data API v2
- Europe PMC REST
- 本地种子摘要（`src/ingest/local_docs.py`）与 `data/raw/local/`

## 局限与后续

- 语料为窄领域演示集，非全科覆盖。
- 离线模式回答为占位，正式演示请配置真实 LLM/Embedding。
- 未做生产级权限、审计与患者数据隔离。
- 后续可加强：重排模型、Ragas 全量指标、指南 PDF 批量入库、MCP 工具封装。
