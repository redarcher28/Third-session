# 证据智能助手 MVP

OpenEvidence 风格的三赛道证据助手：临床证据助手、健康营养助手、RAG vs 通用大模型对比评测。

> **伦理边界**：仅供学习与演示，**不构成医疗建议**，不用于真实诊疗，不处理真实患者隐私。引用请人工复核。

## 团队协作文档

任务分配、流程图、时序图、架构与函数接口清单见：

- [docs/团队任务与架构说明.md](docs/团队任务与架构说明.md)
- [docs/RAG提示词设计.md](docs/RAG提示词设计.md)

**待完善接口已分散到各业务模块文件末尾**（格式：函数签名 + 中文备注 + `NotImplementedError`），不再使用集中式 stubs 文件。认领时直接打开对应 `.py` 搜索「【待完善】」。

## 功能概览

| 赛道 | 说明 |
|------|------|
| 一 · 临床 | 专业结构化回答 + 证据面板 + 引用校验 |
| 二 · 营养 | 通俗科普改写 + 同一知识库可追溯引用 |
| 三 · 评测 | Baseline（纯 LLM）vs RAG 假引用/覆盖率对比 |

共用底座：PubMed / ClinicalTrials.gov / Europe PMC / 本地种子摘要 → 切分 → Chroma → 混合检索 → 带引用生成。

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

### Python 3.13 环境注意事项（重要）

若本机是 **Python 3.13**，直接 `pip install -r requirements.txt` 会遇到两个坑：

1. `chromadb` 1.x 依赖的 `pybase64` 目前**没有 Python 3.13 兼容版本**，pip 会报 `No matching distribution found`；
2. 最新版 `fastapi`（0.141+）与 chromadb 存在依赖解析冲突，pip 会长时间回溯。

本仓库已验证可用组合（2026-08，Windows / Python 3.13）：

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate

# 1) 先固定兼容版本安装（chromadb 1.5.9 自带预编译向量内核，无需编译器）
pip install "fastapi==0.116.1" "chromadb==1.5.9" "starlette==0.46.1"

# 2) 其余依赖照常
pip install -r requirements.txt

# 3) pybase64 垫片：chromadb 仅使用 b64encode_as_string / b64decode，
#    将纯 Python 垫片复制进虚拟环境即可，无需编译（见 packaging/pybase64.py）
Copy-Item packaging\pybase64.py .venv\Lib\site-packages\pybase64.py   # Windows
# cp packaging/pybase64.py .venv/lib/python3.13/site-packages/pybase64.py  # Linux/macOS
```

> 说明：`chromadb` 1.5.x 仅在包元数据中声明依赖 `pybase64`，代码里实际只用标准库可替代的两个函数，垫片不影响任何功能。`fastapi` 不能低于 0.116、`starlette` 不能低于 0.46，否则新版 streamlit 启动会报 `DEFAULT_EXCLUDED_CONTENT_TYPES` 导入错误。若条件允许，直接安装 **Python 3.12** 也可以绕开上述问题（pybase64 与 chroma-hnswlib 均有 3.12 预编译包）。

编辑 `.env`，填入 OpenAI 兼容接口：

```env
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
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

赛道专项自检（可选）：

```bash
python scripts/check_clinical.py    # 赛道一 · 临床证据助手
python scripts/check_nutrition.py   # 赛道二 · 健康营养助手
```

### 3. 启动界面 / API

```bash
# Streamlit 三 Tab 演示
streamlit run src/app/ui.py

# FastAPI
uvicorn src.app.api:app --reload --port 8000
# POST /ask  {"question":"...","track":"clinical"}
# POST /eval/run
```

### 4. 跑评测（赛道三）

```bash
python scripts/run_eval.py
```

结果写入 `data/eval/results/benchmark_results.json` 与 `benchmark_summary.md`。

## 演示路径建议

1. **临床 Tab**：问「高血压为什么要长期吃药」→ 展示证据卡片与 `[n]` 引用。
2. **营养 Tab**：问「地中海饮食证据」→ 对比更通俗的表述，引用仍可点开。
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
