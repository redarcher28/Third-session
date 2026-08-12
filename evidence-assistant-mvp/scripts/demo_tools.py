# -*- coding: utf-8 -*-
"""
Tool 五步演示脚本（材料 D2AM 第 14 页）。

演示流程：
    ① 提问（贯穿问题：限钠干预对高血压患者血压管理的高质量证据）
    ② 选择 search_pubmed 并展示调用参数
    ③ 投影前 3 条结果（候选文献 ≠ 已支持结论）
    ④ 调用 verify_citation 核验引用
    ⑤ 把核验通过的元数据交给证据卡模块

故障预案：
    - 网络/MCP 失效 → 自动展示预先缓存的 JSON（cached=true）
    - 演示超时 → 停在第 ③ 步并说明候选文献尚未核验

用法:
    python scripts/demo_tools.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.pubmed_tool import (  # noqa: E402
    format_evidence_card,
    search_pubmed,
    verify_citation,
)

QUESTION = "限钠干预对高血压患者的血压管理，有哪些高质量证据？"


def step(title: str) -> None:
    print(f"\n{'=' * 60}\n【{title}】\n{'=' * 60}")


def main() -> None:
    step("① 提问")
    print(f"贯穿问题：{QUESTION}")

    step("② 选择工具 search_pubmed，展示调用参数")
    params = {
        "query": "sodium reduction hypertension",
        "year_from": 2013,
        "article_type": None,
        "retmax": 5,
    }
    print(f"search_pubmed({json.dumps(params, ensure_ascii=False)})")
    r = search_pubmed(**params)

    step("③ 投影前 3 条结果（候选文献 ≠ 已支持结论）")
    if not r.get("ok"):
        print(f"工具调用失败：{r.get('error')}")
        if r.get("cached"):
            print(f"（故障预案）已切换缓存：{r.get('stale_note')}")
        else:
            print(f"提示：{r.get('hint', '')}")
        print("演示预案：在此停止（第③步），说明网络故障与缓存兜底方案。")
        return
    items = r.get("items", [])
    if not items:
        print("无匹配文献。演示预案：说明检索式与过滤条件，展示空结果处理。")
        return
    for i, it in enumerate(items[:3], start=1):
        print(
            f"  [{i}] PMID {it['pmid']} | {it['year']} | {it['title'][:50]}"
            f"\n      等级={it['evidence_level']} | {it['url']}"
        )
    print(f"（共 {r.get('count')} 条；以上为候选，尚未核验）")

    step("④ 调用 verify_citation 核验引用")
    first = items[0]["pmid"]
    print(f"verify_citation('pmid:{first}')")
    v = verify_citation(f"pmid:{first}")
    if not v.get("ok"):
        print(f"核验失败：{v.get('error')}；预案：展示缓存或说明核验不可用。")
        return
    if v.get("cached"):
        print(f"（缓存）{v.get('stale_note')}")
    print(f"存在={v.get('exists')} | 撤稿提示={v.get('withdrawn')} | {v.get('note')}")

    step("⑤ 生成证据卡（核验通过后）")
    print(format_evidence_card(v.get("metadata", {})))

    print("\n演示结束：候选 → 核验 → 证据卡链路完整。")


if __name__ == "__main__":
    main()
