# -*- coding: utf-8 -*-
"""
PDF 一键入库工具：PDF → Markdown → data/raw/local/。

用法:
    python scripts/ingest_pdf_to_local.py 资料.pdf [更多.pdf ...]
    python scripts/ingest_pdf_to_local.py --dir 某文件夹

流程:
    1. 有文字层 → pymupdf4llm 转 Markdown（保留标题层级，表格不拆散）；
       无文字层（扫描件）→ RapidOCR 逐页识别兜底；
    2. 输出到 data/raw/local/<文件名>.md；
    3. 之后跑 `python scripts/build_kb.py --skip-live` 即可入库——
       超长文档由任务⑧ split_long_local_markdown 按标题自动切分成章节文档。
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

# Allow running as `python scripts/ingest_pdf_to_local.py` from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_MIN_TEXT_CHARS = 80  # 首页文字低于此值视为扫描件，走 OCR

_PAGE_MARK_RE = re.compile(r"^===== 第\d+页 =====$", re.MULTILINE)


def clean_ocr_text(text: str) -> str:
    """
    清洗 OCR 文本（创新点：提升扫描件入库质量）。

    去除「===== 第N页 =====」页标记、折叠多余空行、清理连排空白，
    避免页标记被当作检索内容、噪声挤占 chunk 空间。
    """
    text = _PAGE_MARK_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _plain_text_pdf(pdf_path: Path) -> str:
    """pymupdf 纯文本提取（pymupdf4llm 失败时的兜底）。"""
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    return "\n\n".join(page.get_text() for page in doc)


def pdf_to_markdown(pdf_path: Path) -> str:
    """
    PDF 转 Markdown。

    注意：pymupdf4llm 的内置 OCR 在 Windows 上会把中文输出成乱码（编码 bug），
    因此扫描件一律直接走 RapidOCR 管线，不再交给 pymupdf4llm。
    """
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    sample = "".join(doc[i].get_text() for i in range(min(3, len(doc))))
    if len(sample.strip()) > _MIN_TEXT_CHARS:  # 有文字层
        try:
            import pymupdf4llm

            md = pymupdf4llm.to_markdown(str(pdf_path))
            if len(md.strip()) > _MIN_TEXT_CHARS:
                return md
        except Exception as e:
            logger.warning("pymupdf4llm 失败（%s），退回 pymupdf 纯文本提取", e)
        return _plain_text_pdf(pdf_path)
    logger.info("扫描件（无文字层），改用 RapidOCR 识别 %s", pdf_path)
    return _ocr_pdf(pdf_path)


def _ocr_pdf(pdf_path: Path) -> str:
    """RapidOCR 逐页识别扫描 PDF。"""
    import pymupdf
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    doc = pymupdf.open(str(pdf_path))
    parts: list[str] = []
    tmp = Path(pdf_path).with_suffix(".tmp.png")
    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=200)
            pix.save(str(tmp))
            result, _ = engine(str(tmp))
            parts.append(f"\n===== 第{i + 1}页 =====\n" + "\n".join(t for _, t, _ in (result or [])))
            logger.info("OCR 第 %d/%d 页", i + 1, len(doc))
    finally:
        tmp.unlink(missing_ok=True)
    return "\n".join(parts)


def ingest_pdf(pdf_path: Path) -> Path | None:
    """把单个 PDF 转成 data/raw/local 下的 Markdown。"""
    settings = get_settings()
    out_dir = settings.raw_path / "local"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = pdf_to_markdown(pdf_path)
    if not md.strip():
        logger.warning("跳过（内容为空）: %s", pdf_path.name)
        return None
    out = out_dir / f"{pdf_path.stem}.md"
    out.write_text(clean_ocr_text(md), encoding="utf-8")
    logger.info("已入库 Markdown: %s（%d 字）", out.name, len(md))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF 一键入库（转 Markdown 到 data/raw/local）")
    parser.add_argument("pdfs", nargs="*", type=Path, help="PDF 文件路径")
    parser.add_argument("--dir", type=Path, default=None, help="批量处理某目录下所有 PDF")
    args = parser.parse_args()

    targets: list[Path] = list(args.pdfs)
    if args.dir:
        targets += sorted(args.dir.glob("*.pdf"))
    if not targets:
        parser.error("请提供 PDF 路径或 --dir 目录")

    for p in targets:
        if not p.exists():
            logger.warning("文件不存在，跳过: %s", p)
            continue
        ingest_pdf(p)
    print("完成。下一步: python scripts/build_kb.py --skip-live --stats")


if __name__ == "__main__":
    main()
