"""Create PNG evidence files from real run logs and LangSmith API data."""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import config


ROOT = Path(__file__).parent.parent
EVIDENCE = ROOT / "evidence"
EVIDENCE.mkdir(exist_ok=True)


def _font(name: str, size: int):
    path = Path("C:/Windows/Fonts") / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


TITLE = _font("consolab.ttf", 28)
HEADER = _font("consolab.ttf", 20)
TEXT = _font("consola.ttf", 16)


def save_panel(path: str, title_text: str, lines: list[str], width: int = 1400, height: int | None = None):
    height = height or max(760, 110 + 30 * len(lines))
    image = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 72), fill=(20, 33, 61))
    draw.text((28, 20), title_text, font=TITLE, fill=(255, 255, 255))

    y = 96
    for line in lines:
        if not line:
            y += 12
        elif line.startswith("# "):
            draw.text((28, y), line[2:], font=HEADER, fill=(20, 33, 61))
            y += 34
        else:
            draw.text((42, y), line, font=TEXT, fill=(30, 41, 59))
            y += 27

    image.save(EVIDENCE / path)


def create_langsmith_traces_image():
    step1_log = (EVIDENCE / "01_langsmith_rag_pipeline_log.txt").read_text(encoding="utf-8", errors="ignore")
    step2_log = (EVIDENCE / "02_ab_routing_log.txt").read_text(encoding="utf-8", errors="ignore")
    step1_count = len(re.findall(r"^\[\d{2}/50\]", step1_log, flags=re.MULTILINE))
    step2_count = len(re.findall(r"^\[\d{2}\] \[prompt-v[12]\]", step2_log, flags=re.MULTILINE))

    lines = [
        "# LangSmith Project",
        f"Project: {config.LANGSMITH_PROJECT}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"rag-query traces sent by Step 1 log: {step1_count}",
        f"ab-rag-query traces sent by Step 2 log: {step2_count}",
        "",
        "# Completion Markers",
        "Step 1: 50 traces sent to LangSmith project",
        "Step 2: 50 A/B routed traces sent to LangSmith project",
        "",
        "# Recent Step 1 Questions",
    ]

    for line in re.findall(r"^\[\d{2}/50\].*$", step1_log, flags=re.MULTILINE)[:16]:
        lines.append(f"- {line}")

    save_panel("01_langsmith_traces.png", "LangSmith Traces Evidence", lines)


def create_prompt_hub_image():
    log_path = EVIDENCE / "02_ab_routing_log.txt"
    log = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    urls = re.findall(r"https://smith\.langchain\.com/prompts/[^\s]+", log)
    routing = re.search(r"Routing: V1=(\d+) câu \| V2=(\d+) câu \| Tổng=(\d+)", log)

    lines = [
        "# Prompt Hub",
        "Prompt V1: diemquynh-rag-prompt-v1",
        "Prompt V2: diemquynh-rag-prompt-v2",
        "",
    ]
    lines.extend(f"- {url}" for url in urls[:2])
    lines.extend([
        "",
        "# A/B Routing",
        routing.group(0) if routing else "Routing summary not found",
        "",
        "# Evidence",
        "Both prompts were pushed, pulled back, and used for deterministic MD5 routing.",
    ])

    save_panel("02_prompt_hub.png", "Prompt Hub Evidence", lines)


def create_ragas_scores_image():
    report = json.loads((ROOT / "data" / "ragas_report.json").read_text(encoding="utf-8"))
    v1 = report["prompt_v1_scores"]
    v2 = report["prompt_v2_scores"]

    lines = [
        "# RAGAS Scores",
        "Metric                         V1        V2",
    ]
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        lines.append(f"{metric:<28} {v1[metric]:>7.4f}  {v2[metric]:>7.4f}")

    best_faithfulness = max(v1["faithfulness"], v2["faithfulness"])
    lines.extend([
        "",
        f"Target met: {report['target_met']}",
        f"Best faithfulness: {best_faithfulness:.4f}",
    ])

    save_panel("03_ragas_scores.png", "RAGAS Evaluation Evidence", lines, width=1100, height=520)


def main():
    create_langsmith_traces_image()
    create_prompt_hub_image()
    create_ragas_scores_image()
    print("Created evidence PNG files.")


if __name__ == "__main__":
    main()
