# Extract information from arxiv papers using agent LLMConfig (Requires Python 3.10+)
import json
import os
import sqlite3

import PIL
import requests
from arxiv2text import arxiv_to_text
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from pydantic import BaseModel

from config import config, LLMConfig, LLMClient

# --- Slidev Template (professional layout) ---

FRONTMATTER = """\
---
theme: seriph
background: https://cover.sli.dev
title: "{title}"
info: |
  ## {title}
  {author} — {date}
class: text-center
drawings:
  persist: false
transition: slide-left
mdc: true
export:
  format: pdf
  timeout: 600000
---"""

SLIDE_COVER = """\

# {title}

<div class="pt-12">
  <span class="px-2 py-1 rounded text-sm">
    {author}
  </span>
</div>

<div class="abs-br m-6 flex gap-2">
  <span class="text-sm opacity-50">{date}</span>
</div>"""

SLIDE_TOC = """\

---
layout: default
transition: fade-out
---

# Table of Contents

<Toc text-sm minDepth="1" maxDepth="2" />"""

SLIDE_PROBLEM = """\

---
layout: default
---

# Problem Statement

{problem}"""

SLIDE_APPROACH = """\

---
layout: two-cols
layoutClass: gap-8
---

# Key Approach

{approach}

::right::

<img src="{img1}" class="rounded shadow-lg mt-8" />"""

SLIDE_APPROACH_NO_IMG = """\

---
layout: default
---

# Key Approach

{approach}"""

SLIDE_MODEL = """\

---
layout: default
level: 2
transition: slide-up
---

# Key Steps / Architecture

{model}"""

SLIDE_DATASET = """\

---
layout: two-cols
layoutClass: gap-8
level: 2
transition: slide-up
---

# Dataset & Benchmarks

{dataset}

::right::

<img src="{img2}" class="rounded shadow-lg mt-8" />"""

SLIDE_DATASET_NO_IMG = """\

---
layout: default
level: 2
transition: slide-up
---

# Dataset & Benchmarks

{dataset}"""

SLIDE_EVALUATION = """\

---
layout: two-cols
layoutClass: gap-8
---

# Evaluation & Results

{evaluation}

::right::

<img src="{img3}" class="rounded shadow-lg mt-8" />"""

SLIDE_EVALUATION_NO_IMG = """\

---
layout: default
---

# Evaluation & Results

{evaluation}"""

SLIDE_CONCLUSION = """\

---
layout: default
---

# Conclusion & Future Work

{conclusion}"""

SLIDE_END = """\

---
layout: center
class: text-center
---

# Thank You

<div class="text-sm opacity-50 pt-4">
Questions & Discussion
</div>"""

# --- LLM Prompt ---

SYSTEM_PROMPT = """\
You are an expert at summarizing scientific papers into structured slide content for academic presentations.

OUTPUT FORMAT: Respond with a valid JSON object containing exactly these keys:
- "problem": Problem statement (3-5 bullet points using markdown `- `)
- "approach": Key approach (3-5 bullet points with **bold** key terms)
- "model": Architecture/steps (numbered list `1. ` format, 4-6 steps)
- "dataset": Dataset details (bullet points with dataset name **bolded**, include size/type/source)
- "evaluation": Evaluation results (bullet points with metric names **bolded**, include numbers)
- "conclusion": Conclusion (3-4 bullet points: findings, applications, limitations, future work)

FORMATTING RULES:
- Each section should be 80-150 words
- Use markdown bullet points (`- `) or numbered lists (`1. `)
- Bold key terms, model names, dataset names, and metric values with **double asterisks**
- Keep each bullet point to 1-2 lines max
- Do NOT use headers inside sections (no # or ##)
- Do NOT wrap the JSON in markdown code blocks"""

USER_PROMPT_TEMPLATE = """\
Summarize this paper into structured slide content.

**Title:** {title}
**Authors:** {author}
**Date:** {date}

**Paper Content:**
{paper_text}"""


# --- PDF/Image Processing ---

config_parser = ConfigParser({"output_format": "markdown"})
converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=config_parser.get_llm_service(),
)


def extract_page_figure(s):
    parts = s.split("_")
    page_number = int(parts[2])
    figure_number = int(parts[4].replace(".jpeg", ""))
    return (page_number, figure_number)


def extract_image(elements):
    img_path_list = []
    new_width = 577
    for ele in elements:
        img_paths = []
        os.makedirs("data/", exist_ok=True)
        os.makedirs(f"slides/md/{str(ele.id)}", exist_ok=True)
        response = requests.get(ele.url)
        file_path = f"data/arxiv_{ele.id}.pdf"
        if response.status_code == 200:
            with open(file_path, "wb") as file:
                file.write(response.content)
            print("PDF downloaded successfully")
        else:
            print(f"Failed to download: {response.status_code}")

        rendered = converter(file_path)
        _, _, images = text_from_rendered(rendered)
        for key, value in images.items():
            img_path = f"slides/md/{str(ele.id)}/{key}"
            original_width, original_height = value.size
            aspect_ratio = original_height / original_width
            new_height = int(new_width * aspect_ratio)
            resized_img = value.resize((new_width, new_height), PIL.Image.LANCZOS)
            resized_img.save(img_path)
            img_paths.append(f"/{str(ele.id)}/{key}")
        img_path_list.append(sorted(img_paths, key=extract_page_figure))
    return img_path_list


# --- Slide Assembly ---


def assemble_slides(sections: dict, title: str, author: str, date: str, img_list: list[str]) -> str:
    """Assemble the final Slidev markdown from structured sections."""
    imgs = (img_list + ["", "", ""])[:3]  # Pad to 3

    parts = [
        FRONTMATTER.format(title=title, author=author, date=date),
        SLIDE_COVER.format(title=title, author=author, date=date),
        SLIDE_TOC,
        SLIDE_PROBLEM.format(problem=sections.get("problem", "")),
    ]

    # Approach slide: use two-cols if image available
    if imgs[0]:
        parts.append(SLIDE_APPROACH.format(approach=sections.get("approach", ""), img1=imgs[0]))
    else:
        parts.append(SLIDE_APPROACH_NO_IMG.format(approach=sections.get("approach", "")))

    parts.append(SLIDE_MODEL.format(model=sections.get("model", "")))

    # Dataset slide
    if imgs[1]:
        parts.append(SLIDE_DATASET.format(dataset=sections.get("dataset", ""), img2=imgs[1]))
    else:
        parts.append(SLIDE_DATASET_NO_IMG.format(dataset=sections.get("dataset", "")))

    # Evaluation slide
    if imgs[2]:
        parts.append(SLIDE_EVALUATION.format(evaluation=sections.get("evaluation", ""), img3=imgs[2]))
    else:
        parts.append(SLIDE_EVALUATION_NO_IMG.format(evaluation=sections.get("evaluation", "")))

    parts.append(SLIDE_CONCLUSION.format(conclusion=sections.get("conclusion", "")))
    parts.append(SLIDE_END)

    return "\n".join(parts) + "\n"


def parse_llm_response(response_text: str) -> dict:
    """Parse the LLM JSON response, handling common formatting issues."""
    text = response_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # Remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    return json.loads(text)


# --- Database ---


class ReportSchema(BaseModel):
    id: int
    title: str
    author: str
    date: str
    url: str


def get_urls_from_db(db_path, topic):
    conf = config.get_config(topic)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        f"select id, title, author, created, pdf_url from articles where report is null and {conf['label']} = 1"
    ).fetchall()
    data = []
    for row in rows:
        ele = ReportSchema(
            id=row[0],
            title=row[1],
            author=row[2],
            date=row[3],
            url=row[4],
        )
        data.append(ele)
    conn.close()
    return data


# --- Main Generation ---


def generate_md(elements, db_path, img_path_list):
    llm_config = LLMConfig.from_env()
    llm = LLMClient(llm_config)

    conn = sqlite3.connect(db_path)
    for index, ele in enumerate(elements):
        paper_text = arxiv_to_text(ele.url)
        # Use more text for better summaries (up to 6000 chars)
        paper_content = paper_text[:6000]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    title=ele.title,
                    author=ele.author,
                    date=ele.date,
                    paper_text=paper_content,
                ),
            },
        ]

        try:
            response_text = llm.chat(messages, temperature=0.3)
            sections = parse_llm_response(response_text)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error processing paper {ele.id}: {e}")
            # Fallback: use raw response as problem statement
            sections = {
                "problem": response_text if isinstance(e, json.JSONDecodeError) else "",
                "approach": "",
                "model": "",
                "dataset": "",
                "evaluation": "",
                "conclusion": "",
            }

        res_md = assemble_slides(
            sections=sections,
            title=ele.title,
            author=ele.author,
            date=ele.date,
            img_list=img_path_list[index],
        )

        filepath = f"./slides/md/{ele.id}.md"
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(res_md)
        print(f"Markdown content for article {ele.id} is saved successfully.")
        conn.execute("UPDATE articles SET report = ? WHERE id = ?", (True, ele.id))
        conn.commit()
    conn.close()


if __name__ == "__main__":
    topics = ["RAG", "CLIP", "LLM"]
    db_path = "arxiv_articles.db"
    for topic in topics:
        data = get_urls_from_db(db_path, topic)
        img_path_list = extract_image(data)
        generate_md(data, db_path, img_path_list)
