# Uses slidev-theme-scholarly for academic presentation layouts
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
from pipeline.layout_templates import resolve_template, DEFAULT_LAYOUTS


FRONTMATTER = """\
---
theme: scholarly
background: https://cover.sli.dev
title: "{title}"
info: |
  ## {title}
  {author} — {date}
layout: cover
class: text-center
themeConfig:
  color-theme: {color_theme}
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

SLIDE_SECTION_BACKGROUND = """\

---
layout: section
transition: fade
---

# Background & Motivation"""

SLIDE_PROBLEM = """\

---
layout: bullets
transition: slide-up
---

# Problem Statement

{problem}

<Keywords :keywords='{keywords}' />"""

SLIDE_SECTION_METHOD = """\

---
layout: section
transition: fade
---

# Methodology"""

SLIDE_APPROACH = """\

---
layout: two-cols
layoutClass: gap-8
transition: slide-left
---

# Key Approach

{approach}

::right::

<img src="{img1}" class="rounded shadow-lg mt-8" />"""

SLIDE_APPROACH_NO_IMG = """\

---
layout: two-cols
transition: slide-left
---

# Key Approach

{approach}

::right::

<Block type="info">

**Core Innovation**

{core_innovation}

</Block>"""

SLIDE_MODEL = """\

---
layout: default
transition: slide-up
---

# Key Steps / Architecture

{model}"""

SLIDE_SECTION_EXPERIMENTS = """\

---
layout: section
transition: fade
---

# Experiments & Results"""

SLIDE_DATASET = """\

---
layout: two-cols
layoutClass: gap-8
transition: slide-right
---

# Dataset & Benchmarks

{dataset}

::right::

<img src="{img2}" class="rounded shadow-lg mt-8" />"""

SLIDE_DATASET_NO_IMG = """\

---
layout: bullets
transition: slide-right
---

# Dataset & Benchmarks

{dataset}"""

SLIDE_EVALUATION = """\

---
layout: two-cols
layoutClass: gap-8
transition: slide-up
---

# Evaluation & Results

{evaluation}

::right::

<img src="{img3}" class="rounded shadow-lg mt-8" />"""

SLIDE_EVALUATION_NO_IMG = """\

---
layout: results
transition: slide-up
---

# Evaluation & Results

{evaluation}"""

SLIDE_CONCLUSION = """\

---
layout: default
transition: fade-out
---

# Conclusion & Future Work

{conclusion}"""

SLIDE_END = """\

---
layout: end
---

# Thank You

<div class="text-center text-sm opacity-60 mt-4">
Questions & Discussion
</div>"""


SYSTEM_PROMPT = """\
You are an expert at summarizing scientific papers into structured slide content for professional academic presentations.

OUTPUT FORMAT: Respond with a valid JSON object containing exactly these keys:
- "keywords": A list of 3-5 key terms/topics (e.g., ["reinforcement learning", "embodied AI", "memory systems"])
- "problem": Problem statement (3-5 bullet points using markdown `- `)
- "approach": Key approach (3-5 bullet points with **bold** key terms)
- "core_innovation": A single concise sentence describing the paper's main contribution (used as a highlight block)
- "model": Architecture/steps (numbered list `1. ` format, 4-6 steps)
- "dataset": Dataset details (bullet points with dataset name **bolded**, include size/type/source)
- "evaluation": Evaluation results (bullet points with metric names **bolded**, include numbers)
- "conclusion": Conclusion (3-4 bullet points: findings, applications, limitations, future work)
- "layout_hints": An object recommending the best slidev-theme-scholarly layout for each section based on the paper's semantic content. Keys and allowed values:
  - "problem": "bullets" (list of issues) | "statement" (single bold claim) | "focus" (one key challenge with icon)
  - "approach": "methodology" (step-by-step pipeline) | "two-cols" (text + visual) | "compare" (old vs new approach) | "image-right" (diagram-heavy)
  - "model": "default" (numbered steps) | "methodology" (formal pipeline) | "timeline" (chronological stages)
  - "dataset": "bullets" (simple list) | "two-cols" (stats + visual) | "compare" (multiple datasets side-by-side)
  - "evaluation": "results" (quantitative table/metrics) | "compare" (baseline comparison) | "fact" (single headline metric) | "two-cols" (text + chart)
  - "conclusion": "default" (bullet list) | "bullets" (enhanced bullets) | "statement" (impactful closing)
- "color_theme": Choose ONE color theme that best matches the paper's domain/tone. Options:
  - "classic-blue": general CS, systems, networking, default choice
  - "oxford-burgundy": humanities-adjacent, social science, HCI, ethics
  - "cambridge-green": biology, environmental science, sustainability, health
  - "yale-blue": formal methods, mathematics, theoretical CS, logic
  - "princeton-orange": creative AI, generative models, art/design, multimedia
  - "nordic-blue": NLP, language models, linguistics, cognitive science
  - "warm-sepia": historical analysis, surveys, literature reviews
  - "monochrome": engineering, hardware, robotics, low-level systems
  - "high-contrast": accessibility research, visualization, UI/UX

LAYOUT SELECTION GUIDANCE:
- Use "methodology" when the paper describes a clear step-by-step process or pipeline
- Use "compare" when contrasting approaches, models, or showing baseline comparisons
- Use "results" for sections with multiple quantitative metrics
- Use "fact" when there is one standout finding or statistic
- Use "timeline" for work with chronological stages or evolution
- Use "statement"/"focus" for strong theoretical claims or contributions
- Use "two-cols" when there is a natural split between text and visual content
- Default to "bullets" when content is a straightforward list

COLOR THEME GUIDANCE:
- Match the color to the paper's research domain and tone
- For ML/DL papers: "classic-blue" (general), "nordic-blue" (NLP/language), "princeton-orange" (generative/creative)
- For theory-heavy: "yale-blue"
- For applied/interdisciplinary: "oxford-burgundy" (social), "cambridge-green" (bio/health)
- For systems/hardware: "monochrome"
- For surveys/reviews: "warm-sepia"
- Default to "classic-blue" if unsure

FORMATTING RULES:
- Each section should be 80-150 words
- Use markdown bullet points (`- `) or numbered lists (`1. `)
- Bold key terms, model names, dataset names, and metric values with **double asterisks**
- Keep each bullet point to 1-2 lines max
- Do NOT use headers inside sections (no # or ##)
- Do NOT wrap the JSON in markdown code blocks
- For "keywords": provide a JSON array of short strings (2-3 words each)
- For "core_innovation": one sentence, no bullet points
- For "layout_hints": provide a JSON object with section names as keys
- For "color_theme": provide a single string from the allowed color theme names"""

USER_PROMPT_TEMPLATE = """\
Summarize this paper into structured slide content.

**Title:** {title}
**Authors:** {author}
**Date:** {date}

**Paper Content:**
{paper_text}"""


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

def assemble_slides(sections: dict, title: str, author: str, date: str, img_list: list[str]) -> str:
    """Assemble the final Slidev markdown from structured sections using scholarly theme.

    Uses LLM-provided layout_hints to dynamically select the best scholarly layout
    for each section based on paper semantics. Falls back to fixed templates when
    layout_hints are missing.
    """
    imgs = (img_list + ["", "", ""])[:3]  # Pad to 3

    # Format keywords for the Keywords component
    keywords = sections.get("keywords", [])
    if isinstance(keywords, list):
        keywords_str = json.dumps(keywords)
    else:
        keywords_str = '["research"]'

    core_innovation = sections.get("core_innovation", "Novel contribution to the field.")
    layout_hints = sections.get("layout_hints", {})
    if not isinstance(layout_hints, dict):
        layout_hints = {}

    color_theme = sections.get("color_theme", "classic-blue")
    valid_themes = {
        "classic-blue", "oxford-burgundy", "cambridge-green", "yale-blue",
        "princeton-orange", "nordic-blue", "warm-sepia", "monochrome", "high-contrast",
    }
    if color_theme not in valid_themes:
        color_theme = "classic-blue"

    parts = [
        FRONTMATTER.format(title=title, author=author, date=date, color_theme=color_theme),
        SLIDE_COVER.format(title=title, author=author, date=date),
        SLIDE_SECTION_BACKGROUND,
    ]

    # Problem slide — dynamic layout
    problem_layout = layout_hints.get("problem", DEFAULT_LAYOUTS["problem"])
    problem_tpl = resolve_template("problem", problem_layout, has_image=False)
    parts.append(problem_tpl.format(
        content=sections.get("problem", ""),
        keywords=keywords_str,
    ))

    parts.append(SLIDE_SECTION_METHOD)

    # Approach slide — dynamic layout, image-aware
    approach_layout = layout_hints.get("approach", DEFAULT_LAYOUTS["approach"])
    approach_tpl = resolve_template("approach", approach_layout, has_image=bool(imgs[0]))
    parts.append(approach_tpl.format(
        content=sections.get("approach", ""),
        core_innovation=core_innovation,
        img=imgs[0],
    ))

    # Model slide — dynamic layout
    model_layout = layout_hints.get("model", DEFAULT_LAYOUTS["model"])
    model_tpl = resolve_template("model", model_layout, has_image=False)
    parts.append(model_tpl.format(content=sections.get("model", "")))

    parts.append(SLIDE_SECTION_EXPERIMENTS)

    # Dataset slide — dynamic layout, image-aware
    dataset_layout = layout_hints.get("dataset", DEFAULT_LAYOUTS["dataset"])
    dataset_tpl = resolve_template("dataset", dataset_layout, has_image=bool(imgs[1]))
    parts.append(dataset_tpl.format(
        content=sections.get("dataset", ""),
        img=imgs[1],
    ))

    # Evaluation slide — dynamic layout, image-aware
    eval_layout = layout_hints.get("evaluation", DEFAULT_LAYOUTS["evaluation"])
    eval_tpl = resolve_template("evaluation", eval_layout, has_image=bool(imgs[2]))
    parts.append(eval_tpl.format(
        content=sections.get("evaluation", ""),
        img=imgs[2],
    ))

    # Conclusion slide — dynamic layout
    conclusion_layout = layout_hints.get("conclusion", DEFAULT_LAYOUTS["conclusion"])
    conclusion_tpl = resolve_template("conclusion", conclusion_layout, has_image=False)
    parts.append(conclusion_tpl.format(content=sections.get("conclusion", "")))

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
        f"select id, title, author, created, pdf_url from articles where {conf['label']} = 1 limit 1"
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
    topics = ["LLM"]
    db_path = "arxiv_articles.db"
    for topic in topics:
        data = get_urls_from_db(db_path, topic)
        img_path_list = extract_image(data)
        generate_md(data, db_path, img_path_list)
