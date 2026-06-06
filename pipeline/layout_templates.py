# Semantic layout templates for slidev-theme-scholarly
# Maps (section_name, layout_hint) -> template string with format placeholders.

LAYOUT_TEMPLATES = {
    # --- Problem section ---
    ("problem", "bullets"): """\

---
layout: bullets
transition: slide-up
---

# Problem Statement

{content}

<Keywords :keywords='{keywords}' />""",

    ("problem", "statement"): """\

---
layout: statement
transition: fade
---

# Problem Statement

{content}""",

    ("problem", "focus"): """\

---
layout: focus
transition: slide-up
---

# Core Challenge

{content}

<Keywords :keywords='{keywords}' />""",

    # --- Approach section ---
    ("approach", "methodology"): """\

---
layout: methodology
transition: slide-left
---

# Methodology

{content}

<Block type="info">

**Core Innovation**

{core_innovation}

</Block>""",

    ("approach", "two-cols"): """\

---
layout: two-cols
layoutClass: gap-8
transition: slide-left
---

# Key Approach

{content}

::right::

<Block type="info">

**Core Innovation**

{core_innovation}

</Block>""",

    ("approach", "two-cols-img"): """\

---
layout: two-cols
layoutClass: gap-8
transition: slide-left
---

# Key Approach

{content}

::right::

<img src="{img}" class="rounded shadow-lg mt-8" />""",

    ("approach", "compare"): """\

---
layout: compare
transition: slide-left
---

# Approach Comparison

{content}""",

    ("approach", "image-right"): """\

---
layout: image-right
image: "{img}"
transition: slide-left
---

# Key Approach

{content}""",

    # --- Model section ---
    ("model", "default"): """\

---
layout: default
transition: slide-up
---

# Key Steps / Architecture

{content}""",

    ("model", "methodology"): """\

---
layout: default
class: wide-content
transition: slide-up
---

# Architecture & Pipeline

<div class="grid grid-cols-1 gap-4">

{content}

</div>""",

    ("model", "timeline"): """\

---
layout: timeline
transition: slide-up
---

# Architecture Stages

{content}""",

    # --- Dataset section ---
    ("dataset", "bullets"): """\

---
layout: bullets
transition: slide-right
---

# Dataset & Benchmarks

{content}""",

    ("dataset", "two-cols"): """\

---
layout: two-cols
layoutClass: gap-8
transition: slide-right
---

# Dataset & Benchmarks

{content}

::right::

<img src="{img}" class="rounded shadow-lg mt-8" />""",

    ("dataset", "compare"): """\

---
layout: compare
transition: slide-right
---

# Datasets Comparison

{content}""",

    # --- Evaluation section ---
    ("evaluation", "results"): """\

---
layout: results
transition: slide-up
---

# Evaluation & Results

{content}""",

    ("evaluation", "compare"): """\

---
layout: compare
transition: slide-up
---

# Results Comparison

{content}""",

    ("evaluation", "fact"): """\

---
layout: fact
transition: slide-up
---

# Key Finding

{content}""",

    ("evaluation", "two-cols"): """\

---
layout: two-cols
layoutClass: gap-8
transition: slide-up
---

# Evaluation & Results

{content}

::right::

<img src="{img}" class="rounded shadow-lg mt-8" />""",

    # --- Conclusion section ---
    ("conclusion", "default"): """\

---
layout: default
transition: fade-out
---

# Conclusion & Future Work

{content}""",

    ("conclusion", "bullets"): """\

---
layout: bullets
transition: fade-out
---

# Conclusion & Future Work

{content}""",

    ("conclusion", "statement"): """\

---
layout: statement
transition: fade-out
---

# Key Takeaway

{content}""",
}

# Default layout per section (fallback when layout_hints is missing or unrecognized)
DEFAULT_LAYOUTS = {
    "problem": "bullets",
    "approach": "two-cols",
    "model": "default",
    "dataset": "bullets",
    "evaluation": "results",
    "conclusion": "default",
}


def resolve_template(section: str, layout_hint: str, has_image: bool = False) -> str:
    """Resolve a layout template for a section given the LLM's hint.

    Falls back to default layout if hint is unrecognized.
    For approach/dataset/evaluation with images, prefers image-capable variant.
    """
    # If image available and layout supports it, use image variant
    if has_image and section in ("approach", "dataset", "evaluation"):
        img_key = (section, "two-cols-img") if section == "approach" else (section, "two-cols")
        if img_key in LAYOUT_TEMPLATES:
            return LAYOUT_TEMPLATES[img_key]

    key = (section, layout_hint)
    if key in LAYOUT_TEMPLATES:
        return LAYOUT_TEMPLATES[key]

    # Fallback to default layout for this section
    default = DEFAULT_LAYOUTS.get(section, "default")
    fallback_key = (section, default)
    return LAYOUT_TEMPLATES.get(fallback_key, "")
