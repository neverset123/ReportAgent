---
theme: seriph
background: https://cover.sli.dev
title: Report Cover
info: |
  ## Slidev Template
  Presentation slides for papers.
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true

#remoteAssets: false
# monaco: false
export:
  format: pdf
  timeout: 600000 

---

## {title}
- {author}
- {date}

<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs]
-->

---
transition: fade-out
---

# Table of Contents

<Toc text-sm minDepth="1" maxDepth="2" />

---
transition: slide-right
---


# Problem Statement
{problem}


<!-- <div grid="~ cols-2 gap-2" m="t-2">

```yaml
---
topic: llm
---
```

<img border="rounded" src="https://github.com/slidevjs/themes/blob/main/screenshots/theme-default/01.png?raw=true" alt="">

</div> -->

---

# Key Approach
{approach}

<!-- - 📝 **Point** - content of point
<br>
<br> -->

<!--
You can have `style` tag in markdown to override the style for the current page.
-->

<!--
Innovations
-->

---
transition: slide-up
level: 2
---

# Key Steps/Models
{model}
<!-- Use code snippets and get the highlighting directly, and even types hover!


<!-- <arrow v-click="[4, 5]" x1="350" y1="310" x2="195" y2="334" color="#953" width="2" arrowSize="1" /> -->
<!-- Footer -->
<!-- Inline style -->

<!--
Notes can also sync with clicks
-->

---
transition: slide-up
level: 2
---

# Dataset 
{dataset}
<!-- It supports animations across multiple code snippets.

Add multiple code blocks and wrap them with <code>````md magic-move</code> (four backticks) to enable the magic move. For example: -->


---

# Evaluation 
{evaluation}
---

# Conclusion
{conclusion}
<!-- <div grid="~ cols-2 gap-4">
<div>

You can use Vue components directly inside your slides.

</div>
</div> -->

<!--
Presenter note with **bold**, *italic*, and ~~striked~~ text.
-->

---
class: px-20
---

<!-- # Learn More

[Documentation](https://sli.dev) · [GitHub](https://github.com/slidevjs/slidev) · [Showcases](https://sli.dev/resources/showcases)
 -->