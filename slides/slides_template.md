---
theme: seriph
background: https://cover.sli.dev
title: 'Title of Paper'
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

## Title
- Author
- Date

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

Themes can provide styles, layouts, components, or even configurations for tools. Switching between themes by just **one edit** in your frontmatter:

<div grid="~ cols-2 gap-2" m="t-2">

```yaml
---
topic: llm
---
```

<img border="rounded" src="https://github.com/slidevjs/themes/blob/main/screenshots/theme-default/01.png?raw=true" alt="">

</div>

---

# Contributions

- 📝 **Point** - content of point
<br>
<br>

<!--
You can have `style` tag in markdown to override the style for the current page.
-->

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

<!--
Innovations
-->

---
transition: slide-up
level: 2
---

# Code

Use code snippets and get the highlighting directly, and even types hover!

```ts {all} twoslash
import { computed, ref } from 'vue'

const count = ref(0)
const doubled = computed(() => count.value * 2)

doubled.value = 2
```

<!-- <arrow v-click="[4, 5]" x1="350" y1="310" x2="195" y2="334" color="#953" width="2" arrowSize="1" /> -->
<!-- Footer -->
<!-- Inline style -->
<style>
.footnotes-sep {
  @apply mt-5 opacity-10;
}
.footnotes {
  @apply text-sm opacity-75;
}
.footnote-backref {
  display: none;
}
</style>

<!--
Notes can also sync with clicks
-->

---
transition: slide-up
level: 2
---

# Performance

It supports animations across multiple code snippets.

Add multiple code blocks and wrap them with <code>````md magic-move</code> (four backticks) to enable the magic move. For example:


---

# Conclusion

<div grid="~ cols-2 gap-4">
<div>

You can use Vue components directly inside your slides.

</div>
</div>

<!--
Presenter note with **bold**, *italic*, and ~~striked~~ text.
-->

---
class: px-20
---

<!-- # Learn More

[Documentation](https://sli.dev) · [GitHub](https://github.com/slidevjs/slidev) · [Showcases](https://sli.dev/resources/showcases)
 -->
