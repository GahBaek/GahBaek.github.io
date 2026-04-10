#!/usr/bin/env python3
"""
build.py  —  Run from the ROOT of your site:
    python build.py

Folder structure:
    posts_content/    ← study notes .md files
    papers_content/   ← paper review .md files

Generates:
    pages/posts/      ← individual post HTML pages
    pages/papers/     ← individual paper HTML pages
    pages/posts.html  ← card index
    pages/papers.html ← card index
"""

import os
import re

# ── TAG MAPS ────────────────────────────────────────────────────────────
POSTS_TAG_MAP = {
    "ctf":         "CTF",
    "compiler":    "Compiler",
    "andrew":      "Andrew.Ng",
    "cnn":         "DL",
    "lab":         "DL",
    "overfitting": "DL",
    "relu":        "DL",
    "xor":         "DL",
    "softmax":     "DL",
    "lec":         "ML",
    "머신러닝":     "ML",
    "카운트":       "NLP",
    "언어":         "NLP",
    "데이터":       "ML",
    "improving":   "DL",
}

PAPERS_TAG_MAP = {
    "transformer": "NLP",
    "attention":   "NLP",
    "llm":         "LLM",
    "gpt":         "LLM",
    "bert":        "NLP",
    "diffusion":   "GenAI",
    "gan":         "GenAI",
    "rl":          "RL",
    "survey":      "Survey",
    "vision":      "CV",
    "detection":   "CV",
    "security":    "Security",
    "fuzzing":     "Security",
    "taint":       "Security",
}
# ────────────────────────────────────────────────────────────────────────


def guess_tag(filename, tag_map, default="Paper"):
    low = filename.lower()
    for key, tag in tag_map.items():
        if key in low:
            return tag
    return default


def parse_filename(fname):
    """YYYY-MM-DD-title.md  →  (date_str, title, slug, sort_key)"""
    base = fname.replace(".md", "")
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})-(.*)", base)
    if m:
        year, month, day, rest = m.groups()
        title = rest.replace("-", " ").replace("_", " ").strip()
        date_str = f"{year}·{month}"
        slug = re.sub(r"[^\w가-힣-]", "", rest.replace(" ", "-"))
        return date_str, title, slug, f"{year}-{month}-{day}"
    return "—", base, re.sub(r"\s+", "-", base), "0000-00-00"


def md_to_html_body(md_text):
    """Markdown → HTML (handles headings, lists, code blocks, inline styles)."""
    lines = md_text.split("\n")
    html = []
    in_code = False
    in_ul = False
    in_ol = False
    ol_num = 0

    for line in lines:
        # fenced code block toggle
        if line.startswith("```"):
            if not in_code:
                lang = line[3:].strip() or ""
                html.append(f'<pre><code class="language-{lang}">')
                in_code = True
            else:
                html.append("</code></pre>")
                in_code = False
            continue

        if in_code:
            html.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            continue

        # close open lists if line doesn't continue them
        if in_ul and not re.match(r"^[-*] ", line):
            html.append("</ul>")
            in_ul = False
        if in_ol and not re.match(r"^\d+\. ", line):
            html.append("</ol>")
            in_ol = False

        # headings
        if line.startswith("#### "):
            html.append(f"<h4>{inline(line[5:])}</h4>")
        elif line.startswith("### "):
            html.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.startswith("## "):
            html.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            html.append(f"<h2>{inline(line[2:])}</h2>")   # demote to h2
        # blockquote
        elif line.startswith("> "):
            html.append(f"<blockquote><p>{inline(line[2:])}</p></blockquote>")
        # horizontal rule
        elif re.match(r"^[-*_]{3,}$", line.strip()):
            html.append("<hr>")
        # unordered list
        elif re.match(r"^[-*] ", line):
            if not in_ul:
                html.append("<ul>")
                in_ul = True
            html.append(f"<li>{inline(line[2:])}</li>")
        # ordered list
        elif re.match(r"^\d+\. ", line):
            if not in_ol:
                html.append("<ol>")
                in_ol = True
            html.append(f"<li>{inline(re.sub(r'^\d+\. ', '', line))}</li>")
        # blank line
        elif line.strip() == "":
            html.append("")
        # paragraph
        else:
            html.append(f"<p>{inline(line)}</p>")

    if in_ul:
        html.append("</ul>")
    if in_ol:
        html.append("</ol>")
    if in_code:
        html.append("</code></pre>")

    return "\n".join(html)


def inline(text):
    """Apply inline Markdown: bold, italic, code, links, images."""
    text = re.sub(r"!\[(.+?)\]\((.+?)\)", r'<img src="\2" alt="\1" style="max-width:100%;border-radius:6px;margin:1rem 0;">', text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)",  r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*",       r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*",           r"<em>\1</em>",         text)
    text = re.sub(r"`(.+?)`",             r"<code>\1</code>",      text)
    return text


# ── HTML TEMPLATES ───────────────────────────────────────────────────────

POST_STYLES = """
    .post-header { margin-bottom: 2.5rem; }
    .post-header h1 {
      font-family: 'DM Serif Display', serif;
      font-size: 2rem; font-weight: 400; line-height: 1.2;
      margin-bottom: .6rem;
    }
    .post-meta { display: flex; gap: .8rem; align-items: center; font-size: .78rem; color: var(--muted); }
    .post-tag { background: var(--tag); border-radius: 4px; padding: .1rem .5rem; font-size: .72rem; color: var(--muted); }
    .post-body { line-height: 1.85; font-size: .95rem; }
    .post-body h2 { font-family: 'DM Serif Display', serif; font-size: 1.4rem; font-weight: 400; margin: 2.2rem 0 .8rem; }
    .post-body h3 { font-size: 1rem; font-weight: 600; margin: 1.8rem 0 .5rem; }
    .post-body h4 { font-size: .9rem; font-weight: 600; margin: 1.2rem 0 .4rem; color: var(--muted); }
    .post-body p  { margin-bottom: 1rem; }
    .post-body ul, .post-body ol { margin: .5rem 0 1rem 1.5rem; }
    .post-body li { margin-bottom: .35rem; }
    .post-body blockquote {
      border-left: 3px solid var(--accent); margin: 1.2rem 0;
      padding: .6rem 1rem; background: var(--tag); border-radius: 0 6px 6px 0;
    }
    .post-body blockquote p { margin: 0; color: var(--muted); font-style: italic; }
    .post-body hr { border: none; border-top: 1px solid var(--line); margin: 2rem 0; }
    .post-body pre {
      background: #1a1814; color: #f7f5f0;
      border-radius: 6px; padding: 1.2rem;
      overflow-x: auto; font-size: .82rem; margin: 1.2rem 0;
    }
    .post-body code { font-family: 'DM Mono', monospace; }
    .post-body p code { background: var(--tag); border-radius: 3px; padding: .1rem .35rem; font-size: .85em; }
    .post-body a { color: var(--accent); text-decoration: none; }
    .post-body a:hover { text-decoration: underline; }
    .back-link {
      display: inline-flex; align-items: center; gap: .4rem;
      font-size: .8rem; color: var(--muted); text-decoration: none;
      margin-bottom: 2rem; transition: color .2s;
    }
    .back-link:hover { color: var(--ink); }
    hr.post-divider { border: none; border-top: 1px solid var(--line); margin: 2rem 0; }
"""

def render_post_page(title, date_str, tag, body_html, back_label, back_href):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title} · Gahyun Baek</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="../../css/style.css">
  <style>{POST_STYLES}</style>
</head>
<body>
<nav>
  <a class="nav-logo" href="../../index.html">Gahyun Baek</a>
  <ul class="nav-links">
    <li><a href="../../index.html">Home</a></li>
    <li><a href="../papers.html">Papers</a></li>
    <li><a href="../posts.html">Posts</a></li>
    <li><a href="../blog.html">Blog</a></li>
    <li><a href="../cv.html">CV</a></li>
  </ul>
</nav>
<main class="page">
  <a class="back-link" href="{back_href}">← Back to {back_label}</a>
  <div class="post-header">
    <h1>{title}</h1>
    <div class="post-meta">
      <span>{date_str}</span>
      <span class="post-tag">{tag}</span>
    </div>
  </div>
  <hr class="post-divider">
  <div class="post-body">
{body_html}
  </div>
</main>
<script src="../../js/nav.js"></script>
</body>
</html>
"""


def render_card(date_str, tag, title, href):
    return f"""    <a class="card" href="{href}">
      <div class="card-meta">
        <div class="card-date">{date_str}</div>
        <div class="card-tag">{tag}</div>
      </div>
      <div class="card-body">
        <h3>{title}</h3>
      </div>
    </a>"""


def render_index(page_title, page_sub, cards_html, self_href, nav_active):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{page_title} · Gahyun Baek</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="../css/style.css">
</head>
<body>
<nav>
  <a class="nav-logo" href="../index.html">Gahyun Baek</a>
  <ul class="nav-links">
    <li><a href="../index.html">Home</a></li>
    <li><a href="papers.html"{' class="active"' if nav_active == 'papers' else ''}>Papers</a></li>
    <li><a href="posts.html"{' class="active"' if nav_active == 'posts' else ''}>Posts</a></li>
    <li><a href="blog.html">Blog</a></li>
    <li><a href="cv.html">CV</a></li>
  </ul>
</nav>
<main class="page">
  <h1 class="page-title">{page_title}</h1>
  <p class="page-sub">{page_sub}</p>
  <div class="card-list">
{cards_html}
  </div>
</main>
<script src="../js/nav.js"></script>
</body>
</html>
"""


# ── BUILDER ─────────────────────────────────────────────────────────────

def build_section(src_dir, out_dir, index_path,
                  page_title, page_sub, nav_active,
                  tag_map, tag_default, back_label):

    if not os.path.isdir(src_dir):
        print(f"  [skip] '{src_dir}/' not found — create it and add .md files to use this section.")
        return

    os.makedirs(out_dir, exist_ok=True)

    md_files = sorted(
        [f for f in os.listdir(src_dir) if f.endswith(".md")],
        reverse=True
    )

    if not md_files:
        print(f"  [skip] No .md files in '{src_dir}/'")
        return

    cards = []
    section_slug = os.path.basename(out_dir)   # "posts" or "papers"

    for fname in md_files:
        path = os.path.join(src_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        # strip Jekyll front matter
        raw = re.sub(r"^---[\s\S]+?---\n?", "", raw).strip()

        # fix Jekyll {{ '/assets/...' | relative_url }} → plain path
        raw = re.sub(r"\{\{\s*'(/[^']+)'\s*\|\s*relative_url\s*\}\}", r"../../\1", raw)

        date_str, title, slug, sort_key = parse_filename(fname)
        tag = guess_tag(fname, tag_map, tag_default)

        body_html  = md_to_html_body(raw)
        back_href  = f"../{section_slug}.html"
        page_html  = render_post_page(title, date_str, tag, body_html, back_label, back_href)

        out_path = os.path.join(out_dir, f"{slug}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(page_html)
        print(f"  ✓  {fname}")

        card_href = f"{section_slug}/{slug}.html"
        cards.append((sort_key, render_card(date_str, tag, title, card_href)))

    cards.sort(key=lambda x: x[0], reverse=True)
    cards_html = "\n".join(c for _, c in cards)

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(render_index(page_title, page_sub, cards_html, index_path, nav_active))

    print(f"  ✓  {index_path}  ({len(cards)} entries)\n")


def main():
    print("\n── Building Posts ──────────────────────────────")
    build_section(
        src_dir    = "posts_content",
        out_dir    = "pages/posts",
        index_path = "pages/posts.html",
        page_title = "Posts",
        page_sub   = "Study notes, concepts, and technical write-ups.",
        nav_active = "posts",
        tag_map    = POSTS_TAG_MAP,
        tag_default= "Note",
        back_label = "Posts",
    )

    print("── Building Paper Reviews ──────────────────────")
    build_section(
        src_dir    = "papers_content",
        out_dir    = "pages/papers",
        index_path = "pages/papers.html",
        page_title = "Paper Reviews",
        page_sub   = "Summaries and critiques of papers I've read.",
        nav_active = "papers",
        tag_map    = PAPERS_TAG_MAP,
        tag_default= "Paper",
        back_label = "Papers",
    )

    print("Done! Run:  git add . && git commit -m 'update' && git push")


if __name__ == "__main__":
    main()
