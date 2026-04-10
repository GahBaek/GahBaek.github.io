#!/usr/bin/env python3
"""
build_posts.py
Run this from the ROOT of your site folder:
    python build_posts.py

It will:
1. Read all .md files from posts_content/
2. Generate individual HTML pages in pages/posts/
3. Auto-update pages/posts.html with the card list
"""

import os
import re
import json

# ── CONFIG ──────────────────────────────────────────────────────────────
POSTS_SRC   = "posts_content"          # where your .md files live
POSTS_OUT   = "pages/posts"            # where individual post HTMLs go
POSTS_INDEX = "pages/posts.html"       # the card-list page

# Map folder-name keywords → tag label
TAG_MAP = {
    "ctf":        "CTF",
    "compiler":   "Compiler",
    "andrew":     "Andrew.Ng",
    "cnn":        "DL",
    "lab":        "DL",
    "overfitting":"DL",
    "relu":       "DL",
    "xor":        "DL",
    "softmax":    "DL",
    "lec":        "ML",
    "머신러닝":    "ML",
    "카운트":      "NLP",
    "언어":        "NLP",
    "데이터":      "ML",
    "paper":      "Paper",
    "improving":  "DL",
}
# ────────────────────────────────────────────────────────────────────────


def guess_tag(filename):
    low = filename.lower()
    for key, tag in TAG_MAP.items():
        if key in low:
            return tag
    return "Note"


def parse_filename(fname):
    """
    Expected format: YYYY-MM-DD-title.md
    Returns (date_str, title, slug)
    """
    base = fname.replace(".md", "")
    # try YYYY-MM-DD-... pattern
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})-(.*)", base)
    if m:
        year, month, day, rest = m.groups()
        title = rest.replace("-", " ").replace("_", " ").strip()
        date_str = f"{year}·{month}"
        slug = re.sub(r"[^\w가-힣-]", "", rest.replace(" ", "-"))
        return date_str, title, slug, f"{year}-{month}-{day}"
    # fallback
    return "—", base, re.sub(r"\s+", "-", base), "0000-00-00"


def md_to_html_body(md_text):
    """Very simple Markdown → HTML converter for common patterns."""
    lines = md_text.split("\n")
    html = []
    in_code = False
    in_ul = False

    for line in lines:
        # fenced code block
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

        # close ul if needed
        if in_ul and not line.startswith("- ") and not line.startswith("* "):
            html.append("</ul>")
            in_ul = False

        # headings
        if line.startswith("### "):
            html.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            html.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            html.append(f"<h2>{line[2:]}</h2>")  # demote h1 to h2 inside post
        # unordered list
        elif line.startswith("- ") or line.startswith("* "):
            if not in_ul:
                html.append("<ul>")
                in_ul = True
            html.append(f"<li>{line[2:]}</li>")
        # blank line → paragraph break
        elif line.strip() == "":
            html.append("")
        else:
            # inline: bold, italic, inline code, links
            line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
            line = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         line)
            line = re.sub(r"`(.+?)`",        r"<code>\1</code>",     line)
            line = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', line)
            html.append(f"<p>{line}</p>")

    if in_ul:
        html.append("</ul>")
    if in_code:
        html.append("</code></pre>")

    return "\n".join(html)


def post_page(title, date_str, tag, body_html):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title} · Gahyun Baek</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="../../css/style.css">
  <style>
    .post-header {{ margin-bottom: 2.5rem; }}
    .post-header h1 {{
      font-family: 'DM Serif Display', serif;
      font-size: 2rem; font-weight: 400; line-height: 1.2;
      margin-bottom: .6rem;
    }}
    .post-meta {{ display: flex; gap: .8rem; align-items: center; font-size: .78rem; color: var(--muted); }}
    .post-tag {{
      background: var(--tag); border-radius: 4px;
      padding: .1rem .5rem; font-size: .72rem; color: var(--muted);
    }}
    .post-body {{ line-height: 1.8; font-size: .95rem; }}
    .post-body h2 {{ font-family: 'DM Serif Display', serif; font-size: 1.3rem; font-weight: 400; margin: 2rem 0 .8rem; }}
    .post-body h3 {{ font-size: 1rem; font-weight: 600; margin: 1.5rem 0 .5rem; }}
    .post-body p  {{ margin-bottom: 1rem; color: var(--ink); }}
    .post-body ul {{ margin: .5rem 0 1rem 1.5rem; }}
    .post-body li {{ margin-bottom: .3rem; }}
    .post-body pre {{
      background: #1a1814; color: #f7f5f0;
      border-radius: 6px; padding: 1.2rem;
      overflow-x: auto; font-size: .82rem;
      margin: 1.2rem 0;
    }}
    .post-body code {{ font-family: 'DM Mono', monospace; }}
    .post-body p code {{
      background: var(--tag); border-radius: 3px;
      padding: .1rem .35rem; font-size: .85em;
    }}
    .post-body a {{ color: var(--accent); text-decoration: none; }}
    .post-body a:hover {{ text-decoration: underline; }}
    .back-link {{
      display: inline-flex; align-items: center; gap: .4rem;
      font-size: .8rem; color: var(--muted); text-decoration: none;
      margin-bottom: 2rem;
      transition: color .2s;
    }}
    .back-link:hover {{ color: var(--ink); }}
    hr.post-divider {{ border: none; border-top: 1px solid var(--line); margin: 2rem 0; }}
  </style>
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
  <a class="back-link" href="../posts.html">
    ← Back to Posts
  </a>

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


def card_html(date_str, tag, title, slug):
    return f"""    <a class="card" href="posts/{slug}.html">
      <div class="card-meta">
        <div class="card-date">{date_str}</div>
        <div class="card-tag">{tag}</div>
      </div>
      <div class="card-body">
        <h3>{title}</h3>
      </div>
    </a>"""


def posts_index(cards_html):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Posts · Gahyun Baek</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="../css/style.css">
</head>
<body>

<nav>
  <a class="nav-logo" href="../index.html">Gahyun Baek</a>
  <ul class="nav-links">
    <li><a href="../index.html">Home</a></li>
    <li><a href="papers.html">Papers</a></li>
    <li><a href="posts.html">Posts</a></li>
    <li><a href="blog.html">Blog</a></li>
    <li><a href="cv.html">CV</a></li>
  </ul>
</nav>

<main class="page">
  <h1 class="page-title">Posts</h1>
  <p class="page-sub">Study notes, concepts, and technical write-ups.</p>
  <div class="card-list">
{cards_html}
  </div>
</main>

<script src="../js/nav.js"></script>
</body>
</html>
"""


def main():
    os.makedirs(POSTS_OUT, exist_ok=True)

    md_files = sorted(
        [f for f in os.listdir(POSTS_SRC) if f.endswith(".md")],
        reverse=True  # newest first
    )

    if not md_files:
        print(f"No .md files found in '{POSTS_SRC}/'")
        return

    cards = []

    for fname in md_files:
        path = os.path.join(POSTS_SRC, fname)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        # strip Jekyll front matter (--- ... ---)
        raw = re.sub(r"^---[\s\S]+?---\n?", "", raw).strip()

        date_str, title, slug, sort_key = parse_filename(fname)
        tag = guess_tag(fname)

        # generate individual post page
        body_html = md_to_html_body(raw)
        page_html = post_page(title, date_str, tag, body_html)
        out_path = os.path.join(POSTS_OUT, f"{slug}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(page_html)
        print(f"  ✓ {fname}  →  {out_path}")

        cards.append((sort_key, card_html(date_str, tag, title, slug)))

    # sort cards newest-first
    cards.sort(key=lambda x: x[0], reverse=True)
    cards_html = "\n".join(c for _, c in cards)

    # write posts index
    with open(POSTS_INDEX, "w", encoding="utf-8") as f:
        f.write(posts_index(cards_html))
    print(f"\n  ✓ Updated {POSTS_INDEX} with {len(cards)} posts")
    print("\nDone! Commit and push to deploy.")


if __name__ == "__main__":
    main()
