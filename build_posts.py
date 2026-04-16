#!/usr/bin/env python3
"""
build.py  --  Run from the ROOT of your site:
    python3 build.py

Folder structure:
    posts_content/    <- study notes .md files
    papers_content/   <- paper review .md files
    blog_content/     <- blog post .md files

Generates:
    pages/posts/      <- individual post HTML pages
    pages/papers/     <- individual paper HTML pages
    pages/blog/       <- individual blog HTML pages
    pages/posts.html  <- card index with category filter + pagination
    pages/papers.html <- card index with category filter + pagination
    pages/blog.html   <- card index with category filter + pagination
"""

import os
import re

# -- TAG MAPS ----------------------------------------------------------------
POSTS_TAG_MAP = {
    "ctf":         "CTF",
    "pwn":         "CTF",
    "shell":       "CTF",
    "compiler":    "Compiler",
    "frontend":    "Compiler",
    "backend":     "Compiler",
    "optimizing":  "Compiler",
    "andrew":      "Andrew.Ng",
    "cnn":         "DL",
    "lab":         "DL",
    "overfitting": "DL",
    "relu":        "DL",
    "xor":         "DL",
    "softmax":     "DL",
    "improving":   "DL",
    "lec":         "ML",
    "머신러닝":     "ML",
    "데이터":       "ML",
    "카운트":       "NLP",
    "언어":         "NLP",
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

BLOG_TAG_MAP = {
    "life":     "Life",
    "research": "Research",
    "review":   "Review",
    "travel":   "Travel",
    "essay":    "Essay",
    "opinion":  "Opinion",
}
# ---------------------------------------------------------------------------


def guess_tag(filename, tag_map, default="Note"):
    low = filename.lower()
    for key, tag in tag_map.items():
        if key in low:
            return tag
    return default


def parse_filename(fname):
    """YYYY-MM-DD-title.md -> (date_str, title, slug, sort_key)"""
    base = fname.replace(".md", "")
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})-(.*)", base)
    if m:
        year, month, day, rest = m.groups()
        title = rest.replace("-", " ").replace("_", " ").strip()
        date_str = year + "\u00b7" + month
        slug = re.sub(r"[^\w\uAC00-\uD7A3-]", "", rest.replace(" ", "-"))
        return date_str, title, slug, year + "-" + month + "-" + day
    return "-", base, re.sub(r"\s+", "-", base), "0000-00-00"


def inline(text):
    """Apply inline Markdown: images, links, bold, italic, inline code."""
    text = re.sub(
        r"!\[(.+?)\]\((.+?)\)",
        r'<img src="\2" alt="\1" style="max-width:100%;border-radius:6px;margin:1rem 0;">',
        text,
    )
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         text)
    text = re.sub(r"`(.+?)`",       r"<code>\1</code>",      text)
    return text


def md_to_html_body(md_text):
    """Markdown -> HTML body."""
    lines = md_text.split("\n")
    html = []
    in_code = False
    in_ul = False
    in_ol = False

    for line in lines:
        # fenced code block
        if line.startswith("```"):
            if not in_code:
                lang = line[3:].strip() or ""
                html.append('<pre><code class="language-' + lang + '">')
                in_code = True
            else:
                html.append("</code></pre>")
                in_code = False
            continue

        if in_code:
            html.append(
                line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            continue

        # close open lists
        if in_ul and not re.match(r"^[-*] ", line):
            html.append("</ul>")
            in_ul = False
        if in_ol and not re.match(r"^\d+\. ", line):
            html.append("</ol>")
            in_ol = False

        if line.startswith("#### "):
            html.append("<h4>" + inline(line[5:]) + "</h4>")
        elif line.startswith("### "):
            html.append("<h3>" + inline(line[4:]) + "</h3>")
        elif line.startswith("## "):
            html.append("<h2>" + inline(line[3:]) + "</h2>")
        elif line.startswith("# "):
            html.append("<h2>" + inline(line[2:]) + "</h2>")
        elif line.startswith("> "):
            html.append("<blockquote><p>" + inline(line[2:]) + "</p></blockquote>")
        elif re.match(r"^[-*_]{3,}$", line.strip()):
            html.append("<hr>")
        elif re.match(r"^[-*] ", line):
            if not in_ul:
                html.append("<ul>")
                in_ul = True
            html.append("<li>" + inline(line[2:]) + "</li>")
        elif re.match(r"^\d+\. ", line):
            if not in_ol:
                html.append("<ol>")
                in_ol = True
            ol_text = re.sub(r"^\d+\. ", "", line)
            html.append("<li>" + inline(ol_text) + "</li>")
        elif line.strip() == "":
            html.append("")
        else:
            html.append("<p>" + inline(line) + "</p>")

    if in_ul:
        html.append("</ul>")
    if in_ol:
        html.append("</ol>")
    if in_code:
        html.append("</code></pre>")

    return "\n".join(html)


# -- SHARED STYLES -----------------------------------------------------------

POST_STYLES = """
    .post-header { margin-bottom: 2.5rem; }
    .post-header h1 {
      font-family: 'DM Serif Display', serif;
      font-size: 2rem; font-weight: 400; line-height: 1.2; margin-bottom: .6rem;
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
      background: #1a1814; color: #f7f5f0; border-radius: 6px;
      padding: 1.2rem; overflow-x: auto; font-size: .82rem; margin: 1.2rem 0;
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

INDEX_STYLES = """
    .filter-bar { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: 1.5rem; }
    .filter-btn {
      font-family: 'DM Mono', monospace; font-size: .72rem;
      padding: .3rem .85rem; border-radius: 999px;
      border: 1px solid var(--line); background: var(--card);
      color: var(--muted); cursor: pointer; transition: all .15s;
    }
    .filter-btn:hover { border-color: var(--ink); color: var(--ink); }
    .filter-btn.active { background: var(--ink); color: var(--bg); border-color: var(--ink); }
    .card.hidden { display: none; }
    .pagination {
      display: flex; align-items: center; justify-content: center;
      gap: .6rem; margin-top: 2rem; flex-wrap: wrap;
    }
    .page-btn, .page-num {
      font-family: 'DM Mono', monospace; font-size: .75rem;
      padding: .35rem .8rem; border-radius: 6px;
      border: 1px solid var(--line); text-decoration: none;
      color: var(--muted); background: var(--card); transition: all .15s;
    }
    .page-btn:hover, .page-num:hover { border-color: var(--ink); color: var(--ink); }
    .page-num.active { background: var(--ink); color: var(--bg); border-color: var(--ink); }
"""

FILTER_JS = """<script>
  document.querySelectorAll('.filter-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
      document.querySelectorAll('.filter-btn').forEach(function(b) {
        b.classList.remove('active');
      });
      btn.classList.add('active');
      var tag = btn.getAttribute('data-tag');
      document.querySelectorAll('.card').forEach(function(card) {
        if (tag === 'all' || card.getAttribute('data-tag') === tag) {
          card.classList.remove('hidden');
        } else {
          card.classList.add('hidden');
        }
      });
    });
  });
</script>"""


# -- HTML BUILDERS -----------------------------------------------------------

def make_nav(nav_active, depth=".."):
    papers_cls = ' class="active"' if nav_active == "papers" else ""
    posts_cls  = ' class="active"' if nav_active == "posts"  else ""
    blog_cls   = ' class="active"' if nav_active == "blog"   else ""
    return (
        "<nav>\n"
        '  <a class="nav-logo" href="' + depth + '/index.html">Gahyun Baek</a>\n'
        '  <ul class="nav-links">\n'
        '    <li><a href="' + depth + '/index.html">Home</a></li>\n'
        '    <li><a href="' + depth + '/pages/papers.html"' + papers_cls + '>Papers</a></li>\n'
        '    <li><a href="' + depth + '/pages/posts.html"'  + posts_cls  + '>Posts</a></li>\n'
        '    <li><a href="' + depth + '/pages/blog.html"'   + blog_cls   + '>Blog</a></li>\n'
        '    <li><a href="' + depth + '/pages/cv.html">CV</a></li>\n'
        "  </ul>\n"
        "</nav>"
    )


def render_post_page(title, date_str, tag, body_html, back_label, back_href, nav_active):
    nav = make_nav(nav_active, depth="../..")
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="UTF-8" />\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>\n'
        "  <title>" + title + " \u00b7 Gahyun Baek</title>\n"
        '  <link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">\n'
        '  <link rel="stylesheet" href="../../css/style.css">\n'
        "  <style>" + POST_STYLES + "</style>\n"
        "</head>\n"
        "<body>\n"
        + nav + "\n"
        "<main class=\"page\">\n"
        '  <a class="back-link" href="' + back_href + '">\u2190 Back to ' + back_label + "</a>\n"
        '  <div class="post-header">\n'
        "    <h1>" + title + "</h1>\n"
        '    <div class="post-meta">\n'
        "      <span>" + date_str + "</span>\n"
        '      <span class="post-tag">' + tag + "</span>\n"
        "    </div>\n"
        "  </div>\n"
        '  <hr class="post-divider">\n'
        '  <div class="post-body">\n'
        + body_html + "\n"
        "  </div>\n"
        "</main>\n"
        '<script src="../../js/nav.js"></script>\n'
        "</body>\n"
        "</html>\n"
    )


def render_card(date_str, tag, title, href):
    return (
        '    <a class="card" href="' + href + '" data-tag="' + tag + '">\n'
        '      <div class="card-meta">\n'
        '        <div class="card-date">' + date_str + "</div>\n"
        '        <div class="card-tag">' + tag + "</div>\n"
        "      </div>\n"
        '      <div class="card-body">\n'
        "        <h3>" + title + "</h3>\n"
        "      </div>\n"
        "    </a>"
    )


def render_index_page(page_title, page_sub, all_cards, index_path,
                      nav_active, section_slug, per_page=10):

    # unique tags across ALL cards for the filter bar
    all_tags = sorted(set(tag for tag, _, _ in all_cards))

    filter_btns = '    <button class="filter-btn active" data-tag="all">All</button>\n'
    for t in all_tags:
        filter_btns += (
            '    <button class="filter-btn" data-tag="' + t + '">' + t + "</button>\n"
        )

    total = len(all_cards)
    total_pages = max(1, -(-total // per_page))

    for page_num in range(1, total_pages + 1):
        start = (page_num - 1) * per_page
        chunk = all_cards[start : start + per_page]
        cards_html = "\n".join(card_str for _, _, card_str in chunk)

        # prev / next links
        prev_link = ""
        next_link = ""
        if page_num > 1:
            if page_num == 2:
                prev_href = section_slug + ".html"
            else:
                prev_href = section_slug + "-page" + str(page_num - 1) + ".html"
            prev_link = '<a class="page-btn" href="' + prev_href + '">\u2190 Prev</a>'
        if page_num < total_pages:
            next_href = section_slug + "-page" + str(page_num + 1) + ".html"
            next_link = '<a class="page-btn" href="' + next_href + '">Next \u2192</a>'

        # page number buttons
        page_btns = ""
        for i in range(1, total_pages + 1):
            if i == 1:
                href = section_slug + ".html"
            else:
                href = section_slug + "-page" + str(i) + ".html"
            active_cls = " active" if i == page_num else ""
            page_btns += (
                '<a class="page-num' + active_cls + '" href="' + href + '">' + str(i) + "</a>"
            )

        pagination = ""
        if total_pages > 1:
            pagination = (
                "\n  <div class=\"pagination\">\n"
                "    " + prev_link + "\n"
                '    <div class="page-nums">' + page_btns + "</div>\n"
                "    " + next_link + "\n"
                "  </div>"
            )

        nav = make_nav(nav_active, depth="..")

        html = (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '  <meta charset="UTF-8" />\n'
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>\n'
            "  <title>" + page_title + " \u00b7 Gahyun Baek</title>\n"
            '  <link rel="preconnect" href="https://fonts.googleapis.com">\n'
            '  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">\n'
            '  <link rel="stylesheet" href="../css/style.css">\n'
            "  <style>" + INDEX_STYLES + "</style>\n"
            "</head>\n"
            "<body>\n"
            + nav + "\n"
            "<main class=\"page\">\n"
            "  <h1 class=\"page-title\">" + page_title + "</h1>\n"
            "  <p class=\"page-sub\">" + page_sub + "</p>\n"
            '  <div class="filter-bar">\n'
            + filter_btns
            + "  </div>\n"
            '  <div class="card-list">\n'
            + cards_html + "\n"
            "  </div>"
            + pagination + "\n"
            "</main>\n"
            '<script src="../js/nav.js"></script>\n'
            + FILTER_JS + "\n"
            "</body>\n"
            "</html>\n"
        )

        if page_num == 1:
            out_path = index_path
        else:
            base_dir = os.path.dirname(index_path)
            out_path = os.path.join(base_dir, section_slug + "-page" + str(page_num) + ".html")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

    print("  ✓  " + index_path + "  (" + str(total) + " entries, " + str(total_pages) + " page(s))\n")


# -- SECTION BUILDER ---------------------------------------------------------

def build_section(src_dir, out_dir, index_path,
                  page_title, page_sub, nav_active,
                  tag_map, tag_default, back_label):

    if not os.path.isdir(src_dir):
        print("  [skip] '" + src_dir + "/' not found.")
        return

    os.makedirs(out_dir, exist_ok=True)

    md_files = sorted(
        [f for f in os.listdir(src_dir) if f.endswith(".md")],
        reverse=True,
    )

    if not md_files:
        print("  [skip] No .md files in '" + src_dir + "/'")
        return

    section_slug = os.path.basename(out_dir)
    cards = []  # (sort_key, tag, card_html)

    for fname in md_files:
        path = os.path.join(src_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        # strip Jekyll front matter
        raw = re.sub(r"^---[\s\S]+?---\n?", "", raw).strip()

        # fix Jekyll relative_url syntax
        raw = re.sub(
            r"\{\{\s*'(/[^']+)'\s*\|\s*relative_url\s*\}\}",
            r"../../\1",
            raw,
        )

        date_str, title, slug, sort_key = parse_filename(fname)
        tag = guess_tag(fname, tag_map, tag_default)

        body_html = md_to_html_body(raw)
        back_href = "../" + section_slug + ".html"
        page_html = render_post_page(
            title, date_str, tag, body_html, back_label, back_href, nav_active
        )

        out_path = os.path.join(out_dir, slug + ".html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(page_html)
        print("  ✓  " + fname)

        card_href = section_slug + "/" + slug + ".html"
        cards.append((sort_key, tag, render_card(date_str, tag, title, card_href)))

    cards.sort(key=lambda x: x[0], reverse=True)
    render_index_page(
        page_title, page_sub, cards, index_path,
        nav_active, section_slug, per_page=10
    )


# -- MAIN --------------------------------------------------------------------

def main():
    print("\n-- Building Posts ------------------------------------------")
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

    print("-- Building Paper Reviews ----------------------------------")
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

    print("-- Building Blog -------------------------------------------")
    build_section(
        src_dir    = "blog_content",
        out_dir    = "pages/blog",
        index_path = "pages/blog.html",
        page_title = "Blog",
        page_sub   = "Personal essays, thoughts, and reflections.",
        nav_active = "blog",
        tag_map    = BLOG_TAG_MAP,
        tag_default= "Essay",
        back_label = "Blog",
    )

    print("Done!  git add . && git commit -m 'update' && git push")


if __name__ == "__main__":
    main()
