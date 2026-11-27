---
title: Paper Reviews
permalink: /paper-reviews/
---

<header class="page-heading">
  <h1>Paper Reviews</h1>
  <p>Notes and takeaways from papers on AI and software security.</p>
</header>

{% assign reviews = site.paper_reviews | sort: "date" | reverse %}

<ul class="entry-list">
{% for review in reviews %}
  <li><a href="{{ review.url | relative_url }}">
    <time datetime="{{ review.date | date_to_xmlschema }}">{{ review.date | date: "%Y.%m.%d" }}</time>
    <span class="title">{{ review.title }}</span>
    <span class="tag"></span>
  </a></li>
{% endfor %}
</ul>
