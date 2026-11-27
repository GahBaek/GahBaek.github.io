---
title: Paper Reviews
permalink: /paper-reviews/
---

<header class="page-heading">
  <h1>Paper Reviews</h1>
  <p>Notes and takeaways from papers on AI and software security.</p>
</header>

{% assign reviews = site.paper_reviews | sort: "date" | reverse %}

<ul class="entry-list stacked">
{% for review in reviews %}
  <li><a href="{{ review.url | relative_url }}">
    <span class="title">{{ review.title }}</span>
    <span class="entry-meta">
      <time datetime="{{ review.date | date_to_xmlschema }}">{{ review.date | date: "%Y.%m.%d" }}</time>
      {% if review.pdf %}<span>Slides</span>{% endif %}
    </span>
  </a></li>
{% endfor %}
</ul>
