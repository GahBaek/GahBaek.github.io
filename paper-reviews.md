---
title: Paper Reviews
permalink: /paper-reviews/
---

<header class="page-heading">
  <p class="eyebrow">Reading notes</p>
  <h1>Paper Reviews</h1>
  <p class="lead">Notes and takeaways from papers on AI and software security.</p>
</header>

{% assign reviews = site.paper_reviews | sort: "date" | reverse %}

<div class="review-list">
{% for review in reviews %}
  <a class="review-item" href="{{ review.url | relative_url }}">
    <time class="review-date">{{ review.date | date: "%Y.%m.%d" }}</time>
    <span class="review-title">{{ review.title }}</span>
    <span class="review-arrow" aria-hidden="true">→</span>
  </a>
{% endfor %}
</div>
