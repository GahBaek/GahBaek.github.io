---
title: Paper Reviews
permalink: /paper-reviews/
---

# Paper Reviews

{% assign reviews = site.paper_reviews | sort: "date" | reverse %}

{% for review in reviews %}
<div class="card">
  <h2><a href="{{ review.url | relative_url }}">{{ review.title }}</a></h2>
  {% if review.paper %}<p class="meta">{{ review.paper }}</p>{% endif %}
  <div class="tag-row">
    {% for tag in review.tags %}
      <span class="badge">{{ tag }}</span>
    {% endfor %}
  </div>
  {% if review.excerpt %}<p>{{ review.excerpt }}</p>{% endif %}
</div>
{% endfor %}
