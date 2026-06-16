---
title: Blog
permalink: /blog/
---

# Blog

## Categories

{% for category in site.categories %}
<section class="card">
  <h2>{{ category[0] }}</h2>
  <ul class="list-clean">
    {% for post in category[1] %}
      <li>
        <a href="{{ post.url | relative_url }}"><strong>{{ post.title }}</strong></a>
        <br>
        <span class="meta">{{ post.date | date: "%Y-%m-%d" }}</span>
      </li>
    {% endfor %}
  </ul>
</section>
{% endfor %}

## All Posts

<ul class="list-clean">
{% for post in site.posts %}
  <li>
    <a href="{{ post.url | relative_url }}"><strong>{{ post.title }}</strong></a>
    <br>
    <span class="meta">{{ post.date | date: "%Y-%m-%d" }} · {{ post.categories | join: ", " }}</span>
  </li>
{% endfor %}
</ul>
