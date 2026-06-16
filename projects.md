---
title: Projects
permalink: /projects/
---

# Projects

{% for project in site.data.projects %}
<div class="card">
  <h2>{{ project.title }}</h2>
  <p>{{ project.description }}</p>

  <div class="tag-row">
    {% for tag in project.tags %}
      <span class="badge">{{ tag }}</span>
    {% endfor %}
  </div>

  {% if project.github and project.github != "" %}
    <p><a href="{{ project.github }}">GitHub Repository →</a></p>
  {% endif %}

  {% if project.demo and project.demo != "" %}
    <p><a href="{{ project.demo }}">Demo →</a></p>
  {% endif %}
</div>
{% endfor %}
