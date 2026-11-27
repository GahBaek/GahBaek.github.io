---
title: Home
permalink: /
---

<section class="intro">
  <div>
    <h1>백가현<span class="en">Gahyun Baek</span></h1>
    <p class="role">M.S. student in Software, Sungkyunkwan University<br>SecAI Lab · AI for Security</p>
    <p class="bio">I study security, focusing on web security, program analysis, software security, and network security.</p>
    <ul class="link-row">
      <li><a href="mailto:gusqorrk@gmail.com">Email</a></li>
      <li><a href="https://github.com/gahyunBaek" target="_blank" rel="noopener">GitHub</a></li>
      <li><a href="https://www.linkedin.com/in/gahyun-baek/" target="_blank" rel="noopener">LinkedIn</a></li>
      <li><a href="{{ '/cv/' | relative_url }}">CV</a></li>
    </ul>
  </div>
  <div class="portrait-cell">
    <img class="portrait" src="{{ '/assets/files/Gahyun_Baek.jpg' | relative_url }}" alt="Portrait of Gahyun Baek">
  </div>
</section>

<section class="section">
  <div class="section-head">
    <h2>Recent posts</h2>
    <a href="{{ '/blog/' | relative_url }}">All posts →</a>
  </div>
  <ul class="entry-list">
  {% for post in site.posts limit: 5 %}
    <li><a href="{{ post.url | relative_url }}">
      <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%Y.%m.%d" }}</time>
      <span class="title">{{ post.title }}</span>
      <span class="tag">{{ post.categories | first }}</span>
    </a></li>
  {% endfor %}
  </ul>
</section>

{% assign reviews = site.paper_reviews | sort: "date" | reverse %}
<section class="section">
  <div class="section-head">
    <h2>Paper reviews</h2>
    <a href="{{ '/paper-reviews/' | relative_url }}">All reviews →</a>
  </div>
  <ul class="entry-list stacked">
  {% for review in reviews limit: 3 %}
    <li><a href="{{ review.url | relative_url }}">
      <span class="title">{{ review.title }}</span>
      <span class="entry-meta">
        <time datetime="{{ review.date | date_to_xmlschema }}">{{ review.date | date: "%Y.%m.%d" }}</time>
        {% if review.pdf %}<span>Slides</span>{% endif %}
      </span>
    </a></li>
  {% endfor %}
  </ul>
</section>
