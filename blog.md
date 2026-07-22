---
title: Blog
permalink: /blog/
---

#### Blog

##### Categories

<div class="tag-row">
{% for category in site.categories %}
  <a class="badge" href="{{ '/blog/' | relative_url }}{{ category[0] }}/">{{ category[0] }}</a>
{% endfor %}
</div>

##### All Posts

<ul class="list-clean" id="all-posts-list">
{% for post in site.posts %}
  <li>
    <a href="{{ post.url | relative_url }}"><strong>{{ post.title }}</strong></a>
    <br>
    <span class="meta">{{ post.date | date: "%Y-%m-%d" }} · {{ post.categories | join: ", " }}</span>
  </li>
{% endfor %}
</ul>

<div class="pagination" id="all-posts-pagination">
  <button type="button" class="pagination-btn" data-page-nav="prev">Prev</button>
  <span class="pagination-status"></span>
  <button type="button" class="pagination-btn" data-page-nav="next">Next</button>
</div>

<script>
(function () {
  var PAGE_SIZE = 10;
  var list = document.getElementById('all-posts-list');
  var pagination = document.getElementById('all-posts-pagination');
  if (!list || !pagination) return;

  var items = Array.prototype.slice.call(list.children);
  var totalPages = Math.max(1, Math.ceil(items.length / PAGE_SIZE));
  var status = pagination.querySelector('.pagination-status');
  var prevBtn = pagination.querySelector('[data-page-nav="prev"]');
  var nextBtn = pagination.querySelector('[data-page-nav="next"]');
  var currentPage = 1;

  function pageFromHash() {
    var match = location.hash.match(/all-posts-page-(\d+)/);
    var page = match ? parseInt(match[1], 10) : 1;
    return Math.min(Math.max(page, 1), totalPages);
  }

  function render(page) {
    currentPage = page;
    items.forEach(function (item, index) {
      item.style.display = (Math.floor(index / PAGE_SIZE) + 1 === page) ? '' : 'none';
    });
    status.textContent = page + ' / ' + totalPages;
    prevBtn.disabled = page <= 1;
    nextBtn.disabled = page >= totalPages;
  }

  pagination.style.display = totalPages > 1 ? 'flex' : 'none';
  render(pageFromHash());

  prevBtn.addEventListener('click', function () {
    if (currentPage > 1) { location.hash = 'all-posts-page-' + (currentPage - 1); render(currentPage - 1); }
  });
  nextBtn.addEventListener('click', function () {
    if (currentPage < totalPages) { location.hash = 'all-posts-page-' + (currentPage + 1); render(currentPage + 1); }
  });
})();
</script>
