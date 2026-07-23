---
title: Guestbook
permalink: /guestbook/
---

#### Guestbook

<p class="meta">방문 기록이나 하고 싶은 말을 자유롭게 남겨주세요. (GitHub 계정으로 로그인 후 작성할 수 있어요)</p>

<div id="giscus-container" class="giscus-container"></div>

<script>
(function () {
  var theme = document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
  var script = document.createElement('script');
  script.src = 'https://giscus.app/client.js';
  script.setAttribute('data-repo', 'GahBaek/GahBaek.github.io');
  script.setAttribute('data-repo-id', 'R_kgDOTdm8JQ');
  script.setAttribute('data-category', 'Guestbook');
  script.setAttribute('data-category-id', 'DIC_kwDOTdm8Jc4DByJK');
  script.setAttribute('data-mapping', 'specific');
  script.setAttribute('data-term', 'guestbook');
  script.setAttribute('data-strict', '0');
  script.setAttribute('data-reactions-enabled', '1');
  script.setAttribute('data-emit-metadata', '0');
  script.setAttribute('data-input-position', 'bottom');
  script.setAttribute('data-theme', theme);
  script.setAttribute('data-lang', 'en');
  script.crossOrigin = 'anonymous';
  script.async = true;
  document.getElementById('giscus-container').appendChild(script);
})();
</script>
