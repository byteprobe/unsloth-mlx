/* ===== MLX-Tune Docs - Main JS ===== */

(function () {
  'use strict';

  // --- Theme Toggle ---
  const THEME_KEY = 'mlx-tune-theme';

  function getPreferredTheme() {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored) return stored;
    return 'light'; // default to light
  }

  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_KEY, theme);
    const btn = document.querySelector('.theme-toggle');
    if (btn) btn.textContent = theme === 'dark' ? '☀️' : '🌙';
  }

  // Apply theme immediately (before DOM ready to avoid flash)
  setTheme(getPreferredTheme());

  document.addEventListener('DOMContentLoaded', function () {
    // Theme toggle button
    const themeBtn = document.querySelector('.theme-toggle');
    if (themeBtn) {
      themeBtn.textContent = getPreferredTheme() === 'dark' ? '☀️' : '🌙';
      themeBtn.addEventListener('click', function () {
        const current = document.documentElement.getAttribute('data-theme');
        setTheme(current === 'dark' ? 'light' : 'dark');
      });
    }

    // --- Mobile Menu Toggle ---
    const mobileToggle = document.querySelector('.mobile-toggle');
    const navLinks = document.querySelector('.nav-links');
    if (mobileToggle && navLinks) {
      mobileToggle.addEventListener('click', function () {
        navLinks.classList.toggle('mobile-open');
        mobileToggle.textContent = navLinks.classList.contains('mobile-open') ? '✕' : '☰';
      });
    }

    // --- Copy to Clipboard ---
    // Hero install command
    const heroInstall = document.querySelector('.hero-install');
    if (heroInstall) {
      heroInstall.addEventListener('click', function () {
        const text = heroInstall.getAttribute('data-copy') || heroInstall.textContent.trim();
        copyToClipboard(text, heroInstall);
      });
    }

    // Code block copy buttons
    document.querySelectorAll('.code-block-wrapper').forEach(function (wrapper) {
      const btn = wrapper.querySelector('.copy-btn');
      const code = wrapper.querySelector('code');
      if (btn && code) {
        btn.addEventListener('click', function () {
          copyToClipboard(code.textContent, btn, 'Copied!');
        });
      }
    });

    function copyToClipboard(text, element, tooltipText) {
      navigator.clipboard.writeText(text).then(function () {
        // Show tooltip
        const tooltip = element.querySelector('.copied-tooltip');
        if (tooltip) {
          tooltip.classList.add('show');
          setTimeout(function () { tooltip.classList.remove('show'); }, 1500);
        } else if (tooltipText) {
          const orig = element.textContent;
          element.textContent = tooltipText;
          setTimeout(function () { element.textContent = orig; }, 1500);
        }
      });
    }

    // --- Sidebar Scroll Spy ---
    const sidebarLinks = document.querySelectorAll('.sidebar-nav a');
    if (sidebarLinks.length > 0) {
      const headings = [];
      sidebarLinks.forEach(function (link) {
        const id = link.getAttribute('href');
        if (id && id.startsWith('#')) {
          const el = document.querySelector(id);
          if (el) headings.push({ el: el, link: link });
        }
      });

      function updateScrollSpy() {
        const scrollPos = window.scrollY + 120;
        let current = headings[0];
        for (var i = 0; i < headings.length; i++) {
          if (headings[i].el.offsetTop <= scrollPos) {
            current = headings[i];
          }
        }
        sidebarLinks.forEach(function (l) { l.classList.remove('active'); });
        if (current) current.link.classList.add('active');
      }

      window.addEventListener('scroll', updateScrollSpy, { passive: true });
      updateScrollSpy();
    }

    // --- Collapsible Sections ---
    document.querySelectorAll('.collapsible-header').forEach(function (header) {
      header.addEventListener('click', function () {
        header.classList.toggle('open');
        const content = header.nextElementSibling;
        if (content && content.classList.contains('collapsible-content')) {
          content.classList.toggle('open');
        }
      });
    });

    // --- Example Code Toggles ---
    document.querySelectorAll('.example-toggle').forEach(function (btn) {
      btn.addEventListener('click', function () {
        const card = btn.closest('.example-card');
        const code = card.querySelector('.example-code');
        if (code) {
          code.classList.toggle('open');
          btn.textContent = code.classList.contains('open') ? 'Hide Code' : 'View Code';
        }
      });
    });

    // --- Active Nav Link ---
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    document.querySelectorAll('.nav-links a').forEach(function (link) {
      const href = link.getAttribute('href');
      if (href === currentPage || (currentPage === '' && href === 'index.html')) {
        link.classList.add('active');
      }
    });
  });
})();
