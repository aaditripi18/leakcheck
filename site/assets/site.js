// Theme toggle and copy buttons, shared by both pages.
document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const dark = document.documentElement.classList.toggle("dark");
    try { localStorage.setItem("leakcheck-theme", dark ? "dark" : "light"); } catch (e) { /* storage blocked */ }
  });
});

document.querySelectorAll("[data-copy]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const src = document.querySelector(btn.dataset.copy);
    const label = btn.querySelector("span");
    try {
      await navigator.clipboard.writeText(src.textContent);
      if (label) label.textContent = "Copied";
    } catch (e) {
      if (label) label.textContent = "Select and copy";
    }
    setTimeout(() => { if (label) label.textContent = "Copy"; }, 1800);
  });
});

// Bar widths live in data-w: the CSP forbids inline style attributes, but allows
// setting styles from this (same-origin) script.
document.querySelectorAll("[data-w]").forEach((el) => { el.style.width = `${el.dataset.w}%`; });
