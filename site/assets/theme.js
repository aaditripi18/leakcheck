// Runs in <head> so the page never flashes the wrong theme. Saved choice wins,
// otherwise follow the system setting. Storage may be blocked; that is fine.
(function () {
  var saved = null;
  try { saved = localStorage.getItem("leakcheck-theme"); } catch (e) {}
  var dark = saved ? saved === "dark" : !!(window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches);
  if (dark) document.documentElement.classList.add("dark");
})();
