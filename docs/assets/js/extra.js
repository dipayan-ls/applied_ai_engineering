document$.subscribe(function () {
  // Animate elements on scroll
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) e.target.classList.add("visible");
      });
    },
    { threshold: 0.1 }
  );
  document.querySelectorAll(".volume-card, .stat-tile, .path-step").forEach((el) => {
    observer.observe(el);
  });
});
