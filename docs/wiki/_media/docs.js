// Add language labels and accessible copy controls to highlighted examples.
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("div.highlight").forEach((block) => {
    const code = block.querySelector("pre");
    if (!code) return;
    const wrapper = block.parentElement;
    const languageClass = [...wrapper.classList].find((name) => name.startsWith("highlight-"));
    const language = languageClass ? languageClass.slice("highlight-".length) : "text";
    const label = document.createElement("span");
    label.className = "code-language";
    label.textContent = {bash: "Bash", python: "Python", bibtex: "BibTeX", default: "Python"}[language] || language;
    wrapper.appendChild(label);

    const button = document.createElement("button");
    button.type = "button";
    button.className = "code-copy";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy code example");
    button.setAttribute("aria-live", "polite");
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = "Copied";
      } catch {
        button.textContent = "Copy unavailable";
      }
      window.setTimeout(() => { button.textContent = "Copy"; }, 2000);
    });
    wrapper.appendChild(button);
  });
});
