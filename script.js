const input = document.getElementById("text-input");
const count = document.getElementById("count");
const button = document.getElementById("analyze-btn");
const error = document.getElementById("error");
const result = document.getElementById("result");

input.addEventListener("input", () => count.textContent = input.value.length);

button.addEventListener("click", async () => {
  const text = input.value.trim();
  error.textContent = "";
  if (!text) { error.textContent = "Please enter some text to analyze."; return; }

  button.disabled = true;
  button.textContent = "Analyzing...";
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");

    document.getElementById("sentiment").textContent = data.sentiment;
    document.getElementById("confidence").textContent = data.confidence + "%";
    document.getElementById("emoji").textContent =
      data.sentiment === "Positive" ? "😊" :
      data.sentiment === "Negative" ? "😢" : "😐";

    const scores = document.getElementById("scores");
    scores.innerHTML = "";
    Object.entries(data.scores).forEach(([label, value]) => {
      const row = document.createElement("div");
      row.className = "score-row";
      const name = document.createElement("span");
      name.textContent = label;
      const track = document.createElement("div");
      track.className = "track";
      const fill = document.createElement("div");
      fill.className = "fill";
      fill.style.width = value + "%";
      track.appendChild(fill);
      const percent = document.createElement("strong");
      percent.textContent = value + "%";
      row.append(name, track, percent);
      scores.appendChild(row);
    });
    result.classList.remove("hidden");
  } catch (err) {
    error.textContent = err.message;
  } finally {
    button.disabled = false;
    button.innerHTML = 'Analyze sentiment <span>→</span>';
  }
});
