document.addEventListener("DOMContentLoaded", function () {
  let currentFile = null;
  let objectUrl = null;
  let analysisHistory = [];

  const fileInput = document.getElementById("file-input");
  const dropZone = document.getElementById("drop-zone");
  const responseUrlInput = document.getElementById("response-url");
  const analyzeBtn = document.getElementById("analyze-btn");
  const resultsSection = document.getElementById("results-section");
  const fileInfoDisplay = document.getElementById("file-info");
  const videoPlayer = document.getElementById("video-player");
  const loadingOverlay = document.getElementById("loading-overlay");

  initEventListeners();

  function initEventListeners() {
    if (fileInput) {
      fileInput.addEventListener("change", function (event) {
        if (event.target.files && event.target.files[0]) {
          handleFileSelect(event.target.files[0]);
        }
      });
    }

    if (dropZone) {
      dropZone.addEventListener("dragover", function (event) {
        event.preventDefault();
        dropZone.classList.add("drag-over");
      });

      dropZone.addEventListener("dragleave", function () {
        dropZone.classList.remove("drag-over");
      });

      dropZone.addEventListener("drop", function (event) {
        event.preventDefault();
        dropZone.classList.remove("drag-over");

        if (event.dataTransfer.files.length > 0) {
          handleFileSelect(event.dataTransfer.files[0]);
        }
      });
    }

    updateAnalyzeButton();
    loadHistory();
  }

  function handleFileSelect(file) {
    if (!file.type.startsWith("video/")) {
      alert("Пожалуйста, выберите видеофайл");
      return;
    }

    const maxSize = 200 * 1024 * 1024;
    if (file.size > maxSize) {
      alert("Файл слишком большой. Максимальный размер: 200MB");
      return;
    }

    currentFile = file;

    if (fileInfoDisplay) {
      fileInfoDisplay.innerHTML = `
        <div class="file-selected">
          <div class="file-header">
            <strong>Выбран файл:</strong> ${escapeHtml(file.name)}
            <button class="remove-file-btn" onclick="removeFile()">× Удалить</button>
          </div>
          <small>Размер: ${formatFileSize(file.size)}</small>
        </div>
      `;
    }

    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
    }

    objectUrl = URL.createObjectURL(file);
    createVideoPreview(objectUrl);
    showResultsSection();
    updateAnalyzeButton();
  }

  function createVideoPreview(src) {
    if (!videoPlayer) {
      return;
    }

    videoPlayer.src = src;
    videoPlayer.load();
    resetCurrentAnalysis();
  }

  function removeFile() {
    currentFile = null;

    if (fileInfoDisplay) {
      fileInfoDisplay.innerHTML = "";
    }

    if (fileInput) {
      fileInput.value = "";
    }

    if (videoPlayer) {
      videoPlayer.src = "";
    }

    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }

    updateAnalyzeButton();
  }

  function clearResponseUrl() {
    if (responseUrlInput) {
      responseUrlInput.value = "";
    }
  }

  function showResultsSection() {
    if (resultsSection) {
      resultsSection.style.display = "block";
    }
  }

  async function analyzeVideo() {
    if (!currentFile) {
      alert("Выберите видеофайл перед запуском анализа");
      return;
    }

    showResultsSection();
    showLoading(true);
    resetCurrentAnalysis();
    addLogEntry(`Загрузка файла: ${currentFile.name}`);

    try {
      const callbackUrl = responseUrlInput ? responseUrlInput.value.trim() : "";
      const uploadResult = await uploadVideo(currentFile, callbackUrl);

      updateStatus("analysis-status", "В очереди", "info");
      addLogEntry(`Создана задача: ${uploadResult.id}`);

      const finalStatus = await waitForFinalStatus(uploadResult.id);
      handleAnalysisResponse(finalStatus, currentFile.name);
      saveToHistory(finalStatus, currentFile.name, true);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      updateStatus("analysis-status", "Ошибка", "danger");
      addLogEntry(`Ошибка: ${message}`);
      alert(`Ошибка обработки: ${message}`);
    } finally {
      showLoading(false);
    }
  }

  async function uploadVideo(file, callbackUrl) {
    const formData = new FormData();
    formData.append("data", file);

    if (callbackUrl) {
      formData.append("response_url", callbackUrl);
    }

    const response = await fetch("/api/v1/Klin/upload", {
      method: "POST",
      body: formData,
    });

    const payload = await readJsonResponse(response);
    if (!response.ok) {
      throw new Error(payload.detail || `HTTP ${response.status}`);
    }

    return payload;
  }

  async function waitForFinalStatus(klinId) {
    const startTime = Date.now();
    const timeoutMs = 10 * 60 * 1000;
    const intervalMs = 2000;

    while (Date.now() - startTime < timeoutMs) {
      const response = await fetch(`/api/v1/Klin/${klinId}`);
      const payload = await readJsonResponse(response);

      if (!response.ok) {
        throw new Error(payload.detail || `HTTP ${response.status}`);
      }

      if (payload.state === "FINISHED" || payload.state === "ERROR") {
        return payload;
      }

      updateStatus("analysis-status", `Обработка (${payload.state})`, "info");
      await sleep(intervalMs);
    }

    throw new Error("Таймаут ожидания результата анализа");
  }

  function handleAnalysisResponse(data, sourceName) {
    const state = data.state || "UNKNOWN";

    if (state === "ERROR") {
      updateStatus("analysis-status", "Ошибка", "danger");
      updateStatus("prediction-result", "ОШИБКА", "danger");
      updateStatus("confidence-level", "0%", "info");
      addLogEntry(`Обработка завершилась ошибкой: ${data.mae || "unknown error"}`);
      return;
    }

    updateStatus("analysis-status", "Завершено", "success");

    const maeResults = parseMaeResults(data.mae);
    const bestResult = pickBestMaeResult(maeResults);

    const prediction = bestResult ? String(bestResult.answer) : "НЕИЗВЕСТНО";
    const confidence = bestResult
      ? normalizeConfidence(bestResult.confident) * 100
      : 0;

    updateStatus("prediction-result", prediction.toUpperCase(), "success");
    updateStatus("confidence-level", `${confidence.toFixed(1)}%`, "info");

    const totalFramesEl = document.getElementById("total-frames");
    if (totalFramesEl) {
      totalFramesEl.textContent = String(maeResults.length);
    }

    const durationEl = document.getElementById("video-duration");
    if (durationEl) {
      durationEl.textContent = "--";
    }

    const fpsEl = document.getElementById("video-fps");
    if (fpsEl) {
      fpsEl.textContent = "--";
    }

    const processingTimeEl = document.getElementById("processing-time");
    if (processingTimeEl) {
      processingTimeEl.textContent = "--";
    }

    addLogEntry(`✓ Файл: ${sourceName}`);
    addLogEntry(`✓ Класс: ${prediction}`);
    addLogEntry(`✓ Уверенность: ${confidence.toFixed(1)}%`);
    addLogEntry("--------------------------------------------------");
  }

  function parseMaeResults(rawMae) {
    if (!rawMae) {
      return [];
    }

    try {
      const parsed = JSON.parse(rawMae);
      return Array.isArray(parsed) ? parsed : [];
    } catch (_error) {
      return [];
    }
  }

  function pickBestMaeResult(results) {
    if (!results.length) {
      return null;
    }

    return results.reduce((best, current) => {
      const currentConfidence = normalizeConfidence(current.confident);
      const bestConfidence = normalizeConfidence(best.confident);
      return currentConfidence > bestConfidence ? current : best;
    });
  }

  function normalizeConfidence(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric < 0) {
      return 0;
    }
    if (numeric > 1) {
      return numeric / 100;
    }
    return numeric;
  }

  async function readJsonResponse(response) {
    try {
      return await response.json();
    } catch (_error) {
      return {};
    }
  }

  function saveToHistory(data, sourceName, isFile = true) {
    const historyEntry = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      sourceName,
      isFile,
      data,
      state: data.state,
    };

    analysisHistory.unshift(historyEntry);
    if (analysisHistory.length > 10) {
      analysisHistory = analysisHistory.slice(0, 10);
    }

    try {
      localStorage.setItem("klin_analysis_history", JSON.stringify(analysisHistory));
    } catch (_error) {
      // silently ignore localStorage errors
    }
  }

  function loadHistory() {
    try {
      const saved = localStorage.getItem("klin_analysis_history");
      if (saved) {
        analysisHistory = JSON.parse(saved);
      }
    } catch (_error) {
      analysisHistory = [];
    }
  }

  function updateStatus(elementId, text, type = "normal") {
    const element = document.getElementById(elementId);
    if (!element) {
      return;
    }

    element.textContent = text;
    element.className = "status-value";

    if (type === "success") {
      element.classList.add("success");
    } else if (type === "danger") {
      element.classList.add("danger");
    } else if (type === "info") {
      element.classList.add("info");
    }
  }

  function addLogEntry(message) {
    const logContainer = document.getElementById("event-log");
    if (!logContainer) {
      return;
    }

    const time = new Date().toLocaleTimeString();
    const entry = document.createElement("div");
    entry.className = "log-entry";
    entry.innerHTML = `<span class="log-time">${time}</span><span class="log-message">${escapeHtml(message)}</span>`;
    logContainer.appendChild(entry);
    logContainer.scrollTop = logContainer.scrollHeight;
  }

  function showLoading(show) {
    if (!loadingOverlay) {
      return;
    }

    loadingOverlay.style.display = show ? "flex" : "none";
  }

  function updateAnalyzeButton() {
    if (!analyzeBtn) {
      return;
    }

    analyzeBtn.disabled = !currentFile;
    analyzeBtn.textContent = currentFile
      ? "Загрузить и обработать"
      : "Загрузить и обработать";
  }

  function resetCurrentAnalysis() {
    updateStatus("analysis-status", "Ожидание данных");
    updateStatus("prediction-result", "--");
    updateStatus("confidence-level", "--");

    const processingTimeEl = document.getElementById("processing-time");
    const durationEl = document.getElementById("video-duration");
    const totalFramesEl = document.getElementById("total-frames");
    const fpsEl = document.getElementById("video-fps");

    if (processingTimeEl) {
      processingTimeEl.textContent = "--";
    }
    if (durationEl) {
      durationEl.textContent = "--";
    }
    if (totalFramesEl) {
      totalFramesEl.textContent = "--";
    }
    if (fpsEl) {
      fpsEl.textContent = "--";
    }
  }

  function clearLog() {
    const logContainer = document.getElementById("event-log");
    if (logContainer) {
      logContainer.innerHTML = "";
    }
  }

  function formatFileSize(bytes) {
    if (bytes === 0) {
      return "0 Байт";
    }

    const k = 1024;
    const sizes = ["Байт", "КБ", "МБ", "ГБ"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function sleep(ms) {
    return new Promise((resolve) => {
      setTimeout(resolve, ms);
    });
  }

  window.analyzeVideo = analyzeVideo;
  window.removeFile = removeFile;
  window.clearResponseUrl = clearResponseUrl;
  window.clearLog = clearLog;
});
