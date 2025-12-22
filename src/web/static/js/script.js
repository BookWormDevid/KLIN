document.addEventListener('DOMContentLoaded', function () {
  // Initialize variables
  let currentFile = null;
  let currentUrl = null;
  let objectUrl = null;
  let analysisHistory = []; // История анализов

  // DOM Elements
  const fileInput = document.getElementById('file-input');
  const dropZone = document.getElementById('drop-zone');
  const videoUrlInput = document.getElementById('video-url');
  const analyzeBtn = document.getElementById('analyze-btn');
  const resultsSection = document.getElementById('results-section');
  const fileInfoDisplay = document.getElementById('file-info');
  const videoPlayer = document.getElementById('video-player');
  const loadingOverlay = document.getElementById('loading-overlay');

  // Initialize event listeners
  initEventListeners();

  function initEventListeners() {
    // File input change
    if (fileInput) {
      fileInput.addEventListener('change', function (e) {
        if (e.target.files && e.target.files[0]) {
          handleFileSelect(e.target.files[0]);
        }
      });
    }

    // Drag and drop
    if (dropZone) {
      dropZone.addEventListener('dragover', function (e) {
        e.preventDefault();
        dropZone.classList.add('drag-over');
      });

      dropZone.addEventListener('dragleave', function () {
        dropZone.classList.remove('drag-over');
      });

      dropZone.addEventListener('drop', function (e) {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
          handleFileSelect(e.dataTransfer.files[0]);
        }
      });
    }

    // URL input
    if (videoUrlInput) {
      videoUrlInput.addEventListener('input', function () {
        currentUrl = this.value.trim();
        updateAnalyzeButton();
      });
    }

    // Update button state on load
    updateAnalyzeButton();

    // Восстанавливаем историю из localStorage
    loadHistory();
  }

  function handleFileSelect(file) {
    // Validate file type
    if (!file.type.startsWith('video/')) {
      alert('Пожалуйста, выберите видеофайл (MP4, AVI, MOV, MKV)');
      return;
    }

    // Check file size (max 500MB)
    const maxSize = 500 * 1024 * 1024; // 500MB in bytes
    if (file.size > maxSize) {
      alert('Файл слишком большой. Максимальный размер: 500MB');
      return;
    }

    // Display file info with remove button
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

    // Set current file and clear URL
    currentFile = file;
    currentUrl = null;

    // Clear URL input
    if (videoUrlInput) {
      videoUrlInput.value = '';
    }

    // Create video preview
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
    }
    objectUrl = URL.createObjectURL(file);
    createVideoPreview(objectUrl);

    // Show results section immediately
    showResultsSection();

    // Update button
    updateAnalyzeButton();
  }

  function createVideoPreview(src) {
    if (!videoPlayer) return;

    videoPlayer.src = src;
    videoPlayer.load();

    // Show results section
    showResultsSection();

    // Reset current analysis results (but keep history)
    resetCurrentAnalysis();
  }

  function showResultsSection() {
    if (resultsSection) {
      resultsSection.style.display = 'block';
    }
  }

  function removeFile() {
    // Clear current file
    currentFile = null;

    // Clear file info display
    if (fileInfoDisplay) {
      fileInfoDisplay.innerHTML = '';
    }

    // Clear file input
    if (fileInput) {
      fileInput.value = '';
    }

    // Clear video player
    if (videoPlayer) {
      videoPlayer.src = '';
    }

    // Release object URL
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }

    // Update button
    updateAnalyzeButton();

    // Don't hide results section - keep history visible
    // Only hide if we have no URL either
    if (!currentUrl && !currentFile) {
      // But we might want to keep results section visible for history
      // So we'll keep it visible
    }
  }

  function clearUrl() {
    // Clear current URL
    currentUrl = null;

    // Clear URL input
    if (videoUrlInput) {
      videoUrlInput.value = '';
    }

    // Clear video player
    if (videoPlayer) {
      videoPlayer.src = '';
    }

    // Update button
    updateAnalyzeButton();
  }

  function testUrl() {
    if (!videoUrlInput) return;

    const url = videoUrlInput.value.trim();

    if (!url) {
      alert('Введите URL видео для проверки');
      return;
    }

    // Validate URL format
    if (!isValidUrl(url)) {
      alert('Некорректный URL. URL должен начинаться с http:// или https://');
      return;
    }

    // Check for video extensions
    const videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
    const hasVideoExtension = videoExtensions.some(ext => url.toLowerCase().includes(ext));

    // Also check for YouTube
    const isYouTube = url.includes('youtube.com') || url.includes('youtu.be');

    if (!hasVideoExtension && !isYouTube) {
      alert('Внимание: URL не похож на прямую ссылку на видео файл.\nСистема попытается обработать ссылку, но могут возникнуть ошибки.');
    } else {
      alert('Ссылка корректна! Нажмите "Начать анализ" для обработки видео.');
    }

    // Set current URL and clear file
    currentUrl = url;
    currentFile = null;

    // Clear file info
    if (fileInfoDisplay) {
      fileInfoDisplay.innerHTML = '';
    }

    // Clear file input
    if (fileInput) {
      fileInput.value = '';
    }

    // Clear video preview
    if (videoPlayer) {
      videoPlayer.src = '';
    }

    // Show results section for URL input
    showResultsSection();
    resetCurrentAnalysis();

    // Update button
    updateAnalyzeButton();
  }

  function isValidUrl(string) {
    try {
      const url = new URL(string);
      return url.protocol === 'http:' || url.protocol === 'https:';
    } catch (_) {
      return false;
    }
  }

  async function analyzeVideo() {
    // Check if we have something to analyze
    if (!currentFile && (!currentUrl || currentUrl.trim() === '')) {
      alert('Пожалуйста, выберите видеофайл или введите URL видео');
      return;
    }

    // Show loading
    showLoading(true);
    addLogEntry('Начало анализа видео...');

    try {
      let result;

      if (currentFile) {
        // Analyze file
        addLogEntry(`Анализ файла: ${currentFile.name}`);
        result = await analyzeFile(currentFile);
      } else {
        // Analyze URL
        addLogEntry(`Анализ URL: ${currentUrl}`);
        result = await analyzeUrl(currentUrl);
      }

      // Handle response
      if (result && result.success) {
        addLogEntry('Анализ завершен успешно');
        handleAnalysisResponse(result, currentFile ? currentFile.name : currentUrl, !!currentFile);

        // Save to history
        saveToHistory(result, currentFile ? currentFile.name : currentUrl, !!currentFile);
      } else {
        const errorMsg = result ? result.error : 'Неизвестная ошибка';
        addLogEntry(`Ошибка анализа: ${errorMsg}`);
        alert(`Ошибка анализа: ${errorMsg}`);
      }
    } catch (error) {
      console.error('Analysis error:', error);
      addLogEntry(`Ошибка сети: ${error.message}`);
      alert(`Ошибка сети: ${error.message}`);
    } finally {
      // Hide loading
      showLoading(false);
    }
  }

  async function analyzeFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/analyze/file', {
      method: 'POST',
      body: formData
    });

    return await response.json();
  }

  async function analyzeUrl(url) {
    const response = await fetch('/api/analyze/url', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: url })
    });

    return await response.json();
  }

  function handleAnalysisResponse(data, sourceName, isFile = true) {
    // Update status
    updateStatus('analysis-status', 'Завершено', 'success');

    // Get prediction
    const prediction = data.predicted_class || 'unknown';
    let predictionText = 'НЕИЗВЕСТНО';
    let predictionClass = 'normal';

    if (prediction === 'violent' || prediction === 'АГРЕССИЯ') {
      predictionText = 'АГРЕССИЯ';
      predictionClass = 'danger';
    } else if (prediction === 'non_violent' || prediction === 'nonviolent' || prediction === 'НОРМА') {
      predictionText = 'НОРМА';
      predictionClass = 'success';
    }

    // Get confidence
    const confidencePercent = data.confidence_percent || (data.confidence * 100) || 0;

    // Update UI
    updateStatus('prediction-result', predictionText, predictionClass);
    updateStatus('confidence-level', `${confidencePercent.toFixed(1)}%`, 'info');

    // Update video info
    if (data.processing_info) {
      const procTimeEl = document.getElementById('processing-time');
      if (procTimeEl) {
        procTimeEl.textContent = `${data.processing_info.processing_time_seconds || 0} сек`;
      }

      const durationEl = document.getElementById('video-duration');
      if (durationEl && data.processing_info.video_duration) {
        const seconds = data.processing_info.video_duration;
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        durationEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
      }

      const totalFramesEl = document.getElementById('total-frames');
      if (totalFramesEl) {
        totalFramesEl.textContent = data.processing_info.total_frames || '--';
      }

      const fpsEl = document.getElementById('video-fps');
      if (fpsEl) {
        fpsEl.textContent = data.processing_info.video_fps ?
          data.processing_info.video_fps.toFixed(2) : '--';
      }
    }

    // Add to log
    const sourceType = isFile ? 'Файл' : 'URL';
    addLogEntry(`✓ ${sourceType}: ${sourceName}`);
    addLogEntry(`✓ Результат: ${predictionText} (${confidencePercent.toFixed(1)}%)`);
    addLogEntry('─'.repeat(50));
  }

  function saveToHistory(data, sourceName, isFile = true) {
    const historyEntry = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      sourceName: sourceName,
      isFile: isFile,
      data: data,
      prediction: data.predicted_class,
      confidence: data.confidence_percent || (data.confidence * 100) || 0
    };

    analysisHistory.unshift(historyEntry); // Add to beginning

    // Keep only last 10 entries
    if (analysisHistory.length > 10) {
      analysisHistory = analysisHistory.slice(0, 10);
    }

    // Save to localStorage
    saveHistory();

    // Update history display if needed
    updateHistoryDisplay();
  }

  function saveHistory() {
    try {
      localStorage.setItem('klin_analysis_history', JSON.stringify(analysisHistory));
    } catch (e) {
      console.error('Failed to save history:', e);
    }
  }

  function loadHistory() {
    try {
      const saved = localStorage.getItem('klin_analysis_history');
      if (saved) {
        analysisHistory = JSON.parse(saved);
        updateHistoryDisplay();
      }
    } catch (e) {
      console.error('Failed to load history:', e);
    }
  }

  function updateHistoryDisplay() {
    // Optional: Add a history panel if you want
    // For now, we just keep it in localStorage
  }

  function updateStatus(elementId, text, type = 'normal') {
    const element = document.getElementById(elementId);
    if (!element) return;

    element.textContent = text;
    element.className = 'status-value';

    if (type === 'success') {
      element.classList.add('success');
    } else if (type === 'danger') {
      element.classList.add('danger');
    } else if (type === 'info') {
      element.classList.add('info');
    }
  }

  function addLogEntry(message) {
    const logContainer = document.getElementById('event-log');
    if (!logContainer) return;

    const time = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `<span class="log-time">${time}</span><span class="log-message">${escapeHtml(message)}</span>`;
    logContainer.appendChild(entry);
    logContainer.scrollTop = logContainer.scrollHeight;
  }

  function showLoading(show) {
    if (!loadingOverlay) return;
    loadingOverlay.style.display = show ? 'flex' : 'none';
  }

  function updateAnalyzeButton() {
    if (!analyzeBtn) return;

    // Enable button if we have a file OR a valid URL
    const hasFile = currentFile !== null;
    const hasUrl = currentUrl && currentUrl.trim() !== '';

    analyzeBtn.disabled = !(hasFile || hasUrl);

    // Update button text
    if (hasFile) {
      analyzeBtn.textContent = 'Начать анализ файла';
    } else if (hasUrl) {
      analyzeBtn.textContent = 'Начать анализ по ссылке';
    } else {
      analyzeBtn.textContent = 'Начать анализ';
    }
  }

  function resetCurrentAnalysis() {
    const fields = [
      'analysis-status', 'prediction-result', 'confidence-level',
      'processing-time', 'video-duration', 'total-frames', 'video-fps'
    ];

    fields.forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.textContent = '--';
        el.className = 'status-value';
      }
    });
  }

  function clearLog() {
    const logContainer = document.getElementById('event-log');
    if (logContainer) {
      logContainer.innerHTML = '';
    }
  }

  // Utility functions
  function formatFileSize(bytes) {
    if (bytes === 0) return '0 Байт';
    const k = 1024;
    const sizes = ['Байт', 'КБ', 'МБ', 'ГБ'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Make functions available globally
  window.testUrl = testUrl;
  window.analyzeVideo = analyzeVideo;
  window.removeFile = removeFile;
  window.clearUrl = clearUrl;
  window.clearLog = clearLog;
});