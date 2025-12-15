// static/js/script.js
// Patched by assistant — fixes selectors, prevents silent exceptions, improves UX/logging

(() => {
  'use strict';

  // Global state
  let currentFile = null;
  let currentUrl = null;
  let activeMethod = null; // 'file' or 'url'
  let objectUrl = null;

  // Utility: safe query
  function $id(id) { return document.getElementById(id); }

  // Initialize
  document.addEventListener('DOMContentLoaded', () => {
    try {
      initializeEventListeners();
      addLogEntry('Система инициализирована');
    } catch (err) {
      console.error('Init error:', err);
      addLogEntry('❌ Ошибка инициализации: ' + (err && err.message ? err.message : err));
    }
  });

  function initializeEventListeners() {
    // Elements
    const fileInput = $id('file-input');
    const dropZone = $id('drop-zone');
    const urlInput = $id('video-url');
    const analyzeBtn = $id('analyze-btn');

    // Guard
    if (!fileInput || !dropZone || !urlInput || !analyzeBtn) {
      console.error('Missing DOM elements on init');
      addLogEntry('❌ Ошибка: отсутствуют элементы интерфейса');
      return;
    }

    // Fix: reliable upload-method selectors
    const uploadMethods = document.querySelectorAll('.upload-method');
    const fileMethod = uploadMethods[0] || null;
    const urlMethod = uploadMethods[1] || null;

    // File input change
    fileInput.addEventListener('change', (e) => {
      if (e && e.target && e.target.files && e.target.files[0]) {
        handleFileSelect(e.target.files[0], fileMethod);
      }
    });

    // NOTE: We intentionally DO NOT attach a dropZone click -> fileInput.click()
    // to avoid double-open behaviour when the browse button is inside the drop zone.
    // Drag + drop
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      const files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length > 0) {
        handleFileSelect(files[0], fileMethod);
      }
    });

    // URL input
    urlInput.addEventListener('input', (e) => handleUrlInput(e, urlMethod));
    urlInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') testUrl(urlMethod);
    });

    // Clean up object URL when page unloads
    window.addEventListener('beforeunload', () => {
      if (objectUrl) {
        try { URL.revokeObjectURL(objectUrl); } catch (e) { /* ignore */ }
      }
    });

    // Make sure analyze button state is correct initially
    updateAnalyzeButton();
  }

  function handleFileSelect(file, fileMethod = null) {
    try {
      if (!file || !file.type || !file.type.startsWith('video/')) {
        showError('Пожалуйста, выберите корректный видеофайл');
        return;
      }

      // limit file size client-side to avoid huge uploads (adjust as needed)
      const MAX_MB = 500;
      if (file.size > MAX_MB * 1024 * 1024) {
        showError(`Файл слишком большой — максимальный размер ${MAX_MB} MB`);
        return;
      }

      // update UI
      const fileInfo = $id('file-info');
      if (fileInfo) {
        fileInfo.innerHTML = `
          <div class="file-selected">
            <strong>Выбран файл:</strong> ${escapeHtml(file.name)}<br>
            <small>Размер: ${formatFileSize(file.size)} • Тип: ${escapeHtml(file.type)}</small>
          </div>
        `;
      }

      // set active method & state
      setActiveMethod('file', fileMethod);
      currentFile = file;
      currentUrl = null;
      updateAnalyzeButton();

      // preview: revoke previous object URL then create new
      if (objectUrl) {
        try { URL.revokeObjectURL(objectUrl); } catch (e) { /* ignore */ }
        objectUrl = null;
      }
      objectUrl = URL.createObjectURL(file);
      createVideoPreview(objectUrl);

      addLogEntry(`Файл загружен: ${file.name}`);
    } catch (err) {
      console.error('handleFileSelect error', err);
      showError('Ошибка при выборе файла: ' + (err && err.message ? err.message : err));
      addLogEntry('❌ Ошибка при выборе файла: ' + (err && err.message ? err.message : err));
    }
  }

  function handleUrlInput(e, urlMethod = null) {
    const url = (e && e.target && e.target.value) ? e.target.value.trim() : '';
    if (url && isValidVideoUrl(url)) {
      setActiveMethod('url', urlMethod);
      currentUrl = url;
    } else {
      currentUrl = null;
    }
    updateAnalyzeButton();
  }

  // Test URL accessibility (uses HEAD; some servers block HEAD — consider GET fallback)
  function testUrl(urlMethod = null) {
    const urlInput = $id('video-url');
    const url = urlInput ? urlInput.value.trim() : '';

    if (!url) {
      showError('Пожалуйста, введите ссылку на видео');
      return;
    }
    if (!isValidVideoUrl(url)) {
      showError('Пожалуйста, введите корректную ссылку на видео (mp4, avi, mov, mkv, webm)');
      return;
    }

    setActiveMethod('url', urlMethod);
    currentUrl = url;
    updateAnalyzeButton();

    showLoading(true, 'Проверка доступности ссылки...');
    // First try HEAD, fallback to GET for servers that disallow HEAD
    fetch(url, { method: 'HEAD' })
      .then(resp => {
        showLoading(false);
        if (resp.ok) {
          createVideoPreview(url);
          addLogEntry(`Ссылка проверена успешно: ${url}`);
          showSuccess('Ссылка доступна и готова к анализу!');
        } else {
          // fallback to small-range GET attempt
          return fetch(url, { method: 'GET', headers: { Range: 'bytes=0-1023' } })
            .then(r => {
              if (r.ok || r.status === 206) {
                createVideoPreview(url);
                addLogEntry(`Ссылка проверена успешно (GET fallback): ${url}`);
                showSuccess('Ссылка доступна и готова к анализу!');
              } else {
                showError('Ссылка недоступна (сервер вернул ' + r.status + ')');
              }
            });
        }
      })
      .catch(err => {
        showLoading(false);
        // try GET fallback
        fetch(url, { method: 'GET', headers: { Range: 'bytes=0-1023' } })
          .then(r => {
            showLoading(false);
            if (r.ok || r.status === 206) {
              createVideoPreview(url);
              addLogEntry(`Ссылка проверена успешно (GET fallback): ${url}`);
              showSuccess('Ссылка доступна и готова к анализу!');
            } else {
              showError('Не удалось получить доступ к ссылке. Код: ' + r.status);
            }
          })
          .catch(() => {
            showError('Не удалось получить доступ к ссылке. Проверьте подключение или CORS.');
          });
      });
  }

  function setActiveMethod(method, methodElement = null) {
    activeMethod = method;

    // Visual feedback for method selection
    const uploadMethods = document.querySelectorAll('.upload-method');
    if (uploadMethods && uploadMethods.length >= 1) {
      uploadMethods.forEach((el) => el.classList.remove('active'));
      if (methodElement) {
        methodElement.classList.add('active');
      } else {
        // fallback: toggle first(0) or second(1)
        if (method === 'file' && uploadMethods[0]) uploadMethods[0].classList.add('active');
        if (method === 'url' && uploadMethods[1]) uploadMethods[1].classList.add('active');
      }
    }
  }

  function createVideoPreview(src) {
    const videoPlayer = $id('video-player');
    if (!videoPlayer) return;
    // prefer setting srcObject for MediaStream — here we use src
    try {
      videoPlayer.src = src;
      // Attempt to show poster/controls immediately
      videoPlayer.load();
    } catch (e) {
      console.warn('Could not set video src:', e);
    }
    console.log("Previewing video:", src);

    // show results section robustly
    const results = $id('results-section');
    if (results) {
      results.classList.remove && results.classList.remove('hidden');
      results.style.display = 'block';
    }

    // reset previous results (visual)
    resetResults();
  }

  function analyzeVideo() {
    try {
      if (!currentFile && !currentUrl) {
        showError('Пожалуйста, выберите видеофайл или введите ссылку на видео');
        return;
      }

      showLoading(true, 'Анализ видео...');
      const analyzeBtn = $id('analyze-btn');
      if (analyzeBtn) analyzeBtn.disabled = true;
      addLogEntry('Запуск анализа видео...');

      if (activeMethod === 'file' && currentFile) {
        analyzeFile(currentFile);
      } else if (activeMethod === 'url' && currentUrl) {
        analyzeUrl(currentUrl);
      } else {
        showError('Не удалось определить метод анализа (file/url).');
        showLoading(false);
        if (analyzeBtn) analyzeBtn.disabled = false;
      }
    } catch (err) {
      console.error('analyzeVideo error', err);
      showLoading(false);
      showError('Ошибка при запуске анализа: ' + (err && err.message ? err.message : err));
    }
  }

  function analyzeFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/analyze/file', {
      method: 'POST',
      body: formData
    })
      .then(parseJsonSafe)
      .then(data => {
        showLoading(false);
        handleAnalysisResponse(data);
      })
      .catch(err => {
        showLoading(false);
        showError('Ошибка анализа: ' + (err && err.message ? err.message : err));
        addLogEntry('❌ Ошибка анализа (file): ' + (err && err.message ? err.message : err));
      })
      .finally(() => {
        const analyzeBtn = $id('analyze-btn');
        if (analyzeBtn) analyzeBtn.disabled = false;
      });
  }

  function analyzeUrl(url) {
    fetch('/api/analyze/url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    })
      .then(parseJsonSafe)
      .then(data => {
        showLoading(false);
        handleAnalysisResponse(data);
      })
      .catch(err => {
        showLoading(false);
        showError('Ошибка анализа: ' + (err && err.message ? err.message : err));
        addLogEntry('❌ Ошибка анализа (url): ' + (err && err.message ? err.message : err));
      })
      .finally(() => {
        const analyzeBtn = $id('analyze-btn');
        if (analyzeBtn) analyzeBtn.disabled = false;
      });
  }

  function parseJsonSafe(resp) {
    if (!resp) return Promise.reject(new Error('Нет ответа от сервера'));
    return resp.json().catch(() => Promise.reject(new Error('Неверный JSON в ответе сервера')));
  }

  function handleAnalysisResponse(data) {
    if (!data) {
      showError('Пустой ответ от сервера');
      return;
    }

    if (data.success) {
      updateStatus('analysis-status', 'Завершено', 'success');

      const predictionText = (data.result === 'violent') ? 'АГРЕССИЯ' : 'НОРМА';
      const predictionType = (data.result === 'violent') ? 'danger' : 'success';

      updateStatus('prediction-result', predictionText, predictionType);
      updateStatus('confidence-level', `${data.confidence}%`, 'info');

      const procEl = $id('processing-time');
      if (procEl) procEl.textContent = data.processing_time ? `${data.processing_time}с` : '--';

      if (data.details) {
        const durEl = $id('video-duration');
        const framesEl = $id('frames-analyzed');
        if (durEl) durEl.textContent = (data.details.video_duration !== undefined && data.details.video_duration !== null) ? `${data.details.video_duration}с` : '--';
        if (framesEl) framesEl.textContent = data.details.frames_analyzed || '--';
      }

      const resultText = data.result === 'violent' ? 'агрессия' : 'норма';
      addLogEntry(`Анализ завершен: ${resultText} (уверенность ${data.confidence}%)`);
      showSuccess('Анализ успешно завершен!');
    } else {
      showError(data.error || 'Ошибка анализа');
      addLogEntry('❌ Ошибка анализа (response): ' + (data.error || JSON.stringify(data)));
    }
  }

  // UI helpers
  function updateAnalyzeButton() {
    const analyzeBtn = $id('analyze-btn');
    if (!analyzeBtn) return;
    analyzeBtn.disabled = !(currentFile || currentUrl);
  }

  function updateStatus(elementId, text, type = 'normal') {
    const element = $id(elementId);
    if (!element) return;
    element.textContent = text;
    element.className = `status-value ${type}`;
  }

  function addLogEntry(message) {
    const logContainer = $id('event-log');
    if (!logContainer) return;
    const time = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `<span class="log-time">${time}</span><span class="log-message">${escapeHtml(message)}</span>`;
    logContainer.appendChild(entry);
    logContainer.scrollTop = logContainer.scrollHeight;
  }

  function showLoading(show, message = 'Обработка видео...') {
    const overlay = $id('loading-overlay');
    const messageElement = $id('loading-message');
    if (!overlay) return;
    if (show) {
      if (messageElement) messageElement.textContent = message;
      overlay.style.display = 'flex';
    } else {
      overlay.style.display = 'none';
    }
  }

  function showError(message) {
    alert('Ошибка: ' + message);
    addLogEntry('❌ ' + message);
  }

  function showSuccess(message) {
    addLogEntry('✅ ' + message);
  }

  function resetResults() {
    updateStatus('analysis-status', '--');
    updateStatus('prediction-result', '--');
    updateStatus('confidence-level', '--');
    const proc = $id('processing-time'); if (proc) proc.textContent = '--';
    const dur = $id('video-duration'); if (dur) dur.textContent = '--';
    const frames = $id('frames-analyzed'); if (frames) frames.textContent = '--';
  }

  // small utilities
  function formatFileSize(bytes) {
    if (bytes === 0) return '0 Байт';
    const k = 1024;
    const sizes = ['Байт', 'КБ', 'МБ', 'ГБ'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function isValidVideoUrl(url) {
    try {
      new URL(url);
      const videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.wmv'];
      return videoExtensions.some(ext => url.toLowerCase().includes(ext));
    } catch {
      return false;
    }
  }

  function escapeHtml(unsafe) {
    if (!unsafe) return '';
    return unsafe.replace(/[&<"'>]/g, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[m]));
  }

  // safe JSON stringify for UI debug (not used now but handy)
  function safeStringify(o) {
    try { return JSON.stringify(o, null, 2); } catch { return String(o); }
  }

  // Expose functions used by inline HTML onclick=""
  window.analyzeVideo = analyzeVideo;
  window.testUrl = testUrl;

  // end of IIFE
})();
