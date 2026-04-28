export const appConfig = {
    // Используем относительный путь, чтобы работало и локально, и на сервере
    apiBaseUrl: import.meta.env.VITE_API_BASE_URL || '/api/v1',
    requestTimeoutMs: 30000,
    maxVideoSizeBytes: 200 * 1024 * 1024,
    historyStorageKey: 'klin_analysis_history',
    pollIntervalMs: 2000,
    streamPollIntervalMs: 2000,
};
