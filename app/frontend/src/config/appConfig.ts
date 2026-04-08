export const appConfig = {
    apiBaseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8008/api/v1',
    requestTimeoutMs: 15000,
    maxVideoSizeBytes: 200 * 1024 * 1024,
    historyStorageKey: 'klin_analysis_history',
    feedbackStorageKey: 'klin_feedback',
    pollIntervalMs: 2000,
    streamPollIntervalMs: 2000,
};