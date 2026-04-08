export interface HistoryItem {
    id: string;
    timestamp: string;
    sourceName: string;
    feedbackStatus: 'confirmed' | 'rejected' | null;
}