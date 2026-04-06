import { HistoryItem } from '../../domain/entities/HistoryItem';
import { appConfig } from '../../../config/appConfig';

export class LocalStorageHistoryRepository {
    private storageKey = appConfig.historyStorageKey;

    getHistory(): HistoryItem[] {
        const raw = localStorage.getItem(this.storageKey);
        if (!raw) return [];
        try {
            return JSON.parse(raw);
        } catch {
            return [];
        }
    }

    saveHistory(history: HistoryItem[]): void {
        localStorage.setItem(this.storageKey, JSON.stringify(history));
    }

    addItem(item: HistoryItem): void {
        const current = this.getHistory();
        const updated = [item, ...current].slice(0, 20);
        this.saveHistory(updated);
    }

    updateFeedback(id: string, status: 'confirmed' | 'rejected'): void {
        const current = this.getHistory();
        const updated = current.map(item =>
            item.id === id ? { ...item, feedbackStatus: status } : item
        );
        this.saveHistory(updated);
    }
}