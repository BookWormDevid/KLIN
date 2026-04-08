import React from 'react';
import { HistoryItem } from '../../core/domain/entities/HistoryItem';
import './HistoryList.css';

interface HistoryListProps {
    items: HistoryItem[];
    onLoad: (id: string) => void;
    onFeedback: (id: string, isConfirmed: boolean) => void;
}

export const HistoryList: React.FC<HistoryListProps> = ({ items, onLoad, onFeedback }) => {
    if (items.length === 0) {
        return <div className="history-empty">История пуста</div>;
    }

    return (
        <div className="history-list">
            {items.map(item => (
                <div key={item.id} className="history-item" onClick={() => onLoad(item.id)}>
                    <div className="history-info">
                        <span className="history-name">{item.sourceName}</span>
                        <span className="history-date">{item.timestamp}</span>
                    </div>
                    {!item.feedbackStatus ? (
                        <div className="history-feedback">
                            <button onClick={(e) => { e.stopPropagation(); onFeedback(item.id, true); }}>Верно</button>
                            <button onClick={(e) => { e.stopPropagation(); onFeedback(item.id, false); }}>Ошибка</button>
                        </div>
                    ) : (
                        <div className="history-status">
                            {item.feedbackStatus === 'confirmed' ? '✓ Отправлено в БД' : '✗ Отклонено'}
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
};