import React from 'react';
import './LogPanel.css';

interface LogEntry {
    time: string;
    msg: string;
}

interface LogPanelProps {
    logs: LogEntry[];
    onClear: () => void;
}

export const LogPanel: React.FC<LogPanelProps> = ({ logs, onClear }) => {
    return (
        <div className="log-panel">
            <div className="log-header">
                <span>Журнал</span>
                <button onClick={onClear}>ОЧИСТИТЬ</button>
            </div>
            <div className="log-list">
                {logs.slice().reverse().map((log, i) => (
                    <div key={i} className="log-entry">
                        <span className="log-time">[{log.time}]</span>
                        <span className="log-msg">{log.msg}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};