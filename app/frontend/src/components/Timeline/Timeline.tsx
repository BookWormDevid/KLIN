import React from 'react';
import './Timeline.css';

interface TimelineEvent {
    time: [number, number];
    class: string;
}

interface TimelineProps {
    events: TimelineEvent[];
    duration: number;
    currentTime: number;
    onSeek: (time: number) => void;
}

export const Timeline: React.FC<TimelineProps> = ({ events, duration, currentTime, onSeek }) => {
    const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const percent = (e.clientX - rect.left) / rect.width;
        onSeek(percent * duration);
    };

    return (
        <div className="timeline-container">
            <div className="timeline-track" onClick={handleClick}>
                {events.map((ev, idx) => (
                    <div
                        key={idx}
                        className="timeline-event"
                        style={{
                            left: `${(ev.time[0] / duration) * 100}%`,
                            width: `${((ev.time[1] - ev.time[0]) / duration) * 100}%`,
                        }}
                        title={`${ev.class} (${ev.time[0].toFixed(1)}-${ev.time[1].toFixed(1)}s)`}
                    />
                ))}
                <div
                    className="timeline-cursor"
                    style={{ left: `${(currentTime / duration) * 100}%` }}
                />
            </div>
        </div>
    );
};
