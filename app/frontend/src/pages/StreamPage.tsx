import React, { useState, useRef, useEffect } from 'react';
import { VideoPlayer } from '../components/VideoPlayer/VideoPlayer';
import type { VideoPlayerRef } from '../components/VideoPlayer/VideoPlayer';
import { BBoxCanvas } from '../components/BBoxCanvas/BBoxCanvas';
import { useStream } from '../hooks/useStream';
import { useServices } from '../core/context/ServicesContext';

export const StreamPage: React.FC = () => {
    const { streamRepo } = useServices();
    const [cameraUrl, setCameraUrl] = useState('');
    const [cameraId, setCameraId] = useState('');
    const { streamState, isActive, error, start, stop } = useStream(streamRepo);
    const playerRef = useRef<VideoPlayerRef>(null);
    const [currentTime, setCurrentTime] = useState(0);
    const [demoVideoUrl, setDemoVideoUrl] = useState<string | null>(null);

    // ----- Состояние мигающей надписи -----
    const [alertVisible, setAlertVisible] = useState(false);
    const alertTimerRef = useRef<number | null>(null);

    // ----- История событий стрима -----
    const [streamEvents, setStreamEvents] = useState<
        { time: string; cameraId: string; label: string; confidence: string }[]
    >([]);

    // Определяем, есть ли сейчас агрессия
    const hasAggression =
        streamState?.last_x3d_label === 'True' ||
        (streamState?.last_mae_label && streamState.last_mae_label !== 'Normal');

    // Управление видимостью надписи с задержкой
    useEffect(() => {
        if (hasAggression) {
            setAlertVisible(true);
            if (alertTimerRef.current) {
                clearTimeout(alertTimerRef.current);
                alertTimerRef.current = null;
            }
        } else {
            if (alertVisible) {
                alertTimerRef.current = window.setTimeout(() => {
                    setAlertVisible(false);
                }, 3000);
            }
        }
        return () => {
            if (alertTimerRef.current) clearTimeout(alertTimerRef.current);
        };
    }, [hasAggression, alertVisible]);

    // Добавление событий в ленту (только MAE, без дублирования подряд)
    useEffect(() => {
        if (!streamState) return;
        const now = new Date().toLocaleTimeString();
        const camId = streamState.camera_id || 'unknown';

        if (streamState.last_mae_label && streamState.last_mae_label !== 'Normal') {
            const confidence = streamState.last_mae_confidence != null
                ? streamState.last_mae_confidence.toFixed(2)
                : '?';
            const label = streamState.last_mae_label;

            setStreamEvents(prev => {
                const last = prev[prev.length - 1];
                if (last && last.cameraId === camId && last.label === label) {
                    return prev;
                }
                return [...prev, { time: now, cameraId: camId, label, confidence }];
            });
        }
    }, [streamState]);

    const handleStart = () => {
        if (!cameraUrl.trim()) return;
        const id = cameraId.trim() || `cam_${Date.now()}`;
        setDemoVideoUrl(cameraUrl);
        start(cameraUrl, id);
    };

    const handleStop = () => {
        stop();
        setDemoVideoUrl(null);
    };

    return (
        <div className="stream-page">
            <div className="stream-controls">
                <input
                    type="text"
                    placeholder="URL камеры (RTSP / HTTP / тестовое видео)"
                    value={cameraUrl}
                    onChange={e => setCameraUrl(e.target.value)}
                    className="stream-url-input"
                />
                <input
                    type="text"
                    placeholder="ID камеры (опционально)"
                    value={cameraId}
                    onChange={e => setCameraId(e.target.value)}
                    className="stream-id-input"
                />
                <button onClick={handleStart} disabled={isActive}>
                    Запустить стрим
                </button>
                <button onClick={handleStop} disabled={!isActive}>
                    Остановить
                </button>
            </div>

            {error && <div className="stream-error">Ошибка: {error}</div>}

            <div className="stream-main" style={{ display: 'flex', gap: '1rem' }}>
                <div style={{ flex: 1 }}>
                    {streamState && (
                        <div className="stream-info">
                            <div>Камера: {streamState.camera_id || '—'}</div>
                            <div>
                                Шанс агрессии:{' '}
                                {streamState.last_x3d_label === 'True'
                                    ? streamState.last_x3d_confidence != null
                                        ? (streamState.last_x3d_confidence * 100).toFixed(1) + '%'
                                        : '—'
                                    : streamState.last_x3d_label === 'False'
                                        ? streamState.last_x3d_confidence != null
                                            ? (streamState.last_x3d_confidence * 100).toFixed(1) + '%'
                                            : '—'
                                        : '—'}
                            </div>
                            <div>
                                Вид аномалии:{' '}
                                {streamState.last_mae_label && streamState.last_mae_label !== 'Normal'
                                    ? `${streamState.last_mae_label} (${streamState.last_mae_confidence != null
                                        ? (streamState.last_mae_confidence * 100).toFixed(1) + '%'
                                        : '—'
                                    })`
                                    : 'Не обнаружена'}
                            </div>
                        </div>
                    )}

                    <div className="stream-video-container" style={{ aspectRatio: '16/9', width: '100%', position: 'relative' }}>
                        {isActive && demoVideoUrl ? (
                            <>
                                <VideoPlayer
                                    ref={playerRef}
                                    url={demoVideoUrl}
                                    onProgress={setCurrentTime}
                                />
                                {streamState && (
                                    <BBoxCanvas
                                        videoElement={playerRef.current?.getVideoElement() || null}
                                        yoloData={{}}
                                        currentTime={currentTime}
                                    />
                                )}
                            </>
                        ) : (
                            <div className="stream-placeholder">
                                {isActive
                                    ? 'Стрим активен (видео не доступно для плеера)'
                                    : 'Введите URL камеры и нажмите «Запустить стрим»'}
                            </div>
                        )}
                    </div>

                    {alertVisible && (
                        <div className="aggression-alert" style={{ marginTop: '0.5rem' }}>
                            ⚠️ ОБНАРУЖЕНА АГРЕССИЯ ⚠️
                        </div>
                    )}
                </div>

                <div className="stream-events" style={{ width: '320px', maxHeight: '500px', overflowY: 'auto' }}>
                    <h3>Журнал событий</h3>
                    {streamEvents.length === 0 && <p>Нет событий</p>}
                    {streamEvents.slice().reverse().map((ev, i) => (
                        <div key={i} className="stream-event-entry">
                            <span className="event-time">[{ev.time}]</span>{' '}
                            <strong>{ev.cameraId}</strong> – {ev.label} ({ev.confidence})
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};
