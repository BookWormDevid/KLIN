import React, { useState, useRef, useEffect } from 'react';
import { VideoPlayer } from '../components/VideoPlayer/VideoPlayer';
import type { VideoPlayerRef } from '../components/VideoPlayer/VideoPlayer';
import { BBoxCanvas } from '../components/BBoxCanvas/BBoxCanvas';
import { useStream } from '../hooks/useStream';

// Заглушка для демонстрации bbox в стриме (случайные прямоугольники)
const MockBBoxProvider: React.FC<{ videoElement: HTMLVideoElement | null; currentTime: number }> = ({
    videoElement,
    currentTime,
}) => {
    const [mockYolo, setMockYolo] = useState<Record<number, number[][]> | null>(null);

    useEffect(() => {
        // Генерируем случайные bbox раз в 2 секунды
        const interval = setInterval(() => {
            const t = Math.floor(currentTime);
            if (t <= 0) return;
            setMockYolo({
                [t]: [
                    [100, 150, 300, 450],
                    [350, 200, 500, 400],
                ],
            });
        }, 2000);
        return () => clearInterval(interval);
    }, [currentTime]);

    return (
        <BBoxCanvas
            videoElement={videoElement}
            yoloData={mockYolo}
            currentTime={currentTime}
            label="СТРИМ: АГРЕССИЯ"
        />
    );
};

export const StreamPage: React.FC = () => {
    const [cameraUrl, setCameraUrl] = useState('');
    const [cameraId, setCameraId] = useState('');
    const { streamState, isActive, error, start, stop } = useStream();
    const playerRef = useRef<VideoPlayerRef>(null);
    const [currentTime, setCurrentTime] = useState(0);

    // Для тестирования без реальной камеры можно использовать тестовое видео
    const [demoVideoUrl, setDemoVideoUrl] = useState<string | null>(null);

    const handleStart = () => {
        if (!cameraUrl.trim()) return;
        const id = cameraId.trim() || `cam_${Date.now()}`;
        // Имитация: стрим "активен", показываем введённую ссылку как тестовое видео
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

            {streamState && (
                <div className="stream-info">
                    <div>Статус: {streamState.state}</div>
                    <div>ID стрима: {streamState.id}</div>
                    <div>
                        X3D: {streamState.last_x3d_label ?? '—'}
                        {streamState.last_x3d_confidence && ` (${streamState.last_x3d_confidence.toFixed(2)})`}
                    </div>
                    <div>
                        MAE: {streamState.last_mae_label ?? '—'}
                        {streamState.last_mae_confidence && ` (${streamState.last_mae_confidence.toFixed(2)})`}
                    </div>
                    <div>Обнаруженные объекты: {streamState.objects?.join(', ') || '—'}</div>
                </div>
            )}

            {/* Плеер для стрима (показываем тестовое видео) */}
            <div className="stream-video-container">
                {isActive && demoVideoUrl ? (
                    <>
                        <VideoPlayer
                            ref={playerRef}
                            url={demoVideoUrl}
                            onProgress={setCurrentTime}
                        />
                        <MockBBoxProvider
                            videoElement={playerRef.current?.getVideoElement() || null}
                            currentTime={currentTime}
                        />
                    </>
                ) : (
                    <div className="stream-placeholder">
                        {isActive
                            ? 'Стрим активен (видео не доступно для плеера)'
                            : 'Введите URL камеры и нажмите «Запустить стрим»'}
                    </div>
                )}
            </div>
        </div>
    );
};
