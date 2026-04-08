import React, { useState, useRef, useEffect } from 'react';
import { useStream } from '../hooks/useStream';
import { VideoPlayer, VideoPlayerRef } from '../components/VideoPlayer/VideoPlayer';
import { BBoxCanvas } from '../components/BBoxCanvas/BBoxCanvas';

export const StreamPage: React.FC = () => {
    const [cameraUrl, setCameraUrl] = useState('');
    const [cameraId, setCameraId] = useState('');
    const { streamState, isActive, error, start, stop } = useStream();
    const playerRef = useRef<VideoPlayerRef>(null);
    // Для стриминга нет реальных bbox, поэтому не рисуем канвас

    const handleStart = () => {
        if (!cameraUrl.trim()) return;
        const id = cameraId.trim() || `cam_${Date.now()}`;
        start(cameraUrl, id);
    };

    const handleStop = () => {
        stop();
    };

    return (
        <div className="stream-page">
            <div className="stream-controls">
                <input
                    type="text"
                    placeholder="URL камеры (RTSP, HTTP, файл)"
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
                <button onClick={handleStart} disabled={isActive}>Запустить стрим</button>
                <button onClick={handleStop} disabled={!isActive}>Остановить</button>
            </div>

            {error && <div className="stream-error">Ошибка: {error}</div>}

            {streamState && (
                <div className="stream-info">
                    <div>Статус: {streamState.state}</div>
                    <div>ID стрима: {streamState.id}</div>
                    <div>Последнее X3D: {streamState.last_x3d_label} ({streamState.last_x3d_confidence?.toFixed(2)})</div>
                    <div>Последнее MAE: {streamState.last_mae_label} ({streamState.last_mae_confidence?.toFixed(2)})</div>
                    <div>Обнаруженные объекты: {streamState.objects?.join(', ') || '-'}</div>
                </div>
            )}

            {isActive && cameraUrl && (
                <div className="stream-video-wrapper">
                    <VideoPlayer ref={playerRef} url={cameraUrl} />
                    {/* BBox не рисуем из-за отсутствия данных */}
                </div>
            )}
        </div>
    );
};