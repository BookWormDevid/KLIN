import React, { useState, useRef, useEffect } from 'react';
import { VideoPlayer, VideoPlayerRef } from '../components/VideoPlayer/VideoPlayer';
import { BBoxCanvas } from '../components/BBoxCanvas/BBoxCanvas';
import { Timeline } from '../components/Timeline/Timeline';
import { LogPanel } from '../components/LogPanel/LogPanel';
import { HistoryList } from '../components/HistoryList/HistoryList';
import { HttpClient } from '../core/infrastructure/http/httpClient';
import { ApiVideoRepository } from '../core/infrastructure/repositories/apiVideoRepository';
import { ApiFeedbackRepository } from '../core/infrastructure/repositories/apiFeedbackRepository';
import { LocalStorageHistoryRepository } from '../core/infrastructure/repositories/localStorageHistoryRepository';
import { uploadVideo, getHistory, getAnalysisStatus } from '../core/application/usecases';
import { Analysis, MaeEvent } from '../core/domain/entities/Analysis';
import { HistoryItem } from '../core/domain/entities/HistoryItem';

const http = new HttpClient();
const videoRepo = new ApiVideoRepository(http);
const feedbackRepo = new ApiFeedbackRepository(http);
const historyRepo = new LocalStorageHistoryRepository();

export const UploadPage: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [url, setUrl] = useState('');
    const [videoUrl, setVideoUrl] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<Analysis | null>(null);
    const [logs, setLogs] = useState<{ time: string; msg: string }[]>([]);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [history, setHistory] = useState<HistoryItem[]>([]);
    const playerRef = useRef<VideoPlayerRef>(null);

    const addLog = (msg: string) => {
        const time = new Date().toLocaleTimeString();
        setLogs(prev => [...prev, { time, msg }].slice(-50));
    };

    const loadHistory = async () => {
        try {
            const analyses = await getHistory(videoRepo, 20);
            const items: HistoryItem[] = analyses.map(a => ({
                id: a.id,
                timestamp: new Date().toLocaleString(),
                sourceName: a.id.slice(0, 8),
                feedbackStatus: null,
            }));
            setHistory(items);
            historyRepo.saveHistory(items);
        } catch (err) {
            console.error('Failed to load history', err);
        }
    };

    const loadAnalysisById = async (id: string) => {
        try {
            addLog(`Загрузка анализа ${id}...`);
            const analysis = await getAnalysisStatus(videoRepo, id);
            setResult(analysis);
            // Восстановить видео невозможно, так как нет API для скачивания
            setVideoUrl(null);
            addLog('Данные восстановлены (видео недоступно)');
        } catch (err: any) {
            addLog(`Ошибка: ${err.message}`);
        }
    };

    const handleUpload = async () => {
        if (!file && !url) return;
        setLoading(true);
        setResult(null);
        addLog('Отправка на сервер...');

        try {
            let sourceFile = file;
            if (!sourceFile && url) {
                addLog('Загрузка видео по URL...');
                const response = await fetch(url);
                const blob = await response.blob();
                sourceFile = new File([blob], 'video.mp4', { type: blob.type });
            }
            if (!sourceFile) throw new Error('Не удалось получить видео');

            const localUrl = URL.createObjectURL(sourceFile);
            setVideoUrl(localUrl);

            const analysis = await uploadVideo(videoRepo, sourceFile, (status) => {
                addLog(`Статус: ${status.state}`);
                if (status.state === 'FINISHED') setResult(status);
                if (status.state === 'ERROR') addLog('Ошибка обработки');
            });

            await loadHistory();
            setResult(analysis);
            addLog('Анализ завершён');
        } catch (err: any) {
            addLog(`Ошибка: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    const handleFeedback = async (id: string, isConfirmed: boolean) => {
        await feedbackRepo.sendFeedback(id, isConfirmed);
        await loadHistory();
        addLog(isConfirmed ? 'Фидбек отправлен' : 'Инцидент отклонён');
    };

    const currentBBoxes = (): Array<[number, number, number, number]> | null => {
        if (!result?.yolo) return null;
        // yolo – это объект { timestamp: [[x1,y1,x2,y2], ...] }
        const times = Object.keys(result.yolo).map(Number).sort((a, b) => a - b);
        if (times.length === 0) return null;
        let bestKey = times[0];
        for (const t of times) {
            if (t <= currentTime) bestKey = t;
            else break;
        }
        const boxes = result.yolo[bestKey];
        if (!boxes || !Array.isArray(boxes)) return null;
        return boxes as Array<[number, number, number, number]>;
    };

    const timelineEvents = result?.mae?.map(ev => ({
        time: ev.time,
        class: ev.answer,
    })) || [];

    useEffect(() => {
        loadHistory();
        return () => {
            if (videoUrl && videoUrl.startsWith('blob:')) URL.revokeObjectURL(videoUrl);
        };
    }, []);

    return (
        <div className="upload-page">
            <div className="upload-controls">
                <input type="file" accept="video/*" onChange={e => setFile(e.target.files?.[0] || null)} />
                <input type="text" placeholder="Ссылка на видео (будет скачано)" value={url} onChange={e => setUrl(e.target.value)} />
                <button onClick={handleUpload} disabled={loading}>{loading ? 'Обработка...' : 'Запустить анализ'}</button>
            </div>

            <div className="video-container">
                {videoUrl ? (
                    <>
                        <VideoPlayer ref={playerRef} url={videoUrl} onProgress={setCurrentTime} onDuration={setDuration} />
                        <BBoxCanvas videoElement={playerRef.current?.getVideoElement() || null} detections={currentBBoxes()} label="АГРЕССИЯ" />
                    </>
                ) : (
                    <div className="video-placeholder">Видео не загружено</div>
                )}
                {timelineEvents.length > 0 && duration > 0 && (
                    <Timeline
                        events={timelineEvents}
                        duration={duration}
                        currentTime={currentTime}
                        onSeek={(t) => playerRef.current?.seekTo(t)}
                    />
                )}
            </div>

            <div className="side-panels">
                <LogPanel logs={logs} onClear={() => setLogs([])} />
                <HistoryList items={history} onLoad={loadAnalysisById} onFeedback={handleFeedback} />
            </div>
        </div>
    );
};