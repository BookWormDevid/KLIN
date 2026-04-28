import React, { useState, useRef, useEffect, useCallback } from 'react';
import { VideoPlayer } from '../components/VideoPlayer/VideoPlayer';
import type { VideoPlayerRef } from '../components/VideoPlayer/VideoPlayer';
import { BBoxCanvas } from '../components/BBoxCanvas/BBoxCanvas';
import { Timeline } from '../components/Timeline/Timeline';
import { LogPanel } from '../components/LogPanel/LogPanel';
import { HistoryList } from '../components/HistoryList/HistoryList';
import { HttpClient } from '../core/infrastructure/http/httpClient';
import { ApiVideoRepository } from '../core/infrastructure/repositories/apiVideoRepository';
import { ApiFeedbackRepository } from '../core/infrastructure/repositories/apiFeedbackRepository';
import { LocalStorageHistoryRepository } from '../core/infrastructure/repositories/localStorageHistoryRepository';
import { uploadVideo, getHistory, getAnalysisStatus } from '../core/application/usecases';
import type { Analysis } from '../core/domain/entities/Analysis';
import type { HistoryItem } from '../core/domain/entities/HistoryItem';

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

    const addLog = useCallback((msg: string) => {
        const time = new Date().toLocaleTimeString();
        setLogs(prev => [...prev, { time, msg }].slice(-100));
    }, []);

    const loadHistory = useCallback(async () => {
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
    }, []);

    const loadAnalysisById = async (id: string) => {
        try {
            addLog(`Загрузка анализа ${id}...`);
            const analysis = await getAnalysisStatus(videoRepo, id);
            setResult(analysis);
            setVideoUrl(null);
            addLog('Данные восстановлены (видео недоступно)');
        } catch (err: any) {
            addLog(`Ошибка: ${err.message}`);
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selected = e.target.files?.[0] || null;
        if (!selected) return;
        if (!selected.type.startsWith('video/')) {
            addLog(`❌ Файл "${selected.name}" не является видео`);
            return;
        }
        setFile(selected);
        if (videoUrl && videoUrl.startsWith('blob:')) URL.revokeObjectURL(videoUrl);
        const objectUrl = URL.createObjectURL(selected);
        setVideoUrl(objectUrl);
        setResult(null);
        addLog(`📁 Выбран файл: ${selected.name}`);
    };

    useEffect(() => {
        return () => {
            if (videoUrl && videoUrl.startsWith('blob:')) URL.revokeObjectURL(videoUrl);
        };
    }, [videoUrl]);

    const handleUpload = async () => {
        if (!file && !url.trim()) {
            addLog('⚠️ Выберите файл или введите ссылку');
            return;
        }
        setLoading(true);
        addLog('⏳ Отправка видео на анализ...');

        try {
            let sourceFile = file;
            if (!sourceFile && url.trim()) {
                addLog('📥 Скачивание видео по ссылке...');
                const response = await fetch(url.trim());
                if (!response.ok) throw new Error(`Ошибка загрузки: ${response.status}`);
                const blob = await response.blob();
                if (!blob.type.startsWith('video/')) throw new Error('Файл не является видео');
                sourceFile = new File([blob], 'downloaded_video.mp4', { type: blob.type });
                if (videoUrl && videoUrl.startsWith('blob:')) URL.revokeObjectURL(videoUrl);
                setVideoUrl(URL.createObjectURL(sourceFile));
            }
            if (!sourceFile) throw new Error('Не удалось получить видео');

            const analysis = await uploadVideo(videoRepo, sourceFile, (status) => {
                if (status.state !== 'FINISHED') addLog(`Статус: ${status.state}`);
                if (status.state === 'ERROR') addLog('❌ Ошибка обработки');
                setResult(status);
            });
            setResult(analysis);

            const historyItem: HistoryItem = {
                id: analysis.id,
                timestamp: new Date().toLocaleString(),
                sourceName: analysis.id.slice(0, 8),
                feedbackStatus: null,
            };
            historyRepo.addItem(historyItem);
            await loadHistory();

            addLog('✅ Анализ завершён');
        } catch (err: any) {
            addLog(`❌ Ошибка: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    const handleFeedback = async (id: string, isConfirmed: boolean) => {
        await feedbackRepo.sendFeedback(id, isConfirmed);
        historyRepo.updateFeedback(id, isConfirmed ? 'confirmed' : 'rejected');
        await loadHistory();
        addLog(isConfirmed ? '👍 Фидбек: верно' : '👎 Фидбек: ошибка');
    };

    const timelineEvents = result?.mae?.map(ev => ({ time: ev.time, class: ev.answer })) || [];

    useEffect(() => { loadHistory(); }, [loadHistory]);

    return (
        <div className="upload-page">
            <div className="upload-main">
                <div className="upload-player-section">
                    <div className="video-container">
                        {videoUrl ? (
                            <>
                                <VideoPlayer
                                    ref={playerRef}
                                    url={videoUrl}
                                    onProgress={setCurrentTime}
                                    onDuration={setDuration}
                                />
                                <BBoxCanvas
                                    videoElement={playerRef.current?.getVideoElement() || null}
                                    yoloData={result?.yolo ?? null}
                                    currentTime={currentTime}
                                    label="АГРЕССИЯ"
                                />
                                {duration > 0 && timelineEvents.length > 0 && (
                                    <Timeline
                                        events={timelineEvents}
                                        duration={duration}
                                        currentTime={currentTime}
                                        onSeek={t => playerRef.current?.seekTo(t)}
                                    />
                                )}
                            </>
                        ) : (
                            <div className="video-placeholder">Выберите видео для предпросмотра</div>
                        )}
                    </div>

                    <div className="upload-controls">
                        <input type="file" accept="video/*" onChange={handleFileChange} />
                        <input type="text" placeholder="Ссылка на видео" value={url} onChange={e => setUrl(e.target.value)} />
                        <button onClick={handleUpload} disabled={loading}>
                            {loading ? 'Обработка...' : 'Запустить анализ'}
                        </button>
                    </div>
                </div>

                <div className="upload-panels">
                    <LogPanel logs={logs} onClear={() => setLogs([])} />
                    <HistoryList items={history} onLoad={loadAnalysisById} onFeedback={handleFeedback} />
                </div>
            </div>
        </div>
    );
};
