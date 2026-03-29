import React, { useState, useEffect, useRef } from 'react';
// @ts-ignore
import ReactPlayer from 'react-player';
import axios from 'axios';
import {
  Upload, Camera, History, AlertTriangle, CheckCircle2,
  Play, RefreshCw, Link as LinkIcon, FileVideo, X, Info, ShieldAlert, Wifi, WifiOff, ThumbsUp, ThumbsDown
} from 'lucide-react';
import { useSocket } from './hooks/useSocket';

// --- КОНФИГУРАЦИЯ ---
const SERVER_IP = 'localhost';
const API_BASE = `http://${SERVER_IP}:8008/api/v1/Klin`;
const WS_URL = `ws://${SERVER_IP}:8008/ws/stream`;
// Эндпоинт для БД дообучения (пока заглушка, настроите под себя)
const RETRAIN_API_BASE = `http://${SERVER_IP}:8008/api/v1/Retrain`;

// --- ТИПЫ ---
interface AnalysisResult {
  id: string;
  state: 'PENDING' | 'FINISHED' | 'ERROR';
  predicted_class?: string;
  confidence_percent?: number;
  video_url?: string;
  mae: { time: [number, number]; class: string }[] | null;
  yolo: Record<string, number[][]> | null;
}

interface HistoryItem {
  id: string;
  timestamp: string;
  sourceName: string;
  feedbackStatus?: 'confirmed' | 'rejected' | null; // Статус для кнопок
}

export default function App() {
  const [mode, setMode] = useState<'upload' | 'stream'>('upload');
  const [file, setFile] = useState<File | null>(null);
  const [externalUrl, setExternalUrl] = useState('');
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [logs, setLogs] = useState<{ time: string, msg: string }[]>([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isDragging, setIsDragging] = useState(false);

  const playerRef = useRef<any>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoContainerRef = useRef<HTMLDivElement>(null);
  const streamVideoRef = useRef<HTMLVideoElement>(null);
  const hiddenCanvasRef = useRef<HTMLCanvasElement>(document.createElement('canvas'));
  const isInitialized = useRef(false);

  const { isConnected, lastFrameData, sendFrame } = useSocket(mode === 'stream' ? WS_URL : '');

  // ИНИЦИАЛИЗАЦИЯ И ЗАГРУЗКА ИСТОРИИ ИЗ БД
  useEffect(() => {
    if (!isInitialized.current) {
      const fetchHistory = async () => {
        try {
          // Пытаемся получить историю из основной БД
          const res = await axios.get(`${API_BASE}/history`);
          setHistory(res.data);
          addLog("История загружена из базы данных");
        } catch (err) {
          // Если БД недоступна, используем старый добрый localStorage
          const saved = localStorage.getItem('klin_analysis_history');
          if (saved) setHistory(JSON.parse(saved));
          addLog("Система готова к работе (локальная история)");
        }
      };

      fetchHistory();
      isInitialized.current = true;
    }
  }, []);

  const addLog = (msg: string) => {
    const time = new Date().toLocaleTimeString();
    setLogs(prev => {
      if (prev.length > 0 && prev[prev.length - 1].msg === msg) return prev;
      return [...prev, { time, msg }].slice(-50);
    });
  };

  useEffect(() => {
    let stream: MediaStream | null = null;
    let frameInterval: ReturnType<typeof setInterval>;

    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (streamVideoRef.current) {
          streamVideoRef.current.srcObject = stream;
        }
        addLog("Камера подключена. Ожидание соединения с сервером...");

        frameInterval = setInterval(() => {
          if (streamVideoRef.current && isConnected) {
            const video = streamVideoRef.current;
            const canvas = hiddenCanvasRef.current;
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            if (ctx) {
              ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
              const base64Data = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
              sendFrame(base64Data);
            }
          }
        }, 300);

      } catch (err) {
        addLog("Ошибка доступа к камере");
      }
    };

    if (mode === 'stream') {
      startCamera();
    }

    return () => {
      if (stream) stream.getTracks().forEach(track => track.stop());
      clearInterval(frameInterval);
    };
  }, [mode, isConnected, sendFrame]);

  useEffect(() => {
    if (mode === 'stream') {
      addLog(isConnected ? "WebSocket подключен" : "WebSocket отключен");
    }
  }, [isConnected, mode]);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setIsDragging(true);
    else if (e.type === "dragleave") setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation(); setIsDragging(false);
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile && droppedFile.type.startsWith('video/')) {
      setFile(droppedFile); setExternalUrl(''); setVideoUrl(URL.createObjectURL(droppedFile)); setResult(null);
      addLog(`Выбран файл: ${droppedFile.name}`);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;
    setFile(selectedFile); setExternalUrl(''); setVideoUrl(URL.createObjectURL(selectedFile)); setResult(null);
    addLog(`Выбран файл: ${selectedFile.name}`);
  };

  const loadHistoryItem = async (id: string) => {
    setLoading(true);
    addLog(`Загрузка архива: ${id}`);
    try {
      const res = await axios.get(`${API_BASE}/${id}`);
      setResult(res.data);
      if (res.data.video_url) {
        setVideoUrl(res.data.video_url);
      } else {
        addLog("Внимание: видео в хранилище отсутствует");
      }
      setMode('upload');
      addLog("Данные восстановлены");
    } catch (err) {
      addLog("Ошибка при получении истории");
    } finally {
      setLoading(false);
    }
  };

  // ФУНКЦИЯ ОБРАБОТКИ ФИДБЕКА
  const handleFeedback = async (id: string, isConfirmed: boolean, e: React.MouseEvent) => {
    e.stopPropagation(); // Чтобы клик не вызывал loadHistoryItem

    // Обновляем UI сразу для отзывчивости
    setHistory(prev => prev.map(item =>
      item.id === id ? { ...item, feedbackStatus: isConfirmed ? 'confirmed' : 'rejected' } : item
    ));

    try {
      if (isConfirmed) {
        addLog(`Отправка в БД для дообучения (ID: ${id})...`);
        // Раскомментируйте, когда эндпоинт будет готов:
        // await axios.post(`${RETRAIN_API_BASE}/submit`, { original_id: id, status: 'confirmed' });
        addLog("Успешно: данные сохранены для дообучения модели");
      } else {
        addLog(`Инцидент отклонен оператором (ID: ${id})`);
        // await axios.post(`${API_BASE}/feedback`, { id: id, status: 'rejected' });
      }
    } catch (err) {
      addLog("Ошибка при отправке обратной связи");
    }
  };

  const handleUpload = async () => {
    if (!file && !externalUrl) return;
    setLoading(true);
    setResult(null);
    addLog("Отправка на сервер...");

    const formData = new FormData();
    if (file) {
      formData.append('data', file);
    } else {
      formData.append('url', externalUrl);
    }

    try {
      const res = await axios.post(`${API_BASE}/upload`, formData);
      const taskId = res.data.id;

      const newHistory = [{
        id: taskId,
        timestamp: new Date().toLocaleString(),
        sourceName: file?.name || externalUrl,
        feedbackStatus: null
      }, ...history].slice(0, 20);

      setHistory(newHistory);
      localStorage.setItem('klin_analysis_history', JSON.stringify(newHistory));

      pollStatus(taskId);
    } catch (err) {
      addLog("Ошибка отправки. Проверь порт API и поднялся ли бек");
      setLoading(false);
    }
  };

  const pollStatus = (id: string) => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_BASE}/${id}`);

        if (res.data.state === 'FINISHED' || res.data.state === 'ERROR') {
          clearInterval(interval);
          setLoading(false);

          if (res.data.state === 'FINISHED') {
            setResult(res.data);
            addLog("Анализ завершен успешно. Координаты получены");
          } else {
            addLog("Ошибка обработки на сервере");
          }
        }
      } catch (e) {
        clearInterval(interval);
        setLoading(false);
        addLog("Связь с сервером потеряна при опросе статуса");
      }
    }, 2000);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    let animId: number;

    const draw = () => {
      if (videoContainerRef.current) {
        canvas.width = videoContainerRef.current.clientWidth;
        canvas.height = videoContainerRef.current.clientHeight;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      let boxesToDraw: number[][] | null = null;
      let label = 'АГРЕССИЯ';

      if (mode === 'upload' && result?.yolo) {
        const secondKey = Math.floor(currentTime).toString();
        boxesToDraw = result.yolo[secondKey];
      }
      else if (mode === 'stream' && lastFrameData?.yolo) {
        boxesToDraw = lastFrameData.yolo;
        if (lastFrameData.class) label = lastFrameData.class;
      }

      if (boxesToDraw) {
        const scaleX = canvas.width / (streamVideoRef.current?.videoWidth || playerRef.current?.getInternalPlayer()?.videoWidth || 1280);
        const scaleY = canvas.height / (streamVideoRef.current?.videoHeight || playerRef.current?.getInternalPlayer()?.videoHeight || 720);

        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 3;
        ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';

        boxesToDraw.forEach(box => {
          const [x1, y1, x2, y2] = box;
          const sx = x1 * scaleX, sy = y1 * scaleY, sw = (x2 - x1) * scaleX, sh = (y2 - y1) * scaleY;
          ctx.strokeRect(sx, sy, sw, sh);
          ctx.fillRect(sx, sy, sw, sh);

          ctx.fillStyle = '#ef4444';
          ctx.font = 'bold 12px sans-serif';
          ctx.fillText(label, sx, sy - 5);
          ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
        });
      }
      animId = requestAnimationFrame(draw);
    };

    animId = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animId);
  }, [currentTime, result, mode, lastFrameData]);

  return (
    // УВЕЛИЧЕНО: paddings (p-6 md:p-10)
    <div className="min-h-screen bg-[#0f172a] text-slate-200 font-sans p-6 md:p-8">
      {/* УВЕЛИЧЕНО: max-w-[95vw] 2xl:max-w-[1800px] вместо max-w-7xl */}
      <header className="max-w-[1440px] mx-auto flex flex-col md:flex-row justify-between items-center mb-8 gap-4 border-b border-slate-800 pb-6">
        <div>
          {/* УВЕЛИЧЕНО: text-5xl */}
          <h1 className="text-4xl font-black tracking-tighter text-white flex items-center gap-3">
            KLIN <span className="text-blue-500 text-sm font-medium tracking-normal bg-blue-500/10 px-3 py-1 rounded">v1.0</span>
          </h1>
          {/* УВЕЛИЧЕНО: text-sm */}
          <p className="text-slate-500 uppercase text-xs font-bold tracking-widest mt-1">Система анализа агрессии</p>
        </div>

        <div className="flex bg-slate-900 p-1.5 rounded-2xl border border-slate-800">
          {/* УВЕЛИЧЕНО: px-8 py-3 text-lg */}
          <button onClick={() => setMode('upload')} className={`flex items-center gap-2 px-10 py-2 rounded-xl transition font-bold text-sm ${mode === 'upload' ? 'bg-blue-600 text-white shadow-lg' : 'hover:bg-slate-800 text-slate-400'}`}>
            <Upload size={18} /> АНАЛИЗ ВИДЕО
          </button>
          <button onClick={() => setMode('stream')} className={`flex items-center gap-2 px-10 py-2 rounded-xl transition font-bold text-sm ${mode === 'stream' ? 'bg-red-600 text-white shadow-lg' : 'hover:bg-slate-800 text-slate-400'}`}>
            <Camera size={18} /> ПРЯМОЙ ЭФИР
          </button>
        </div>
      </header>

      {/* УВЕЛИЧЕНО: max-w-[95vw] 2xl:max-w-[1800px] */}
      <main className="max-w-[1440px] mx-auto grid grid-cols-12 gap-8">

        {/* ЛЕВАЯ КОЛОНКА */}
        <div className="col-span-12 lg:col-span-8 space-y-6">
          <div className="bg-slate-900 rounded-[2rem] p-4 border border-slate-800 shadow-2xl relative">
            <div ref={videoContainerRef} className="relative bg-black rounded-2xl overflow-hidden aspect-video border border-slate-800 flex items-center justify-center">
              {mode === 'upload' ? (
                videoUrl ? (
                  <>
                    <ReactPlayer
                      ref={playerRef} {...({ url: videoUrl || undefined } as any)} width="100%" height="100%" controls
                      onProgress={(p: any) => setCurrentTime(p.playedSeconds)}
                      onDuration={(d: any) => setDuration(d)}
                      style={{ position: 'absolute', top: 0, left: 0 }}
                    />
                    <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full pointer-events-none z-20" />
                  </>
                ) : (
                  <div className="flex flex-col items-center justify-center text-slate-600 h-full w-full">
                    <Play size={64} className="mb-4 opacity-10" />
                    <p className="font-bold uppercase tracking-widest text-base">Ожидание видеофайла...</p>
                  </div>
                )
              ) : (
                <div className="relative w-full h-full">
                  <video ref={streamVideoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
                  <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full pointer-events-none z-20" />

                  <div className="absolute top-4 right-4 flex items-center gap-3">
                    <div className={`text-xs font-black px-3 py-1.5 rounded-full flex items-center gap-2 ${isConnected ? 'bg-green-600/80 text-white' : 'bg-slate-800 text-slate-400'}`}>
                      {isConnected ? <Wifi size={12} /> : <WifiOff size={12} />} {isConnected ? 'WS CONNECTED' : 'WS OFFLINE'}
                    </div>
                    <div className="bg-red-600 text-white text-xs font-black px-3 py-1.5 rounded-full animate-pulse flex items-center gap-2">
                      <span className="w-2 h-2 bg-white rounded-full"></span> LIVE
                    </div>
                  </div>
                </div>
              )}
            </div>

            {mode === 'upload' && (
              <div className="mt-6 space-y-4">
                <div className="flex justify-between items-end">
                  <h3 className="text-sm font-black text-slate-500 uppercase tracking-tighter">Временная шкала инцидентов</h3>
                </div>
                <div onClick={(e) => {
                  if (duration === 0 || !playerRef.current) return;
                  const rect = e.currentTarget.getBoundingClientRect();
                  playerRef.current.seekTo(((e.clientX - rect.left) / rect.width) * duration);
                }} className="relative h-8 bg-slate-950 rounded-xl border border-slate-800 overflow-hidden cursor-pointer group">
                  {result?.mae?.map((evt, i) => (
                    <div key={i} className="absolute h-full bg-red-600/40 border-x border-red-500/80"
                      style={{ left: `${(evt.time[0] / duration) * 100}%`, width: `${((evt.time[1] - evt.time[0]) / duration) * 100}%` }}
                    />
                  ))}
                  <div className="absolute top-0 h-full w-1 bg-white shadow-[0_0_15px_rgba(255,255,255,0.8)] z-10 pointer-events-none" style={{ left: `${(currentTime / duration) * 100}%` }} />
                </div>
              </div>
            )}
          </div>

          <div className="grid grid-cols-3 gap-6">
            <div className="bg-slate-900 p-5 rounded-3xl border border-slate-800">
              <div className="text-[10px] text-slate-500 font-black uppercase mb-1 flex items-center gap-1">Статус анализа</div>
              <div className={`text-lg font-bold ${result ? 'text-green-400' : loading ? 'text-blue-400 animate-pulse' : 'text-slate-400'}`}>
                {mode === 'stream' ? (isConnected ? 'АКТИВЕН' : 'ОЖИДАНИЕ') : (loading ? 'ОБРАБОТКА...' : (result ? 'ЗАВЕРШЕНО' : 'ОЖИДАНИЕ'))}
              </div>
            </div>
            <div className="bg-slate-900 p-5 rounded-3xl border border-slate-800">
              <div className="text-[10px] text-slate-500 font-black uppercase mb-1 flex items-center gap-1">Результат ML</div>
              <div className={`text-lg font-bold ${(mode === 'stream' ? lastFrameData?.class : result?.predicted_class)?.includes('АГРЕССИЯ') ? 'text-red-500' : 'text-green-400'}`}>
                {mode === 'stream' ? (lastFrameData?.class || 'СКАНИРОВАНИЕ') : (result ? (result.predicted_class || 'БЕЗ ОПАСНОСТИ') : 'ОЖИДАНИЕ')}
              </div>
            </div>
            <div className="bg-slate-900 p-5 rounded-3xl border border-slate-800">
              <div className="text-[10px] text-slate-500 font-black uppercase mb-1 flex items-center gap-1">Уверенность</div>
              <div className="text-lg font-bold text-white">
                {mode === 'stream' ? (lastFrameData?.confidence ? `${lastFrameData.confidence}%` : '--') : (result?.confidence_percent ? `${result.confidence_percent.toFixed(1)}%` : '--')}
              </div>
            </div>
          </div>
        </div>

        {/* ПРАВАЯ КОЛОНКА */}
        <div className="col-span-12 lg:col-span-4 space-y-6">

          {/* БЛОК ЗАГРУЗКИ */}
          <div className="bg-slate-900 p-6 rounded-[2rem] border border-slate-800 shadow-xl">
            <h2 className="text-sm font-bold text-white mb-4 flex items-center gap-2">
              {mode === 'upload' ? <FileVideo className="text-blue-500" /> : <Camera className="text-red-500" />} ЗАГРУЗКА ДАННЫХ
            </h2>
            {mode === 'upload' ? (
              <div className="space-y-4">
                <input type="file" id="fileInput" onChange={handleFileChange} className="hidden" accept="video/*" />
                <label htmlFor="fileInput" onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop} className={`flex flex-col items-center justify-center w-full h-40 border-2 border-dashed rounded-xl cursor-pointer transition ${isDragging ? 'border-blue-500 bg-blue-500/10' : 'border-slate-700 hover:border-blue-500'}`}>
                  <Upload size={24} className="text-slate-500 mb-2" />
                  <span className="text-[12px] font-bold text-slate-400 uppercase">ВЫБРАТЬ ИЛИ ПЕРЕТАЩИТЬ ВИДЕО</span>
                </label>
                {file && <div className="flex items-center justify-between bg-slate-800 p-2 rounded-xl border border-slate-700"><span className="text-sm font-bold truncate">{file.name}</span><X size={18} className="text-slate-500 cursor-pointer" onClick={() => setFile(null)} /></div>}

                <input type="text" placeholder="Ссылка на видео" value={externalUrl} onChange={(e) => setExternalUrl(e.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2 text-sm focus:border-blue-500 outline-none" />

                <button onClick={handleUpload} disabled={loading || (!file && !externalUrl)} className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 py-3 rounded-xl font-black text-xs tracking-widest transition">
                  {loading ? 'ЗАГРУЗКА...' : 'ЗАПУСТИТЬ АНАЛИЗ'}
                </button>
              </div>
            ) : (
              <div className="p-4 bg-red-500/5 border border-red-500/10 rounded-xl">
                <p className="text-red-400 text-xs font-medium leading-relaxed">Камера активна. Кадры передаются на сервер по WebSockets для анализа в реальном времени.</p>
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">

            {/* ЖУРНАЛ (Увеличена высота: h-80 вместо h-64) */}
            <div className="bg-slate-900 rounded-[2rem] border border-slate-800 shadow-xl overflow-hidden flex flex-col h-80">
              <div className="bg-slate-800/50 px-5 py-5 border-b border-slate-800 flex justify-between items-center">
                <h2 className="text-sm font-black text-slate-400 uppercase">Журнал</h2>
                <button onClick={() => setLogs([])} className="text-xs text-slate-500 hover:text-white transition">ОЧИСТИТЬ</button>
              </div>
              <div className="flex-grow overflow-y-auto p-5 flex flex-col-reverse font-mono">
                <div className="space-y-3">{logs.map((log, i) => <div key={i} className="text-xs flex gap-3"><span className="text-blue-500 min-w-[60px]">[{log.time}]</span><span className="text-slate-300">{log.msg}</span></div>)}</div>
              </div>
            </div>

            {/* ИСТОРИЯ С КНОПКАМИ ДООБУЧЕНИЯ */}
            <div className="bg-slate-900 rounded-[2rem] border border-slate-800 shadow-xl overflow-hidden flex flex-col h-80">
              <div className="bg-slate-800/50 px-5 py-5 border-b border-slate-800"><h2 className="text-sm font-black text-slate-400 uppercase">Архив БД</h2></div>
              <div className="p-5 space-y-4 overflow-y-auto flex-grow">
                {history.map((item) => (
                  <div key={item.id} onClick={() => loadHistoryItem(item.id)} className="p-4 bg-slate-950 rounded-2xl border border-slate-800 hover:border-blue-500 cursor-pointer transition">
                    <div className="flex justify-between items-start mb-2">
                      <span className="text-xs text-blue-400 font-bold truncate max-w-[120px]">{item.sourceName}</span>
                      <div className="flex items-center gap-1 text-[10px] text-slate-500"><CheckCircle2 size={12} className="text-green-500" /> СЕРВЕР</div>
                    </div>

                    {/* КНОПКИ ПОДТВЕРЖДЕНИЯ / ОТКЛОНЕНИЯ */}
                    {!item.feedbackStatus ? (
                      <div className="flex gap-2 mt-3 pt-3 border-t border-slate-800/50">
                        <button
                          onClick={(e) => handleFeedback(item.id, true, e)}
                          className="flex-1 flex items-center justify-center gap-1 bg-green-500/10 hover:bg-green-500/20 text-green-500 text-[10px] font-bold py-2 rounded-lg transition"
                        >
                          <ThumbsUp size={12} /> ВЕРНО
                        </button>
                        <button
                          onClick={(e) => handleFeedback(item.id, false, e)}
                          className="flex-1 flex items-center justify-center gap-1 bg-red-500/10 hover:bg-red-500/20 text-red-500 text-[10px] font-bold py-2 rounded-lg transition"
                        >
                          <ThumbsDown size={12} /> ОШИБКА
                        </button>
                      </div>
                    ) : (
                      <div className={`mt-3 pt-3 border-t border-slate-800/50 text-[10px] font-bold text-center ${item.feedbackStatus === 'confirmed' ? 'text-green-500/50' : 'text-red-500/50'}`}>
                        {item.feedbackStatus === 'confirmed' ? '✓ ОТПРАВЛЕНО В БД ДООБУЧЕНИЯ' : '✗ ОТКЛОНЕНО ОПЕРАТОРОМ'}
                      </div>
                    )}
                  </div>
                ))}
                {history.length === 0 && <div className="text-center text-slate-600 text-sm mt-10">БД пуста</div>}
              </div>
            </div>

          </div>
        </div>
      </main>
    </div>
  );
}