import React, { useState, useEffect, useRef } from 'react';
// @ts-ignore
import ReactPlayer from 'react-player';
import axios from 'axios';
import {
  Upload, Camera, History, AlertTriangle, CheckCircle2,
  Play, RefreshCw, Link as LinkIcon, FileVideo, X, Info, ShieldAlert
} from 'lucide-react';

// --- ТИПЫ ---
interface AnalysisResult {
  id: string;
  state: 'PENDING' | 'FINISHED' | 'ERROR';
  predicted_class?: string;
  confidence_percent?: number;
  video_url?: string; // Поле для S3 ссылки
  processing_info?: {
    processing_time_seconds: number;
    video_duration: number;
    total_frames: number;
    video_fps: number;
  };
  mae: any[] | null;
  yolo: Record<string, number[][]> | null;
}

interface HistoryItem {
  id: string;
  timestamp: string;
  sourceName: string;
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
  const isInitialized = useRef(false);

  // Инициализация 
  useEffect(() => {
    if (!isInitialized.current) {
      const saved = localStorage.getItem('klin_analysis_history');
      if (saved) setHistory(JSON.parse(saved));
      addLog("Система готова к работе");
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

  // Drag & Drop логика
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setIsDragging(true);
    else if (e.type === "dragleave") setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile && droppedFile.type.startsWith('video/')) {
      setFile(droppedFile);
      setExternalUrl('');
      setVideoUrl(URL.createObjectURL(droppedFile));
      addLog(`Выбран файл (drop): ${droppedFile.name}`);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;
    if (!selectedFile.type.startsWith('video/')) {
      alert('Ошибка: Пожалуйста, выберите видеофайл (MP4, AVI, MOV)');
      return;
    }
    setFile(selectedFile);
    setExternalUrl('');
    setVideoUrl(URL.createObjectURL(selectedFile));
    addLog(`Выбран файл: ${selectedFile.name}`);
  };

  // Функция возврата к ранее обработанному видео
  const loadHistoryItem = async (id: string) => {
    setLoading(true);
    addLog(`Загрузка архива: ${id}`);
    try {
      const res = await axios.get(`http://localhost:8000/api/v1/Klin/${id}`);
      setResult(res.data);
      if (res.data.video_url) setVideoUrl(res.data.video_url);
      addLog("Данные восстановлены из S3");
    } catch (err) {
      addLog("Ошибка при получении истории");
    } finally {
      setLoading(false);
    }
  };

  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (duration === 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const clickedTime = (x / rect.width) * duration;
    playerRef.current?.seekTo(clickedTime);
  };

  const handleUpload = async () => {
    if (!file && !externalUrl) return;
    setLoading(true);
    addLog("Запуск анализа...");

    const formData = new FormData();
    if (file) formData.append('data', file);
    else formData.append('url', externalUrl);

    try {
      const res = await axios.post('http://localhost:8000/api/v1/Klin/upload', formData);
      const taskId = res.data.id;

      const newHistory = [{ id: taskId, timestamp: new Date().toLocaleString(), sourceName: file?.name || externalUrl }, ...history].slice(0, 10);
      setHistory(newHistory);
      localStorage.setItem('klin_analysis_history', JSON.stringify(newHistory));

      pollStatus(taskId);
    } catch (err) {
      addLog("Ошибка отправки на сервер");
      setLoading(false);
      alert("Ошибка! Проверьте Docker-контейнеры.");
    }
  };

  const pollStatus = async (id: string) => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`http://localhost:8000/api/v1/Klin/${id}`);
        if (res.data.state === 'FINISHED') {
          setResult(res.data);
          if (res.data.video_url) setVideoUrl(res.data.video_url);
          setLoading(false);
          addLog("Анализ успешно завершен");
          clearInterval(interval);
        }
      } catch (e) {
        clearInterval(interval);
      }
    }, 3000);
  };

  // Отрисовка BBox (Bounding Boxes)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !result?.yolo) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const secondKey = Math.floor(currentTime).toString();
      const boxes = result.yolo ? result.yolo[secondKey] : null;

      if (boxes) {
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 3;
        boxes.forEach(box => {
          const [x1, y1, x2, y2] = box;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        });
      }
      requestAnimationFrame(draw);
    };
    const anim = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(anim);
  }, [currentTime, result]);

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 font-sans p-4 md:p-8">
      {/* HEADER */}
      <header className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center mb-8 gap-4 border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-4xl font-black tracking-tighter text-white flex items-center gap-2">
            КЛИН <span className="text-blue-500 text-lg font-medium tracking-normal bg-blue-500/10 px-2 py-1 rounded">v1.0</span>
          </h1>
          <p className="text-slate-500 uppercase text-xs font-bold tracking-widest mt-1">Система анализа агрессии с видеопотоков</p>
        </div>

        <div className="flex bg-slate-900 p-1 rounded-2xl border border-slate-800">
          <button onClick={() => setMode('upload')} className={`flex items-center gap-2 px-6 py-2 rounded-xl transition font-bold ${mode === 'upload' ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' : 'hover:bg-slate-800 text-slate-400'}`}>
            <Upload size={18} /> АНАЛИЗ ВИДЕО
          </button>
          <button onClick={() => setMode('stream')} className={`flex items-center gap-2 px-6 py-2 rounded-xl transition font-bold ${mode === 'stream' ? 'bg-red-600 text-white shadow-lg shadow-red-900/20' : 'hover:bg-slate-800 text-slate-400'}`}>
            <Camera size={18} /> ПРЯМОЙ ЭФИР
          </button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-12 gap-8">

        {/* ЛЕВАЯ ЧАСТЬ: Плеер и Шкала */}
        <div className="col-span-12 lg:col-span-8 space-y-6">
          <div className="bg-slate-900 rounded-3xl p-4 border border-slate-800 shadow-2xl">
            <div className="relative bg-black rounded-2xl overflow-hidden aspect-video border border-slate-800">
              {videoUrl ? (
                <>
                  <ReactPlayer
                    ref={playerRef}
                    url={videoUrl}
                    width="100%" height="100%" controls
                    onProgress={(p: any) => setCurrentTime(p.playedSeconds)}
                    onDuration={(d: any) => setDuration(d)}
                  />
                  <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full pointer-events-none z-20" width={1280} height={720} />
                </>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-slate-600">
                  <Play size={64} className="mb-4 opacity-10" />
                  <p className="font-bold uppercase tracking-widest text-sm">Ожидание загрузки данных...</p>
                </div>
              )}
            </div>

            {/* ТАЙМЛАЙН */}
            <div className="mt-6 space-y-3">
              <div className="flex justify-between items-end">
                <h3 className="text-xs font-black text-slate-500 uppercase tracking-tighter">Временная шкала инцидентов</h3>
                <span className="text-[10px] bg-red-500/20 text-red-400 px-2 py-0.5 rounded-full font-bold">Клик для перемотки</span>
              </div>
              <div onClick={handleTimelineClick} className="relative h-8 bg-slate-950 rounded-xl border border-slate-800 overflow-hidden cursor-pointer group">
                {result?.mae?.map((evt: any, i: number) => (
                  <div key={i} className="absolute h-full bg-red-600/40 border-x border-red-500/50"
                    style={{ left: `${(evt.time[0] / duration) * 100}%`, width: `${((evt.time[1] - evt.time[0]) / duration) * 100}%` }}
                  />
                ))}
                <div className="absolute top-0 h-full w-1 bg-white shadow-[0_0_15px_rgba(255,255,255,0.8)] z-10 transition-all"
                  style={{ left: `${(currentTime / duration) * 100}%` }} />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="bg-slate-900 p-4 rounded-2xl border border-slate-800">
              <div className="text-[10px] text-slate-500 font-black uppercase mb-1 flex items-center gap-1">
                <RefreshCw size={10} /> Статус анализа
              </div>
              <div className={`text-sm font-bold ${result ? 'text-green-400' : 'text-blue-400 animate-pulse'}`}>
                {loading ? 'ОБРАБОТКА...' : (result ? 'ЗАВЕРШЕНО' : 'ОЖИДАНИЕ')}
              </div>
            </div>
            <div className="bg-slate-900 p-4 rounded-2xl border border-slate-800">
              <div className="text-[10px] text-slate-500 font-black uppercase mb-1 flex items-center gap-1">
                <ShieldAlert size={10} /> Результат ML
              </div>
              <div className={`text-sm font-bold ${result?.predicted_class === 'АГРЕССИЯ' ? 'text-red-500' : 'text-green-400'}`}>
                {result?.predicted_class || '--'}
              </div>
            </div>
            <div className="bg-slate-900 p-4 rounded-2xl border border-slate-800">
              <div className="text-[10px] text-slate-500 font-black uppercase mb-1 flex items-center gap-1">
                <Info size={10} /> Уверенность
              </div>
              <div className="text-sm font-bold text-white">
                {result?.confidence_percent ? `${result.confidence_percent.toFixed(1)}%` : '--'}
              </div>
            </div>
          </div>
        </div>

        {/* ПРАВАЯ ЧАСТЬ: Загрузка и Логи */}
        <div className="col-span-12 lg:col-span-4 space-y-6">
          <div className="bg-slate-900 p-6 rounded-3xl border border-slate-800 shadow-xl">
            <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <FileVideo className="text-blue-500" /> {mode === 'upload' ? 'ЗАГРУЗКА ВИДЕО' : 'RTSP ПОТОК'}
            </h2>

            {mode === 'upload' ? (
              <div className="space-y-4">
                <div className="relative group">
                  <input type="file" id="fileInput" onChange={handleFileChange} className="hidden" accept="video/*" />
                  <label
                    htmlFor="fileInput"
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    className={`flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-2xl cursor-pointer transition ${isDragging ? 'border-blue-500 bg-blue-500/10' : 'border-slate-700 hover:border-blue-500 hover:bg-blue-500/5'}`}
                  >
                    <Upload className={`${isDragging ? 'text-blue-400 animate-bounce' : 'text-slate-500'} mb-2`} />
                    <span className="text-xs font-bold text-slate-400 uppercase tracking-tighter">
                      {isDragging ? 'ОТПУСТИТЕ ФАЙЛ' : 'ВЫБРАТЬ ИЛИ ПЕРЕТАЩИТЬ ФАЙЛ'}
                    </span>
                  </label>
                </div>

                {file && (
                  <div className="flex items-center justify-between bg-slate-800 p-3 rounded-xl">
                    <div className="flex items-center gap-2 overflow-hidden">
                      <FileVideo size={16} className="text-blue-400 flex-shrink-0" />
                      <span className="text-xs font-bold truncate">{file.name}</span>
                    </div>
                    <X size={16} className="text-slate-500 cursor-pointer" onClick={() => { setFile(null); setVideoUrl(null) }} />
                  </div>
                )}

                <div className="flex items-center gap-2 py-2">
                  <div className="h-px bg-slate-800 flex-grow" />
                  <span className="text-[10px] font-black text-slate-600">ИЛИ ССЫЛКА</span>
                  <div className="h-px bg-slate-800 flex-grow" />
                </div>

                <div className="relative">
                  <input
                    type="text"
                    placeholder="https://..."
                    value={externalUrl}
                    onChange={(e) => { setExternalUrl(e.target.value); if (e.target.value) setFile(null); }}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-sm focus:border-blue-500 outline-none transition"
                  />
                  <LinkIcon size={14} className="absolute right-4 top-3.5 text-slate-600" />
                </div>

                <button
                  onClick={handleUpload}
                  disabled={loading || (!file && !externalUrl)}
                  className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-800 py-4 rounded-2xl font-black text-xs tracking-widest transition shadow-lg shadow-blue-600/20 flex justify-center items-center gap-2"
                >
                  {loading ? <RefreshCw className="animate-spin" size={18} /> : 'ЗАПУСТИТЬ АНАЛИЗ'}
                </button>
              </div>
            ) : (
              <div className="p-4 bg-red-500/5 border border-red-500/20 rounded-2xl">
                <p className="text-red-400 text-[11px] leading-relaxed font-medium">
                  ВНИМАНИЕ: Режим прямого эфира использует протокол RTSP. События будут отображаться в журнале в реальном времени.
                </p>
              </div>
            )}
          </div>

          {/* ЖУРНАЛ СОБЫТИЙ */}
          <div className="bg-slate-900 rounded-3xl border border-slate-800 shadow-xl overflow-hidden">
            <div className="bg-slate-800/50 px-6 py-4 border-b border-slate-800 flex justify-between items-center">
              <h2 className="text-xs font-black tracking-widest text-slate-400 uppercase">Журнал событий</h2>
              <button onClick={() => setLogs([])} className="text-[10px] text-slate-500 hover:text-white transition">ОЧИСТИТЬ</button>
            </div>
            <div className="h-48 overflow-y-auto p-4 space-y-2 font-mono">
              {logs.map((log, i) => (
                <div key={i} className="text-[10px] flex gap-2">
                  <span className="text-blue-500">[{log.time}]</span>
                  <span className="text-slate-400">{log.msg}</span>
                </div>
              ))}
              {logs.length === 0 && <div className="text-[10px] text-slate-600 italic text-center py-8">Журнал пуст</div>}
            </div>
          </div>

          {/* ИСТОРИЯ */}
          <div className="bg-slate-900 rounded-3xl border border-slate-800 shadow-xl overflow-hidden">
            <div className="bg-slate-800/50 px-6 py-4 border-b border-slate-800">
              <h2 className="text-xs font-black tracking-widest text-slate-400 uppercase">Последние анализы</h2>
            </div>
            <div className="p-4 space-y-3">
              {history.map((item) => (
                <div
                  key={item.id}
                  onClick={() => loadHistoryItem(item.id)}
                  className="group p-3 bg-slate-950 rounded-xl border border-slate-800 hover:border-blue-500 transition cursor-pointer"
                >
                  <div className="flex justify-between items-start mb-1">
                    <span className="text-[10px] text-blue-500 font-bold truncate max-w-[120px]">{item.sourceName}</span>
                    <span className="text-[9px] text-slate-600">{item.timestamp}</span>
                  </div>
                  <div className="flex items-center gap-1 text-[9px] text-slate-500">
                    <CheckCircle2 size={10} className="text-green-500" /> Нажмите для просмотра
                  </div>
                </div>
              ))}
              {history.length === 0 && <div className="text-[10px] text-slate-600 text-center py-4 uppercase">История пуста</div>}
            </div>
          </div>

        </div>
      </main>

      <footer className="max-w-7xl mx-auto mt-12 pt-6 border-t border-slate-800 flex justify-between items-center text-slate-600">
        <p className="text-[10px] font-bold tracking-widest">КЛИН • AI AGGRESSION DETECTION SYSTEM</p>
        <p className="text-[10px]">© 2026 KLIN VISION LAB</p>
      </footer>
    </div>
  );
}