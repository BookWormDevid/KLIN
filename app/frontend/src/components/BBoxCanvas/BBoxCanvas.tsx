import React, { useRef, useEffect } from 'react';
import './BBoxCanvas.css';

interface BBoxCanvasProps {
    videoElement: HTMLVideoElement | null;
    yoloData: Record<number, number[][]> | null;   // ключ – timestamp (сек)
    currentTime: number;
    label?: string;
}

export const BBoxCanvas: React.FC<BBoxCanvasProps> = ({
    videoElement,
    yoloData,
    currentTime,
    label = 'АГРЕССИЯ'
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !videoElement) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const draw = () => {
            const videoRect = videoElement.getBoundingClientRect();
            canvas.width = videoRect.width;
            canvas.height = videoRect.height;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (!yoloData) return;

            // Найти ближайший timestamp <= currentTime
            const timestamps = Object.keys(yoloData).map(Number).sort((a, b) => a - b);
            let bestTimestamp: number | null = null;
            for (const t of timestamps) {
                if (t <= currentTime) bestTimestamp = t;
                else break;
            }
            if (bestTimestamp === null) return;

            const detections = yoloData[bestTimestamp];
            if (!detections || detections.length === 0) return;

            const videoWidth = videoElement.videoWidth;
            const videoHeight = videoElement.videoHeight;
            const scaleX = canvas.width / videoWidth;
            const scaleY = canvas.height / videoHeight;

            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 3;
            ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';

            detections.forEach(([x1, y1, x2, y2]) => {
                const sx = x1 * scaleX;
                const sy = y1 * scaleY;
                const sw = (x2 - x1) * scaleX;
                const sh = (y2 - y1) * scaleY;
                ctx.strokeRect(sx, sy, sw, sh);
                ctx.fillRect(sx, sy, sw, sh);
            });

            // Подпись
            ctx.fillStyle = '#ef4444';
            ctx.font = 'bold 14px sans-serif';
            ctx.fillText(label, 10, 30);
        };

        const animation = requestAnimationFrame(draw);
        return () => cancelAnimationFrame(animation);
    }, [videoElement, yoloData, currentTime, label]);

    return <canvas ref={canvasRef} className="bbox-canvas" />;
};
