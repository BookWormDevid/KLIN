import React, { useRef, forwardRef, useImperativeHandle } from 'react';
import './VideoPlayer.css';

interface VideoPlayerProps {
    url: string | null;
    onProgress?: (currentTime: number) => void;
    onDuration?: (duration: number) => void;
}

export interface VideoPlayerRef {
    getVideoElement: () => HTMLVideoElement | null;
    seekTo: (time: number) => void;
}

export const VideoPlayer = forwardRef<VideoPlayerRef, VideoPlayerProps>(
    ({ url, onProgress, onDuration }, ref) => {
        const videoRef = useRef<HTMLVideoElement>(null);

        useImperativeHandle(ref, () => ({
            getVideoElement: () => videoRef.current,
            seekTo: (time: number) => {
                if (videoRef.current) videoRef.current.currentTime = time;
            },
        }));

        const handleTimeUpdate = () => {
            if (videoRef.current && onProgress) onProgress(videoRef.current.currentTime);
        };

        const handleLoadedMetadata = () => {
            if (videoRef.current && onDuration) onDuration(videoRef.current.duration);
        };

        if (!url) return <div className="video-placeholder">Нет видео</div>;

        return (
            <div className="video-player-wrapper">
                <video
                    ref={videoRef}
                    src={url}
                    controls
                    className="video-element"
                    onTimeUpdate={handleTimeUpdate}
                    onLoadedMetadata={handleLoadedMetadata}
                />
            </div>
        );
    }
);
