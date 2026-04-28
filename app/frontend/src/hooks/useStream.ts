import { useEffect, useRef, useState } from 'react';
import { HttpClient } from '../core/infrastructure/http/httpClient';
import { ApiStreamRepository } from '../core/infrastructure/repositories/apiStreamRepository';
import { startStream, stopStream, getStreamStatus } from '../core/application/usecases';
import type { StreamState } from '../core/domain/entities/StreamState';
import { appConfig } from '../config/appConfig';

export function useStream() {
    const [streamState, setStreamState] = useState<StreamState | null>(null);
    const [isActive, setIsActive] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const repoRef = useRef<ApiStreamRepository | null>(null);
    const pollIntervalRef = useRef<number | null>(null);
    const currentStreamIdRef = useRef<string | null>(null);

    useEffect(() => {
        repoRef.current = new ApiStreamRepository(new HttpClient());
        return () => {
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
            if (currentStreamIdRef.current) {
                stopStream(repoRef.current!, currentStreamIdRef.current).catch(console.error);
            }
        };
    }, []);

    const start = async (cameraUrl: string, cameraId: string) => {
        if (!repoRef.current) return;
        setError(null);
        try {
            const state = await startStream(repoRef.current, cameraUrl, cameraId);
            currentStreamIdRef.current = state.id;
            setStreamState(state);
            setIsActive(true);
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = window.setInterval(async () => {
                if (!currentStreamIdRef.current) return;
                try {
                    const updated = await getStreamStatus(repoRef.current!, currentStreamIdRef.current);
                    setStreamState(updated);
                    if (updated.state === 'STOPPED' || updated.state === 'ERROR') {
                        setIsActive(false);
                        if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
                    }
                } catch (err) {
                    console.error('Poll error', err);
                }
            }, appConfig.streamPollIntervalMs);
        } catch (err: any) {
            setError(err.message || 'Failed to start stream');
            setIsActive(false);
        }
    };

    const stop = async () => {
        if (!repoRef.current || !currentStreamIdRef.current) return;
        try {
            await stopStream(repoRef.current, currentStreamIdRef.current);
            setIsActive(false);
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
        } catch (err: any) {
            setError(err.message || 'Failed to stop stream');
        }
    };

    return { streamState, isActive, error, start, stop };
}
