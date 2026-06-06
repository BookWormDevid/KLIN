import { useEffect, useRef, useState } from 'react';
import type { StreamState } from '../core/domain/entities/StreamState';
import { appConfig } from '../config/appConfig';
import type { ApiStreamRepository } from '../core/infrastructure/repositories/apiStreamRepository';

export function useStream(streamRepo: ApiStreamRepository) {
    const [streamState, setStreamState] = useState<StreamState | null>(null);
    const [isActive, setIsActive] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const pollIntervalRef = useRef<number | null>(null);
    const currentStreamIdRef = useRef<string | null>(null);

    useEffect(() => {
        return () => {
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
        };
    }, []);

    const start = async (cameraUrl: string, cameraId: string) => {
        setError(null);
        try {
            const state = await streamRepo.startStream(cameraUrl, cameraId);
            currentStreamIdRef.current = state.id;
            setStreamState(state);
            setIsActive(true);
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = window.setInterval(async () => {
                if (!currentStreamIdRef.current) return;
                try {
                    const updated = await streamRepo.getStatus(currentStreamIdRef.current);
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
        if (!currentStreamIdRef.current) return;
        try {
            await streamRepo.stopStream(currentStreamIdRef.current);
            setIsActive(false);
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
        } catch (err: any) {
            setError(err.message || 'Failed to stop stream');
        }
    };

    return { streamState, isActive, error, start, stop };
}
