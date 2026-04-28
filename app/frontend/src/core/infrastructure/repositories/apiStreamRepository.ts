import { HttpClient } from '../http/httpClient';
import type { StreamState, StreamProcessingState } from '../../domain/entities/StreamState';

export class ApiStreamRepository {
    private http: HttpClient;

    constructor(http: HttpClient) {
        this.http = http;
    }

    async startStream(cameraUrl: string, cameraId: string): Promise<StreamState> {
        const raw = await this.http.post<any>('/Klin_Stream/upload', {
            camera_url: cameraUrl,
            camera_id: cameraId,
        });
        return this.parseStreamState(raw);
    }

    async stopStream(streamId: string): Promise<void> {
        await this.http.post(`/Klin_Stream/${streamId}/stop`, {});
    }

    async getStatus(streamId: string): Promise<StreamState> {
        const raw = await this.http.get<any>(`/Klin_Stream/${streamId}`);
        return this.parseStreamState(raw);
    }

    // Заглушки для будущей реализации WebSocket
    isConnected(): boolean {
        // Пока WebSocket не используется, всегда возвращаем false
        return false;
    }

    sendFrame(frameBase64: string): void {
        // В будущем здесь будет отправка кадра через WebSocket
        console.warn('sendFrame is not implemented yet');
    }

    private parseStreamState(raw: any): StreamState {
        return {
            id: raw.id,
            camera_id: raw.camera_id,
            camera_url: raw.camera_url,
            state: raw.state as StreamProcessingState,
            last_x3d_label: raw.last_x3d_label ?? null,
            last_x3d_confidence: raw.last_x3d_confidence ?? null,
            last_mae_label: raw.last_mae_label ?? null,
            last_mae_confidence: raw.last_mae_confidence ?? null,
            objects: raw.objects ?? null,
            all_classes: raw.all_classes ?? null,
        };
    }
}
