import { HttpClient } from '../http/httpClient';
import type { StreamState, StreamProcessingState } from '../../domain/entities/StreamState';
import { appConfig } from '../../../config/appConfig';

function createMockStreamState(overrides: Partial<StreamState> = {}): StreamState {
    return {
        id: overrides.id || `stream-mock-${Date.now()}`,
        camera_id: overrides.camera_id || 'mock-cam',
        camera_url: overrides.camera_url || null,
        state: (overrides.state as StreamProcessingState) || 'PROCESSING',
        last_x3d_label: 'violence',
        last_x3d_confidence: 0.87,
        last_mae_label: 'Fighting',
        last_mae_confidence: 0.92,
        objects: ['person'],
        all_classes: ['Fighting'],
    };
}

export class ApiStreamRepository {
    private http: HttpClient;

    constructor(http: HttpClient) {
        this.http = http;
    }

    async startStream(cameraUrl: string, cameraId: string): Promise<StreamState> {
        if (appConfig.useMocks) {
            await new Promise(res => setTimeout(res, 500));
            return createMockStreamState({ camera_url: cameraUrl, camera_id: cameraId });
        }
        const raw = await this.http.post<any>('/Klin_Stream/upload', {
            camera_url: cameraUrl,
            camera_id: cameraId,
        });
        return this.parseStreamState(raw);
    }

    async stopStream(streamId: string): Promise<void> {
        if (appConfig.useMocks) return;
        await this.http.post(`/Klin_Stream/${streamId}/stop`, {});
    }

    async getStatus(streamId: string): Promise<StreamState> {
        if (appConfig.useMocks) {
            await new Promise(res => setTimeout(res, 300));
            return createMockStreamState({ id: streamId });
        }
        const raw = await this.http.get<any>(`/Klin_Stream/${streamId}`);
        return this.parseStreamState(raw);
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
