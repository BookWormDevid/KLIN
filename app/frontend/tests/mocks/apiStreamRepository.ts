export type StreamProcessingState = 'PENDING' | 'PROCESSING' | 'STOPPED' | 'ERROR';
export interface StreamState {
    id: string;
    camera_id: string;
    camera_url: string | null;
    state: StreamProcessingState;
    last_x3d_label: string | null;
    last_x3d_confidence: number | null;
    last_mae_label: string | null;
    last_mae_confidence: number | null;
    objects: string[] | null;
    all_classes: string[] | null;
}

export class ApiStreamRepository {
    private mockState: StreamState = {
        id: 'mock-stream-1', camera_id: 'cam1', camera_url: null, state: 'PROCESSING',
        last_x3d_label: null, last_x3d_confidence: null,
        last_mae_label: null, last_mae_confidence: null,
        objects: null, all_classes: null,
    };

    async startStream(cameraUrl: string, cameraId: string): Promise<StreamState> {
        await new Promise(r => setTimeout(r, 500));
        this.mockState = { ...this.mockState, camera_url: cameraUrl, camera_id: cameraId, state: 'PROCESSING' };
        return this.mockState;
    }
    async stopStream(streamId: string): Promise<void> {
        await new Promise(r => setTimeout(r, 300));
        this.mockState.state = 'STOPPED';
    }
    async getStatus(streamId: string): Promise<StreamState> {
        await new Promise(r => setTimeout(r, 200));
        const now = Date.now();
        if (now % 10000 > 5000) {
            this.mockState.last_x3d_label = 'True';
            this.mockState.last_x3d_confidence = 0.7 + Math.random() * 0.3;
            this.mockState.last_mae_label = 'Fighting';
            this.mockState.last_mae_confidence = 0.88;
        } else {
            this.mockState.last_x3d_label = 'False';
            this.mockState.last_x3d_confidence = 0.1;
            this.mockState.last_mae_label = 'Normal';
            this.mockState.last_mae_confidence = 0.95;
        }
        return { ...this.mockState };
    }
}
