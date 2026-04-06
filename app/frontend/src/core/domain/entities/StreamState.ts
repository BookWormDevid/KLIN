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