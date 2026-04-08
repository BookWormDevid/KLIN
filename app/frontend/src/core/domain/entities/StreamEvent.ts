export type StreamEventType = 'YOLO' | 'MAE' | 'X3D_VIOLENCE' | 'STOP_STREAM' | 'STREAM_STOPPED';

export interface StreamEvent {
    id: string;
    type: StreamEventType;
    camera_id: string;
    stream_id: string;
    payload: any;
}