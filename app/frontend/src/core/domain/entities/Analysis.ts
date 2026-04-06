export type ProcessingState = 'PENDING' | 'PROCESSING' | 'FINISHED' | 'ERROR';

export interface MaeEvent {
    time: [number, number];
    answer: string;
    confident: number;
}

export interface Analysis {
    id: string;
    state: ProcessingState;
    x3d: Record<string, number> | null;
    mae: MaeEvent[] | null;
    yolo: Record<number, number[][]> | null;
    objects: string[] | null;
    all_classes: string[] | null;
}