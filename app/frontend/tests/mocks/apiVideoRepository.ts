// локальные типы, идентичные боевым
export type ProcessingState = 'PENDING' | 'PROCESSING' | 'FINISHED' | 'ERROR';
export interface MaeEvent { time: [number, number]; answer: string; confident: number; }
export interface Analysis {
    id: string;
    state: ProcessingState;
    x3d: Record<string, number> | null;
    mae: MaeEvent[] | null;
    yolo: Record<number, number[][]> | null;
    objects: string[] | null;
    all_classes: string[] | null;
    created_at?: string;
}

const MOCK_YOLO: Record<number, number[][]> = { 1.5: [[100, 150, 300, 450], [350, 200, 500, 400]] };
const MOCK_MAE: MaeEvent[] = [
    { time: [0, 2.5], answer: 'Fighting', confident: 0.92 },
    { time: [3.0, 5.5], answer: 'Assault', confident: 0.88 },
    { time: [6.0, 8.0], answer: 'Normal', confident: 0.99 },
];

export class ApiVideoRepository {
    async upload(file: File): Promise<Analysis> {
        await new Promise(r => setTimeout(r, 800));
        return {
            id: `mock-${Date.now()}`,
            state: 'FINISHED',
            x3d: { True: 0.95 },
            mae: MOCK_MAE,
            yolo: MOCK_YOLO,
            objects: ['person'],
            all_classes: ['Fighting', 'Assault', 'Normal'],
            created_at: new Date().toISOString(),
        };
    }
    async getStatus(id: string): Promise<Analysis> {
        await new Promise(r => setTimeout(r, 300));
        return { id, state: 'FINISHED', x3d: { True: 0.95 }, mae: MOCK_MAE, yolo: MOCK_YOLO, objects: ['person'], all_classes: ['Fighting'] };
    }
    async getHistory(limit?: number): Promise<Analysis[]> {
        const items: Analysis[] = [];
        for (let i = 0; i < 5; i++) {
            items.push({
                id: `hist-${i}-${Date.now()}`,
                state: 'FINISHED',
                x3d: { True: 0.9 + i * 0.02 },
                mae: MOCK_MAE,
                yolo: MOCK_YOLO,
                objects: ['person'],
                all_classes: ['Fighting'],
                created_at: new Date().toISOString(),
            });
        }
        return items;
    }
}
