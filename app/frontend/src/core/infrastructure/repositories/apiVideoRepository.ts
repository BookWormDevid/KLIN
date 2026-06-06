import { HttpClient } from '../http/httpClient';
import type { Analysis, ProcessingState } from '../../domain/entities/Analysis';
import { appConfig } from '../../../config/appConfig';

// ---------- МОКИ ----------
const MOCK_YOLO: Record<number, number[][]> = {
    1.5: [[100, 150, 300, 450], [350, 200, 500, 400]],
    3.0: [[200, 100, 400, 350]],
    5.2: [[150, 180, 380, 420], [420, 250, 550, 390]],
};

const MOCK_MAE = [
    { time: [0, 2.5] as [number, number], answer: 'Fighting', confident: 0.92 },
    { time: [3.0, 5.5] as [number, number], answer: 'Assault', confident: 0.88 },
    { time: [6.0, 8.0] as [number, number], answer: 'Normal', confident: 0.99 },
];

function createMockAnalysis(overrides: Partial<Analysis> = {}): Analysis {
    return {
        id: overrides.id || `mock-${Date.now()}`,
        state: (overrides.state as ProcessingState) || 'FINISHED',
        x3d: { True: 0.95 },
        mae: MOCK_MAE,
        yolo: MOCK_YOLO,
        objects: ['person'],
        all_classes: ['Fighting', 'Assault', 'Normal'],

    };
}
// ---------------------------

export class ApiVideoRepository {
    private http: HttpClient;

    constructor(http: HttpClient) {
        this.http = http;
    }

    async upload(file: File, responseUrl?: string): Promise<Analysis> {
        if (appConfig.useMocks) {
            // Имитация задержки сети
            await new Promise(res => setTimeout(res, 800));
            return createMockAnalysis();
        }
        const formData = new FormData();
        formData.append('data', file);
        if (responseUrl) formData.append('response_url', responseUrl);
        const raw = await this.http.postForm<any>('/Klin/upload', formData);
        return this.parseAnalysis(raw);
    }

    async getStatus(id: string): Promise<Analysis> {
        if (appConfig.useMocks) {
            await new Promise(res => setTimeout(res, 300));
            return createMockAnalysis({ id });
        }
        const raw = await this.http.get<any>(`/Klin/${id}`);
        return this.parseAnalysis(raw);
    }

    async getHistory(limit: number = 100): Promise<Analysis[]> {
        if (appConfig.useMocks) {
            await new Promise(res => setTimeout(res, 300));
            const items: Analysis[] = [];
            for (let i = 0; i < 5; i++) {
                items.push(createMockAnalysis({ id: `hist-${i}-${Date.now()}` }));
            }
            return items;
        }
        const rawList = await this.http.get<any[]>('/Klin/');
        return rawList.slice(0, limit).map(item => this.parseAnalysis(item));
    }

    private parseAnalysis(raw: any): Analysis {
        return {
            id: raw.id,
            state: raw.state as ProcessingState,
            x3d: raw.x3d ? JSON.parse(raw.x3d) : null,
            mae: raw.mae ? JSON.parse(raw.mae) : null,
            yolo: raw.yolo ? JSON.parse(raw.yolo) : null,
            objects: raw.objects ?? null,
            all_classes: raw.all_classes ?? null,

        };
    }
}
