import { HttpClient } from '../http/httpClient';
import type { Analysis, ProcessingState } from '../../domain/entities/Analysis';

export class ApiVideoRepository {
    private http: HttpClient;

    constructor(http: HttpClient) {
        this.http = http;
    }

    async upload(file: File, responseUrl?: string): Promise<Analysis> {
        const formData = new FormData();
        formData.append('data', file);
        if (responseUrl) formData.append('response_url', responseUrl);
        const raw = await this.http.postForm<any>('/Klin/upload', formData);
        return this.parseAnalysis(raw);
    }

    async getStatus(id: string): Promise<Analysis> {
        const raw = await this.http.get<any>(`/Klin/${id}`);
        return this.parseAnalysis(raw);
    }

    async getHistory(limit: number = 100): Promise<Analysis[]> {
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
