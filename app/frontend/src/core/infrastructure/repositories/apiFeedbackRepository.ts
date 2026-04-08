import { HttpClient } from '../http/httpClient';

export class ApiFeedbackRepository {
    constructor(private http: HttpClient) { }

    async sendFeedback(analysisId: string, isConfirmed: boolean): Promise<void> {
        // Заглушка, так как в бэкенде нет эндпоинта
        console.log(`[FEEDBACK] ${analysisId} -> ${isConfirmed ? 'confirmed' : 'rejected'}`);
        const key = `feedback_${analysisId}`;
        localStorage.setItem(key, isConfirmed ? 'confirmed' : 'rejected');
    }
}