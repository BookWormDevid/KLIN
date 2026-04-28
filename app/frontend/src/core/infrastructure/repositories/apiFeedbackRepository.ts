import { HttpClient } from '../http/httpClient';

export class ApiFeedbackRepository {
    private http: HttpClient;

    constructor(http: HttpClient) {
        this.http = http;
    }

    async sendFeedback(analysisId: string, isConfirmed: boolean): Promise<void> {
        console.log(`[FEEDBACK] ${analysisId} -> ${isConfirmed ? 'confirmed' : 'rejected'}`);
        const key = `feedback_${analysisId}`;
        localStorage.setItem(key, isConfirmed ? 'confirmed' : 'rejected');
    }
}
