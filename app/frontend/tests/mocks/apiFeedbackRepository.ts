export class ApiFeedbackRepository {
    async sendFeedback(analysisId: string, isConfirmed: boolean): Promise<void> {
        console.log(`Mock feedback: ${analysisId} -> ${isConfirmed ? 'confirmed' : 'rejected'}`);
    }
}
