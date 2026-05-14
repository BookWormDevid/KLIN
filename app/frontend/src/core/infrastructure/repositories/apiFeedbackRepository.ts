export class ApiFeedbackRepository {
    async sendFeedback(analysisId: string, isConfirmed: boolean): Promise<void> {
        // Мок – ничего не делаем, только логируем
        console.log(`🥸 Mock feedback: ${analysisId} -> ${isConfirmed ? 'confirmed' : 'rejected'}`);
    }
}
