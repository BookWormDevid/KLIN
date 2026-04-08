import { ApiFeedbackRepository } from '../../infrastructure/repositories/apiFeedbackRepository';
import { LocalStorageHistoryRepository } from '../../infrastructure/repositories/localStorageHistoryRepository';

export async function sendFeedback(
    apiFeedbackRepo: ApiFeedbackRepository,
    historyRepo: LocalStorageHistoryRepository,
    analysisId: string,
    isConfirmed: boolean
): Promise<void> {
    await apiFeedbackRepo.sendFeedback(analysisId, isConfirmed);
    historyRepo.updateFeedback(analysisId, isConfirmed ? 'confirmed' : 'rejected');
}