import { ApiVideoRepository } from '../../infrastructure/repositories/apiVideoRepository';
import type { Analysis } from '../../domain/entities/Analysis';

export async function getAnalysisStatus(
    videoRepo: ApiVideoRepository,
    id: string
): Promise<Analysis> {
    return videoRepo.getStatus(id);
}
