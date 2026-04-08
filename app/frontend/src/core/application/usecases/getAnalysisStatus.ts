import { ApiVideoRepository } from '../../infrastructure/repositories/apiVideoRepository';
import { Analysis } from '../../domain/entities/Analysis';

export async function getAnalysisStatus(
    videoRepo: ApiVideoRepository,
    id: string
): Promise<Analysis> {
    return videoRepo.getStatus(id);
}