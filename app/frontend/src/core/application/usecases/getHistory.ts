import { ApiVideoRepository } from '../../infrastructure/repositories/apiVideoRepository';
import { Analysis } from '../../domain/entities/Analysis';

export async function getHistory(videoRepo: ApiVideoRepository, limit = 20): Promise<Analysis[]> {
    return videoRepo.getHistory(limit);
}