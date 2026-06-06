import { ApiStreamRepository } from '../../infrastructure/repositories/apiStreamRepository';
import type { StreamState } from '../../domain/entities/StreamState';

export async function getStreamStatus(
    streamRepo: ApiStreamRepository,
    streamId: string
): Promise<StreamState> {
    return streamRepo.getStatus(streamId);
}
