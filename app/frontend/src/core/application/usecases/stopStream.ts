import { ApiStreamRepository } from '../../infrastructure/repositories/apiStreamRepository';

export async function stopStream(
    streamRepo: ApiStreamRepository,
    streamId: string
): Promise<void> {
    return streamRepo.stopStream(streamId);
}