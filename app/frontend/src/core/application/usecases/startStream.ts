import { ApiStreamRepository } from '../../infrastructure/repositories/apiStreamRepository';
import { StreamState } from '../../domain/entities/StreamState';

export async function startStream(
    streamRepo: ApiStreamRepository,
    cameraUrl: string,
    cameraId: string
): Promise<StreamState> {
    return streamRepo.startStream(cameraUrl, cameraId);
}