import { ApiStreamRepository } from '../../infrastructure/repositories/apiStreamRepository';

export function sendFrame(streamRepo: ApiStreamRepository, frameBase64: string): void {
    if (streamRepo.isConnected()) {
        streamRepo.sendFrame(frameBase64);
    } else {
        console.warn('Stream not connected, cannot send frame');
    }
}