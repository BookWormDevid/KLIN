import { ApiVideoRepository } from '../../infrastructure/repositories/apiVideoRepository';
import { isValidVideoFile, isFileSizeValid } from '../../domain/services/validation';
import { appConfig } from '../../../config/appConfig';
import type { Analysis } from '../../domain/entities/Analysis';

export async function uploadVideo(
    videoRepo: ApiVideoRepository,
    file: File,
    onStatusUpdate?: (analysis: Analysis) => void
): Promise<Analysis> {
    if (!isValidVideoFile(file)) {
        throw new Error('Пожалуйста, выберите видеофайл');
    }
    if (!isFileSizeValid(file, appConfig.maxVideoSizeBytes)) {
        throw new Error('Файл слишком большой (макс. 200 МБ)');
    }

    const initial = await videoRepo.upload(file);
    if (onStatusUpdate) onStatusUpdate(initial);

    const pollInterval = appConfig.pollIntervalMs;
    const maxAttempts = 60; // 2 минуты
    let attempts = 0;

    return new Promise((resolve, reject) => {
        const interval = setInterval(async () => {
            attempts++;
            try {
                const status = await videoRepo.getStatus(initial.id);
                if (onStatusUpdate) onStatusUpdate(status);
                if (status.state === 'FINISHED' || status.state === 'ERROR') {
                    clearInterval(interval);
                    resolve(status);
                } else if (attempts >= maxAttempts) {
                    clearInterval(interval);
                    reject(new Error('Timeout waiting for analysis'));
                }
            } catch (err) {
                clearInterval(interval);
                reject(err);
            }
        }, pollInterval);
    });
}
