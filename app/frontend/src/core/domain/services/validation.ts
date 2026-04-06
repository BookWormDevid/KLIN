export const isValidVideoFile = (file: File): boolean => {
    return file.type.startsWith('video/');
};

export const isValidVideoUrl = (url: string): boolean => {
    return /^https?:\/\/.+\.(mp4|avi|mov|mkv|webm)$/i.test(url);
};

export const isFileSizeValid = (file: File, maxBytes: number): boolean => {
    return file.size <= maxBytes;
};